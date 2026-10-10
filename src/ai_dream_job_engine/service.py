"""Privacy-safe orchestration for local job analysis."""

from __future__ import annotations

import hashlib
import importlib
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.decision import decide_job
from ai_dream_job_engine.evidence import build_evidence_chunks, validate_quote
from ai_dream_job_engine.models import (
    AnalysisRecord,
    AnalysisSummary,
    CandidateProfile,
    DataClass,
    DecisionResult,
    DecisionSignal,
    DecisionTrace,
    ExtractedFact,
    Rubric,
)
from ai_dream_job_engine.providers.base import (
    DecisionSignalProvider,
    JobFactExtractor,
)


_FAILURE_CATEGORY = "analysis_failed"
_FAILURE_MESSAGE = "Job analysis failed safely."


class AnalysisServiceError(RuntimeError):
    """A stable, actionable service error without provider details."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class JobAnalysisService:
    """Run providers once, persist safe inputs, and recompute locally.

    Provider, evidence, and decision failures are saved as sanitized failed
    records and then raised as AnalysisServiceError. Recompute replaces the
    same stable analysis ID in place and preserves its original creation time.
    """

    def __init__(
        self,
        store: LocalStore,
        fact_extractor: JobFactExtractor,
        signal_provider: DecisionSignalProvider,
        *,
        id_factory: Callable[[], str] | None = None,
        utc_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._fact_extractor = fact_extractor
        self._signal_provider = signal_provider
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._utc_clock = utc_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic

    def analyze_job(self, job_description: str) -> AnalysisRecord:
        if not job_description or not job_description.strip():
            raise AnalysisServiceError(
                "empty_job_description",
                "Job description is empty; provide public job text before analysis.",
            )
        profile, rubric = self._active_inputs()
        analysis_id = self._id_factory()
        input_hash = _input_hash(job_description)
        created_at = self._utc_clock()
        started_at = self._monotonic_clock()

        try:
            facts = sorted(
                self._fact_extractor.extract(job_description, rubric),
                key=lambda fact: fact.fact_id,
            )
            self._validate_fact_evidence(job_description, facts, rubric)
            signals = sorted(
                self._signal_provider.judge(job_description, rubric.questions),
                key=lambda signal: signal.question_id,
            )
            result = self._decision(facts, signals, profile, rubric)
            trace = self._trace(
                analysis_id,
                facts,
                signals,
                profile,
                rubric,
                result,
            )
            record = AnalysisRecord(
                id=analysis_id,
                input_hash=input_hash,
                facts=facts,
                signals=signals,
                result=result,
                trace=trace,
                status="success",
                latency_ms=_latency_ms(started_at, self._monotonic_clock()),
                created_at=created_at,
            )
        except Exception:
            failed = self._failed_record(
                analysis_id,
                input_hash,
                created_at,
                started_at,
            )
            self._store.save_analysis(failed)
            raise AnalysisServiceError(
                _FAILURE_CATEGORY,
                _FAILURE_MESSAGE,
            ) from None

        self._store.save_analysis(record)
        return record

    def recompute_decision(self, analysis_id: str) -> AnalysisRecord:
        existing = self._find_analysis(analysis_id)
        if existing is None:
            raise AnalysisServiceError(
                "analysis_not_found",
                "Analysis was not found; choose a successful analysis ID.",
            )
        if existing.status != "success" or existing.result is None:
            raise AnalysisServiceError(
                "analysis_not_successful",
                "Analysis cannot be recomputed; choose a successful analysis ID.",
            )
        profile, rubric = self._active_inputs()
        started_at = self._monotonic_clock()

        try:
            result = self._decision(
                existing.facts,
                existing.signals,
                profile,
                rubric,
                previous_trace=existing.trace,
            )
            trace = self._trace(
                existing.id,
                existing.facts,
                existing.signals,
                profile,
                rubric,
                result,
                previous_trace=existing.trace,
            )
            recomputed = existing.model_copy(
                update={
                    "result": result,
                    "trace": trace,
                    "latency_ms": _latency_ms(
                        started_at, self._monotonic_clock()
                    ),
                    "error_category": None,
                    "error": None,
                }
            )
        except Exception:
            failed = self._failed_record(
                existing.id,
                existing.input_hash,
                existing.created_at,
                started_at,
            )
            self._store.save_analysis(failed)
            raise AnalysisServiceError(
                _FAILURE_CATEGORY,
                _FAILURE_MESSAGE,
            ) from None

        self._store.save_analysis(recomputed)
        return recomputed

    def get_decision_trace(self, analysis_id: str) -> DecisionTrace:
        record = self._find_analysis(analysis_id)
        if record is None:
            raise AnalysisServiceError(
                "analysis_not_found",
                "Analysis was not found; choose a successful analysis ID.",
            )
        if record.status != "success" or record.trace is None:
            raise AnalysisServiceError(
                "trace_unavailable",
                "Decision trace is unavailable; choose a successful analysis ID.",
            )
        return record.trace

    def list_analyzed_jobs(self) -> list[AnalysisSummary]:
        summaries = []
        for record in self._store.list_analyses():
            if record.status != "success" or record.result is None:
                continue
            summaries.append(
                AnalysisSummary(
                    analysis_id=record.id,
                    recommendation=record.result.recommendation,
                    job_title=_text_fact(record.facts, "job_title", "title"),
                    company=_text_fact(record.facts, "company", "company_name"),
                    created_at=record.created_at,
                )
            )
        return summaries

    def _active_inputs(self) -> tuple[CandidateProfile, Rubric]:
        profile = self._store.load_profile()
        if profile is None:
            raise AnalysisServiceError(
                "missing_profile",
                "Candidate profile is missing; initialize the profile before analysis.",
            )
        rubric = self._store.load_rubric()
        if rubric is None:
            raise AnalysisServiceError(
                "missing_rubric",
                "Decision rubric is missing; initialize the rubric before analysis.",
            )
        return profile, rubric

    def _decision(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
        profile: CandidateProfile,
        rubric: Rubric,
        *,
        previous_trace: DecisionTrace | None = None,
    ) -> DecisionResult:
        result = decide_job(
            facts,
            signals,
            _profile_with_evidence_refs(profile),
            rubric,
        )
        if previous_trace is None:
            prompt_versions = self._prompt_versions()
            model_versions = self._model_versions(facts, signals)
        else:
            prompt_versions = previous_trace.prompt_versions
            model_versions = previous_trace.model_versions
        return result.model_copy(
            update={
                "prompt_version": ";".join(
                    f"{key}:{value}" for key, value in prompt_versions.items()
                ),
                "model_versions": model_versions,
            }
        )

    def _trace(
        self,
        analysis_id: str,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
        profile: CandidateProfile,
        rubric: Rubric,
        result: DecisionResult,
        *,
        previous_trace: DecisionTrace | None = None,
    ) -> DecisionTrace:
        if previous_trace is None:
            providers = self._provider_ids(facts, signals)
            models = self._model_versions(facts, signals)
            prompts = self._prompt_versions()
        else:
            providers = previous_trace.providers
            models = previous_trace.model_versions
            prompts = previous_trace.prompt_versions

        active_claim_ids = [
            claim.id
            for claim in profile.claims
            if claim.confirmed
        ]
        evidence = [item for fact in facts for item in fact.evidence]
        evidence += [item for signal in signals for item in signal.evidence]
        evidence += [
            item
            for contribution in (
                result.preference_contributions
                + result.qualification_contributions
            )
            for item in contribution.candidate_evidence
        ]
        evidence += [
            item
            for gap in result.gaps
            for item in gap.candidate_evidence
        ]
        return DecisionTrace(
            analysis_id=analysis_id,
            input_ids=sorted(
                {
                    profile.id,
                    *active_claim_ids,
                    *(fact.fact_id for fact in facts),
                    *(signal.question_id for signal in signals),
                }
            ),
            evidence=sorted(set(evidence)),
            providers=providers,
            model_versions=models,
            rubric_version=rubric.version,
            prompt_versions=prompts,
            question_versions={
                question.id: question.version
                for question in sorted(rubric.questions, key=lambda item: item.id)
            },
            result_version=result.result_version,
            outbound_data_classes=[DataClass.PUBLIC_JOB_DATA],
        )

    def _provider_ids(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
    ) -> dict[str, str]:
        return {
            "facts": _stage_metadata(
                self._fact_extractor, facts, "provider", "fact extractor"
            ),
            "signals": _stage_metadata(
                self._signal_provider, signals, "provider", "signal provider"
            ),
        }

    def _model_versions(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
    ) -> dict[str, str]:
        return {
            "facts": _stage_metadata(
                self._fact_extractor, facts, "model", "fact model"
            ),
            "signals": _stage_metadata(
                self._signal_provider, signals, "model", "signal model"
            ),
        }

    def _prompt_versions(self) -> dict[str, str]:
        versions = {}
        fact_version = _prompt_version(self._fact_extractor)
        signal_version = _prompt_version(self._signal_provider)
        if fact_version is not None:
            versions["facts"] = fact_version
        if signal_version is not None:
            versions["signals"] = signal_version
        if not versions:
            raise ValueError("provider prompt version metadata is missing")
        return versions

    def _failed_record(
        self,
        analysis_id: str,
        input_hash: str,
        created_at: datetime,
        started_at: float,
    ) -> AnalysisRecord:
        return AnalysisRecord(
            id=analysis_id,
            input_hash=input_hash,
            status="failed",
            latency_ms=_latency_ms(started_at, self._monotonic_clock()),
            error_category=_FAILURE_CATEGORY,
            error=_FAILURE_MESSAGE,
            created_at=created_at,
        )

    def _find_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        return next(
            (
                record
                for record in self._store.list_analyses()
                if record.id == analysis_id
            ),
            None,
        )

    @staticmethod
    def _validate_fact_evidence(
        job_description: str,
        facts: list[ExtractedFact],
        rubric: Rubric,
    ) -> None:
        definitions = {definition.id: definition for definition in rubric.facts}
        document = build_evidence_chunks(job_description)
        for fact in facts:
            definition = definitions.get(fact.fact_id)
            if definition is None:
                raise ValueError("fact is not defined by the active rubric")
            if (
                definition.requires_evidence
                and not fact.is_unknown
                and not fact.evidence
            ):
                raise ValueError("required fact evidence is missing")
            for quote in fact.evidence:
                validate_quote(document, quote)


def _input_hash(job_description: str) -> str:
    digest = hashlib.sha256(job_description.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _latency_ms(started_at: float, finished_at: float) -> int:
    return max(0, round((finished_at - started_at) * 1000))


def _profile_with_evidence_refs(profile: CandidateProfile) -> CandidateProfile:
    claims = []
    for claim in profile.claims:
        payload = claim.model_dump(mode="python")
        if claim.source_quote is not None:
            payload["source_quote"] = f"claim:{claim.id}"
        if claim.provenance is not None:
            payload["provenance"] = f"claim:{claim.id}"
        claims.append(type(claim).model_validate(payload))
    return profile.model_copy(update={"claims": claims})


def _stage_metadata(
    provider: Any,
    outputs: list[Any],
    field: str,
    label: str,
) -> str:
    direct = getattr(provider, field, None)
    values = {getattr(output, field, None) for output in outputs}
    values.discard(None)
    if direct:
        values.add(direct)
    if len(values) != 1:
        raise ValueError(f"{label} metadata is missing or inconsistent")
    return str(values.pop())


def _prompt_version(provider: Any) -> str | None:
    direct = getattr(provider, "prompt_version", None)
    if direct:
        return str(direct)
    module = importlib.import_module(type(provider).__module__)
    module_version = getattr(module, "PROMPT_VERSION", None)
    if module_version:
        return str(module_version)
    return None


def _text_fact(facts: list[ExtractedFact], *fact_ids: str) -> str | None:
    for fact_id in fact_ids:
        fact = next((item for item in facts if item.fact_id == fact_id), None)
        if fact is not None and not fact.is_unknown and isinstance(fact.value, str):
            return fact.value
    return None
