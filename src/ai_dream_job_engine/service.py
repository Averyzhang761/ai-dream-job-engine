"""Privacy-safe orchestration for local job analysis."""

from __future__ import annotations

import hashlib
import importlib
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, NoReturn, TypeVar

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
_STORAGE_CATEGORY = "storage_failed"
_STORAGE_MESSAGE = "Local analysis storage failed safely."
_ID_CATEGORY = "analysis_id_collision"
_ID_MESSAGE = "A unique analysis ID could not be allocated safely."
_AUDIT_CATEGORY = "failure_audit_persistence_failed"
_AUDIT_MESSAGE = "Analysis failed, and its audit record could not be persisted safely."
_JD_REF_PATTERN = re.compile(
    r"jd-evidence:v1:([0-9a-f]{64}):([0-9]+):([0-9]+):([0-9a-f]{64})\Z"
)
_T = TypeVar("_T")


class AnalysisServiceError(RuntimeError):
    """A fixed service error without provider or storage details."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _DecisionSurface:
    input_ids: frozenset[str]
    fact_ids: frozenset[str]
    question_versions: dict[str, str]


class _PersistedRecordSafety:
    """Project sensitive inputs and validate every record before storage."""

    def __init__(
        self,
        input_hash: str,
        rubric: Rubric,
        profile: CandidateProfile,
        raw_job: str | None,
    ) -> None:
        self.input_hash = input_hash
        self.rubric = rubric
        self.profile = _safe_profile(profile)
        self.raw_job = raw_job
        self.job_digest = input_hash.removeprefix("sha256:")

    def project_inputs(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
    ) -> tuple[list[ExtractedFact], list[DecisionSignal]]:
        return self.project_facts(facts), self.project_signals(signals)

    def project_facts(
        self,
        facts: list[ExtractedFact],
    ) -> list[ExtractedFact]:
        if self.raw_job is None:
            raise ValueError("raw job required")
        self._complete_facts(facts)
        document = build_evidence_chunks(self.raw_job)
        safe_facts = []
        for fact in sorted(facts, key=lambda item: item.fact_id):
            if _contains_exact(fact.value, self.raw_job):
                raise ValueError("raw job fact value")
            definition = next(
                item for item in self.rubric.facts if item.id == fact.fact_id
            )
            if (
                definition.requires_evidence
                and not fact.is_unknown
                and not fact.evidence
            ):
                raise ValueError("missing fact evidence")
            refs = []
            for quote in fact.evidence:
                evidence = validate_quote(document, quote)
                refs.append(
                    self._jd_ref(
                        evidence.start_offset,
                        evidence.end_offset,
                        quote,
                    )
                )
            safe_facts.append(fact.model_copy(update={"evidence": refs}))

        return safe_facts

    def project_signals(
        self,
        signals: list[DecisionSignal],
    ) -> list[DecisionSignal]:
        self._complete_signals(signals)
        return [
            signal.model_copy(
                update={
                    "evidence": [
                        ref
                        for value in signal.evidence
                        if (ref := self._signal_ref(value)) is not None
                    ]
                }
            )
            for signal in sorted(signals, key=lambda item: item.question_id)
        ]

    def validate(
        self,
        record: AnalysisRecord,
        *,
        require_current_inputs: bool = True,
    ) -> AnalysisRecord:
        if record.input_hash != self.input_hash:
            raise ValueError("input hash changed")
        if self.raw_job is not None and _contains_exact(
            record.model_dump(mode="python"), self.raw_job
        ):
            raise ValueError("record contains raw job")
        if record.status == "failed":
            if record.facts or record.signals or record.result or record.trace:
                raise ValueError("failed record has partial success")
            return record
        if record.result is None or record.trace is None:
            raise ValueError("success is incomplete")
        if require_current_inputs:
            self._complete(record.facts, record.signals)
        self._versions(record)
        self._evidence(record)
        surface = _decision_surface(self.profile, self.rubric)
        if set(record.trace.input_ids) != surface.input_ids:
            raise ValueError("trace input refs incomplete")
        if record.trace.outbound_data_classes != [DataClass.PUBLIC_JOB_DATA]:
            raise ValueError("unsafe outbound data class")
        return record

    def validate_stored(self, record: AnalysisRecord) -> None:
        trace = record.trace
        result = record.result
        if record.status != "success" or trace is None or result is None:
            raise ValueError("stored analysis is not successful")
        fact_ids = [item.fact_id for item in record.facts]
        signal_keys = [
            (item.question_id, item.question_version) for item in record.signals
        ]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("stored facts are duplicated")
        if len(signal_keys) != len(set(signal_keys)):
            raise ValueError("stored signals are duplicated")
        if trace.analysis_id != record.id:
            raise ValueError("stored trace identity mismatch")
        result_input_ids = {
            input_id
            for item in (
                result.preference_contributions
                + result.qualification_contributions
                + result.gaps
            )
            for input_id in item.input_ids
        }
        stored_input_ids = set(fact_ids) | set(trace.question_versions) | result_input_ids
        traced_input_ids = {
            item
            for item in trace.input_ids
            if not item.startswith("profile-ref:v1:")
        }
        if (
            not set(trace.question_versions).issubset(traced_input_ids)
            or not traced_input_ids.issubset(stored_input_ids)
        ):
            raise ValueError("stored trace inputs are incomplete")
        expected_stages = {
            *({"facts"} if record.facts else set()),
            *({"signals"} if record.signals else set()),
        }
        for versions in (
            trace.providers,
            trace.model_versions,
            trace.prompt_versions,
        ):
            if set(versions) != expected_stages or any(
                not item for item in versions.values()
            ):
                raise ValueError("stored stage metadata is incomplete")
        if result.model_versions != trace.model_versions:
            raise ValueError("stored model versions mismatch")
        if trace.result_version != result.result_version:
            raise ValueError("stored result version mismatch")
        if result.rubric_version != trace.rubric_version:
            raise ValueError("stored rubric versions mismatch")
        if record.facts and (
            {item.provider for item in record.facts} != {trace.providers["facts"]}
            or {item.model for item in record.facts}
            != {trace.model_versions["facts"]}
        ):
            raise ValueError("stored fact metadata mismatch")
        if record.signals and (
            {item.provider for item in record.signals}
            != {trace.providers["signals"]}
            or {item.model for item in record.signals}
            != {trace.model_versions["signals"]}
        ):
            raise ValueError("stored signal metadata mismatch")
        self._evidence(record)
        if trace.outbound_data_classes != [DataClass.PUBLIC_JOB_DATA]:
            raise ValueError("stored trace has unsafe outbound data")

    def _complete(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
    ) -> None:
        self._complete_facts(facts)
        self._complete_signals(signals)

    def _complete_facts(self, facts: list[ExtractedFact]) -> None:
        expected_facts = {item.id for item in self.rubric.facts}
        actual_facts = [item.fact_id for item in facts]
        if (
            len(actual_facts) != len(set(actual_facts))
            or set(actual_facts) != expected_facts
        ):
            raise ValueError("fact set incomplete")

    def _complete_signals(self, signals: list[DecisionSignal]) -> None:
        expected_signals = {
            (item.id, item.version) for item in self.rubric.questions
        }
        actual_signals = [
            (item.question_id, item.question_version) for item in signals
        ]
        if (
            len(actual_signals) != len(set(actual_signals))
            or set(actual_signals) != expected_signals
        ):
            raise ValueError("signal set incomplete")

    def _versions(self, record: AnalysisRecord) -> None:
        trace = record.trace
        result = record.result
        if trace is None or result is None:
            raise ValueError("versions missing")
        expected_stages = {
            *({"facts"} if record.facts else set()),
            *({"signals"} if record.signals else set()),
        }
        for versions in (
            trace.providers,
            trace.model_versions,
            trace.prompt_versions,
        ):
            if set(versions) != expected_stages or any(
                not item for item in versions.values()
            ):
                raise ValueError("stage metadata incomplete")
        surface = _decision_surface(self.profile, self.rubric)
        if trace.question_versions != surface.question_versions:
            raise ValueError("question versions incomplete")
        if trace.rubric_version != self.rubric.version:
            raise ValueError("trace rubric stale")
        if result.rubric_version != self.rubric.version:
            raise ValueError("result rubric stale")
        if trace.result_version != result.result_version:
            raise ValueError("result version mismatch")
        if result.model_versions != trace.model_versions:
            raise ValueError("model versions mismatch")

    def _evidence(self, record: AnalysisRecord) -> None:
        trace = record.trace
        result = record.result
        if trace is None or result is None:
            raise ValueError("evidence missing")
        job_refs = [
            *(item for fact in record.facts for item in fact.evidence),
            *(item for signal in record.signals for item in signal.evidence),
            *(
                item
                for contribution in (
                    result.preference_contributions
                    + result.qualification_contributions
                )
                for item in contribution.job_evidence
            ),
            *(item for gap in result.gaps for item in gap.job_evidence),
        ]
        profile_refs = [
            *(
                item
                for contribution in (
                    result.preference_contributions
                    + result.qualification_contributions
                )
                for item in contribution.candidate_evidence
            ),
            *(item for gap in result.gaps for item in gap.candidate_evidence),
        ]
        for item in job_refs:
            _parse_jd_ref(item, self.job_digest)
        if any(not item.startswith("profile-ref:v1:") for item in profile_refs):
            raise ValueError("raw profile evidence")
        for item in trace.evidence:
            if item.startswith("profile-ref:v1:"):
                continue
            _parse_jd_ref(item, self.job_digest)

    def _jd_ref(self, start: int, end: int, value: str) -> str:
        ref = (
            f"jd-evidence:v1:{self.job_digest}:{start}:{end}:{_digest(value)}"
        )
        _parse_jd_ref(ref, self.job_digest)
        return ref

    def _signal_ref(self, value: str) -> str | None:
        """Keep only evidence resolvable against a user-resupplied JD."""
        if self.raw_job is None:
            raise ValueError("raw job required")
        start = self.raw_job.find(value)
        if value and start >= 0:
            return self._jd_ref(start, start + len(value), value)
        return None

class JobAnalysisService:
    """Orchestrate providers and persist only safety-projected records.

    New analyses and failed recompute attempts are append-only. Successful
    recompute updates its original stable ID without allocating an attempt ID.
    Unconfigured provider stages are skipped, and non-JD model rationale is
    discarded. Pipeline and storage errors expose only fixed safe messages;
    failure-record persistence faults are reported distinctly.
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
        input_hash = _input_hash(job_description)
        safety = _PersistedRecordSafety(
            input_hash, rubric, profile, job_description
        )
        analysis_id = self._allocate_id()
        created_at = self._utc_clock()
        started_at = self._monotonic_clock()
        failed = False
        try:
            facts = []
            if rubric.facts:
                raw_facts = self._fact_extractor.extract(job_description, rubric)
                facts = safety.project_facts(raw_facts)
            signals = []
            if rubric.questions:
                raw_signals = self._signal_provider.judge(
                    job_description, rubric.questions
                )
                signals = safety.project_signals(raw_signals)
            metadata = self._metadata(facts, signals)
            result = _decision(
                facts,
                signals,
                safety.profile,
                rubric,
                metadata["models"],
                metadata["prompts"],
            )
            trace = _trace(
                analysis_id,
                facts,
                signals,
                safety.profile,
                rubric,
                result,
                metadata,
            )
            record = safety.validate(
                AnalysisRecord(
                    id=analysis_id,
                    input_hash=input_hash,
                    facts=facts,
                    signals=signals,
                    result=result,
                    trace=trace,
                    status="success",
                    latency_ms=_latency_ms(
                        started_at, self._monotonic_clock()
                    ),
                    created_at=created_at,
                )
            )
        except Exception:
            failed = True
        if failed:
            self._raise_failed_analysis(
                safety,
                input_hash,
                created_at,
                started_at,
                analysis_id=analysis_id,
            )
        self._insert_or_raise(record)
        return record

    def recompute_decision(self, analysis_id: str) -> AnalysisRecord:
        existing = self._find_analysis(analysis_id)
        if existing is None:
            raise AnalysisServiceError(
                "analysis_not_found",
                "Analysis was not found; choose a successful analysis ID.",
            )
        if (
            existing.status != "success"
            or existing.result is None
            or existing.trace is None
        ):
            raise AnalysisServiceError(
                "analysis_not_successful",
                "Analysis cannot be recomputed; choose a successful analysis ID.",
            )
        profile, rubric = self._active_inputs()
        safety = _PersistedRecordSafety(
            existing.input_hash, rubric, profile, None
        )
        started_at = self._monotonic_clock()
        metadata = {
            "providers": existing.trace.providers,
            "models": existing.trace.model_versions,
            "prompts": existing.trace.prompt_versions,
        }
        failed = False
        try:
            safety.validate_stored(existing)
            result = _decision(
                existing.facts,
                existing.signals,
                safety.profile,
                rubric,
                metadata["models"],
                metadata["prompts"],
            )
            trace = _trace(
                existing.id,
                existing.facts,
                existing.signals,
                safety.profile,
                rubric,
                result,
                metadata,
            )
            recomputed = safety.validate(
                existing.model_copy(
                    update={
                        "result": result,
                        "trace": trace,
                        "latency_ms": _latency_ms(
                            started_at, self._monotonic_clock()
                        ),
                        "error_category": None,
                        "error": None,
                    }
                ),
                require_current_inputs=False,
            )
        except Exception:
            failed = True
        if failed:
            self._raise_failed_analysis(
                safety,
                existing.input_hash,
                self._utc_clock(),
                started_at,
                source_analysis_id=existing.id,
            )
        self._save_or_raise(recomputed)
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
        return [
            AnalysisSummary(
                analysis_id=record.id,
                recommendation=record.result.recommendation,
                job_title=_text_fact(record.facts, "job_title", "title"),
                company=_text_fact(record.facts, "company", "company_name"),
                created_at=record.created_at,
            )
            for record in self._store_call(self._store.list_analyses)
            if record.status == "success" and record.result is not None
        ]

    def _active_inputs(self) -> tuple[CandidateProfile, Rubric]:
        profile = self._store_call(self._store.load_profile)
        if profile is None:
            raise AnalysisServiceError(
                "missing_profile",
                "Candidate profile is missing; initialize the profile before analysis.",
            )
        rubric = self._store_call(self._store.load_rubric)
        if rubric is None:
            raise AnalysisServiceError(
                "missing_rubric",
                "Decision rubric is missing; initialize the rubric before analysis.",
            )
        return profile, rubric

    def _allocate_id(self) -> str:
        failed = False
        try:
            analysis_id = self._id_factory()
        except Exception:
            failed = True
            analysis_id = ""
        if failed or not isinstance(analysis_id, str) or not analysis_id:
            raise AnalysisServiceError(_ID_CATEGORY, _ID_MESSAGE) from None
        if any(
            item.id == analysis_id
            for item in self._store_call(self._store.list_analyses)
        ):
            raise AnalysisServiceError(_ID_CATEGORY, _ID_MESSAGE)
        return analysis_id

    def _metadata(
        self,
        facts: list[ExtractedFact],
        signals: list[DecisionSignal],
    ) -> dict[str, dict[str, str]]:
        metadata = {"providers": {}, "models": {}, "prompts": {}}
        for stage, provider, outputs, label in (
            ("facts", self._fact_extractor, facts, "fact"),
            ("signals", self._signal_provider, signals, "signal"),
        ):
            if not outputs:
                continue
            metadata["providers"][stage] = _stage_metadata(
                provider, outputs, "provider", f"{label} provider"
            )
            metadata["models"][stage] = _stage_metadata(
                provider, outputs, "model", f"{label} model"
            )
            metadata["prompts"][stage] = _prompt_version(provider)
        return metadata

    def _find_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        records = self._store_call(self._store.list_analyses)
        return next((item for item in records if item.id == analysis_id), None)

    def _insert_or_raise(self, record: AnalysisRecord) -> None:
        failed = False
        try:
            self._store.insert_analysis(record)
        except Exception:
            failed = True
        if failed:
            raise AnalysisServiceError(
                _STORAGE_CATEGORY, _STORAGE_MESSAGE
            ) from None

    def _save_or_raise(self, record: AnalysisRecord) -> None:
        failed = False
        try:
            self._store.save_analysis(record)
        except Exception:
            failed = True
        if failed:
            raise AnalysisServiceError(
                _STORAGE_CATEGORY, _STORAGE_MESSAGE
            ) from None

    def _raise_failed_analysis(
        self,
        safety: _PersistedRecordSafety,
        input_hash: str,
        created_at: datetime,
        started_at: float,
        *,
        analysis_id: str | None = None,
        source_analysis_id: str | None = None,
    ) -> NoReturn:
        failed = False
        try:
            attempt_id = analysis_id or self._allocate_id()
            record = safety.validate(
                AnalysisRecord(
                    id=attempt_id,
                    source_analysis_id=source_analysis_id,
                    input_hash=input_hash,
                    status="failed",
                    latency_ms=_latency_ms(started_at, self._monotonic_clock()),
                    error_category=_FAILURE_CATEGORY,
                    error=_FAILURE_MESSAGE,
                    created_at=created_at,
                )
            )
            self._store.insert_analysis(record)
        except Exception:
            failed = True
        if failed:
            raise AnalysisServiceError(_AUDIT_CATEGORY, _AUDIT_MESSAGE) from None
        raise AnalysisServiceError(_FAILURE_CATEGORY, _FAILURE_MESSAGE) from None

    def _store_call(self, operation: Callable[[], _T]) -> _T:
        failed = False
        value: _T | None = None
        try:
            value = operation()
        except Exception:
            failed = True
        if failed:
            raise AnalysisServiceError(
                _STORAGE_CATEGORY, _STORAGE_MESSAGE
            ) from None
        return value  # type: ignore[return-value]


def _decision(
    facts: list[ExtractedFact],
    signals: list[DecisionSignal],
    profile: CandidateProfile,
    rubric: Rubric,
    models: dict[str, str],
    prompts: dict[str, str],
) -> DecisionResult:
    result = decide_job(facts, signals, profile, rubric)
    return result.model_copy(
        update={
            "prompt_version": ";".join(
                f"{key}:{value}" for key, value in sorted(prompts.items())
            ),
            "model_versions": models,
        }
    )


def _trace(
    analysis_id: str,
    facts: list[ExtractedFact],
    signals: list[DecisionSignal],
    profile: CandidateProfile,
    rubric: Rubric,
    result: DecisionResult,
    metadata: dict[str, dict[str, str]],
) -> DecisionTrace:
    surface = _decision_surface(profile, rubric)
    evidence = [
        item
        for fact in facts
        if fact.fact_id in surface.fact_ids
        for item in fact.evidence
    ]
    evidence += [
        item
        for signal in signals
        if signal.question_id in surface.question_versions
        for item in signal.evidence
    ]
    evidence += [
        item
        for contribution in (
            result.preference_contributions + result.qualification_contributions
        )
        for item in contribution.candidate_evidence
    ]
    evidence += [item for gap in result.gaps for item in gap.candidate_evidence]
    return DecisionTrace(
        analysis_id=analysis_id,
        input_ids=sorted(surface.input_ids),
        evidence=sorted(set(evidence)),
        providers=metadata["providers"],
        model_versions=metadata["models"],
        rubric_version=rubric.version,
        prompt_versions=metadata["prompts"],
        question_versions=surface.question_versions,
        result_version=result.result_version,
        outbound_data_classes=[DataClass.PUBLIC_JOB_DATA],
    )


def _decision_surface(
    profile: CandidateProfile,
    rubric: Rubric,
) -> _DecisionSurface:
    fact_ids = frozenset(item.id for item in rubric.facts)
    question_versions = {
        item.id: item.version
        for item in sorted(rubric.questions, key=lambda item: item.id)
    }
    return _DecisionSurface(
        input_ids=frozenset(
            {
                profile.id,
                *(claim.id for claim in profile.claims if claim.confirmed),
                *fact_ids,
                *question_versions,
            }
        ),
        fact_ids=fact_ids,
        question_versions=question_versions,
    )


def _parse_jd_ref(value: str, expected_job_digest: str) -> tuple[int, int, str]:
    match = _JD_REF_PATTERN.fullmatch(value)
    if match is None or match.group(1) != expected_job_digest:
        raise ValueError("invalid JD evidence reference")
    start = int(match.group(2))
    end = int(match.group(3))
    if start > end:
        raise ValueError("invalid JD evidence span")
    return start, end, match.group(4)


def _input_hash(value: str) -> str:
    return f"sha256:{_digest(value)}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _opaque_ref(namespace: str, *values: str) -> str:
    return f"profile-ref:v1:{_digest(chr(0).join((namespace, *values)))}"


def _safe_profile(profile: CandidateProfile) -> CandidateProfile:
    safe_profile_id = _opaque_ref("profile", profile.id)
    claims = []
    for claim in profile.claims:
        safe_claim_id = _opaque_ref("claim", profile.id, claim.id)
        payload = claim.model_dump(mode="python")
        payload["id"] = safe_claim_id
        if claim.source_quote is not None:
            payload["source_quote"] = safe_claim_id
        if claim.provenance is not None:
            payload["provenance"] = safe_claim_id
        claims.append(type(claim).model_validate(payload))
    return profile.model_copy(
        update={
            "id": safe_profile_id,
            "claims": claims,
            "import_receipts": [],
        }
    )


def _contains_exact(value: Any, target: str) -> bool:
    if isinstance(value, str):
        return target in value
    if isinstance(value, dict):
        return any(
            _contains_exact(key, target) or _contains_exact(item, target)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_exact(item, target) for item in value)
    return False


def _stage_metadata(
    provider: Any,
    outputs: list[Any],
    field: str,
    label: str,
) -> str:
    if not outputs:
        raise ValueError(f"{label} metadata missing")
    values = {getattr(item, field, None) for item in outputs}
    if None in values or "" in values:
        raise ValueError(f"{label} metadata missing")
    direct = getattr(provider, field, None)
    if direct:
        values.add(direct)
    if len(values) != 1:
        raise ValueError(f"{label} metadata inconsistent")
    return str(values.pop())


def _prompt_version(provider: Any) -> str:
    direct = getattr(provider, "prompt_version", None)
    if direct:
        return str(direct)
    module = importlib.import_module(type(provider).__module__)
    version = getattr(module, "PROMPT_VERSION", None)
    if version:
        return str(version)
    raise ValueError("provider prompt version metadata missing")


def _latency_ms(started_at: float, finished_at: float) -> int:
    return max(0, round((finished_at - started_at) * 1000))


def _text_fact(facts: list[ExtractedFact], *fact_ids: str) -> str | None:
    for fact_id in fact_ids:
        fact = next((item for item in facts if item.fact_id == fact_id), None)
        if fact is not None and not fact.is_unknown and isinstance(fact.value, str):
            return fact.value
    return None
