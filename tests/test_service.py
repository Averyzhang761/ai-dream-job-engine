from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimSource,
    DataClass,
    DecisionQuestion,
    DecisionSignal,
    ExtractedFact,
    FactDefinition,
    RecommendationLevel,
    Rubric,
    ScoringDirection,
    ValueDimension,
)
from ai_dream_job_engine.providers.base import ProviderError
from ai_dream_job_engine.service import AnalysisServiceError, JobAnalysisService


JOB_DESCRIPTION = "Senior Python Engineer at Acme. Remote role."
NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


class FakeFactExtractor:
    provider = "fake-facts"
    model = "facts-model-v1"
    prompt_version = "facts-prompt-v1"

    def __init__(
        self,
        *,
        quote: str = "Senior Python Engineer",
        value: object = "Senior Python Engineer",
        fail: bool = False,
        outputs: list[ExtractedFact] | None = None,
    ):
        self.quote = quote
        self.value = value
        self.fail = fail
        self.outputs = outputs
        self.calls: list[tuple[str, str]] = []

    def extract(self, job_description: str, rubric: Rubric) -> list[ExtractedFact]:
        self.calls.append((job_description, rubric.version))
        if self.fail:
            raise ProviderError(
                "provider leaked api-key-secret and private@example.com"
            )
        if self.outputs is not None:
            return self.outputs
        return [
            ExtractedFact(
                fact_id="job_title",
                value=self.value,
                evidence=[self.quote],
                provider=self.provider,
                model=self.model,
            )
        ]


class FakeSignalProvider:
    provider = "fake-signals"
    model = "signals-model-v1"
    prompt_version = "signals-prompt-v1"

    def __init__(self, *, outputs: list[DecisionSignal] | None = None) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, list[str]]] = []

    def judge(
        self, job_description: str, questions: list[DecisionQuestion]
    ) -> list[DecisionSignal]:
        self.calls.append((job_description, [question.id for question in questions]))
        if self.outputs is not None:
            return self.outputs
        return [
            DecisionSignal(
                question_id="role_fit",
                question_version=questions[0].version,
                probability=1,
                evidence=["public role-fit signal"],
                provider=self.provider,
                model=self.model,
            )
        ]


def _rubric(*, version: str = "rubric-v1") -> Rubric:
    return Rubric(
        version=version,
        facts=[
            FactDefinition(
                id="job_title",
                name="Job title",
                definition="The advertised role title.",
                data_type="text",
                extraction_instructions="Extract the exact title.",
            )
        ],
        dimensions=[
            ValueDimension(
                id="role_fit",
                name="Role fit",
                definition="Whether the role fits the candidate.",
                signal_ids=["role_fit"],
                weight=1,
                scoring_direction=ScoringDirection.HIGHER_IS_BETTER,
            )
        ],
        questions=[
            DecisionQuestion(
                id="role_fit",
                version="question-v1",
                question="Does this role fit the candidate?",
            )
        ],
        recommendation_thresholds={
            RecommendationLevel.PRIORITY_APPLY: 0.8,
            RecommendationLevel.APPLY_AFTER_REVIEW: 0.6,
            RecommendationLevel.HOLD: 0.4,
            RecommendationLevel.SKIP: 0,
        },
    )


def _service(
    tmp_path,
    *,
    profile: CandidateProfile | None = None,
    rubric: Rubric | None = None,
    extractor: FakeFactExtractor | None = None,
    signal_provider: FakeSignalProvider | None = None,
    id_factory: Callable[[], str] | None = None,
) -> tuple[JobAnalysisService, LocalStore, FakeFactExtractor, FakeSignalProvider]:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    if profile is not None:
        store.save_profile(profile)
    if rubric is not None:
        store.save_rubric(rubric)
    facts = extractor or FakeFactExtractor()
    signals = signal_provider or FakeSignalProvider()
    ticks = iter([10.0, 10.025, 20.0, 20.007])
    service = JobAnalysisService(
        store,
        facts,
        signals,
        id_factory=id_factory or (lambda: "analysis-1"),
        utc_clock=lambda: NOW,
        monotonic_clock=lambda: next(ticks),
    )
    return service, store, facts, signals


def test_analyze_job_runs_extract_signals_decide_and_persist(tmp_path) -> None:
    service, store, extractor, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )

    record = service.analyze_job(JOB_DESCRIPTION)

    assert extractor.calls == [(JOB_DESCRIPTION, "rubric-v1")]
    assert signal_provider.calls == [(JOB_DESCRIPTION, ["role_fit"])]
    assert record.status == "success"
    assert record.result is not None
    assert record.result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert record.input_hash == "sha256:" + hashlib.sha256(
        JOB_DESCRIPTION.encode("utf-8")
    ).hexdigest()
    assert record.latency_ms == 25
    assert store.list_analyses() == [record]
    assert service.list_analyzed_jobs()[0].job_title == "Senior Python Engineer"


def test_recompute_reuses_normalized_inputs_without_model_calls(tmp_path) -> None:
    ids = iter(["analysis-1", "recompute-attempt-1"])
    service, store, extractor, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    updated_rubric = _rubric(version="rubric-v2")
    store.save_rubric(updated_rubric)
    extractor.provider = "replacement-facts"
    extractor.model = "replacement-facts-v2"
    extractor.prompt_version = "replacement-facts-prompt-v2"
    signal_provider.provider = "replacement-signals"
    signal_provider.model = "replacement-signals-v2"
    signal_provider.prompt_version = "replacement-signals-prompt-v2"

    recomputed = service.recompute_decision(original.id)

    assert len(extractor.calls) == 1
    assert len(signal_provider.calls) == 1
    assert recomputed.id == original.id
    assert recomputed.created_at == original.created_at
    assert recomputed.facts == original.facts
    assert recomputed.signals == original.signals
    assert recomputed.result is not None
    assert recomputed.result.rubric_version == "rubric-v2"
    assert recomputed.trace is not None
    assert recomputed.trace.rubric_version == "rubric-v2"
    assert recomputed.trace.providers == original.trace.providers
    assert recomputed.trace.model_versions == original.trace.model_versions
    assert recomputed.trace.prompt_versions == original.trace.prompt_versions
    assert store.list_analyses() == [recomputed]


def test_missing_profile_returns_actionable_error(tmp_path) -> None:
    service, store, extractor, signal_provider = _service(
        tmp_path,
        rubric=_rubric(),
    )

    with pytest.raises(
        AnalysisServiceError,
        match="Candidate profile is missing; initialize the profile before analysis.",
    ) as exc_info:
        service.analyze_job(JOB_DESCRIPTION)

    assert exc_info.value.code == "missing_profile"
    assert extractor.calls == []
    assert signal_provider.calls == []
    assert store.list_analyses() == []


def test_invalid_quote_does_not_persist_success(tmp_path) -> None:
    service, store, _, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=FakeFactExtractor(quote="not present in the public JD"),
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.analyze_job(JOB_DESCRIPTION)

    records = store.list_analyses()
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].result is None
    assert records[0].trace is None
    assert records[0].facts == []
    assert signal_provider.calls == []


def test_provider_failure_records_failure_not_decision(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=FakeFactExtractor(fail=True),
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.analyze_job(JOB_DESCRIPTION)

    record = store.list_analyses()[0]
    serialized = record.model_dump_json()
    assert record.status == "failed"
    assert record.error_category == "analysis_failed"
    assert record.error == "Job analysis failed safely."
    assert record.result is None
    assert record.trace is None
    for restricted in (
        JOB_DESCRIPTION,
        "api-key-secret",
        "private@example.com",
        "provider leaked",
    ):
        assert restricted not in serialized
    assert service.list_analyzed_jobs() == []


def test_trace_records_outbound_data_class_without_raw_restricted_content(
    tmp_path,
) -> None:
    restricted_profile_value = "private-resume-secret@example.com"
    profile = CandidateProfile(
        claims=[
            CandidateClaim(
                id="private-claim",
                claim_type="job_title",
                value=restricted_profile_value,
                source=ClaimSource.USER_CONFIRMED,
                confirmed=True,
                provenance="private memory content",
            )
        ]
    )
    service, _, _, _ = _service(
        tmp_path,
        profile=profile,
        rubric=_rubric(),
    )

    record = service.analyze_job(JOB_DESCRIPTION)
    trace = service.get_decision_trace(record.id)

    assert trace.outbound_data_classes == [DataClass.PUBLIC_JOB_DATA]
    assert restricted_profile_value not in trace.model_dump_json()
    assert "private memory content" not in trace.model_dump_json()


def test_trace_contains_provider_rubric_prompt_and_result_versions(tmp_path) -> None:
    service, _, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )

    record = service.analyze_job(JOB_DESCRIPTION)
    trace = service.get_decision_trace(record.id)

    assert trace.providers == {
        "facts": "fake-facts",
        "signals": "fake-signals",
    }
    assert trace.model_versions == {
        "facts": "facts-model-v1",
        "signals": "signals-model-v1",
    }
    assert trace.prompt_versions == {
        "facts": "facts-prompt-v1",
        "signals": "signals-prompt-v1",
    }
    assert trace.question_versions == {"role_fit": "question-v1"}
    assert trace.rubric_version == "rubric-v1"
    assert trace.result_version == "1"


def test_success_projects_all_evidence_to_auditable_opaque_refs(tmp_path) -> None:
    signals = FakeSignalProvider(
        outputs=[
            DecisionSignal(
                question_id="role_fit",
                question_version="question-v1",
                probability=1,
                evidence=[JOB_DESCRIPTION],
                provider="fake-signals",
                model="signals-model-v1",
            )
        ]
    )
    service, _, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=FakeFactExtractor(quote=JOB_DESCRIPTION),
        signal_provider=signals,
    )

    record = service.analyze_job(JOB_DESCRIPTION)

    serialized = record.model_dump_json()
    assert JOB_DESCRIPTION not in serialized
    assert all(item.startswith("jd-evidence:v1:") for item in record.facts[0].evidence)
    assert all(
        item.startswith("jd-evidence:v1:") for item in record.signals[0].evidence
    )
    assert record.trace is not None
    assert all(
        item.startswith(("jd-evidence:v1:", "profile-ref:v1:"))
        for item in record.trace.evidence
    )
    offset_ref = record.facts[0].evidence[0]
    assert f":0:{len(JOB_DESCRIPTION)}:" in offset_ref


def test_full_raw_jd_value_is_rejected_without_persistence(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=FakeFactExtractor(
            quote="Senior Python Engineer",
            value=JOB_DESCRIPTION,
        ),
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.analyze_job(JOB_DESCRIPTION)

    record = store.list_analyses()[0]
    assert record.status == "failed"
    assert JOB_DESCRIPTION not in record.model_dump_json()


def test_failed_recompute_appends_attempt_and_preserves_success(tmp_path) -> None:
    ids = iter(["analysis-1", "recompute-failure-1"])
    service, store, extractor, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    store.save_profile(
        CandidateProfile(
            claims=[
                CandidateClaim(
                    id="bad-claim",
                    claim_type="not-in-rubric",
                    value=True,
                    source=ClaimSource.USER_CONFIRMED,
                    confirmed=True,
                    provenance="private",
                )
            ]
        )
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.recompute_decision(original.id)

    records = store.list_analyses()
    assert records[0] == original
    assert records[1].id == "recompute-failure-1"
    assert records[1].status == "failed"
    assert records[1].source_analysis_id == original.id
    assert len(extractor.calls) == 1
    assert len(signal_provider.calls) == 1


def test_duplicate_generated_id_fails_before_providers_without_overwrite(
    tmp_path,
) -> None:
    service, store, extractor, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )
    original = service.analyze_job(JOB_DESCRIPTION)

    with pytest.raises(
        AnalysisServiceError,
        match="A unique analysis ID could not be allocated safely.",
    ) as exc_info:
        service.analyze_job("Different public job description.")

    assert exc_info.value.code == "analysis_id_collision"
    assert store.list_analyses() == [original]
    assert len(extractor.calls) == 1
    assert len(signal_provider.calls) == 1


def test_recompute_failure_id_collision_leaves_success_untouched(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )
    original = service.analyze_job(JOB_DESCRIPTION)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.recompute_decision(original.id)

    assert exc_info.value.code == "analysis_id_collision"
    assert store.list_analyses() == [original]


@pytest.mark.parametrize("provider_fails", [False, True])
def test_store_failures_are_fixed_errors_without_secret_context(
    tmp_path,
    monkeypatch,
    provider_fails: bool,
) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=FakeFactExtractor(fail=provider_fails),
    )

    def raise_secret(_record):
        raise RuntimeError("sqlite /private/path api-key-secret")

    monkeypatch.setattr(store, "insert_analysis", raise_secret, raising=False)

    with pytest.raises(
        AnalysisServiceError,
        match="Local analysis storage failed safely."
        if not provider_fails
        else "Job analysis failed safely.",
    ) as exc_info:
        service.analyze_job(JOB_DESCRIPTION)

    assert "api-key-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("extractor", "signals"),
    [
        (FakeFactExtractor(outputs=[]), FakeSignalProvider()),
        (
            FakeFactExtractor(
                outputs=[
                    ExtractedFact(
                        fact_id="job_title",
                        value="Senior Python Engineer",
                        evidence=["Senior Python Engineer"],
                        provider="fake-facts",
                        model="facts-model-v1",
                    ),
                    ExtractedFact(
                        fact_id="job_title",
                        value="Senior Python Engineer",
                        evidence=["Senior Python Engineer"],
                        provider="fake-facts",
                        model="facts-model-v1",
                    ),
                ]
            ),
            FakeSignalProvider(),
        ),
        (FakeFactExtractor(), FakeSignalProvider(outputs=[])),
        (
            FakeFactExtractor(),
            FakeSignalProvider(
                outputs=[
                    DecisionSignal(
                        question_id="role_fit",
                        question_version="wrong-version",
                        probability=1,
                        provider="fake-signals",
                        model="signals-model-v1",
                    )
                ]
            ),
        ),
        (
            FakeFactExtractor(),
            FakeSignalProvider(
                outputs=[
                    DecisionSignal(
                        question_id="role_fit",
                        question_version="question-v1",
                        probability=1,
                        provider="fake-signals",
                        model="signals-model-v1",
                    ),
                    DecisionSignal(
                        question_id="role_fit",
                        question_version="question-v1",
                        probability=1,
                        provider="fake-signals",
                        model="signals-model-v1",
                    ),
                ]
            ),
        ),
    ],
)
def test_incomplete_or_wrong_version_outputs_cannot_persist_success(
    tmp_path,
    extractor,
    signals,
) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        extractor=extractor,
        signal_provider=signals,
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.analyze_job(JOB_DESCRIPTION)

    assert [record.status for record in store.list_analyses()] == ["failed"]


def test_both_provider_stages_require_prompt_metadata(tmp_path) -> None:
    signals = FakeSignalProvider()
    signals.prompt_version = None
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        signal_provider=signals,
    )

    with pytest.raises(AnalysisServiceError, match="Job analysis failed safely."):
        service.analyze_job(JOB_DESCRIPTION)

    assert [record.status for record in store.list_analyses()] == ["failed"]


def test_profile_identifiers_and_source_text_become_opaque_refs(tmp_path) -> None:
    profile_id = "profile-private@example.com"
    claim_id = "resume says secret phone 555-0100"
    profile = CandidateProfile(
        id=profile_id,
        claims=[
            CandidateClaim(
                id=claim_id,
                claim_type="job_title",
                value="private value",
                source=ClaimSource.USER_CONFIRMED,
                confirmed=True,
                source_quote="private source quote",
                provenance="private memory provenance",
            )
        ],
    )
    service, _, _, _ = _service(
        tmp_path,
        profile=profile,
        rubric=_rubric(),
    )

    record = service.analyze_job(JOB_DESCRIPTION)
    serialized = record.model_dump_json()

    for restricted in (
        profile_id,
        claim_id,
        "private value",
        "private source quote",
        "private memory provenance",
    ):
        assert restricted not in serialized
    assert record.trace is not None
    assert any(
        item.startswith("profile-ref:v1:") for item in record.trace.input_ids
    )
    assert record.trace.outbound_data_classes == [DataClass.PUBLIC_JOB_DATA]


def test_failed_recompute_storage_failure_keeps_prior_success(
    tmp_path,
    monkeypatch,
) -> None:
    ids = iter(["analysis-1", "recompute-failure-1"])
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    store.save_profile(
        CandidateProfile(
            claims=[
                CandidateClaim(
                    id="invalid",
                    claim_type="not-in-rubric",
                    value=True,
                    source=ClaimSource.USER_CONFIRMED,
                    confirmed=True,
                    provenance="secret",
                )
            ]
        )
    )

    def reject_failure(_record):
        raise RuntimeError("storage secret")

    monkeypatch.setattr(store, "insert_analysis", reject_failure)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.recompute_decision(original.id)

    assert str(exc_info.value) == "Job analysis failed safely."
    assert store.list_analyses() == [original]
