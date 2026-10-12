from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import pytest

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.mcp_server import (
    ConfiguredAnalysisApplication,
    LocalProfileApplication,
)
from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimSource,
    DataClass,
    DecisionQuestion,
    DecisionSignal,
    ExtractedFact,
    FactDataType,
    FactDefinition,
    JobReviewStatus,
    RecommendationLevel,
    Rubric,
    RubricPatch,
    ScoringDirection,
    ValueDimension,
)
from ai_dream_job_engine.providers.base import ProviderError
from ai_dream_job_engine.providers.jev import JevSignalProvider
from ai_dream_job_engine.rubric import apply_rubric_patch, load_default_rubric
from ai_dream_job_engine.service import AnalysisServiceError, JobAnalysisService
from ai_dream_job_engine.workspace import WorkspaceApplication


JOB_DESCRIPTION = "Senior Python Engineer at Acme. Remote role."
NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
ANALYSIS_ID = "00000000-0000-4000-8000-000000000001"
FAILURE_ID = "00000000-0000-4000-8000-000000000002"
UNUSED_ID = "00000000-0000-4000-8000-000000000003"
OTHER_ID = "00000000-0000-4000-8000-000000000004"


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
        id_factory=id_factory or (lambda: ANALYSIS_ID),
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
    assert record.input_length == len(JOB_DESCRIPTION)
    assert record.decision_surface is not None
    assert record.decision_surface.fact_ids == ["job_title"]
    assert record.decision_surface.question_versions == {
        "role_fit": "question-v1"
    }
    assert record.decision_surface.profile_refs == sorted(
        record.decision_surface.profile_refs
    )
    assert record.trace.input_ids == sorted(
        {
            *record.decision_surface.fact_ids,
            *record.decision_surface.question_versions,
            *record.decision_surface.profile_refs,
        }
    )
    assert record.latency_ms == 25
    assert store.list_analyses() == [record]
    assert service.list_analyzed_jobs()[0].job_title == "Senior Python Engineer"


def test_list_analyzed_jobs_excludes_trace_integrity_invalid_success(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )
    record = service.analyze_job(JOB_DESCRIPTION)
    store.save_analysis(
        record.model_copy(
            update={
                "trace": record.trace.model_copy(
                    update={"analysis_id": OTHER_ID}
                )
            }
        )
    )

    assert service.list_analyzed_jobs() == []


def test_recompute_reuses_normalized_inputs_without_model_calls(tmp_path) -> None:
    ids = iter([ANALYSIS_ID, UNUSED_ID])
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
    assert original.decision_revision == 1
    assert recomputed.decision_revision == 2
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


def test_shared_sqlite_recomputes_serialize_revisions_and_stale_readiness(
    tmp_path, monkeypatch
) -> None:
    rubric = _rubric()
    rubric.facts[0] = rubric.facts[0].model_copy(
        update={"candidate_comparison": "exact", "qualification_weight": 1}
    )
    profile = CandidateProfile(
        claims=[
            CandidateClaim(
                id="confirmed-job-title",
                claim_type="job_title",
                value="Senior Python Engineer",
                source=ClaimSource.USER_CONFIRMED,
                confirmed=True,
                source_quote="Senior Python Engineer",
            )
        ]
    )
    service, first_store, _, _ = _service(
        tmp_path,
        profile=profile,
        rubric=rubric,
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    assert original.result is not None
    assert original.result.recommendation is RecommendationLevel.PRIORITY_APPLY

    database_path = tmp_path / "store.sqlite3"
    second_store = LocalStore.open(database_path)
    second_store.initialize()
    first_application = ConfiguredAnalysisApplication(first_store)
    second_application = ConfiguredAnalysisApplication(second_store)
    workspace = WorkspaceApplication(
        first_store,
        profile_application=LocalProfileApplication(first_store),
        analysis_application=first_application,
    )

    reads_aligned = threading.Barrier(2)
    thread_state = threading.local()
    original_find = JobAnalysisService._find_analysis

    def aligned_find(service, analysis_id):
        record = original_find(service, analysis_id)
        if threading.current_thread() is not threading.main_thread() and not getattr(
            thread_state, "aligned", False
        ):
            thread_state.aligned = True
            reads_aligned.wait(timeout=2)
        return record

    monkeypatch.setattr(JobAnalysisService, "_find_analysis", aligned_find)

    ready_saved = threading.Event()
    first_write = first_store.compare_and_swap_analysis
    second_write = second_store.compare_and_swap_analysis

    def save_ready_after_first_write(*args, **kwargs):
        saved = first_write(*args, **kwargs)
        record = args[0]
        if saved is not False and record.decision_revision == 2:
            workspace.set_review_status(
                original.id, JobReviewStatus.READY_TO_APPLY
            )
            ready_saved.set()
        return saved

    def wait_for_ready_before_second_write(*args, **kwargs):
        record = args[0]
        if record.decision_revision == 2:
            assert ready_saved.wait(timeout=2)
        return second_write(*args, **kwargs)

    monkeypatch.setattr(
        first_store, "compare_and_swap_analysis", save_ready_after_first_write
    )
    monkeypatch.setattr(
        second_store, "compare_and_swap_analysis", wait_for_ready_before_second_write
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            first_application.recompute_decision, original.id
        )
        second_future = executor.submit(
            second_application.recompute_decision, original.id
        )
        recomputed = [first_future.result(timeout=3), second_future.result(timeout=3)]

    assert sorted(record.decision_revision for record in recomputed) == [2, 3]
    assert first_store.list_analyses()[0].decision_revision == 3
    assert first_store.load_job_review(original.id).decision_revision == 2
    assert (
        workspace.get_job_detail(original.id).summary.review_status
        is JobReviewStatus.UNREVIEWED
    )


def test_configured_default_persists_and_recomputes_with_audit_support(tmp_path) -> None:
    rubric = _rubric()
    rubric.facts[0] = rubric.facts[0].model_copy(
        update={
            "data_type": FactDataType.BOOLEAN,
            "missing_policy": "default",
            "default_value": True,
        }
    )
    rubric.dimensions[0] = rubric.dimensions[0].model_copy(
        update={"fact_ids": ["job_title"], "signal_ids": []}
    )
    rubric.questions = []
    extractor = FakeFactExtractor(
        outputs=[
            ExtractedFact(
                fact_id="job_title",
                is_unknown=True,
                provider="fake-facts",
                model="facts-model-v1",
            )
        ]
    )
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=rubric,
        extractor=extractor,
    )

    original = service.analyze_job(JOB_DESCRIPTION)
    recomputed = service.recompute_decision(original.id)

    for record in (original, store.list_analyses()[0], recomputed):
        assert record.result is not None
        contribution = record.result.preference_contributions[0]
        assert contribution.audit_support == ["configured-default:job_title"]
        assert contribution.job_evidence == []


def test_real_jev_adapter_contributes_default_signal_dimensions(tmp_path) -> None:
    rubric = load_default_rubric()
    values = {
        "role_family": "applied AI",
        "ai_scope": "LLM evaluation",
        "product_customer_scope": "customers",
        "ownership_scope": "own launch",
        "infrastructure_scope": "model serving",
        "work_arrangement_terms": ["remote"],
        "compensation_terms": "salary range",
        "work_authorization_requirement": [],
        "required_skills": ["Python"],
        "minimum_experience_years": 3,
        "applied_ai_requirements": ["evaluation"],
    }
    extractor = FakeFactExtractor(
        outputs=[
            ExtractedFact(
                fact_id=fact.id,
                value=values[fact.id],
                evidence=["Senior Python Engineer"],
                provider="fake-facts",
                model="facts-model-v1",
            )
            for fact in rubric.facts
        ]
    )

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "question_id": question.id,
                        "question_version": question.version,
                        "noul": 0 if question.id == "pure_infrastructure" else 1,
                    }
                    for question in rubric.questions
                ]
            },
        )

    store = LocalStore.open(tmp_path / "jev.sqlite3")
    store.initialize()
    store.save_profile(CandidateProfile())
    store.save_rubric(rubric)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service = JobAnalysisService(
            store,
            extractor,
            JevSignalProvider(api_key="test-key", client=client),
            id_factory=lambda: ANALYSIS_ID,
        )
        record = service.analyze_job(JOB_DESCRIPTION)

    assert record.result is not None
    contributions = {
        item.subject_id: item for item in record.result.preference_contributions
    }
    signal_dimensions = {
        "applied_ai_depth",
        "product_customer_proximity",
        "infrastructure_balance",
    }
    assert signal_dimensions <= contributions.keys()
    assert all(
        any(ref.startswith("signal:jev:jev-latest:") for ref in item.audit_support)
        for name, item in contributions.items()
        if name in signal_dimensions
    )


def test_patch_and_recompute_record_distinct_rubric_revision(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path, profile=CandidateProfile(), rubric=_rubric()
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    current = apply_rubric_patch(
        store.load_rubric(),
        RubricPatch(dimension_updates={"role_fit": {"weight": 2}}),
    )
    store.save_rubric(current)

    recomputed = service.recompute_decision(original.id)

    assert original.result is not None and original.trace is not None
    assert recomputed.result is not None and recomputed.trace is not None
    assert recomputed.result.rubric_revision == current.revision
    assert recomputed.trace.rubric_revision == current.revision
    assert recomputed.result.rubric_revision != original.result.rubric_revision


def test_recompute_uses_current_rubric_missing_and_stale_semantics(tmp_path) -> None:
    ids = iter([ANALYSIS_ID, UNUSED_ID])
    signal_provider = FakeSignalProvider(
        outputs=[
            DecisionSignal(
                question_id="role_fit",
                question_version="question-v1",
                probability=1,
                evidence=["Remote role"],
                provider="fake-signals",
                model="signals-model-v1",
            )
        ]
    )
    service, store, extractor, signal_provider = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        signal_provider=signal_provider,
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    current = _rubric(version="rubric-v2").model_copy(
        update={
            "facts": [
                FactDefinition(
                    id="new_fact",
                    name="New fact",
                    definition="A newly configured job fact.",
                    data_type="text",
                    extraction_instructions="Extract the new fact.",
                )
            ],
            "dimensions": [
                _rubric().dimensions[0].model_copy(
                    update={
                        "fact_ids": ["new_fact"],
                        "signal_ids": ["role_fit", "new_signal"],
                    }
                )
            ],
            "questions": [
                DecisionQuestion(
                    id="role_fit",
                    version="question-v2",
                    question="Does this current role fit?",
                ),
                DecisionQuestion(
                    id="new_signal",
                    version="question-v1",
                    question="Is this newly configured signal favorable?",
                ),
            ]
        }
    )
    store.save_rubric(current)

    recomputed = service.recompute_decision(original.id)

    assert recomputed.status == "success"
    assert recomputed.signals == original.signals
    assert recomputed.result is not None
    assert recomputed.result.needs_review is True
    assert {gap.subject_id for gap in recomputed.result.gaps} == {
        "new_fact",
        "role_fit",
        "new_signal",
    }
    assert recomputed.trace is not None
    assert {
        item
        for item in recomputed.trace.input_ids
        if not item.startswith("profile-ref:v1:")
    } == {"new_fact", "role_fit", "new_signal"}
    assert recomputed.trace.question_versions == {
        "role_fit": "question-v2",
        "new_signal": "question-v1",
    }
    assert recomputed.decision_surface is not None
    assert recomputed.decision_surface.fact_ids == ["new_fact"]
    assert recomputed.decision_surface.question_versions == {
        "role_fit": "question-v2",
        "new_signal": "question-v1",
    }
    assert recomputed.trace.input_ids == sorted(
        {
            *recomputed.decision_surface.fact_ids,
            *recomputed.decision_surface.question_versions,
            *recomputed.decision_surface.profile_refs,
        }
    )
    assert original.facts[0].evidence[0] not in recomputed.trace.evidence
    assert original.signals[0].evidence[0] in recomputed.trace.evidence
    assert len(extractor.calls) == 1
    assert len(signal_provider.calls) == 1


def test_recompute_rejects_stored_input_trace_integrity_mismatch(tmp_path) -> None:
    ids = iter([ANALYSIS_ID, FAILURE_ID])
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    tampered = original.model_copy(
        update={
            "trace": original.trace.model_copy(
                update={"input_ids": [*original.trace.input_ids, "unbound-input"]}
            )
        }
    )
    store.save_analysis(tampered)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.recompute_decision(original.id)

    assert exc_info.value.code == "trace_integrity_failed"
    assert store.list_analyses() == [tampered]


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


@pytest.mark.parametrize(
    "malformed_ref",
    [
        "wrong-scheme:v1:{job}:0:1:{quote}",
        "jd-evidence:v2:{job}:0:1:{quote}",
        "jd-evidence:v1:{short_job}:0:1:{quote}",
        "jd-evidence:v1:{upper_job}:0:1:{quote}",
        "jd-evidence:v1:{other_job}:0:1:{quote}",
        "jd-evidence:v1:{job}:-1:1:{quote}",
        "jd-evidence:v1:{job}:zero:1:{quote}",
        "jd-evidence:v1:{job}:2:1:{quote}",
        "jd-evidence:v1:{job}:0:0:{quote}",
        "jd-evidence:v1:{job}:00:1:{quote}",
        "jd-evidence:v1:{job}:0:01:{quote}",
        "jd-evidence:v1:{job}:{input_length}:{past_end}:{quote}",
        "jd-evidence:v1:{job}:0:1:{short_quote}",
        "jd-evidence:v1:{job}:0:1:{upper_quote}",
        "jd-evidence:v1:{job}:0:1:{quote}:trailing",
        "jd-evidence:v1:{job}:0:44:" + JOB_DESCRIPTION,
    ],
)
def test_recompute_rejects_malformed_or_payload_bearing_jd_refs(
    tmp_path,
    malformed_ref,
) -> None:
    ids = iter([ANALYSIS_ID, FAILURE_ID])
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: next(ids),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    job_digest = original.input_hash.removeprefix("sha256:")
    quote_digest = "a" * 64
    poisoned_ref = malformed_ref.format(
        job=job_digest,
        short_job=job_digest[:-1],
        upper_job=job_digest.upper(),
        other_job="0" * 64,
        input_length=len(JOB_DESCRIPTION),
        past_end=len(JOB_DESCRIPTION) + 1,
        quote=quote_digest,
        short_quote=quote_digest[:-1],
        upper_quote=quote_digest.upper(),
    )
    poisoned = original.model_copy(
        update={
            "signals": [
                original.signals[0].model_copy(update={"evidence": [poisoned_ref]})
            ]
        }
    )
    store.save_analysis(poisoned)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.recompute_decision(original.id)

    assert exc_info.value.code == "trace_integrity_failed"
    assert store.list_analyses() == [poisoned]


@pytest.mark.parametrize(
    "corruption",
    [
        "trace_id_mismatch",
        "unsafe_ids",
        "outbound_class",
        "input_length",
        "unbound_input",
        "extra_profile_input",
        "self_declared_question",
        "extra_evidence",
        "providers",
        "models",
        "prompt_versions",
        "rubric_version",
        "rubric_revision",
        "result_version",
        "jd_ref",
        "profile_ref",
    ],
)
def test_trace_read_rejects_malformed_stored_privacy_or_integrity_data(
    tmp_path,
    corruption,
) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    trace = original.trace
    target_id = original.id
    input_length = original.input_length
    if corruption == "trace_id_mismatch":
        trace = trace.model_copy(update={"analysis_id": OTHER_ID})
    elif corruption == "unsafe_ids":
        target_id = "private resume secret@example.com"
        trace = trace.model_copy(update={"analysis_id": target_id})
    elif corruption == "outbound_class":
        trace = trace.model_copy(
            update={"outbound_data_classes": [DataClass.RESTRICTED_PERSONAL_DATA]}
        )
    elif corruption == "input_length":
        input_length = 1
    elif corruption == "unbound_input":
        trace = trace.model_copy(
            update={
                "input_ids": [
                    *trace.input_ids,
                    "PRIVATE RESUME secret@example.com",
                ]
            }
        )
    elif corruption == "extra_profile_input":
        trace = trace.model_copy(
            update={
                "input_ids": sorted(
                    [*trace.input_ids, "profile-ref:v1:" + "0" * 64]
                )
            }
        )
    elif corruption == "self_declared_question":
        trace = trace.model_copy(
            update={
                "input_ids": sorted([*trace.input_ids, "invented_question"]),
                "question_versions": {
                    **trace.question_versions,
                    "invented_question": "invented-v1",
                },
            }
        )
    elif corruption == "extra_evidence":
        trace = trace.model_copy(
            update={
                "evidence": [
                    *trace.evidence,
                    "jd-evidence:v1:"
                    + original.input_hash.removeprefix("sha256:")
                    + ":0:1:"
                    + "a" * 64,
                ]
            }
        )
    elif corruption == "providers":
        trace = trace.model_copy(
            update={"providers": {**trace.providers, "facts": "PRIVATE API KEY"}}
        )
    elif corruption == "models":
        trace = trace.model_copy(
            update={"model_versions": {**trace.model_versions, "facts": "wrong"}}
        )
    elif corruption == "prompt_versions":
        trace = trace.model_copy(
            update={"prompt_versions": {**trace.prompt_versions, "facts": "wrong"}}
        )
    elif corruption == "rubric_version":
        trace = trace.model_copy(update={"rubric_version": "PRIVATE RESUME"})
    elif corruption == "rubric_revision":
        trace = trace.model_copy(update={"rubric_revision": "PRIVATE RESUME"})
    elif corruption == "result_version":
        trace = trace.model_copy(update={"result_version": "wrong"})
    elif corruption == "jd_ref":
        trace = trace.model_copy(
            update={
                "evidence": [
                    "jd-evidence:v1:"
                    + original.input_hash.removeprefix("sha256:")
                    + f":0:{len(JOB_DESCRIPTION)}:{JOB_DESCRIPTION}"
                ]
            }
        )
    else:
        trace = trace.model_copy(
            update={"evidence": ["profile-ref:v1:private@example.com"]}
        )
    poisoned = original.model_copy(
        update={
            "id": target_id,
            "input_length": input_length,
            "trace": trace,
        }
    )
    store.save_analysis(poisoned)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.get_decision_trace(target_id)

    assert exc_info.value.code == "trace_integrity_failed"
    assert str(exc_info.value) == (
        "Stored decision trace failed privacy and integrity validation."
    )
    assert "private" not in str(exc_info.value)
    assert JOB_DESCRIPTION not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_empty_stage_success_rejects_zero_input_length_on_trace_read(tmp_path) -> None:
    rubric = Rubric(
        version="empty-v1",
        recommendation_thresholds={
            RecommendationLevel.PRIORITY_APPLY: 0.8,
            RecommendationLevel.APPLY_AFTER_REVIEW: 0.6,
            RecommendationLevel.HOLD: 0.4,
            RecommendationLevel.SKIP: 0,
        },
    )
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=rubric,
    )
    original = service.analyze_job(JOB_DESCRIPTION)
    tampered = original.model_copy(update={"input_length": 0})
    store.save_analysis(tampered)

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.get_decision_trace(original.id)

    assert exc_info.value.code == "trace_integrity_failed"


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
    ids = iter([ANALYSIS_ID, FAILURE_ID])
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
    assert records[1].id == FAILURE_ID
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


def test_successful_recompute_does_not_allocate_unused_failure_id(tmp_path) -> None:
    generated_ids = []

    def same_id() -> str:
        generated_ids.append(ANALYSIS_ID)
        return ANALYSIS_ID

    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=same_id,
    )
    original = service.analyze_job(JOB_DESCRIPTION)

    recomputed = service.recompute_decision(original.id)

    assert recomputed.id == original.id
    assert generated_ids == [ANALYSIS_ID]
    assert store.list_analyses() == [recomputed]


def test_failed_recompute_id_collision_reports_missing_audit_record(tmp_path) -> None:
    service, store, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        id_factory=lambda: ANALYSIS_ID,
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

    with pytest.raises(AnalysisServiceError) as exc_info:
        service.recompute_decision(original.id)

    assert exc_info.value.code == "failure_audit_persistence_failed"
    assert str(exc_info.value) == (
        "Analysis failed, and its audit record could not be persisted safely."
    )
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
        else "Analysis failed, and its audit record could not be persisted safely.",
    ) as exc_info:
        service.analyze_job(JOB_DESCRIPTION)

    assert "api-key-secret" not in str(exc_info.value)
    if provider_fails:
        assert exc_info.value.code == "failure_audit_persistence_failed"
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


@pytest.mark.parametrize(
    "rubric",
    [
        Rubric(
            version="empty-v1",
            recommendation_thresholds={
                RecommendationLevel.PRIORITY_APPLY: 0.8,
                RecommendationLevel.APPLY_AFTER_REVIEW: 0.6,
                RecommendationLevel.HOLD: 0.4,
                RecommendationLevel.SKIP: 0,
            },
        ),
        _rubric().model_copy(update={"dimensions": [], "questions": []}),
        _rubric().model_copy(update={"facts": []}),
    ],
)
def test_unconfigured_provider_stages_are_skipped_and_empty_sets_are_complete(
    tmp_path,
    rubric,
) -> None:
    service, _, extractor, signals = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=rubric,
    )

    record = service.analyze_job(JOB_DESCRIPTION)

    assert record.status == "success"
    assert len(extractor.calls) == (1 if rubric.facts else 0)
    assert len(signals.calls) == (1 if rubric.questions else 0)
    expected_stages = {
        *({"facts"} if rubric.facts else set()),
        *({"signals"} if rubric.questions else set()),
    }
    assert set(record.trace.providers) == expected_stages
    assert set(record.trace.model_versions) == expected_stages
    assert set(record.trace.prompt_versions) == expected_stages


def test_non_jd_signal_rationale_is_discarded_not_hashed(tmp_path) -> None:
    rationale = "model-only chain of thought rationale"
    signals = FakeSignalProvider(
        outputs=[
            DecisionSignal(
                question_id="role_fit",
                question_version="question-v1",
                probability=1,
                evidence=[rationale],
                provider="fake-signals",
                model="signals-model-v1",
            )
        ]
    )
    service, _, _, _ = _service(
        tmp_path,
        profile=CandidateProfile(),
        rubric=_rubric(),
        signal_provider=signals,
    )

    record = service.analyze_job(JOB_DESCRIPTION)

    assert record.signals[0].evidence == []
    assert record.result is not None
    assert record.result.needs_review is True
    assert record.trace is not None
    assert rationale not in record.model_dump_json()
    assert "provider-evidence:" not in record.model_dump_json()


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
    ids = iter([ANALYSIS_ID, FAILURE_ID])
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

    assert exc_info.value.code == "failure_audit_persistence_failed"
    assert str(exc_info.value) == (
        "Analysis failed, and its audit record could not be persisted safely."
    )
    assert store.list_analyses() == [original]
