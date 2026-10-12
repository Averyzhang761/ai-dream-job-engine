from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.mcp_server import ApplicationError, LocalProfileApplication
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    DataClass,
    DecisionContribution,
    DecisionResult,
    DecisionSurfaceManifest,
    DecisionTrace,
    ExtractedFact,
    Gap,
    GapKind,
    JobReviewRecord,
    JobReviewStatus,
    RecommendationLevel,
    Rubric,
)
from ai_dream_job_engine.service import AnalysisServiceError
from ai_dream_job_engine.rubric import rubric_revision
from ai_dream_job_engine.workspace import (
    FULL_JD_NOT_PERSISTED,
    WorkspaceApplication,
    WorkspaceError,
)


NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
ANALYSIS_ID = "00000000-0000-4000-8000-000000000001"
OTHER_ANALYSIS_ID = "00000000-0000-4000-8000-000000000002"
JOB_TEXT = "x"
JOB_DIGEST = hashlib.sha256(JOB_TEXT.encode()).hexdigest()
JOB_REF = f"jd-evidence:v1:{JOB_DIGEST}:0:1:{JOB_DIGEST}"


class IntegrityAnalysisService:
    def __init__(self, store: LocalStore) -> None:
        self._store = store

    def get_decision_trace(self, analysis_id: str) -> DecisionTrace:
        record = next(
            (item for item in self._store.list_analyses() if item.id == analysis_id),
            None,
        )
        if record is None or record.trace is None:
            raise AnalysisServiceError("analysis_not_found", "unsafe detail")
        if record.trace.analysis_id != record.id:
            raise AnalysisServiceError("trace_integrity_failed", "unsafe detail")
        return record.trace


class NoPreviewProfileApplication:
    def get_claim_preview(self, preview_id: str):
        raise AssertionError("not used")


def _store(tmp_path) -> LocalStore:
    store = LocalStore.open(tmp_path / "workspace.sqlite3")
    store.initialize()
    return store


def _ready_store(tmp_path) -> LocalStore:
    store = _store(tmp_path)
    store.save_profile(CandidateProfile())
    store.save_rubric(Rubric(version="rubric-v1"))
    return store


def _workspace(store: LocalStore) -> WorkspaceApplication:
    return WorkspaceApplication(
        store,
        profile_application=NoPreviewProfileApplication(),
        analysis_application=IntegrityAnalysisService(store),
    )


def _analysis(
    *,
    analysis_id: str = ANALYSIS_ID,
    created_at: datetime = NOW,
    recommendation: RecommendationLevel = RecommendationLevel.PRIORITY_APPLY,
    gaps: list[Gap] | None = None,
    decision_revision: int = 1,
    rubric_revision_value: str | None = None,
    profile_refs: list[str] | None = None,
    evidence_catalog: dict[str, str] | None = None,
) -> AnalysisRecord:
    gaps = gaps or []
    rubric_revision_value = rubric_revision_value or rubric_revision(
        Rubric(version="rubric-v1")
    )
    fact_ids = ["company", "job_title", "requirement"]
    facts = [
        ExtractedFact(
            fact_id="company",
            value="Acme",
            evidence=[JOB_REF],
            provider="fixture",
            model="fixture-v1",
        ),
        ExtractedFact(
            fact_id="job_title",
            value="AI Engineer",
            evidence=[JOB_REF],
            provider="fixture",
            model="fixture-v1",
        ),
        ExtractedFact(
            fact_id="requirement",
            value=True,
            evidence=[JOB_REF],
            provider="fixture",
            model="fixture-v1",
        ),
    ]
    contribution = DecisionContribution(
        subject_id="preference",
        score=0.9,
        weight=1,
        input_ids=["requirement"],
        job_evidence=[JOB_REF],
    )
    result = DecisionResult(
        preference_score=0.9,
        qualification_score=0.8,
        recommendation=recommendation,
        gaps=gaps,
        preference_contributions=[contribution],
        qualification_contributions=[
            contribution.model_copy(update={"subject_id": "qualification"})
        ],
        rubric_version="rubric-v1",
        rubric_revision=rubric_revision_value,
        prompt_version="facts:fixture-prompt-v1",
        model_versions={"facts": "fixture-v1"},
    )
    return AnalysisRecord(
        id=analysis_id,
        decision_revision=decision_revision,
        input_hash=f"sha256:{JOB_DIGEST}",
        input_length=len(JOB_TEXT),
        facts=facts,
        result=result,
        trace=DecisionTrace(
            analysis_id=analysis_id,
            input_ids=fact_ids,
            evidence=[JOB_REF],
            providers={"facts": "fixture"},
            model_versions={"facts": "fixture-v1"},
            rubric_version="rubric-v1",
            rubric_revision=rubric_revision_value,
            prompt_versions={"facts": "fixture-prompt-v1"},
            result_version="1",
            outbound_data_classes=[DataClass.PUBLIC_JOB_DATA],
        ),
        decision_surface=DecisionSurfaceManifest(
            fact_ids=fact_ids,
            profile_refs=profile_refs or [],
        ),
        evidence_catalog=evidence_catalog or {},
        status="success",
        latency_ms=1,
        created_at=created_at,
    )


def _gap(kind: GapKind) -> Gap:
    return Gap(
        kind=kind,
        subject_id=f"{kind.value}-subject",
        message=f"{kind.value} message",
        input_ids=["requirement"],
        job_evidence=[JOB_REF],
        missing_evidence=kind is GapKind.EVIDENCE_GAP,
    )


def test_snapshot_reports_setup_actions_when_profile_and_rubric_are_missing(
    tmp_path,
) -> None:
    snapshot = _workspace(_store(tmp_path)).get_snapshot()

    assert snapshot.profile_ready is False
    assert snapshot.rubric_ready is False
    assert snapshot.setup_actions == ["initialize_profile", "initialize_rubric"]
    assert snapshot.active_rubric is None
    assert snapshot.confirmed_claims == []
    assert snapshot.jobs == []


def test_snapshot_lists_only_successful_integrity_valid_analyses(tmp_path) -> None:
    store = _store(tmp_path)
    older = _analysis(created_at=NOW - timedelta(hours=1))
    newer = _analysis(analysis_id=OTHER_ANALYSIS_ID, created_at=NOW)
    invalid = _analysis(
        analysis_id="00000000-0000-4000-8000-000000000003",
        created_at=NOW + timedelta(hours=1),
    )
    invalid = invalid.model_copy(
        update={
            "trace": invalid.trace.model_copy(update={"analysis_id": ANALYSIS_ID})
        }
    )
    failed = AnalysisRecord(
        id="00000000-0000-4000-8000-000000000004",
        input_hash=f"sha256:{JOB_DIGEST}",
        status="failed",
        latency_ms=1,
        created_at=NOW + timedelta(hours=2),
    )
    for record in (older, newer, invalid, failed):
        store.insert_analysis(record)

    snapshot = _workspace(store).get_snapshot()

    assert [item.analysis_id for item in snapshot.jobs] == [
        OTHER_ANALYSIS_ID,
        ANALYSIS_ID,
    ]


def test_snapshot_joins_review_status_without_mutating_analysis(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = _analysis()
    store.insert_analysis(analysis)
    store.save_job_review(
        JobReviewRecord(
            analysis_id=analysis.id,
            status=JobReviewStatus.KEPT,
            updated_at=NOW,
        )
    )

    snapshot = _workspace(store).get_snapshot()

    assert snapshot.jobs[0].review_status is JobReviewStatus.KEPT
    assert store.list_analyses() == [analysis]


def test_job_detail_preserves_all_four_gap_kinds_and_evidence(tmp_path) -> None:
    store = _store(tmp_path)
    gaps = [_gap(kind) for kind in GapKind]
    store.insert_analysis(_analysis(gaps=gaps))

    detail = _workspace(store).get_job_detail(ANALYSIS_ID)

    assert set(detail.gaps) == set(GapKind)
    assert [detail.gaps[kind][0].kind for kind in GapKind] == list(GapKind)
    assert all(
        detail.gaps[kind][0].job_evidence == [FULL_JD_NOT_PERSISTED]
        for kind in GapKind
    )
    assert detail.preference_contributions[0].job_evidence == [
        FULL_JD_NOT_PERSISTED
    ]
    assert detail.qualification_contributions[0].job_evidence == [
        FULL_JD_NOT_PERSISTED
    ]
    assert detail.trace_evidence == [FULL_JD_NOT_PERSISTED]


@pytest.mark.parametrize(
    ("recommendation", "gaps", "expected_codes", "expected_subject_ids"),
    [
        (
            RecommendationLevel.APPLY_AFTER_REVIEW,
            [],
            ["recommendation_not_priority"],
            [[]],
        ),
        (
            RecommendationLevel.PRIORITY_APPLY,
            [_gap(GapKind.HARD_BLOCKER)],
            ["hard_blocker"],
            [["hard_blocker-subject"]],
        ),
        (
            RecommendationLevel.PRIORITY_APPLY,
            [_gap(GapKind.EVIDENCE_GAP)],
            ["unresolved_evidence_gap"],
            [["evidence_gap-subject"]],
        ),
    ],
)
def test_job_detail_returns_server_owned_ready_blockers(
    tmp_path,
    recommendation,
    gaps,
    expected_codes,
    expected_subject_ids,
) -> None:
    store = _ready_store(tmp_path)
    store.insert_analysis(_analysis(recommendation=recommendation, gaps=gaps))

    detail = _workspace(store).get_job_detail(ANALYSIS_ID)

    assert [item.code for item in detail.ready_to_apply_blockers] == expected_codes
    assert [item.subject_ids for item in detail.ready_to_apply_blockers] == expected_subject_ids
    assert all(item.message for item in detail.ready_to_apply_blockers)


def test_job_detail_has_no_ready_blockers_when_server_gate_allows_ready(tmp_path) -> None:
    store = _ready_store(tmp_path)
    store.insert_analysis(_analysis())

    detail = _workspace(store).get_job_detail(ANALYSIS_ID)

    assert detail.summary.can_mark_ready_to_apply is True
    assert detail.ready_to_apply_blockers == []


def test_ready_to_apply_requires_priority_recommendation(tmp_path) -> None:
    store = _store(tmp_path)
    store.insert_analysis(
        _analysis(recommendation=RecommendationLevel.APPLY_AFTER_REVIEW)
    )

    with pytest.raises(WorkspaceError, match="review_status_ineligible"):
        _workspace(store).set_review_status(
            ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY
        )

    assert store.list_job_reviews() == []


def test_ready_to_apply_rejects_hard_blocker(tmp_path) -> None:
    store = _store(tmp_path)
    store.insert_analysis(_analysis(gaps=[_gap(GapKind.HARD_BLOCKER)]))

    with pytest.raises(WorkspaceError, match="review_status_ineligible"):
        _workspace(store).set_review_status(
            ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY
        )

    assert store.list_job_reviews() == []


def test_ready_to_apply_rejects_unresolved_evidence_gap(tmp_path) -> None:
    store = _store(tmp_path)
    store.insert_analysis(_analysis(gaps=[_gap(GapKind.EVIDENCE_GAP)]))

    with pytest.raises(WorkspaceError, match="review_status_ineligible"):
        _workspace(store).set_review_status(
            ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY
        )

    assert store.list_job_reviews() == []


def test_ready_to_apply_requires_new_action_after_each_decision_revision(
    tmp_path,
) -> None:
    store = _ready_store(tmp_path)
    store.insert_analysis(_analysis(decision_revision=1))
    workspace = _workspace(store)

    ready = workspace.set_review_status(
        ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY
    )
    assert ready.summary.review_status is JobReviewStatus.READY_TO_APPLY
    assert store.load_job_review(ANALYSIS_ID).decision_revision == 1

    store.save_analysis(
        _analysis(
            decision_revision=2,
            recommendation=RecommendationLevel.APPLY_AFTER_REVIEW,
        )
    )
    assert (
        workspace.get_job_detail(ANALYSIS_ID).summary.review_status
        is JobReviewStatus.UNREVIEWED
    )

    store.save_analysis(_analysis(decision_revision=3))
    assert (
        workspace.get_job_detail(ANALYSIS_ID).summary.review_status
        is JobReviewStatus.UNREVIEWED
    )

    ready_again = workspace.set_review_status(
        ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY
    )
    assert ready_again.summary.review_status is JobReviewStatus.READY_TO_APPLY
    assert store.load_job_review(ANALYSIS_ID).decision_revision == 3


def test_keep_and_ignore_are_always_explicit_user_actions(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = _analysis(
        recommendation=RecommendationLevel.SKIP,
        gaps=[_gap(GapKind.HARD_BLOCKER)],
    )
    store.insert_analysis(analysis)
    workspace = _workspace(store)

    kept = workspace.set_review_status(ANALYSIS_ID, JobReviewStatus.KEPT)
    ignored = workspace.set_review_status(ANALYSIS_ID, JobReviewStatus.IGNORED)

    assert kept.summary.review_status is JobReviewStatus.KEPT
    assert ignored.summary.review_status is JobReviewStatus.IGNORED
    assert store.load_job_review(ANALYSIS_ID).status is JobReviewStatus.IGNORED
    assert store.list_analyses() == [analysis]


def test_unknown_analysis_does_not_create_review_state(tmp_path) -> None:
    store = _store(tmp_path)

    with pytest.raises(WorkspaceError, match="analysis_not_found"):
        _workspace(store).set_review_status(
            "00000000-0000-4000-8000-000000000099",
            JobReviewStatus.KEPT,
        )

    assert store.list_job_reviews() == []


def _profile_workspace(
    store: LocalStore,
) -> tuple[WorkspaceApplication, LocalProfileApplication]:
    profile_application = LocalProfileApplication(store)
    return (
        WorkspaceApplication(
            store,
            profile_application=profile_application,
            analysis_application=IntegrityAnalysisService(store),
        ),
        profile_application,
    )


def _resume_claim() -> CandidateClaim:
    return CandidateClaim(
        id="resume-claim-1",
        claim_type="python",
        value="five years",
        source=ClaimSource.RESUME_EVIDENCE,
        source_quote="Built Python services",
    )


def test_expired_claim_preview_returns_safe_error(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_profile(CandidateProfile())
    workspace, profile_application = _profile_workspace(store)
    issued = profile_application.review_profile_claims([_resume_claim()])
    profile_application.confirm_profile_claims(
        issued.preview_id,
        [ClaimDecision(claim_id="resume-claim-1", action="reject")],
    )

    with pytest.raises(ApplicationError, match="profile_preview_invalid"):
        workspace.get_claim_preview(issued.preview_id)


def test_expired_agent_memory_claim_preview_returns_safe_error(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_profile(CandidateProfile())
    workspace, profile_application = _profile_workspace(store)
    candidate = CandidateClaim(
        id="memory-claim-1",
        claim_type="python",
        value="five years",
        source=ClaimSource.AGENT_SUGGESTED,
        source_agent="career-agent",
        provenance="agent-memory://career-agent/memory-claim-1",
    )
    issued = profile_application.preview_agent_memory_import(
        "career-agent", ["skills"], [candidate]
    )
    profile_application.confirm_agent_memory_import(
        issued.preview_id,
        [ClaimDecision(claim_id="memory-claim-1", action="reject")],
    )

    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        workspace.get_claim_preview(issued.preview_id)


def test_claim_preview_does_not_mutate_profile(tmp_path) -> None:
    store = _store(tmp_path)
    original = CandidateProfile()
    store.save_profile(original)
    workspace, profile_application = _profile_workspace(store)
    issued = profile_application.review_profile_claims([_resume_claim()])

    first = workspace.get_claim_preview(issued.preview_id)
    first.candidates[0].value = "caller mutation"
    second = workspace.get_claim_preview(issued.preview_id)

    assert first.kind == "profile_claims"
    assert second.candidates[0].value == "five years"
    assert store.load_profile() == original


def test_missing_profile_and_rubric_make_analysis_stale_and_block_ready(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_analysis(_analysis())
    workspace = _workspace(store)

    snapshot = workspace.get_snapshot()
    detail = workspace.get_job_detail(ANALYSIS_ID)

    assert set(snapshot.jobs[0].stale_reasons) == {"profile_missing", "rubric_missing"}
    assert detail.summary.can_mark_ready_to_apply is False
    assert detail.ready_to_apply_blockers[0].code == "stale_analysis"
    with pytest.raises(WorkspaceError, match="review_status_ineligible"):
        workspace.set_review_status(ANALYSIS_ID, JobReviewStatus.READY_TO_APPLY)


def test_changed_profile_evidence_marks_analysis_stale_after_restart(tmp_path) -> None:
    store = _store(tmp_path)
    original_profile = CandidateProfile()
    rubric = Rubric(version="rubric-v1")
    store.save_profile(original_profile)
    store.save_rubric(rubric)
    record = _analysis(
        rubric_revision_value=rubric_revision(rubric),
        profile_refs=[],
    )
    store.save_analysis(record)
    changed = CandidateProfile(
        claims=[
            CandidateClaim(
                id="new-evidence",
                claim_type="python",
                value="production Python",
                source=ClaimSource.USER_CONFIRMED,
                confirmed=True,
            )
        ]
    )
    store.save_profile(changed)

    restarted = _workspace(LocalStore.open(tmp_path / "workspace.sqlite3"))
    detail = restarted.get_job_detail(ANALYSIS_ID)

    assert "profile_changed" in detail.summary.stale_reasons
    assert detail.summary.can_mark_ready_to_apply is False


def test_job_detail_resolves_real_service_references_for_app(tmp_path) -> None:
    store = _store(tmp_path)
    claim = CandidateClaim(
        id="python-evidence",
        claim_type="python",
        value="production Python",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        source_quote="Built production Python services",
    )
    profile = CandidateProfile(claims=[claim])
    rubric = Rubric(version="rubric-v1")
    store.save_profile(profile)
    store.save_rubric(rubric)
    from ai_dream_job_engine.service import current_decision_surface_manifest

    surface = current_decision_surface_manifest(profile, rubric)
    record = _analysis(
        rubric_revision_value=rubric_revision(rubric),
        profile_refs=surface.profile_refs,
        evidence_catalog={JOB_REF: "x"},
    )
    from ai_dream_job_engine.service import candidate_evidence_catalog
    profile_ref = next(iter(candidate_evidence_catalog(profile)), None)
    if profile_ref:
        contribution = record.result.preference_contributions[0].model_copy(
            update={"candidate_evidence": [profile_ref]}
        )
        record = record.model_copy(
            update={
                "result": record.result.model_copy(
                    update={"preference_contributions": [contribution]}
                )
            }
        )
    store.save_analysis(record)

    detail = _workspace(store).get_job_detail(ANALYSIS_ID)

    assert detail.preference_contributions[0].job_evidence == ["x"]
    if profile_ref:
        assert detail.preference_contributions[0].candidate_evidence == [
            "Built production Python services"
        ]
