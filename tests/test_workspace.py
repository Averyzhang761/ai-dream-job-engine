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
from ai_dream_job_engine.workspace import WorkspaceApplication, WorkspaceError


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
) -> AnalysisRecord:
    gaps = gaps or []
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
        rubric_revision="sha256:fixture",
        prompt_version="facts:fixture-prompt-v1",
        model_versions={"facts": "fixture-v1"},
    )
    return AnalysisRecord(
        id=analysis_id,
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
            rubric_revision="sha256:fixture",
            prompt_versions={"facts": "fixture-prompt-v1"},
            result_version="1",
            outbound_data_classes=[DataClass.PUBLIC_JOB_DATA],
        ),
        decision_surface=DecisionSurfaceManifest(fact_ids=fact_ids),
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
    assert all(detail.gaps[kind][0].job_evidence == [JOB_REF] for kind in GapKind)
    assert detail.preference_contributions[0].job_evidence == [JOB_REF]
    assert detail.qualification_contributions[0].job_evidence == [JOB_REF]
    assert detail.trace_evidence == [JOB_REF]


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
