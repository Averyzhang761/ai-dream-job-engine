"""Server-owned workspace queries for reviewed job decisions."""

from __future__ import annotations

from typing import Protocol

from ai_dream_job_engine.models import (
    ActiveRubricMetadata,
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimPreviewView,
    GapKind,
    JobReviewRecord,
    JobReviewStatus,
    RecommendationLevel,
    ReadyToApplyBlocker,
    WorkspaceJobDetail,
    WorkspaceJobSummary,
    WorkspaceSnapshot,
)
from ai_dream_job_engine.service import (
    AnalysisServiceError,
    candidate_evidence_catalog,
    current_decision_surface_manifest,
)
from ai_dream_job_engine.rubric import rubric_revision


ANALYSIS_NOT_FOUND = (
    "analysis_not_found",
    "Analysis was not found; choose a successful analysis ID.",
)
FULL_JD_NOT_PERSISTED = "完整 JD 原文未持久化，请重新打开来源核对"
ANALYSIS_UNAVAILABLE = (
    "analysis_unavailable",
    "Analysis is unavailable; choose a successful integrity-valid analysis.",
)
STORAGE_FAILURE = (
    "storage_failure",
    "Local data could not be read or saved safely.",
)
REVIEW_STATUS_INELIGIBLE = (
    "review_status_ineligible",
    "Only priority-apply jobs without hard blockers or unresolved evidence gaps can be ready to apply.",
)


class WorkspaceError(RuntimeError):
    """A fixed workspace error without storage or profile details."""

    def __init__(self, public_error: tuple[str, str]) -> None:
        self.code, self.safe_message = public_error
        super().__init__(f"{self.code}: {self.safe_message}")


class _Store(Protocol):
    def load_profile(self): ...

    def load_rubric(self): ...

    def list_analyses(self): ...

    def list_job_reviews(self): ...

    def load_job_review(self, analysis_id: str): ...

    def save_job_review(self, record: JobReviewRecord): ...


class _AnalysisApplication(Protocol):
    def get_decision_trace(self, analysis_id: str): ...


class WorkspaceApplication:
    def __init__(
        self,
        store: _Store,
        *,
        profile_application,
        analysis_application: _AnalysisApplication,
    ) -> None:
        self._store = store
        self._profile_application = profile_application
        self._analysis_application = analysis_application

    def get_snapshot(self) -> WorkspaceSnapshot:
        profile = self._store_call(self._store.load_profile)
        rubric = self._store_call(self._store.load_rubric)
        setup_actions = []
        if profile is None:
            setup_actions.append("initialize_profile")
        if rubric is None:
            setup_actions.append("initialize_rubric")

        confirmed_claims: list[CandidateClaim] = []
        if profile is not None:
            confirmed_claims = [
                claim.model_copy(deep=True)
                for claim in profile.claims
                if claim.confirmed
            ]
        active_rubric = None
        if rubric is not None:
            active_rubric = ActiveRubricMetadata(
                id=rubric.id,
                version=rubric.version,
                revision=rubric.revision,
            )

        reviews = {
            review.analysis_id: review
            for review in self._store_call(self._store.list_job_reviews)
        }
        jobs = [
            self._summary(
                record,
                _effective_review_status(
                    record,
                    reviews.get(record.id),
                    self._stale_reasons(record, profile, rubric),
                ),
                self._stale_reasons(record, profile, rubric),
            )
            for record in self._valid_analyses()
        ]
        jobs.sort(key=lambda item: (item.created_at, item.analysis_id), reverse=True)
        return WorkspaceSnapshot(
            profile_ready=profile is not None,
            rubric_ready=rubric is not None,
            setup_actions=setup_actions,
            active_rubric=active_rubric,
            confirmed_claims=confirmed_claims,
            jobs=jobs,
        )

    def get_job_detail(self, analysis_id: str) -> WorkspaceJobDetail:
        record = self._get_valid_record(analysis_id)
        profile = self._store_call(self._store.load_profile)
        rubric = self._store_call(self._store.load_rubric)
        stale_reasons = self._stale_reasons(record, profile, rubric)
        review = self._store_call(lambda: self._store.load_job_review(analysis_id))
        status = _effective_review_status(record, review, stale_reasons)
        result = record.result
        trace = record.trace
        if result is None or trace is None:
            raise WorkspaceError(ANALYSIS_UNAVAILABLE)
        evidence = dict(record.evidence_catalog)
        if profile is not None:
            evidence.update(candidate_evidence_catalog(profile))
        return WorkspaceJobDetail(
            summary=self._summary(record, status, stale_reasons),
            gaps={
                kind: [
                    _resolve_evidence(gap, evidence)
                    for gap in result.gaps
                    if gap.kind is kind
                ]
                for kind in GapKind
            },
            preference_contributions=[
                _resolve_evidence(item, evidence)
                for item in result.preference_contributions
            ],
            qualification_contributions=[
                _resolve_evidence(item, evidence)
                for item in result.qualification_contributions
            ],
            trace_evidence=[
                _resolve_evidence_ref(item, evidence) for item in trace.evidence
            ],
            ready_to_apply_blockers=_ready_to_apply_blockers(
                record, stale_reasons
            ),
        )

    def set_review_status(
        self, analysis_id: str, status: JobReviewStatus
    ) -> WorkspaceJobDetail:
        record = self._get_valid_record(analysis_id)
        profile = self._store_call(self._store.load_profile)
        rubric = self._store_call(self._store.load_rubric)
        stale_reasons = self._stale_reasons(record, profile, rubric)
        if (
            status is JobReviewStatus.READY_TO_APPLY
            and not _can_mark_ready_to_apply(record, stale_reasons)
        ):
            raise WorkspaceError(REVIEW_STATUS_INELIGIBLE)
        self._store_call(
            lambda: self._store.save_job_review(
                JobReviewRecord(
                    analysis_id=analysis_id,
                    status=status,
                    decision_revision=(
                        record.decision_revision
                        if status is JobReviewStatus.READY_TO_APPLY
                        else None
                    ),
                )
            )
        )
        return self.get_job_detail(analysis_id)

    def _get_valid_record(self, analysis_id: str) -> AnalysisRecord:
        records = self._store_call(self._store.list_analyses)
        record = next((item for item in records if item.id == analysis_id), None)
        if record is None:
            raise WorkspaceError(ANALYSIS_NOT_FOUND)
        if not self._is_valid(record):
            raise WorkspaceError(ANALYSIS_UNAVAILABLE)
        return record

    def get_claim_preview(self, preview_id: str) -> ClaimPreviewView:
        return self._profile_application.get_claim_preview(preview_id)

    def get_profile(self) -> CandidateProfile:
        return self._profile_application.get_profile()

    def _valid_analyses(self) -> list[AnalysisRecord]:
        return [
            record
            for record in self._store_call(self._store.list_analyses)
            if self._is_valid(record)
        ]

    def _is_valid(self, record: AnalysisRecord) -> bool:
        if record.status != "success" or record.result is None or record.trace is None:
            return False
        try:
            trace = self._analysis_application.get_decision_trace(record.id)
        except (AnalysisServiceError, WorkspaceError):
            return False
        return trace == record.trace

    @staticmethod
    def _summary(
        record: AnalysisRecord,
        review_status: JobReviewStatus,
        stale_reasons: list[str],
    ) -> WorkspaceJobSummary:
        result = record.result
        if result is None:
            raise WorkspaceError(ANALYSIS_UNAVAILABLE)
        return WorkspaceJobSummary(
            analysis_id=record.id,
            title=_text_fact(record, "job_title", "title"),
            company=_text_fact(record, "company", "company_name"),
            recommendation=result.recommendation,
            preference_score=result.preference_score,
            qualification_score=result.qualification_score,
            top_unresolved_gap=(
                result.gaps[0].model_copy(deep=True) if result.gaps else None
            ),
            review_status=review_status,
            created_at=record.created_at,
            can_mark_ready_to_apply=_can_mark_ready_to_apply(
                record, stale_reasons
            ),
            stale_reasons=stale_reasons,
        )

    @staticmethod
    def _stale_reasons(record, profile, rubric) -> list[str]:
        reasons: list[str] = []
        if profile is None:
            reasons.append("profile_missing")
        if rubric is None:
            reasons.append("rubric_missing")
        if rubric is not None and record.result is not None:
            if record.result.rubric_revision != rubric_revision(rubric):
                reasons.append("rubric_changed")
        if profile is not None and rubric is not None:
            current = current_decision_surface_manifest(profile, rubric)
            if (
                record.decision_surface is None
                or record.decision_surface.profile_refs != current.profile_refs
            ):
                reasons.append("profile_changed")
        return reasons

    @staticmethod
    def _store_call(operation):
        try:
            return operation()
        except WorkspaceError:
            raise
        except Exception:
            raise WorkspaceError(STORAGE_FAILURE) from None


def _can_mark_ready_to_apply(
    record: AnalysisRecord, stale_reasons: list[str]
) -> bool:
    return not _ready_to_apply_blockers(record, stale_reasons)


def _ready_to_apply_blockers(
    record: AnalysisRecord, stale_reasons: list[str]
) -> list[ReadyToApplyBlocker]:
    result = record.result
    if result is None:
        raise WorkspaceError(ANALYSIS_UNAVAILABLE)

    blockers: list[ReadyToApplyBlocker] = []
    if stale_reasons:
        blockers.append(
            ReadyToApplyBlocker(
                code="stale_analysis",
                message="候选人证据或岗位标准已变化，请重新计算",
                subject_ids=list(stale_reasons),
            )
        )
    if result.recommendation is not RecommendationLevel.PRIORITY_APPLY:
        blockers.append(
            ReadyToApplyBlocker(
                code="recommendation_not_priority",
                message="推荐等级不是优先申请",
            )
        )

    hard_blocker_ids = [
        gap.subject_id for gap in result.gaps if gap.kind is GapKind.HARD_BLOCKER
    ]
    if hard_blocker_ids:
        blockers.append(
            ReadyToApplyBlocker(
                code="hard_blocker",
                message="存在硬性阻断",
                subject_ids=hard_blocker_ids,
            )
        )

    evidence_gap_ids = [
        gap.subject_id for gap in result.gaps if gap.kind is GapKind.EVIDENCE_GAP
    ]
    if evidence_gap_ids:
        blockers.append(
            ReadyToApplyBlocker(
                code="unresolved_evidence_gap",
                message="存在未解决的证据差距",
                subject_ids=evidence_gap_ids,
            )
        )
    return blockers


def _effective_review_status(
    record: AnalysisRecord,
    review: JobReviewRecord | None,
    stale_reasons: list[str],
) -> JobReviewStatus:
    if review is None:
        return JobReviewStatus.UNREVIEWED
    if (
        review.status is JobReviewStatus.READY_TO_APPLY
        and (
            review.decision_revision != record.decision_revision
            or bool(stale_reasons)
        )
    ):
        return JobReviewStatus.UNREVIEWED
    return review.status


def _text_fact(record: AnalysisRecord, *fact_ids: str) -> str | None:
    for fact_id in fact_ids:
        fact = next((item for item in record.facts if item.fact_id == fact_id), None)
        if fact is not None and not fact.is_unknown and isinstance(fact.value, str):
            value = fact.value.strip()
            if value:
                return value
    return None


def _resolve_evidence(item, catalog: dict[str, str]):
    return item.model_copy(
        update={
            "job_evidence": [
                _resolve_evidence_ref(value, catalog)
                for value in item.job_evidence
            ],
            "candidate_evidence": [
                catalog.get(value, value) for value in item.candidate_evidence
            ],
        },
        deep=True,
    )


def _resolve_evidence_ref(value: str, catalog: dict[str, str]) -> str:
    resolved = catalog.get(value)
    if resolved is not None:
        return resolved
    if value.startswith("jd-evidence:v1:"):
        return FULL_JD_NOT_PERSISTED
    return value
