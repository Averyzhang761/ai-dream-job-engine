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
    WorkspaceJobDetail,
    WorkspaceJobSummary,
    WorkspaceSnapshot,
)
from ai_dream_job_engine.service import AnalysisServiceError


ANALYSIS_NOT_FOUND = (
    "analysis_not_found",
    "Analysis was not found; choose a successful analysis ID.",
)
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
                _effective_review_status(record, reviews.get(record.id)),
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
        review = self._store_call(lambda: self._store.load_job_review(analysis_id))
        status = _effective_review_status(record, review)
        result = record.result
        trace = record.trace
        if result is None or trace is None:
            raise WorkspaceError(ANALYSIS_UNAVAILABLE)
        return WorkspaceJobDetail(
            summary=self._summary(record, status),
            gaps={
                kind: [gap.model_copy(deep=True) for gap in result.gaps if gap.kind is kind]
                for kind in GapKind
            },
            preference_contributions=[
                item.model_copy(deep=True) for item in result.preference_contributions
            ],
            qualification_contributions=[
                item.model_copy(deep=True) for item in result.qualification_contributions
            ],
            trace_evidence=list(trace.evidence),
        )

    def set_review_status(
        self, analysis_id: str, status: JobReviewStatus
    ) -> WorkspaceJobDetail:
        record = self._get_valid_record(analysis_id)
        if (
            status is JobReviewStatus.READY_TO_APPLY
            and not _can_mark_ready_to_apply(record)
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
        record: AnalysisRecord, review_status: JobReviewStatus
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
            can_mark_ready_to_apply=_can_mark_ready_to_apply(record),
        )

    @staticmethod
    def _store_call(operation):
        try:
            return operation()
        except WorkspaceError:
            raise
        except Exception:
            raise WorkspaceError(STORAGE_FAILURE) from None


def _can_mark_ready_to_apply(record: AnalysisRecord) -> bool:
    result = record.result
    return bool(
        result is not None
        and result.recommendation is RecommendationLevel.PRIORITY_APPLY
        and not any(
            gap.kind in {GapKind.HARD_BLOCKER, GapKind.EVIDENCE_GAP}
            for gap in result.gaps
        )
    )


def _effective_review_status(
    record: AnalysisRecord, review: JobReviewRecord | None
) -> JobReviewStatus:
    if review is None:
        return JobReviewStatus.UNREVIEWED
    if (
        review.status is JobReviewStatus.READY_TO_APPLY
        and review.decision_revision != record.decision_revision
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
