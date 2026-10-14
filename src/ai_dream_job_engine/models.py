"""Validated domain contracts for the AI Dream Job Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DomainModel(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}


class ClaimSource(StrEnum):
    RESUME_EVIDENCE = "resume_evidence"
    USER_CONFIRMED = "user_confirmed"
    AGENT_SUGGESTED = "agent_suggested"


class FactDataType(StrEnum):
    BOOLEAN = "boolean"
    NUMBER = "number"
    TEXT = "text"
    ENUM = "enum"
    LIST = "list"


class GapKind(StrEnum):
    PREFERENCE_MISMATCH = "preference_mismatch"
    HARD_BLOCKER = "hard_blocker"
    CAPABILITY_GAP = "capability_gap"
    EVIDENCE_GAP = "evidence_gap"


class RecommendationLevel(StrEnum):
    PRIORITY_APPLY = "priority_apply"
    APPLY_AFTER_REVIEW = "apply_after_review"
    HOLD = "hold"
    SKIP = "skip"


class JobReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    KEPT = "kept"
    IGNORED = "ignored"
    READY_TO_APPLY = "ready_to_apply"


class SignalStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class DataClass(StrEnum):
    PUBLIC_JOB_DATA = "public_job_data"
    SELECTED_CANDIDATE_CLAIMS = "selected_candidate_claims"
    RESTRICTED_PERSONAL_DATA = "restricted_personal_data"


class ScoringDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class CriterionCategory(StrEnum):
    ROLE_CONTENT = "role_content"
    WORK_CONDITIONS = "work_conditions"
    GROWTH_TEAM = "growth_team"
    COMPENSATION_CONSTRAINTS = "compensation_constraints"


class RequirementLevel(StrEnum):
    IGNORED = "ignored"
    NORMAL = "normal"
    STRONG = "strong"
    REQUIRED = "required"


class CriterionOrigin(StrEnum):
    TEMPLATE = "template"
    CUSTOM = "custom"


class ClaimImportOrigin(DomainModel):
    source_agent: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    candidate_value: Any
    confirmed_value: Any


class ClaimImportReceipt(DomainModel):
    source_agent: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    confirmed_claim_id: str = Field(min_length=1)
    confirmed_value: Any
    origin: ClaimImportOrigin

    @model_validator(mode="after")
    def identity_matches_origin(self) -> ClaimImportReceipt:
        if self.source_agent != self.origin.source_agent:
            raise ValueError("receipt source_agent must match origin")
        if self.candidate_id != self.origin.candidate_id:
            raise ValueError("receipt candidate_id must match origin")
        return self


class AgentMemoryConnection(DomainModel):
    source_agent: str = Field(min_length=1)
    scope: list[str] = Field(min_length=1)
    connected: bool = True
    generation: int = Field(default=0, ge=0)


class CandidateClaim(DomainModel):
    id: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    value: Any
    value_is_exhaustive: bool = True
    source: ClaimSource
    confirmed: bool = False
    source_quote: str | None = None
    provenance: str | None = None
    source_agent: str | None = None
    observed_at: datetime | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    import_origin: ClaimImportOrigin | None = None

    @model_validator(mode="after")
    def agent_suggestions_are_not_directly_confirmed(self) -> CandidateClaim:
        if self.source is ClaimSource.AGENT_SUGGESTED and self.confirmed:
            raise ValueError(
                "agent_suggested claims cannot be confirmed directly; "
                "confirmation creates a user_confirmed claim"
            )
        if self.import_origin is not None and (
            self.source is not ClaimSource.USER_CONFIRMED or not self.confirmed
        ):
            raise ValueError("import_origin requires a confirmed user_confirmed claim")
        return self


class CandidateProfile(DomainModel):
    id: str = "default"
    claims: list[CandidateClaim] = Field(default_factory=list)
    import_receipts: list[ClaimImportReceipt] = Field(default_factory=list)
    memory_connections: list[AgentMemoryConnection] = Field(default_factory=list)
    version: str = "1"
    storage_revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactDefinition(DomainModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    data_type: FactDataType
    extraction_instructions: str = Field(min_length=1)
    requires_evidence: bool = True
    missing_policy: Literal["unknown", "default"] = "unknown"
    default_value: Any | None = None
    enum_values: list[str] = Field(default_factory=list)
    candidate_comparison: Literal[
        "none",
        "exact",
        "allowed_alternatives",
        "required_subset",
        "candidate_at_least",
    ] = "none"
    qualification_weight: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def configured_defaults_have_a_value(self) -> FactDefinition:
        if self.missing_policy == "default" and self.default_value is None:
            raise ValueError("default_value is required when missing_policy is default")
        if self.candidate_comparison == "none" and self.qualification_weight != 0:
            raise ValueError(
                "qualification_weight must be zero without candidate comparison"
            )
        if self.candidate_comparison != "none" and self.qualification_weight <= 0:
            raise ValueError(
                "qualification_weight must be positive for candidate comparison"
            )
        if self.candidate_comparison in {
            "allowed_alternatives",
            "required_subset",
        } and self.data_type is not FactDataType.LIST:
            raise ValueError(
                f"{self.candidate_comparison} requires list data_type"
            )
        if (
            self.candidate_comparison == "candidate_at_least"
            and self.data_type is not FactDataType.NUMBER
        ):
            raise ValueError("candidate_at_least requires number data_type")
        return self


class ValueDimension(DomainModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    category: CriterionCategory
    desired_value: str = Field(min_length=1)
    requirement_level: RequirementLevel
    origin: CriterionOrigin
    examples: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    weight: float = Field(ge=0)
    legacy_weight_override: float | None = Field(default=None, ge=0)
    scoring_direction: ScoringDirection
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    hard_constraint: bool = False
    threshold: float | None = None
    explanation_template: str | None = None

    @model_validator(mode="before")
    @classmethod
    def add_user_facing_criterion_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("topic_id", payload.get("id"))
        payload.setdefault("category", CriterionCategory.ROLE_CONTENT)
        payload.setdefault("desired_value", payload.get("definition"))
        payload.setdefault("origin", CriterionOrigin.TEMPLATE)

        explicit_level = payload.get("requirement_level")
        if explicit_level is None:
            weight = payload.get("weight", 0)
            if payload.get("hard_constraint", False):
                payload["requirement_level"] = RequirementLevel.REQUIRED
            elif weight == 0:
                payload["requirement_level"] = RequirementLevel.IGNORED
            elif isinstance(weight, (int, float)) and weight >= 1.5:
                payload["requirement_level"] = RequirementLevel.STRONG
            else:
                payload["requirement_level"] = RequirementLevel.NORMAL
            return payload

        try:
            level = RequirementLevel(explicit_level)
        except ValueError:
            return payload
        projection = {
            RequirementLevel.IGNORED: (0.0, False),
            RequirementLevel.NORMAL: (1.0, False),
            RequirementLevel.STRONG: (1.5, False),
            RequirementLevel.REQUIRED: (1.0, True),
        }[level]
        if (
            "weight" in payload
            and payload["weight"] != projection[0]
            and payload.get("legacy_weight_override") != payload["weight"]
        ):
            raise ValueError("weight must match requirement_level")
        if (
            "hard_constraint" in payload
            and payload["hard_constraint"] is not projection[1]
        ):
            raise ValueError("hard_constraint must match requirement_level")
        payload.setdefault("weight", projection[0])
        payload.setdefault("hard_constraint", projection[1])
        return payload


class DecisionQuestion(DomainModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evaluation: Literal["provider", "local_keywords"] = "provider"
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def local_questions_have_explicit_keywords(self) -> DecisionQuestion:
        if (
            self.evaluation == "local_keywords"
            and not self.positive_keywords
            and not self.negative_keywords
        ):
            raise ValueError("local keyword questions require explicit keywords")
        return self


class PreferenceConflictKind(StrEnum):
    NONE = "none"
    DUPLICATE = "duplicate"
    DIRECT_CONFLICT = "direct_conflict"
    SEMANTIC_TENSION = "semantic_tension"
    MANUAL_REVIEW = "manual_review"


class CustomPreferenceProposal(DomainModel):
    statement: str = Field(min_length=1)
    topic_id: str | None = None
    category: CriterionCategory | None = None
    name: str | None = None
    desired_value: str | None = None
    requirement_level: RequirementLevel | None = None
    evidence_guidance: str | None = None
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    related_topic_ids: list[str] = Field(default_factory=list)


class PendingPreference(CustomPreferenceProposal):
    id: str = Field(min_length=1)
    conflict_kind: PreferenceConflictKind
    conflicting_dimension_id: str | None = None


class PreferenceResolution(DomainModel):
    action: Literal[
        "accept_new",
        "replace_existing",
        "keep_existing",
        "accept_tension",
    ]
    edited_proposal: CustomPreferenceProposal | None = None
    expected_storage_revision: int | None = Field(default=None, ge=0)


class Rubric(DomainModel):
    id: str = "default"
    version: str = Field(min_length=1)
    revision: str = ""
    storage_revision: int = Field(default=0, ge=0)
    facts: list[FactDefinition] = Field(default_factory=list)
    dimensions: list[ValueDimension] = Field(default_factory=list)
    questions: list[DecisionQuestion] = Field(default_factory=list)
    pending_preferences: list[PendingPreference] = Field(default_factory=list)
    recommendation_thresholds: dict[RecommendationLevel, float] = Field(
        default_factory=dict
    )


class EvidenceChunk(DomainModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def offsets_match_text(self) -> EvidenceChunk:
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("evidence chunk offsets must match its text")
        return self


class EvidenceDocument(DomainModel):
    text: str
    chunks: list[EvidenceChunk]

    @model_validator(mode="after")
    def chunks_are_a_lossless_partition(self) -> EvidenceDocument:
        if not self.text:
            if self.chunks:
                raise ValueError("empty evidence documents cannot have chunks")
            return self
        if not self.chunks:
            raise ValueError("evidence chunks must cover the document")

        seen_ids: set[str] = set()
        next_offset = 0
        for chunk in self.chunks:
            if chunk.id in seen_ids:
                raise ValueError("evidence chunk IDs must be unique")
            if chunk.start_offset != next_offset or chunk.end_offset > len(self.text):
                raise ValueError("evidence chunks must be ordered and contiguous")
            if self.text[chunk.start_offset : chunk.end_offset] != chunk.text:
                raise ValueError("evidence chunk text must match the document slice")
            seen_ids.add(chunk.id)
            next_offset = chunk.end_offset
        if next_offset != len(self.text):
            raise ValueError("evidence chunks must cover the document")
        return self


class EvidenceRef(DomainModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class ExtractedFact(DomainModel):
    fact_id: str = Field(min_length=1)
    value: Any | None = None
    is_unknown: bool = False
    evidence: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class DecisionSignal(DomainModel):
    question_id: str = Field(min_length=1)
    question_version: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0, le=1)
    status: SignalStatus = SignalStatus.AVAILABLE
    provider: str | None = None
    model: str | None = None
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def available_signals_have_a_probability(self) -> DecisionSignal:
        if self.status is SignalStatus.AVAILABLE and self.probability is None:
            raise ValueError("available signals require a probability")
        return self


class Gap(DomainModel):
    kind: GapKind
    subject_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    input_ids: list[str] = Field(min_length=1)
    job_evidence: list[str] = Field(default_factory=list)
    candidate_evidence: list[str] = Field(default_factory=list)
    missing_evidence: bool = False


class DecisionContribution(DomainModel):
    subject_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0)
    input_ids: list[str] = Field(min_length=1)
    job_evidence: list[str] = Field(default_factory=list)
    candidate_evidence: list[str] = Field(default_factory=list)
    audit_support: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_source_evidence(self) -> DecisionContribution:
        if not self.job_evidence and not self.candidate_evidence and not self.audit_support:
            raise ValueError("every contribution requires evidence or audit support")
        return self


class DecisionResult(DomainModel):
    preference_score: float | None = Field(default=None, ge=0, le=1)
    qualification_score: float | None = Field(default=None, ge=0, le=1)
    recommendation: RecommendationLevel
    gaps: list[Gap] = Field(default_factory=list)
    preference_contributions: list[DecisionContribution] = Field(default_factory=list)
    qualification_contributions: list[DecisionContribution] = Field(
        default_factory=list
    )
    rubric_version: str = Field(min_length=1)
    rubric_revision: str = Field(default="unversioned", min_length=1)
    prompt_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    result_version: str = "1"
    needs_review: bool = False

    @model_validator(mode="after")
    def every_gap_has_evidence(self) -> DecisionResult:
        gaps_without_evidence = [
            gap.subject_id
            for gap in self.gaps
            if not gap.job_evidence
            and not gap.candidate_evidence
            and not (
                gap.kind is GapKind.EVIDENCE_GAP and gap.missing_evidence
            )
        ]
        if gaps_without_evidence:
            joined_ids = ", ".join(gaps_without_evidence)
            raise ValueError(f"every gap requires evidence: {joined_ids}")
        return self


class OutboundDisclosure(DomainModel):
    provider: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)
    purpose: str = Field(min_length=1)
    data_class: DataClass
    confirmed: bool = False


class RubricPatch(DomainModel):
    expected_storage_revision: int | None = Field(default=None, ge=0)
    fact_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dimension_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    question_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    threshold_updates: dict[RecommendationLevel, float] = Field(default_factory=dict)


class ClaimDecision(DomainModel):
    claim_id: str = Field(min_length=1)
    action: Literal["accept", "edit", "reject"]
    edited_value: Any | None = None

    @model_validator(mode="after")
    def edited_value_matches_action(self) -> ClaimDecision:
        supplied = "edited_value" in self.model_fields_set
        if self.action == "edit" and not supplied:
            raise ValueError("edited_value must be supplied for edit")
        if self.action != "edit" and supplied:
            raise ValueError("edited_value is only allowed for edit")
        return self


class MemoryImportPreview(DomainModel):
    source_agent: str = Field(min_length=1)
    scope: list[str]
    candidates: list[CandidateClaim] = Field(default_factory=list)
    added_claim_ids: list[str] = Field(default_factory=list)
    changed_claim_ids: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)


class DecisionTrace(DomainModel):
    analysis_id: str = Field(min_length=1)
    input_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    providers: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    rubric_version: str = Field(min_length=1)
    rubric_revision: str = Field(default="unversioned", min_length=1)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    question_versions: dict[str, str] = Field(default_factory=dict)
    result_version: str = Field(min_length=1)
    outbound_data_classes: list[DataClass] = Field(default_factory=list)


class DecisionSurfaceManifest(DomainModel):
    fact_ids: list[str] = Field(default_factory=list)
    question_versions: dict[str, str] = Field(default_factory=dict)
    profile_refs: list[str] = Field(default_factory=list)


class AnalysisRecord(DomainModel):
    id: str = Field(min_length=1)
    decision_revision: int = Field(default=1, ge=1)
    source_analysis_id: str | None = None
    input_hash: str = Field(min_length=1)
    input_length: int | None = Field(default=None, ge=0)
    facts: list[ExtractedFact] = Field(default_factory=list)
    signals: list[DecisionSignal] = Field(default_factory=list)
    result: DecisionResult | None = None
    trace: DecisionTrace | None = None
    decision_surface: DecisionSurfaceManifest | None = None
    evidence_catalog: dict[str, str] = Field(default_factory=dict)
    status: Literal["success", "failed"]
    latency_ms: int = Field(ge=0)
    error_category: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobReviewRecord(DomainModel):
    analysis_id: str = Field(min_length=1)
    status: JobReviewStatus
    decision_revision: int | None = Field(default=None, ge=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("updated_at")
    @classmethod
    def updated_at_is_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class ActiveRubricMetadata(DomainModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    revision: str


class WorkspaceJobSummary(DomainModel):
    analysis_id: str = Field(min_length=1)
    title: str | None = None
    company: str | None = None
    recommendation: RecommendationLevel
    preference_score: float | None = Field(default=None, ge=0, le=1)
    qualification_score: float | None = Field(default=None, ge=0, le=1)
    top_unresolved_gap: Gap | None = None
    review_status: JobReviewStatus = JobReviewStatus.UNREVIEWED
    created_at: datetime
    can_mark_ready_to_apply: bool
    stale_reasons: list[
        Literal[
            "profile_missing",
            "rubric_missing",
            "profile_changed",
            "rubric_changed",
        ]
    ] = Field(default_factory=list)


class ReadyToApplyBlocker(DomainModel):
    code: Literal[
        "stale_analysis",
        "recommendation_not_priority",
        "hard_blocker",
        "unresolved_evidence_gap",
    ]
    message: str = Field(min_length=1)
    subject_ids: list[str] = Field(default_factory=list)


class WorkspaceJobDetail(DomainModel):
    summary: WorkspaceJobSummary
    gaps: dict[GapKind, list[Gap]]
    preference_contributions: list[DecisionContribution]
    qualification_contributions: list[DecisionContribution]
    trace_evidence: list[str]
    ready_to_apply_blockers: list[ReadyToApplyBlocker] = Field(default_factory=list)


class WorkspaceSnapshot(DomainModel):
    profile_ready: bool
    rubric_ready: bool
    setup_actions: list[Literal["initialize_profile", "initialize_rubric"]]
    active_rubric: ActiveRubricMetadata | None = None
    confirmed_claims: list[CandidateClaim]
    jobs: list[WorkspaceJobSummary]


class ClaimPreviewView(DomainModel):
    preview_id: str = Field(min_length=32)
    kind: Literal["profile_claims", "agent_memory"]
    candidates: list[CandidateClaim]
    source_agent: str | None = None
    scope: list[str] = Field(default_factory=list)
    added_claim_ids: list[str] = Field(default_factory=list)
    changed_claim_ids: list[str] = Field(default_factory=list)
    conflicting_claim_ids: list[str] = Field(default_factory=list)


class AnalysisSummary(DomainModel):
    analysis_id: str = Field(min_length=1)
    recommendation: RecommendationLevel | None = None
    job_title: str | None = None
    company: str | None = None
    created_at: datetime


class EvaluationCase(DomainModel):
    id: str = Field(min_length=1)
    job_description: str = Field(min_length=1)
    human_recommendation: RecommendationLevel
    human_gap_labels: list[GapKind]
    system_recommendation: RecommendationLevel | None = None
    system_gap_labels: list[GapKind] | None = None
    system_preference_score: float | None = Field(default=None, ge=0, le=1)
    system_qualification_score: float | None = Field(default=None, ge=0, le=1)
    human_top_five_usefulness: float | None = Field(default=None, ge=0, le=1)
    human_review_time_seconds: float | None = Field(default=None, ge=0)
    notes: str | None = None


class EvaluationReport(DomainModel):
    cases: list[EvaluationCase] = Field(default_factory=list)
    attempted_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    rerun_attempted_count: int = Field(ge=0)
    rerun_completed_count: int = Field(ge=0)
    rerun_failed_count: int = Field(ge=0)
    run_complete: bool
    recommendation_agreement: float | None = Field(default=None, ge=0, le=1)
    blocker_precision: float | None = Field(ge=0, le=1)
    gap_type_confusion_counts: dict[
        str, dict[str, Annotated[int, Field(ge=0)]]
    ]
    unsupported_fact_count: int = Field(ge=0)
    unsupported_signal_count: int = Field(ge=0)
    unsupported_gap_count: int = Field(ge=0)
    unsupported_contribution_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    rerun_stability: float | None = Field(default=None, ge=0, le=1)
    gap_label_agreement: float | None = Field(default=None, ge=0, le=1)
    top_five_case_ids: list[str] = Field(default_factory=list, max_length=5)
    evaluation_runtime_seconds: float = Field(ge=0)
    top_five_usefulness: float | None = Field(default=None, ge=0, le=1)
    review_time_seconds: float | None = Field(default=None, ge=0)
    failure_case_ids: list[str] = Field(default_factory=list)
    rerun_failure_case_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def run_counts_are_consistent(self) -> EvaluationReport:
        if self.attempted_count != len(self.cases):
            raise ValueError("attempted count must match report cases")
        if self.completed_count + self.failed_count != self.attempted_count:
            raise ValueError("completed and failed counts must match attempted count")
        if self.rerun_attempted_count != self.completed_count:
            raise ValueError("every completed first run requires a rerun")
        if (
            self.rerun_completed_count + self.rerun_failed_count
            != self.rerun_attempted_count
        ):
            raise ValueError("rerun completed and failed counts must match attempts")
        if self.run_complete != (
            self.failed_count == 0 and self.rerun_failed_count == 0
        ):
            raise ValueError("run_complete must reflect first-run and rerun failures")
        unsupported_total = (
            self.unsupported_fact_count
            + self.unsupported_signal_count
            + self.unsupported_gap_count
            + self.unsupported_contribution_count
        )
        if self.unsupported_claim_count != unsupported_total:
            raise ValueError("unsupported total must match its class counts")
        if len(self.top_five_case_ids) != min(5, self.completed_count):
            raise ValueError("top-five IDs must match completed case count")
        return self
