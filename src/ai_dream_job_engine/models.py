"""Validated domain contracts for the AI Dream Job Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class CandidateClaim(DomainModel):
    id: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    value: Any
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
    version: str = "1"
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
    candidate_comparison: Literal["none", "exact", "allowed_alternatives"] = "none"
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
        if (
            self.candidate_comparison == "allowed_alternatives"
            and self.data_type is not FactDataType.LIST
        ):
            raise ValueError("allowed_alternatives requires list data_type")
        return self


class ValueDimension(DomainModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    weight: float = Field(ge=0)
    scoring_direction: ScoringDirection
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    hard_constraint: bool = False
    threshold: float | None = None
    explanation_template: str | None = None


class DecisionQuestion(DomainModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    question: str = Field(min_length=1)


class Rubric(DomainModel):
    id: str = "default"
    version: str = Field(min_length=1)
    facts: list[FactDefinition] = Field(default_factory=list)
    dimensions: list[ValueDimension] = Field(default_factory=list)
    questions: list[DecisionQuestion] = Field(default_factory=list)
    recommendation_thresholds: dict[RecommendationLevel, float] = Field(
        default_factory=dict
    )


class EvidenceChunk(DomainModel):
    id: str = Field(min_length=1)
    text: str
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

    @model_validator(mode="after")
    def has_source_evidence(self) -> DecisionContribution:
        if not self.job_evidence and not self.candidate_evidence:
            raise ValueError("every contribution requires evidence")
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
    rubric_version: str = Field(min_length=1)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    result_version: str = Field(min_length=1)
    outbound_data_classes: list[DataClass] = Field(default_factory=list)


class AnalysisRecord(DomainModel):
    id: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    facts: list[ExtractedFact] = Field(default_factory=list)
    signals: list[DecisionSignal] = Field(default_factory=list)
    result: DecisionResult | None = None
    trace: DecisionTrace | None = None
    status: Literal["success", "failed"]
    latency_ms: int = Field(ge=0)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    notes: str | None = None


class EvaluationReport(DomainModel):
    cases: list[EvaluationCase] = Field(default_factory=list)
    recommendation_agreement: float = Field(ge=0, le=1)
    blocker_precision: float | None = Field(ge=0, le=1)
    gap_type_confusion_counts: dict[
        GapKind, dict[GapKind, Annotated[int, Field(ge=0)]]
    ]
    unsupported_claim_count: int = Field(ge=0)
    rerun_stability: float = Field(ge=0, le=1)
    top_five_usefulness: float = Field(ge=0, le=1)
    review_time_seconds: float = Field(ge=0)
    failure_case_ids: list[str] = Field(default_factory=list)
