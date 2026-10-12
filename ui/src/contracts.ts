export type RecommendationLevel =
  | "priority_apply"
  | "apply_after_review"
  | "hold"
  | "skip";

export type JobReviewStatus =
  | "unreviewed"
  | "kept"
  | "ignored"
  | "ready_to_apply";

export type GapKind =
  | "preference_mismatch"
  | "hard_blocker"
  | "capability_gap"
  | "evidence_gap";

export interface Gap {
  kind: GapKind;
  subject_id: string;
  message: string;
  input_ids: string[];
  job_evidence: string[];
  candidate_evidence: string[];
  missing_evidence: boolean;
}

export interface CandidateClaim {
  id: string;
  claim_type: string;
  value: unknown;
  value_is_exhaustive: boolean;
  source: "resume_evidence" | "user_confirmed" | "agent_suggested";
  confirmed: boolean;
  source_quote: string | null;
  provenance: string | null;
  source_agent: string | null;
  observed_at: string | null;
  confidence: number | null;
  import_origin: ClaimImportOrigin | null;
}

export interface ClaimImportOrigin {
  source_agent: string;
  candidate_id: string;
  provenance: string;
  claim_type: string;
  candidate_value: unknown;
  confirmed_value: unknown;
}

export interface CandidateProfile {
  id: string;
  claims: CandidateClaim[];
  import_receipts: Array<Record<string, unknown>>;
  memory_connections?: Array<{
    source_agent: string;
    scope: string[];
    connected: boolean;
  }>;
  version: string;
  storage_revision?: number;
  updated_at: string;
}

export type FactDataType = "boolean" | "number" | "text" | "enum" | "list";
export type CandidateComparison =
  | "none"
  | "exact"
  | "allowed_alternatives"
  | "required_subset"
  | "candidate_at_least";

export interface FactDefinition {
  id: string;
  name: string;
  definition: string;
  data_type: FactDataType;
  extraction_instructions: string;
  requires_evidence: boolean;
  missing_policy: "unknown" | "default";
  default_value: unknown | null;
  enum_values: string[];
  candidate_comparison: CandidateComparison;
  qualification_weight: number;
}

export interface ValueDimension {
  id: string;
  name: string;
  definition: string;
  examples: string[];
  fact_ids: string[];
  signal_ids: string[];
  weight: number;
  scoring_direction: "higher_is_better" | "lower_is_better";
  positive_keywords: string[];
  negative_keywords: string[];
  hard_constraint: boolean;
  threshold: number | null;
  explanation_template: string | null;
}

export interface DecisionQuestion {
  id: string;
  version: string;
  question: string;
}

export interface Rubric {
  id: string;
  version: string;
  revision: string;
  facts: FactDefinition[];
  dimensions: ValueDimension[];
  questions: DecisionQuestion[];
  recommendation_thresholds: Record<RecommendationLevel, number>;
}

export interface RubricPatch {
  fact_updates?: Record<string, Partial<FactDefinition>>;
  dimension_updates?: Record<string, Partial<ValueDimension>>;
  threshold_updates?: Partial<Record<RecommendationLevel, number>>;
}

export interface ClaimDecision {
  claim_id: string;
  action: "accept" | "edit" | "reject";
  edited_value?: unknown;
}

export interface WorkspaceJobSummary {
  analysis_id: string;
  title: string | null;
  company: string | null;
  recommendation: RecommendationLevel;
  preference_score: number | null;
  qualification_score: number | null;
  top_unresolved_gap: Gap | null;
  review_status: JobReviewStatus;
  created_at: string;
  can_mark_ready_to_apply: boolean;
  stale_reasons?: Array<
    | "profile_missing"
    | "rubric_missing"
    | "profile_changed"
    | "rubric_changed"
  >;
}

export interface DecisionContribution {
  subject_id: string;
  score: number;
  weight: number;
  input_ids: string[];
  job_evidence: string[];
  candidate_evidence: string[];
  audit_support: string[];
}

export interface WorkspaceJobDetail {
  summary: WorkspaceJobSummary;
  gaps: Record<GapKind, Gap[]>;
  preference_contributions: DecisionContribution[];
  qualification_contributions: DecisionContribution[];
  trace_evidence: string[];
  ready_to_apply_blockers: ReadyToApplyBlocker[];
}

export interface ReadyToApplyBlocker {
  code:
    | "recommendation_not_priority"
    | "stale_analysis"
    | "hard_blocker"
    | "unresolved_evidence_gap";
  message: string;
  subject_ids: string[];
}

export interface WorkspaceSnapshot {
  profile_ready: boolean;
  rubric_ready: boolean;
  setup_actions: Array<"initialize_profile" | "initialize_rubric">;
  active_rubric: {
    id: string;
    version: string;
    revision: string;
  } | null;
  confirmed_claims: CandidateClaim[];
  jobs: WorkspaceJobSummary[];
}

export interface ClaimPreviewView {
  preview_id: string;
  kind: "profile_claims" | "agent_memory";
  candidates: CandidateClaim[];
  source_agent: string | null;
  scope: string[];
  added_claim_ids: string[];
  changed_claim_ids: string[];
  conflicting_claim_ids: string[];
}

export interface WorkspaceAppPayload {
  view: "jobs" | "claim_review";
  snapshot: WorkspaceSnapshot;
  claim_preview: ClaimPreviewView | null;
}

export interface ToolResultEnvelope {
  structuredContent?: unknown;
  _meta?: Record<string, unknown>;
  isError?: boolean;
}

export function privatePayload(value: ToolResultEnvelope): unknown {
  const namespace = value._meta?.aiDreamJobEngine;
  if (namespace && typeof namespace === "object" && "payload" in namespace) {
    return (namespace as { payload: unknown }).payload;
  }
  return value.structuredContent;
}

export function isWorkspaceAppPayload(value: unknown): value is WorkspaceAppPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<WorkspaceAppPayload>;
  return (
    (candidate.view === "jobs" || candidate.view === "claim_review") &&
    Boolean(candidate.snapshot) &&
    typeof candidate.snapshot?.profile_ready === "boolean" &&
    typeof candidate.snapshot?.rubric_ready === "boolean" &&
    Array.isArray(candidate.snapshot?.setup_actions) &&
    Array.isArray(candidate.snapshot?.confirmed_claims) &&
    Array.isArray(candidate.snapshot?.jobs)
  );
}
