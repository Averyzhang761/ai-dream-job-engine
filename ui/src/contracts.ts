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
  import_origin: Record<string, unknown> | null;
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
  isError?: boolean;
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
