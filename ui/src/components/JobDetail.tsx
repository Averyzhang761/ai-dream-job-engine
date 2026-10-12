import type {
  Gap,
  GapKind,
  JobReviewStatus,
  WorkspaceJobDetail,
} from "../contracts";
import { formatScore } from "./JobList";
import { recommendationLabel, StatusBadge } from "./StatusBadge";

const gapSections: Array<{ kind: GapKind; label: string }> = [
  { kind: "hard_blocker", label: "硬性阻断" },
  { kind: "preference_mismatch", label: "偏好差距" },
  { kind: "capability_gap", label: "能力差距" },
  { kind: "evidence_gap", label: "证据差距" },
];

export function JobDetail({
  detail,
  busy,
  onReview,
}: {
  detail: WorkspaceJobDetail;
  busy: boolean;
  onReview: (status: JobReviewStatus) => void;
}) {
  const { summary } = detail;
  const title = summary.title ?? "未提供岗位名称";
  const company = summary.company ?? "未提供公司名称";

  return (
    <article className="job-detail" aria-label="岗位判断详情">
      <header className="job-detail-header">
        <div>
          <p className="job-detail-company">{company}</p>
          <h2>{title}</h2>
        </div>
        <StatusBadge reviewStatus={summary.review_status} />
      </header>

      <section className="decision-summary" aria-labelledby="decision-heading">
        <div className="decision-heading-row">
          <h3 id="decision-heading">判断</h3>
          <StatusBadge recommendation={summary.recommendation} />
        </div>
        <p>服务端建议：{recommendationLabel(summary.recommendation)}</p>
      </section>

      <section className="score-grid" aria-label="匹配分数">
        <div>
          <span>偏好匹配</span>
          <strong>{formatScore(summary.preference_score)}</strong>
        </div>
        <div>
          <span>胜任力证据</span>
          <strong>{formatScore(summary.qualification_score)}</strong>
        </div>
      </section>

      <div className="gap-groups">
        {gapSections.map(({ kind, label }) => (
          <GapSection key={kind} label={label} gaps={detail.gaps[kind]} />
        ))}
      </div>

      <EvidenceSection
        title="JD 与候选人证据"
        jobEvidence={detail.preference_contributions.flatMap((item) => item.job_evidence)}
        candidateEvidence={detail.qualification_contributions.flatMap(
          (item) => item.candidate_evidence,
        )}
        traceEvidence={detail.trace_evidence}
      />

      <section className="review-section" aria-labelledby="review-heading">
        <h3 id="review-heading">审核状态</h3>
        <div className="review-actions">
          <button type="button" disabled={busy} onClick={() => onReview("ignored")}>忽略</button>
          <button type="button" disabled={busy} onClick={() => onReview("kept")}>保留</button>
          <button
            type="button"
            className="review-action-primary"
            disabled={busy || !summary.can_mark_ready_to_apply}
            onClick={() => onReview("ready_to_apply")}
          >
            加入待申请
          </button>
        </div>
        {!summary.can_mark_ready_to_apply && (
          <p className="action-reason">服务端判定此岗位暂不可加入待申请</p>
        )}
      </section>
    </article>
  );
}

function GapSection({ label, gaps }: { label: string; gaps: Gap[] }) {
  return (
    <section className="gap-section" aria-labelledby={`gap-${label}`}>
      <h3 id={`gap-${label}`}>{label}</h3>
      {gaps.length === 0 ? (
        <p className="empty-line">无</p>
      ) : (
        <ul>
          {gaps.map((gap, index) => (
            <li key={`${gap.subject_id}-${index}`}>
              <p>{gap.message}</p>
              <details>
                <summary>查看证据</summary>
                <EvidenceLists
                  jobEvidence={gap.job_evidence}
                  candidateEvidence={gap.candidate_evidence}
                />
              </details>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function EvidenceSection({
  title,
  jobEvidence,
  candidateEvidence,
  traceEvidence,
}: {
  title: string;
  jobEvidence: string[];
  candidateEvidence: string[];
  traceEvidence: string[];
}) {
  return (
    <section className="evidence-section">
      <details>
        <summary>{title}</summary>
        <EvidenceLists
          jobEvidence={jobEvidence.length ? jobEvidence : traceEvidence}
          candidateEvidence={candidateEvidence}
        />
      </details>
    </section>
  );
}

function EvidenceLists({
  jobEvidence,
  candidateEvidence,
}: {
  jobEvidence: string[];
  candidateEvidence: string[];
}) {
  return (
    <div className="evidence-columns">
      <EvidenceList title="JD 证据" items={jobEvidence} />
      <EvidenceList title="候选人证据" items={candidateEvidence} />
    </div>
  );
}

function EvidenceList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h4>{title}</h4>
      {items.length ? (
        <ul>{items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
      ) : (
        <p className="empty-line">暂无证据</p>
      )}
    </div>
  );
}
