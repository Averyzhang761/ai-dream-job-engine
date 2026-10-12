import type { WorkspaceJobSummary } from "../contracts";
import { StatusBadge } from "./StatusBadge";

export function formatScore(value: number | null): string {
  return value === null ? "未评分" : `${Math.round(value * 100)}%`;
}

export function JobList({
  jobs,
  selectedId,
  onSelect,
}: {
  jobs: WorkspaceJobSummary[];
  selectedId: string | null;
  onSelect: (analysisId: string) => void;
}) {
  if (jobs.length === 0) {
    return <div className="state-message">尚无已分析岗位</div>;
  }

  return (
    <ul className="job-list" aria-label="岗位队列">
      {jobs.map((job) => {
        const title = job.title ?? "未提供岗位名称";
        const company = job.company ?? "未提供公司名称";
        return (
          <li key={job.analysis_id} className="job-list-item">
            <button
              type="button"
              className="job-list-button"
              aria-current={selectedId === job.analysis_id ? "true" : undefined}
              aria-label={`${title}，${company}`}
              onClick={() => onSelect(job.analysis_id)}
            >
              <span className="job-list-heading">
                <strong>{title}</strong>
                <StatusBadge recommendation={job.recommendation} />
              </span>
              <span className="job-company">{company}</span>
              <span className="job-scores">
                <span>偏好 {formatScore(job.preference_score)}</span>
                <span>胜任证据 {formatScore(job.qualification_score)}</span>
              </span>
              {job.top_unresolved_gap && (
                <span className="job-top-gap">
                  {job.top_unresolved_gap.message}
                </span>
              )}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
