import type {
  JobReviewStatus,
  RecommendationLevel,
} from "../contracts";

const recommendationLabels: Record<RecommendationLevel, string> = {
  priority_apply: "优先申请",
  apply_after_review: "审核后申请",
  hold: "暂缓",
  skip: "跳过",
};

const reviewLabels: Record<JobReviewStatus, string> = {
  unreviewed: "未审核",
  kept: "已保留",
  ignored: "已忽略",
  ready_to_apply: "待申请",
};

export function recommendationLabel(value: RecommendationLevel): string {
  return recommendationLabels[value];
}

export function StatusBadge({
  recommendation,
  reviewStatus,
}: {
  recommendation?: RecommendationLevel;
  reviewStatus?: JobReviewStatus;
}) {
  const value = recommendation ?? reviewStatus;
  if (!value) return null;
  const label = recommendation
    ? recommendationLabels[recommendation]
    : reviewLabels[reviewStatus!];
  return (
    <span className={`status-badge status-badge--${value}`}>
      {label}
    </span>
  );
}
