import type { CriterionCategory, RequirementLevel, ValueDimension } from "./contracts";

export const CRITERION_CATEGORIES: ReadonlyArray<{
  id: CriterionCategory;
  label: string;
  description: string;
}> = [
  { id: "role_content", label: "岗位内容", description: "做什么，以及与 AI、产品和客户的距离" },
  { id: "work_conditions", label: "工作方式", description: "地点、远程、出差与日常协作条件" },
  { id: "growth_team", label: "成长与团队", description: "责任边界、团队环境与成长空间" },
  { id: "compensation_constraints", label: "回报与约束", description: "薪资、工作许可与不可妥协条件" },
];

const labels: Record<RequirementLevel, string> = {
  ignored: "不考虑",
  normal: "普通偏好",
  strong: "强偏好",
  required: "必须满足",
};

export function requirementLabel(level: RequirementLevel) {
  return labels[level];
}

export function summarizeCategory(dimensions: ValueDimension[]) {
  const values = dimensions
    .filter((item) => item.requirement_level !== "ignored")
    .map((item) => item.desired_value)
    .filter(Boolean);
  if (!values.length) return "尚未设置";
  const visible = values.slice(0, 3).join("、");
  return values.length > 3 ? `${visible} +${values.length - 3}` : visible;
}

export function dimensionPatchForRequirement(level: RequirementLevel) {
  const projection = {
    ignored: { weight: 0, hard_constraint: false },
    normal: { weight: 1, hard_constraint: false },
    strong: { weight: 1.5, hard_constraint: false },
    required: { weight: 1, hard_constraint: true },
  }[level];
  return { requirement_level: level, ...projection };
}
