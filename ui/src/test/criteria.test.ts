import { describe, expect, it } from "vitest";

import { CRITERION_CATEGORIES, dimensionPatchForRequirement, requirementLabel, summarizeCategory } from "../criteria";
import type { ValueDimension } from "../contracts";

const dimension = (desiredValue: string): ValueDimension => ({
  id: desiredValue, topic_id: desiredValue, category: "role_content", name: desiredValue,
  desired_value: desiredValue, requirement_level: "normal", origin: "template",
  definition: "definition", examples: [], fact_ids: ["role"], signal_ids: [],
  weight: 1, scoring_direction: "higher_is_better", positive_keywords: [],
  negative_keywords: [], hard_constraint: false, threshold: null, explanation_template: "evidence",
});

describe("criteria view model", () => {
  it("keeps the four approved categories in a stable order", () => {
    expect(CRITERION_CATEGORIES.map((item) => item.id)).toEqual([
      "role_content", "work_conditions", "growth_team", "compensation_constraints",
    ]);
  });

  it("summarizes at most three values and handles empty categories", () => {
    expect(summarizeCategory([])).toBe("尚未设置");
    expect(summarizeCategory(["A", "B", "C", "D"].map(dimension))).toBe("A、B、C +1");
  });

  it("projects each requirement level to one consistent backend patch", () => {
    expect(requirementLabel("required")).toBe("必须满足");
    expect(dimensionPatchForRequirement("ignored")).toEqual({ requirement_level: "ignored", weight: 0, hard_constraint: false });
    expect(dimensionPatchForRequirement("strong")).toEqual({ requirement_level: "strong", weight: 1.5, hard_constraint: false });
    expect(dimensionPatchForRequirement("required")).toEqual({ requirement_level: "required", weight: 1, hard_constraint: true });
  });
});
