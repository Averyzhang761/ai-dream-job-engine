import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { McpAppBridge } from "../bridge";
import { RubricView } from "../components/RubricView";
import type { Rubric } from "../contracts";

const rubric: Rubric = {
  id: "default",
  version: "1",
  revision: "r1",
  facts: [
    {
      id: "location",
      name: "工作地点",
      definition: "岗位所在地点",
      data_type: "text",
      extraction_instructions: "提取明确地点",
      requires_evidence: true,
      missing_policy: "unknown",
      default_value: null,
      enum_values: [],
      candidate_comparison: "exact",
      qualification_weight: 1,
    },
  ],
  dimensions: [
    {
      id: "product_facing",
      name: "产品协作",
      definition: "是否贴近用户",
      examples: ["客户共创"],
      fact_ids: ["location"],
      signal_ids: ["product_signal"],
      weight: 1,
      scoring_direction: "higher_is_better",
      positive_keywords: ["customer"],
      negative_keywords: ["back office"],
      hard_constraint: false,
      threshold: 0.6,
      explanation_template: "根据产品协作证据判断",
    },
  ],
  questions: [{ id: "product_signal", version: "1", question: "是否面向产品？" }],
  recommendation_thresholds: {
    priority_apply: 0.8,
    apply_after_review: 0.6,
    hold: 0.3,
    skip: 0,
  },
};

function bridge() {
  const calls: Array<{ name: string; args: object }> = [];
  const value: McpAppBridge = {
    initialize: async () => undefined,
    subscribeToToolResults: () => () => undefined,
    sendMessage: async () => undefined,
    callTool: async <T,>(name: string, args: object) => {
      calls.push({ name, args });
      if (name === "get_rubric") return structuredClone(rubric) as T;
      if (name === "update_rubric") return { ...rubric, revision: "r2" } as T;
      if (name === "recompute_decision") return { id: "analysis-1" } as T;
      throw new Error("unexpected tool");
    },
  };
  return { value, calls };
}

describe("RubricView", () => {
  afterEach(() => cleanup());

  it("renders server definitions and sends only changed declarative fields", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(
      <RubricView
        bridge={mock.value}
        selectedAnalysisId="analysis-1"
        onRecomputed={vi.fn()}
      />,
    );

    expect(await screen.findByDisplayValue("工作地点")).toBeVisible();
    expect(screen.getByDisplayValue("产品协作")).toBeVisible();
    const name = screen.getByLabelText("事实名称 location");
    await user.clear(name);
    await user.type(name, "首选地点");
    await user.click(screen.getByRole("button", { name: "保存标准" }));

    await waitFor(() =>
      expect(mock.calls).toContainEqual({
        name: "update_rubric",
        args: { patch: { fact_updates: { location: { name: "首选地点" } } } },
      }),
    );
    expect(screen.getByRole("button", { name: "重新计算当前岗位" })).toBeVisible();
  });

  it("rejects incomplete additions and invalid numbers before a tool call", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(<RubricView bridge={mock.value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    await screen.findByDisplayValue("工作地点");

    await user.click(screen.getByRole("button", { name: "新增事实字段" }));
    await user.click(screen.getByRole("button", { name: "保存标准" }));
    expect(await screen.findByText(/请补全新增事实字段/)).toBeVisible();
    expect(mock.calls.filter((call) => call.name === "update_rubric")).toHaveLength(0);

    cleanup();
    const numericMock = bridge();
    render(<RubricView bridge={numericMock.value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    const weight = await screen.findByLabelText("维度权重 product_facing");
    await user.clear(weight);
    await user.type(weight, "not-a-number");
    await user.click(screen.getByRole("button", { name: "保存标准" }));
    expect(await screen.findByText(/请输入有效数字/)).toBeVisible();
    expect(numericMock.calls.filter((call) => call.name === "update_rubric")).toHaveLength(0);
  });

  it("recomputes only after an explicit action", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    const onRecomputed = vi.fn();
    render(
      <RubricView
        bridge={mock.value}
        selectedAnalysisId="analysis-1"
        onRecomputed={onRecomputed}
      />,
    );
    await user.clear(await screen.findByLabelText("事实名称 location"));
    await user.type(screen.getByLabelText("事实名称 location"), "地点偏好");
    await user.click(screen.getByRole("button", { name: "保存标准" }));
    expect(mock.calls.some((call) => call.name === "recompute_decision")).toBe(false);

    await user.click(await screen.findByRole("button", { name: "重新计算当前岗位" }));
    await waitFor(() =>
      expect(mock.calls).toContainEqual({
        name: "recompute_decision",
        args: { analysis_id: "analysis-1" },
      }),
    );
    expect(onRecomputed).toHaveBeenCalledWith("analysis-1");
  });
});
