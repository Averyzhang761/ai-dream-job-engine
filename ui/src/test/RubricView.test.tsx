import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { McpAppBridge } from "../bridge";
import { RubricView } from "../components/RubricView";
import type { CriterionCategory, Rubric, ToolResultEnvelope, ValueDimension } from "../contracts";

const categories: CriterionCategory[] = ["role_content", "work_conditions", "growth_team", "compensation_constraints"];
const dimension = (index: number): ValueDimension => ({
  id: `standard_${index}`, topic_id: `topic_${index}`, category: categories[index % categories.length],
  name: `标准 ${index}`, desired_value: `选择 ${index}`, requirement_level: "normal",
  origin: index === 4 ? "custom" : "template", definition: `判断说明 ${index}`, examples: [],
  fact_ids: ["role"], signal_ids: [], weight: 1, scoring_direction: "higher_is_better",
  positive_keywords: [], negative_keywords: [], hard_constraint: false, threshold: null,
  explanation_template: `查看岗位中的证据 ${index}`,
});

const rubric: Rubric = {
  id: "default", version: "1", revision: "r1", storage_revision: 0, facts: [],
  dimensions: Array.from({ length: 24 }, (_, index) => dimension(index + 1)), questions: [],
  pending_preferences: [],
  recommendation_thresholds: { priority_apply: 0.8, apply_after_review: 0.6, hold: 0.3, skip: 0 },
};

function bridge(initial = rubric) {
  const calls: Array<{ name: string; args: object }> = [];
  const value: McpAppBridge = {
    initialize: async () => undefined,
    subscribeToToolResults: () => () => undefined,
    sendMessage: async () => undefined,
    callTool: async <T,>(name: string, args: object) => {
      calls.push({ name, args });
      if (name === "get_rubric") return structuredClone(initial) as T;
      if (name === "update_rubric") return { ...rubric, revision: "r2" } as T;
      if (name === "recompute_decision") return { id: "analysis-1" } as T;
      throw new Error("unexpected tool");
    },
  };
  return { value, calls };
}

describe("RubricView", () => {
  afterEach(() => cleanup());

  it("shows four themes instead of a 24-row standards table", async () => {
    render(<RubricView bridge={bridge().value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "我的求职画像" })).toBeVisible();
    expect(screen.getAllByRole("button", { name: /\u7f16\u8f91/ })).toHaveLength(4);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByText("判断字段")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看全部 24 条标准" })).toBeVisible();
  });

  it("edits one category with human-facing columns and no internal fields", async () => {
    const user = userEvent.setup();
    render(<RubricView bridge={bridge().value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "编辑 岗位内容" }));
    const editor = screen.getByRole("region", { name: "编辑岗位内容" });
    expect(within(editor).getByText("标准")).toBeVisible();
    expect(within(editor).getByText("我的选择")).toBeVisible();
    expect(within(editor).getByText("要求程度")).toBeVisible();
    expect(within(editor).getAllByRole("combobox")).toHaveLength(6);
    expect(within(editor).queryByText("标准 2")).not.toBeInTheDocument();
    expect(screen.queryByText("权重")).not.toBeInTheDocument();
    expect(screen.queryByText("阈值")).not.toBeInTheDocument();
    await user.click(within(editor).getAllByText("系统如何判断")[0]);
    expect(within(editor).getByText("查看岗位中的证据 4")).toBeVisible();
    expect(within(editor).getByText("自定义")).toBeVisible();
  });

  it("changes a requirement within two clicks and saves its consistent projection", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(<RubricView bridge={mock.value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "编辑 岗位内容" }));
    await user.selectOptions(screen.getByLabelText("要求程度 标准 4"), "required");
    await user.click(screen.getByRole("button", { name: "保存更改" }));
    await waitFor(() => expect(mock.calls).toContainEqual({
      name: "update_rubric",
      args: { patch: { expected_storage_revision: 0, dimension_updates: { standard_4: { requirement_level: "required", weight: 1, hard_constraint: true } } } },
    }));
  });

  it("offers all-criteria search and filters without horizontal overflow", async () => {
    const user = userEvent.setup();
    const withPending = structuredClone(rubric);
    withPending.pending_preferences = [{
      id: "draft-filter", statement: "希望和客户共创", topic_id: "customer_contact",
      category: "role_content", name: "客户共创", desired_value: "直接与客户共创",
      requirement_level: "strong", evidence_guidance: "查看客户共创证据", fact_ids: ["role"],
      positive_keywords: ["客户共创"], negative_keywords: ["纯内部平台"],
      signal_ids: [], related_topic_ids: [], conflict_kind: "none", conflicting_dimension_id: null,
    }];
    render(<RubricView bridge={bridge(withPending).value} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "查看全部 24 条标准" }));
    const workspace = screen.getByLabelText("全部筛选标准");
    expect(within(workspace).getByRole("searchbox", { name: "搜索标准" })).toBeVisible();
    expect(within(workspace).getByLabelText("按主题筛选")).toBeVisible();
    expect(within(workspace).getByLabelText("按要求程度筛选")).toBeVisible();
    expect(within(workspace).getByLabelText("按处理状态筛选")).toBeVisible();
    await user.selectOptions(within(workspace).getByLabelText("按处理状态筛选"), "pending");
    expect(within(workspace).getByText("客户共创")).toBeVisible();
    expect(within(workspace).getByRole("button", { name: "处理" })).toBeVisible();
    expect(workspace).toHaveClass("criteria-workspace");
    expect(workspace).not.toHaveStyle({ overflowX: "auto" });
  });

  it("hands off explicit text and opens the private staged result for confirmation", async () => {
    const user = userEvent.setup();
    const sent: string[] = [];
    const listeners: Array<(result: ToolResultEnvelope) => void> = [];
    const staged = structuredClone(rubric);
    staged.pending_preferences = [{
      id: "draft-1", statement: "希望和客户共创", topic_id: "customer_contact",
      category: "role_content", name: "客户共创", desired_value: "直接与客户共创",
      requirement_level: "strong", evidence_guidance: "查看客户共创证据", fact_ids: ["role"],
      positive_keywords: ["客户共创"], negative_keywords: ["纯内部平台"],
      signal_ids: [], related_topic_ids: [], conflict_kind: "none", conflicting_dimension_id: null,
    }];
    const testBridge: McpAppBridge = {
      initialize: async () => undefined,
      callTool: async <T,>(name: string) => {
        if (name === "get_rubric") return structuredClone(rubric) as T;
        throw new Error("unexpected tool");
      },
      sendMessage: async (message) => { sent.push(message); },
      subscribeToToolResults: (next) => { listeners.push(next); return () => undefined; },
    };
    render(<RubricView bridge={testBridge} selectedAnalysisId={null} onRecomputed={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "添加偏好" }));
    await user.type(screen.getByLabelText("用一句话描述偏好"), "希望和客户共创");
    await user.click(screen.getByRole("button", { name: "整理偏好" }));
    expect(sent).toHaveLength(1);
    expect(sent[0]).toContain("希望和客户共创");
    expect(sent[0]).toContain("不要读取 Agent memory");
    listeners[0]?.({ _meta: { aiDreamJobEngine: { payload: staged } } });
    expect(await screen.findByRole("region", { name: "确认自定义偏好" })).toBeVisible();
    expect(screen.getByRole("button", { name: "确认添加" })).toBeVisible();
  });
});
