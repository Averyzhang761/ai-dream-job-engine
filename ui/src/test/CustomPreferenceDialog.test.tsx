import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CustomPreferenceDialog } from "../components/CustomPreferenceDialog";
import { PendingPreferenceReview } from "../components/PendingPreferenceReview";
import type { PendingPreference } from "../contracts";

const pending = (conflict_kind: PendingPreference["conflict_kind"]): PendingPreference => ({
  id: "draft-1", statement: "我想直接接触客户", topic_id: "customer_contact",
  category: "role_content", name: "客户接触", desired_value: "直接与客户共创",
  requirement_level: "strong", evidence_guidance: "查看客户访谈与共创证据",
  fact_ids: ["product_customer_scope"], signal_ids: [], related_topic_ids: [],
  conflict_kind, conflicting_dimension_id: conflict_kind === "none" ? null : "product_customer_proximity",
});

describe("CustomPreferenceDialog", () => {
  afterEach(() => cleanup());

  it("sends natural language only after one explicit structure action", async () => {
    const user = userEvent.setup();
    const onAgentHandoff = vi.fn().mockResolvedValue(undefined);
    render(<CustomPreferenceDialog onClose={vi.fn()} onAgentHandoff={onAgentHandoff} onManualStage={vi.fn()} />);
    await user.type(screen.getByLabelText("用一句话描述偏好"), "我希望经常和客户交流");
    expect(onAgentHandoff).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "整理偏好" }));
    expect(onAgentHandoff).toHaveBeenCalledTimes(1);
    expect(onAgentHandoff).toHaveBeenCalledWith("我希望经常和客户交流");
  });

  it("preserves the original text and exposes manual fields after Agent failure", async () => {
    const user = userEvent.setup();
    render(<CustomPreferenceDialog onClose={vi.fn()} onAgentHandoff={vi.fn().mockRejectedValue(new Error("failed"))} onManualStage={vi.fn()} />);
    const input = screen.getByLabelText("用一句话描述偏好");
    await user.type(input, "我要混合办公");
    await user.click(screen.getByRole("button", { name: "整理偏好" }));
    expect(input).toHaveValue("我要混合办公");
    expect(await screen.findByLabelText("偏好名称")).toBeVisible();
    expect(screen.getByText("未能连接 Agent，可以手动整理。")).toBeVisible();
  });
});

describe("PendingPreferenceReview", () => {
  afterEach(() => cleanup());

  it("offers only keep-existing for a duplicate", () => {
    render(<PendingPreferenceReview preference={pending("duplicate")} busy={false} onResolve={vi.fn()} />);
    expect(screen.getByRole("button", { name: "保留现有标准" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "直接新增" })).not.toBeInTheDocument();
  });

  it("requires an explicit side for direct conflicts and tension", () => {
    const { rerender } = render(<PendingPreferenceReview preference={pending("direct_conflict")} busy={false} onResolve={vi.fn()} />);
    expect(screen.getByRole("button", { name: "用新偏好替换" })).toBeVisible();
    expect(screen.getByRole("button", { name: "保留现有标准" })).toBeVisible();
    rerender(<PendingPreferenceReview preference={pending("semantic_tension")} busy={false} onResolve={vi.fn()} />);
    expect(screen.getByText("这项偏好与现有标准可能存在张力。")).toBeVisible();
    expect(screen.getByRole("button", { name: "保留两项" })).toBeVisible();
  });

  it("does not confirm manual review until required fields are complete", async () => {
    const user = userEvent.setup();
    const incomplete = { ...pending("manual_review"), name: null, desired_value: null };
    render(<PendingPreferenceReview preference={incomplete} busy={false} onResolve={vi.fn()} />);
    expect(screen.getByRole("button", { name: "确认添加" })).toBeDisabled();
    await user.type(screen.getByLabelText("偏好名称"), "客户接触");
    await user.type(screen.getByLabelText("我的选择"), "直接共创");
    expect(screen.getByRole("button", { name: "确认添加" })).toBeEnabled();
  });
});
