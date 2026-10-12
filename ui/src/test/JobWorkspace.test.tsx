import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { App } from "../App";
import type { McpAppBridge } from "../bridge";
import type {
  Gap,
  ToolResultEnvelope,
  WorkspaceAppPayload,
  WorkspaceJobDetail,
  WorkspaceJobSummary,
} from "../contracts";

const evidenceGap: Gap = {
  kind: "evidence_gap",
  subject_id: "production_ai",
  message: "缺少生产环境 AI 项目证据",
  input_ids: ["production_ai"],
  job_evidence: ["Ship production AI systems"],
  candidate_evidence: [],
  missing_evidence: true,
};

function summary(overrides: Partial<WorkspaceJobSummary> = {}): WorkspaceJobSummary {
  return {
    analysis_id: "analysis-1",
    title: "Applied AI Engineer",
    company: "Example AI",
    recommendation: "apply_after_review",
    preference_score: 0.82,
    qualification_score: 0.64,
    top_unresolved_gap: evidenceGap,
    review_status: "unreviewed",
    created_at: "2026-10-11T18:00:00Z",
    can_mark_ready_to_apply: false,
    ...overrides,
  };
}

function detail(jobSummary = summary()): WorkspaceJobDetail {
  return {
    summary: jobSummary,
    gaps: {
      preference_mismatch: [{ ...evidenceGap, kind: "preference_mismatch", subject_id: "work_mode", message: "工作方式与偏好不一致", missing_evidence: false, job_evidence: ["Five days onsite"], candidate_evidence: ["偏好混合办公"] }],
      hard_blocker: [],
      capability_gap: [],
      evidence_gap: [evidenceGap],
    },
    preference_contributions: [{ subject_id: "product_facing", score: 0.9, weight: 1, input_ids: ["product_facing"], job_evidence: ["Partner with customers"], candidate_evidence: [], audit_support: [] }],
    qualification_contributions: [],
    trace_evidence: ["Partner with customers", "Ship production AI systems"],
  };
}

function payload(jobs: WorkspaceJobSummary[]): WorkspaceAppPayload {
  return {
    view: "jobs",
    snapshot: {
      profile_ready: true,
      rubric_ready: true,
      setup_actions: [],
      active_rubric: { id: "default", version: "1", revision: "r1" },
      confirmed_claims: [],
      jobs,
    },
    claim_preview: null,
  };
}

function bridgeWithPayload(
  initialPayload: WorkspaceAppPayload,
  toolImpl: (name: string, args: object) => Promise<unknown> = async (name, args) => {
    if (name === "get_workspace_job") {
      const selected = initialPayload.snapshot.jobs.find(
        (job) => job.analysis_id === (args as { analysis_id: string }).analysis_id,
      );
      if (!selected) throw new Error("missing");
      return detail(selected);
    }
    return {};
  },
): McpAppBridge {
  let listener: ((result: ToolResultEnvelope) => void) | undefined;
  return {
    initialize: async () => {
      queueMicrotask(() => listener?.({ structuredContent: initialPayload, isError: false }));
    },
    callTool: async <T,>(name: string, args: object) =>
      (await toolImpl(name, args)) as T,
    sendMessage: async () => undefined,
    subscribeToToolResults: (nextListener) => {
      listener = nextListener;
      return () => { listener = undefined; };
    },
  };
}

describe("job workspace", () => {
  afterEach(() => cleanup());

  it("shows loading, then an empty queue", async () => {
    render(<App bridge={bridgeWithPayload(payload([]))} />);
    expect(
      within(screen.getByRole("tabpanel", { name: "岗位" })).getByText(
        "正在连接本地工作区…",
      ),
    ).toBeVisible();
    expect(await screen.findByText("尚无已分析岗位")).toBeVisible();
  });

  it("renders server-ordered summaries, fallback text, labels, scores, and top gap", async () => {
    const jobs = [
      summary({ analysis_id: "newest", title: null, company: null, recommendation: "priority_apply", preference_score: 0.91, qualification_score: 0.8, created_at: "2026-10-11T20:00:00Z" }),
      summary({ analysis_id: "review", recommendation: "apply_after_review", created_at: "2026-10-11T19:00:00Z" }),
      summary({ analysis_id: "hold", recommendation: "hold", created_at: "2026-10-11T18:00:00Z" }),
      summary({ analysis_id: "skip", recommendation: "skip", created_at: "2026-10-11T17:00:00Z" }),
    ];
    render(<App bridge={bridgeWithPayload(payload(jobs))} />);

    const list = await screen.findByRole("list", { name: "岗位队列" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(4);
    expect(items[0]).toHaveTextContent("未提供岗位名称");
    expect(items[0]).toHaveTextContent("未提供公司名称");
    expect(items[0]).toHaveTextContent("优先申请");
    expect(items[0]).toHaveTextContent("偏好 91%");
    expect(items[0]).toHaveTextContent("胜任证据 80%");
    expect(items[0]).toHaveTextContent("缺少生产环境 AI 项目证据");
    expect(items[1]).toHaveTextContent("审核后申请");
    expect(items[2]).toHaveTextContent("暂缓");
    expect(items[3]).toHaveTextContent("跳过");
  });

  it("loads selection in place and renders classified detail evidence", async () => {
    const user = userEvent.setup();
    const first = summary({ analysis_id: "first" });
    const second = summary({ analysis_id: "second", title: "Forward Deployed Engineer", top_unresolved_gap: null });
    render(<App bridge={bridgeWithPayload(payload([first, second]))} />);

    const shell = screen.getByTestId("app-shell");
    await user.click(await screen.findByRole("button", { name: /Forward Deployed Engineer/ }));
    expect(screen.getByTestId("app-shell")).toBe(shell);
    expect(await screen.findByRole("heading", { name: "Forward Deployed Engineer" })).toBeVisible();
    expect(screen.getByText("偏好差距")).toBeVisible();
    expect(screen.getByText("证据差距")).toBeVisible();
    const jobDetail = screen.getByRole("article", { name: "岗位判断详情" });
    expect(within(jobDetail).getByText("缺少生产环境 AI 项目证据")).toBeVisible();
    await user.click(within(jobDetail).getAllByText("查看证据")[1]);
    expect(within(jobDetail).getByText("Ship production AI systems")).toBeVisible();
  });

  it("persists keep and ignore only from returned server detail", async () => {
    const user = userEvent.setup();
    let current = summary({ analysis_id: "actionable" });
    const bridge = bridgeWithPayload(payload([current]), async (name, args) => {
      if (name === "get_workspace_job") return detail(current);
      if (name === "set_job_review_status") {
        current = { ...current, review_status: (args as { status: "kept" | "ignored" }).status };
        return detail(current);
      }
      throw new Error("unexpected");
    });
    render(<App bridge={bridge} />);

    await user.click(await screen.findByRole("button", { name: "保留" }));
    expect(await screen.findByText("已保留此岗位")).toBeVisible();
    expect(screen.getByText("已保留")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "忽略" }));
    expect(await screen.findByText("已忽略此岗位")).toBeVisible();
    expect(screen.getByText("已忽略")).toBeVisible();
  });

  it("enables ready only from the server flag and explains a disabled action", async () => {
    const user = userEvent.setup();
    render(<App bridge={bridgeWithPayload(payload([summary({ analysis_id: "ineligible" })]))} />);
    expect(await screen.findByRole("button", { name: "加入待申请" })).toBeDisabled();
    expect(screen.getByText("服务端判定此岗位暂不可加入待申请")).toBeVisible();

    cleanup();
    let eligible = summary({ analysis_id: "eligible", recommendation: "priority_apply", can_mark_ready_to_apply: true, top_unresolved_gap: null });
    const bridge = bridgeWithPayload(payload([eligible]), async (name) => {
      if (name === "get_workspace_job") return detail(eligible);
      eligible = { ...eligible, review_status: "ready_to_apply" };
      return detail(eligible);
    });
    render(<App bridge={bridge} />);
    await user.click(await screen.findByRole("button", { name: "加入待申请" }));
    expect(await screen.findByText("已加入待申请队列")).toBeVisible();
    expect(screen.getByText("待申请")).toBeVisible();
  });

  it("refreshes selected detail after a stale-state server rejection", async () => {
    const user = userEvent.setup();
    const stale = summary({ analysis_id: "stale", recommendation: "priority_apply", can_mark_ready_to_apply: true, top_unresolved_gap: null });
    const refreshed = { ...stale, recommendation: "apply_after_review" as const, can_mark_ready_to_apply: false, top_unresolved_gap: evidenceGap };
    let detailReads = 0;
    const bridge = bridgeWithPayload(payload([stale]), async (name) => {
      if (name === "get_workspace_job") {
        detailReads += 1;
        return detail(detailReads === 1 ? stale : refreshed);
      }
      if (name === "set_job_review_status") throw new Error("private server detail");
      throw new Error("unexpected");
    });
    render(<App bridge={bridge} />);

    await user.click(await screen.findByRole("button", { name: "加入待申请" }));
    expect(await screen.findByText("状态未更新；已刷新服务端结果")).toBeVisible();
    expect(
      within(screen.getByRole("article", { name: "岗位判断详情" })).getByText(
        "审核后申请",
      ),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "加入待申请" })).toBeDisabled();
    expect(screen.queryByText("private server detail")).not.toBeInTheDocument();
  });
});
