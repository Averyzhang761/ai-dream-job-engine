import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import type { McpAppBridge } from "../bridge";
import { EvidenceView } from "../components/EvidenceView";
import type { CandidateClaim, CandidateProfile, ClaimPreviewView } from "../contracts";

const resumeClaim: CandidateClaim = {
  id: "resume-1",
  claim_type: "skill",
  value: "Python",
  value_is_exhaustive: true,
  source: "resume_evidence",
  confirmed: true,
  source_quote: "Built Python services",
  provenance: "resume.docx",
  source_agent: null,
  observed_at: null,
  confidence: 0.9,
  import_origin: null,
};

const profile: CandidateProfile = {
  id: "default",
  version: "1",
  updated_at: "2026-10-11T20:00:00Z",
  claims: [
    resumeClaim,
    { ...resumeClaim, id: "user-1", value: "Bay Area", source: "user_confirmed", source_quote: null, provenance: "用户填写" },
    {
      ...resumeClaim,
      id: "agent-1",
      value: "Production AI",
      source: "user_confirmed",
      source_quote: null,
      provenance: "project note",
      source_agent: "Codex",
      import_origin: {
        source_agent: "Codex",
        candidate_id: "candidate-1",
        provenance: "project note",
        claim_type: "skill",
        candidate_value: "Production AI",
        confirmed_value: "Production AI",
      },
    },
  ],
  import_receipts: [],
  memory_connections: [{ source_agent: "Codex", scope: ["项目"], connected: true, generation: 1 }],
};

const preview: ClaimPreviewView = {
  preview_id: "p".repeat(32),
  kind: "agent_memory",
  candidates: [
    { ...resumeClaim, id: "candidate-a", value: "RAG", source: "agent_suggested", confirmed: false, provenance: "memory/project-a", source_agent: "Codex" },
    { ...resumeClaim, id: "candidate-b", value: "FDE interest", source: "agent_suggested", confirmed: false, provenance: "memory/preference", source_agent: "Codex" },
  ],
  source_agent: "Codex",
  scope: ["项目", "职业偏好"],
  added_claim_ids: ["candidate-a"],
  changed_claim_ids: ["candidate-b"],
  conflicting_claim_ids: ["candidate-b"],
};

function bridge(confirmFails = false) {
  const calls: Array<{ name: string; args: object }> = [];
  const messages: string[] = [];
  const value: McpAppBridge = {
    initialize: async () => undefined,
    subscribeToToolResults: () => () => undefined,
    sendMessage: async (message) => { messages.push(message); },
    callTool: async <T,>(name: string, args: object) => {
      calls.push({ name, args });
      if (name === "get_candidate_profile") return structuredClone(profile) as T;
      if (name.startsWith("confirm_") || name === "disconnect_agent_memory") {
        if (confirmFails) throw new Error("private preview error");
        return { ...profile, claims: [...profile.claims, { ...resumeClaim, id: "confirmed-new", value: "RAG", source: "user_confirmed" }] } as T;
      }
      throw new Error("unexpected tool");
    },
  };
  return { value, calls, messages };
}

describe("EvidenceView", () => {
  afterEach(() => cleanup());

  it("groups confirmed evidence by provenance without exposing unconfirmed claims", async () => {
    render(<EvidenceView bridge={bridge().value} initialClaims={[]} claimPreview={null} />);
    expect(await screen.findByRole("heading", { name: "简历证据" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "用户确认" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Agent 导入" })).toBeVisible();
    expect(screen.getByText("Built Python services")).toBeVisible();
    expect(screen.getByText("project note")).toBeVisible();
  });

  it("keeps Agent Memory off until the user chooses scope and sends a preview request", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(<EvidenceView bridge={mock.value} initialClaims={profile.claims} claimPreview={null} />);
    expect(mock.messages).toHaveLength(0);
    expect(screen.getByRole("button", { name: "让 Agent 生成候选" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "项目" }));
    await user.click(screen.getByRole("button", { name: "让 Agent 生成候选" }));
    expect(mock.messages).toHaveLength(1);
    expect(mock.messages[0]).toContain("项目");
    expect(mock.messages[0]).toContain("预览");
    expect(mock.calls.some((call) => call.name.startsWith("confirm_"))).toBe(false);
  });

  it("requires accept, edit, or reject for every claim before confirmation", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(<EvidenceView bridge={mock.value} initialClaims={profile.claims} claimPreview={preview} />);
    const review = await screen.findByRole("region", { name: "候选证据确认" });
    await user.click(within(review).getByRole("button", { name: "提交确认" }));
    expect(await screen.findByText("请逐条选择接受、修改或拒绝")).toBeVisible();
    expect(mock.calls.filter((call) => call.name.startsWith("confirm_"))).toHaveLength(0);

    const rows = within(review).getAllByRole("group");
    await user.click(within(rows[0]).getByRole("radio", { name: "接受" }));
    await user.click(within(rows[1]).getByRole("radio", { name: "修改" }));
    await user.clear(within(rows[1]).getByLabelText("修改后的值"));
    await user.type(within(rows[1]).getByLabelText("修改后的值"), "FDE and Applied AI");
    await user.click(within(review).getByRole("button", { name: "提交确认" }));

    await waitFor(() => expect(mock.calls).toContainEqual({
      name: "confirm_agent_memory_import",
      args: {
        preview_id: preview.preview_id,
        decisions: [
          { claim_id: "candidate-a", action: "accept" },
          { claim_id: "candidate-b", action: "edit", edited_value: "FDE and Applied AI" },
        ],
      },
    }));
  });

  it("clears editable preview values after an invalid or expired preview", async () => {
    const user = userEvent.setup();
    const mock = bridge(true);
    render(<EvidenceView bridge={mock.value} initialClaims={profile.claims} claimPreview={preview} />);
    const review = await screen.findByRole("region", { name: "候选证据确认" });
    const rows = within(review).getAllByRole("group");
    await user.click(within(rows[0]).getByRole("radio", { name: "拒绝" }));
    await user.click(within(rows[1]).getByRole("radio", { name: "修改" }));
    await user.type(within(rows[1]).getByLabelText("修改后的值"), "private edit");
    await user.click(within(review).getByRole("button", { name: "提交确认" }));

    expect(await screen.findByText("预览已失效，请重新生成")).toBeVisible();
    expect(screen.queryByDisplayValue("private edit")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重新生成预览" }));
    expect(mock.messages.at(-1)).toContain("重新生成");
  });

  it("shows server preview categories", async () => {
    render(<EvidenceView bridge={bridge().value} initialClaims={profile.claims} claimPreview={preview} />);
    const review = await screen.findByRole("region", { name: "候选证据确认" });
    expect(within(review).getByText("新增")).toBeVisible();
    expect(within(review).getByText("变更")).toBeVisible();
    expect(within(review).getByText("冲突")).toBeVisible();
  });

  it("disconnects an Agent source with explicit preserve or destructive delete choices", async () => {
    const user = userEvent.setup();
    const mock = bridge();
    render(<EvidenceView bridge={mock.value} initialClaims={profile.claims} claimPreview={null} />);
    await screen.findByText("Production AI");

    await user.click(screen.getByRole("button", { name: "断开 Codex" }));
    await user.click(screen.getByRole("radio", { name: "保留已导入证据" }));
    await user.click(screen.getByRole("button", { name: "确认断开" }));
    await waitFor(() => expect(mock.calls).toContainEqual({
      name: "disconnect_agent_memory",
      args: { source_agent: "Codex", delete_imported: false },
    }));

    await user.click(screen.getByRole("button", { name: "断开 Codex" }));
    await user.click(screen.getByRole("radio", { name: "删除此来源导入的证据" }));
    expect(screen.getByRole("button", { name: "确认删除并断开" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "我确认删除 Codex 导入的证据" }));
    await user.click(screen.getByRole("button", { name: "确认删除并断开" }));
    await waitFor(() => expect(mock.calls).toContainEqual({
      name: "disconnect_agent_memory",
      args: { source_agent: "Codex", delete_imported: true },
    }));
  });

  it("does not infer an Agent connection from retained imported claims", async () => {
    const disconnectedProfile = {
      ...profile,
      memory_connections: [{ source_agent: "Codex", scope: ["项目"], connected: false, generation: 2 }],
    };
    const disconnectedBridge = bridge();
    disconnectedBridge.value.callTool = async <T,>(name: string) => {
      if (name === "get_candidate_profile") return disconnectedProfile as T;
      throw new Error("unexpected tool");
    };

    render(<EvidenceView bridge={disconnectedBridge.value} initialClaims={disconnectedProfile.claims} claimPreview={null} />);
    expect(await screen.findByText("Production AI")).toBeVisible();
    expect(screen.queryByRole("button", { name: "断开 Codex" })).not.toBeInTheDocument();
  });

  it("does not let an old preview response hide a newer preview", async () => {
    const user = userEvent.setup();
    const pending = deferred<CandidateProfile>();
    const calls: Array<{ name: string; args: object }> = [];
    const raceBridge: McpAppBridge = {
      initialize: async () => undefined,
      subscribeToToolResults: () => () => undefined,
      sendMessage: async () => undefined,
      callTool: async <T,>(name: string, args: object) => {
        calls.push({ name, args });
        if (name === "get_candidate_profile") return profile as T;
        return pending.promise as Promise<T>;
      },
    };
    const previewB = {
      ...preview,
      preview_id: "b".repeat(32),
      candidates: [{ ...preview.candidates[0], id: "candidate-new", value: "New B claim" }],
      added_claim_ids: ["candidate-new"],
      changed_claim_ids: [],
    };
    const rendered = render(<EvidenceView bridge={raceBridge} initialClaims={profile.claims} claimPreview={preview} />);
    const firstReview = await screen.findByRole("region", { name: "候选证据确认" });
    for (const row of within(firstReview).getAllByRole("group")) {
      await user.click(within(row).getByRole("radio", { name: "接受" }));
    }
    await user.click(within(firstReview).getByRole("button", { name: "提交确认" }));

    rendered.rerender(<EvidenceView bridge={raceBridge} initialClaims={profile.claims} claimPreview={previewB} />);
    expect(await screen.findByText("New B claim")).toBeVisible();
    pending.resolve(profile);
    await waitFor(() => expect(screen.getByText("New B claim")).toBeVisible());
    expect(screen.getByRole("region", { name: "候选证据确认" })).toBeVisible();
    expect(calls.filter((call) => call.name === "confirm_agent_memory_import")).toHaveLength(1);
  });

  it("rejects invalid edited numbers and booleans while preserving strings", async () => {
    const user = userEvent.setup();
    const typedPreview: ClaimPreviewView = {
      ...preview,
      candidates: [
        { ...preview.candidates[0], id: "number", value: 3 },
        { ...preview.candidates[0], id: "boolean", value: true },
        { ...preview.candidates[0], id: "string", value: "  exact text  " },
      ],
      added_claim_ids: ["number", "boolean", "string"],
      changed_claim_ids: [],
    };
    const mock = bridge();
    render(<EvidenceView bridge={mock.value} initialClaims={profile.claims} claimPreview={typedPreview} />);
    const rows = await screen.findAllByRole("group");
    for (const row of rows) await user.click(within(row).getByRole("radio", { name: "修改" }));
    const edits = screen.getAllByLabelText("修改后的值");
    await user.clear(edits[0]);
    await user.type(edits[0], "Infinity");
    await user.clear(edits[1]);
    await user.type(edits[1], "yes");
    await user.click(screen.getByRole("button", { name: "提交确认" }));
    expect(await screen.findByText(/数字必须是有限值/)).toBeVisible();
    expect(mock.calls.filter((call) => call.name.startsWith("confirm_"))).toHaveLength(0);

    await user.clear(edits[0]);
    await user.type(edits[0], "4.5");
    await user.click(screen.getByRole("button", { name: "提交确认" }));
    expect(await screen.findByText(/布尔值只能填写 true 或 false/)).toBeVisible();
    expect(mock.calls.filter((call) => call.name.startsWith("confirm_"))).toHaveLength(0);

    await user.clear(edits[1]);
    await user.type(edits[1], "false");
    await user.click(screen.getByRole("button", { name: "提交确认" }));
    await waitFor(() => expect(mock.calls).toContainEqual({
      name: "confirm_agent_memory_import",
      args: {
        preview_id: typedPreview.preview_id,
        decisions: [
          { claim_id: "number", action: "edit", edited_value: 4.5 },
          { claim_id: "boolean", action: "edit", edited_value: false },
          { claim_id: "string", action: "edit", edited_value: "  exact text  " },
        ],
      },
    }));
  });
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => { resolve = next; });
  return { promise, resolve };
}
