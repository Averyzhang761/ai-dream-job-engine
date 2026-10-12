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
  conflicting_claim_ids: [],
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
      if (name.startsWith("confirm_")) {
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
    await user.click(screen.getByRole("checkbox", { name: "项目" }));
    await user.click(screen.getByRole("button", { name: "同步 Agent Memory" }));
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
});
