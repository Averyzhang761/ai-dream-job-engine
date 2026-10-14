import { useEffect, useState } from "react";

import { CRITERION_CATEGORIES, requirementLabel } from "../criteria";
import type { CriterionCategory, CustomPreferenceProposal, PendingPreference, PreferenceResolution, RequirementLevel } from "../contracts";

interface PendingPreferenceReviewProps {
  preference: PendingPreference;
  busy: boolean;
  onResolve: (resolution: PreferenceResolution) => void;
}

const levels: RequirementLevel[] = ["ignored", "normal", "strong", "required"];

export function PendingPreferenceReview({ preference, busy, onResolve }: PendingPreferenceReviewProps) {
  const [proposal, setProposal] = useState<CustomPreferenceProposal>(() => editable(preference));
  useEffect(() => setProposal(editable(preference)), [preference]);
  const set = (changes: Partial<CustomPreferenceProposal>) => setProposal((current) => ({ ...current, ...changes }));
  const complete = Boolean(proposal.topic_id?.trim() && proposal.category && proposal.name?.trim() && proposal.desired_value?.trim() && proposal.requirement_level && proposal.evidence_guidance?.trim());
  const resolve = (action: PreferenceResolution["action"]) => onResolve({ action, edited_proposal: proposal });

  return <section className="pending-review" aria-label="确认自定义偏好">
    <header><div><span>待确认偏好</span><h2>{proposal.name || "尚未命名"}</h2></div><strong>{conflictLabel(preference.conflict_kind)}</strong></header>
    {preference.conflict_kind === "semantic_tension" && <p className="conflict-warning">这项偏好与现有标准可能存在张力。</p>}
    <div className="preference-form">
      <label>偏好名称<input aria-label="偏好名称" value={proposal.name ?? ""} onChange={(event) => set({ name: event.target.value })} /></label>
      <label>主题标识<input aria-label="主题标识" value={proposal.topic_id ?? ""} onChange={(event) => set({ topic_id: event.target.value })} /></label>
      <label>所属主题<select aria-label="所属主题" value={proposal.category ?? ""} onChange={(event) => set({ category: event.target.value as CriterionCategory })}><option value="" disabled>请选择</option>{CRITERION_CATEGORIES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
      <label>我的选择<input aria-label="我的选择" value={proposal.desired_value ?? ""} onChange={(event) => set({ desired_value: event.target.value })} /></label>
      <label>要求程度<select aria-label="要求程度" value={proposal.requirement_level ?? ""} onChange={(event) => set({ requirement_level: event.target.value as RequirementLevel })}><option value="" disabled>请选择</option>{levels.map((item) => <option key={item} value={item}>{requirementLabel(item)}</option>)}</select></label>
      <label>系统如何找证据<textarea aria-label="系统如何找证据" value={proposal.evidence_guidance ?? ""} onChange={(event) => set({ evidence_guidance: event.target.value })} /></label>
    </div>
    <div className="pending-resolution-actions">
      {(preference.conflict_kind === "none" || preference.conflict_kind === "manual_review") && <button className="primary-button" type="button" disabled={!complete || busy} onClick={() => resolve("accept_new")}>确认添加</button>}
      {preference.conflict_kind === "duplicate" && <><button className="secondary-button" type="button" disabled={busy} onClick={() => resolve("keep_existing")}>保留现有标准</button><button className="text-button" type="button" disabled={!complete || proposal.topic_id === preference.topic_id || busy} onClick={() => resolve("accept_new")}>按修改内容新增</button></>}
      {preference.conflict_kind === "direct_conflict" && <><button className="primary-button" type="button" disabled={!complete || busy} onClick={() => resolve("replace_existing")}>用新偏好替换</button><button className="secondary-button" type="button" disabled={busy} onClick={() => resolve("keep_existing")}>保留现有标准</button></>}
      {preference.conflict_kind === "semantic_tension" && <><button className="primary-button" type="button" disabled={!complete || busy} onClick={() => resolve("accept_tension")}>保留两项</button><button className="secondary-button" type="button" disabled={busy} onClick={() => resolve("keep_existing")}>仅保留现有标准</button></>}
    </div>
  </section>;
}

function editable(preference: PendingPreference): CustomPreferenceProposal {
  const { id: _id, conflict_kind: _conflict, conflicting_dimension_id: _conflicting, ...proposal } = preference;
  return proposal;
}

function conflictLabel(kind: PendingPreference["conflict_kind"]) {
  return { none: "可添加", duplicate: "内容重复", direct_conflict: "直接冲突", semantic_tension: "可能存在张力", manual_review: "需补全" }[kind];
}
