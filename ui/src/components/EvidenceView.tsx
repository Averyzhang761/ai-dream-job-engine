import { Brain, Link2Off, RefreshCw, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { McpAppBridge } from "../bridge";
import type {
  CandidateClaim,
  CandidateProfile,
  ClaimDecision,
  ClaimPreviewView,
} from "../contracts";

interface EvidenceViewProps {
  bridge: McpAppBridge;
  initialClaims: CandidateClaim[];
  claimPreview: ClaimPreviewView | null;
}

type ClaimAction = ClaimDecision["action"];

const scopeOptions = ["项目", "技能", "职业偏好", "工作约束"];

function displayValue(value: unknown) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function evidenceGroup(claim: CandidateClaim) {
  if (claim.import_origin) return "agent";
  if (claim.source === "resume_evidence") return "resume";
  return "user";
}

function parseEditedValue(
  original: unknown,
  edited: string,
): { value?: unknown; error?: string } {
  if (typeof original === "string") return { value: edited };
  if (typeof original === "number") {
    const trimmed = edited.trim();
    const value = Number(trimmed);
    return trimmed !== "" && Number.isFinite(value)
      ? { value }
      : { error: "修改后的数字必须是有限值" };
  }
  if (typeof original === "boolean") {
    const normalized = edited.trim().toLowerCase();
    if (normalized !== "true" && normalized !== "false") {
      return { error: "修改后的布尔值只能填写 true 或 false" };
    }
    return { value: normalized === "true" };
  }
  if (Array.isArray(original)) {
    try {
      const value = JSON.parse(edited);
      return Array.isArray(value)
        ? { value }
        : { error: "修改后的列表必须是有效 JSON 数组" };
    } catch {
      return { error: "修改后的列表必须是有效 JSON 数组" };
    }
  }
  if (original !== null && typeof original === "object") {
    try {
      const value = JSON.parse(edited);
      return value !== null && typeof value === "object" && !Array.isArray(value)
        ? { value }
        : { error: "修改后的对象必须是有效 JSON 对象" };
    } catch {
      return { error: "修改后的对象必须是有效 JSON 对象" };
    }
  }
  return { value: edited };
}

function previewCategories(preview: ClaimPreviewView, claimId: string) {
  return [
    preview.added_claim_ids.includes(claimId) ? "新增" : null,
    preview.changed_claim_ids.includes(claimId) ? "变更" : null,
    preview.conflicting_claim_ids.includes(claimId) ? "冲突" : null,
  ].filter((value): value is string => Boolean(value));
}

export function EvidenceView({
  bridge,
  initialClaims,
  claimPreview,
}: EvidenceViewProps) {
  const [profile, setProfile] = useState<CandidateProfile | null>(null);
  const [scopes, setScopes] = useState<string[]>([]);
  const [actions, setActions] = useState<Record<string, ClaimAction>>({});
  const [editedValues, setEditedValues] = useState<Record<string, string>>({});
  const [previewUnavailable, setPreviewUnavailable] = useState(false);
  const [previewCompleted, setPreviewCompleted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [disconnectSource, setDisconnectSource] = useState<string | null>(null);
  const [disconnectMode, setDisconnectMode] = useState<"preserve" | "delete">("preserve");
  const [deleteConfirmed, setDeleteConfirmed] = useState(false);
  const previewIdRef = useRef(claimPreview?.preview_id);
  const previewRequestRef = useRef(0);
  previewIdRef.current = claimPreview?.preview_id;

  useEffect(() => {
    let active = true;
    bridge.callTool<CandidateProfile>("get_candidate_profile", {}).then((next) => {
      if (active) setProfile(next);
    }).catch(() => {
      if (active) setProfile({ id: "default", claims: initialClaims, import_receipts: [], version: "1", updated_at: "" });
    });
    return () => { active = false; };
  }, [bridge, initialClaims]);

  useEffect(() => {
    setActions({});
    setEditedValues({});
    setPreviewUnavailable(false);
    setPreviewCompleted(false);
    setBusy(false);
    setError("");
    previewRequestRef.current += 1;
  }, [claimPreview?.preview_id]);

  const groups = useMemo(() => {
    const confirmed = (profile?.claims ?? initialClaims).filter((claim) => claim.confirmed);
    return {
      resume: confirmed.filter((claim) => evidenceGroup(claim) === "resume"),
      user: confirmed.filter((claim) => evidenceGroup(claim) === "user"),
      agent: confirmed.filter((claim) => evidenceGroup(claim) === "agent"),
    };
  }, [initialClaims, profile]);

  const memorySources = useMemo(
    () => (profile?.memory_connections ?? [])
      .filter((connection) => connection.connected)
      .map((connection) => connection.source_agent),
    [profile],
  );

  const askAgent = async (regenerate = false) => {
    if (!regenerate && scopes.length === 0) {
      setError("请至少选择一个同步范围");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const selectedScope = regenerate ? (claimPreview?.scope ?? scopes) : scopes;
      await bridge.sendMessage(
        `${regenerate ? "请重新生成" : "请生成"} Agent Memory 候选证据预览。仅使用我授权的范围：${selectedScope.join("、")}。不要直接写入 Candidate Profile；请调用 preview_agent_memory_import，然后用 open_claim_review 打开逐条确认界面。`,
      );
      setMessage("已请求 Agent 生成预览；收到候选信息后仍需逐条确认。");
    } catch {
      setError("Agent 未接受请求，请稍后重试");
    } finally {
      setBusy(false);
    }
  };

  const submitPreview = async () => {
    if (!claimPreview) return;
    if (claimPreview.candidates.some((claim) => !actions[claim.id])) {
      setError("请逐条选择接受、修改或拒绝");
      return;
    }
    const decisions: ClaimDecision[] = [];
    for (const claim of claimPreview.candidates) {
      const action = actions[claim.id];
      if (action === "edit") {
        const parsed = parseEditedValue(
          claim.value,
          editedValues[claim.id] ?? displayValue(claim.value),
        );
        if (parsed.error) {
          setError(parsed.error);
          return;
        }
        decisions.push({ claim_id: claim.id, action, edited_value: parsed.value });
      } else {
        decisions.push({ claim_id: claim.id, action });
      }
    }
    const previewId = claimPreview.preview_id;
    const requestGeneration = ++previewRequestRef.current;
    setBusy(true);
    setError("");
    try {
      const tool = claimPreview.kind === "agent_memory"
        ? "confirm_agent_memory_import"
        : "confirm_profile_claims";
      const updated = await bridge.callTool<CandidateProfile>(tool, {
        preview_id: previewId,
        decisions,
      });
      if (
        previewIdRef.current !== previewId ||
        previewRequestRef.current !== requestGeneration
      ) return;
      setProfile(updated);
      setActions({});
      setEditedValues({});
      setPreviewCompleted(true);
      setMessage("候选证据已按你的逐条决定更新");
    } catch {
      if (
        previewIdRef.current !== previewId ||
        previewRequestRef.current !== requestGeneration
      ) return;
      setActions({});
      setEditedValues({});
      setPreviewUnavailable(true);
      setError("预览已失效，请重新生成");
    } finally {
      if (
        previewIdRef.current === previewId &&
        previewRequestRef.current === requestGeneration
      ) setBusy(false);
    }
  };

  const disconnectMemory = async () => {
    if (!disconnectSource || (disconnectMode === "delete" && !deleteConfirmed)) return;
    setBusy(true);
    setError("");
    try {
      const updated = await bridge.callTool<CandidateProfile>("disconnect_agent_memory", {
        source_agent: disconnectSource,
        delete_imported: disconnectMode === "delete",
      });
      setProfile(updated);
      setMessage(disconnectMode === "delete" ? "已断开来源并删除其导入证据" : "已断开来源；已导入证据予以保留");
      setDisconnectSource(null);
      setDisconnectMode("preserve");
      setDeleteConfirmed(false);
    } catch {
      setError("未能断开 Agent Memory 来源，请重试");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="evidence-view">
      <header className="editor-header">
        <div><h2>我的证据</h2><p>这里只显示已经确认、可以参与岗位判断的个人事实。</p></div>
        <ShieldCheck aria-hidden="true" size={22} />
      </header>

      <div className="evidence-groups">
        <EvidenceGroup title="简历证据" claims={groups.resume} />
        <EvidenceGroup title="用户确认" claims={groups.user} />
        <EvidenceGroup title="Agent 导入" claims={groups.agent} />
      </div>

      <section className="memory-sync" aria-labelledby="memory-sync-heading">
        <div className="section-heading"><h3 id="memory-sync-heading">Agent Memory</h3><span>默认关闭</span></div>
        <p>只请求所选范围的候选信息。Agent 不能直接修改你的证据。</p>
        <div className="scope-options">
          {scopeOptions.map((scope) => <label className="checkbox-field" key={scope}><input type="checkbox" checked={scopes.includes(scope)} onChange={(event) => setScopes(event.target.checked ? [...scopes, scope] : scopes.filter((item) => item !== scope))} />{scope}</label>)}
        </div>
        <button className="secondary-button" type="button" disabled={busy || scopes.length === 0} onClick={() => askAgent(false)}><Brain size={15} aria-hidden="true" />同步 Agent Memory</button>
        {memorySources.length > 0 && <div className="memory-sources"><strong>已确认来源</strong>{memorySources.map((source) => <button type="button" key={source} onClick={() => { setDisconnectSource(source); setDisconnectMode("preserve"); setDeleteConfirmed(false); }}><Link2Off size={14} aria-hidden="true" />断开 {source}</button>)}</div>}
        {disconnectSource && <fieldset className="disconnect-panel"><legend>断开 {disconnectSource}</legend><label><input type="radio" name="disconnect-mode" checked={disconnectMode === "preserve"} onChange={() => { setDisconnectMode("preserve"); setDeleteConfirmed(false); }} />保留已导入证据</label><label className="destructive-choice"><input type="radio" name="disconnect-mode" checked={disconnectMode === "delete"} onChange={() => setDisconnectMode("delete")} />删除此来源导入的证据</label>{disconnectMode === "delete" && <label className="destructive-confirm"><input type="checkbox" checked={deleteConfirmed} onChange={(event) => setDeleteConfirmed(event.target.checked)} />我确认删除 {disconnectSource} 导入的证据</label>}<div className="disconnect-actions"><button type="button" onClick={() => setDisconnectSource(null)}>取消</button><button className={disconnectMode === "delete" ? "danger-button" : "secondary-button"} type="button" disabled={busy || (disconnectMode === "delete" && !deleteConfirmed)} onClick={disconnectMemory}>{disconnectMode === "delete" ? "确认删除并断开" : "确认断开"}</button></div></fieldset>}
      </section>

      {claimPreview && !previewUnavailable && !previewCompleted && (
        <section className="claim-review" role="region" aria-label="候选证据确认">
          <div className="section-heading"><h3>候选证据确认</h3><span>{claimPreview.kind === "agent_memory" ? `来自 ${claimPreview.source_agent ?? "当前 Agent"}` : "来自简历解析"}</span></div>
          {claimPreview.scope.length > 0 && <p className="preview-scope">授权范围：{claimPreview.scope.join("、")}</p>}
          <div className="claim-review-list">
            {claimPreview.candidates.map((claim) => (
              <fieldset className="claim-review-row" key={claim.id}>
                <legend>{claim.claim_type}</legend>
                <div className="preview-categories">{previewCategories(claimPreview, claim.id).map((category) => <span className={`preview-category preview-category--${category === "冲突" ? "conflict" : "change"}`} key={category}>{category}</span>)}</div>
                <strong>{displayValue(claim.value)}</strong>
                <p>{claim.provenance || claim.source_quote || "未提供来源说明"}</p>
                <div className="claim-actions">
                  {(["accept", "edit", "reject"] as ClaimAction[]).map((action) => <label key={action}><input type="radio" name={`action-${claim.id}`} checked={actions[claim.id] === action} onChange={() => setActions({ ...actions, [claim.id]: action })} />{{ accept: "接受", edit: "修改", reject: "拒绝" }[action]}</label>)}
                </div>
                {actions[claim.id] === "edit" && <label>修改后的值<input aria-label="修改后的值" value={editedValues[claim.id] ?? displayValue(claim.value)} onChange={(event) => setEditedValues({ ...editedValues, [claim.id]: event.target.value })} /></label>}
              </fieldset>
            ))}
          </div>
          <button className="primary-button" type="button" disabled={busy} onClick={submitPreview}>提交确认</button>
        </section>
      )}

      {previewUnavailable && <div className="preview-recovery"><button type="button" onClick={() => askAgent(true)}><RefreshCw size={14} aria-hidden="true" />重新生成预览</button></div>}
      {error && <p className="form-message form-message--error" role="alert">{error}</p>}
      {message && <p className="form-message" role="status">{message}</p>}
    </div>
  );
}

function EvidenceGroup({ title, claims }: { title: string; claims: CandidateClaim[] }) {
  return (
    <section className="evidence-group">
      <div className="section-heading"><h3>{title}</h3><span>{claims.length}</span></div>
      {claims.length === 0 ? <p className="empty-line">暂无已确认信息</p> : <ul>{claims.map((claim) => <li key={claim.id}><div><strong>{claim.claim_type}</strong><span>{displayValue(claim.value)}</span></div>{claim.source_quote && <blockquote>{claim.source_quote}</blockquote>}<small>{claim.provenance || claim.import_origin?.provenance || "用户确认"}</small></li>)}</ul>}
    </section>
  );
}
