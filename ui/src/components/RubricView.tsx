import { RotateCcw, Save } from "lucide-react";
import { useEffect, useState } from "react";

import type { McpAppBridge } from "../bridge";
import { dimensionPatchForRequirement } from "../criteria";
import type { CriterionCategory, RequirementLevel, Rubric, RubricPatch, ValueDimension } from "../contracts";
import { CriteriaEditor, type CriteriaEditorMode } from "./CriteriaEditor";
import { CriteriaOverview } from "./CriteriaOverview";

interface RubricViewProps {
  bridge: McpAppBridge;
  selectedAnalysisId: string | null;
  staleJobCount?: number;
  selectedJobIsStale?: boolean;
  onSaved?: () => void;
  onRecomputed: (analysisId: string) => void;
}

const defaultCopy: Record<string, { name: string; desired: string }> = {
  role_family_fit: { name: "岗位方向", desired: "AI 工程、应用科学或 AI 产品岗位" },
  applied_ai_depth: { name: "应用 AI 深度", desired: "构建并评估面向产品或工作流的 AI 行为" },
  product_customer_proximity: { name: "产品与客户距离", desired: "影响产品决策，并直接从用户或客户获得反馈" },
  ownership_impact: { name: "端到端负责与影响", desired: "端到端负责工作，并产生可衡量的影响" },
  infrastructure_balance: { name: "纯基础设施占比", desired: "避免没有 AI 或产品结果的纯基础设施工作" },
  work_arrangement: { name: "办公方式", desired: "符合已确认的地点与到岗偏好" },
  compensation: { name: "薪资待遇", desired: "符合已确认的薪资要求" },
  work_authorization: { name: "工作许可", desired: "避免与已确认的工作许可事实冲突" },
};

function localize(rubric: Rubric): Rubric {
  return {
    ...rubric,
    pending_preferences: rubric.pending_preferences ?? [],
    dimensions: rubric.dimensions.map((item) => {
      const copy = defaultCopy[item.id];
      return copy ? { ...item, name: copy.name, desired_value: copy.desired } : item;
    }),
  };
}

export function RubricView({
  bridge,
  selectedAnalysisId,
  staleJobCount = 0,
  selectedJobIsStale = false,
  onSaved = () => undefined,
  onRecomputed,
}: RubricViewProps) {
  const [baseline, setBaseline] = useState<Rubric | null>(null);
  const [draft, setDraft] = useState<Rubric | null>(null);
  const [mode, setMode] = useState<"overview" | CriteriaEditorMode>("overview");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    bridge.callTool<Rubric>("get_rubric", {}).then((value) => {
      if (!active) return;
      const next = localize(value);
      setBaseline(next);
      setDraft(structuredClone(next));
    }).catch(() => active && setError("无法载入筛选标准，请重试"));
    return () => { active = false; };
  }, [bridge]);

  if (!draft || !baseline) {
    return <div className={`state-message${error ? " state-message--error" : ""}`}>{error || "正在载入筛选标准…"}</div>;
  }

  const changeRequirement = (id: string, level: RequirementLevel) => {
    setDraft((current) => current && ({
      ...current,
      dimensions: current.dimensions.map((item) => item.id === id ? { ...item, ...dimensionPatchForRequirement(level) } : item),
    }));
  };

  const save = async () => {
    const dimensionUpdates: Record<string, Partial<ValueDimension>> = {};
    draft.dimensions.forEach((item) => {
      const original = baseline.dimensions.find((candidate) => candidate.id === item.id);
      if (!original) return;
      if (item.requirement_level !== original.requirement_level) {
        dimensionUpdates[item.id] = dimensionPatchForRequirement(item.requirement_level);
      }
    });
    if (!Object.keys(dimensionUpdates).length) {
      setStatus("没有需要保存的更改");
      return;
    }
    const patch: RubricPatch = { dimension_updates: dimensionUpdates };
    setBusy(true);
    setError("");
    try {
      const updated = localize(await bridge.callTool<Rubric>("update_rubric", { patch }));
      setBaseline(updated);
      setDraft(structuredClone(updated));
      setStatus("标准已保存；现有岗位需要重新计算");
      onSaved();
    } catch {
      setError("标准未保存，请重试");
    } finally {
      setBusy(false);
    }
  };

  const recompute = async () => {
    if (!selectedAnalysisId || busy) return;
    setBusy(true);
    try {
      await bridge.callTool("recompute_decision", { analysis_id: selectedAnalysisId });
      onRecomputed(selectedAnalysisId);
      setStatus("当前岗位已按新标准重新计算");
    } catch {
      setError("重新计算失败，请稍后重试");
    } finally {
      setBusy(false);
    }
  };

  const openCategory = (category: CriterionCategory) => setMode({ kind: "category", category });

  return <div className="criteria-view" aria-label="筛选标准工作区">
    {mode === "overview" ? (
      <CriteriaOverview
        dimensions={draft.dimensions}
        pendingCount={draft.pending_preferences.length}
        onEditCategory={openCategory}
        onViewAll={() => setMode({ kind: "all" })}
        onAddPreference={() => setStatus("自定义偏好编辑器即将打开")}
      />
    ) : (
      <>
        <CriteriaEditor dimensions={draft.dimensions} mode={mode} onBack={() => setMode("overview")} onRequirementChange={changeRequirement} />
        <div className="criteria-actions"><button className="primary-button" type="button" disabled={busy} onClick={save}><Save size={15} aria-hidden="true" />保存更改</button></div>
      </>
    )}
    {error && <p className="form-message form-message--error" role="alert">{error}</p>}
    {status && <p className="form-message" role="status">{status}</p>}
    {staleJobCount > 0 && <div className="recompute-bar"><span>{staleJobCount} 个岗位需要重新计算。</span><button type="button" disabled={!selectedAnalysisId || !selectedJobIsStale || busy} onClick={recompute}><RotateCcw size={14} aria-hidden="true" />重新计算当前岗位</button></div>}
  </div>;
}
