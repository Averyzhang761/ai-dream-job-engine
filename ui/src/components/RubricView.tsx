import { Plus, RotateCcw, Save, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { McpAppBridge } from "../bridge";
import type {
  CandidateComparison,
  FactDefinition,
  RecommendationLevel,
  Rubric,
  RubricPatch,
  ValueDimension,
} from "../contracts";

interface RubricViewProps {
  bridge: McpAppBridge;
  selectedAnalysisId: string | null;
  staleJobCount?: number;
  selectedJobIsStale?: boolean;
  onSaved?: () => void;
  onRecomputed: (analysisId: string) => void;
}

const recommendationLevels: RecommendationLevel[] = [
  "priority_apply",
  "apply_after_review",
  "hold",
  "skip",
];

const levelLabels: Record<RecommendationLevel, string> = {
  priority_apply: "优先申请",
  apply_after_review: "审核后申请",
  hold: "暂缓",
  skip: "跳过",
};

const comparisonLabels: Record<CandidateComparison, string> = {
  none: "不比较候选人证据",
  exact: "精确匹配",
  allowed_alternatives: "允许替代项",
  required_subset: "满足必需子集",
  candidate_at_least: "候选人至少达到",
};

const emptyFact = (): FactDefinition => ({
  id: "",
  name: "",
  definition: "",
  data_type: "text",
  extraction_instructions: "",
  requires_evidence: true,
  missing_policy: "unknown",
  default_value: null,
  enum_values: [],
  candidate_comparison: "none",
  qualification_weight: 0,
});

const emptyDimension = (): ValueDimension => ({
  id: "",
  name: "",
  definition: "",
  examples: [],
  fact_ids: [],
  signal_ids: [],
  weight: 1,
  scoring_direction: "higher_is_better",
  positive_keywords: [],
  negative_keywords: [],
  hard_constraint: false,
  threshold: null,
  explanation_template: "",
});

function sameValue(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function changedFields<T extends { id: string }>(original: T, next: T) {
  return Object.fromEntries(
    Object.entries(next).filter(
      ([key, value]) => key !== "id" && !sameValue(value, original[key as keyof T]),
    ),
  );
}

function splitList(value: string) {
  return value.split(/[,，\n]/).map((item) => item.trim()).filter(Boolean);
}

function parseDefaultInput(fact: FactDefinition, value: string): unknown {
  if (fact.data_type === "boolean") return value === "true";
  if (fact.data_type === "number") return value;
  if (fact.data_type === "list") return splitList(value);
  return value;
}

function normalizedFact(fact: FactDefinition): FactDefinition {
  if (fact.default_value === null || fact.data_type !== "number") return fact;
  return { ...fact, default_value: Number(fact.default_value) };
}

function factDefaultError(fact: FactDefinition): string | null {
  if (fact.default_value === null) {
    return fact.missing_policy === "default" ? "配置默认值时不能为空" : null;
  }
  if (fact.data_type === "number") {
    const raw = fact.default_value;
    if ((typeof raw !== "number" && typeof raw !== "string") || String(raw).trim() === "" || !Number.isFinite(Number(raw))) {
      return "数字默认值必须是有限数字";
    }
  }
  if (fact.data_type === "boolean" && typeof fact.default_value !== "boolean") {
    return "布尔默认值必须是 true 或 false";
  }
  if (fact.data_type === "text" && typeof fact.default_value !== "string") {
    return "文本默认值必须是字符串";
  }
  if (fact.data_type === "enum" && (typeof fact.default_value !== "string" || !fact.enum_values.includes(fact.default_value))) {
    return "枚举默认值必须属于已声明选项";
  }
  if (fact.data_type === "list" && !Array.isArray(fact.default_value)) {
    return "列表默认值必须是列表";
  }
  return null;
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
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedFactIndex, setSelectedFactIndex] = useState(0);
  const [selectedDimensionIndex, setSelectedDimensionIndex] = useState(0);

  useEffect(() => {
    let active = true;
    bridge.callTool<Rubric>("get_rubric", {}).then((rubric) => {
      if (!active) return;
      setBaseline(rubric);
      setDraft(structuredClone(rubric));
    }).catch(() => {
      if (active) setError("无法载入岗位标准，请重试");
    });
    return () => { active = false; };
  }, [bridge]);

  const knownInputs = useMemo(() => ({
    facts: new Set(draft?.facts.map((fact) => fact.id).filter(Boolean) ?? []),
    signals: new Set(draft?.questions.map((question) => question.id) ?? []),
  }), [draft]);

  if (!draft || !baseline) {
    return <div className={`state-message${error ? " state-message--error" : ""}`}>{error || "正在载入岗位标准…"}</div>;
  }

  const updateFact = (index: number, changes: Partial<FactDefinition>) => {
    setDraft((current) => current && ({
      ...current,
      facts: current.facts.map((fact, itemIndex) => itemIndex === index ? { ...fact, ...changes } : fact),
    }));
  };

  const updateDimension = (index: number, changes: Partial<ValueDimension>) => {
    setDraft((current) => current && ({
      ...current,
      dimensions: current.dimensions.map((dimension, itemIndex) => itemIndex === index ? { ...dimension, ...changes } : dimension),
    }));
  };

  const addFact = () => {
    const index = draft.facts.length;
    setDraft({ ...draft, facts: [...draft.facts, emptyFact()] });
    setSelectedFactIndex(index);
  };

  const addDimension = () => {
    const index = draft.dimensions.length;
    setDraft({ ...draft, dimensions: [...draft.dimensions, emptyDimension()] });
    setSelectedDimensionIndex(index);
  };

  const validate = () => {
    const newFacts = draft.facts.slice(baseline.facts.length);
    if (draft.facts.some((fact) => !fact.id.trim() || !fact.name.trim() || !fact.definition.trim() || !fact.extraction_instructions.trim())) {
      return newFacts.length > 0
        ? "请补全新增事实字段的 ID、名称、定义和提取说明"
        : "请补全事实字段的名称、定义和提取说明";
    }
    if (draft.facts.some((fact) => !Number.isFinite(fact.qualification_weight) || fact.qualification_weight < 0)) {
      return "请输入有效数字：事实权重必须是大于或等于 0 的数字";
    }
    if (new Set(draft.facts.map((fact) => fact.id)).size !== draft.facts.length || new Set(draft.dimensions.map((dimension) => dimension.id)).size !== draft.dimensions.length) {
      return "事实字段和价值维度的 ID 不能重复";
    }
    const defaultError = draft.facts.map(factDefaultError).find(Boolean);
    if (defaultError) return defaultError;
    if (draft.facts.some((fact) =>
      (fact.data_type === "enum" && fact.enum_values.length === 0) ||
      (fact.candidate_comparison === "none" && fact.qualification_weight !== 0) ||
      (fact.candidate_comparison !== "none" && fact.qualification_weight <= 0) ||
      (["allowed_alternatives", "required_subset"].includes(fact.candidate_comparison) && fact.data_type !== "list") ||
      (fact.candidate_comparison === "candidate_at_least" && fact.data_type !== "number")
    )) {
      return "事实字段的数据类型、默认值、比较方式和胜任力权重不一致";
    }
    const newDimensions = draft.dimensions.slice(baseline.dimensions.length);
    if (draft.dimensions.some((dimension) => !dimension.id.trim() || !dimension.name.trim() || !dimension.definition.trim() || dimension.fact_ids.length + dimension.signal_ids.length === 0)) {
      return newDimensions.length > 0
        ? "请补全新增价值维度，并至少选择一个事实或信号"
        : "请补全价值维度，并至少选择一个事实或信号";
    }
    if (draft.dimensions.some((dimension) => !Number.isFinite(dimension.weight) || dimension.weight < 0 || (dimension.threshold !== null && (!Number.isFinite(dimension.threshold) || dimension.threshold < 0 || dimension.threshold > 1)))) {
      return "请输入有效数字：权重需大于或等于 0，阈值需在 0 到 1 之间";
    }
    if (draft.dimensions.some((dimension) => dimension.fact_ids.some((id) => !knownInputs.facts.has(id)) || dimension.signal_ids.some((id) => !knownInputs.signals.has(id)))) {
      return "价值维度只能引用当前已定义的事实和信号";
    }
    const factsById = new Map(draft.facts.map((fact) => [fact.id, fact]));
    if (draft.dimensions.some((dimension) => dimension.hard_constraint && !dimension.fact_ids.some((id) => factsById.get(id)?.candidate_comparison !== "none"))) {
      return "硬性约束必须引用一个启用候选人比较的事实字段";
    }
    const thresholds = draft.recommendation_thresholds;
    if (recommendationLevels.some((level) => !Number.isFinite(thresholds[level]) || thresholds[level] < 0 || thresholds[level] > 1) || !(thresholds.priority_apply > thresholds.apply_after_review && thresholds.apply_after_review > thresholds.hold && thresholds.hold > thresholds.skip && thresholds.skip === 0)) {
      return "请输入有效数字：推荐阈值需依次降低，且跳过阈值必须为 0";
    }
    return "";
  };

  const save = async () => {
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }
    const patch: RubricPatch = {};
    const factUpdates: Record<string, Partial<FactDefinition>> = {};
    draft.facts.forEach((fact, index) => {
      const normalized = normalizedFact(fact);
      const changes = index < baseline.facts.length ? changedFields(baseline.facts[index], normalized) : Object.fromEntries(Object.entries(normalized).filter(([key]) => key !== "id"));
      if (Object.keys(changes).length) factUpdates[fact.id] = changes;
    });
    const dimensionUpdates: Record<string, Partial<ValueDimension>> = {};
    draft.dimensions.forEach((dimension, index) => {
      const changes = index < baseline.dimensions.length ? changedFields(baseline.dimensions[index], dimension) : Object.fromEntries(Object.entries(dimension).filter(([key]) => key !== "id"));
      if (Object.keys(changes).length) dimensionUpdates[dimension.id] = changes;
    });
    const thresholdUpdates = Object.fromEntries(recommendationLevels.filter((level) => draft.recommendation_thresholds[level] !== baseline.recommendation_thresholds[level]).map((level) => [level, draft.recommendation_thresholds[level]]));
    if (Object.keys(factUpdates).length) patch.fact_updates = factUpdates;
    if (Object.keys(dimensionUpdates).length) patch.dimension_updates = dimensionUpdates;
    if (Object.keys(thresholdUpdates).length) patch.threshold_updates = thresholdUpdates;
    if (!Object.keys(patch).length) {
      setStatus("没有需要保存的更改");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const updated = await bridge.callTool<Rubric>("update_rubric", { patch });
      setBaseline(updated);
      setDraft(structuredClone(updated));
      onSaved();
      setStatus("标准已保存；现有岗位需要重新计算");
    } catch {
      setError("标准未保存，请检查字段后重试");
    } finally {
      setBusy(false);
    }
  };

  const recompute = async () => {
    if (!selectedAnalysisId || busy) return;
    setBusy(true);
    setError("");
    try {
      await bridge.callTool("recompute_decision", { analysis_id: selectedAnalysisId });
      setStatus("当前岗位已按新标准重新计算");
      onRecomputed(selectedAnalysisId);
    } catch {
      setError("重新计算失败，请稍后重试");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="editor-view" aria-label="岗位标准编辑器">
      <header className="editor-header">
        <div><h2>我的标准</h2><p>定义提取哪些事实，以及这些事实如何影响推荐。</p></div>
        <button className="primary-button" type="button" disabled={busy} onClick={save}><Save size={15} aria-hidden="true" />保存标准</button>
      </header>

      <section className="editor-section" aria-labelledby="facts-heading">
        <div className="section-heading"><div><h3 id="facts-heading">事实字段</h3><span>{draft.facts.length} 项</span></div><button aria-label="新增事实字段" type="button" onClick={addFact}><Plus size={14} aria-hidden="true" />新增</button></div>
        <label className="definition-selector">选择要修改的事实<select aria-label="选择事实字段" value={selectedFactIndex} onChange={(event) => setSelectedFactIndex(Number(event.target.value))}>{draft.facts.map((fact, index) => <option value={index} key={`${fact.id || "new"}-${index}`}>{fact.name || "新事实"} · {fact.data_type} · 权重 {fact.qualification_weight}</option>)}</select></label>
        {draft.facts[selectedFactIndex] && (() => {
          const fact = draft.facts[selectedFactIndex];
          const index = selectedFactIndex;
          return <fieldset className="definition-editor definition-editor--selected">
            <legend>编辑：{fact.name || "新事实字段"}</legend>
            <label>名称<input aria-label={`事实名称 ${fact.id}`} value={fact.name} onChange={(event) => updateFact(index, { name: event.target.value })} /></label>
            <label>候选人比较<select value={fact.candidate_comparison} onChange={(event) => updateFact(index, { candidate_comparison: event.target.value as CandidateComparison })}>{Object.entries(comparisonLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            <label className="wide-field">定义<textarea value={fact.definition} onChange={(event) => updateFact(index, { definition: event.target.value })} /></label>
            <label>胜任力权重<input aria-label={`事实权重 ${fact.id}`} inputMode="decimal" value={String(fact.qualification_weight)} onChange={(event) => updateFact(index, { qualification_weight: Number(event.target.value) })} /></label>
            <label className="checkbox-field"><input type="checkbox" checked={fact.requires_evidence} onChange={(event) => updateFact(index, { requires_evidence: event.target.checked })} />要求 JD 原文证据</label>
            <details className="advanced-fields wide-field">
              <summary><SlidersHorizontal size={14} aria-hidden="true" />高级设置</summary>
              <div className="advanced-field-grid">
                <label>ID<input value={fact.id} disabled={index < baseline.facts.length} onChange={(event) => updateFact(index, { id: event.target.value.trim() })} /></label>
                <label>数据类型<select value={fact.data_type} onChange={(event) => updateFact(index, { data_type: event.target.value as FactDefinition["data_type"] })}><option value="boolean">布尔</option><option value="number">数字</option><option value="text">文本</option><option value="enum">枚举</option><option value="list">列表</option></select></label>
                <label className="wide-field">提取说明<textarea value={fact.extraction_instructions} onChange={(event) => updateFact(index, { extraction_instructions: event.target.value })} /></label>
                <label>缺失处理<select value={fact.missing_policy} onChange={(event) => updateFact(index, { missing_policy: event.target.value as FactDefinition["missing_policy"] })}><option value="unknown">标记未知</option><option value="default">使用默认值</option></select></label>
                <label>枚举值（逗号分隔）<input value={fact.enum_values.join(", ")} onChange={(event) => updateFact(index, { enum_values: splitList(event.target.value) })} /></label>
                <label>默认值{fact.data_type === "boolean" ? <select aria-label={`事实默认值 ${fact.id}`} disabled={fact.missing_policy !== "default"} value={String(fact.default_value ?? false)} onChange={(event) => updateFact(index, { default_value: event.target.value === "true" })}><option value="false">否</option><option value="true">是</option></select> : <input aria-label={`事实默认值 ${fact.id}`} disabled={fact.missing_policy !== "default"} value={fact.default_value === null ? "" : Array.isArray(fact.default_value) ? fact.default_value.join(", ") : String(fact.default_value)} onChange={(event) => updateFact(index, { default_value: parseDefaultInput(fact, event.target.value) })} />}</label>
              </div>
            </details>
          </fieldset>;
        })()}
      </section>

      <section className="editor-section" aria-labelledby="dimensions-heading">
        <div className="section-heading"><div><h3 id="dimensions-heading">价值维度</h3><span>{draft.dimensions.length} 项</span></div><button aria-label="新增价值维度" type="button" onClick={addDimension}><Plus size={14} aria-hidden="true" />新增</button></div>
        <label className="definition-selector">选择要修改的维度<select aria-label="选择价值维度" value={selectedDimensionIndex} onChange={(event) => setSelectedDimensionIndex(Number(event.target.value))}>{draft.dimensions.map((dimension, index) => <option value={index} key={`${dimension.id || "new"}-${index}`}>{dimension.name || "新维度"} · 权重 {dimension.weight}{dimension.hard_constraint ? " · 硬约束" : ""}</option>)}</select></label>
        {draft.dimensions[selectedDimensionIndex] && (() => {
          const dimension = draft.dimensions[selectedDimensionIndex];
          const index = selectedDimensionIndex;
          return <fieldset className="definition-editor definition-editor--selected">
            <legend>编辑：{dimension.name || "新价值维度"}</legend>
            <label>名称<input aria-label={`维度名称 ${dimension.id}`} value={dimension.name} onChange={(event) => updateDimension(index, { name: event.target.value })} /></label>
            <label>权重<input aria-label={`维度权重 ${dimension.id}`} inputMode="decimal" value={Number.isNaN(dimension.weight) ? "" : String(dimension.weight)} onChange={(event) => updateDimension(index, { weight: event.target.value === "" ? Number.NaN : Number(event.target.value) })} /></label>
            <label className="wide-field">定义<textarea value={dimension.definition} onChange={(event) => updateDimension(index, { definition: event.target.value })} /></label>
            <label>阈值<input inputMode="decimal" value={dimension.threshold ?? ""} onChange={(event) => updateDimension(index, { threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
            <label className="checkbox-field"><input type="checkbox" checked={dimension.hard_constraint} onChange={(event) => updateDimension(index, { hard_constraint: event.target.checked })} />硬性约束</label>
            <details className="advanced-fields wide-field">
              <summary><SlidersHorizontal size={14} aria-hidden="true" />高级设置</summary>
              <div className="advanced-field-grid">
                <label>ID<input value={dimension.id} disabled={index < baseline.dimensions.length} onChange={(event) => updateDimension(index, { id: event.target.value.trim() })} /></label>
                <label>计分方向<select value={dimension.scoring_direction} onChange={(event) => updateDimension(index, { scoring_direction: event.target.value as ValueDimension["scoring_direction"] })}><option value="higher_is_better">越高越好</option><option value="lower_is_better">越低越好</option></select></label>
                <label>正向关键词<input value={dimension.positive_keywords.join(", ")} onChange={(event) => updateDimension(index, { positive_keywords: splitList(event.target.value) })} /></label>
                <label>负向关键词<input value={dimension.negative_keywords.join(", ")} onChange={(event) => updateDimension(index, { negative_keywords: splitList(event.target.value) })} /></label>
                <label>示例<input value={dimension.examples.join(", ")} onChange={(event) => updateDimension(index, { examples: splitList(event.target.value) })} /></label>
                <label className="wide-field">解释模板<textarea value={dimension.explanation_template ?? ""} onChange={(event) => updateDimension(index, { explanation_template: event.target.value || null })} /></label>
                <div className="reference-picker wide-field"><span>引用事实</span>{draft.facts.filter((fact) => fact.id).map((fact) => <label className="checkbox-field" key={fact.id}><input type="checkbox" checked={dimension.fact_ids.includes(fact.id)} onChange={(event) => updateDimension(index, { fact_ids: event.target.checked ? [...dimension.fact_ids, fact.id] : dimension.fact_ids.filter((id) => id !== fact.id) })} />{fact.name || fact.id}</label>)}</div>
                <div className="reference-picker wide-field"><span>引用信号</span>{draft.questions.map((question) => <label className="checkbox-field" key={question.id}><input type="checkbox" checked={dimension.signal_ids.includes(question.id)} onChange={(event) => updateDimension(index, { signal_ids: event.target.checked ? [...dimension.signal_ids, question.id] : dimension.signal_ids.filter((id) => id !== question.id) })} />{question.question}</label>)}</div>
              </div>
            </details>
          </fieldset>;
        })()}
      </section>

      <details className="editor-section threshold-panel"><summary><span><strong>推荐阈值</strong><small>{recommendationLevels.map((level) => draft.recommendation_thresholds[level]).join(" / ")}</small></span><SlidersHorizontal size={14} aria-hidden="true" /></summary><div className="field-grid">{recommendationLevels.map((level) => <label key={level}>{levelLabels[level]}<input aria-label={`${levelLabels[level]}阈值`} inputMode="decimal" value={Number.isNaN(draft.recommendation_thresholds[level]) ? "" : String(draft.recommendation_thresholds[level])} onChange={(event) => setDraft({ ...draft, recommendation_thresholds: { ...draft.recommendation_thresholds, [level]: event.target.value === "" ? Number.NaN : Number(event.target.value) } })} /></label>)}</div></details>

      {error && <p className="form-message form-message--error" role="alert">{error}</p>}
      {status && <p className="form-message" role="status">{status}</p>}
      {staleJobCount > 0 && <div className="recompute-bar"><span>{staleJobCount} 个岗位需要重新计算。</span><button type="button" disabled={!selectedAnalysisId || !selectedJobIsStale || busy} onClick={recompute}><RotateCcw size={14} aria-hidden="true" />重新计算当前岗位</button>{!selectedAnalysisId && <small>请先在“岗位”中选择一项。</small>}{selectedAnalysisId && !selectedJobIsStale && <small>当前岗位已是最新结果。</small>}</div>}
    </div>
  );
}
