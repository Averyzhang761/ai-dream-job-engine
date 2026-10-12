import { Pencil, Plus, RotateCcw, Save, Search, SlidersHorizontal, X } from "lucide-react";
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

const weightOptions = [
  { value: 0, label: "不参与" },
  { value: 0.5, label: "较低" },
  { value: 0.75, label: "次要" },
  { value: 1, label: "一般" },
  { value: 1.5, label: "重要" },
  { value: 2, label: "最高" },
];

const defaultChineseCopy: Record<string, { sourceName: string; name: string; definition: string }> = {
  role_family: { sourceName: "Role family", name: "岗位方向", definition: "岗位属于哪一种工程、产品或研究方向。" },
  ai_scope: { sourceName: "Applied AI scope", name: "AI 工作内容", definition: "岗位在多大程度上直接构建、评估或运营 AI 产品能力。" },
  product_customer_scope: { sourceName: "Product and customer scope", name: "产品与客户接触", definition: "岗位与产品决策、用户或客户的直接接触程度。" },
  ownership_scope: { sourceName: "Ownership and impact scope", name: "职责与影响力", definition: "岗位需要负责的决策、交付范围和可衡量结果。" },
  infrastructure_scope: { sourceName: "Infrastructure scope", name: "基础设施占比", definition: "岗位中缺少明确 AI 或产品结果的通用平台、基础设施工作占比。" },
  work_arrangement_terms: { sourceName: "Work arrangement", name: "办公方式", definition: "岗位写明的地点、远程、混合办公、到岗和出差要求。" },
  compensation_terms: { sourceName: "Compensation", name: "薪资待遇", definition: "岗位写明的薪资范围和其他报酬条件。" },
  work_authorization_requirement: { sourceName: "Work authorization requirement", name: "工作许可要求", definition: "岗位写明的工作许可、公民身份、安全审查或签证支持要求。" },
  required_skills: { sourceName: "Required skills", name: "必需技能", definition: "岗位明确要求的技能和技术。" },
  minimum_experience_years: { sourceName: "Minimum experience", name: "最低经验年限", definition: "岗位明确要求的最低相关工作年限。" },
  applied_ai_requirements: { sourceName: "Applied AI requirements", name: "应用 AI 要求", definition: "岗位明确要求的应用 AI 能力。" },
  role_family_fit: { sourceName: "Role family fit", name: "岗位方向匹配", definition: "岗位是否属于 AI 工程、应用科学或 AI 产品方向。" },
  applied_ai_depth: { sourceName: "Applied AI depth", name: "应用 AI 深度", definition: "岗位是否真正负责构建和评估产品或工作流中的 AI 行为。" },
  product_customer_proximity: { sourceName: "Product and customer proximity", name: "产品与客户距离", definition: "岗位能否直接影响产品决策并从用户或客户处获得反馈。" },
  ownership_impact: { sourceName: "Ownership and impact", name: "端到端负责与影响", definition: "岗位要求的端到端责任和可衡量影响程度。" },
  infrastructure_balance: { sourceName: "Pure infrastructure concentration", name: "纯基础设施占比", definition: "岗位是否过度集中于通用基础设施，而缺少明确的 AI 或产品结果。" },
  work_arrangement: { sourceName: "Work arrangement", name: "办公方式匹配", definition: "岗位地点和到岗要求是否符合已确认偏好。" },
  compensation: { sourceName: "Compensation", name: "薪资匹配", definition: "岗位薪资信息是否符合已确认要求。" },
  work_authorization: { sourceName: "Work authorization", name: "工作许可匹配", definition: "岗位的工作许可要求是否与已确认的候选人事实冲突。" },
};

function localizeDefaultRubric(rubric: Rubric): Rubric {
  const localize = <T extends FactDefinition | ValueDimension>(item: T): T => {
    const copy = defaultChineseCopy[item.id];
    return copy && item.name === copy.sourceName
      ? { ...item, name: copy.name, definition: copy.definition }
      : item;
  };
  return {
    ...rubric,
    facts: rubric.facts.map(localize),
    dimensions: rubric.dimensions.map(localize),
  };
}

function WeightSelect({ id, value, onChange }: { id: string; value: number; onChange: (value: number) => void }) {
  const isPreset = weightOptions.some((option) => option.value === value);
  return <select aria-label={`重要程度 ${id}`} value={value} onChange={(event) => onChange(Number(event.target.value))}>{!isPreset && <option value={value}>自定义 · {value}</option>}{weightOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select>;
}

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
  const [activeCollection, setActiveCollection] = useState<"dimensions" | "facts">("dimensions");
  const [editingItem, setEditingItem] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let active = true;
    bridge.callTool<Rubric>("get_rubric", {}).then((rubric) => {
      if (!active) return;
      const localized = localizeDefaultRubric(rubric);
      setBaseline(localized);
      setDraft(structuredClone(localized));
    }).catch(() => {
      if (active) setError("无法载入岗位标准，请重试");
    });
    return () => { active = false; };
  }, [bridge]);

  const knownInputs = useMemo(() => ({
    facts: new Set(draft?.facts.map((fact) => fact.id).filter(Boolean) ?? []),
    signals: new Set(draft?.questions.map((question) => question.id) ?? []),
  }), [draft]);

  const visibleDimensions = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return (draft?.dimensions ?? []).map((item, index) => ({ item, index })).filter(({ item }) => !normalized || `${item.name} ${item.definition}`.toLocaleLowerCase().includes(normalized));
  }, [draft, query]);

  const visibleFacts = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return (draft?.facts ?? []).map((item, index) => ({ item, index })).filter(({ item }) => !normalized || `${item.name} ${item.definition}`.toLocaleLowerCase().includes(normalized));
  }, [draft, query]);

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
    setActiveCollection("facts");
    setEditingItem(`fact:${index}`);
  };

  const addDimension = () => {
    const index = draft.dimensions.length;
    setDraft({ ...draft, dimensions: [...draft.dimensions, emptyDimension()] });
    setActiveCollection("dimensions");
    setEditingItem(`dimension:${index}`);
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
      const updated = localizeDefaultRubric(await bridge.callTool<Rubric>("update_rubric", { patch }));
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

  const editingDimensionIndex = editingItem?.startsWith("dimension:")
    ? Number(editingItem.split(":")[1])
    : null;
  const editingFactIndex = editingItem?.startsWith("fact:")
    ? Number(editingItem.split(":")[1])
    : null;
  const editingDimension = editingDimensionIndex === null ? null : draft.dimensions[editingDimensionIndex];
  const editingFact = editingFactIndex === null ? null : draft.facts[editingFactIndex];

  return (
    <div className="editor-view" aria-label="岗位标准编辑器">
      <header className="editor-header">
        <div><h2>我的标准</h2><p>{draft.dimensions.length} 条求职标准 · {draft.facts.length} 个判断字段</p></div>
        <button className="primary-button" type="button" disabled={busy} onClick={save}><Save size={15} aria-hidden="true" />保存标准</button>
      </header>

      <section className="standards-matrix" aria-label="标准矩阵">
        <div className="standards-toolbar">
          <div className="collection-switch" role="tablist" aria-label="标准类型">
            <button type="button" role="tab" aria-selected={activeCollection === "dimensions"} onClick={() => { setActiveCollection("dimensions"); setEditingItem(null); }}>求职标准 <span>{draft.dimensions.length}</span></button>
            <button type="button" role="tab" aria-selected={activeCollection === "facts"} onClick={() => { setActiveCollection("facts"); setEditingItem(null); }}>判断字段 <span>{draft.facts.length}</span></button>
          </div>
          <label className="standards-search"><Search size={14} aria-hidden="true" /><span className="sr-only">搜索标准</span><input type="search" placeholder="搜索名称或说明" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
          <details className="add-standard-menu"><summary><Plus size={14} aria-hidden="true" />新增</summary><div><button type="button" onClick={addDimension}>求职标准</button><button type="button" onClick={addFact}>判断字段</button></div></details>
        </div>

        <div className={`standards-workspace${editingItem ? " standards-workspace--editing" : ""}`}>
        {activeCollection === "dimensions" ? <div className="standards-list" role="table" aria-label="求职标准列表">
          <div className="standards-list-header" role="row"><span role="columnheader">标准</span><span role="columnheader">重要程度</span><span role="columnheader">硬性</span><span className="sr-only" role="columnheader">编辑</span></div>
          {visibleDimensions.map(({ item: dimension, index }) => {
            const key = `dimension:${index}`;
            const selected = editingItem === key;
            return <div className={`standard-row${selected ? " standard-row--selected" : ""}`} role="row" key={`${dimension.id || "new"}-${index}`}>
              <button className="standard-name" type="button" role="cell" aria-pressed={selected} onClick={() => setEditingItem(key)}><strong>{dimension.name || "未命名标准"}</strong><span>{dimension.definition || "尚未填写说明"}</span></button>
              <div className="standard-control" role="cell"><WeightSelect id={dimension.id} value={dimension.weight} onChange={(weight) => updateDimension(index, { weight })} /></div>
              <label className="compact-toggle" role="cell"><input aria-label={`硬性标准 ${dimension.id}`} type="checkbox" checked={dimension.hard_constraint} onChange={(event) => updateDimension(index, { hard_constraint: event.target.checked })} /><span>硬性</span></label>
              <button className="icon-button" type="button" aria-label={`编辑 ${dimension.name || "未命名标准"}`} title="编辑" onClick={() => setEditingItem(key)}><Pencil size={15} aria-hidden="true" /></button>
            </div>;
          })}
          {visibleDimensions.length === 0 && <p className="empty-line">没有匹配的求职标准</p>}
        </div> : <div className="standards-list" role="table" aria-label="判断字段列表">
          <div className="standards-list-header" role="row"><span role="columnheader">字段</span><span role="columnheader">匹配用途</span><span role="columnheader">证据</span><span className="sr-only" role="columnheader">编辑</span></div>
          {visibleFacts.map(({ item: fact, index }) => {
            const key = `fact:${index}`;
            const selected = editingItem === key;
            return <div className={`standard-row${selected ? " standard-row--selected" : ""}`} role="row" key={`${fact.id || "new"}-${index}`}>
              <button className="standard-name" type="button" role="cell" aria-pressed={selected} onClick={() => setEditingItem(key)}><strong>{fact.name || "未命名字段"}</strong><span>{fact.definition || "尚未填写说明"}</span></button>
              <div className="standard-control" role="cell"><select aria-label={`匹配用途 ${fact.id}`} value={fact.candidate_comparison} onChange={(event) => { const comparison = event.target.value as CandidateComparison; updateFact(index, { candidate_comparison: comparison, qualification_weight: comparison === "none" ? 0 : fact.qualification_weight || 1 }); }}>{Object.entries(comparisonLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></div>
              <label className="compact-toggle" role="cell"><input aria-label={`原文证据 ${fact.id}`} type="checkbox" checked={fact.requires_evidence} onChange={(event) => updateFact(index, { requires_evidence: event.target.checked })} /><span>原文</span></label>
              <button className="icon-button" type="button" aria-label={`编辑 ${fact.name || "未命名字段"}`} title="编辑" onClick={() => setEditingItem(key)}><Pencil size={15} aria-hidden="true" /></button>
            </div>;
          })}
          {visibleFacts.length === 0 && <p className="empty-line">没有匹配的判断字段</p>}
        </div>}

        {(editingDimension || editingFact) && <aside className="standard-drawer" aria-label="标准编辑面板">
          <header className="standard-drawer-header">
            <div><span>{editingDimension ? "求职标准" : "判断字段"}</span><h3>{(editingDimension ?? editingFact)?.name || "未命名"}</h3></div>
            <button className="icon-button" type="button" aria-label="关闭编辑面板" title="关闭" onClick={() => setEditingItem(null)}><X size={17} aria-hidden="true" /></button>
          </header>
          {editingDimension && editingDimensionIndex !== null && <div className="standard-detail" role="region" aria-label={`编辑 ${editingDimension.name || "未命名标准"}`}>
            <label>名称<input aria-label={`维度名称 ${editingDimension.id}`} value={editingDimension.name} onChange={(event) => updateDimension(editingDimensionIndex, { name: event.target.value })} /></label>
            <label>一句话说明<textarea value={editingDimension.definition} onChange={(event) => updateDimension(editingDimensionIndex, { definition: event.target.value })} /></label>
            <div className="drawer-quick-controls"><label>重要程度<WeightSelect id={editingDimension.id} value={editingDimension.weight} onChange={(weight) => updateDimension(editingDimensionIndex, { weight })} /></label><label>最低通过分<input aria-label="最低通过分" inputMode="decimal" value={editingDimension.threshold === null || Number.isNaN(editingDimension.threshold) ? "" : editingDimension.threshold} onChange={(event) => updateDimension(editingDimensionIndex, { threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label></div>
            <label className="drawer-toggle"><input type="checkbox" checked={editingDimension.hard_constraint} onChange={(event) => updateDimension(editingDimensionIndex, { hard_constraint: event.target.checked })} /><span><strong>必须满足</strong><small>未达到时直接阻止推荐</small></span></label>
            <details className="advanced-fields"><summary><SlidersHorizontal size={14} aria-hidden="true" />判断依据与技术设置</summary><div className="advanced-field-grid">
              <label>ID<input value={editingDimension.id} disabled={editingDimensionIndex < baseline.dimensions.length} onChange={(event) => updateDimension(editingDimensionIndex, { id: event.target.value.trim() })} /></label>
              <label>计分方向<select value={editingDimension.scoring_direction} onChange={(event) => updateDimension(editingDimensionIndex, { scoring_direction: event.target.value as ValueDimension["scoring_direction"] })}><option value="higher_is_better">越高越好</option><option value="lower_is_better">越低越好</option></select></label>
              <label>正向关键词<input value={editingDimension.positive_keywords.join(", ")} onChange={(event) => updateDimension(editingDimensionIndex, { positive_keywords: splitList(event.target.value) })} /></label>
              <label>负向关键词<input value={editingDimension.negative_keywords.join(", ")} onChange={(event) => updateDimension(editingDimensionIndex, { negative_keywords: splitList(event.target.value) })} /></label>
              <label>示例<input value={editingDimension.examples.join(", ")} onChange={(event) => updateDimension(editingDimensionIndex, { examples: splitList(event.target.value) })} /></label>
              <label>解释模板<textarea value={editingDimension.explanation_template ?? ""} onChange={(event) => updateDimension(editingDimensionIndex, { explanation_template: event.target.value || null })} /></label>
              <div className="reference-picker"><span>引用事实</span>{draft.facts.filter((fact) => fact.id).map((fact) => <label className="checkbox-field" key={fact.id}><input type="checkbox" checked={editingDimension.fact_ids.includes(fact.id)} onChange={(event) => updateDimension(editingDimensionIndex, { fact_ids: event.target.checked ? [...editingDimension.fact_ids, fact.id] : editingDimension.fact_ids.filter((id) => id !== fact.id) })} />{fact.name || fact.id}</label>)}</div>
              <div className="reference-picker"><span>引用信号</span>{draft.questions.map((question) => <label className="checkbox-field" key={question.id}><input type="checkbox" checked={editingDimension.signal_ids.includes(question.id)} onChange={(event) => updateDimension(editingDimensionIndex, { signal_ids: event.target.checked ? [...editingDimension.signal_ids, question.id] : editingDimension.signal_ids.filter((id) => id !== question.id) })} />{question.question}</label>)}</div>
            </div></details>
          </div>}
          {editingFact && editingFactIndex !== null && <div className="standard-detail" role="region" aria-label={`编辑 ${editingFact.name || "未命名字段"}`}>
            <label>名称<input aria-label={`事实名称 ${editingFact.id}`} value={editingFact.name} onChange={(event) => updateFact(editingFactIndex, { name: event.target.value })} /></label>
            <label>一句话说明<textarea value={editingFact.definition} onChange={(event) => updateFact(editingFactIndex, { definition: event.target.value })} /></label>
            <label>匹配用途<select aria-label={`编辑匹配用途 ${editingFact.id}`} value={editingFact.candidate_comparison} onChange={(event) => { const comparison = event.target.value as CandidateComparison; updateFact(editingFactIndex, { candidate_comparison: comparison, qualification_weight: comparison === "none" ? 0 : editingFact.qualification_weight || 1 }); }}>{Object.entries(comparisonLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            {editingFact.candidate_comparison !== "none" && <label>胜任力权重<input aria-label={`事实权重 ${editingFact.id}`} inputMode="decimal" value={String(editingFact.qualification_weight)} onChange={(event) => updateFact(editingFactIndex, { qualification_weight: Number(event.target.value) })} /></label>}
            <label className="drawer-toggle"><input type="checkbox" checked={editingFact.requires_evidence} onChange={(event) => updateFact(editingFactIndex, { requires_evidence: event.target.checked })} /><span><strong>必须提供原文证据</strong><small>判断结果可以回溯到 JD</small></span></label>
            <details className="advanced-fields"><summary><SlidersHorizontal size={14} aria-hidden="true" />提取规则与技术设置</summary><div className="advanced-field-grid">
              <label>ID<input value={editingFact.id} disabled={editingFactIndex < baseline.facts.length} onChange={(event) => updateFact(editingFactIndex, { id: event.target.value.trim() })} /></label>
              <label>数据类型<select value={editingFact.data_type} onChange={(event) => updateFact(editingFactIndex, { data_type: event.target.value as FactDefinition["data_type"] })}><option value="boolean">布尔</option><option value="number">数字</option><option value="text">文本</option><option value="enum">枚举</option><option value="list">列表</option></select></label>
              <label>提取说明<textarea value={editingFact.extraction_instructions} onChange={(event) => updateFact(editingFactIndex, { extraction_instructions: event.target.value })} /></label>
              <label>缺失处理<select value={editingFact.missing_policy} onChange={(event) => updateFact(editingFactIndex, { missing_policy: event.target.value as FactDefinition["missing_policy"] })}><option value="unknown">标记未知</option><option value="default">使用默认值</option></select></label>
              <label>枚举值（逗号分隔）<input value={editingFact.enum_values.join(", ")} onChange={(event) => updateFact(editingFactIndex, { enum_values: splitList(event.target.value) })} /></label>
              <label>默认值{editingFact.data_type === "boolean" ? <select aria-label={`事实默认值 ${editingFact.id}`} disabled={editingFact.missing_policy !== "default"} value={String(editingFact.default_value ?? false)} onChange={(event) => updateFact(editingFactIndex, { default_value: event.target.value === "true" })}><option value="false">否</option><option value="true">是</option></select> : <input aria-label={`事实默认值 ${editingFact.id}`} disabled={editingFact.missing_policy !== "default"} value={editingFact.default_value === null ? "" : Array.isArray(editingFact.default_value) ? editingFact.default_value.join(", ") : String(editingFact.default_value)} onChange={(event) => updateFact(editingFactIndex, { default_value: parseDefaultInput(editingFact, event.target.value) })} />}</label>
            </div></details>
          </div>}
        </aside>}
        </div>
      </section>

      <details className="editor-section threshold-panel"><summary><span><strong>推荐阈值</strong><small>{recommendationLevels.map((level) => draft.recommendation_thresholds[level]).join(" / ")}</small></span><SlidersHorizontal size={14} aria-hidden="true" /></summary><div className="field-grid">{recommendationLevels.map((level) => <label key={level}>{levelLabels[level]}<input aria-label={`${levelLabels[level]}阈值`} inputMode="decimal" value={Number.isNaN(draft.recommendation_thresholds[level]) ? "" : String(draft.recommendation_thresholds[level])} onChange={(event) => setDraft({ ...draft, recommendation_thresholds: { ...draft.recommendation_thresholds, [level]: event.target.value === "" ? Number.NaN : Number(event.target.value) } })} /></label>)}</div></details>

      {error && <p className="form-message form-message--error" role="alert">{error}</p>}
      {status && <p className="form-message" role="status">{status}</p>}
      {staleJobCount > 0 && <div className="recompute-bar"><span>{staleJobCount} 个岗位需要重新计算。</span><button type="button" disabled={!selectedAnalysisId || !selectedJobIsStale || busy} onClick={recompute}><RotateCcw size={14} aria-hidden="true" />重新计算当前岗位</button>{!selectedAnalysisId && <small>请先在“岗位”中选择一项。</small>}{selectedAnalysisId && !selectedJobIsStale && <small>当前岗位已是最新结果。</small>}</div>}
    </div>
  );
}
