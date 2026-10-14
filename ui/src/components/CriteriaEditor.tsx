import { ArrowLeft, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { CRITERION_CATEGORIES, requirementLabel } from "../criteria";
import type { CriterionCategory, PendingPreference, RequirementLevel, ValueDimension } from "../contracts";

export type CriteriaEditorMode =
  | { kind: "category"; category: CriterionCategory }
  | { kind: "all" };

interface CriteriaEditorProps {
  dimensions: ValueDimension[];
  pendingPreferences: PendingPreference[];
  mode: CriteriaEditorMode;
  onBack: () => void;
  onReviewPending: (id: string) => void;
  onRequirementChange: (id: string, level: RequirementLevel) => void;
}

const levels: RequirementLevel[] = ["ignored", "normal", "strong", "required"];

export function CriteriaEditor({ dimensions, pendingPreferences, mode, onBack, onReviewPending, onRequirementChange }: CriteriaEditorProps) {
  const [query, setQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<CriterionCategory | "all">("all");
  const [levelFilter, setLevelFilter] = useState<RequirementLevel | "all">("all");
  const [originFilter, setOriginFilter] = useState<"all" | "custom">("all");
  const [attentionFilter, setAttentionFilter] = useState<"all" | "pending">("all");
  const category = mode.kind === "category" ? CRITERION_CATEGORIES.find((item) => item.id === mode.category) : null;

  const visible = useMemo(() => dimensions.filter((item) => {
    if (mode.kind === "category" && item.category !== mode.category) return false;
    if (mode.kind === "all" && categoryFilter !== "all" && item.category !== categoryFilter) return false;
    if (levelFilter !== "all" && item.requirement_level !== levelFilter) return false;
    if (originFilter === "custom" && item.origin !== "custom") return false;
    const text = `${item.name} ${item.desired_value} ${item.definition}`.toLocaleLowerCase();
    return !query.trim() || text.includes(query.trim().toLocaleLowerCase());
  }), [categoryFilter, dimensions, levelFilter, mode, originFilter, query]);
  const visiblePending = useMemo(() => pendingPreferences.filter((item) => {
    if (attentionFilter !== "pending") return false;
    if (categoryFilter !== "all" && item.category !== categoryFilter) return false;
    const text = `${item.name ?? ""} ${item.desired_value ?? ""} ${item.statement}`.toLocaleLowerCase();
    return !query.trim() || text.includes(query.trim().toLocaleLowerCase());
  }), [attentionFilter, categoryFilter, pendingPreferences, query]);
  const visibleDimensions = attentionFilter === "pending" ? [] : visible;

  const title = category?.label ?? "全部筛选标准";
  return (
    <section className="criteria-workspace" aria-label={mode.kind === "all" ? "全部筛选标准" : `编辑${title}`}>
      <header className="criteria-editor-header">
        <button className="icon-button" type="button" aria-label="返回求职画像" onClick={onBack}><ArrowLeft size={17} aria-hidden="true" /></button>
        <div><h2>{title}</h2><p>{attentionFilter === "pending" ? `${visiblePending.length} 项待确认` : `${visibleDimensions.length} 条标准`}</p></div>
      </header>

      {mode.kind === "all" && (
        <div className="criteria-filters">
          <label className="criteria-search"><Search size={14} aria-hidden="true" /><span className="sr-only">搜索标准</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
          <label><span className="sr-only">按主题筛选</span><select aria-label="按主题筛选" value={categoryFilter} onChange={(event) => setCategoryFilter(event.target.value as CriterionCategory | "all")}><option value="all">全部主题</option>{CRITERION_CATEGORIES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          <label><span className="sr-only">按要求程度筛选</span><select aria-label="按要求程度筛选" value={levelFilter} onChange={(event) => setLevelFilter(event.target.value as RequirementLevel | "all")}><option value="all">全部程度</option>{levels.map((level) => <option key={level} value={level}>{requirementLabel(level)}</option>)}</select></label>
          <label><span className="sr-only">按来源筛选</span><select aria-label="按来源筛选" value={originFilter} onChange={(event) => setOriginFilter(event.target.value as "all" | "custom")}><option value="all">全部来源</option><option value="custom">仅自定义</option></select></label>
          <label><span className="sr-only">按处理状态筛选</span><select aria-label="按处理状态筛选" value={attentionFilter} onChange={(event) => setAttentionFilter(event.target.value as "all" | "pending")}><option value="all">已生效标准</option><option value="pending">待确认</option></select></label>
        </div>
      )}

      <div className="criteria-list" role="table" aria-label={`${title}标准`}>
        <div className="criteria-row criteria-row--header" role="row"><span role="columnheader">标准</span><span role="columnheader">我的选择</span><span role="columnheader">要求程度</span></div>
        {visibleDimensions.map((item) => <div className="criteria-row" role="row" key={item.id}>
          <div className="criterion-name" role="cell"><strong>{item.name}</strong>{item.origin === "custom" && <span className="origin-badge">自定义</span>}</div>
          <div className="criterion-choice" role="cell">{item.desired_value}</div>
          <div className="criterion-requirement" role="cell"><select aria-label={`要求程度 ${item.name}`} value={item.requirement_level} onChange={(event) => onRequirementChange(item.id, event.target.value as RequirementLevel)}>{levels.map((level) => <option value={level} key={level}>{requirementLabel(level)}</option>)}</select></div>
          <details className="criterion-judgment"><summary>系统如何判断</summary><div><p>{item.explanation_template || item.definition}</p><p>信息不足时标记为未知，不自动猜测。</p><p>{item.requirement_level === "required" ? "未满足时会阻止推荐。" : "用于调整想不想要这份工作的评分。"}</p></div></details>
        </div>)}
        {visiblePending.map((item) => <div className="criteria-row" role="row" key={item.id}>
          <div className="criterion-name" role="cell"><strong>{item.name || "尚未命名"}</strong><span className="pending-badge">待确认</span></div>
          <div className="criterion-choice" role="cell">{item.desired_value || item.statement}</div>
          <div className="criterion-requirement" role="cell"><button className="secondary-button" type="button" onClick={() => onReviewPending(item.id)}>处理</button></div>
        </div>)}
      </div>
    </section>
  );
}
