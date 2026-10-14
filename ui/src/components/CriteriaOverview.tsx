import { ChevronRight, Plus } from "lucide-react";

import { CRITERION_CATEGORIES, summarizeCategory } from "../criteria";
import type { CriterionCategory, ValueDimension } from "../contracts";

interface CriteriaOverviewProps {
  dimensions: ValueDimension[];
  pendingCount: number;
  onEditCategory: (category: CriterionCategory) => void;
  onViewAll: () => void;
  onAddPreference: () => void;
}

export function CriteriaOverview({ dimensions, pendingCount, onEditCategory, onViewAll, onAddPreference }: CriteriaOverviewProps) {
  return (
    <section className="criteria-overview" aria-label="求职画像概览">
      <header className="criteria-heading">
        <div>
          <h2>我的求职画像</h2>
          <p>先看四个主题，需要时再进入具体标准。</p>
        </div>
        <button className="primary-button" type="button" onClick={onAddPreference}>
          <Plus size={15} aria-hidden="true" />添加偏好
        </button>
      </header>

      {pendingCount > 0 && <button className="pending-link" type="button" onClick={onViewAll}>{pendingCount} 项需要处理</button>}

      <div className="criteria-category-list">
        {CRITERION_CATEGORIES.map((category) => {
          const items = dimensions.filter((item) => item.category === category.id);
          const summary = summarizeCategory(items);
          return (
            <article className="criteria-category" key={category.id}>
              <div className="criteria-category-copy">
                <h3>{category.label}</h3>
                <p>{category.description}</p>
                <strong aria-label={`${category.label}：${summary}`}>{summary}</strong>
              </div>
              <button className="secondary-button" type="button" aria-label={`编辑 ${category.label}`} onClick={() => onEditCategory(category.id)}>
                编辑 <ChevronRight size={15} aria-hidden="true" />
              </button>
            </article>
          );
        })}
      </div>

      <button className="text-button criteria-view-all" type="button" onClick={onViewAll}>
        查看全部 {dimensions.length} 条标准 <ChevronRight size={15} aria-hidden="true" />
      </button>
    </section>
  );
}
