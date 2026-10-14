import { X } from "lucide-react";
import { useState } from "react";

import { CRITERION_CATEGORIES, requirementLabel } from "../criteria";
import type { CriterionCategory, CustomPreferenceProposal, RequirementLevel } from "../contracts";

interface CustomPreferenceDialogProps {
  onClose: () => void;
  onAgentHandoff: (statement: string) => Promise<void>;
  onManualStage: (proposal: CustomPreferenceProposal) => Promise<void> | void;
}

const levels: RequirementLevel[] = ["ignored", "normal", "strong", "required"];

export function CustomPreferenceDialog({ onClose, onAgentHandoff, onManualStage }: CustomPreferenceDialogProps) {
  const [statement, setStatement] = useState("");
  const [manual, setManual] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState("");
  const [topicId, setTopicId] = useState("");
  const [category, setCategory] = useState<CriterionCategory>("role_content");
  const [name, setName] = useState("");
  const [desiredValue, setDesiredValue] = useState("");
  const [level, setLevel] = useState<RequirementLevel>("normal");
  const [guidance, setGuidance] = useState("");
  const [positiveKeywords, setPositiveKeywords] = useState("");
  const [negativeKeywords, setNegativeKeywords] = useState("");
  const [relatedTopics, setRelatedTopics] = useState("");

  const structure = async () => {
    if (!statement.trim() || waiting) return;
    setWaiting(true);
    setError("");
    try {
      await onAgentHandoff(statement.trim());
    } catch {
      setManual(true);
      setError("未能连接 Agent，可以手动整理。");
    } finally {
      setWaiting(false);
    }
  };

  const complete = Boolean(topicId.trim() && name.trim() && desiredValue.trim() && guidance.trim() && (positiveKeywords.trim() || negativeKeywords.trim()));
  const submitManual = async () => {
    if (!complete) return;
    await onManualStage({
      statement: statement.trim(), topic_id: topicId.trim(), category, name: name.trim(),
      desired_value: desiredValue.trim(), requirement_level: level,
      evidence_guidance: guidance.trim(),
      positive_keywords: splitTerms(positiveKeywords),
      negative_keywords: splitTerms(negativeKeywords),
      fact_ids: [], signal_ids: [],
      related_topic_ids: relatedTopics.split(/[,，]/).map((item) => item.trim()).filter(Boolean),
    });
  };

  return <div className="preference-modal" role="dialog" aria-modal="true" aria-labelledby="preference-dialog-title">
    <div className="preference-dialog">
      <header><div><h2 id="preference-dialog-title">添加偏好</h2><p>先用自然语言表达，再确认结构化结果。</p></div><button className="icon-button" type="button" aria-label="关闭添加偏好" onClick={onClose}><X size={17} aria-hidden="true" /></button></header>
      <label className="preference-statement">用一句话描述偏好<textarea aria-label="用一句话描述偏好" value={statement} onChange={(event) => setStatement(event.target.value)} placeholder="例如：我希望经常与客户共创，而不是只做后台平台。" /></label>
      {!manual && <div className="preference-actions"><button className="primary-button" type="button" disabled={!statement.trim() || waiting} onClick={structure}>{waiting ? "等待 Agent…" : "整理偏好"}</button><button className="text-button" type="button" onClick={() => setManual(true)}>手动填写</button></div>}
      {error && <p className="form-message form-message--error" role="alert">{error}</p>}
      {manual && <div className="preference-form">
        <label>偏好名称<input aria-label="偏好名称" value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>主题标识<input aria-label="主题标识" value={topicId} onChange={(event) => setTopicId(event.target.value)} placeholder="customer_contact" /></label>
        <label>所属主题<select aria-label="所属主题" value={category} onChange={(event) => setCategory(event.target.value as CriterionCategory)}>{CRITERION_CATEGORIES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        <label>我的选择<input aria-label="我的选择" value={desiredValue} onChange={(event) => setDesiredValue(event.target.value)} /></label>
        <label>要求程度<select aria-label="要求程度" value={level} onChange={(event) => setLevel(event.target.value as RequirementLevel)}>{levels.map((item) => <option key={item} value={item}>{requirementLabel(item)}</option>)}</select></label>
        <label>系统如何找证据<textarea aria-label="系统如何找证据" value={guidance} onChange={(event) => setGuidance(event.target.value)} /></label>
        <label>符合时常见表述<input aria-label="符合时常见表述" value={positiveKeywords} onChange={(event) => setPositiveKeywords(event.target.value)} placeholder="客户访谈，customer discovery" /></label>
        <label>不符合时常见表述<input aria-label="不符合时常见表述" value={negativeKeywords} onChange={(event) => setNegativeKeywords(event.target.value)} placeholder="纯内部平台，no customer contact" /></label>
        <label>相关主题（可选）<input aria-label="相关主题" value={relatedTopics} onChange={(event) => setRelatedTopics(event.target.value)} /></label>
        <button className="primary-button" type="button" disabled={!complete} onClick={submitManual}>暂存并检查冲突</button>
      </div>}
    </div>
  </div>;
}

function splitTerms(value: string): string[] {
  return value.split(/[,\uff0c]/).map((item) => item.trim()).filter(Boolean);
}
