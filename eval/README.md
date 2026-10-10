# Private evaluation protocol

The repository ships only an invented synthetic fixture for code-path checks. Real job descriptions, candidate details, and private evaluation reports must remain local and uncommitted.

## 20-JD protocol

1. Place exactly 20 private cases in `eval/private/`. Before any system run, record the human recommendation and human gap labels for every case.
2. Validate that every case has those human fields and no system result. Stop if a system result is already present or a human label changed after analysis began.
3. Run the configured `JobAnalysisService` through the dependency-injected evaluation CLI and write the report under `eval/reports/`.
4. Review each disagreement. Record whether the rubric, evidence, provider output, or human judgment needs revision; do not relabel a case merely to improve agreement.
5. Report descriptive agreement, blocker precision, paired gap-label confusion counts, unsupported claims, rerun stability, top-five label agreement, and review-run time with the limitations below.

## Interpretation

Human labels are prior judgments, not ground truth. `recommendation_agreement` must never be renamed or described as accuracy. `blocker_precision` is `null` when the system predicts no blockers. Gap confusion counts cover paired labels after exact matches; unmatched labels remain visible on each case. Top-five usefulness is the precision of up to five system gap labels against the prior human labels, not a user-satisfaction score. Review time is local harness elapsed time, not a controlled human-time study.

Twenty cases are a private review set, not a statistically representative benchmark. Do not publish raw cases, copied JD text, profile data, credentials, or reports. This repository implements the protocol only; the private 20-JD milestone is pending.

## JSONL fields

Each UTF-8 JSONL line uses `EvaluationCase`: `id`, `job_description`, `human_recommendation`, `human_gap_labels`, and optional private notes. Leave `system_recommendation` and `system_gap_labels` absent. Invalid-line errors identify only the line number and never echo job text.
