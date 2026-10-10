# Private evaluation protocol

The repository ships only an invented synthetic fixture for code-path checks. Real job descriptions, candidate details, and private evaluation reports must remain local and uncommitted.

## 20-JD protocol

1. Place exactly 20 private cases in `eval/private/`. Before any system run, record the human recommendation and human gap labels for every case.
2. Validate that every case has those human fields and no system result. Stop if a system result is already present or a human label changed after analysis began.
3. Run the configured `JobAnalysisService` through the dependency-injected evaluation CLI and write the report under `eval/reports/`. A first-run failure remains in place with null system fields; the atomic partial report has `run_complete=false`, and the CLI exits nonzero.
4. Review each disagreement. Record whether the rubric, evidence, provider output, or human judgment needs revision; do not relabel a case merely to improve agreement. Only now may the human add `human_top_five_usefulness` and `human_review_time_seconds`.
5. Report descriptive agreement, blocker precision, lossless gap-label counts, unsupported claims, rerun stability, top-five gap-label agreement, evaluation runtime, and any explicitly supplied human review metrics with the limitations below.

## Interpretation

Human labels are prior judgments, not ground truth. `recommendation_agreement` must never be renamed or described as accuracy and is `null` when no case completes. `blocker_precision` is `null` when the system predicts no blockers. Gap counts use sets: exact matches are diagonal, false negatives use `__missing_system__`, and false positives use `__missing_human__`.

`top_five_gap_label_agreement` is automated label overlap, not usefulness or quality. `evaluation_runtime_seconds` is service execution time, not review time. `top_five_usefulness` is the mean explicit human usefulness rating, and `review_time_seconds` is the total explicit human review time; each aggregate is populated only when every completed case supplies its corresponding post-review field, otherwise it is `null`. `rerun_stability` is also `null` when no first run completes. None of these fields establishes real-world quality or statistical accuracy.

Twenty cases are a private review set, not a statistically representative benchmark. Do not publish raw cases, copied JD text, profile data, credentials, or reports. This repository implements the protocol only; the private 20-JD milestone is pending.

## JSONL fields

Each UTF-8 JSONL line uses `EvaluationCase`: `id`, `job_description`, `human_recommendation`, `human_gap_labels`, and optional private notes. Leave `system_recommendation` and `system_gap_labels` absent. Post-review copies may add `human_top_five_usefulness` and `human_review_time_seconds`. Invalid and duplicate-line errors identify line numbers and never echo IDs or job text.
