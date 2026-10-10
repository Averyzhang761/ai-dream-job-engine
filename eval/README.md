# Private evaluation protocol

The repository ships only an invented synthetic fixture for code-path checks. Real job descriptions, candidate details, and private evaluation reports must remain local and uncommitted.

## 20-JD protocol

1. Place exactly 20 private cases in `eval/private/`. Before any system run, record the human recommendation and human gap labels for every case.
2. Validate that every case has those human fields and no system result. Stop if a system result is already present or a human label changed after analysis began.
3. Run the configured `JobAnalysisService` through the dependency-injected evaluation CLI. The first pass stores `system_recommendation`, `system_preference_score`, and `system_qualification_score` on report copies, then ranks completed jobs by recommendation tier, descending preference score, descending qualification score, and stable case ID. The required stability rerun is counted separately.
4. Review the generated ranking and disagreements. Record whether the rubric, evidence, provider output, or human judgment needs revision; do not relabel a case merely to improve agreement. Only now may the human add `human_top_five_usefulness` for ranked jobs and `human_review_time_seconds`. Generate the next report from cases without pre-populated system fields. These values are post-system feedback, not pre-system ground truth.
5. Write the report under `eval/reports/`. A first-run or rerun failure makes `run_complete=false`; the atomic partial report preserves successful first-pass metrics, and the CLI exits nonzero.

## Interpretation

Human labels are prior judgments, not ground truth. `recommendation_agreement` must never be renamed or described as accuracy and is `null` when no case completes. `blocker_precision` is `null` when the system predicts no blockers. Gap counts use sets: exact matches are diagonal, false negatives use `__missing_system__`, and false positives use `__missing_human__`.

`gap_label_agreement` is overall set overlap, not usefulness or quality. `evaluation_runtime_seconds` is service execution time, not review time. `top_five_usefulness` is the mean explicit human rating over exactly the first `min(5, completed)` ranked jobs and is `null` if any selected job lacks a rating. `review_time_seconds` is total explicit human review time and is `null` unless every completed case supplies it.

Rerun attempts, completions, failures, and failure case IDs are separate. Any rerun exception makes `rerun_stability` null and the overall run incomplete. A completed changed rerun is nondeterministic and lowers stability, but is not a rerun failure.

The unsupported-source breakdown counts: known extracted facts with no evidence; available decision signals missing provider, model, question ID, or question version; decision gaps with no evidence unless `missing_evidence` is explicit; and decision contributions with no evidence. An available signal probability need not quote the JD. Candidate raw claims are intentionally absent from the privacy-safe `AnalysisRecord`; the audit therefore cannot evaluate their raw source content. The total is the sum of the four class counts. None of these fields establishes real-world quality or statistical accuracy.

Twenty cases are a private review set, not a statistically representative benchmark. Do not publish raw cases, copied JD text, profile data, credentials, or reports. This repository implements the protocol only; the private 20-JD milestone is pending.

## JSONL fields

Each UTF-8 JSONL line uses `EvaluationCase`: `id`, `job_description`, `human_recommendation`, `human_gap_labels`, and optional private notes. Leave `system_recommendation` and `system_gap_labels` absent. Post-review copies may add `human_top_five_usefulness` and `human_review_time_seconds`. Invalid and duplicate-line errors identify line numbers and never echo IDs or job text.
