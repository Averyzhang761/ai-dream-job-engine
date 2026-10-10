# AI Dream Job Engine

AI Dream Job Engine is a local-first decision aid for evaluating one public job description against a user-controlled rubric and confirmed candidate claims. It produces an evidence-backed recommendation for human review; it does not determine whether a person is qualified or predict hiring outcomes.

## Product thesis

Job-search tools should help a person make a deliberate decision without taking control of the search. The core keeps profile claims, rubrics, analyses, and audit records local. A host Agent can call the same bounded operations while the user retains approval over memory imports and external-model disclosures.

## Differentiator

Recommendations are traceable to versioned rubric inputs and evidence references. Unknown information remains unknown, human judgments remain distinct from system output, and provider failures are recorded without persisting raw job text or secret-bearing errors.

## Architecture

- **Local core:** Pydantic domain contracts, deterministic decision logic, and a local SQLite store.
- **Provider adapters:** OpenAI-compatible extraction and judgment adapters, plus JEV support.
- **Agent-native plugin:** a Codex plugin and MCP server expose the core's bounded workflows. The host Agent owns schedules and notifications; a `no_change` result is quiet.

There is no standalone cloud account or application service.

## Installation

Python 3.12 or 3.13 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Start the local MCP server with:

```bash
./scripts/start-mcp.sh
```

The helper script requires [`uv`](https://docs.astral.sh/uv/). A clean environment can instead use the installed Python module directly, without `uv`:

```bash
python -m ai_dream_job_engine.mcp_server
```

## Model configuration

Model credentials stay in environment variables and must not be committed. Set `AI_DREAM_JOB_FACT_PROVIDER` to `openai` or `openrouter`, set `AI_DREAM_JOB_FACT_MODEL`, and provide the corresponding `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. OpenRouter uses its fixed API base. JEV decision signals use `TYPESAFE_API_KEY` and the adapter's pinned JEV endpoint/model. Provider adapters are not created when the evaluation module is imported.

External models receive public job-description text only by default. Selected candidate claims require a separate, explicit disclosure and consent step.

## Memory consent

Agent memory is never silently imported. The user previews proposed claims, chooses the scope, and explicitly accepts, edits, or rejects each candidate. Imported claims retain provenance and can be reviewed locally.

## One-JD quickstart

1. Create or review the local candidate profile and rubric.
2. Submit one public job description through the host Agent's `analyze_job` tool.
3. Review the recommendation, gaps, evidence references, and disclosure record before acting.

For a network-free harness smoke test, run:

```bash
python scripts/run_evaluation.py eval/synthetic_cases.jsonl /tmp/ai-dream-job-eval.json --offline
```

The offline mode is a deterministic fake for the invented fixture. It is not a model-quality result.

## Evidence boundaries

The engine distinguishes public job evidence, user-confirmed candidate claims, and restricted personal data. Evidence references support review but do not turn a human label into ground truth. Missing or ambiguous evidence can force review. A recommendation is decision support, not hiring advice or a verified statement about an employer.

## Privacy

Data classes are `public_job_data`, `selected_candidate_claims`, and `restricted_personal_data`. Outbound disclosures identify provider, purpose, fields, data class, and confirmation state. Public JD text is the only default external-model payload. Restricted personal data is not sent by the implemented provider flow.

Local profiles, rubrics, and analysis records default to `.data/ai-dream-job-engine.sqlite3` in the project root. Set `AI_DREAM_JOB_DATA_PATH` to override that location. Private evaluation cases belong under `eval/private/`, generated private reports under `eval/reports/`, and local MCP state under `local-mcp-data/`; these paths are ignored by Git. There is no telemetry.

## Evaluation method

The protocol has two phases. Before analysis, the human records recommendation and gap labels; the harness rejects pre-populated system results and does not mutate source cases. After analysis, a human may explicitly add `human_top_five_usefulness` and `human_review_time_seconds` while reviewing the output.

Automated fields are recommendation **agreement**, blocker precision with `null` when no blocker was predicted, lossless gap-label counts with missing-side sentinels, unsupported-claim count, rerun stability, `top_five_gap_label_agreement`, and `evaluation_runtime_seconds`. The report retains every attempted case in input order and marks partial runs with counts and `run_complete=false`. `top_five_usefulness` and `review_time_seconds` remain `null` unless every completed case contains the corresponding explicit human post-review value. A partial report is written atomically, but the CLI exits nonzero.

These are descriptive review signals, not statistical accuracy estimates. The committed cases are synthetic and only exercise the offline path. See [eval/README.md](eval/README.md) for the private 20-JD protocol. That real evaluation has not been completed and no real JD dataset is included.

## Limitations

The current implementation analyzes user-supplied job text with a local core and Agent-native plugin. Model outputs can be incomplete or wrong, evidence references still require human review, and the synthetic fixture cannot establish real-world usefulness. Evaluation runtime is machine execution time; review time is accepted only from explicit human post-review input. Neither is a controlled usability study.

Not implemented: job search, Gmail sync, application generation or auto-apply, application submission, cloud accounts, paid tiers, telemetry, a hosted service, or website, desktop, browser, or MCP App UI.

## Deferred roadmap

Search, inbox integrations, richer interfaces, application workflows, and hosted services are deferred. Any future implementation must preserve explicit consent, local/private-data boundaries, quiet `no_change` behavior, and human control. Deferred items are not commitments.

## License

MIT. See [LICENSE](LICENSE).
