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
- **MCP App:** a compact side panel for reviewing jobs, editing the rubric, and confirming candidate evidence. Python remains the only owner of persisted state and decisions.

There is no standalone cloud account or application service.

## Installation

Python 3.12 or 3.13 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
npm ci
npm run ui:check
npm run ui:build
python -m pytest -q
```

`npm run ui:build` writes the single self-contained MCP App document to
`src/ai_dream_job_engine/ui/workspace-v1.html`. Rebuild it before packaging
the Python distribution.

Start the local MCP server with:

```bash
./scripts/start-mcp.sh
```

The helper script requires [`uv`](https://docs.astral.sh/uv/). A clean environment can instead use the installed Python module directly, without `uv`:

```bash
python -m ai_dream_job_engine.mcp_server
```

### Local Codex plugin installation

This repository is a local Codex marketplace and contains one plugin. From the
repository root, build the UI, register the marketplace, and install the
unambiguous local selector:

```bash
npm ci
npm run ui:build
./scripts/build-codex-plugin.sh
codex plugin marketplace add "$PWD"
codex plugin add ai-dream-job-engine@ai-dream-job-engine-local
```

The two `codex plugin` commands intentionally update the current user's Codex
marketplace and plugin configuration; the repository never runs them during
build, test, or server startup. The bundle step writes only runtime files to
ignored `dist/codex-plugin`; it excludes the development virtual environment,
Node modules, tests, and local data. After installation, fully quit and reopen Codex
Desktop, open this repository as the project, and start a new conversation so
the skill and MCP server are loaded. Ask the Agent to open the AI Dream Job
Engine workspace. Re-run `codex plugin add
ai-dream-job-engine@ai-dream-job-engine-local` after rebuilding a version that
you want Codex to install.

To verify the server independently of Codex, run `./scripts/start-mcp.sh` with
an MCP Inspector or client. The render entrypoint is `open_job_workspace`; its
resource is `ui://ai-dream-job-engine/workspace-v1.html` with media type
`text/html;profile=mcp-app`. Existing non-UI tools remain callable when a host
cannot render MCP Apps.

The startup script uses `uv run --locked`, so it refuses to resolve a dependency
set different from the tracked `uv.lock`. Check and reproduce the lock plus MCP
stdio transcript with:

```bash
uv lock --check
python scripts/verify_mcp_stdio.py \
  --output output/playwright/mcp-protocol-verification.json
```

To verify the built wheel rather than an editable checkout, use a fresh
temporary environment and point the same stdio verifier at its interpreter:

```bash
WHEEL_SMOKE_ROOT="$(mktemp -d)"
mkdir -p "$WHEEL_SMOKE_ROOT/dist"
python -m pip wheel . --no-deps --no-build-isolation \
  --wheel-dir "$WHEEL_SMOKE_ROOT/dist"
python -m venv "$WHEEL_SMOKE_ROOT/venv"
"$WHEEL_SMOKE_ROOT/venv/bin/python" -m pip install \
  "$WHEEL_SMOKE_ROOT"/dist/ai_dream_job_engine-*.whl
python scripts/verify_mcp_stdio.py \
  --python "$WHEEL_SMOKE_ROOT/venv/bin/python"
```

## Model configuration

Model credentials stay in environment variables and must not be committed. Set `AI_DREAM_JOB_FACT_PROVIDER` to `openai` or `openrouter`, set `AI_DREAM_JOB_FACT_MODEL`, and provide the corresponding `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. OpenRouter uses its fixed API base. JEV decision signals use `TYPESAFE_API_KEY` and the adapter's pinned JEV endpoint/model. Provider adapters are not created when the evaluation module is imported.

External models receive public job-description text only by default. Selected candidate claims require a separate, explicit disclosure and consent step.

## Memory consent

Agent memory is never silently imported. The user previews proposed claims, chooses the scope, and explicitly accepts, edits, or rejects each candidate. Imported claims retain provenance and can be reviewed locally.

## One-JD quickstart

1. Create or review the local candidate profile and rubric.
2. Submit one public job description through the host Agent's `analyze_job` tool.
3. Ask the Agent to open the job workspace, then review the recommendation, leading gap, evidence references, and disclosure record.
4. Choose `Keep`, `Ignore`, or, only when the server allows it, `Ready to apply`.

`ready_to_apply` is a local review state. It never opens a browser, signs in,
fills a form, or submits an application. Browser login handoff, automatic job
search, Gmail synchronization, form filling, and application submission remain
deferred and require separate consent-preserving designs.

### Manual 30-second acceptance

The automated harness measures interaction latency, not human comprehension.
To perform the product acceptance check, start with one already analyzed,
synthetic or explicitly approved JD visible in the default `岗位` view:

1. Start a stopwatch when the populated workspace becomes visible.
2. Without assistance, state the recommendation and the leading unresolved gap.
3. Open that gap's evidence and identify the JD and candidate evidence shown.
4. Choose `保留` or `忽略`, then stop the timer.

Record the elapsed time and whether any help was needed. The 30-second target is
met only when a person completes all four steps in at most 30 seconds without
help locating a control.

For a network-free harness smoke test, run:

```bash
python scripts/run_evaluation.py eval/synthetic_cases.jsonl /tmp/ai-dream-job-eval.json --offline
```

The offline mode is a deterministic fake for the invented fixture. It is not a model-quality result.

## Evidence boundaries

The engine distinguishes public job evidence, user-confirmed candidate claims, and restricted personal data. Evidence references support review but do not turn a human label into ground truth. Missing or ambiguous evidence can force review. A recommendation is decision support, not hiring advice or a verified statement about an employer.

## Privacy

Data classes are `public_job_data`, `selected_candidate_claims`, and `restricted_personal_data`. Outbound disclosures identify provider, purpose, fields, data class, and confirmation state. Public JD text is the only default external-model payload. Restricted personal data is not sent by the implemented provider flow.

Local profiles, rubrics, and analysis records use the user data directory by default. On macOS the database is `~/Library/Application Support/ai-dream-job-engine/ai-dream-job-engine.sqlite3`; elsewhere it uses `$XDG_DATA_HOME/ai-dream-job-engine/ai-dream-job-engine.sqlite3`, or `~/.local/share/ai-dream-job-engine/ai-dream-job-engine.sqlite3` when `XDG_DATA_HOME` is unset. Set `AI_DREAM_JOB_DATA_PATH` to override that location. Private evaluation cases belong under `eval/private/`, generated private reports under `eval/reports/`, and local MCP state under `local-mcp-data/`; these paths are ignored by Git. There is no telemetry.

## Evaluation method

The protocol has two phases. Before analysis, the human records recommendation and gap labels; the harness rejects pre-populated system results and does not mutate source cases. The first run records system recommendation, preference score, qualification score, and a deterministic job ranking. The user then reviews that ranking, adds `human_top_five_usefulness` for ranked jobs and `human_review_time_seconds`, and runs a new report from cases that still omit pre-populated system fields. Those post-review values are feedback on system output, not pre-system ground truth.

Automated fields are recommendation **agreement**, blocker precision with `null` when no blocker was predicted, lossless gap-label counts with missing-side sentinels, `gap_label_agreement`, unsupported-source counts, rerun stability, and `evaluation_runtime_seconds`. The report retains every attempted case in input order. It separately records `rerun_attempted_count`, `rerun_completed_count`, `rerun_failed_count`, and rerun failure IDs. A rerun exception makes stability `null`, sets `run_complete=false`, and causes a nonzero CLI exit while preserving the first-pass report. A completed but changed rerun is nondeterministic, not failed, and lowers stability.

Top-five jobs are the first `min(5, completed)` after sorting by recommendation tier (`priority_apply`, `apply_after_review`, `hold`, `skip`), then descending preference score, descending qualification score, and stable case ID. `top_five_usefulness` is populated only when every selected job has explicit post-review usefulness; `review_time_seconds` likewise comes only from explicit human review input. Partial reports are written atomically before the CLI exits nonzero.

The unsupported-source audit counts known facts without evidence; available signals missing provider, model, question ID, or question version; gaps without evidence unless `missing_evidence` is explicit; and contributions without source evidence or explicit audit support. Signal probabilities do not require JD quotes. Candidate raw claims are intentionally absent from the privacy-safe `AnalysisRecord`, so this audit cannot inspect their raw source content.

These are descriptive review signals, not statistical accuracy estimates. The committed cases are synthetic and only exercise the offline path. See [eval/README.md](eval/README.md) for the private 20-JD protocol. That real evaluation has not been completed and no real JD dataset is included.

## Limitations

The current implementation analyzes user-supplied job text with a local core and Agent-native plugin. Model outputs can be incomplete or wrong, evidence references still require human review, and the synthetic fixture cannot establish real-world usefulness. Evaluation runtime is machine execution time; review time is accepted only from explicit human post-review input. Neither is a controlled usability study.

Not implemented: job search, Gmail sync, application generation or auto-apply, application submission, browser login handoff, cloud accounts, paid tiers, telemetry, a hosted service, or website, desktop, or browser-extension UI.

## Deferred roadmap

Search, inbox integrations, richer interfaces, application workflows, and hosted services are deferred. Any future implementation must preserve explicit consent, local/private-data boundaries, quiet `no_change` behavior, and human control. Deferred items are not commitments.

## License

MIT. See [LICENSE](LICENSE).
