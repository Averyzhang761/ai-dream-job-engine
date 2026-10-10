# AI Dream Job Engine Core MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Codex plugin that creates a user-confirmed candidate profile, applies a configurable AI-job rubric to one JD, and returns a deterministic, evidence-backed recommendation that separates preference, capability, and evidence gaps.

**Architecture:** A Python package owns domain contracts, SQLite state, provider adapters, and a pure decision engine. The MCP server exposes a small tool surface; a Codex Skill supplies the conversational workflow and consent rules. OpenAI/OpenRouter extract facts, JEV supplies probabilistic decision signals, and only deterministic Python rules produce the recommendation.

**Tech Stack:** Python 3.12, Pydantic 2, SQLite, OpenAI Python SDK, HTTPX, MCP Python SDK v2, pytest, Codex plugin manifest and Skill.

**Spec:** `docs/superpowers/specs/2026-10-10-ai-dream-job-engine-design.md`

## Global Constraints

- Work only in `ai-dream-job-engine`; do not modify `ai-job-fit-analyzer-submission`.
- Do not copy the old repository's Git history, real JD dataset, personal job library, credentials, logs, cache, or virtual environment.
- Personal state is local SQLite; API keys come only from environment variables or host secret storage.
- Remote providers receive public JD content by default, never the full Candidate Profile. Full resume transmission requires a separate disclosure and explicit user confirmation.
- JEV receives JD content and configured decision questions only; it never receives resume, Agent-memory, email, or Candidate Profile data.
- No telemetry. Logs redact restricted personal data and credentials.
- Agent memory import is off by default. The MCP server receives only user-authorized candidate claims supplied by the host Agent; it never scans host memory.
- Scheduling belongs to the host Agent. The plugin must not create its own scheduler, heartbeat process, or notification center.
- Future automation-facing workflows must use `no_change`, `updated`, `needs_review`, or `failed`; `no_change` is quiet by default and consequential actions still require user confirmation.
- `agent_suggested` claims cannot influence decisions until the user confirms them.
- Resume absence means `evidence_gap`, never `capability_gap`.
- Models extract facts or return decision signals; deterministic Python code owns weights, blockers, thresholds, and recommendations.
- First milestone analyzes one supplied JD at a time. Job discovery, Gmail, material generation, automatic submission, cloud sync, website, browser extension, desktop app, and MCP App UI are out of scope.
- Never report agreement with provisional labels as model accuracy.
- Use synthetic fixtures in the public repository; keep the 20 real evaluation JDs local and ignored by Git.

## Review Focus

- A JD requirement absent from the resume must create an `evidence_gap`, not a capability gap; Task 5 pins this behavior.
- Unconfirmed Agent-memory claims must never affect scoring; Tasks 4 and 5 pin this behavior.
- Missing or unknown facts must stay unknown and trigger review where configured, never silently become a pass or hard failure; Tasks 3 and 5 pin this behavior.
- A confirmed hard-constraint conflict must override otherwise high scores; Task 5 pins this behavior.
- Malformed model output or a quote not found in the source JD must fail safely and must not persist a successful decision; Tasks 6 and 7 pin this behavior.
- A provider payload must contain only the data classes approved for that operation; Tasks 6 and 7 pin this behavior.

---

## Planned File Structure

```text
ai-dream-job-engine/
├── .codex-plugin/plugin.json
├── .mcp.json
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── scripts/
│   ├── run_evaluation.py
│   └── start-mcp.sh
├── skills/ai-dream-job/SKILL.md
├── src/ai_dream_job_engine/
│   ├── __init__.py
│   ├── database.py
│   ├── decision.py
│   ├── evidence.py
│   ├── mcp_server.py
│   ├── models.py
│   ├── profile.py
│   ├── rubric.py
│   ├── service.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── jev.py
│   │   └── openai_compatible.py
│   └── templates/ai_job_v1.json
├── eval/
│   ├── README.md
│   └── synthetic_cases.jsonl
└── tests/
    ├── test_database.py
    ├── test_decision.py
    ├── test_evaluation.py
    ├── test_mcp_server.py
    ├── test_models.py
    ├── test_profile.py
    ├── test_providers.py
    ├── test_rubric.py
    └── test_service.py
```

## Task 1: Project Foundation and Domain Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ai_dream_job_engine/__init__.py`
- Create: `src/ai_dream_job_engine/models.py`
- Create: `src/ai_dream_job_engine/providers/__init__.py`
- Create: `src/ai_dream_job_engine/providers/base.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Approved design specification.
- Produces: Pydantic models `CandidateClaim`, `CandidateProfile`, `FactDefinition`, `ValueDimension`, `DecisionQuestion`, `Rubric`, `ExtractedFact`, `DecisionSignal`, `Gap`, `DecisionResult`, and `OutboundDisclosure`; protocols `JobFactExtractor`, `CandidateClaimExtractor`, and `DecisionSignalProvider`.

- [ ] **Step 1: Create the package metadata and failing model tests**

  Set Python to `>=3.12,<3.14`; runtime dependencies to `pydantic>=2.12,<3`, `openai>=3.24,<4`, `httpx>=0.28,<1`, and `mcp[cli]>=2,<3`; dev dependencies to `pytest>=9,<10` and `pytest-asyncio>=1.2,<2`.

  Add tests named `test_agent_suggested_claim_requires_confirmation`, `test_fact_definition_rejects_unknown_type`, `test_dimension_rejects_negative_weight`, `test_signal_probability_must_be_between_zero_and_one`, and `test_decision_result_requires_evidence_for_every_gap`.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `python -m pytest tests/test_models.py -v`

  Expected: collection fails because `ai_dream_job_engine.models` does not exist.

- [ ] **Step 3: Implement the minimal domain contracts and provider protocols**

  Required enums and values:

  - `ClaimSource`: `resume_evidence`, `user_confirmed`, `agent_suggested`.
  - `FactDataType`: `boolean`, `number`, `text`, `enum`, `list`.
  - `GapKind`: `preference_mismatch`, `hard_blocker`, `capability_gap`, `evidence_gap`.
  - `RecommendationLevel`: `priority_apply`, `apply_after_review`, `hold`, `skip`.
  - `SignalStatus`: `available`, `unavailable`, `invalid`.
  - `DataClass`: `public_job_data`, `selected_candidate_claims`, `restricted_personal_data`.

  Protocol signatures:

  - `JobFactExtractor.extract(job_description: str, rubric: Rubric) -> list[ExtractedFact]`
  - `CandidateClaimExtractor.extract(resume_text: str) -> list[CandidateClaim]`
  - `DecisionSignalProvider.judge(job_description: str, questions: list[DecisionQuestion]) -> list[DecisionSignal]`

  Also define the cross-task records used later: `RubricPatch`, `ClaimDecision`, `MemoryImportPreview`, `AnalysisRecord`, `AnalysisSummary`, `DecisionTrace`, `EvaluationCase`, and `EvaluationReport`.

- [ ] **Step 4: Run the model tests and full suite**

  Run: `python -m pytest tests/test_models.py -v`

  Expected: all model tests pass.

- [ ] **Step 5: Commit the domain foundation**

  Commit: `feat: define core decision contracts`

## Task 2: Local SQLite Store

**Files:**
- Create: `src/ai_dream_job_engine/database.py`
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: Domain models from Task 1.
- Produces: `LocalStore.open(path: Path) -> LocalStore`, `initialize() -> None`, `save_profile(profile: CandidateProfile) -> None`, `load_profile() -> CandidateProfile | None`, `save_rubric(rubric: Rubric) -> None`, `load_rubric() -> Rubric | None`, `save_analysis(record: AnalysisRecord) -> None`, and `list_analyses() -> list[AnalysisRecord]`.

- [ ] **Step 1: Write failing persistence tests**

  Add `test_initialize_is_idempotent`, `test_profile_round_trip_preserves_claim_sources`, `test_rubric_round_trip_preserves_version`, `test_analysis_round_trip_preserves_decision_trace`, and `test_invalid_saved_payload_raises_store_error` using a temporary SQLite file.

- [ ] **Step 2: Run tests and verify the store is missing**

  Run: `python -m pytest tests/test_database.py -v`

  Expected: FAIL because `LocalStore` is undefined.

- [ ] **Step 3: Implement the smallest versioned SQLite schema**

  Use `sqlite3` from the standard library. Store validated JSON payloads in three tables: `profiles`, `rubrics`, and `analyses`, each with `schema_version`, timestamps, and stable IDs. Keep migration version in `PRAGMA user_version`; version 1 is sufficient.

- [ ] **Step 4: Run persistence tests**

  Run: `python -m pytest tests/test_database.py -v`

  Expected: all persistence tests pass and no files are written outside the test temporary directory.

- [ ] **Step 5: Commit local persistence**

  Commit: `feat: add local sqlite store`

## Task 3: Default AI-Job Rubric and Declarative Updates

**Files:**
- Create: `src/ai_dream_job_engine/templates/ai_job_v1.json`
- Create: `src/ai_dream_job_engine/rubric.py`
- Test: `tests/test_rubric.py`

**Interfaces:**
- Consumes: `Rubric`, `FactDefinition`, `ValueDimension`, and `DecisionQuestion` from Task 1.
- Produces: `load_default_rubric() -> Rubric` and `apply_rubric_patch(rubric: Rubric, patch: RubricPatch) -> Rubric`.

- [ ] **Step 1: Write failing default-template and patch tests**

  Add `test_default_rubric_contains_ai_job_dimensions`, `test_default_questions_include_product_applied_ai_and_infrastructure`, `test_patch_can_change_weight_and_threshold`, `test_patch_cannot_reference_missing_fact_or_signal`, and `test_missing_fact_policy_remains_unknown`.

  Assert the default template includes dimensions `role_family_fit`, `applied_ai_depth`, `product_customer_proximity`, `ownership_impact`, `infrastructure_balance`, `work_arrangement`, `compensation`, and `work_authorization`; decision questions `product_facing`, `applied_ai`, and `pure_infrastructure`; and a rubric version of `ai-job-v1`.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_rubric.py -v`

  Expected: FAIL because the template and loader do not exist.

- [ ] **Step 3: Implement the default JSON template and validated patching**

  Include definitions, examples, positive/negative keywords, weights, hard-constraint flags, and missing-value policy. `apply_rubric_patch` may add or update declarative fields but must reject arbitrary code, unknown references, duplicate IDs, and negative weights.

- [ ] **Step 4: Run rubric tests**

  Run: `python -m pytest tests/test_rubric.py -v`

  Expected: all rubric tests pass.

- [ ] **Step 5: Commit the configurable rubric**

  Commit: `feat: add configurable AI job rubric`

## Task 4: Candidate Profile and Consent-Based Agent Memory Import

**Files:**
- Create: `src/ai_dream_job_engine/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `CandidateClaim`, `CandidateProfile`, `CandidateClaimExtractor`, and `LocalStore`.
- Produces: `propose_resume_claims(resume_text: str, extractor: CandidateClaimExtractor) -> list[CandidateClaim]`, `preview_memory_import(profile: CandidateProfile, source_agent: str, scope: list[str], candidates: list[CandidateClaim]) -> MemoryImportPreview`, `confirm_claims(profile: CandidateProfile, decisions: list[ClaimDecision]) -> CandidateProfile`, and `disconnect_memory_source(profile: CandidateProfile, source_agent: str, delete_imported: bool) -> CandidateProfile`.

- [ ] **Step 1: Write failing profile and consent tests**

  Add `test_resume_claims_keep_exact_source_quote`, `test_memory_import_is_preview_only`, `test_unconfirmed_memory_claim_is_not_active`, `test_confirmed_memory_claim_becomes_user_confirmed_and_keeps_provenance`, `test_remote_resume_extraction_requires_explicit_disclosure_confirmation`, `test_refresh_reports_conflict_without_overwrite`, and `test_disconnect_can_remove_only_selected_source`.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_profile.py -v`

  Expected: FAIL because profile services do not exist.

- [ ] **Step 3: Implement profile proposal, diff, confirmation, and disconnect logic**

  The service receives candidate memory claims from the host Agent; it never reads Agent memory itself. Deduplicate by normalized claim key, preserve every source, and require an explicit `ClaimDecision` before changing active profile data. A remote resume extractor cannot run without a confirmed `OutboundDisclosure` naming the provider, fields, purpose, and `restricted_personal_data` classification.

- [ ] **Step 4: Run profile tests**

  Run: `python -m pytest tests/test_profile.py -v`

  Expected: all profile tests pass, including the unconfirmed-memory boundary.

- [ ] **Step 5: Commit candidate-profile workflows**

  Commit: `feat: add consent based candidate profile`

## Task 5: Pure Decision Engine and Gap Classification

**Files:**
- Create: `src/ai_dream_job_engine/decision.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Consumes: Confirmed `CandidateProfile`, `Rubric`, `ExtractedFact`, and `DecisionSignal` values.
- Produces: `decide_job(facts: list[ExtractedFact], signals: list[DecisionSignal], profile: CandidateProfile, rubric: Rubric) -> DecisionResult`.

- [ ] **Step 1: Write failing decision-table tests**

  Add table-driven tests for:

  - high preference plus strong candidate evidence -> `priority_apply`;
  - high preference plus missing resume evidence -> `apply_after_review` with `evidence_gap`;
  - explicit user-confirmed missing capability -> `hold` with `capability_gap`;
  - low preference plus strong capability -> `skip` with `preference_mismatch`;
  - confirmed work-authorization conflict -> `skip` with `hard_blocker`, regardless of other scores;
  - unconfirmed Agent-memory claim does not change either score;
  - unknown required fact follows the rubric's review policy rather than silently passing or failing;
  - identical normalized inputs return identical serialized results.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_decision.py -v`

  Expected: FAIL because `decide_job` is undefined.

- [ ] **Step 3: Implement deterministic scoring and recommendation precedence**

  Precedence is: confirmed hard blocker -> unresolved required evidence -> weighted preference and qualification scores. Only active confirmed claims may support qualification. Every score contribution and gap must include its input IDs and evidence references.

- [ ] **Step 4: Run decision tests twice**

  Run twice as separate commands: `python -m pytest tests/test_decision.py -v`

  Expected: both runs pass with identical result fixtures.

- [ ] **Step 5: Commit the decision core**

  Commit: `feat: add deterministic job decisions`

## Task 6: Evidence Validation and Model Provider Adapters

**Files:**
- Create: `src/ai_dream_job_engine/evidence.py`
- Create: `src/ai_dream_job_engine/providers/openai_compatible.py`
- Create: `src/ai_dream_job_engine/providers/jev.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: Provider protocols, rubric definitions, and domain models from Task 1.
- Produces: `build_evidence_chunks(text: str) -> EvidenceDocument`, `validate_quote(document: EvidenceDocument, quote: str) -> EvidenceRef`, `OpenAICompatibleProvider`, and `JevSignalProvider`.

- [ ] **Step 1: Write failing adapter tests with no network access**

  Add `test_openai_direct_uses_official_base_url`, `test_openrouter_uses_openrouter_base_url`, `test_job_extraction_payload_excludes_candidate_profile`, `test_provider_returns_validated_job_facts`, `test_quote_not_in_jd_is_rejected`, `test_malformed_structured_output_raises_provider_error`, `test_jev_payload_contains_only_jd_and_questions`, `test_jev_maps_noul_probability`, `test_jev_missing_answer_returns_invalid_signal`, and `test_missing_api_key_fails_before_network_call`.

  Inject fake OpenAI clients and `httpx.MockTransport`; never call live services in unit tests.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_providers.py -v`

  Expected: FAIL because evidence and provider adapters do not exist.

- [ ] **Step 3: Implement evidence-preserving OpenAI/OpenRouter extraction**

  `OpenAICompatibleProvider` accepts provider name, API key, and model. OpenAI uses its official endpoint; OpenRouter uses `https://openrouter.ai/api/v1`. Validate every returned JD quote against the original source before constructing domain models.

- [ ] **Step 4: Implement the JEV adapter**

  POST to `https://api.typesafe.ai/v1/systemone` with model `jev-latest`, versioned questions, and `TYPESAFE_API_KEY`. Convert each `noul` answer to a `DecisionSignal` probability. Missing, malformed, or out-of-range values become invalid signals and do not silently default to false.

- [ ] **Step 5: Run provider tests and commit**

  Run: `python -m pytest tests/test_providers.py -v`

  Expected: all provider tests pass without network access.

  Commit: `feat: add evidence backed model adapters`

## Task 7: Analysis Service and Recompute Path

**Files:**
- Create: `src/ai_dream_job_engine/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `LocalStore`, provider protocols, `decide_job`, active profile, and active rubric.
- Produces: `JobAnalysisService.analyze_job(job_description: str) -> AnalysisRecord`, `recompute_decision(analysis_id: str) -> AnalysisRecord`, `get_decision_trace(analysis_id: str) -> DecisionTrace`, and `list_analyzed_jobs() -> list[AnalysisSummary]`.

- [ ] **Step 1: Write failing orchestration tests with fake providers**

  Add `test_analyze_job_runs_extract_signals_decide_and_persist`, `test_recompute_reuses_normalized_inputs_without_model_calls`, `test_missing_profile_returns_actionable_error`, `test_invalid_quote_does_not_persist_success`, `test_provider_failure_records_failure_not_decision`, `test_trace_records_outbound_data_class_without_raw_restricted_content`, and `test_trace_contains_provider_rubric_prompt_and_result_versions`.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_service.py -v`

  Expected: FAIL because `JobAnalysisService` does not exist.

- [ ] **Step 3: Implement the application service**

  Keep orchestration here, not in MCP handlers. Persist raw input hash, normalized facts, signals, result, latency, provider/model IDs, and versions. Store failure records separately from successful decisions.

- [ ] **Step 4: Run service tests**

  Run: `python -m pytest tests/test_service.py -v`

  Expected: all service tests pass and recompute performs zero provider calls.

- [ ] **Step 5: Commit the analysis workflow**

  Commit: `feat: orchestrate job analysis and traces`

## Task 8: MCP Server, Codex Plugin, and Skill

**Files:**
- Create: `src/ai_dream_job_engine/mcp_server.py`
- Create: `.codex-plugin/plugin.json`
- Create: `.mcp.json`
- Create: `scripts/start-mcp.sh`
- Create: `skills/ai-dream-job/SKILL.md`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Profile, rubric, store, and analysis services from Tasks 2-7.
- Produces: MCP tools named `initialize_profile`, `review_profile_claims`, `confirm_profile_claims`, `preview_agent_memory_import`, `confirm_agent_memory_import`, `disconnect_agent_memory`, `get_rubric`, `update_rubric`, `analyze_job`, `recompute_decision`, `get_decision_trace`, and `list_analyzed_jobs`.

- [ ] **Step 1: Write failing MCP contract and manifest tests**

  Add `test_expected_tool_names_are_registered`, `test_tool_schemas_use_domain_contracts`, `test_memory_preview_requires_explicit_source_and_scope`, `test_analyze_tool_returns_structured_result`, `test_plugin_manifest_points_to_mcp_config_and_skill`, `test_skill_declares_host_owned_automation_boundary`, and `test_start_script_uses_repo_local_project`.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_mcp_server.py -v`

  Expected: FAIL because the MCP server and plugin files do not exist.

- [ ] **Step 3: Implement the MCP Python SDK v2 server**

  Use `from mcp.server import MCPServer` and stdio transport. MCP handlers validate inputs and delegate immediately to services; they contain no scoring or provider logic.

- [ ] **Step 4: Add the Codex packaging and conversational Skill**

  The plugin manifest uses name `ai-dream-job-engine`, version `0.1.0`, `skills: "./skills/"`, and `mcpServers: "./.mcp.json"`. The Skill must ask consent before memory import, require claim confirmation, label unknowns, avoid unsupported capability claims, and require user confirmation before any future application action. It must state that schedules and notifications are owned by the host Agent, define quiet `no_change` behavior, and reserve future coarse-grained automation workflows for job discovery and Gmail status sync. `scripts/start-mcp.sh` starts the project through `uv run --project` and does not embed secrets.

- [ ] **Step 5: Run MCP tests and a local protocol smoke test**

  Run: `python -m pytest tests/test_mcp_server.py -v`

  Run: `uv run mcp dev src/ai_dream_job_engine/mcp_server.py`

  Expected: tests pass; MCP Inspector lists the expected tools without importing personal data or requiring API keys at startup.

- [ ] **Step 6: Commit the Agent plugin surface**

  Commit: `feat: expose the decision engine as a Codex plugin`

## Task 9: Evaluation Harness and Open-Source Delivery

**Files:**
- Create: `scripts/run_evaluation.py`
- Create: `eval/synthetic_cases.jsonl`
- Create: `eval/README.md`
- Create: `tests/test_evaluation.py`
- Create: `README.md`
- Create: `LICENSE`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `JobAnalysisService` and serialized human labels.
- Produces: `run_evaluation(cases: list[EvaluationCase], service: JobAnalysisService) -> EvaluationReport` and a CLI that accepts a local JSONL path and writes a local report.

- [ ] **Step 1: Write failing evaluation tests**

  Add `test_case_requires_human_judgment_before_system_result`, `test_report_separates_agreement_from_accuracy`, `test_report_counts_gap_classification`, `test_real_eval_paths_are_gitignored`, and `test_synthetic_fixture_contains_no_personal_data`.

- [ ] **Step 2: Run tests and verify they fail**

  Run: `python -m pytest tests/test_evaluation.py -v`

  Expected: FAIL because the evaluation harness does not exist.

- [ ] **Step 3: Implement the evaluation harness and synthetic fixtures**

  Require each local case to contain the user's pre-recorded recommendation and gap labels before analysis. Report recommendation agreement, blocker precision, gap-type confusion counts, unsupported-claim count, rerun stability, top-five usefulness fields, and review time. Label all comparisons as agreement, not accuracy.

- [ ] **Step 4: Add open-source documentation and privacy boundaries**

  README sections: product thesis, what is different, installation, model configuration, memory-consent flow, one-JD quickstart, evidence boundaries, privacy data classes and outbound disclosures, local data location, evaluation method, limitations, and deferred roadmap. Use the MIT license. Ignore `.env`, SQLite files, local evaluation cases/reports, resume files, caches, and virtual environments.

- [ ] **Step 5: Run all static and automated checks**

  Run: `python -m pytest -v`

  Run: `python -m compileall -q src scripts`

  Run: `git diff --check`

  Expected: all tests pass, compilation succeeds, and the diff check is clean.

- [ ] **Step 6: Run the private 20-JD milestone with the user**

  Create the local ignored case file only after the user records independent judgments. Run the CLI, inspect every disagreement, and save a local failure-review report. Do not commit real JDs, resume content, or private reports.

- [ ] **Step 7: Commit the validated open-source MVP**

  Commit: `docs: add evaluation and open source guide`

## Final Verification

- [ ] Confirm `git status --short` is clean except explicitly ignored local evaluation artifacts.
- [ ] Confirm the old AI Job Fit Analyzer still shows only its pre-existing untracked files.
- [ ] Install the plugin from the new repository in a clean local environment.
- [ ] Verify profile initialization, memory preview and confirmation, rubric update, one-JD analysis, recompute, trace, and local listing end to end.
- [ ] Verify startup does not require API keys and live analysis gives a clear configuration error when the selected provider key is missing.
- [ ] Verify no resume text, real JD text, Agent-memory content, SQLite database, or credential is tracked by Git.
- [ ] Request a final whole-branch code review before publishing the repository.
