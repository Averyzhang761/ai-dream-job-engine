# AI Dream Job Engine MCP App UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local MCP App side panel that lets a user review analyzed AI jobs, understand recommendation evidence and gaps, manage personal standards and confirmed evidence, and persist an explicit review status without creating a website or duplicating decision logic.

**Architecture:** Keep Python as the only owner of persisted state and decisions. Add a small workspace application layer plus MCP data/action tools, then register one versioned `text/html;profile=mcp-app` resource and decoupled render tools. Build the UI as a single bundled React/TypeScript document that talks to the existing MCP server through the standard MCP Apps bridge; use OpenAI metadata only for the Codex thread-side-panel entrypoint.

**Tech Stack:** Python 3.12/3.13, Pydantic 2, SQLite, MCP Python SDK 2.3+, React 19, TypeScript, Vite 8, `@modelcontextprotocol/ext-apps`, Vitest, Testing Library, Playwright CLI.

**Spec:** `docs/superpowers/specs/2026-10-10-ai-dream-job-engine-design.md`

## Global Constraints

- Work only in `ai-dream-job-engine`; do not modify the personal workbench or the old AI Job Fit Analyzer.
- The MCP App has exactly three top-level views: `岗位`, `我的标准`, and `我的证据`; the default view is `岗位`.
- Keep profile, rubric, review status, analyses, and audit data in the existing local SQLite database. No cloud account, telemetry, or secondary browser storage is an authority.
- The UI may format data and collect user choices, but it must not calculate scores, classify gaps, decide eligibility, or interpret evidence.
- Use the MCP Apps standard bridge for initialization, tool results, and `tools/call`. Feature-detect optional OpenAI extensions; do not branch on the host product name.
- Package the component as one versioned local HTML resource at `ui://ai-dream-job-engine/workspace-v1.html` with MIME type `text/html;profile=mcp-app`; load no remote fonts, scripts, images, or styles.
- Register a Codex thread-side-panel entrypoint. Existing MCP tools remain useful when the host cannot render MCP Apps.
- Agent Memory stays off by default. The UI can request a host-Agent preview, but only confirmed claims may enter the active profile.
- A missing resume claim is an `evidence_gap`, not a `capability_gap`.
- Only a `priority_apply` analysis with no `hard_blocker` and no unresolved `evidence_gap` may be marked `ready_to_apply`; this state never submits an application.
- No job search, Gmail sync, form filling, browser login, CAPTCHA handling, application submission, or scheduler is implemented in this plan.
- Use white, graphite, status green, and warning red; color communicates status only. Avoid nested cards, oversized headings, decorative gradients, and marketing copy.
- Support a 360 px narrow panel without clipped controls; provide visible keyboard focus and respect `prefers-reduced-motion`.

## Review Focus

- A user with no initialized profile or rubric must see one precise setup action instead of a blank or crashing panel; Task 2 and Task 4 test this.
- A direct tool call must not mark an ineligible analysis `ready_to_apply`, even if the UI is bypassed; Task 2 tests the server-side gate.
- Failed, missing, or integrity-invalid analyses must not appear as successful decisions or expose unsafe error details; Tasks 2 and 3 test this.
- An expired or mismatched resume/Agent-memory preview must fail without changing the profile and leave the user able to request a new preview; Tasks 2 and 6 test this.
- A host without the MCP Apps bridge, or a resource bundle missing from the installed package, must fail visibly and leave the existing conversational tools usable; Tasks 3, 4, and 7 test this.

---

## Planned File Structure

```text
ai-dream-job-engine/
├── package.json
├── package-lock.json
├── ui/
│   ├── index.html
│   ├── src/
│   │   ├── App.tsx
│   │   ├── bridge.ts
│   │   ├── contracts.ts
│   │   ├── styles.css
│   │   ├── components/
│   │   │   ├── EvidenceView.tsx
│   │   │   ├── JobDetail.tsx
│   │   │   ├── JobList.tsx
│   │   │   ├── RubricView.tsx
│   │   │   └── StatusBadge.tsx
│   │   └── test/
│   │       ├── App.test.tsx
│   │       ├── EvidenceView.test.tsx
│   │       ├── JobWorkspace.test.tsx
│   │       └── RubricView.test.tsx
│   ├── tsconfig.json
│   └── vite.config.ts
├── src/ai_dream_job_engine/
│   ├── database.py
│   ├── mcp_app.py
│   ├── mcp_server.py
│   ├── models.py
│   ├── workspace.py
│   └── ui/workspace-v1.html
├── tests/
│   ├── test_database.py
│   ├── test_mcp_app.py
│   ├── test_mcp_server.py
│   ├── test_models.py
│   └── test_workspace.py
└── docs/superpowers/plans/2026-10-11-mcp-app-ui-implementation.md
```

## Task 1: Persisted Job Review State

**Files:**
- Modify: `src/ai_dream_job_engine/models.py`
- Modify: `src/ai_dream_job_engine/database.py`
- Test: `tests/test_models.py`
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: Existing `AnalysisRecord`, `RecommendationLevel`, `GapKind`, and schema-versioned `LocalStore`.
- Produces: `JobReviewStatus`, `JobReviewRecord`, `LocalStore.save_job_review(record: JobReviewRecord) -> None`, `LocalStore.load_job_review(analysis_id: str) -> JobReviewRecord | None`, and `LocalStore.list_job_reviews() -> list[JobReviewRecord]`.

- [ ] **Step 1: Write failing model tests**

  Add `test_job_review_status_has_only_supported_values` and `test_job_review_record_requires_analysis_id`. Assert the exact status values `unreviewed`, `kept`, `ignored`, and `ready_to_apply`.

- [ ] **Step 2: Run the focused model tests**

  Run: `.venv/bin/python -m pytest tests/test_models.py -k job_review -v`

  Expected: FAIL because `JobReviewStatus` and `JobReviewRecord` do not exist.

- [ ] **Step 3: Implement the review-state contracts**

  Add `JobReviewStatus(StrEnum)` and `JobReviewRecord(DomainModel)` with `analysis_id: str`, `status: JobReviewStatus`, and UTC `updated_at: datetime`. Do not put review state inside `AnalysisRecord`; analyses remain immutable decision records.

- [ ] **Step 4: Run the focused model tests**

  Run: `.venv/bin/python -m pytest tests/test_models.py -k job_review -v`

  Expected: PASS.

- [ ] **Step 5: Write failing migration and round-trip tests**

  Add `test_version_one_database_migrates_to_review_state_schema`, `test_job_review_round_trip`, `test_job_review_upsert_preserves_one_row_per_analysis`, and `test_review_state_schema_rejects_unexpected_objects`. Build a real version-1 fixture using the existing three-table schema, then initialize it with the new store.

- [ ] **Step 6: Run the focused database tests**

  Run: `.venv/bin/python -m pytest tests/test_database.py -k 'review or migrates' -v`

  Expected: FAIL because the store has no `job_reviews` table or methods.

- [ ] **Step 7: Implement schema version 2 and review persistence**

  Migrate version 1 to version 2 by creating `job_reviews` with the same validated JSON envelope used by the existing tables. Preserve all existing rows. Update strict schema validation and add the three store methods from the Interfaces block.

- [ ] **Step 8: Run database tests and the full Python suite**

  Run: `.venv/bin/python -m pytest tests/test_database.py -v && .venv/bin/python -m pytest -q`

  Expected: all tests pass.

- [ ] **Step 9: Commit the persisted review state**

  Commit: `feat: persist job review state`

## Task 2: Workspace Queries and Server-Side Review Gates

**Files:**
- Create: `src/ai_dream_job_engine/workspace.py`
- Modify: `src/ai_dream_job_engine/models.py`
- Modify: `src/ai_dream_job_engine/mcp_server.py`
- Test: `tests/test_workspace.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `LocalStore`, active profile/rubric applications, existing analysis service, and Task 1 review records.
- Produces: `WorkspaceJobSummary`, `WorkspaceJobDetail`, `WorkspaceSnapshot`, `ClaimPreviewView`; `WorkspaceApplication.get_snapshot() -> WorkspaceSnapshot`, `get_job_detail(analysis_id: str) -> WorkspaceJobDetail`, `set_review_status(analysis_id: str, status: JobReviewStatus) -> WorkspaceJobDetail`, `get_claim_preview(preview_id: str) -> ClaimPreviewView`, and `get_profile() -> CandidateProfile`.

- [ ] **Step 1: Write failing snapshot tests**

  Add `test_snapshot_reports_setup_actions_when_profile_and_rubric_are_missing`, `test_snapshot_lists_only_successful_integrity_valid_analyses`, `test_snapshot_joins_review_status_without_mutating_analysis`, and `test_job_detail_preserves_all_four_gap_kinds_and_evidence`.

- [ ] **Step 2: Run the snapshot tests**

  Run: `.venv/bin/python -m pytest tests/test_workspace.py -k 'snapshot or detail' -v`

  Expected: FAIL because `WorkspaceApplication` does not exist.

- [ ] **Step 3: Implement read models and workspace queries**

  `WorkspaceJobSummary` contains analysis ID, title, company, recommendation, preference score, qualification score, top unresolved gap, review status, creation time, and `can_mark_ready_to_apply`. `WorkspaceJobDetail` contains the summary, all gaps grouped by `GapKind`, contributions, and trace evidence. `WorkspaceSnapshot` contains profile/rubric readiness, active rubric metadata, confirmed claims, and summaries sorted newest first.

- [ ] **Step 4: Run the snapshot tests**

  Run: `.venv/bin/python -m pytest tests/test_workspace.py -k 'snapshot or detail' -v`

  Expected: PASS.

- [ ] **Step 5: Write failing action-gate and preview tests**

  Add `test_ready_to_apply_requires_priority_recommendation`, `test_ready_to_apply_rejects_hard_blocker`, `test_ready_to_apply_rejects_unresolved_evidence_gap`, `test_keep_and_ignore_are_always_explicit_user_actions`, `test_unknown_analysis_does_not_create_review_state`, `test_expired_claim_preview_returns_safe_error`, and `test_claim_preview_does_not_mutate_profile`.

- [ ] **Step 6: Run the action tests**

  Run: `.venv/bin/python -m pytest tests/test_workspace.py -k 'ready or review or preview' -v`

  Expected: FAIL because action methods are missing.

- [ ] **Step 7: Implement status validation and preview retrieval**

  Enforce `ready_to_apply` eligibility in Python before persistence. Reuse the server-issued preview registries; expose immutable preview data by ID without adding a second preview store. Translate missing, expired, or mismatched previews to the existing fixed safe error categories.

- [ ] **Step 8: Expose data/action MCP tools**

  Add `get_workspace_snapshot`, `get_workspace_job`, `set_job_review_status`, `get_candidate_profile`, and `get_claim_preview`. Keep these tools useful without a UI and return typed structured output. Do not attach a UI template to these data/action tools.

- [ ] **Step 9: Run workspace, MCP, safety, and full tests**

  Run: `.venv/bin/python -m pytest tests/test_workspace.py tests/test_mcp_server.py tests/test_mcp_server_safety.py -v && .venv/bin/python -m pytest -q`

  Expected: all tests pass and no exception text exposes profile values, JD text, or credentials.

- [ ] **Step 10: Commit workspace operations**

  Commit: `feat: add workspace query and review tools`

## Task 3: MCP App Resource and Render Tools

**Files:**
- Create: `src/ai_dream_job_engine/mcp_app.py`
- Create: `src/ai_dream_job_engine/ui/workspace-v1.html`
- Modify: `src/ai_dream_job_engine/mcp_server.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mcp_app.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: Task 2 `WorkspaceApplication`, `WorkspaceSnapshot`, and `ClaimPreviewView`.
- Produces: `WORKSPACE_RESOURCE_URI = "ui://ai-dream-job-engine/workspace-v1.html"`, `load_workspace_html() -> str`, `open_job_workspace() -> WorkspaceAppPayload`, and `open_claim_review(preview_id: str) -> WorkspaceAppPayload`.

- [ ] **Step 1: Write failing resource-registration tests**

  Add `test_workspace_resource_uses_versioned_ui_uri_and_mcp_app_mime`, `test_workspace_resource_is_self_contained_and_has_no_remote_domains`, `test_open_workspace_tool_links_only_to_workspace_resource`, `test_thread_entrypoint_metadata_is_present`, and `test_existing_data_tools_have_no_output_template`.

- [ ] **Step 2: Run the MCP App tests**

  Run: `.venv/bin/python -m pytest tests/test_mcp_app.py -v`

  Expected: FAIL because the resource and render tools are missing.

- [ ] **Step 3: Implement packaged-resource loading**

  Read `ui/workspace-v1.html` with `importlib.resources`, require a nonempty bundled document, and return it through `@server.resource` using `text/html;profile=mcp-app`. Add package data for `ui/*.html`. Use a zero-network CSP: empty `connectDomains`, `resourceDomains`, and `frameDomains`.

- [ ] **Step 4: Implement decoupled render tools**

  Register `open_job_workspace` and `open_claim_review` with `_meta.ui.resourceUri` set to `WORKSPACE_RESOURCE_URI`. Add `"openai/ui": {"entrypoints": [{"type": "thread"}]}` only to `open_job_workspace`. Render tools return typed initial data; they do not mutate profile, rubric, review state, or analysis data.

- [ ] **Step 5: Test missing-bundle failure and tool-only fallback**

  Add `test_missing_ui_bundle_fails_with_fixed_public_error` and assert the server can still list and call pre-existing tools when the UI resource cannot be read.

- [ ] **Step 6: Run MCP App, MCP server, and full Python tests**

  Run: `.venv/bin/python -m pytest tests/test_mcp_app.py tests/test_mcp_server.py tests/test_mcp_server_safety.py -v && .venv/bin/python -m pytest -q`

  Expected: all tests pass.

- [ ] **Step 7: Commit the MCP App server surface**

  Commit: `feat: register dream job MCP App`

## Task 4: UI Bundle, Bridge, and Accessible Shell

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `ui/index.html`
- Create: `ui/tsconfig.json`
- Create: `ui/vite.config.ts`
- Create: `ui/src/contracts.ts`
- Create: `ui/src/bridge.ts`
- Create: `ui/src/App.tsx`
- Create: `ui/src/styles.css`
- Create: `ui/src/test/App.test.tsx`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 3 `WorkspaceAppPayload`; MCP Apps `ui/initialize`, `ui/notifications/tool-result`, `tools/call`, and `ui/message`.
- Produces: `createMcpAppBridge(): McpAppBridge`, `McpAppBridge.callTool<T>(name: string, args: object) -> Promise<T>`, React `App`, and deterministic build output copied to `src/ai_dream_job_engine/ui/workspace-v1.html`.

- [ ] **Step 1: Add the minimal UI toolchain and failing shell tests**

  Pin React 19, TypeScript, Vite 8, Vitest, Testing Library, jsdom, Lucide React, and `@modelcontextprotocol/ext-apps`. Add scripts `ui:build`, `ui:test`, and `ui:check`. Test the three labels `岗位`, `我的标准`, and `我的证据`, default selection of `岗位`, keyboard tab switching, and a 360 px layout class without horizontal overflow.

- [ ] **Step 2: Run the shell tests**

  Run: `npm run ui:test -- --run ui/src/test/App.test.tsx`

  Expected: FAIL because the bridge and app shell do not exist.

- [ ] **Step 3: Implement the standard MCP Apps bridge wrapper**

  Keep protocol details inside `bridge.ts`. Initialize through the shared bridge, subscribe to tool-result notifications, expose typed `tools/call`, and expose `sendMessage` for requests that require the host Agent. Return a visible `UI unavailable; use the conversation tools` state if initialization is unsupported or times out; never silently use mock data.

- [ ] **Step 4: Implement the three-view shell and token system**

  Use CSS custom properties for white, graphite, status green, warning red, borders, spacing, focus, and typography. Use a compact segmented control for the three views, an `aria-live` status region, stable narrow-panel dimensions, no nested cards, and reduced-motion media rules.

- [ ] **Step 5: Build one self-contained HTML document**

  Configure Vite to inline JavaScript and CSS into one `workspace-v1.html`, then copy it to `src/ai_dream_job_engine/ui/`. The built document must not reference `http:`, `https:`, external fonts, or separate local assets.

- [ ] **Step 6: Run UI tests, build, and package-resource tests**

  Run: `npm run ui:check && npm run ui:build && .venv/bin/python -m pytest tests/test_mcp_app.py -v`

  Expected: TypeScript passes, UI tests pass, the bundle builds, and Python can load the packaged HTML.

- [ ] **Step 7: Commit the UI foundation**

  Commit: `feat: add MCP App UI foundation`

## Task 5: Job Queue, Decision Detail, and Review Actions

**Files:**
- Create: `ui/src/components/StatusBadge.tsx`
- Create: `ui/src/components/JobList.tsx`
- Create: `ui/src/components/JobDetail.tsx`
- Create: `ui/src/test/JobWorkspace.test.tsx`
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/styles.css`

**Interfaces:**
- Consumes: `McpAppBridge`, `WorkspaceJobSummary`, `WorkspaceJobDetail`, `get_workspace_job`, and `set_job_review_status`.
- Produces: `JobList`, `JobDetail`, and the complete `岗位` view.

- [ ] **Step 1: Write failing job-workspace tests**

  Add tests for newest-first summaries, company/title fallback text, four Chinese recommendation labels, preference/qualification score formatting, top-gap preview, loading and empty states, and selection without remounting the whole app.

- [ ] **Step 2: Run the job-workspace tests**

  Run: `npm run ui:test -- --run ui/src/test/JobWorkspace.test.tsx`

  Expected: FAIL because job components do not exist.

- [ ] **Step 3: Implement list and detail rendering**

  The detail order is fixed: recommendation and one-sentence conclusion; preference and qualification evidence scores; hard blockers; preference, capability, and evidence gaps; JD and candidate evidence; actions. Use semantic sections and disclose evidence on demand without hiding the gap classification.

- [ ] **Step 4: Write failing review-action tests**

  Test `保留`, `忽略`, and eligible `加入待申请`; disabled `加入待申请` must show the server-provided reason. Test server rejection after stale UI state and verify the UI refreshes the selected detail instead of pretending success.

- [ ] **Step 5: Run review-action tests**

  Run: `npm run ui:test -- --run ui/src/test/JobWorkspace.test.tsx`

  Expected: FAIL because actions are not wired.

- [ ] **Step 6: Wire review actions through MCP tools**

  Call `set_job_review_status`, replace local summary/detail only with the returned server value, and announce success or fixed safe error text in `aria-live`. Never infer eligibility in the UI beyond displaying server fields.

- [ ] **Step 7: Run job UI tests and build**

  Run: `npm run ui:test -- --run ui/src/test/JobWorkspace.test.tsx && npm run ui:build`

  Expected: PASS.

- [ ] **Step 8: Commit the job decision workspace**

  Commit: `feat: add job decision workspace`

## Task 6: Standards, Evidence, and Claim Confirmation

**Files:**
- Create: `ui/src/components/RubricView.tsx`
- Create: `ui/src/components/EvidenceView.tsx`
- Create: `ui/src/test/RubricView.test.tsx`
- Create: `ui/src/test/EvidenceView.test.tsx`
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/styles.css`
- Modify: `src/ai_dream_job_engine/mcp_server.py`
- Test: `tests/test_mcp_app.py`

**Interfaces:**
- Consumes: Existing `get_rubric`, `update_rubric`, `recompute_decision`, `get_candidate_profile`, `get_claim_preview`, `confirm_profile_claims`, and `confirm_agent_memory_import`; Task 4 `McpAppBridge`.
- Produces: complete `我的标准`, `我的证据`, and claim-review modes.

- [ ] **Step 1: Write failing standards tests**

  Test rendering facts and value dimensions from server data; adding a complete fact definition and value dimension; editing names, definitions, extraction instructions, positive/negative keywords, weights, hard constraints, and recommendation thresholds; rejecting incomplete definitions and invalid numeric input before a tool call; preserving untouched declarative fields; saving through `update_rubric`; and explicitly recomputing the selected analysis through `recompute_decision`.

- [ ] **Step 2: Run standards tests**

  Run: `npm run ui:test -- --run ui/src/test/RubricView.test.tsx`

  Expected: FAIL because `RubricView` does not exist.

- [ ] **Step 3: Implement the generated declarative editor**

  Render controls from the active rubric rather than hard-coding the user's preferences. Provide `新增事实字段` and `新增价值维度` forms that collect every required declarative field and reference only currently defined fact/signal IDs. Send the smallest `RubricPatch` containing additions or changed fields. After save, mark displayed analyses as needing recomputation and provide one clear `重新计算当前岗位` action; replace the detail only with the returned record.

- [ ] **Step 4: Write failing evidence and claim-review tests**

  Test grouped confirmed claims with provenance; separate resume, user-confirmed, and Agent-derived evidence; default-off Agent Memory sync; `ui/message` request with selected scope; accept/edit/reject for every issued claim; invalid or expired preview recovery; and no profile mutation before confirmation.

- [ ] **Step 5: Run evidence tests**

  Run: `npm run ui:test -- --run ui/src/test/EvidenceView.test.tsx`

  Expected: FAIL because evidence and review flows do not exist.

- [ ] **Step 6: Implement evidence display and host-Agent handoff**

  `同步 Agent Memory` sends a scoped message to the host Agent; it does not read memory itself. Render `open_claim_review` payloads in the same component, require one decision per candidate, and call the matching confirmation tool only after explicit submission. On expired preview, retain no editable values and offer `重新生成预览` through the host Agent.

- [ ] **Step 7: Run UI, MCP App, and full tests**

  Run: `npm run ui:check && npm run ui:build && .venv/bin/python -m pytest tests/test_mcp_app.py tests/test_mcp_server.py -v && .venv/bin/python -m pytest -q`

  Expected: all tests pass.

- [ ] **Step 8: Commit standards and evidence workflows**

  Commit: `feat: add standards and evidence views`

## Task 7: Packaging, Documentation, and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `.codex-plugin/plugin.json` only if the installed Codex build requires explicit extension metadata beyond MCP tool metadata.
- Create: `tests/fixtures/workspace-demo.sqlite3` only if a generated temporary fixture cannot drive the visual harness; otherwise do not commit a database.
- Modify: `.gitignore`

**Interfaces:**
- Consumes: completed Python package, MCP resource, UI bundle, and existing plugin manifest.
- Produces: reproducible install/build instructions and verified Codex side-panel behavior.

- [ ] **Step 1: Add build and installation documentation**

  Document Python install, `npm ci`, `npm run ui:build`, Python tests, UI tests, MCP server start, plugin installation, local database location, and the fact that `ready_to_apply` does not submit. Keep browser login and automatic application explicitly deferred.

- [ ] **Step 2: Verify a clean build from tracked inputs**

  Run: `npm ci && npm run ui:check && npm run ui:build && .venv/bin/python -m pytest -q && git diff --check`

  Expected: all checks pass and rebuilding the UI produces no unexplained tracked diff.

- [ ] **Step 3: Verify MCP discovery and resource output**

  Start the local MCP server through `./scripts/start-mcp.sh`, list tools/resources with MCP Inspector or the repository's test client, call `open_job_workspace`, and read `ui://ai-dream-job-engine/workspace-v1.html`.

  Expected: both render tools are discoverable, the resource returns `text/html;profile=mcp-app`, and existing non-UI tools still work.

- [ ] **Step 4: Open the installed plugin in Codex and test the real workflow**

  Verify the thread-side-panel entrypoint, all three views, one stored JD, one gap disclosure, `保留`, `忽略`, eligible/ineligible `加入待申请`, rubric save plus selected-job recompute, and one claim preview confirmation. Use only synthetic or user-approved local data.

  Expected: UI values match the corresponding MCP tool output and persist after closing and reopening Codex.

- [ ] **Step 5: Time the primary decision path**

  Starting from an already analyzed JD in the default `岗位` view, have the user locate the recommendation, identify the leading gap, inspect its evidence, and choose `保留` or `忽略`.

  Expected: the path completes within 30 seconds without opening another product or reading raw JSON. Record the measured time in the verification notes; do not call the target met if the user needs assistance locating a control.

- [ ] **Step 6: Capture visual evidence at narrow and wide panel sizes**

  Use Playwright CLI against a local component harness, capture screenshots at 360×800 and 900×900, and inspect them for blank rendering, clipped text, overlapping controls, focus visibility, and evidence readability. Store temporary screenshots under `output/playwright/` and keep them untracked unless the project later adopts visual baselines.

- [ ] **Step 7: Run final privacy and repository checks**

  Run: `git status --short`, inspect the database path used by the real test, and search tracked files for private JD text, resume text, API keys, cookies, and generated SQLite files.

  Expected: only intended source/docs changes are tracked; no private data, credentials, cookies, or local database is committed.

- [ ] **Step 8: Commit documentation and verification fixes**

  Commit: `docs: document MCP App workflow`

## Official Protocol References

- MCP Apps UI and decoupled render-tool pattern: `https://developers.openai.com/plugins/build/chatgpt-ui`
- Plugin UI metadata and bridge reference: `https://developers.openai.com/plugins/reference`
- Codex/ChatGPT plugin side-panel entrypoints: `https://developers.openai.com/plugins/build/extensions`

