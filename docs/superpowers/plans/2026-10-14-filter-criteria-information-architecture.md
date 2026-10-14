# Filter Criteria Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standards configuration table with a four-theme job-profile summary, focused theme editing, full criteria management, and a confirmed custom-preference workflow.

**Architecture:** Keep `Rubric.dimensions` as the decision engine's effective criteria, but add user-facing metadata and a single requirement-level source of truth. Pending custom preferences live outside effective dimensions until the user resolves duplicates or conflicts. The MCP App renders a three-level information architecture and delegates natural-language structuring to the host Agent while preserving an editable local confirmation step.

**Tech Stack:** Python 3.12, Pydantic 2, FastMCP/MCP Apps, SQLite, React 19, TypeScript 5.9, Vitest, Testing Library, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-filter-criteria-information-architecture.md`

## Global Constraints

- The standards home view shows four themes only: `role_content`, `work_conditions`, `growth_team`, and `compensation_constraints`.
- Requirement level is exactly one of `ignored`, `normal`, `strong`, or `required`; `required` is the only user-facing hard-blocking state.
- Weight and hard-constraint values remain internal compatibility fields and must be derived from requirement level, never edited independently in the primary UI.
- A decision topic has at most one effective criterion; unresolved drafts never affect scoring.
- Judgment fields, numeric weights, thresholds, keywords, and internal IDs stay out of the primary UI.
- Criteria affect “Do I want it?” only; candidate evidence and JD requirements continue to determine “Can I prove fit?”.
- No new dependency, cloud account, or direct external-model call is introduced.
- Custom preference text is sent to the host Agent only after the user clicks the explicit structure action.
- Existing stored rubrics must load without manual migration.

## Review Focus

- A legacy stored rubric containing only `weight` and `hard_constraint` loads with the correct requirement level and no data loss; Task 1 tests this migration.
- Twenty or more effective criteria do not render as twenty rows on the standards home view; Task 4 tests the four-theme summary.
- Two custom preferences for the same topic cannot silently become effective together; Task 2 tests duplicate and direct-conflict handling.
- A host Agent failure leaves the user's text intact and offers manual structured confirmation; Task 5 tests the fallback.
- Narrow MCP App widths do not cause horizontal page scrolling or hide the edit action; Task 4 adds a responsive DOM/CSS regression assertion and final screenshot verification.

---

### Task 1: Make Requirement Level the User-Facing Source of Truth

**Files:**
- Modify: `src/ai_dream_job_engine/models.py`
- Modify: `src/ai_dream_job_engine/rubric.py`
- Modify: `src/ai_dream_job_engine/decision.py`
- Modify: `src/ai_dream_job_engine/templates/ai_job_v1.json`
- Modify: `tests/test_models.py`
- Modify: `tests/test_rubric.py`
- Modify: `tests/test_decision.py`

**Interfaces:**
- Produces: `CriterionCategory`, `RequirementLevel`, and `CriterionOrigin` string enums.
- Produces: `ValueDimension.topic_id`, `category`, `desired_value`, `requirement_level`, and `origin`.
- Produces: `requirement_weight(level: RequirementLevel) -> float` and `requirement_is_hard(level: RequirementLevel) -> bool` in `rubric.py`.
- Preserves: legacy `weight` and `hard_constraint` fields as validated internal projections for decision and stored-data compatibility.

- [ ] **Step 1: Write failing model and migration tests**

Add tests asserting:

```python
assert ValueDimension.model_validate(legacy_payload).requirement_level == "required"
assert ValueDimension.model_validate({**legacy_payload, "weight": 0, "hard_constraint": False}).requirement_level == "ignored"
assert ValueDimension.model_validate(new_payload).weight == 1.5
assert ValueDimension.model_validate(new_payload).hard_constraint is False
```

Also assert mismatched explicit internal fields are rejected instead of silently choosing one.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_models.py tests/test_rubric.py tests/test_decision.py -q`

Expected: FAIL because the enums, metadata fields, and migration rules do not exist.

- [ ] **Step 3: Add the enum and compatibility contracts**

In `models.py`, add:

```python
class CriterionCategory(StrEnum):
    ROLE_CONTENT = "role_content"
    WORK_CONDITIONS = "work_conditions"
    GROWTH_TEAM = "growth_team"
    COMPENSATION_CONSTRAINTS = "compensation_constraints"

class RequirementLevel(StrEnum):
    IGNORED = "ignored"
    NORMAL = "normal"
    STRONG = "strong"
    REQUIRED = "required"

class CriterionOrigin(StrEnum):
    TEMPLATE = "template"
    CUSTOM = "custom"
```

Extend `ValueDimension` with the exact produced fields above. Add a `mode="before"` validator that derives missing metadata from legacy values, then an `after` validator that enforces the mapping `ignored=(0,false)`, `normal=(1,false)`, `strong=(1.5,false)`, and `required=(1,true)`.

- [ ] **Step 4: Add helpers and migrate the bundled template**

Implement the two helper signatures in `rubric.py`. Add explicit metadata to all eight bundled dimensions, using each current dimension ID as its `topic_id`; assign the approved four categories and concise Chinese-ready desired values. Keep the current scoring direction, references, thresholds, keywords, and explanation templates unchanged.

- [ ] **Step 5: Update decision reads without changing score behavior**

Use the validated internal projections in `decision.py`; add regression assertions that ignored dimensions contribute nothing, strong dimensions contribute weight `1.5`, and required dimensions still produce hard blockers through their referenced candidate-comparison facts.

- [ ] **Step 6: Run the focused tests**

Run: `python -m pytest tests/test_models.py tests/test_rubric.py tests/test_decision.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ai_dream_job_engine/models.py src/ai_dream_job_engine/rubric.py src/ai_dream_job_engine/decision.py src/ai_dream_job_engine/templates/ai_job_v1.json tests/test_models.py tests/test_rubric.py tests/test_decision.py
git commit -m "refactor: unify criterion requirement levels"
```

### Task 2: Add Pending Custom Preferences and Deterministic Conflict Resolution

**Files:**
- Create: `src/ai_dream_job_engine/preferences.py`
- Modify: `src/ai_dream_job_engine/models.py`
- Modify: `src/ai_dream_job_engine/rubric.py`
- Modify: `tests/test_rubric.py`
- Create: `tests/test_preferences.py`

**Interfaces:**
- Consumes: `CriterionCategory`, `RequirementLevel`, `CriterionOrigin`, and the effective `ValueDimension` contract from Task 1.
- Produces: `PreferenceConflictKind = none | duplicate | direct_conflict | semantic_tension | manual_review`.
- Produces: `CustomPreferenceProposal`, `PendingPreference`, and `PreferenceResolution` Pydantic models.
- Produces: `stage_preference(rubric: Rubric, proposal: CustomPreferenceProposal) -> Rubric`.
- Produces: `resolve_preference(rubric: Rubric, draft_id: str, resolution: PreferenceResolution) -> Rubric`.
- Extends: `Rubric.pending_preferences: list[PendingPreference]` and includes it in rubric revision hashing.

- [ ] **Step 1: Write failing lifecycle tests**

Cover these exact cases:

```python
def test_new_topic_stages_without_affecting_effective_dimensions(): ...
def test_same_topic_and_value_is_duplicate(): ...
def test_same_topic_and_different_value_is_direct_conflict(): ...
def test_declared_related_topic_is_semantic_tension(): ...
def test_incomplete_proposal_requires_manual_review(): ...
def test_replace_resolution_leaves_one_effective_topic_rule(): ...
def test_unresolved_draft_does_not_change_decision_result(): ...
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_preferences.py tests/test_rubric.py -q`

Expected: FAIL because preference lifecycle models and functions do not exist.

- [ ] **Step 3: Implement immutable staging and conflict classification**

In `preferences.py`, classify by normalized `topic_id` first. Equal desired value and requirement level is `duplicate`; a different value or level on the same topic is `direct_conflict`; valid `related_topic_ids` produce `semantic_tension`; missing required structure produces `manual_review`; otherwise use `none`. Never append a proposal directly to `dimensions`.

- [ ] **Step 4: Implement explicit resolutions**

Support exact actions `accept_new`, `replace_existing`, `keep_existing`, and `accept_tension`. Reject invalid action/conflict combinations. `accept_new` is valid only with no conflict, `replace_existing` removes the effective same-topic dimension first, and `accept_tension` requires distinct topic IDs. Every successful resolution removes the draft.

- [ ] **Step 5: Preserve revision and patch validation**

Update rubric construction and revision logic to retain pending drafts. Add regression tests proving an unresolved draft changes storage/revision state but not scoring inputs, while resolving it changes the effective rubric revision.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_preferences.py tests/test_rubric.py tests/test_decision.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ai_dream_job_engine/preferences.py src/ai_dream_job_engine/models.py src/ai_dream_job_engine/rubric.py tests/test_preferences.py tests/test_rubric.py
git commit -m "feat: stage and resolve custom preferences"
```

### Task 3: Expose the Preference Workflow Through MCP Tools

**Files:**
- Modify: `src/ai_dream_job_engine/mcp_server.py`
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_mcp_server_safety.py`
- Modify: `tests/test_mcp_app.py`

**Interfaces:**
- Consumes: `stage_preference` and `resolve_preference` from Task 2.
- Produces: `LocalRubricApplication.stage_custom_preference(proposal: CustomPreferenceProposal) -> Rubric`.
- Produces: `LocalRubricApplication.resolve_custom_preference(draft_id: str, resolution: PreferenceResolution) -> Rubric`.
- Produces MCP tools: `stage_custom_preference` with `model` and `app` visibility; `resolve_custom_preference` with `app` visibility.
- Preserves compare-and-swap behavior and fixed safe error messages.

- [ ] **Step 1: Write failing MCP contract and concurrency tests**

Assert the tool schemas expose only declared proposal/resolution fields, the staging tool returns private app rubric data, and two concurrent resolutions cannot silently overwrite one another. Verify malformed proposals return fixed errors without echoing private preference text.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_mcp_server.py tests/test_mcp_server_safety.py tests/test_mcp_app.py -q`

Expected: FAIL because the tools and application methods do not exist.

- [ ] **Step 3: Add application methods using the existing rubric lock and CAS loop**

Follow `update_rubric`'s storage pattern. Do not create a second persistence path or a new table; pending preferences remain part of the rubric singleton.

- [ ] **Step 4: Register the two MCP tools**

The model-visible staging tool description must tell the host Agent to structure only the user's supplied preference into `topic_id`, category, name, desired value, requirement level, evidence guidance, and optional related topic IDs. It must not infer candidate facts or inspect Agent memory.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_mcp_server.py tests/test_mcp_server_safety.py tests/test_mcp_app.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ai_dream_job_engine/mcp_server.py tests/test_mcp_server.py tests/test_mcp_server_safety.py tests/test_mcp_app.py
git commit -m "feat: expose custom preference workflow"
```

### Task 4: Build the Three-Level Criteria Workspace

**Files:**
- Create: `ui/src/criteria.ts`
- Create: `ui/src/components/CriteriaOverview.tsx`
- Create: `ui/src/components/CriteriaEditor.tsx`
- Modify: `ui/src/components/RubricView.tsx`
- Modify: `ui/src/contracts.ts`
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/styles.css`
- Create: `ui/src/test/criteria.test.ts`
- Rewrite: `ui/src/test/RubricView.test.tsx`
- Modify: `ui/src/test/App.test.tsx`

**Interfaces:**
- Consumes: rubric criterion metadata and pending preferences from Tasks 1–3.
- Produces: `CRITERION_CATEGORIES`, `requirementLabel`, `summarizeCategory`, and `dimensionPatchForRequirement` in `ui/src/criteria.ts`.
- Produces: `CriteriaOverview` for four summaries and actionable pending count.
- Produces: `CriteriaEditor` with modes `{ kind: "category"; category } | { kind: "all" }`.
- Keeps: `RubricView` responsible for loading, draft state, validation, save, and recompute only.

- [ ] **Step 1: Write failing pure view-model tests**

Assert four categories always render in the approved order, empty categories show `尚未设置`, summaries show at most three desired values plus `+N`, and requirement changes produce exactly the internal weight/hard projection from Task 1.

- [ ] **Step 2: Write failing component tests for 20+ criteria**

Render 24 dimensions and assert the home view contains exactly four theme edit buttons, no criteria table, no “判断字段” tab, and one `查看全部 24 条标准` action. Assert the primary navigation label is `筛选标准`. Open one category and assert only that category's rows appear with columns `标准 / 我的选择 / 要求程度`, a custom-origin badge when applicable, and an expandable `系统如何判断` disclosure. Change one requirement level and verify the rule is editable within two clicks from the overview.

- [ ] **Step 3: Run the focused UI tests and verify they fail**

Run: `npm run ui:test -- --run ui/src/test/criteria.test.ts ui/src/test/RubricView.test.tsx ui/src/test/App.test.tsx`

Expected: FAIL because the new view model and components do not exist.

- [ ] **Step 4: Implement the pure criteria adapter**

Keep mapping and summary logic out of React. `dimensionPatchForRequirement(level)` returns `requirement_level`, `weight`, and `hard_constraint` together so the UI cannot produce contradictory state.

- [ ] **Step 5: Implement overview and focused editing**

Rename the primary navigation item to `筛选标准`. The home screen title is `我的求职画像`; the top action is `添加偏好`. Category editing and all-criteria management use the same row component and dropdowns. The all view adds search plus filters for category, requirement level, custom origin, and pending items. Each rule may disclose `系统如何判断`, showing evidence guidance, unknown handling, and scoring impact without raw IDs, numeric weights, or thresholds. Judgment fields and recommendation thresholds are removed from this user-facing view, not deleted from the backend.

- [ ] **Step 6: Add narrow-width styles and accessibility behavior**

Use full-width unframed sections, stable grid tracks, no nested cards, visible focus states, labels for icon buttons, and a single-column row fallback below 560px. Truncated summaries expose the full text through an accessible label or description. Add a DOM assertion that the app root and criteria workspace do not opt into horizontal overflow.

- [ ] **Step 7: Run focused UI tests**

Run: `npm run ui:test -- --run ui/src/test/criteria.test.ts ui/src/test/RubricView.test.tsx ui/src/test/App.test.tsx`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add ui/src/criteria.ts ui/src/components/CriteriaOverview.tsx ui/src/components/CriteriaEditor.tsx ui/src/components/RubricView.tsx ui/src/contracts.ts ui/src/App.tsx ui/src/styles.css ui/src/test/criteria.test.ts ui/src/test/RubricView.test.tsx ui/src/test/App.test.tsx
git commit -m "feat: add job profile criteria workspace"
```

### Task 5: Complete the Custom Preference Confirmation Loop

**Files:**
- Create: `ui/src/components/CustomPreferenceDialog.tsx`
- Create: `ui/src/components/PendingPreferenceReview.tsx`
- Modify: `ui/src/components/RubricView.tsx`
- Modify: `ui/src/contracts.ts`
- Modify: `ui/src/styles.css`
- Modify: `ui/src/test/RubricView.test.tsx`
- Create: `ui/src/test/CustomPreferenceDialog.test.tsx`

**Interfaces:**
- Consumes: `McpAppBridge.sendMessage`, `stage_custom_preference`, and `resolve_custom_preference`.
- Produces: editable proposal form fields `statement`, `topic_id`, `category`, `name`, `desired_value`, `requirement_level`, `evidence_guidance`, and `related_topic_ids`.
- Produces: conflict-specific actions matching Task 2's exact resolution strings.
- Preserves user input on Agent failure and never makes a draft effective before confirmation.

- [ ] **Step 1: Write failing interaction tests**

Cover these flows:

```text
natural-language input -> explicit "整理偏好" -> sendMessage called once
Agent/tool result -> editable structured pending draft appears
duplicate -> offers modify existing or discard, not a second effective rule
direct conflict -> requires keep-existing or replace-existing
semantic tension -> warning plus explicit accept-tension
Agent failure -> original text remains and manual fields become editable
manual-review draft -> cannot confirm until required fields are complete
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `npm run ui:test -- --run ui/src/test/CustomPreferenceDialog.test.tsx ui/src/test/RubricView.test.tsx`

Expected: FAIL because the components do not exist.

- [ ] **Step 3: Implement explicit Agent handoff**

On `整理偏好`, send one bounded message instructing the host Agent to call `stage_custom_preference` with no information beyond the user's statement and currently visible criterion topic metadata. Do not auto-send on typing, opening, or blur.

- [ ] **Step 4: Implement editable preview and conflict actions**

Render structured fields before resolution. Call `resolve_custom_preference` only from an explicit user action, refresh the rubric from the returned value, and close the dialog only after success.

- [ ] **Step 5: Implement failure and pending affordances**

If Agent handoff fails or times out, keep the original statement and reveal the same structured fields for manual completion. A completed manual form calls `stage_custom_preference` directly from the app, then follows the same confirmation path. The overview shows only an actionable `N 项需要处理` link when pending drafts exist.

- [ ] **Step 6: Run focused UI tests**

Run: `npm run ui:test -- --run ui/src/test/CustomPreferenceDialog.test.tsx ui/src/test/RubricView.test.tsx`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ui/src/components/CustomPreferenceDialog.tsx ui/src/components/PendingPreferenceReview.tsx ui/src/components/RubricView.tsx ui/src/contracts.ts ui/src/styles.css ui/src/test/CustomPreferenceDialog.test.tsx ui/src/test/RubricView.test.tsx
git commit -m "feat: add confirmed custom preference flow"
```

### Task 6: Verify, Bundle, and Document the Shipped Workspace

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-14-filter-criteria-information-architecture.md`
- Regenerate: `src/ai_dream_job_engine/ui/workspace-v1.html`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: a built MCP App bundle matching source and a short user-facing README description of the three-level criteria workflow.

- [ ] **Step 1: Run all backend tests**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run all frontend checks**

Run: `npm run ui:check`

Expected: TypeScript passes and all Vitest tests pass.

- [ ] **Step 3: Build the production MCP App bundle**

Run: `npm run ui:build`

Expected: `src/ai_dream_job_engine/ui/workspace-v1.html` is regenerated without errors.

- [ ] **Step 4: Verify the real UI at wide and narrow widths**

Open the MCP App using the project tool, capture desktop and narrow screenshots, and verify: four summaries on the home view; no 20-row list on entry; category edit is reachable; custom preference remains pending until confirmation; no overlap or horizontal page scroll.

- [ ] **Step 5: Update documentation and status**

Document the new workflow in `README.md` and change the spec status to `已实现并验证` only after Steps 1–4 pass.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/superpowers/specs/2026-10-14-filter-criteria-information-architecture.md src/ai_dream_job_engine/ui/workspace-v1.html
git commit -m "docs: document criteria workspace"
```

- [ ] **Step 7: Review the final branch diff**

Run: `git diff b424155..HEAD --stat && git status --short`

Expected: only criterion-domain, MCP workflow, MCP App UI, tests, bundle, and documentation changes; working tree clean.
