# Final Whole-Branch Review Fixes

## Implementation commit

- `1634d4e634f8a89408c9ac2d42aad479c77a9331` - `fix: harden private workspace state`

## Fixed

1. MCP App hydration is private by contract. Render and app-only tools place full UI payloads under `_meta.aiDreamJobEngine.payload`; model-visible `structuredContent` contains only a non-sensitive status summary. Tests assert exact tool visibility metadata and the absence of confirmed or candidate claims from model-visible output.
2. Workspace staleness is server-authoritative. Stored analyses are checked against the current rubric digest and confirmed candidate evidence references on every snapshot, detail read, and review update. Missing or changed inputs expose stale reasons, invalidate prior ready state after restart, and block direct `ready_to_apply` calls.
3. Evidence is readable in the local app. Successful analyses retain only cited JD excerpts, never an entire supplied JD, and candidate references resolve from the current confirmed local profile. Workspace detail returns readable evidence while model-facing analysis records keep opaque references.
4. Agent Memory authorization is persisted separately from imported claims. Disconnecting revokes future source sync; confirmed imports can be retained or explicitly deleted. The UI lists active sources from persisted connection state rather than inferring connections from retained claims.
5. Profile and rubric singleton writes use SQLite revision CAS. Concurrent distinct profile confirmations and rubric edits retry and merge. Concurrent writes to the same rubric field return `rubric_update_conflict` instead of silently overwriting.

## Verification

- Focused backend regression: `182 passed`
- Full Python suite: `342 passed in 3.43s`
- UI typecheck and tests: `5 files passed`, `31 tests passed`
- Production UI build: successful, self-contained `workspace-v1.html`
- `git diff --check`: passed
- Privacy scan: no known private fixture markers in production Python/UI bundle; no direct claim/profile/candidate reads from model-visible structured content

## Residual concerns

- Readable JD evidence is intentionally limited to cited excerpts stored in the local SQLite database. Full JD text remains excluded, but users should still treat the local database as private job-search data.
- Agent Memory becomes connected when an explicitly scoped preview is requested. Claims still require per-item confirmation before entering the candidate profile.
