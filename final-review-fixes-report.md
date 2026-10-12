# Final Whole-Branch Review Fixes

## Implementation commit

- `1634d4e634f8a89408c9ac2d42aad479c77a9331` - `fix: harden private workspace state`
- `2658db2a0320747529d28517b6169dc920e4fec5` - `fix: close final privacy replay gaps`

## Fixed

1. MCP App hydration is private by contract. Render and app-only tools place full UI payloads under `_meta.aiDreamJobEngine.payload`; model-visible `structuredContent` contains only a non-sensitive status summary. Tests assert exact tool visibility metadata and the absence of confirmed or candidate claims from model-visible output.
2. Workspace staleness is server-authoritative. Stored analyses are checked against the current rubric digest and confirmed candidate evidence references on every snapshot, detail read, and review update. Missing or changed inputs expose stale reasons, invalidate prior ready state after restart, and block direct `ready_to_apply` calls.
3. Evidence is readable in the local app. Successful analyses retain only cited JD excerpts, never an entire supplied JD, and candidate references resolve from the current confirmed local profile. Workspace detail returns readable evidence while model-facing analysis records keep opaque references.
4. Agent Memory authorization is persisted separately from imported claims. Disconnecting revokes future source sync; confirmed imports can be retained or explicitly deleted. The UI lists active sources from persisted connection state rather than inferring connections from retained claims.
5. Profile and rubric singleton writes use SQLite revision CAS. Concurrent distinct profile confirmations and rubric edits retry and merge. Concurrent writes to the same rubric field return `rubric_update_conflict` instead of silently overwriting.
6. Resume and Agent Memory preview tools remain available for Agent orchestration, but model-visible results contain only status, preview ID, and candidate count. Candidate values are available only to the App through `_meta.aiDreamJobEngine.payload`.
7. Disconnecting Agent Memory invalidates every outstanding preview for that source. Confirmation rechecks persisted authorization during each CAS attempt, so a disconnect committed by another application instance wins over an overlapping confirmation and cannot silently reconnect the source.
8. A full-JD evidence reference that cannot be resolved locally is rendered as `完整 JD 原文未持久化，请重新打开来源核对`; the local App never receives the opaque hash as display text.

## Verification

- Focused disconnect/concurrency regression: `32 passed`
- Full Python suite: `348 passed in 3.37s`
- UI typecheck and tests: `5 files passed`, `32 tests passed`
- Production UI build: successful, self-contained `workspace-v1.html`
- `git diff --check`: passed
- Privacy scan: no known private fixture markers in production Python/UI bundle; no direct claim/profile/candidate reads from model-visible structured content

## Residual concerns

- Readable JD evidence is intentionally limited to cited excerpts stored in the local SQLite database. Full JD text remains excluded, but users should still treat the local database as private job-search data.
- Agent Memory becomes connected when an explicitly scoped preview is requested. Claims still require per-item confirmation before entering the candidate profile.
