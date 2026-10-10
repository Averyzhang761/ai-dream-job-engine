---
name: ai-dream-job
description: Review candidate evidence and make traceable job decisions.
---

# AI Dream Job

Keep the conversation concise and evidence-backed.

1. Explain local-first privacy before profile setup. Ask consent before any
   Agent memory import, accept only host-supplied candidates, show the preview,
   and require per-claim confirmation.
2. Never make unsupported capability claims. Label each issue as an unknown or
   evidence gap, or as a capability gap, based only on confirmed evidence.
3. External models receive the public job description only by default. Do not
   send resumes, profile claims, memory, credentials, or personal files.
4. This plugin does not auto-apply. Require explicit user confirmation before
   any future application action.
5. The host Agent owns schedules and notifications. Use quiet `no_change` when
   nothing actionable changed.

Job discovery and Gmail status sync are reserved for future coarse workflows.
Do not imply that discovery, Gmail sync, scheduling, notifications, or job
application actions are implemented by this plugin.
