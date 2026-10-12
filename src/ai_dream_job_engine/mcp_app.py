"""Packaged MCP App resource and typed initial render payloads."""

from __future__ import annotations

from importlib import resources
from typing import Literal, Protocol

from pydantic import BaseModel

from ai_dream_job_engine.models import ClaimPreviewView, WorkspaceSnapshot


WORKSPACE_RESOURCE_URI = "ui://ai-dream-job-engine/workspace-v1.html"
MCP_APP_MIME_TYPE = "text/html;profile=mcp-app"
PRIVATE_PAYLOAD_META_KEY = "aiDreamJobEngine"
UI_RESOURCE_META = {
    "ui": {
        "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
        }
    }
}
UI_BUNDLE_UNAVAILABLE = (
    "ui_bundle_unavailable",
    "The local interface is unavailable; use the conversation tools.",
)


class MCPAppResourceError(RuntimeError):
    """Fixed public error for a missing or empty packaged interface."""

    def __init__(self) -> None:
        self.code, self.safe_message = UI_BUNDLE_UNAVAILABLE
        super().__init__(f"{self.code}: {self.safe_message}")


class WorkspaceAppPayload(BaseModel):
    """Server-owned initial data for the shared workspace interface."""

    view: Literal["jobs", "claim_review"]
    snapshot: WorkspaceSnapshot
    claim_preview: ClaimPreviewView | None = None


class WorkspaceReader(Protocol):
    def get_snapshot(self) -> WorkspaceSnapshot: ...

    def get_claim_preview(self, preview_id: str) -> ClaimPreviewView: ...


def load_workspace_html() -> str:
    """Load the self-contained interface bundled with the Python package."""
    try:
        document = (
            resources.files("ai_dream_job_engine")
            .joinpath("ui", "workspace-v1.html")
            .read_text(encoding="utf-8")
        )
    except Exception:
        raise MCPAppResourceError() from None
    if not document.strip():
        raise MCPAppResourceError()
    return document


def open_job_workspace(application: WorkspaceReader) -> WorkspaceAppPayload:
    """Return immutable initial data for the default jobs view."""
    return WorkspaceAppPayload(
        view="jobs",
        snapshot=application.get_snapshot(),
    )


def open_claim_review(
    application: WorkspaceReader, preview_id: str
) -> WorkspaceAppPayload:
    """Return immutable initial data for one server-issued claim preview."""
    return WorkspaceAppPayload(
        view="claim_review",
        snapshot=application.get_snapshot(),
        claim_preview=application.get_claim_preview(preview_id),
    )
