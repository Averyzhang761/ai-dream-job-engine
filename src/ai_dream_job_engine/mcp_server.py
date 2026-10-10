"""Privacy-safe stdio MCP adapter for the decision engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Protocol

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.models import (
    AnalysisRecord,
    AnalysisSummary,
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    DecisionTrace,
    MemoryImportPreview,
    Rubric,
    RubricPatch,
)
from ai_dream_job_engine.profile import (
    confirm_claims,
    disconnect_memory_source,
    preview_memory_import,
)
from ai_dream_job_engine.providers.base import ProviderError
from ai_dream_job_engine.providers.jev import JevSignalProvider
from ai_dream_job_engine.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from ai_dream_job_engine.rubric import apply_rubric_patch, load_default_rubric
from ai_dream_job_engine.service import AnalysisServiceError, JobAnalysisService


NonEmptyString = Annotated[str, Field(min_length=1)]
NonEmptyStrings = Annotated[list[str], Field(min_length=1)]


class ProfileApplication(Protocol):
    def initialize_profile(self, profile: CandidateProfile) -> CandidateProfile: ...

    def review_profile_claims(self) -> CandidateProfile: ...

    def confirm_profile_claims(
        self, decisions: list[ClaimDecision]
    ) -> CandidateProfile: ...

    def preview_agent_memory_import(
        self,
        source_agent: str,
        scope: list[str],
        candidates: list[CandidateClaim],
    ) -> MemoryImportPreview: ...

    def confirm_agent_memory_import(
        self,
        preview: MemoryImportPreview,
        decisions: list[ClaimDecision],
    ) -> CandidateProfile: ...

    def disconnect_agent_memory(
        self, source_agent: str, delete_imported: bool
    ) -> CandidateProfile: ...


class RubricApplication(Protocol):
    def get_rubric(self) -> Rubric: ...

    def update_rubric(self, patch: RubricPatch) -> Rubric: ...


class AnalysisApplication(Protocol):
    def analyze_job(self, job_description: str) -> AnalysisRecord: ...

    def recompute_decision(self, analysis_id: str) -> AnalysisRecord: ...

    def get_decision_trace(self, analysis_id: str) -> DecisionTrace: ...

    def list_analyzed_jobs(self) -> list[AnalysisSummary]: ...


@dataclass(frozen=True)
class MCPServices:
    profile: ProfileApplication
    rubric: RubricApplication
    analysis: AnalysisApplication


class _LazyLocalStore:
    """Initialize the caller-selected local database only on first use."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._store: LocalStore | None = None
        self._lock = Lock()

    def _ready(self) -> LocalStore:
        with self._lock:
            if self._store is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                store = LocalStore.open(self._path)
                store.initialize()
                self._store = store
        return self._store

    def save_profile(self, profile: CandidateProfile) -> None:
        self._ready().save_profile(profile)

    def load_profile(self) -> CandidateProfile | None:
        return self._ready().load_profile()

    def save_rubric(self, rubric: Rubric) -> None:
        self._ready().save_rubric(rubric)

    def load_rubric(self) -> Rubric | None:
        return self._ready().load_rubric()

    def save_analysis(self, record: AnalysisRecord) -> None:
        self._ready().save_analysis(record)

    def insert_analysis(self, record: AnalysisRecord) -> None:
        self._ready().insert_analysis(record)

    def list_analyses(self) -> list[AnalysisRecord]:
        return self._ready().list_analyses()


class LocalProfileApplication:
    def __init__(self, store: _LazyLocalStore) -> None:
        self._store = store

    def initialize_profile(self, profile: CandidateProfile) -> CandidateProfile:
        self._store.save_profile(profile)
        return profile

    def review_profile_claims(self) -> CandidateProfile:
        return self._profile()

    def confirm_profile_claims(
        self, decisions: list[ClaimDecision]
    ) -> CandidateProfile:
        updated = confirm_claims(self._profile(), decisions)
        self._store.save_profile(updated)
        return updated

    def preview_agent_memory_import(
        self,
        source_agent: str,
        scope: list[str],
        candidates: list[CandidateClaim],
    ) -> MemoryImportPreview:
        if not source_agent.strip() or not scope or any(not item.strip() for item in scope):
            raise ValueError("memory source and scope are required")
        return preview_memory_import(
            self._profile(), source_agent, scope, candidates
        )

    def confirm_agent_memory_import(
        self,
        preview: MemoryImportPreview,
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        if (
            not preview.source_agent.strip()
            or not preview.scope
            or any(not item.strip() for item in preview.scope)
        ):
            raise ValueError("a complete memory preview is required")
        updated = confirm_claims(
            self._profile(), decisions, preview=preview
        )
        self._store.save_profile(updated)
        return updated

    def disconnect_agent_memory(
        self, source_agent: str, delete_imported: bool
    ) -> CandidateProfile:
        updated = disconnect_memory_source(
            self._profile(), source_agent, delete_imported
        )
        self._store.save_profile(updated)
        return updated

    def _profile(self) -> CandidateProfile:
        profile = self._store.load_profile()
        if profile is None:
            raise ValueError("profile is not initialized")
        return profile


class LocalRubricApplication:
    def __init__(self, store: _LazyLocalStore) -> None:
        self._store = store

    def get_rubric(self) -> Rubric:
        rubric = self._store.load_rubric()
        if rubric is None:
            rubric = load_default_rubric()
            self._store.save_rubric(rubric)
        return rubric

    def update_rubric(self, patch: RubricPatch) -> Rubric:
        updated = apply_rubric_patch(self.get_rubric(), patch)
        self._store.save_rubric(updated)
        return updated


class ConfiguredAnalysisApplication:
    def __init__(self, store: _LazyLocalStore) -> None:
        self._store = store

    def analyze_job(self, job_description: str) -> AnalysisRecord:
        profile = self._store.load_profile()
        rubric = self._store.load_rubric()
        if profile is None or rubric is None:
            return self._service().analyze_job(job_description)
        try:
            facts = self._fact_provider() if rubric.facts else None
            signals = self._signal_provider() if rubric.questions else None
        except (ProviderError, ValueError):
            raise AnalysisServiceError(
                "provider_not_configured",
                "Selected analysis provider is not configured.",
            ) from None
        return self._service(facts, signals).analyze_job(job_description)

    def recompute_decision(self, analysis_id: str) -> AnalysisRecord:
        return self._service().recompute_decision(analysis_id)

    def get_decision_trace(self, analysis_id: str) -> DecisionTrace:
        return self._service().get_decision_trace(analysis_id)

    def list_analyzed_jobs(self) -> list[AnalysisSummary]:
        return self._service().list_analyzed_jobs()

    def _service(self, facts=None, signals=None) -> JobAnalysisService:
        return JobAnalysisService(
            self._store,
            facts or _UnavailableFactProvider(),
            signals or _UnavailableSignalProvider(),
        )

    @staticmethod
    def _fact_provider() -> OpenAICompatibleProvider:
        provider = os.getenv("AI_DREAM_JOB_FACT_PROVIDER", "openai")
        if provider not in {"openai", "openrouter"}:
            raise ValueError("unsupported provider")
        key_name = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        return OpenAICompatibleProvider(
            provider=provider,
            api_key=os.getenv(key_name, ""),
            model=os.getenv("AI_DREAM_JOB_FACT_MODEL", ""),
        )

    @staticmethod
    def _signal_provider() -> JevSignalProvider:
        return JevSignalProvider(api_key=os.getenv("TYPESAFE_API_KEY"))


class _UnavailableFactProvider:
    def extract(self, job_description, rubric):
        raise ProviderError("fact provider is unavailable")


class _UnavailableSignalProvider:
    def judge(self, job_description, questions):
        raise ProviderError("signal provider is unavailable")


def default_services() -> MCPServices:
    project_root = Path(__file__).resolve().parents[2]
    configured_path = os.getenv("AI_DREAM_JOB_DATA_PATH")
    database_path = (
        Path(configured_path).expanduser()
        if configured_path
        else project_root / ".data" / "ai-dream-job-engine.sqlite3"
    )
    store = _LazyLocalStore(database_path)
    return MCPServices(
        profile=LocalProfileApplication(store),
        rubric=LocalRubricApplication(store),
        analysis=ConfiguredAnalysisApplication(store),
    )


def _call_safely(label: str, operation):
    try:
        return operation()
    except AnalysisServiceError as error:
        raise ToolError(str(error)) from None
    except Exception:
        raise ToolError(f"{label} failed safely.") from None


def build_server(services: MCPServices | None = None) -> MCPServer:
    applications = services or default_services()
    server = MCPServer(
        "ai-dream-job-engine",
        description="Local-first, evidence-backed job decision engine.",
        version="0.1.0",
        log_level="ERROR",
    )

    @server.tool()
    def initialize_profile(profile: CandidateProfile) -> CandidateProfile:
        """Initialize the local candidate profile from host-supplied claims."""
        return _call_safely(
            "Profile initialization",
            lambda: applications.profile.initialize_profile(profile),
        )

    @server.tool()
    def review_profile_claims() -> CandidateProfile:
        """Review current claims before confirming or editing them."""
        return _call_safely(
            "Profile review", applications.profile.review_profile_claims
        )

    @server.tool()
    def confirm_profile_claims(
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        """Apply explicit per-claim user decisions."""
        return _call_safely(
            "Claim confirmation",
            lambda: applications.profile.confirm_profile_claims(decisions),
        )

    @server.tool()
    def preview_agent_memory_import(
        source_agent: NonEmptyString,
        scope: NonEmptyStrings,
        candidates: list[CandidateClaim],
    ) -> MemoryImportPreview:
        """Preview explicit host-supplied memory candidates without mutation."""
        return _call_safely(
            "Memory preview",
            lambda: applications.profile.preview_agent_memory_import(
                source_agent, scope, candidates
            ),
        )

    @server.tool()
    def confirm_agent_memory_import(
        preview: MemoryImportPreview,
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        """Confirm selected claims from a prior memory preview."""
        return _call_safely(
            "Memory confirmation",
            lambda: applications.profile.confirm_agent_memory_import(
                preview, decisions
            ),
        )

    @server.tool()
    def disconnect_agent_memory(
        source_agent: NonEmptyString,
        delete_imported: bool,
    ) -> CandidateProfile:
        """Disconnect a source and optionally delete its confirmed imports."""
        return _call_safely(
            "Memory disconnect",
            lambda: applications.profile.disconnect_agent_memory(
                source_agent, delete_imported
            ),
        )

    @server.tool()
    def get_rubric() -> Rubric:
        """Return the active local decision rubric."""
        return _call_safely("Rubric read", applications.rubric.get_rubric)

    @server.tool()
    def update_rubric(patch: RubricPatch) -> Rubric:
        """Apply a validated declarative patch to the active rubric."""
        return _call_safely(
            "Rubric update", lambda: applications.rubric.update_rubric(patch)
        )

    @server.tool()
    def analyze_job(job_description: NonEmptyString) -> AnalysisRecord:
        """Analyze host-supplied public job text with configured providers."""
        return _call_safely(
            "Job analysis",
            lambda: applications.analysis.analyze_job(job_description),
        )

    @server.tool()
    def recompute_decision(analysis_id: NonEmptyString) -> AnalysisRecord:
        """Recompute a stored decision without another provider call."""
        return _call_safely(
            "Decision recompute",
            lambda: applications.analysis.recompute_decision(analysis_id),
        )

    @server.tool()
    def get_decision_trace(analysis_id: NonEmptyString) -> DecisionTrace:
        """Return the validated structured trace for a stored decision."""
        return _call_safely(
            "Decision trace",
            lambda: applications.analysis.get_decision_trace(analysis_id),
        )

    @server.tool()
    def list_analyzed_jobs() -> list[AnalysisSummary]:
        """List structured summaries of successful local analyses."""
        return _call_safely(
            "Analysis listing", applications.analysis.list_analyzed_jobs
        )

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
