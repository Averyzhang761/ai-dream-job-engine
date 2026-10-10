"""Privacy-safe stdio MCP adapter for the decision engine."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Annotated, Callable, Protocol

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

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

_DATA_PATH_ENV = "AI_DREAM_JOB_DATA_PATH"
_FACT_PROVIDER_ENV = "AI_DREAM_JOB_FACT_PROVIDER"
_FACT_MODEL_ENV = "AI_DREAM_JOB_FACT_MODEL"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"
_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
_JEV_KEY_ENV = "TYPESAFE_API_KEY"
APPLICATION_CONFIG_ENV_VARS = frozenset(
    {
        _DATA_PATH_ENV,
        _FACT_PROVIDER_ENV,
        _FACT_MODEL_ENV,
        _OPENAI_KEY_ENV,
        _OPENROUTER_KEY_ENV,
        _JEV_KEY_ENV,
    }
)

PROFILE_NOT_INITIALIZED = (
    "profile_not_initialized",
    "Initialize the candidate profile before this action.",
)
MEMORY_PREVIEW_INVALID = (
    "memory_preview_invalid",
    "Request a new memory preview before confirming claims.",
)
INVALID_CLAIM_DECISIONS = (
    "invalid_claim_decisions",
    "Review the claim decisions and submit valid per-claim actions.",
)
INVALID_RUBRIC_PATCH = (
    "invalid_rubric_patch",
    "Review the rubric patch and submit valid rubric changes.",
)
STORAGE_FAILURE = (
    "storage_failure",
    "Local data could not be saved or loaded; retry the action.",
)
ANALYSIS_PROVIDER_CONFIG = (
    "analysis_provider_config",
    "Configure the selected analysis provider before analyzing a job.",
)
RUBRIC_NOT_INITIALIZED = (
    "rubric_not_initialized",
    "Initialize the decision rubric before analyzing a job.",
)
ANALYSIS_NOT_FOUND = (
    "analysis_not_found",
    "Choose an existing successful analysis and retry.",
)
ANALYSIS_UNAVAILABLE = (
    "analysis_unavailable",
    "Choose a successful analysis before this action.",
)
ANALYSIS_FAILED = (
    "analysis_failed",
    "Job analysis failed safely; verify provider availability and retry.",
)
INVALID_JOB_DESCRIPTION = (
    "invalid_job_description",
    "Provide a non-empty public job description.",
)
TRACE_INTEGRITY_FAILURE = (
    "trace_integrity_failure",
    "Stored decision trace could not be verified; rerun the analysis.",
)
ANALYSIS_FAILURE = (
    "analysis_failure",
    "The analysis action could not be completed safely.",
)
INTERNAL_FAILURE = (
    "internal_failure",
    "The action could not be completed safely; retry it.",
)


class ApplicationError(RuntimeError):
    """A fixed public error that cannot carry source exception context."""

    def __init__(self, public_error: tuple[str, str]) -> None:
        self.code, self.safe_message = public_error
        super().__init__(f"{self.code}: {self.safe_message}")


class IssuedMemoryImportPreview(BaseModel):
    preview_id: str = Field(min_length=32)
    preview: MemoryImportPreview


@dataclass(frozen=True)
class _RegisteredPreview:
    profile_digest: str
    binding_digest: str
    preview: MemoryImportPreview


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
    ) -> IssuedMemoryImportPreview: ...

    def confirm_agent_memory_import(
        self,
        preview_id: str,
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

    def __init__(self, path_factory: Callable[[], Path]) -> None:
        self._path_factory = path_factory
        self._store: LocalStore | None = None
        self._lock = Lock()

    def _ready(self) -> LocalStore:
        with self._lock:
            if self._store is None:
                path = self._path_factory()
                path.parent.mkdir(parents=True, exist_ok=True)
                store = LocalStore.open(path)
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
    def __init__(self, store) -> None:
        self._store = store
        self._lock = RLock()
        self._previews: dict[str, _RegisteredPreview] = {}

    def initialize_profile(self, profile: CandidateProfile) -> CandidateProfile:
        with self._lock:
            self._save_profile(profile)
            return profile

    def review_profile_claims(self) -> CandidateProfile:
        with self._lock:
            return self._profile()

    def confirm_profile_claims(
        self, decisions: list[ClaimDecision]
    ) -> CandidateProfile:
        with self._lock:
            try:
                updated = confirm_claims(self._profile(), decisions)
            except ValueError:
                raise ApplicationError(INVALID_CLAIM_DECISIONS) from None
            self._save_profile(updated)
            return updated

    def preview_agent_memory_import(
        self,
        source_agent: str,
        scope: list[str],
        candidates: list[CandidateClaim],
    ) -> IssuedMemoryImportPreview:
        with self._lock:
            if (
                not source_agent.strip()
                or not scope
                or any(not item.strip() for item in scope)
            ):
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            profile = self._profile()
            try:
                preview = preview_memory_import(
                    profile, source_agent, scope, candidates
                )
            except ValueError:
                raise ApplicationError(MEMORY_PREVIEW_INVALID) from None
            preview_id = self._new_preview_id()
            registered_preview = preview.model_copy(deep=True)
            self._previews[preview_id] = _RegisteredPreview(
                profile_digest=_model_digest(profile),
                binding_digest=_preview_binding_digest(
                    profile, registered_preview
                ),
                preview=registered_preview,
            )
            return IssuedMemoryImportPreview(
                preview_id=preview_id,
                preview=preview.model_copy(deep=True),
            )

    def confirm_agent_memory_import(
        self,
        preview_id: str,
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        with self._lock:
            issued = self._previews.get(preview_id)
            if issued is None:
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            profile = self._profile()
            if _model_digest(profile) != issued.profile_digest:
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            try:
                reproduced = preview_memory_import(
                    profile,
                    issued.preview.source_agent,
                    issued.preview.scope,
                    issued.preview.candidates,
                )
            except ValueError:
                raise ApplicationError(MEMORY_PREVIEW_INVALID) from None
            if (
                _preview_binding_digest(profile, reproduced)
                != issued.binding_digest
            ):
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            try:
                updated = confirm_claims(
                    profile, decisions, preview=issued.preview
                )
            except ValueError:
                raise ApplicationError(INVALID_CLAIM_DECISIONS) from None
            self._save_profile(updated)
            del self._previews[preview_id]
            return updated

    def disconnect_agent_memory(
        self, source_agent: str, delete_imported: bool
    ) -> CandidateProfile:
        with self._lock:
            try:
                updated = disconnect_memory_source(
                    self._profile(), source_agent, delete_imported
                )
            except ValueError:
                raise ApplicationError(INVALID_CLAIM_DECISIONS) from None
            self._save_profile(updated)
            return updated

    def _profile(self) -> CandidateProfile:
        try:
            profile = self._store.load_profile()
        except Exception:
            raise ApplicationError(STORAGE_FAILURE) from None
        if profile is None:
            raise ApplicationError(PROFILE_NOT_INITIALIZED)
        return profile

    def _save_profile(self, profile: CandidateProfile) -> None:
        try:
            self._store.save_profile(profile)
        except Exception:
            raise ApplicationError(STORAGE_FAILURE) from None

    def _new_preview_id(self) -> str:
        while True:
            preview_id = secrets.token_urlsafe(32)
            if preview_id not in self._previews:
                return preview_id


class LocalRubricApplication:
    def __init__(self, store) -> None:
        self._store = store
        self._lock = RLock()

    def get_rubric(self) -> Rubric:
        with self._lock:
            return self._get_rubric()

    def update_rubric(self, patch: RubricPatch) -> Rubric:
        with self._lock:
            try:
                updated = apply_rubric_patch(self._get_rubric(), patch)
            except ValueError:
                raise ApplicationError(INVALID_RUBRIC_PATCH) from None
            self._save_rubric(updated)
            return updated

    def _get_rubric(self) -> Rubric:
        try:
            rubric = self._store.load_rubric()
        except Exception:
            raise ApplicationError(STORAGE_FAILURE) from None
        if rubric is None:
            rubric = load_default_rubric()
            self._save_rubric(rubric)
        return rubric

    def _save_rubric(self, rubric: Rubric) -> None:
        try:
            self._store.save_rubric(rubric)
        except Exception:
            raise ApplicationError(STORAGE_FAILURE) from None


class ConfiguredAnalysisApplication:
    def __init__(self, store) -> None:
        self._store = store

    def analyze_job(self, job_description: str) -> AnalysisRecord:
        try:
            profile = self._store.load_profile()
            rubric = self._store.load_rubric()
        except Exception:
            raise ApplicationError(STORAGE_FAILURE) from None
        if profile is None or rubric is None:
            return self._service().analyze_job(job_description)
        try:
            facts = self._fact_provider() if rubric.facts else None
            signals = self._signal_provider() if rubric.questions else None
        except (ProviderError, ValueError):
            raise ApplicationError(ANALYSIS_PROVIDER_CONFIG) from None
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
        provider = os.getenv(_FACT_PROVIDER_ENV, "openai")
        if provider not in {"openai", "openrouter"}:
            raise ValueError("unsupported provider")
        key_name = (
            _OPENAI_KEY_ENV if provider == "openai" else _OPENROUTER_KEY_ENV
        )
        return OpenAICompatibleProvider(
            provider=provider,
            api_key=os.getenv(key_name, ""),
            model=os.getenv(_FACT_MODEL_ENV, ""),
        )

    @staticmethod
    def _signal_provider() -> JevSignalProvider:
        return JevSignalProvider(api_key=os.getenv(_JEV_KEY_ENV))


class _UnavailableFactProvider:
    def extract(self, job_description, rubric):
        raise ProviderError("fact provider is unavailable")


class _UnavailableSignalProvider:
    def judge(self, job_description, questions):
        raise ProviderError("signal provider is unavailable")


def _model_digest(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _preview_binding_digest(
    profile: CandidateProfile, preview: MemoryImportPreview
) -> str:
    payload = json.dumps(
        {
            "profile": profile.model_dump(mode="json"),
            "preview": preview.model_dump(mode="json"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _default_database_path() -> Path:
    configured_path = os.getenv(_DATA_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".data" / "ai-dream-job-engine.sqlite3"


def default_services() -> MCPServices:
    store = _LazyLocalStore(_default_database_path)
    return MCPServices(
        profile=LocalProfileApplication(store),
        rubric=LocalRubricApplication(store),
        analysis=ConfiguredAnalysisApplication(store),
    )


def _analysis_public_error(error: AnalysisServiceError) -> tuple[str, str]:
    if error.code == "missing_profile":
        return PROFILE_NOT_INITIALIZED
    if error.code == "missing_rubric":
        return RUBRIC_NOT_INITIALIZED
    if error.code == "analysis_not_found":
        return ANALYSIS_NOT_FOUND
    if error.code in {"analysis_not_successful", "trace_unavailable"}:
        return ANALYSIS_UNAVAILABLE
    if error.code == "analysis_failed":
        return ANALYSIS_FAILED
    if error.code in {"empty_job_description", "invalid_job_description"}:
        return INVALID_JOB_DESCRIPTION
    if error.code == "trace_integrity_failed":
        return TRACE_INTEGRITY_FAILURE
    if error.code == "provider_not_configured":
        return ANALYSIS_PROVIDER_CONFIG
    if error.code in {
        "storage_failed",
        "failure_audit_persistence_failed",
    }:
        return STORAGE_FAILURE
    return ANALYSIS_FAILURE


def _call_safely(operation):
    try:
        return operation()
    except ApplicationError as error:
        raise ToolError(str(error)) from None
    except AnalysisServiceError as error:
        public_error = _analysis_public_error(error)
        raise ToolError(
            f"{public_error[0]}: {public_error[1]}"
        ) from None
    except Exception:
        raise ToolError(
            f"{INTERNAL_FAILURE[0]}: {INTERNAL_FAILURE[1]}"
        ) from None


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
            lambda: applications.profile.initialize_profile(profile),
        )

    @server.tool()
    def review_profile_claims() -> CandidateProfile:
        """Review current claims before confirming or editing them."""
        return _call_safely(applications.profile.review_profile_claims)

    @server.tool()
    def confirm_profile_claims(
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        """Apply explicit per-claim user decisions."""
        return _call_safely(
            lambda: applications.profile.confirm_profile_claims(decisions),
        )

    @server.tool()
    def preview_agent_memory_import(
        source_agent: NonEmptyString,
        scope: NonEmptyStrings,
        candidates: list[CandidateClaim],
    ) -> IssuedMemoryImportPreview:
        """Preview explicit host-supplied memory candidates without mutation."""
        return _call_safely(
            lambda: applications.profile.preview_agent_memory_import(
                source_agent, scope, candidates
            ),
        )

    @server.tool()
    def confirm_agent_memory_import(
        preview_id: NonEmptyString,
        decisions: list[ClaimDecision],
    ) -> CandidateProfile:
        """Confirm selected claims from a prior memory preview."""
        return _call_safely(
            lambda: applications.profile.confirm_agent_memory_import(
                preview_id, decisions
            ),
        )

    @server.tool()
    def disconnect_agent_memory(
        source_agent: NonEmptyString,
        delete_imported: bool,
    ) -> CandidateProfile:
        """Disconnect a source and optionally delete its confirmed imports."""
        return _call_safely(
            lambda: applications.profile.disconnect_agent_memory(
                source_agent, delete_imported
            ),
        )

    @server.tool()
    def get_rubric() -> Rubric:
        """Return the active local decision rubric."""
        return _call_safely(applications.rubric.get_rubric)

    @server.tool()
    def update_rubric(patch: RubricPatch) -> Rubric:
        """Apply a validated declarative patch to the active rubric."""
        return _call_safely(
            lambda: applications.rubric.update_rubric(patch)
        )

    @server.tool()
    def analyze_job(job_description: NonEmptyString) -> AnalysisRecord:
        """Analyze host-supplied public job text with configured providers."""
        return _call_safely(
            lambda: applications.analysis.analyze_job(job_description),
        )

    @server.tool()
    def recompute_decision(analysis_id: NonEmptyString) -> AnalysisRecord:
        """Recompute a stored decision without another provider call."""
        return _call_safely(
            lambda: applications.analysis.recompute_decision(analysis_id),
        )

    @server.tool()
    def get_decision_trace(analysis_id: NonEmptyString) -> DecisionTrace:
        """Return the validated structured trace for a stored decision."""
        return _call_safely(
            lambda: applications.analysis.get_decision_trace(analysis_id),
        )

    @server.tool()
    def list_analyzed_jobs() -> list[AnalysisSummary]:
        """List structured summaries of successful local analyses."""
        return _call_safely(applications.analysis.list_analyzed_jobs)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
