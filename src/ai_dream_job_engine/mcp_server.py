"""Privacy-safe stdio MCP adapter for the decision engine."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Annotated, Callable, Protocol

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field

from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.mcp_app import (
    MCP_APP_MIME_TYPE,
    PRIVATE_PAYLOAD_META_KEY,
    UI_RESOURCE_META,
    WORKSPACE_RESOURCE_URI,
    WorkspaceAppPayload,
    load_workspace_html,
    open_claim_review as build_claim_review_payload,
    open_job_workspace as build_job_workspace_payload,
)
from ai_dream_job_engine.models import (
    AnalysisRecord,
    AnalysisSummary,
    CandidateClaim,
    CandidateProfile,
    ClaimPreviewView,
    ClaimDecision,
    CustomPreferenceProposal,
    DecisionTrace,
    JobReviewStatus,
    MemoryImportPreview,
    PreferenceResolution,
    Rubric,
    RubricPatch,
    WorkspaceJobDetail,
    WorkspaceSnapshot,
)
from ai_dream_job_engine.preferences import resolve_preference, stage_preference
from ai_dream_job_engine.profile import (
    confirm_claims,
    disconnect_memory_source,
    preview_memory_import,
    profile_with_memory_connection,
)
from ai_dream_job_engine.providers.base import ProviderError
from ai_dream_job_engine.providers.jev import JevSignalProvider
from ai_dream_job_engine.providers.openai_compatible import (
    OpenAICompatibleProvider,
)
from ai_dream_job_engine.rubric import apply_rubric_patch, load_default_rubric
from ai_dream_job_engine.service import AnalysisServiceError, JobAnalysisService
from ai_dream_job_engine.workspace import WorkspaceApplication, WorkspaceError


NonEmptyString = Annotated[str, Field(min_length=1)]
NonEmptyStrings = Annotated[list[str], Field(min_length=1)]

_DATA_PATH_ENV = "AI_DREAM_JOB_DATA_PATH"
_FACT_PROVIDER_ENV = "AI_DREAM_JOB_FACT_PROVIDER"
_FACT_MODEL_ENV = "AI_DREAM_JOB_FACT_MODEL"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"
_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
_JEV_KEY_ENV = "TYPESAFE_API_KEY"
_PROFILE_PREVIEW_PREFIX = "profile_"
_MEMORY_PREVIEW_PREFIX = "memory_"
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
PROFILE_PREVIEW_INVALID = (
    "profile_preview_invalid",
    "Review pending resume claims before confirming them.",
)
INVALID_CLAIM_DECISIONS = (
    "invalid_claim_decisions",
    "Review the claim decisions and submit valid per-claim actions.",
)
INVALID_RUBRIC_PATCH = (
    "invalid_rubric_patch",
    "Review the rubric patch and submit valid rubric changes.",
)
INVALID_CUSTOM_PREFERENCE = (
    "invalid_custom_preference",
    "Review the custom preference and submit a valid resolution.",
)
RUBRIC_UPDATE_CONFLICT = (
    "rubric_update_conflict",
    "The same rubric fields changed concurrently; reload and retry your changes.",
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
REVIEW_STATUS_INELIGIBLE = (
    "review_status_ineligible",
    "Only priority-apply jobs without hard blockers or unresolved evidence gaps can be ready to apply.",
)


class ApplicationError(RuntimeError):
    """A fixed public error that cannot carry source exception context."""

    def __init__(self, public_error: tuple[str, str]) -> None:
        self.code, self.safe_message = public_error
        super().__init__(f"{self.code}: {self.safe_message}")


class IssuedMemoryImportPreview(BaseModel):
    preview_id: str = Field(min_length=32)
    preview: MemoryImportPreview


class IssuedProfileClaimsPreview(BaseModel):
    preview_id: str = Field(min_length=32)
    candidates: list[CandidateClaim]


@dataclass(frozen=True)
class _RegisteredPreview:
    profile_digest: str
    binding_digest: str
    generation: int
    preview: MemoryImportPreview


@dataclass(frozen=True)
class _RegisteredProfilePreview:
    profile_digest: str
    binding_digest: str
    candidates: tuple[CandidateClaim, ...]


class ProfileApplication(Protocol):
    def get_profile(self) -> CandidateProfile: ...

    def get_claim_preview(self, preview_id: str) -> ClaimPreviewView: ...

    def initialize_profile(
        self, profile: CandidateProfile | None = None
    ) -> CandidateProfile: ...

    def review_profile_claims(
        self, proposals: list[CandidateClaim]
    ) -> IssuedProfileClaimsPreview: ...

    def confirm_profile_claims(
        self, preview_id: str, decisions: list[ClaimDecision]
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

    def stage_custom_preference(
        self, proposal: CustomPreferenceProposal
    ) -> Rubric: ...

    def resolve_custom_preference(
        self, draft_id: str, resolution: PreferenceResolution
    ) -> Rubric: ...


class AnalysisApplication(Protocol):
    def analyze_job(self, job_description: str) -> AnalysisRecord: ...

    def recompute_decision(self, analysis_id: str) -> AnalysisRecord: ...

    def get_decision_trace(self, analysis_id: str) -> DecisionTrace: ...

    def list_analyzed_jobs(self) -> list[AnalysisSummary]: ...


class WorkspaceOperations(Protocol):
    def get_snapshot(self) -> WorkspaceSnapshot: ...

    def get_job_detail(self, analysis_id: str) -> WorkspaceJobDetail: ...

    def set_review_status(
        self, analysis_id: str, status: JobReviewStatus
    ) -> WorkspaceJobDetail: ...

    def get_profile(self) -> CandidateProfile: ...

    def get_claim_preview(self, preview_id: str) -> ClaimPreviewView: ...


@dataclass(frozen=True)
class MCPServices:
    profile: ProfileApplication
    rubric: RubricApplication
    analysis: AnalysisApplication
    workspace: WorkspaceOperations | None = None


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

    def compare_and_swap_profile(
        self, profile: CandidateProfile, expected_storage_revision: int
    ) -> bool:
        return self._ready().compare_and_swap_profile(
            profile, expected_storage_revision
        )

    def load_profile(self) -> CandidateProfile | None:
        return self._ready().load_profile()

    def save_rubric(self, rubric: Rubric) -> None:
        self._ready().save_rubric(rubric)

    def compare_and_swap_rubric(
        self, rubric: Rubric, expected_storage_revision: int
    ) -> bool:
        return self._ready().compare_and_swap_rubric(
            rubric, expected_storage_revision
        )

    def load_rubric(self) -> Rubric | None:
        return self._ready().load_rubric()

    def save_analysis(self, record: AnalysisRecord) -> None:
        self._ready().save_analysis(record)

    def compare_and_swap_analysis(
        self, record: AnalysisRecord, expected_decision_revision: int
    ) -> bool:
        return self._ready().compare_and_swap_analysis(
            record, expected_decision_revision
        )

    def insert_analysis(self, record: AnalysisRecord) -> None:
        self._ready().insert_analysis(record)

    def list_analyses(self) -> list[AnalysisRecord]:
        return self._ready().list_analyses()

    def save_job_review(self, record) -> None:
        self._ready().save_job_review(record)

    def load_job_review(self, analysis_id: str):
        return self._ready().load_job_review(analysis_id)

    def list_job_reviews(self):
        return self._ready().list_job_reviews()


class LocalProfileApplication:
    def __init__(self, store) -> None:
        self._store = store
        self._lock = RLock()
        self._previews: dict[str, _RegisteredPreview] = {}
        self._profile_previews: dict[str, _RegisteredProfilePreview] = {}

    def initialize_profile(
        self, profile: CandidateProfile | None = None
    ) -> CandidateProfile:
        with self._lock:
            requested = profile or CandidateProfile()
            if requested.claims or requested.import_receipts:
                raise ApplicationError(INVALID_CLAIM_DECISIONS)
            empty = requested.model_copy(
                update={"claims": [], "import_receipts": []}, deep=True
            )
            self._save_profile(empty)
            self._previews.clear()
            self._profile_previews.clear()
            return empty

    def get_profile(self) -> CandidateProfile:
        with self._lock:
            return self._profile().model_copy(deep=True)

    def get_claim_preview(self, preview_id: str) -> ClaimPreviewView:
        with self._lock:
            profile = self._profile()
            profile_preview = self._profile_previews.get(preview_id)
            if profile_preview is not None:
                if (
                    _model_digest(profile) != profile_preview.profile_digest
                    or _profile_preview_digest(profile, profile_preview.candidates)
                    != profile_preview.binding_digest
                ):
                    raise ApplicationError(PROFILE_PREVIEW_INVALID)
                return ClaimPreviewView(
                    preview_id=preview_id,
                    kind="profile_claims",
                    candidates=[
                        item.model_copy(deep=True)
                        for item in profile_preview.candidates
                    ],
                )

            memory_preview = self._previews.get(preview_id)
            if memory_preview is None:
                error = (
                    MEMORY_PREVIEW_INVALID
                    if preview_id.startswith(_MEMORY_PREVIEW_PREFIX)
                    else PROFILE_PREVIEW_INVALID
                )
                raise ApplicationError(error)
            if _model_digest(profile) != memory_preview.profile_digest:
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            if not _memory_preview_is_current(profile, memory_preview):
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            try:
                reproduced = preview_memory_import(
                    profile,
                    memory_preview.preview.source_agent,
                    memory_preview.preview.scope,
                    memory_preview.preview.candidates,
                )
            except ValueError:
                raise ApplicationError(MEMORY_PREVIEW_INVALID) from None
            if (
                _preview_binding_digest(profile, reproduced)
                != memory_preview.binding_digest
            ):
                raise ApplicationError(MEMORY_PREVIEW_INVALID)
            preview = memory_preview.preview
            return ClaimPreviewView(
                preview_id=preview_id,
                kind="agent_memory",
                candidates=[item.model_copy(deep=True) for item in preview.candidates],
                source_agent=preview.source_agent,
                scope=list(preview.scope),
                added_claim_ids=list(preview.added_claim_ids),
                changed_claim_ids=list(preview.changed_claim_ids),
                conflicting_claim_ids=list(preview.conflicting_claim_ids),
            )

    def review_profile_claims(
        self, proposals: list[CandidateClaim]
    ) -> IssuedProfileClaimsPreview:
        with self._lock:
            profile = self._profile()
            try:
                _validate_resume_proposals(proposals)
            except ValueError:
                raise ApplicationError(PROFILE_PREVIEW_INVALID) from None
            preview_id = self._new_preview_id(_PROFILE_PREVIEW_PREFIX)
            candidates = tuple(item.model_copy(deep=True) for item in proposals)
            self._profile_previews[preview_id] = _RegisteredProfilePreview(
                profile_digest=_model_digest(profile),
                binding_digest=_profile_preview_digest(profile, candidates),
                candidates=candidates,
            )
            return IssuedProfileClaimsPreview(
                preview_id=preview_id,
                candidates=[item.model_copy(deep=True) for item in candidates],
            )

    def confirm_profile_claims(
        self, preview_id: str, decisions: list[ClaimDecision]
    ) -> CandidateProfile:
        with self._lock:
            issued = self._profile_previews.get(preview_id)
            if issued is None:
                raise ApplicationError(PROFILE_PREVIEW_INVALID)
            if {item.claim_id for item in decisions} != {
                item.id for item in issued.candidates
            }:
                raise ApplicationError(PROFILE_PREVIEW_INVALID)
            def apply(latest: CandidateProfile) -> CandidateProfile:
                try:
                    staged = latest.model_copy(
                        update={"claims": [*latest.claims, *issued.candidates]},
                        deep=True,
                    )
                    confirmed = confirm_claims(staged, decisions)
                    proposal_ids = {item.id for item in issued.candidates}
                    return confirmed.model_copy(
                        update={
                            "claims": [
                                claim
                                for claim in confirmed.claims
                                if claim.id not in proposal_ids
                            ]
                        },
                        deep=True,
                    )
                except ValueError:
                    raise ApplicationError(INVALID_CLAIM_DECISIONS) from None

            updated = self._update_profile(apply)
            del self._profile_previews[preview_id]
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
            def connect(latest: CandidateProfile) -> CandidateProfile:
                try:
                    preview = preview_memory_import(
                        latest, source_agent, scope, candidates
                    )
                    return profile_with_memory_connection(latest, preview)
                except ValueError:
                    raise ApplicationError(MEMORY_PREVIEW_INVALID) from None

            profile = self._update_profile(connect)
            try:
                preview = preview_memory_import(profile, source_agent, scope, candidates)
            except ValueError:
                raise ApplicationError(MEMORY_PREVIEW_INVALID) from None
            preview_id = self._new_preview_id(_MEMORY_PREVIEW_PREFIX)
            registered_preview = preview.model_copy(deep=True)
            connection = next(
                item
                for item in profile.memory_connections
                if item.source_agent == source_agent
            )
            self._previews[preview_id] = _RegisteredPreview(
                profile_digest=_model_digest(profile),
                binding_digest=_preview_binding_digest(
                    profile, registered_preview
                ),
                generation=connection.generation,
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
            def apply(latest: CandidateProfile) -> CandidateProfile:
                if not _memory_preview_is_current(latest, issued):
                    raise ApplicationError(MEMORY_PREVIEW_INVALID)
                try:
                    reproduced = preview_memory_import(
                        latest,
                        issued.preview.source_agent,
                        issued.preview.scope,
                        issued.preview.candidates,
                    )
                    return confirm_claims(latest, decisions, preview=reproduced)
                except ValueError:
                    raise ApplicationError(INVALID_CLAIM_DECISIONS) from None

            updated = self._update_profile(apply)
            del self._previews[preview_id]
            return updated

    def disconnect_agent_memory(
        self, source_agent: str, delete_imported: bool
    ) -> CandidateProfile:
        with self._lock:
            def disconnect(latest: CandidateProfile) -> CandidateProfile:
                try:
                    return disconnect_memory_source(
                        latest, source_agent, delete_imported
                    )
                except ValueError:
                    raise ApplicationError(INVALID_CLAIM_DECISIONS) from None

            updated = self._update_profile(disconnect)
            self._previews = {
                preview_id: preview
                for preview_id, preview in self._previews.items()
                if preview.preview.source_agent != source_agent
            }
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

    def _update_profile(self, operation) -> CandidateProfile:
        while True:
            current = self._profile()
            updated = operation(current).model_copy(
                update={"storage_revision": current.storage_revision + 1},
                deep=True,
            )
            try:
                compare_and_swap = getattr(
                    self._store, "compare_and_swap_profile", None
                )
                if compare_and_swap is None:
                    self._store.save_profile(updated)
                    saved = True
                else:
                    saved = compare_and_swap(updated, current.storage_revision)
            except Exception:
                raise ApplicationError(STORAGE_FAILURE) from None
            if saved:
                return updated

    def _new_preview_id(self, prefix: str) -> str:
        while True:
            preview_id = f"{prefix}{secrets.token_urlsafe(32)}"
            if (
                preview_id not in self._previews
                and preview_id not in self._profile_previews
            ):
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
            base: Rubric | None = None
            while True:
                current = self._get_rubric()
                if base is None:
                    base = current
                elif _rubric_patch_conflicts(base, current, patch):
                    raise ApplicationError(RUBRIC_UPDATE_CONFLICT)
                try:
                    updated = apply_rubric_patch(current, patch).model_copy(
                        update={"storage_revision": current.storage_revision + 1}
                    )
                except ValueError:
                    raise ApplicationError(INVALID_RUBRIC_PATCH) from None
                try:
                    compare_and_swap = getattr(
                        self._store, "compare_and_swap_rubric", None
                    )
                    if compare_and_swap is None:
                        self._store.save_rubric(updated)
                        saved = True
                    else:
                        saved = compare_and_swap(
                            updated, current.storage_revision
                        )
                    if saved:
                        return updated
                except Exception:
                    raise ApplicationError(STORAGE_FAILURE) from None

    def stage_custom_preference(
        self, proposal: CustomPreferenceProposal
    ) -> Rubric:
        with self._lock:
            return self._mutate_preference(
                lambda rubric: stage_preference(rubric, proposal)
            )

    def resolve_custom_preference(
        self, draft_id: str, resolution: PreferenceResolution
    ) -> Rubric:
        with self._lock:
            return self._mutate_preference(
                lambda rubric: resolve_preference(
                    rubric, draft_id, resolution
                )
            )

    def _mutate_preference(
        self, operation: Callable[[Rubric], Rubric]
    ) -> Rubric:
        while True:
            current = self._get_rubric()
            try:
                updated = operation(current).model_copy(
                    update={"storage_revision": current.storage_revision + 1}
                )
            except ValueError:
                raise ApplicationError(INVALID_CUSTOM_PREFERENCE) from None
            try:
                compare_and_swap = getattr(
                    self._store, "compare_and_swap_rubric", None
                )
                if compare_and_swap is None:
                    self._store.save_rubric(updated)
                    saved = True
                else:
                    saved = compare_and_swap(
                        updated, current.storage_revision
                    )
                if saved:
                    return updated
            except Exception:
                raise ApplicationError(STORAGE_FAILURE) from None

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


def _rubric_patch_conflicts(
    base: Rubric, current: Rubric, patch: RubricPatch
) -> bool:
    for collection_name, updates in (
        ("facts", patch.fact_updates),
        ("dimensions", patch.dimension_updates),
        ("questions", patch.question_updates),
    ):
        base_items = {
            item.id: item.model_dump(mode="python")
            for item in getattr(base, collection_name)
        }
        current_items = {
            item.id: item.model_dump(mode="python")
            for item in getattr(current, collection_name)
        }
        for item_id, changes in updates.items():
            if item_id not in base_items or item_id not in current_items:
                if base_items.get(item_id) != current_items.get(item_id):
                    return True
                continue
            for field in changes:
                if base_items[item_id].get(field) != current_items[item_id].get(field):
                    return True
    return any(
        base.recommendation_thresholds.get(level)
        != current.recommendation_thresholds.get(level)
        for level in patch.threshold_updates
    )


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


def _memory_preview_is_current(
    profile: CandidateProfile, issued: _RegisteredPreview
) -> bool:
    connection = next(
        (
            item
            for item in profile.memory_connections
            if item.source_agent == issued.preview.source_agent
        ),
        None,
    )
    return bool(
        connection is not None
        and connection.connected
        and connection.scope == issued.preview.scope
        and connection.generation == issued.generation
    )


def _validate_resume_proposals(proposals: list[CandidateClaim]) -> None:
    ids = [item.id for item in proposals]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate resume proposal IDs")
    for proposal in proposals:
        if (
            proposal.source.value != "resume_evidence"
            or proposal.confirmed
            or not proposal.source_quote
            or proposal.import_origin is not None
            or proposal.source_agent is not None
        ):
            raise ValueError("resume proposals must be pending quoted resume evidence")


def _profile_preview_digest(
    profile: CandidateProfile, candidates: tuple[CandidateClaim, ...]
) -> str:
    payload = json.dumps(
        {
            "profile": profile.model_dump(mode="json"),
            "candidates": [item.model_dump(mode="json") for item in candidates],
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
    if sys.platform == "darwin":
        data_root = Path.home() / "Library" / "Application Support"
    else:
        configured_root = os.getenv("XDG_DATA_HOME")
        data_root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path.home() / ".local" / "share"
        )
    return data_root / "ai-dream-job-engine" / "ai-dream-job-engine.sqlite3"


def default_services() -> MCPServices:
    store = _LazyLocalStore(_default_database_path)
    profile = LocalProfileApplication(store)
    rubric = LocalRubricApplication(store)
    analysis = ConfiguredAnalysisApplication(store)
    return MCPServices(
        profile=profile,
        rubric=rubric,
        analysis=analysis,
        workspace=WorkspaceApplication(
            store,
            profile_application=profile,
            analysis_application=analysis,
        ),
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


def _workspace_public_error(error: WorkspaceError) -> tuple[str, str]:
    if error.code == "analysis_not_found":
        return ANALYSIS_NOT_FOUND
    if error.code == "analysis_unavailable":
        return ANALYSIS_UNAVAILABLE
    if error.code == "storage_failure":
        return STORAGE_FAILURE
    if error.code == "review_status_ineligible":
        return REVIEW_STATUS_INELIGIBLE
    return INTERNAL_FAILURE


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
    except WorkspaceError as error:
        public_error = _workspace_public_error(error)
        raise ToolError(
            f"{public_error[0]}: {public_error[1]}"
        ) from None
    except Exception:
        raise ToolError(
            f"{INTERNAL_FAILURE[0]}: {INTERNAL_FAILURE[1]}"
        ) from None


def _private_app_result(payload, *, summary: dict | None = None) -> CallToolResult:
    private_payload = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else payload
    )
    return CallToolResult(
        content=[TextContent(type="text", text="Local app data is ready.")],
        structured_content=summary or {"status": "ok"},
        meta={PRIVATE_PAYLOAD_META_KEY: {"payload": private_payload}},
    )


def build_server(services: MCPServices | None = None) -> MCPServer:
    applications = services or default_services()
    server = MCPServer(
        "ai-dream-job-engine",
        description="Local-first, evidence-backed job decision engine.",
        version="0.1.0",
        log_level="ERROR",
    )

    @server.resource(
        WORKSPACE_RESOURCE_URI,
        name="AI Dream Job Workspace",
        description="Self-contained local job decision workspace.",
        mime_type=MCP_APP_MIME_TYPE,
        meta=UI_RESOURCE_META,
    )
    def workspace_resource() -> str:
        return load_workspace_html()

    @server.tool(
        meta={
            "ui": {
                "resourceUri": WORKSPACE_RESOURCE_URI,
                "visibility": ["model", "app"],
            },
            "openai/ui": {"entrypoints": [{"type": "thread"}]},
        },
        structured_output=False,
    )
    def open_job_workspace() -> CallToolResult:
        """Open the local workspace with server-owned initial job data."""
        payload = _call_safely(
            lambda: build_job_workspace_payload(applications.workspace)
        )
        return _private_app_result(
            payload,
            summary={
                "status": "ok",
                "view": "jobs",
                "profile_ready": payload.snapshot.profile_ready,
                "rubric_ready": payload.snapshot.rubric_ready,
                "job_count": len(payload.snapshot.jobs),
            },
        )

    @server.tool(
        meta={
            "ui": {
                "resourceUri": WORKSPACE_RESOURCE_URI,
                "visibility": ["model", "app"],
            }
        },
        structured_output=False,
    )
    def open_claim_review(preview_id: NonEmptyString) -> CallToolResult:
        """Open one immutable server-issued claim preview in the workspace."""
        payload = _call_safely(
            lambda: build_claim_review_payload(
                applications.workspace, preview_id
            )
        )
        return _private_app_result(
            payload,
            summary={"status": "ok", "view": "claim_review"},
        )

    @server.tool()
    def initialize_profile(
        profile: CandidateProfile | None = None,
    ) -> CandidateProfile:
        """Initialize an empty local candidate profile."""
        return _call_safely(
            lambda: applications.profile.initialize_profile(profile),
        )

    @server.tool(
        meta={"ui": {"visibility": ["model", "app"]}},
        structured_output=False,
    )
    def review_profile_claims(
        proposals: list[CandidateClaim],
    ) -> CallToolResult:
        """Issue a bound preview for pending host-supplied resume claims."""
        payload = _call_safely(
            lambda: applications.profile.review_profile_claims(proposals)
        )
        return _private_app_result(
            payload,
            summary={
                "status": "ok",
                "preview_id": payload.preview_id,
                "candidate_count": len(payload.candidates),
            },
        )

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def confirm_profile_claims(
        preview_id: NonEmptyString,
        decisions: list[ClaimDecision],
    ) -> CallToolResult:
        """Apply a complete set of explicit per-claim resume decisions."""
        return _private_app_result(_call_safely(
            lambda: applications.profile.confirm_profile_claims(
                preview_id, decisions
            ),
        ))

    @server.tool(
        meta={"ui": {"visibility": ["model", "app"]}},
        structured_output=False,
    )
    def preview_agent_memory_import(
        source_agent: NonEmptyString,
        scope: NonEmptyStrings,
        candidates: list[CandidateClaim],
    ) -> CallToolResult:
        """Preview explicit host-supplied memory candidates without mutation."""
        payload = _call_safely(
            lambda: applications.profile.preview_agent_memory_import(
                source_agent, scope, candidates
            ),
        )
        return _private_app_result(
            payload,
            summary={
                "status": "ok",
                "preview_id": payload.preview_id,
                "candidate_count": len(payload.preview.candidates),
            },
        )

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def confirm_agent_memory_import(
        preview_id: NonEmptyString,
        decisions: list[ClaimDecision],
    ) -> CallToolResult:
        """Apply a complete set of explicit per-claim memory decisions."""
        return _private_app_result(_call_safely(
            lambda: applications.profile.confirm_agent_memory_import(
                preview_id, decisions
            ),
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def disconnect_agent_memory(
        source_agent: NonEmptyString,
        delete_imported: bool,
    ) -> CallToolResult:
        """Disconnect a source and optionally delete its confirmed imports."""
        return _private_app_result(_call_safely(
            lambda: applications.profile.disconnect_agent_memory(
                source_agent, delete_imported
            ),
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def get_rubric() -> CallToolResult:
        """Return the active local decision rubric."""
        return _private_app_result(_call_safely(applications.rubric.get_rubric))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def update_rubric(patch: RubricPatch) -> CallToolResult:
        """Apply a validated declarative patch to the active rubric."""
        return _private_app_result(_call_safely(
            lambda: applications.rubric.update_rubric(patch)
        ))

    @server.tool(
        meta={"ui": {"visibility": ["model", "app"]}},
        structured_output=False,
    )
    def stage_custom_preference(
        proposal: CustomPreferenceProposal,
    ) -> CallToolResult:
        """Stage a structured preference for explicit user review."""
        return _private_app_result(_call_safely(
            lambda: applications.rubric.stage_custom_preference(proposal)
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def resolve_custom_preference(
        draft_id: NonEmptyString,
        resolution: PreferenceResolution,
    ) -> CallToolResult:
        """Apply an explicit user resolution to a staged preference."""
        return _private_app_result(_call_safely(
            lambda: applications.rubric.resolve_custom_preference(
                draft_id, resolution
            )
        ))

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

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def get_workspace_snapshot() -> CallToolResult:
        """Return setup state, confirmed claims, and reviewed job summaries."""
        return _private_app_result(_call_safely(
            lambda: applications.workspace.get_snapshot()
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def get_workspace_job(analysis_id: NonEmptyString) -> CallToolResult:
        """Return one integrity-valid decision with grouped gaps and evidence."""
        return _private_app_result(_call_safely(
            lambda: applications.workspace.get_job_detail(analysis_id)
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def set_job_review_status(
        analysis_id: NonEmptyString,
        status: JobReviewStatus,
    ) -> CallToolResult:
        """Persist an explicit review status after server-side eligibility checks."""
        return _private_app_result(_call_safely(
            lambda: applications.workspace.set_review_status(analysis_id, status)
        ))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def get_candidate_profile() -> CallToolResult:
        """Return the active local candidate profile."""
        return _private_app_result(_call_safely(applications.workspace.get_profile))

    @server.tool(meta={"ui": {"visibility": ["app"]}}, structured_output=False)
    def get_claim_preview(preview_id: NonEmptyString) -> CallToolResult:
        """Return an immutable view of a server-issued claim preview."""
        return _private_app_result(_call_safely(
            lambda: applications.workspace.get_claim_preview(preview_id)
        ))

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
