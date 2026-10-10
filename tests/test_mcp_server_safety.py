import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from ai_dream_job_engine.mcp_server import (
    APPLICATION_CONFIG_ENV_VARS,
    ApplicationError,
    ConfiguredAnalysisApplication,
    LocalProfileApplication,
    LocalRubricApplication,
    MCPServices,
    build_server,
)
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    Rubric,
    RubricPatch,
)
from ai_dream_job_engine.rubric import load_default_rubric
from ai_dream_job_engine.service import AnalysisServiceError


class MemoryStore:
    def __init__(self, *, profile=None, rubric=None):
        self.profile = profile
        self.rubric = rubric
        self.analyses = []

    def save_profile(self, profile):
        self.profile = profile.model_copy(deep=True)

    def load_profile(self):
        return self.profile.model_copy(deep=True) if self.profile else None

    def save_rubric(self, rubric):
        self.rubric = rubric.model_copy(deep=True)

    def load_rubric(self):
        return self.rubric.model_copy(deep=True) if self.rubric else None

    def save_analysis(self, record):
        raise AssertionError("not used")

    def insert_analysis(self, record):
        raise AssertionError("not used")

    def list_analyses(self):
        return list(self.analyses)


class StaticRubricService:
    def get_rubric(self):
        return Rubric(version="rubric-v1")

    def update_rubric(self, patch):
        return Rubric(version="rubric-v1")


class StaticAnalysisService:
    def analyze_job(self, job_description):
        return AnalysisRecord(
            id="analysis-1",
            input_hash="sha256:test",
            status="failed",
            latency_ms=0,
        )

    def recompute_decision(self, analysis_id):
        raise AssertionError("not used")

    def get_decision_trace(self, analysis_id):
        raise AssertionError("not used")

    def list_analyzed_jobs(self):
        return []


def _agent_candidate(candidate_id="memory-1", value="Python"):
    return CandidateClaim(
        id=candidate_id,
        claim_type="skill",
        value=value,
        source=ClaimSource.AGENT_SUGGESTED,
        provenance=f"agent-memory://career-agent/{candidate_id}",
        source_agent="career-agent",
    )


def _resume_candidate(candidate_id, claim_type, value):
    return CandidateClaim(
        id=candidate_id,
        claim_type=claim_type,
        value=value,
        source=ClaimSource.RESUME_EVIDENCE,
    )


def _profile_server(profile_application):
    return build_server(
        MCPServices(
            profile=profile_application,
            rubric=StaticRubricService(),
            analysis=StaticAnalysisService(),
        )
    )


def test_memory_preview_is_server_issued_exact_and_single_use() -> None:
    store = MemoryStore(profile=CandidateProfile())
    application = LocalProfileApplication(store)
    candidate = _agent_candidate()

    first = application.preview_agent_memory_import(
        "career-agent", ["skills"], [candidate]
    )
    second = application.preview_agent_memory_import(
        "career-agent", ["skills"], [candidate]
    )

    assert first.preview_id != second.preview_id
    assert len(first.preview_id) >= 32
    first.preview.candidates[0].value = "forged caller value"
    updated = application.confirm_agent_memory_import(
        first.preview_id,
        [ClaimDecision(claim_id="memory-1", action="accept")],
    )
    assert updated.claims[-1].value == "Python"
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import(
            first.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )


def test_memory_confirmation_rejects_forged_and_stale_preview_ids() -> None:
    store = MemoryStore(profile=CandidateProfile())
    application = LocalProfileApplication(store)
    issued = application.preview_agent_memory_import(
        "career-agent", ["skills"], [_agent_candidate()]
    )

    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import("forged-preview-id", [])

    application.initialize_profile(CandidateProfile(version="2"))
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import(issued.preview_id, [])


def test_memory_preview_is_consumed_only_after_successful_save() -> None:
    class FailOnceStore(MemoryStore):
        fail_next_save = False

        def save_profile(self, profile):
            if self.fail_next_save:
                self.fail_next_save = False
                raise RuntimeError("/private/path API_KEY=secret")
            super().save_profile(profile)

    store = FailOnceStore(profile=CandidateProfile())
    application = LocalProfileApplication(store)
    issued = application.preview_agent_memory_import(
        "career-agent", ["skills"], [_agent_candidate()]
    )
    decisions = [ClaimDecision(claim_id="memory-1", action="accept")]
    store.fail_next_save = True

    with pytest.raises(ApplicationError, match="storage_failure"):
        application.confirm_agent_memory_import(issued.preview_id, decisions)

    updated = application.confirm_agent_memory_import(issued.preview_id, decisions)
    assert any(claim.confirmed for claim in updated.claims)


def test_profile_read_transform_save_operations_are_serialized() -> None:
    class BlockingFirstSaveStore(MemoryStore):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.first_save_started = threading.Event()
            self.release_first_save = threading.Event()
            self.save_count = 0

        def save_profile(self, profile):
            self.save_count += 1
            if self.save_count == 1:
                self.first_save_started.set()
                assert self.release_first_save.wait(2)
            super().save_profile(profile)

    profile = CandidateProfile(
        claims=[
            _resume_candidate("claim-a", "skill", "Python"),
            _resume_candidate("claim-b", "location", "Remote"),
        ]
    )
    store = BlockingFirstSaveStore(profile=profile)
    application = LocalProfileApplication(store)
    second_started = threading.Event()

    def confirm_second():
        second_started.set()
        return application.confirm_profile_claims(
            [ClaimDecision(claim_id="claim-b", action="accept")]
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            application.confirm_profile_claims,
            [ClaimDecision(claim_id="claim-a", action="accept")],
        )
        assert store.first_save_started.wait(2)
        second = executor.submit(confirm_second)
        assert second_started.wait(2)
        store.release_first_save.set()
        first.result(timeout=2)
        second.result(timeout=2)

    confirmed_types = {
        claim.claim_type for claim in store.profile.claims if claim.confirmed
    }
    assert confirmed_types == {"skill", "location"}


def test_rubric_read_transform_save_operations_are_serialized() -> None:
    class BlockingFirstSaveStore(MemoryStore):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.first_save_started = threading.Event()
            self.release_first_save = threading.Event()
            self.save_count = 0

        def save_rubric(self, rubric):
            self.save_count += 1
            if self.save_count == 1:
                self.first_save_started.set()
                assert self.release_first_save.wait(2)
            super().save_rubric(rubric)

    store = BlockingFirstSaveStore(rubric=load_default_rubric())
    application = LocalRubricApplication(store)
    second_started = threading.Event()
    first_patch = RubricPatch(
        fact_updates={"role_family": {"name": "Role family updated"}}
    )
    second_patch = RubricPatch(
        fact_updates={"ai_scope": {"name": "AI scope updated"}}
    )

    def update_second():
        second_started.set()
        return application.update_rubric(second_patch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(application.update_rubric, first_patch)
        assert store.first_save_started.wait(2)
        second = executor.submit(update_second)
        assert second_started.wait(2)
        store.release_first_save.set()
        first.result(timeout=2)
        second.result(timeout=2)

    names = {fact.id: fact.name for fact in store.rubric.facts}
    assert names["role_family"] == "Role family updated"
    assert names["ai_scope"] == "AI scope updated"


@pytest.mark.asyncio
async def test_server_construction_and_listing_do_not_read_application_config(
    monkeypatch,
) -> None:
    original_getenv = os.getenv
    expected_watchlist = {
        "AI_DREAM_JOB_DATA_PATH",
        "AI_DREAM_JOB_FACT_PROVIDER",
        "AI_DREAM_JOB_FACT_MODEL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TYPESAFE_API_KEY",
    }
    assert APPLICATION_CONFIG_ENV_VARS == expected_watchlist

    def forbidden_getenv(key, *args, **kwargs):
        if key in APPLICATION_CONFIG_ENV_VARS:
            raise AssertionError("application environment read during construction")
        return original_getenv(key, *args, **kwargs)

    monkeypatch.setattr(os, "getenv", forbidden_getenv)

    tools = await build_server().list_tools()

    assert len(tools) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_code", "public_code", "safe_message"),
    [
        (
            "missing_rubric",
            "rubric_not_initialized",
            "Initialize the decision rubric before analyzing a job.",
        ),
        (
            "analysis_not_found",
            "analysis_not_found",
            "Choose an existing successful analysis and retry.",
        ),
        (
            "analysis_not_successful",
            "analysis_unavailable",
            "Choose a successful analysis before this action.",
        ),
        (
            "trace_unavailable",
            "analysis_unavailable",
            "Choose a successful analysis before this action.",
        ),
        (
            "analysis_failed",
            "analysis_failed",
            "Job analysis failed safely; verify provider availability and retry.",
        ),
        (
            "empty_job_description",
            "invalid_job_description",
            "Provide a non-empty public job description.",
        ),
        (
            "trace_integrity_failed",
            "trace_integrity_failure",
            "Stored decision trace could not be verified; rerun the analysis.",
        ),
        (
            "provider_not_configured",
            "analysis_provider_config",
            "Configure the selected analysis provider before analyzing a job.",
        ),
        (
            "missing_profile",
            "profile_not_initialized",
            "Initialize the candidate profile before this action.",
        ),
        (
            "storage_failed",
            "storage_failure",
            "Local data could not be saved or loaded; retry the action.",
        ),
        (
            "failure_audit_persistence_failed",
            "storage_failure",
            "Local data could not be saved or loaded; retry the action.",
        ),
        (
            "unexpected_private_code",
            "analysis_failure",
            "The analysis action could not be completed safely.",
        ),
    ],
)
async def test_expected_analysis_errors_have_fixed_public_recovery_messages(
    service_code, public_code, safe_message
) -> None:
    class RaisingAnalysisService(StaticAnalysisService):
        def analyze_job(self, job_description):
            raise AnalysisServiceError(
                service_code,
                "/private/path secret-id OPENAI_API_KEY=secret",
            )

    server = build_server(
        MCPServices(
            profile=LocalProfileApplication(
                MemoryStore(profile=CandidateProfile())
            ),
            rubric=StaticRubricService(),
            analysis=RaisingAnalysisService(),
        )
    )

    with pytest.raises(ToolError) as raised:
        await server.call_tool(
            "analyze_job", {"job_description": "Public job text"}
        )

    _assert_safe_error(
        raised.value,
        public_code,
        safe_message,
        "/private/path",
        "secret-id",
        "OPENAI_API_KEY",
        "unexpected_private_code",
    )


@pytest.mark.asyncio
async def test_public_errors_are_fixed_and_do_not_leak_context(monkeypatch) -> None:
    empty_store = MemoryStore()
    empty_profile = LocalProfileApplication(empty_store)
    server = _profile_server(empty_profile)

    with pytest.raises(ToolError) as missing:
        await server.call_tool("review_profile_claims", {})
    _assert_safe_error(
        missing.value,
        "profile_not_initialized",
        "Initialize the candidate profile before this action.",
    )

    empty_profile.initialize_profile(
        CandidateProfile(
            claims=[_resume_candidate("private-claim-id", "skill", "secret-value")]
        )
    )
    with pytest.raises(ToolError) as decisions:
        await server.call_tool(
            "confirm_profile_claims",
            {"decisions": [{"claim_id": "unknown-private-id", "action": "accept"}]},
        )
    _assert_safe_error(
        decisions.value,
        "invalid_claim_decisions",
        "Review the claim decisions and submit valid per-claim actions.",
        "unknown-private-id",
    )

    with pytest.raises(ToolError) as preview:
        await server.call_tool(
            "confirm_agent_memory_import",
            {"preview_id": "forged-private-token", "decisions": []},
        )
    _assert_safe_error(
        preview.value,
        "memory_preview_invalid",
        "Request a new memory preview before confirming claims.",
        "forged-private-token",
    )

    rubric_application = LocalRubricApplication(
        MemoryStore(rubric=load_default_rubric())
    )
    rubric_server = build_server(
        MCPServices(
            profile=empty_profile,
            rubric=rubric_application,
            analysis=StaticAnalysisService(),
        )
    )
    with pytest.raises(ToolError) as rubric:
        await rubric_server.call_tool(
            "update_rubric",
            {"patch": {"threshold_updates": {"skip": 0.5}}},
        )
    _assert_safe_error(
        rubric.value,
        "invalid_rubric_patch",
        "Review the rubric patch and submit valid rubric changes.",
    )

    class FailingStore(MemoryStore):
        def save_profile(self, profile):
            raise RuntimeError("/private/path OPENAI_API_KEY=secret")

    storage_server = _profile_server(LocalProfileApplication(FailingStore()))
    with pytest.raises(ToolError) as storage:
        await storage_server.call_tool(
            "initialize_profile", {"profile": CandidateProfile().model_dump(mode="json")}
        )
    _assert_safe_error(
        storage.value,
        "storage_failure",
        "Local data could not be saved or loaded; retry the action.",
        "/private/path",
        "OPENAI_API_KEY",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_DREAM_JOB_FACT_MODEL", raising=False)
    analysis_store = MemoryStore(
        profile=CandidateProfile(), rubric=load_default_rubric()
    )
    provider_server = build_server(
        MCPServices(
            profile=LocalProfileApplication(analysis_store),
            rubric=LocalRubricApplication(analysis_store),
            analysis=ConfiguredAnalysisApplication(analysis_store),
        )
    )
    with pytest.raises(ToolError) as provider:
        await provider_server.call_tool(
            "analyze_job", {"job_description": "Public job text"}
        )
    _assert_safe_error(
        provider.value,
        "analysis_provider_config",
        "Configure the selected analysis provider before analyzing a job.",
    )


def _assert_safe_error(error, code, message, *forbidden):
    rendered = str(error)
    assert f"{code}: {message}" in rendered
    for value in forbidden:
        assert value not in rendered
