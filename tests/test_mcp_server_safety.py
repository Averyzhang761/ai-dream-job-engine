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
    _default_database_path,
    build_server,
)
from ai_dream_job_engine.database import LocalStore
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    CustomPreferenceProposal,
    JobReviewStatus,
    PreferenceResolution,
    Rubric,
    RubricPatch,
)
from ai_dream_job_engine.rubric import load_default_rubric
from ai_dream_job_engine.service import AnalysisServiceError
from ai_dream_job_engine.workspace import (
    STORAGE_FAILURE as WORKSPACE_STORAGE_FAILURE,
    WorkspaceError,
)


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


class BarrierRubricStore(LocalStore):
    def __init__(self, path, barrier):
        super().__init__(path)
        self._barrier = barrier
        self._first_cas = True

    def compare_and_swap_rubric(self, rubric, expected_storage_revision):
        if self._first_cas:
            self._first_cas = False
            self._barrier.wait(timeout=5)
        return super().compare_and_swap_rubric(rubric, expected_storage_revision)


class BarrierProfileStore(LocalStore):
    def __init__(self, path, barrier, disconnect_written=None):
        super().__init__(path)
        self._barrier = barrier
        self._disconnect_written = disconnect_written
        self._first_cas = True
        self._armed = False
        self._prefer_disconnect = False

    def arm(self, *, prefer_disconnect=False):
        self._armed = True
        self._prefer_disconnect = prefer_disconnect

    def compare_and_swap_profile(self, profile, expected_storage_revision):
        if self._armed and self._first_cas:
            self._first_cas = False
            self._barrier.wait(timeout=5)
            if self._prefer_disconnect:
                is_disconnect = any(
                    not connection.connected
                    for connection in profile.memory_connections
                )
                if is_disconnect:
                    saved = super().compare_and_swap_profile(
                        profile, expected_storage_revision
                    )
                    self._disconnect_written.set()
                    return saved
                assert self._disconnect_written.wait(timeout=5)
        return super().compare_and_swap_profile(profile, expected_storage_revision)


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
        source_quote=f"Resume evidence for {candidate_id}",
    )


def test_profile_initialization_rejects_claims_and_resume_confirmation_needs_preview() -> None:
    store = MemoryStore()
    application = LocalProfileApplication(store)
    forged = _resume_candidate("resume-1", "skill", "Python").model_copy(
        update={"confirmed": True}
    )

    with pytest.raises(ApplicationError, match="invalid_claim_decisions"):
        application.initialize_profile(CandidateProfile(claims=[forged]))

    initialized = application.initialize_profile(CandidateProfile())
    assert initialized.claims == []
    proposal = _resume_candidate("resume-1", "skill", "Python").model_copy(
        update={"source_quote": "Built Python services."}
    )
    issued = application.review_profile_claims([proposal])

    with pytest.raises(ApplicationError, match="profile_preview_invalid"):
        application.confirm_profile_claims(
            "forged-preview", [ClaimDecision(claim_id="resume-1", action="accept")]
        )

    confirmed = application.confirm_profile_claims(
        issued.preview_id, [ClaimDecision(claim_id="resume-1", action="accept")]
    )
    assert any(claim.confirmed and claim.value == "Python" for claim in confirmed.claims)


@pytest.mark.asyncio
async def test_mcp_profile_claims_require_server_preview_and_per_claim_decisions() -> None:
    application = LocalProfileApplication(MemoryStore())
    server = _profile_server(application)
    forged = _resume_candidate("resume-1", "skill", "Python").model_copy(
        update={"confirmed": True}
    )

    with pytest.raises(ToolError, match="invalid_claim_decisions"):
        await server.call_tool(
            "initialize_profile",
            {"profile": CandidateProfile(claims=[forged]).model_dump(mode="json")},
        )

    initialized = await server.call_tool("initialize_profile", {})
    assert initialized.structured_content["claims"] == []
    pending = _resume_candidate("resume-1", "skill", "Python")
    issued = await server.call_tool(
        "review_profile_claims",
        {"proposals": [pending.model_dump(mode="json")]},
    )
    preview_id = issued.structured_content["preview_id"]
    assert issued.structured_content == {
        "status": "ok",
        "preview_id": preview_id,
        "candidate_count": 1,
    }
    assert "Python" not in str(issued.structured_content)
    assert "Python" not in " ".join(item.text for item in issued.content)
    assert (
        issued.meta["aiDreamJobEngine"]["payload"]["candidates"][0]["value"]
        == "Python"
    )
    confirmed = await server.call_tool(
        "confirm_profile_claims",
        {
            "preview_id": preview_id,
            "decisions": [{"claim_id": "resume-1", "action": "accept"}],
        },
    )

    assert confirmed.structured_content == {"status": "ok"}
    private_profile = confirmed.meta["aiDreamJobEngine"]["payload"]
    assert any(item["confirmed"] for item in private_profile["claims"])
    assert "Python" not in str(confirmed.structured_content)


@pytest.mark.asyncio
async def test_memory_preview_tool_keeps_candidate_values_out_of_model_content() -> None:
    server = _profile_server(
        LocalProfileApplication(MemoryStore(profile=CandidateProfile()))
    )
    candidate = _agent_candidate(value="private memory capability")

    issued = await server.call_tool(
        "preview_agent_memory_import",
        {
            "source_agent": "career-agent",
            "scope": ["skills"],
            "candidates": [candidate.model_dump(mode="json")],
        },
    )

    preview_id = issued.structured_content["preview_id"]
    assert issued.structured_content == {
        "status": "ok",
        "preview_id": preview_id,
        "candidate_count": 1,
    }
    assert "private memory capability" not in str(issued.structured_content)
    assert "private memory capability" not in " ".join(item.text for item in issued.content)
    assert (
        issued.meta["aiDreamJobEngine"]["payload"]["preview"]["candidates"][0][
            "value"
        ]
        == "private memory capability"
    )


def test_profile_confirmation_persists_only_confirmed_mixed_decisions() -> None:
    class FailOnceStore(MemoryStore):
        fail_next_save = False

        def save_profile(self, profile):
            if self.fail_next_save:
                self.fail_next_save = False
                raise RuntimeError("/private/path API_KEY=secret")
            super().save_profile(profile)

    existing = CandidateClaim(
        id="existing",
        claim_type="existing_fact",
        value="keep me",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        provenance="user confirmation",
    )
    initial = CandidateProfile(claims=[existing])
    store = FailOnceStore(profile=initial)
    application = LocalProfileApplication(store)
    proposals = [
        _resume_candidate("accept", "skill", "Python"),
        _resume_candidate("edit", "location", "San Francisco"),
        _resume_candidate("reject", "compensation", "rejected private value"),
    ]
    issued = application.review_profile_claims(proposals)
    decisions = [
        ClaimDecision(claim_id="accept", action="accept"),
        ClaimDecision(claim_id="edit", action="edit", edited_value="Remote"),
        ClaimDecision(claim_id="reject", action="reject"),
    ]
    store.fail_next_save = True

    with pytest.raises(ApplicationError, match="storage_failure"):
        application.confirm_profile_claims(issued.preview_id, decisions)
    assert store.load_profile() == initial

    application.confirm_profile_claims(issued.preview_id, decisions)
    reloaded = store.load_profile()

    assert reloaded is not None
    assert {claim.value for claim in reloaded.claims} == {
        "keep me",
        "Python",
        "Remote",
    }
    assert all(claim.confirmed for claim in reloaded.claims)
    assert not {"accept", "edit", "reject"} & {
        claim.id for claim in reloaded.claims
    }
    with pytest.raises(ApplicationError, match="profile_preview_invalid"):
        application.confirm_profile_claims(issued.preview_id, decisions)


def test_explicit_database_path_override_is_preserved(monkeypatch, tmp_path) -> None:
    configured = tmp_path / "custom" / "state.sqlite3"
    monkeypatch.setenv("AI_DREAM_JOB_DATA_PATH", str(configured))

    assert _default_database_path() == configured


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
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import(
            first.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )
    second.preview.candidates[0].value = "forged caller value"
    updated = application.confirm_agent_memory_import(
        second.preview_id,
        [ClaimDecision(claim_id="memory-1", action="accept")],
    )
    assert updated.claims[-1].value == "Python"
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import(
            second.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )


@pytest.mark.parametrize("delete_imported", [False, True])
def test_disconnect_revokes_outstanding_memory_previews(delete_imported) -> None:
    store = MemoryStore(profile=CandidateProfile())
    application = LocalProfileApplication(store)
    issued = application.preview_agent_memory_import(
        "career-agent", ["skills"], [_agent_candidate()]
    )

    disconnected = application.disconnect_agent_memory(
        "career-agent", delete_imported
    )

    assert disconnected.memory_connections[0].connected is False
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.get_claim_preview(issued.preview_id)
    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        application.confirm_agent_memory_import(
            issued.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )


def test_disconnect_in_other_instance_revokes_memory_confirmation(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    first_store = LocalStore.open(path)
    second_store = LocalStore.open(path)
    first_store.initialize()
    first_store.save_profile(CandidateProfile())
    first = LocalProfileApplication(first_store)
    second = LocalProfileApplication(second_store)
    issued = first.preview_agent_memory_import(
        "career-agent", ["skills"], [_agent_candidate()]
    )

    second.disconnect_agent_memory("career-agent", False)

    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        first.confirm_agent_memory_import(
            issued.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )
    loaded = first_store.load_profile()
    assert loaded is not None
    assert loaded.memory_connections[0].connected is False
    assert not any(claim.confirmed for claim in loaded.claims)


def test_old_preview_cannot_replay_after_disconnect_and_same_scope_reconnect(
    tmp_path,
) -> None:
    path = tmp_path / "shared.sqlite3"
    first_store = LocalStore.open(path)
    second_store = LocalStore.open(path)
    first_store.initialize()
    first_store.save_profile(CandidateProfile())
    first = LocalProfileApplication(first_store)
    second = LocalProfileApplication(second_store)
    old = first.preview_agent_memory_import(
        "career-agent",
        ["skills"],
        [_agent_candidate("old-memory", "old private claim")],
    )

    second.disconnect_agent_memory("career-agent", True)
    current = second.preview_agent_memory_import(
        "career-agent",
        ["skills"],
        [_agent_candidate("current-memory", "current private claim")],
    )

    with pytest.raises(ApplicationError, match="memory_preview_invalid"):
        first.confirm_agent_memory_import(
            old.preview_id,
            [ClaimDecision(claim_id="old-memory", action="accept")],
        )
    confirmed = second.confirm_agent_memory_import(
        current.preview_id,
        [ClaimDecision(claim_id="current-memory", action="accept")],
    )

    connection = confirmed.memory_connections[0]
    assert connection.connected is True
    assert connection.generation == 3
    assert any(
        claim.confirmed and claim.value == "current private claim"
        for claim in confirmed.claims
    )
    assert not any(claim.value == "old private claim" for claim in confirmed.claims)


def test_concurrent_disconnect_wins_over_memory_confirmation(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    barrier = threading.Barrier(2)
    disconnect_written = threading.Event()
    first_store = BarrierProfileStore(path, barrier, disconnect_written)
    second_store = BarrierProfileStore(path, barrier, disconnect_written)
    first_store.initialize()
    first_store.save_profile(CandidateProfile())
    first = LocalProfileApplication(first_store)
    second = LocalProfileApplication(second_store)
    issued = first.preview_agent_memory_import(
        "career-agent", ["skills"], [_agent_candidate()]
    )
    first_store.arm(prefer_disconnect=True)
    second_store.arm(prefer_disconnect=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        confirm = pool.submit(
            first.confirm_agent_memory_import,
            issued.preview_id,
            [ClaimDecision(claim_id="memory-1", action="accept")],
        )
        disconnect = pool.submit(
            second.disconnect_agent_memory, "career-agent", False
        )
        disconnect.result(timeout=5)
        with pytest.raises(ApplicationError, match="memory_preview_invalid"):
            confirm.result(timeout=5)

    loaded = first_store.load_profile()
    assert loaded is not None
    assert loaded.memory_connections[0].connected is False
    assert not any(claim.confirmed for claim in loaded.claims)


def test_two_profile_applications_merge_distinct_confirmations(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    barrier = threading.Barrier(2)
    first_store = BarrierProfileStore(path, barrier)
    second_store = BarrierProfileStore(path, barrier)
    first_store.initialize()
    first_store.save_profile(CandidateProfile())
    first = LocalProfileApplication(first_store)
    second = LocalProfileApplication(second_store)
    first_preview = first.review_profile_claims(
        [_resume_candidate("claim-a", "skill", "Python")]
    )
    second_preview = second.review_profile_claims(
        [_resume_candidate("claim-b", "location", "Remote")]
    )
    first_store.arm()
    second_store.arm()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            first.confirm_profile_claims,
            first_preview.preview_id,
            [ClaimDecision(claim_id="claim-a", action="accept")],
        )
        second_future = pool.submit(
            second.confirm_profile_claims,
            second_preview.preview_id,
            [ClaimDecision(claim_id="claim-b", action="accept")],
        )
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    loaded = first_store.load_profile()
    assert loaded is not None
    assert {(claim.claim_type, claim.value) for claim in loaded.claims} == {
        ("skill", "Python"),
        ("location", "Remote"),
    }
    assert loaded.storage_revision == 2


def test_two_rubric_applications_merge_distinct_patches(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    barrier = threading.Barrier(2)
    first_store = BarrierRubricStore(path, barrier)
    second_store = BarrierRubricStore(path, barrier)
    first_store.initialize()
    rubric = load_default_rubric()
    first_store.save_rubric(rubric)
    first = LocalRubricApplication(first_store)
    second = LocalRubricApplication(second_store)
    fact_id = rubric.facts[0].id
    dimension_id = rubric.dimensions[0].id

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            first.update_rubric,
            RubricPatch(fact_updates={fact_id: {"name": "Updated fact"}}),
        )
        second_future = pool.submit(
            second.update_rubric,
            RubricPatch(
                dimension_updates={dimension_id: {"definition": "Updated dimension"}}
            ),
        )
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    loaded = first_store.load_rubric()
    assert loaded is not None
    assert next(item for item in loaded.facts if item.id == fact_id).name == "Updated fact"
    assert next(
        item for item in loaded.dimensions if item.id == dimension_id
    ).definition == "Updated dimension"
    assert loaded.storage_revision == 2


def test_two_rubric_applications_reject_overlapping_concurrent_patches(
    tmp_path,
) -> None:
    path = tmp_path / "shared.sqlite3"
    barrier = threading.Barrier(2)
    first_store = BarrierRubricStore(path, barrier)
    second_store = BarrierRubricStore(path, barrier)
    first_store.initialize()
    rubric = load_default_rubric()
    first_store.save_rubric(rubric)
    fact_id = rubric.facts[0].id
    applications = (
        LocalRubricApplication(first_store),
        LocalRubricApplication(second_store),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                application.update_rubric,
                RubricPatch(fact_updates={fact_id: {"name": name}}),
            )
            for application, name in zip(applications, ("First", "Second"), strict=True)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except ApplicationError as error:
                outcomes.append(error)

    assert sum(isinstance(item, Rubric) for item in outcomes) == 1
    conflicts = [item for item in outcomes if isinstance(item, ApplicationError)]
    assert len(conflicts) == 1
    assert conflicts[0].code == "rubric_update_conflict"
    loaded = first_store.load_rubric()
    assert loaded is not None
    assert loaded.storage_revision == 1


def test_two_rubric_applications_cannot_resolve_the_same_preference_twice(
    tmp_path,
) -> None:
    path = tmp_path / "shared-preference.sqlite3"
    setup_store = LocalStore(path)
    setup_store.initialize()
    application = LocalRubricApplication(setup_store)
    staged = application.stage_custom_preference(
        CustomPreferenceProposal(
            statement="Prefer customer discovery",
            topic_id="customer_discovery",
            category="role_content",
            name="Customer discovery",
            desired_value="Direct customer discovery",
            requirement_level="strong",
            evidence_guidance="Look for customer interviews",
            fact_ids=["product_customer_scope"],
        )
    )
    draft_id = staged.pending_preferences[0].id

    barrier = threading.Barrier(2)
    first = LocalRubricApplication(BarrierRubricStore(path, barrier))
    second = LocalRubricApplication(BarrierRubricStore(path, barrier))
    resolution = PreferenceResolution(action="accept_new")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(item.resolve_custom_preference, draft_id, resolution)
            for item in (first, second)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except ApplicationError as error:
                outcomes.append(error)

    assert sum(isinstance(item, Rubric) for item in outcomes) == 1
    errors = [item for item in outcomes if isinstance(item, ApplicationError)]
    assert len(errors) == 1
    assert errors[0].code == "invalid_custom_preference"
    loaded = setup_store.load_rubric()
    assert loaded is not None
    assert len(
        [item for item in loaded.dimensions if item.topic_id == "customer_discovery"]
    ) == 1


def test_invalid_preference_error_does_not_echo_private_text() -> None:
    store = MemoryStore(rubric=load_default_rubric())
    application = LocalRubricApplication(store)
    private_text = "Do not expose this preference"

    with pytest.raises(ApplicationError) as captured:
        application.resolve_custom_preference(
            private_text,
            PreferenceResolution(action="accept_new"),
        )

    assert captured.value.code == "invalid_custom_preference"
    assert private_text not in str(captured.value)


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

    store = BlockingFirstSaveStore(profile=CandidateProfile())
    application = LocalProfileApplication(store)
    first_preview = application.review_profile_claims(
        [_resume_candidate("claim-a", "skill", "Python")]
    )
    second_started = threading.Event()

    def confirm_second():
        second_started.set()
        preview = application.review_profile_claims(
            [_resume_candidate("claim-b", "location", "Remote")]
        )
        return application.confirm_profile_claims(
            preview.preview_id,
            [ClaimDecision(claim_id="claim-b", action="accept")]
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            application.confirm_profile_claims,
            first_preview.preview_id,
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

    assert len(tools) == 21


@pytest.mark.asyncio
async def test_workspace_errors_reuse_fixed_public_recovery_messages() -> None:
    class FailingWorkspaceService:
        def get_snapshot(self):
            raise WorkspaceError(WORKSPACE_STORAGE_FAILURE)

    server = build_server(
        MCPServices(
            profile=LocalProfileApplication(
                MemoryStore(profile=CandidateProfile())
            ),
            rubric=StaticRubricService(),
            analysis=StaticAnalysisService(),
            workspace=FailingWorkspaceService(),
        )
    )

    with pytest.raises(ToolError) as raised:
        await server.call_tool("get_workspace_snapshot", {})

    assert (
        "storage_failure: Local data could not be saved or loaded; retry the action."
        in str(raised.value)
    )


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
        await server.call_tool("review_profile_claims", {"proposals": []})
    _assert_safe_error(
        missing.value,
        "profile_not_initialized",
        "Initialize the candidate profile before this action.",
    )

    empty_profile.initialize_profile(CandidateProfile())
    issued = empty_profile.review_profile_claims(
        [_resume_candidate("private-claim-id", "skill", "secret-value")]
    )
    with pytest.raises(ToolError) as decisions:
        await server.call_tool(
            "confirm_profile_claims",
            {
                "preview_id": issued.preview_id,
                "decisions": [{"claim_id": "unknown-private-id", "action": "accept"}],
            },
        )
    _assert_safe_error(
        decisions.value,
        "profile_preview_invalid",
        "Review pending resume claims before confirming them.",
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
