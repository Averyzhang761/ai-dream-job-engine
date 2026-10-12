import json
from pathlib import Path
import platform
import sys

import pytest
from packaging.markers import Marker

from ai_dream_job_engine.mcp_server import MCPServices, build_server
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimPreviewView,
    ClaimSource,
    JobReviewStatus,
    Rubric,
    WorkspaceSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "initialize_profile",
    "review_profile_claims",
    "confirm_profile_claims",
    "preview_agent_memory_import",
    "confirm_agent_memory_import",
    "disconnect_agent_memory",
    "get_rubric",
    "update_rubric",
    "analyze_job",
    "recompute_decision",
    "get_decision_trace",
    "list_analyzed_jobs",
    "get_workspace_snapshot",
    "get_workspace_job",
    "set_job_review_status",
    "get_candidate_profile",
    "get_claim_preview",
    "open_job_workspace",
    "open_claim_review",
}


class FakeProfileService:
    def initialize_profile(self, profile=None):
        return profile or CandidateProfile()

    def review_profile_claims(self, proposals):
        raise AssertionError("not used")

    def confirm_profile_claims(self, preview_id, decisions):
        raise AssertionError("not used")

    def preview_agent_memory_import(self, source_agent, scope, candidates):
        raise AssertionError("not used")

    def confirm_agent_memory_import(self, preview, decisions):
        raise AssertionError("not used")

    def disconnect_agent_memory(self, source_agent, confirmed):
        raise AssertionError("not used")


class FakeRubricService:
    def get_rubric(self):
        return Rubric(version="rubric-v1")

    def update_rubric(self, patch):
        return Rubric(version="rubric-v1")


class FakeAnalysisService:
    def analyze_job(self, job_description):
        return AnalysisRecord(
            id="analysis-1",
            input_hash="sha256:test",
            status="failed",
            latency_ms=0,
            error_category="test",
            error="safe test result",
        )

    def recompute_decision(self, analysis_id):
        raise AssertionError("not used")

    def get_decision_trace(self, analysis_id):
        raise AssertionError("not used")

    def list_analyzed_jobs(self):
        return []


class FakeWorkspaceService:
    def get_snapshot(self):
        return WorkspaceSnapshot(
            profile_ready=True,
            rubric_ready=False,
            setup_actions=["initialize_rubric"],
            confirmed_claims=[],
            jobs=[],
        )

    def get_job_detail(self, analysis_id):
        raise AssertionError("not used")

    def set_review_status(self, analysis_id, status):
        raise AssertionError("not used")

    def get_profile(self):
        return CandidateProfile()

    def get_claim_preview(self, preview_id):
        return ClaimPreviewView(
            preview_id=preview_id,
            kind="profile_claims",
            candidates=[
                CandidateClaim(
                    id="claim-1",
                    claim_type="python",
                    value="confirmed only after action",
                    source=ClaimSource.RESUME_EVIDENCE,
                    source_quote="Python evidence",
                )
            ],
        )


def _server():
    return build_server(
        MCPServices(
            profile=FakeProfileService(),
            rubric=FakeRubricService(),
            analysis=FakeAnalysisService(),
            workspace=FakeWorkspaceService(),
        )
    )


@pytest.mark.asyncio
async def test_expected_tool_names_are_registered() -> None:
    tools = await _server().list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_tool_schemas_use_domain_contracts() -> None:
    tools = {tool.name: tool for tool in await _server().list_tools()}

    initialize_schema = tools["initialize_profile"].input_schema
    confirm_schema = tools["confirm_profile_claims"].input_schema
    rubric_schema = tools["update_rubric"].input_schema
    assert "CandidateProfile" in initialize_schema["$defs"]
    assert "ClaimDecision" in confirm_schema["$defs"]
    assert set(confirm_schema["required"]) == {"preview_id", "decisions"}
    assert "RubricPatch" in rubric_schema["$defs"]
    review_schema = tools["set_job_review_status"].input_schema
    assert "JobReviewStatus" in review_schema["$defs"]
    assert set(review_schema["required"]) == {"analysis_id", "status"}
    assert tools["get_workspace_job"].output_schema is None


@pytest.mark.asyncio
async def test_workspace_tools_return_typed_structured_output() -> None:
    snapshot = await _server().call_tool("get_workspace_snapshot", {})
    profile = await _server().call_tool("get_candidate_profile", {})
    preview = await _server().call_tool(
        "get_claim_preview", {"preview_id": "p" * 32}
    )

    assert snapshot.is_error is False
    assert snapshot.structured_content == {"status": "ok"}
    assert snapshot.meta["aiDreamJobEngine"]["payload"]["profile_ready"] is True
    assert snapshot.meta["aiDreamJobEngine"]["payload"]["setup_actions"] == ["initialize_rubric"]
    assert profile.meta["aiDreamJobEngine"]["payload"]["id"] == "default"
    assert preview.meta["aiDreamJobEngine"]["payload"]["kind"] == "profile_claims"
    assert preview.meta["aiDreamJobEngine"]["payload"]["candidates"][0]["id"] == "claim-1"


@pytest.mark.asyncio
async def test_memory_preview_requires_explicit_source_and_scope() -> None:
    tools = {tool.name: tool for tool in await _server().list_tools()}
    schema = tools["preview_agent_memory_import"].input_schema
    confirm_schema = tools["confirm_agent_memory_import"].input_schema

    assert {"source_agent", "scope", "candidates"} <= set(schema["required"])
    assert schema["properties"]["source_agent"]["minLength"] == 1
    assert schema["properties"]["scope"]["minItems"] == 1
    assert set(confirm_schema["required"]) == {"preview_id", "decisions"}
    assert "preview" not in confirm_schema["properties"]


@pytest.mark.asyncio
async def test_analyze_tool_returns_structured_result() -> None:
    result = await _server().call_tool(
        "analyze_job", {"job_description": "Public job description"}
    )

    assert result.is_error is False
    assert result.structured_content["id"] == "analysis-1"
    assert result.structured_content["status"] == "failed"


def test_plugin_manifest_points_to_mcp_config_and_skill() -> None:
    manifest = json.loads((ROOT / ".codex-plugin/plugin.json").read_text())

    assert manifest == {
        "name": "ai-dream-job-engine",
        "version": "0.1.0",
        "skills": "./skills/",
        "mcpServers": "./.mcp.json",
    }


def test_skill_declares_host_owned_automation_boundary() -> None:
    skill = (ROOT / "skills/ai-dream-job/SKILL.md").read_text()

    assert "host Agent owns schedules and notifications" in skill
    assert "no_change" in skill
    assert "explicit user confirmation" in skill
    assert "public job description only" in skill
    assert "does not auto-apply" in skill


def test_start_script_uses_repo_local_project() -> None:
    script = (ROOT / "scripts/start-mcp.sh").read_text()

    assert 'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"' in script
    assert 'uv run --project "$REPO_ROOT"' in script
    assert 'PYTHONPATH="$REPO_ROOT/src' in script
    assert "/Users/" not in script


def test_mcp_config_declares_repo_root_working_directory() -> None:
    config = json.loads((ROOT / ".mcp.json").read_text())
    server = config["mcpServers"]["ai-dream-job-engine"]

    assert server == {
        "command": "./scripts/start-mcp.sh",
        "args": [],
        "cwd": ".",
    }


def test_mcp_dependency_uses_only_platform_bound_cryptography_cap() -> None:
    project = (ROOT / "pyproject.toml").read_text()

    assert '"mcp[cli]>=2,<3"' in project
    assert (
        '"cryptography<49; sys_platform == \'darwin\' '
        'and platform_machine == \'x86_64\'"'
    ) in project
    assert 'cryptography<47' not in project
    marker_applies = Marker(
        "sys_platform == 'darwin' and platform_machine == 'x86_64'"
    ).evaluate()
    assert marker_applies is (
        sys.platform == "darwin" and platform.machine() == "x86_64"
    )
