import pytest

from ai_dream_job_engine import mcp_app
from ai_dream_job_engine.mcp_app import (
    MCP_APP_MIME_TYPE,
    WORKSPACE_RESOURCE_URI,
    MCPAppResourceError,
)
from tests.test_mcp_server import _server


@pytest.mark.asyncio
async def test_workspace_resource_uses_versioned_ui_uri_and_mcp_app_mime() -> None:
    server = _server()

    resources = {str(item.uri): item for item in await server.list_resources()}
    resource = resources[WORKSPACE_RESOURCE_URI]
    contents = await server.read_resource(WORKSPACE_RESOURCE_URI)

    assert resource.mime_type == MCP_APP_MIME_TYPE
    assert len(contents) == 1
    assert contents[0].mime_type == MCP_APP_MIME_TYPE
    assert contents[0].content == mcp_app.load_workspace_html()


@pytest.mark.asyncio
async def test_workspace_resource_is_self_contained_and_has_no_remote_domains() -> None:
    server = _server()
    resources = {str(item.uri): item for item in await server.list_resources()}
    resource = resources[WORKSPACE_RESOURCE_URI]
    html = mcp_app.load_workspace_html().lower()

    assert "<html" in html
    assert "http:" not in html
    assert "https:" not in html
    assert resource.meta == {
        "ui": {
            "csp": {
                "connectDomains": [],
                "resourceDomains": [],
                "frameDomains": [],
            }
        }
    }


@pytest.mark.asyncio
async def test_render_tools_keep_private_hydration_out_of_structured_content() -> None:
    server = _server()
    tools = {tool.name: tool for tool in await server.list_tools()}

    workspace = await server.call_tool("open_job_workspace", {})
    preview = await server.call_tool(
        "open_claim_review", {"preview_id": "p" * 32}
    )

    assert tools["open_job_workspace"].meta["ui"]["visibility"] == ["model", "app"]
    assert tools["open_claim_review"].meta["ui"]["visibility"] == ["model", "app"]
    assert "confirmed_claims" not in str(workspace.structured_content)
    assert "confirmed only after action" not in str(preview.structured_content)
    assert workspace.meta["aiDreamJobEngine"]["payload"]["view"] == "jobs"
    assert preview.meta["aiDreamJobEngine"]["payload"]["claim_preview"]["candidates"][0]["id"] == "claim-1"


@pytest.mark.asyncio
async def test_private_ui_tools_are_app_only_and_return_payload_in_meta() -> None:
    server = _server()
    tools = {tool.name: tool for tool in await server.list_tools()}

    for name in {
        "get_workspace_snapshot",
        "get_workspace_job",
        "set_job_review_status",
        "get_candidate_profile",
        "get_claim_preview",
        "get_rubric",
        "update_rubric",
        "resolve_custom_preference",
        "confirm_profile_claims",
        "confirm_agent_memory_import",
        "disconnect_agent_memory",
    }:
        assert tools[name].meta["ui"]["visibility"] == ["app"]

    result = await server.call_tool("get_candidate_profile", {})
    assert result.structured_content == {"status": "ok"}
    assert result.meta["aiDreamJobEngine"]["payload"]["id"] == "default"


@pytest.mark.asyncio
async def test_preview_orchestration_tools_keep_private_payload_in_app_meta() -> None:
    tools = {tool.name: tool for tool in await _server().list_tools()}

    for name in (
        "review_profile_claims",
        "preview_agent_memory_import",
        "stage_custom_preference",
    ):
        assert tools[name].meta["ui"]["visibility"] == ["model", "app"]
        assert tools[name].output_schema is None


@pytest.mark.asyncio
async def test_staged_preference_is_returned_only_in_private_app_payload() -> None:
    server = _server()

    result = await server.call_tool(
        "stage_custom_preference",
        {
            "proposal": {
                "statement": "Private preference text",
                "topic_id": "team_style",
                "category": "growth_team",
                "name": "Team style",
                "desired_value": "Collaborative team",
                "requirement_level": "normal",
                "evidence_guidance": "Look for collaboration evidence",
            }
        },
    )

    assert result.structured_content == {"status": "ok"}
    assert "Private preference text" not in str(result.structured_content)
    assert result.meta["aiDreamJobEngine"]["payload"]["version"] == "rubric-v1"


@pytest.mark.asyncio
async def test_open_workspace_tool_links_only_to_workspace_resource() -> None:
    server = _server()
    tools = {item.name: item for item in await server.list_tools()}

    assert tools["open_job_workspace"].meta["ui"] == {
        "resourceUri": WORKSPACE_RESOURCE_URI,
        "visibility": ["model", "app"],
    }
    assert tools["open_claim_review"].meta == {
        "ui": {
            "resourceUri": WORKSPACE_RESOURCE_URI,
            "visibility": ["model", "app"],
        }
    }

    workspace = await server.call_tool("open_job_workspace", {})
    claim_review = await server.call_tool(
        "open_claim_review", {"preview_id": "p" * 32}
    )

    assert workspace.structured_content["view"] == "jobs"
    assert workspace.meta["aiDreamJobEngine"]["payload"]["snapshot"]["profile_ready"] is True
    assert workspace.meta["aiDreamJobEngine"]["payload"]["claim_preview"] is None
    assert claim_review.structured_content["view"] == "claim_review"
    assert claim_review.meta["aiDreamJobEngine"]["payload"]["claim_preview"]["preview_id"] == "p" * 32
    assert claim_review.meta["aiDreamJobEngine"]["payload"]["claim_preview"]["candidates"][0]["confirmed"] is False


@pytest.mark.asyncio
async def test_claim_confirmation_tools_require_explicit_decisions_and_do_not_open_ui() -> None:
    tools = {item.name: item for item in await _server().list_tools()}

    for name in ("confirm_profile_claims", "confirm_agent_memory_import"):
        tool = tools[name]
        assert set(tool.input_schema["required"]) == {"preview_id", "decisions"}
        assert tool.meta["ui"]["visibility"] == ["app"]
        assert "explicit per-claim" in tool.description


@pytest.mark.asyncio
async def test_thread_entrypoint_metadata_is_present() -> None:
    tools = {item.name: item for item in await _server().list_tools()}

    assert tools["open_job_workspace"].meta["openai/ui"] == {
        "entrypoints": [{"type": "thread"}]
    }
    assert "openai/ui" not in tools["open_claim_review"].meta


@pytest.mark.asyncio
async def test_existing_data_tools_have_no_output_template() -> None:
    tools = {item.name: item for item in await _server().list_tools()}
    render_tools = {"open_job_workspace", "open_claim_review"}

    for name, tool in tools.items():
        if name in render_tools:
            continue
        if tool.meta and "ui" in tool.meta:
            assert "resourceUri" not in tool.meta["ui"]
        assert not tool.meta or "openai/outputTemplate" not in tool.meta


def test_missing_ui_bundle_fails_with_fixed_public_error(monkeypatch) -> None:
    def missing_package(_package):
        raise FileNotFoundError("/private/user/path/workspace-v1.html")

    monkeypatch.setattr(mcp_app.resources, "files", missing_package)

    with pytest.raises(
        MCPAppResourceError,
        match="ui_bundle_unavailable: The local interface is unavailable; use the conversation tools.",
    ):
        mcp_app.load_workspace_html()


@pytest.mark.asyncio
async def test_missing_ui_bundle_does_not_disable_existing_tools(monkeypatch) -> None:
    def missing_package(_package):
        raise FileNotFoundError("/private/user/path/workspace-v1.html")

    monkeypatch.setattr(mcp_app.resources, "files", missing_package)
    server = _server()

    tools = {item.name for item in await server.list_tools()}
    snapshot = await server.call_tool("get_workspace_snapshot", {})

    assert "get_workspace_snapshot" in tools
    assert snapshot.is_error is False
    assert snapshot.meta["aiDreamJobEngine"]["payload"]["profile_ready"] is True
