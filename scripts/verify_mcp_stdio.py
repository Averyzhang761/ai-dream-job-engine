#!/usr/bin/env python3
"""Verify the packaged MCP server over its real stdio transport."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


RESOURCE_URI = "ui://ai-dream-job-engine/workspace-v1.html"
RESOURCE_MIME = "text/html;profile=mcp-app"
RENDER_TOOLS = ["open_claim_review", "open_job_workspace"]


def _server_environment(database_path: Path) -> dict[str, str]:
    environment = {
        "AI_DREAM_JOB_DATA_PATH": str(database_path),
        "PATH": os.environ.get("PATH", ""),
    }
    for name in ("HOME", "LANG", "LC_ALL", "TMPDIR", "UV_CACHE_DIR"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


async def verify(
    repo_root: Path,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ai-dream-job-mcp-") as temp_dir:
        if python_executable is None:
            command = str(repo_root / "scripts" / "start-mcp.sh")
            args: list[str] = []
        else:
            command = str(python_executable.absolute())
            args = ["-m", "ai_dream_job_engine.mcp_server"]
        parameters = StdioServerParameters(
            command=command,
            args=args,
            cwd=repo_root,
            env=_server_environment(Path(temp_dir) / "verification.sqlite3"),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                opened = await session.call_tool("open_job_workspace", {})
                snapshot = await session.call_tool("get_workspace_snapshot", {})
                resource = await session.read_resource(RESOURCE_URI)

    tool_names = {tool.name for tool in tools.tools}
    content = resource.contents[0]
    result = {
        "non_ui_tool_present": "get_workspace_snapshot" in tool_names,
        "open_workspace_is_error": opened.is_error,
        "render_tools": sorted(tool_names.intersection(RENDER_TOOLS)),
        "resource_mime": content.mime_type,
        "resource_nonblank": bool(getattr(content, "text", "").strip()),
        "resources": sorted(str(item.uri) for item in resources.resources),
        "snapshot_is_error": snapshot.is_error,
        "tool_count": len(tool_names),
    }
    expected = {
        "non_ui_tool_present": True,
        "open_workspace_is_error": False,
        "render_tools": RENDER_TOOLS,
        "resource_mime": RESOURCE_MIME,
        "resource_nonblank": True,
        "resources": [RESOURCE_URI],
        "snapshot_is_error": False,
        "tool_count": 19,
    }
    if result != expected:
        raise RuntimeError(
            "MCP stdio verification failed:\n"
            + json.dumps({"expected": expected, "actual": result}, indent=2)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the deterministic JSON transcript to this path.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Launch the MCP module with this Python instead of start-mcp.sh.",
    )
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    result = anyio.run(verify, repo_root, arguments.python)
    transcript = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(transcript, encoding="utf-8")
    sys.stdout.write(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
