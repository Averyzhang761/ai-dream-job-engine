#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
BUNDLE_ROOT="$REPO_ROOT/dist/codex-plugin"

if [ -d "$BUNDLE_ROOT" ]; then
  find "$BUNDLE_ROOT" -depth -delete
fi

mkdir -p "$BUNDLE_ROOT/scripts"
cp -R "$REPO_ROOT/.codex-plugin" "$BUNDLE_ROOT/.codex-plugin"
cp -R "$REPO_ROOT/skills" "$BUNDLE_ROOT/skills"
cp -R "$REPO_ROOT/src" "$BUNDLE_ROOT/src"
find "$BUNDLE_ROOT/src" -type d \( -name '__pycache__' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
find "$BUNDLE_ROOT/src" -type f -name '*.pyc' -delete
cp "$REPO_ROOT/.mcp.json" "$BUNDLE_ROOT/.mcp.json"
cp "$REPO_ROOT/LICENSE" "$BUNDLE_ROOT/LICENSE"
cp "$REPO_ROOT/pyproject.toml" "$BUNDLE_ROOT/pyproject.toml"
cp "$REPO_ROOT/uv.lock" "$BUNDLE_ROOT/uv.lock"
cp "$REPO_ROOT/scripts/start-mcp.sh" "$BUNDLE_ROOT/scripts/start-mcp.sh"
cp "$REPO_ROOT/scripts/verify_mcp_stdio.py" "$BUNDLE_ROOT/scripts/verify_mcp_stdio.py"

printf 'Built Codex plugin bundle: %s\n' "$BUNDLE_ROOT"
