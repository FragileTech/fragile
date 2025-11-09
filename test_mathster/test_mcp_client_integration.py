"""Integration tests for mathster.mcp_client using real MCP servers.

These tests intentionally exercise the live MCP bridges (Gemini, Codex, Claude).
They require the corresponding CLIs, API keys, and an opt-in flag to avoid
accidental external calls during normal CI runs.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Sequence

import pytest
from dotenv import load_dotenv

from mathster.mcp_client import (
    ClaudeCodeMCPClient,
    CodexMCPClient,
    GeminiMCPClient,
)


# Ensure .env-based settings (API keys, MCP commands) are available during tests.
load_dotenv()


def _integration_enabled() -> bool:
    return os.getenv("RUN_MCP_INTEGRATION", "").lower() in {"1", "true", "yes"}


def _skip_unless_integration():
    if not _integration_enabled():
        pytest.skip("Set RUN_MCP_INTEGRATION=1 to enable live MCP integration tests.")


def _require_env(var_name: str):
    if not os.getenv(var_name):
        pytest.skip(f"{var_name} is not set; cannot run live MCP test.")


def _assert_marker_in_response(response: str, marker: str):
    assert marker in response.strip(), f"Expected marker '{marker}' in response: {response}"


def _server_params(env_var: str) -> tuple[str, Sequence[str]]:
    """
    Obtain server command + args from an environment variable.

    The env var should contain the full command (e.g., "gemini mcp-server --stdio").
    """
    value = os.getenv(env_var)
    if not value:
        pytest.skip(
            f"{env_var} is not configured. Provide the exact MCP server command "
            f"(e.g., 'gemini mcp-server --stdio') to run integration tests."
        )

    parts = shlex.split(value)
    if not parts:
        pytest.skip(f"{env_var} is empty after parsing.")
    return parts[0], parts[1:]


@pytest.mark.integration
def test_gemini_client_live_roundtrip():
    """Exercise Gemini MCP client end-to-end."""

    _skip_unless_integration()
    _require_env("GEMINI_API_KEY")
    server_command, server_args = _server_params("GEMINI_MCP_COMMAND")

    async def _run():
        client = GeminiMCPClient(server_command=server_command, server_args=list(server_args))
        response = await client.ask(
            prompt="Reply with the exact string GEMINI_OK (no extra words).",
            model="gemini-2.5-pro",
        )
        _assert_marker_in_response(response, "GEMINI_OK")

    asyncio.run(_run())


@pytest.mark.integration
def test_codex_client_live_roundtrip():
    """Exercise Codex MCP client end-to-end."""

    _skip_unless_integration()
    _require_env("OPENAI_API_KEY")
    server_command, server_args = _server_params("CODEX_MCP_COMMAND")

    async def _run():
        client = CodexMCPClient(server_command=server_command, server_args=list(server_args))
        response = await client.ask(
            prompt="Respond with CODEX_OK exactly.",
            model="gpt-5-codex",
        )
        _assert_marker_in_response(response, "CODEX_OK")

    asyncio.run(_run())


@pytest.mark.integration
def test_claude_client_live_roundtrip():
    """Exercise Claude Code MCP client end-to-end."""

    _skip_unless_integration()
    _require_env("ANTHROPIC_API_KEY")
    server_command, server_args = _server_params("CLAUDE_MCP_COMMAND")

    async def _run():
        client = ClaudeCodeMCPClient(server_command=server_command, server_args=list(server_args))
        response = await client.ask(
            prompt="Output CLAUDE_OK exactly.",
            model="claude-3-5-sonnet",
        )
        _assert_marker_in_response(response, "CLAUDE_OK")

    asyncio.run(_run())
