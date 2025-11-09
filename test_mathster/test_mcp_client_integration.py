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
    CodexMCPClient,
    GeminiMCPClient,
)
from mathster.agents_direct import ClaudeDirectClient


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
    [DEPRECATED] Obtain server command + args from an environment variable.

    This function is no longer used since MCP clients now have hardcoded commands.
    Kept for backward compatibility but can be removed in future.

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
    """Exercise Gemini MCP client end-to-end with Flash model."""

    _skip_unless_integration()
    _require_env("GEMINI_API_KEY")

    async def _run():
        client = GeminiMCPClient()  # Uses hardcoded command
        response = await client.ask(
            prompt="Reply with the exact string GEMINI_OK (no extra words).",
            model="gemini-2.5-flash",
        )
        _assert_marker_in_response(response, "GEMINI_OK")

    asyncio.run(_run())


@pytest.mark.integration
def test_codex_client_live_roundtrip():
    """Exercise Codex MCP client end-to-end with low reasoning effort."""

    _skip_unless_integration()
    _require_env("OPENAI_API_KEY")

    async def _run():
        client = CodexMCPClient()  # Uses hardcoded command
        response = await client.ask(
            prompt="Respond with CODEX_OK exactly.",
            model="gpt-5-codex",
            reasoning_effort="low",
        )
        _assert_marker_in_response(response, "CODEX_OK")

    asyncio.run(_run())


@pytest.mark.integration
def test_claude_client_live_roundtrip():
    """Exercise Claude CLI client end-to-end.

    Note: This now uses direct CLI invocation with `claude --print --model [model] [prompt]`
    instead of MCP server architecture.
    """

    _skip_unless_integration()
    _require_env("ANTHROPIC_API_KEY")

    async def _run():
        client = ClaudeDirectClient()  # Uses direct CLI command
        response = await client.ask(
            prompt="Reply with the exact string CLAUDE_OK (no extra words).",
            model="sonnet",  # Use alias instead of full model name
        )
        _assert_marker_in_response(response, "CLAUDE_OK")

    asyncio.run(_run())


@pytest.mark.integration
@pytest.mark.parametrize("model", [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
])
def test_gemini_models(model):
    """Test Gemini MCP client with Flash model variants (not Pro)."""

    _skip_unless_integration()
    _require_env("GEMINI_API_KEY")

    async def _run():
        client = GeminiMCPClient()
        response = await client.ask(
            prompt=f"Reply with the exact string GEMINI_{model.upper().replace('-', '_').replace('.', '_')}_OK",
            model=model,
        )
        # Check for model-specific marker or generic OK
        assert "OK" in response.strip(), f"Expected 'OK' in response for {model}: {response}"

    asyncio.run(_run())


@pytest.mark.integration
@pytest.mark.parametrize("reasoning_effort", [
    "low",
    "medium",
])
def test_codex_reasoning_effort(reasoning_effort):
    """Test Codex MCP client with low/medium reasoning effort (not high)."""

    _skip_unless_integration()
    _require_env("OPENAI_API_KEY")

    async def _run():
        client = CodexMCPClient()
        response = await client.ask(
            prompt=f"Respond with CODEX_{reasoning_effort.upper()}_OK exactly.",
            model="gpt-5-codex",
            reasoning_effort=reasoning_effort,
        )
        assert "OK" in response.strip(), f"Expected 'OK' in response with {reasoning_effort}: {response}"

    asyncio.run(_run())


@pytest.mark.integration
@pytest.mark.parametrize("model", [
    "sonnet",
    "haiku",
])
def test_claude_models(model):
    """Test Claude CLI client with different model variants (using aliases)."""

    _skip_unless_integration()
    _require_env("ANTHROPIC_API_KEY")

    async def _run():
        client = ClaudeDirectClient()
        response = await client.ask(
            prompt=f"Reply with the exact string CLAUDE_{model.upper()}_OK",
            model=model,
        )
        # Check for model-specific marker or generic OK
        assert "OK" in response.strip(), f"Expected 'OK' in response for {model}: {response}"

    asyncio.run(_run())
