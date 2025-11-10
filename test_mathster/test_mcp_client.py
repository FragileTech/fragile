"""Tests for mathster.mcp_client (isolated from fragile test fixtures)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# Ensure DSPy cache is writable inside the sandbox before importing dspy.
_DSPY_CACHE = Path(os.environ.get("DSPY_CACHEDIR", Path.cwd() / ".dspy_cache"))
_DSPY_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DSPY_CACHEDIR", str(_DSPY_CACHE))

try:
    import dspy
except ImportError:  # pragma: no cover - fallback when DSPy not installed

    class _FallbackBaseLM:
        def __init__(self, model: str, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def forward(self, prompt=None, messages=None, **kwargs):
            msg = "This fallback BaseLM cannot be used directly."
            raise NotImplementedError(msg)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    dspy = SimpleNamespace(BaseLM=_FallbackBaseLM)

from mathster.mcp_client import (  # noqa: E402
    BaseMCPClient,
    ClaudeCodeMCPClient,
    CodexMCPClient,
    GeminiMCPClient,
    MCPConnectionError,
    sync_ask_claude,
    sync_ask_codex,
    sync_ask_gemini,
)


def test_gemini_client_auto_discovers_and_sets_default_args(monkeypatch):
    """Gemini client should auto-discover binary and enforce MCP args."""

    monkeypatch.setattr(
        GeminiMCPClient,
        "_discover_server",
        lambda self: "/fake/bin/gemini",
    )

    client = GeminiMCPClient(server_command=None)

    assert client.server_command == "/fake/bin/gemini"
    assert client.server_args == ["mcp"]


def test_env_overrides_preserve_existing_keys():
    """Providing an API key should not drop other env overrides."""

    client = GeminiMCPClient(
        server_command="gemini",
        auto_discover=False,
        api_key="dummy",
        env={"EXTRA": "1"},
    )

    assert client.env == {"EXTRA": "1", "GEMINI_API_KEY": "dummy"}


def test_claude_client_auto_discovers_and_sets_default_args(monkeypatch):
    """Claude client should auto-discover CLI and set MCP args."""

    monkeypatch.setattr(
        ClaudeCodeMCPClient,
        "_discover_server",
        lambda self: "/fake/bin/claude",
    )

    client = ClaudeCodeMCPClient(server_command=None)

    assert client.server_command == "/fake/bin/claude"
    assert client.server_args == ["mcp"]


def test_claude_env_overrides_respected():
    """ANTHROPIC API key should merge with other env overrides."""

    client = ClaudeCodeMCPClient(
        server_command="claude",
        auto_discover=False,
        api_key="anthropic-key",
        env={"OTHER": "1"},
    )

    assert client.env == {"OTHER": "1", "ANTHROPIC_API_KEY": "anthropic-key"}


def test_gemini_ask_invokes_mcp_tool(monkeypatch):
    """ask() should call into the MCP tool with the right payload."""

    calls = []

    class FakeSession:
        async def call_tool(self, tool_name, arguments):
            calls.append((tool_name, arguments))
            return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    @asynccontextmanager
    async def fake_connect(self):
        yield FakeSession()

    async def _run():
        monkeypatch.setattr(BaseMCPClient, "_connect", fake_connect, raising=False)

        client = GeminiMCPClient(server_command="gemini", auto_discover=False)

        result = await client.ask("hello", model="gemini-2.5-pro")

        assert result == "ok"
        assert calls == [("ask-gemini", {"model": "gemini-2.5-pro", "prompt": "hello"})]

    asyncio.run(_run())


def test_connect_timeout_raises_clear_error(monkeypatch):
    """If the MCP server never initializes, raise MCPConnectionError quickly."""

    class HangingClientSession:
        def __init__(self, _read, _write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            await asyncio.sleep(0.05)

    @asynccontextmanager
    async def fake_stdio(_params):
        yield object(), object()

    async def _run():
        monkeypatch.setattr("mathster.mcp_client.ClientSession", HangingClientSession)
        monkeypatch.setattr("mathster.mcp_client.stdio_client", fake_stdio)

        client = GeminiMCPClient(
            server_command="gemini",
            auto_discover=False,
            connect_timeout=0.01,
        )

        with pytest.raises(MCPConnectionError):
            await client.list_tools()

    asyncio.run(_run())


def test_sync_wrappers_can_back_dspy_lm(monkeypatch):
    """sync_ask_* helpers should be usable inside custom DSPy BaseLM subclasses."""

    monkeypatch.setattr(GeminiMCPClient, "__init__", lambda self, **kwargs: None)
    gemini_mock = AsyncMock(return_value="Gemini processed prompt")
    monkeypatch.setattr(GeminiMCPClient, "ask", gemini_mock)

    monkeypatch.setattr(CodexMCPClient, "__init__", lambda self, **kwargs: None)
    codex_mock = AsyncMock(return_value="Codex answered prompt")
    monkeypatch.setattr(CodexMCPClient, "ask", codex_mock)

    monkeypatch.setattr(ClaudeCodeMCPClient, "__init__", lambda self, **kwargs: None)
    claude_mock = AsyncMock(return_value="Claude reviewed prompt")
    monkeypatch.setattr(ClaudeCodeMCPClient, "ask", claude_mock)

    class MCPDemoLM(dspy.BaseLM):
        """Minimal BaseLM wrapper that calls a synchronous MCP helper."""

        def __init__(self, sync_callable, **kwargs):
            super().__init__(model="test/provider", **kwargs)
            self._sync_callable = sync_callable

        def forward(self, prompt=None, messages=None, **kwargs):
            question = prompt or (messages[-1]["content"] if messages else "")
            text = self._sync_callable(question)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
            )

    gemini_lm = MCPDemoLM(lambda prompt: sync_ask_gemini(prompt, server_command="gemini"))
    codex_lm = MCPDemoLM(lambda prompt: sync_ask_codex(prompt, server_command="codex"))
    claude_lm = MCPDemoLM(
        lambda prompt: sync_ask_claude(prompt, server_command="claude", tool_name="ask-claude")
    )

    gemini_result = gemini_lm.forward(prompt="Gemini?")
    codex_result = codex_lm.forward(prompt="Codex?")
    claude_result = claude_lm.forward(prompt="Claude?")

    assert gemini_result.choices[0].message.content == "Gemini processed prompt"
    assert codex_result.choices[0].message.content == "Codex answered prompt"
    assert claude_result.choices[0].message.content == "Claude reviewed prompt"

    gemini_mock.assert_awaited_once_with(prompt="Gemini?", model="gemini-2.5-pro")
    codex_mock.assert_awaited_once_with(prompt="Codex?", model="gpt-5-codex")
    claude_mock.assert_awaited_once_with(prompt="Claude?", model="claude-3-5-sonnet")


def test_codex_ask_uses_default_tool(monkeypatch):
    """Codex ask should default to the generic 'ask' MCP tool."""

    captured = {}

    class FakeSession:
        async def call_tool(self, tool_name, arguments):
            captured["tool"] = tool_name
            captured["arguments"] = arguments
            return SimpleNamespace(content=[SimpleNamespace(text="codex")])

    @asynccontextmanager
    async def fake_connect(self):
        yield FakeSession()

    async def _run():
        monkeypatch.setattr(BaseMCPClient, "_connect", fake_connect, raising=False)

        client = CodexMCPClient(server_command="codex", auto_discover=False)

        answer = await client.ask(prompt="explain?", model="gpt-5-codex")

        assert answer == "codex"
        assert captured == {
            "tool": "ask",
            "arguments": {"model": "gpt-5-codex", "prompt": "explain?"},
        }

    asyncio.run(_run())


def test_claude_ask_uses_custom_tool(monkeypatch):
    """Claude ask should respect the configured MCP tool name."""

    captured = {}

    class FakeSession:
        async def call_tool(self, tool_name, arguments):
            captured["tool"] = tool_name
            captured["arguments"] = arguments
            return SimpleNamespace(content=[SimpleNamespace(text="claude")])

    @asynccontextmanager
    async def fake_connect(self):
        yield FakeSession()

    async def _run():
        monkeypatch.setattr(BaseMCPClient, "_connect", fake_connect, raising=False)
        client = ClaudeCodeMCPClient(
            server_command="claude",
            auto_discover=False,
            tool_name="custom-claude",
        )
        answer = await client.ask(prompt="review?", model="claude-3-5-sonnet")
        assert answer == "claude"
        assert captured == {
            "tool": "custom-claude",
            "arguments": {"model": "claude-3-5-sonnet", "prompt": "review?"},
        }

    asyncio.run(_run())
