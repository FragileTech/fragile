"""
Low-level MCP client for direct server invocation.

This module provides async MCP clients that communicate with MCP servers
via stdio transport. It's designed for standalone Python scripts that need
to invoke MCP servers outside of Claude Code's runtime.

Key Components:
    - BaseMCPClient: Abstract base for MCP clients
    - GeminiMCPClient: Client for Gemini API via gemini-cli MCP server
    - CodexMCPClient: Client for OpenAI/Codex via MCP server
    - MCPConnectionError: Exception for connection failures

Usage Example:
    ```python
    import asyncio
    from mathster.mcp_client import GeminiMCPClient


    async def main():
        # Create client (will auto-discover server or use provided path)
        client = GeminiMCPClient(server_command="gemini-cli")

        # Invoke Gemini
        response = await client.ask(
            prompt="Explain the Keystone Principle", model="gemini-2.5-pro"
        )
        print(response)


    asyncio.run(main())
    ```

Server Discovery:
    The client attempts to find MCP servers in this order:
    1. Provided server_command parameter
    2. Common installation locations:
       - ~/.local/bin/
       - /usr/local/bin/
       - npm global bin directory
    3. Raises MCPConnectionError if not found

Requirements:
    - MCP SDK (mcp package) installed
    - MCP server executable (gemini-cli, codex, etc.) installed
    - Server configured with necessary API keys
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Awaitable, TypeVar

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


logger = logging.getLogger(__name__)
T = TypeVar("T")

# Load environment variables early so GEMINI/OPENAI keys from .env are available.
load_dotenv()


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""


class BaseMCPClient:
    """
    Base class for MCP clients.

    Provides common functionality for connecting to and communicating
    with MCP servers via stdio transport.

    Attributes:
        server_command: Command to start MCP server
        server_args: Arguments to pass to server
        env: Environment variables for server process
        session: Active MCP session (when connected)
    """

    def __init__(
        self,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        auto_discover: bool = True,
        connect_timeout: float = 20.0,
    ):
        """
        Initialize MCP client.

        Args:
            server_command: Command to execute MCP server (e.g., "gemini-cli")
                If None and auto_discover=True, attempts to find server
            server_args: Arguments to pass to server (e.g., ["mcp-server"])
            env: Environment variable overrides for server process
            auto_discover: If True, attempts to find server in common locations
            connect_timeout: Seconds to wait for MCP server initialization

        Raises:
            MCPConnectionError: If server not found and auto_discover fails
        """
        self.server_command = server_command
        self.server_args = list(server_args or [])
        self.env = env.copy() if env else None
        self.connect_timeout = connect_timeout
        self._session: ClientSession | None = None

        if self.server_command is None and auto_discover:
            self.server_command = self._discover_server()

        if self.server_command is None:
            msg = (
                "MCP server not found. Please install the server or "
                "provide server_command parameter."
            )
            raise MCPConnectionError(msg)

    def _discover_server(self) -> str | None:
        """
        Attempt to discover MCP server in common locations.

        Override this in subclasses to provide server-specific discovery logic.

        Returns:
            str: Path to server executable, or None if not found
        """
        return None

    def _compose_env(self) -> dict[str, str] | None:
        """
        Merge overrides with the current environment for subprocess execution.
        """
        if self.env is None:
            return None

        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    @asynccontextmanager
    async def _connect(self):
        """
        Create connection to MCP server.

        Yields:
            ClientSession: Active MCP session

        Raises:
            MCPConnectionError: If connection fails
        """
        server_params = StdioServerParameters(
            command=self.server_command, args=self.server_args, env=self._compose_env()
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=self.connect_timeout)
                    except asyncio.TimeoutError as exc:
                        raise MCPConnectionError(
                            f"MCP server '{self.server_command}' did not respond within "
                            f"{self.connect_timeout} seconds. Ensure the CLI is started "
                            "in MCP stdio mode (e.g., `gemini mcp`)."
                        ) from exc

                    logger.info(
                        "Connected to MCP server '%s' with args %s",
                        self.server_command,
                        self.server_args,
                    )
                    yield session
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{self.server_command}': {e}"
            ) from e

    async def list_tools(self) -> list[str]:
        """
        List available tools on the MCP server.

        Returns:
            List[str]: Names of available tools

        Raises:
            MCPConnectionError: If connection fails
        """
        async with self._connect() as session:
            response = await session.list_tools()
            return [tool.name for tool in response.tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of tool to invoke
            arguments: Tool arguments as dictionary

        Returns:
            str: Tool response text

        Raises:
            MCPConnectionError: If connection or tool call fails
        """
        async with self._connect() as session:
            try:
                result = await session.call_tool(tool_name, arguments=arguments)

                # Extract text from response
                if result.content:
                    # Content can be text or structured data
                    if hasattr(result.content[0], "text"):
                        return result.content[0].text
                    return str(result.content[0])
                return ""

            except Exception as e:
                raise MCPConnectionError(f"Tool call '{tool_name}' failed: {e}") from e


class GeminiMCPClient(BaseMCPClient):
    """
    MCP client for Gemini API via gemini-cli server.

    This client connects to the @google/gemini-cli MCP server which provides
    access to Google's Gemini models.

    Server Installation:
        npm install -g @google/gemini-cli

    Configuration:
        Set GEMINI_API_KEY environment variable

    Example:
        ```python
        import asyncio


        async def main():
            client = GeminiMCPClient()

            # Simple ask
            response = await client.ask("What is quantum computing?")
            print(response)

            # With specific model
            response = await client.ask(
                prompt="Prove the Keystone Principle", model="gemini-2.5-pro"
            )
            print(response)


        asyncio.run(main())
        ```
    """

    DEFAULT_SERVER_COMMAND = "mcp-gemini-cli"
    DEFAULT_SERVER_ARGS = []

    def __init__(
        self,
        server_command: str | None = None,
        api_key: str | None = None,
        server_args: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize Gemini MCP client.

        Args:
            server_command: Path to mcp-gemini-cli executable
                (default: "mcp-gemini-cli" - hardcoded)
            api_key: Gemini API key (default: from GEMINI_API_KEY env var)
            server_args: CLI arguments (default: [] - no args needed)
            **kwargs: Additional arguments passed to BaseMCPClient
        """
        env_overrides = dict(kwargs.pop("env", {}) or {})
        if api_key:
            env_overrides["GEMINI_API_KEY"] = api_key
        if env_overrides:
            kwargs["env"] = env_overrides

        # Use hardcoded default if not specified
        if server_command is None:
            server_command = self.DEFAULT_SERVER_COMMAND

        resolved_args = (
            list(server_args)
            if server_args is not None
            else list(kwargs.pop("server_args", self.DEFAULT_SERVER_ARGS))
        )
        kwargs["server_args"] = resolved_args

        super().__init__(server_command=server_command, **kwargs)

    def _discover_server(self) -> str | None:
        """
        Return hardcoded MCP server command.

        Returns:
            str: Hardcoded "mcp-gemini-cli" command
        """
        return self.DEFAULT_SERVER_COMMAND

    async def ask(self, prompt: str, model: str = "gemini-2.5-pro") -> str:
        """
        Ask Gemini a question via MCP.

        Args:
            prompt: Question or prompt for Gemini
            model: Gemini model to use (default: gemini-2.5-pro per CLAUDE.md)

        Returns:
            str: Gemini's response

        Raises:
            MCPConnectionError: If connection or call fails
        """
        return await self.call_tool(
            tool_name="geminiChat",  # Tool name from @choplin/mcp-gemini-cli
            arguments={"model": model, "prompt": prompt},
        )


class CodexMCPClient(BaseMCPClient):
    """
    MCP client for OpenAI/Codex API via MCP server.

    This client connects to a Codex MCP server for accessing GPT models.

    Example:
        ```python
        import asyncio


        async def main():
            client = CodexMCPClient(server_command="codex-mcp")

            response = await client.ask("Explain the halting problem")
            print(response)


        asyncio.run(main())
        ```
    """

    DEFAULT_SERVER_COMMAND = "codex"
    DEFAULT_SERVER_ARGS = ["mcp-server"]

    def __init__(
        self,
        server_command: str | None = None,
        api_key: str | None = None,
        server_args: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize Codex MCP client.

        Args:
            server_command: Path to codex executable
                (default: "codex" - hardcoded)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            server_args: Arguments to start the MCP server (default: ["mcp-server"])
            **kwargs: Additional arguments passed to BaseMCPClient
        """
        # Set up environment with API key if provided
        env = dict(kwargs.pop("env", {}) or {})
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if env:
            kwargs["env"] = env

        # Use hardcoded default if not specified
        if server_command is None:
            server_command = self.DEFAULT_SERVER_COMMAND

        resolved_args = (
            list(server_args)
            if server_args is not None
            else list(kwargs.pop("server_args", self.DEFAULT_SERVER_ARGS))
        )
        kwargs["server_args"] = resolved_args

        super().__init__(server_command=server_command, **kwargs)

    def _discover_server(self) -> str | None:
        """
        Return hardcoded MCP server command.

        Returns:
            str: Hardcoded "codex" command
        """
        return self.DEFAULT_SERVER_COMMAND

    async def ask(
        self,
        prompt: str,
        model: str = "gpt-5-codex",
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Ask Codex/GPT a question via MCP.

        Args:
            prompt: Question or prompt
            model: Model to use (default: gpt-5-codex)
            reasoning_effort: Reasoning effort level - "low", "medium", or "high"
                (passed via config object to Codex MCP server)

        Returns:
            str: Model response

        Raises:
            MCPConnectionError: If connection or call fails
        """
        arguments = {"model": model, "prompt": prompt}

        # Add reasoning effort via config object if specified
        if reasoning_effort:
            arguments["config"] = {"model_reasoning_effort": reasoning_effort}

        return await self.call_tool(
            tool_name="codex",  # Tool name from codex mcp-server
            arguments=arguments,
        )


class ClaudeCodeMCPClient(BaseMCPClient):
    """
    MCP client for Anthropic Claude Code via claude CLI server.

    This client connects to the Claude Code MCP bridge exposed by the
    `claude` CLI (part of @anthropic-ai/claude-agent-sdk).
    """

    DEFAULT_SERVER_COMMAND = "claude"
    DEFAULT_SERVER_ARGS = ["mcp", "serve"]

    def __init__(
        self,
        server_command: str | None = None,
        api_key: str | None = None,
        server_args: list[str] | None = None,
        tool_name: str = "ask-claude",
        **kwargs,
    ):
        """
        Initialize Claude Code MCP client.

        Args:
            server_command: Path to claude CLI executable
                (default: "claude" - hardcoded)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            server_args: CLI arguments (default: ["mcp", "serve"])
            tool_name: Name of the MCP tool to invoke (default: "ask-claude")
            **kwargs: Passed to BaseMCPClient
        """
        env = dict(kwargs.pop("env", {}) or {})
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if env:
            kwargs["env"] = env

        # Use hardcoded default if not specified
        if server_command is None:
            server_command = self.DEFAULT_SERVER_COMMAND

        resolved_args = (
            list(server_args)
            if server_args is not None
            else list(kwargs.pop("server_args", self.DEFAULT_SERVER_ARGS))
        )
        kwargs["server_args"] = resolved_args

        self.tool_name = tool_name

        super().__init__(server_command=server_command, **kwargs)

    def _discover_server(self) -> str | None:
        """
        Return hardcoded MCP server command.

        Returns:
            str: Hardcoded "claude" command
        """
        return self.DEFAULT_SERVER_COMMAND

    async def ask(self, prompt: str, model: str = "claude-3-5-sonnet") -> str:
        """
        Ask Claude Code a question via MCP.

        Args:
            prompt: Task or question for Claude
            model: Anthropic model identifier (default: claude-3-5-sonnet)
        """
        return await self.call_tool(
            tool_name=self.tool_name,
            arguments={"model": model, "prompt": prompt},
        )


# Synchronous wrappers for non-async contexts


def sync_ask_gemini(
    prompt: str,
    model: str = "gemini-2.5-pro",
    server_command: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Synchronous wrapper for Gemini MCP client.

    Args:
        prompt: Question for Gemini
        model: Model to use (default: gemini-2.5-pro)
        server_command: Optional path to gemini-cli
        api_key: Optional API key

    Returns:
        str: Gemini's response

    Example:
        ```python
        response = sync_ask_gemini("What is the Keystone Principle?")
        print(response)
        ```
    """

    async def _ask():
        client = GeminiMCPClient(server_command=server_command, api_key=api_key)
        return await client.ask(prompt=prompt, model=model)

    return _run_sync(_ask())


def sync_ask_codex(
    prompt: str,
    model: str = "gpt-5-codex",
    server_command: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Synchronous wrapper for Codex MCP client.

    Args:
        prompt: Question for Codex
        model: Model to use
        server_command: Optional path to codex server
        api_key: Optional API key

    Returns:
        str: Model response

    Example:
        ```python
        response = sync_ask_codex("Explain monads in category theory")
        print(response)
        ```
    """

    async def _ask():
        client = CodexMCPClient(server_command=server_command, api_key=api_key)
        return await client.ask(prompt=prompt, model=model)

    return _run_sync(_ask())


def sync_ask_claude(
    prompt: str,
    model: str = "claude-3-5-sonnet",
    server_command: str | None = None,
    api_key: str | None = None,
    tool_name: str = "ask-claude",
) -> str:
    """
    Synchronous wrapper for Claude Code MCP client.

    Args:
        prompt: Question for Claude Code
        model: Anthropic model to use
        server_command: Optional explicit path to claude CLI
        api_key: Optional Anthropic API key
        tool_name: MCP tool name to invoke (default: ask-claude)
    """

    async def _ask():
        client = ClaudeCodeMCPClient(
            server_command=server_command,
            api_key=api_key,
            tool_name=tool_name,
        )
        return await client.ask(prompt=prompt, model=model)

    return _run_sync(_ask())


def _run_sync(awaitable: Awaitable[T]) -> T:
    """
    Execute an awaitable from synchronous code, guarding against active loops.

    Raises:
        RuntimeError: If called while an asyncio event loop is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if loop.is_running():
        raise RuntimeError(
            "sync_ask_* helpers cannot be executed inside a running asyncio loop. "
            "Await the async client methods instead."
        )

    return loop.run_until_complete(awaitable)


# Example usage
if __name__ == "__main__":
    import sys

    load_dotenv()

    async def test_gemini():
        """Test Gemini MCP client."""
        try:
            client = GeminiMCPClient()
            print("✓ Connected to Gemini MCP server")

            # # List available tools
            # tools = await client.list_tools()
            # print(f"✓ Available tools: {tools}")

            # Ask a question
            response = await client.ask("What is 2+2?", model="gemini-2.5-flash")
            print(f"✓ Response: {response[:100]}...")

            print("\n✓ Gemini MCP client test passed!")

        except MCPConnectionError as e:
            print(f"✗ MCP Connection failed: {e}", file=sys.stderr)
            print("\nTo fix:")
            print("1. Install: npm install -g @google/gemini-cli")
            print("2. Set API key: export GEMINI_API_KEY=your_key")
            sys.exit(1)

    print("Testing Gemini MCP client...")
    asyncio.run(test_gemini())
