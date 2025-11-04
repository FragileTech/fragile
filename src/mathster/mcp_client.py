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
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize MCP client.

        Args:
            server_command: Command to execute MCP server (e.g., "gemini-cli")
                If None and auto_discover=True, attempts to find server
            server_args: Arguments to pass to server (e.g., ["mcp-server"])
            env: Environment variables for server process
            auto_discover: If True, attempts to find server in common locations

        Raises:
            MCPConnectionError: If server not found and auto_discover fails
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.env = env
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
            command=self.server_command, args=self.server_args, env=self.env
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    logger.info(f"Connected to MCP server: {self.server_command}")
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

    def __init__(self, server_command: str | None = None, api_key: str | None = None, **kwargs):
        """
        Initialize Gemini MCP client.

        Args:
            server_command: Path to gemini-cli executable
                (default: auto-discover)
            api_key: Gemini API key (default: from GEMINI_API_KEY env var)
            **kwargs: Additional arguments passed to BaseMCPClient
        """
        # Set up environment with API key if provided
        env = kwargs.get("env", {})
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        kwargs["env"] = env or None

        # Default server args for gemini-cli
        if "server_args" not in kwargs:
            kwargs["server_args"] = []  # gemini-cli doesn't need special args

        super().__init__(server_command=server_command, **kwargs)

    def _discover_server(self) -> str | None:
        """
        Discover gemini-cli in common locations.

        Searches:
        - ~/.local/bin/gemini-cli
        - /usr/local/bin/gemini-cli
        - npm global bin path
        - PATH environment variable

        Returns:
            str: Path to gemini-cli, or None if not found
        """
        import shutil

        # Check common locations
        locations = [
            Path.home() / ".local" / "bin" / "gemini-cli",
            Path("/usr/local/bin/gemini-cli"),
            Path("/usr/bin/gemini-cli"),
        ]

        for path in locations:
            if path.exists() and path.is_file():
                logger.info(f"Found gemini-cli at: {path}")
                return str(path)

        # Check PATH
        found = shutil.which("gemini-cli")
        if found:
            logger.info(f"Found gemini-cli in PATH: {found}")
            return found

        # Try npm global bin
        try:
            import subprocess

            result = subprocess.run(
                ["npm", "bin", "-g"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode == 0:
                npm_bin = Path(result.stdout.strip()) / "gemini-cli"
                if npm_bin.exists():
                    logger.info(f"Found gemini-cli in npm global bin: {npm_bin}")
                    return str(npm_bin)
        except Exception as e:
            logger.debug(f"Could not check npm global bin: {e}")

        logger.warning("gemini-cli not found. Install with: npm install -g @google/gemini-cli")
        return None

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
            tool_name="ask-gemini",  # Tool name from gemini-cli MCP server
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

    def __init__(self, server_command: str | None = None, api_key: str | None = None, **kwargs):
        """
        Initialize Codex MCP client.

        Args:
            server_command: Path to codex MCP server executable
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            **kwargs: Additional arguments passed to BaseMCPClient
        """
        # Set up environment with API key if provided
        env = kwargs.get("env", {})
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        kwargs["env"] = env or None

        super().__init__(server_command=server_command, **kwargs)

    def _discover_server(self) -> str | None:
        """
        Discover codex MCP server in common locations.

        Returns:
            str: Path to codex server, or None if not found
        """
        import shutil

        # Common names for codex MCP servers
        server_names = ["codex", "codex-mcp", "openai-mcp"]

        for name in server_names:
            found = shutil.which(name)
            if found:
                logger.info(f"Found codex server: {found}")
                return found

        logger.warning("Codex MCP server not found")
        return None

    async def ask(self, prompt: str, model: str = "gpt-5-codex") -> str:
        """
        Ask Codex/GPT a question via MCP.

        Args:
            prompt: Question or prompt
            model: Model to use (default: gpt-5-codex)

        Returns:
            str: Model response

        Raises:
            MCPConnectionError: If connection or call fails
        """
        # Note: Tool name may vary depending on MCP server implementation
        # This is a placeholder - actual tool name should match server
        return await self.call_tool(
            tool_name="ask",  # Or "codex", "complete", etc.
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

    return asyncio.run(_ask())


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

    return asyncio.run(_ask())


# Example usage
if __name__ == "__main__":
    import sys

    async def test_gemini():
        """Test Gemini MCP client."""
        try:
            client = GeminiMCPClient()
            print("✓ Connected to Gemini MCP server")

            # List available tools
            tools = await client.list_tools()
            print(f"✓ Available tools: {tools}")

            # Ask a question
            response = await client.ask("What is 2+2?", model="gemini-2.5-pro")
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
