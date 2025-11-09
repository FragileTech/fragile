"""Direct CLI clients for AI agents (non-MCP based).

This module provides direct command-line interface clients for AI agents
that use subprocess execution instead of MCP server architecture.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any


class MCPConnectionError(Exception):
    """Raised when CLI command execution fails."""

    pass


class ClaudeDirectClient:
    """
    Direct CLI client for Anthropic Claude (not MCP-based).

    Uses the `claude --print --model [model] [prompt]` command directly
    instead of MCP server architecture.
    """

    DEFAULT_COMMAND = "claude"

    def __init__(
        self,
        command: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Claude CLI client.

        Args:
            command: Path to claude CLI executable (default: "claude")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Unused, for API compatibility
        """
        self.command = command or self.DEFAULT_COMMAND
        self.api_key = api_key
        self._env = os.environ.copy()
        if api_key:
            self._env["ANTHROPIC_API_KEY"] = api_key

    async def ask(self, prompt: str, model: str = "sonnet") -> str:
        """
        Ask Claude a question via direct CLI invocation.

        Args:
            prompt: Task or question for Claude
            model: Anthropic model identifier or alias (default: "sonnet")
                   Aliases: "sonnet", "opus", "haiku"
                   Full names: "claude-sonnet-4-5-20250929", etc.

        Returns:
            str: Claude's response

        Raises:
            MCPConnectionError: If CLI invocation fails
        """
        # Build command: claude --print --model [model] [prompt]
        cmd = [self.command, "--print", "--model", model, prompt]

        try:
            # Run command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
            )

            stdout, stderr = await process.communicate()

            # Claude writes some logs to stderr but returns valid output on stdout
            # Check stdout first - if we have output, consider it success
            output = stdout.decode("utf-8").strip()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                raise MCPConnectionError(
                    f"Claude CLI command failed (exit code {process.returncode}): {error_msg}"
                )

            return output

        except FileNotFoundError as e:
            raise MCPConnectionError(
                f"Claude CLI not found at '{self.command}'. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            ) from e
        except Exception as e:
            raise MCPConnectionError(f"Failed to call Claude CLI: {e}") from e


# Alias for backward compatibility with existing code
ClaudeCodeMCPClient = ClaudeDirectClient


# Synchronous wrapper for non-async contexts
def sync_ask_claude(
    prompt: str,
    model: str = "sonnet",
    command: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Synchronous wrapper for Claude direct CLI client.

    Args:
        prompt: Question for Claude
        model: Model alias to use (default: "sonnet")
               Aliases: "sonnet", "opus", "haiku"
        command: Optional path to claude CLI
        api_key: Optional API key

    Returns:
        str: Claude's response

    Example:
        ```python
        response = sync_ask_claude("What is the Keystone Principle?")
        print(response)
        ```
    """
    client = ClaudeDirectClient(command=command, api_key=api_key)
    return asyncio.run(client.ask(prompt, model=model))
