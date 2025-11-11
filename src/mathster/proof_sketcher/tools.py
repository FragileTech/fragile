#!/usr/bin/env python3
"""Utility tools for proof sketch review agents (search + Claude access)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from mathster.claude_tool import sync_ask_claude


def configure_search_tool(base_path: str | Path) -> Callable[[str], str]:
    """Create a DSPy ReAct tool that searches files under base_path for a query."""

    base_path = Path(base_path).resolve()

    def search(query: str) -> str:
        path = Path(query.strip())
        if path.is_file():
            return path.read_text()
        # Fallback: treat query as substring to search with ripgrep
        import subprocess

        cmd = [
            "rg",
            "--no-heading",
            "--line-number",
            "--color",
            "never",
            query,
            str(base_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            return result.stdout.strip() or f"No matches for '{query}'"
        except subprocess.CalledProcessError as exc:
            return exc.stdout.strip() or f"No matches for '{query}'"

    search.__name__ = "search_project"
    search.__doc__ = (
        "Search project files via ripgrep. Provide either a file path or a search substring."
    )
    return search


def configure_claude_tool(system_prompt: str) -> Callable[[str], str]:
    """Return a ReAct-compatible tool for interacting with Claude."""

    def ask_claude(prompt: str) -> str:
        return sync_ask_claude(prompt, model="sonnet", system_prompt=system_prompt)

    ask_claude.__name__ = "ask_claude"
    ask_claude.__doc__ = "Consult Claude with framework-specific instructions."
    return ask_claude
