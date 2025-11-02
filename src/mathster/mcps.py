"""
DSPy integration with Claude Code MCP servers.

This module provides custom DSPy LM classes that route LLM calls through
MCP (Model Context Protocol) servers, enabling full DSPy optimization
features while leveraging MCP infrastructure.

Key Components:
    - ClaudeCodeLM: Base LM class for DSPy + MCP integration
    - GeminiMCP: Convenience wrapper for Gemini 2.5 Pro
    - CodexMCP: Convenience wrapper for Codex/GPT-5
    - create_gemini_invoker: Factory for Gemini MCP callable
    - create_codex_invoker: Factory for Codex MCP callable

Usage Example (Standalone Python):
    ```python
    import dspy
    from mathster.mcps import create_gemini_lm

    # Auto-configure with real MCP client (finds gemini-cli automatically)
    lm = create_gemini_lm()
    dspy.configure(lm=lm)

    # Use DSPy as normal
    predictor = dspy.ChainOfThought("question -> answer")
    result = predictor(question="What is the Keystone Principle?")

    # Use optimizers
    from dspy.teleprompt import BootstrapFewShot
    optimizer = BootstrapFewShot(metric=accuracy_metric)
    optimized = optimizer.compile(module, trainset=examples)
    ```

Usage Example (Manual Configuration):
    ```python
    import dspy
    from mathster.mcps import GeminiMCP, create_gemini_invoker

    # Create MCP invoker
    invoker = create_gemini_invoker(server_command="gemini-cli")

    # Configure DSPy
    lm = GeminiMCP(mcp_callable=invoker)
    dspy.configure(lm=lm)
    ```

Architecture:
    DSPy Program/Module
        |
        v
    ClaudeCodeLM(dspy.LM)
        |
        v
    MCP Callable (from create_*_invoker)
        |
        v
    mathster.mcp_client (GeminiMCPClient/CodexMCPClient)
        |
        v
    MCP Server (gemini-cli, codex via stdio)
        |
        v
    Gemini 2.5 Pro / GPT-5

Requirements:
    - MCP SDK: uv add mcp
    - MCP Server: npm install -g @google/gemini-cli
    - API Key: Set GEMINI_API_KEY environment variable

Notes:
    - This module is standalone and doesn't depend on llm_interface.py
    - Supports all DSPy optimization features (BootstrapFewShot, MIPRO, etc.)
    - Single model per LM instance (dual-review can be orchestrated separately)
    - For Claude Code context, use declarative agent instructions instead
"""

from typing import Callable, Optional, Any, List
import dspy
from dspy.primitives.prediction import Prediction

# Import MCP client for standalone scripts
try:
    from mathster.mcp_client import sync_ask_gemini, sync_ask_codex, MCPConnectionError
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    MCPConnectionError = Exception  # Fallback


# Type alias for MCP callable
MCPCallable = Callable[[str, str], str]
"""
Type signature for MCP invocation callable.

Args:
    model (str): Model identifier (e.g., "gemini-2.5-pro")
    prompt (str): Prompt to send to the model

Returns:
    str: Model response text
"""


class ClaudeCodeLM(dspy.LM):
    """
    Custom DSPy LM that routes calls through Claude Code MCP servers.

    This class enables full DSPy functionality (including optimizers like
    BootstrapFewShot, MIPRO) while using Claude Code's MCP infrastructure
    for LLM invocation.

    Attributes:
        mcp_callable: Function that invokes MCP server
        model: Model identifier (e.g., "gemini-2.5-pro")
        history: Call history for DSPy introspection

    Example:
        ```python
        import dspy

        def my_mcp_invoker(model: str, prompt: str) -> str:
            # Invoke MCP in Claude Code environment
            return mcp__gemini-cli__ask-gemini(model=model, prompt=prompt)

        lm = ClaudeCodeLM(
            mcp_callable=my_mcp_invoker,
            model="gemini-2.5-pro"
        )
        dspy.configure(lm=lm)
        ```
    """

    def __init__(
        self,
        mcp_callable: MCPCallable,
        model: str = "gemini-2.5-pro",
        **kwargs
    ):
        """
        Initialize ClaudeCodeLM.

        Args:
            mcp_callable: Function with signature (model: str, prompt: str) -> str
                that invokes the MCP server. In Claude Code context, this would
                typically call mcp__gemini-cli__ask-gemini or mcp__codex__codex.
            model: Model identifier. For Gemini, use "gemini-2.5-pro" (required
                by CLAUDE.md guidelines). For Codex, the model is inferred from
                the MCP server.
            **kwargs: Additional arguments passed to dspy.LM base class.
        """
        super().__init__(model=model, **kwargs)
        self.mcp_callable = mcp_callable
        self.model_name = model
        self.history: List[dict] = []

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        **kwargs
    ) -> List[str]:
        """
        Invoke the LM via MCP server.

        This method is called by DSPy during program execution and optimization.

        Args:
            prompt: Optional text prompt (DSPy may pass this directly)
            messages: Optional message list (DSPy format)
            **kwargs: Additional generation parameters (e.g., temperature, max_tokens)

        Returns:
            List[str]: List of completions (typically single completion)

        Raises:
            ValueError: If neither prompt nor messages provided
        """
        # Build prompt from messages if needed
        if prompt is None:
            if messages is None:
                raise ValueError("Must provide either 'prompt' or 'messages'")
            prompt = self._messages_to_prompt(messages)

        # Invoke MCP
        try:
            response = self.mcp_callable(self.model_name, prompt)
        except Exception as e:
            raise RuntimeError(
                f"MCP invocation failed for model '{self.model_name}': {e}"
            ) from e

        # Record in history for DSPy introspection
        self.history.append({
            "prompt": prompt,
            "response": response,
            "model": self.model_name,
            "kwargs": kwargs,
        })

        # DSPy expects list of completions
        return [response]

    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """
        Convert DSPy message format to plain text prompt.

        DSPy uses message dicts with 'role' and 'content' keys.
        We concatenate them into a single prompt string.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            str: Concatenated prompt text
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if role and content:
                prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)

    def copy(self, **kwargs) -> "ClaudeCodeLM":
        """
        Create a copy of this LM with optionally modified parameters.

        Required by DSPy for creating model variants during optimization.

        Args:
            **kwargs: Parameters to override (e.g., model, temperature)

        Returns:
            ClaudeCodeLM: New instance with updated parameters
        """
        new_model = kwargs.get("model", self.model_name)
        new_callable = kwargs.get("mcp_callable", self.mcp_callable)

        return ClaudeCodeLM(
            mcp_callable=new_callable,
            model=new_model,
            **{k: v for k, v in kwargs.items() if k not in ["model", "mcp_callable"]}
        )

    def inspect_history(self, n: int = 1) -> List[dict]:
        """
        Inspect recent call history.

        Used by DSPy for debugging and optimization introspection.

        Args:
            n: Number of recent calls to return

        Returns:
            List[dict]: Recent call records
        """
        return self.history[-n:] if self.history else []

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ClaudeCodeLM(model='{self.model_name}', calls={len(self.history)})"


class GeminiMCP(ClaudeCodeLM):
    """
    Convenience wrapper for Gemini 2.5 Pro via MCP.

    This class enforces the use of "gemini-2.5-pro" as required by CLAUDE.md
    guidelines (never flash or other variants).

    Example:
        ```python
        import dspy
        from mathster.mcps import GeminiMCP

        lm = GeminiMCP(mcp_callable=my_gemini_invoker)
        dspy.configure(lm=lm)

        # Use with DSPy programs
        predictor = dspy.ChainOfThought("question -> answer")
        result = predictor(question="What is the Keystone Principle?")
        ```
    """

    def __init__(self, mcp_callable: MCPCallable, **kwargs):
        """
        Initialize GeminiMCP.

        Args:
            mcp_callable: Function that invokes mcp__gemini-cli__ask-gemini
            **kwargs: Additional arguments (model is fixed to "gemini-2.5-pro")
        """
        # Enforce model requirement from CLAUDE.md
        if "model" in kwargs and kwargs["model"] != "gemini-2.5-pro":
            raise ValueError(
                "GeminiMCP requires model='gemini-2.5-pro' per CLAUDE.md guidelines. "
                f"Got: {kwargs['model']}"
            )

        super().__init__(
            mcp_callable=mcp_callable,
            model="gemini-2.5-pro",
            **{k: v for k, v in kwargs.items() if k != "model"}
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"GeminiMCP(calls={len(self.history)})"


class CodexMCP(ClaudeCodeLM):
    """
    Convenience wrapper for Codex/GPT-5 via MCP.

    This class is designed for use with mcp__codex__codex, which typically
    routes to GPT-5 or similar high-capability models.

    Example:
        ```python
        import dspy
        from mathster.mcps import CodexMCP

        lm = CodexMCP(mcp_callable=my_codex_invoker)
        dspy.configure(lm=lm)

        # Use with DSPy programs
        predictor = dspy.Predict("context, question -> answer")
        result = predictor(context="...", question="...")
        ```
    """

    def __init__(self, mcp_callable: MCPCallable, **kwargs):
        """
        Initialize CodexMCP.

        Args:
            mcp_callable: Function that invokes mcp__codex__codex
            **kwargs: Additional arguments (model defaults to "gpt-5-codex")
        """
        model = kwargs.pop("model", "gpt-5-codex")
        super().__init__(
            mcp_callable=mcp_callable,
            model=model,
            **kwargs
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"CodexMCP(model='{self.model_name}', calls={len(self.history)})"


# Utility functions

def create_mcp_callable(
    mcp_function: Callable[[str, str], str],
    default_model: Optional[str] = None
) -> MCPCallable:
    """
    Create an MCP callable with optional model override.

    This helper wraps an MCP invocation function to provide a consistent
    interface for ClaudeCodeLM.

    Args:
        mcp_function: Raw MCP function (e.g., mcp__gemini-cli__ask-gemini)
        default_model: Optional default model to use if not specified

    Returns:
        MCPCallable: Wrapped function with (model, prompt) -> response signature

    Example:
        ```python
        # In Claude Code context
        gemini_callable = create_mcp_callable(
            mcp__gemini-cli__ask-gemini,
            default_model="gemini-2.5-pro"
        )

        lm = GeminiMCP(mcp_callable=gemini_callable)
        ```
    """
    def callable_wrapper(model: str, prompt: str) -> str:
        """Wrapper that invokes MCP with model and prompt."""
        use_model = model if model else default_model
        if not use_model:
            raise ValueError("No model specified and no default model set")

        return mcp_function(model=use_model, prompt=prompt)

    return callable_wrapper


def get_gemini_lm(mcp_invoker: Callable) -> GeminiMCP:
    """
    Quick setup for Gemini 2.5 Pro LM.

    Args:
        mcp_invoker: MCP invocation function

    Returns:
        GeminiMCP: Configured LM instance

    Example:
        ```python
        import dspy

        lm = get_gemini_lm(mcp__gemini-cli__ask-gemini)
        dspy.configure(lm=lm)
        ```
    """
    return GeminiMCP(mcp_callable=mcp_invoker)


def get_codex_lm(mcp_invoker: Callable) -> CodexMCP:
    """
    Quick setup for Codex/GPT-5 LM.

    Args:
        mcp_invoker: MCP invocation function

    Returns:
        CodexMCP: Configured LM instance

    Example:
        ```python
        import dspy

        lm = get_codex_lm(mcp__codex__codex)
        dspy.configure(lm=lm)
        ```
    """
    return CodexMCP(mcp_callable=mcp_invoker)


# Factory functions using real MCP client (for standalone scripts)

def create_gemini_invoker(
    server_command: Optional[str] = None,
    api_key: Optional[str] = None
) -> MCPCallable:
    """
    Create Gemini MCP invoker using real MCP client.

    This function creates a callable that uses the actual MCP client
    to communicate with gemini-cli server via stdio.

    Args:
        server_command: Path to gemini-cli executable (auto-discovers if None)
        api_key: Gemini API key (uses GEMINI_API_KEY env var if None)

    Returns:
        MCPCallable: Function with signature (model, prompt) -> response

    Raises:
        ImportError: If MCP client not available
        MCPConnectionError: If gemini-cli server not found

    Example:
        ```python
        import dspy
        from mathster.mcps import GeminiMCP, create_gemini_invoker

        # Create invoker with auto-discovery
        invoker = create_gemini_invoker()

        # Configure DSPy
        lm = GeminiMCP(mcp_callable=invoker)
        dspy.configure(lm=lm)
        ```
    """
    if not MCP_CLIENT_AVAILABLE:
        raise ImportError(
            "MCP client not available. Install with: uv add mcp\n"
            "Also ensure MCP server installed: npm install -g @google/gemini-cli"
        )

    def invoker(model: str, prompt: str) -> str:
        """Invoke Gemini via real MCP client."""
        return sync_ask_gemini(
            prompt=prompt,
            model=model,
            server_command=server_command,
            api_key=api_key
        )

    return invoker


def create_codex_invoker(
    server_command: Optional[str] = None,
    api_key: Optional[str] = None
) -> MCPCallable:
    """
    Create Codex MCP invoker using real MCP client.

    Args:
        server_command: Path to codex MCP server executable (auto-discovers if None)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)

    Returns:
        MCPCallable: Function with signature (model, prompt) -> response

    Raises:
        ImportError: If MCP client not available
        MCPConnectionError: If codex server not found

    Example:
        ```python
        import dspy
        from mathster.mcps import CodexMCP, create_codex_invoker

        # Create invoker
        invoker = create_codex_invoker(server_command="codex-mcp")

        # Configure DSPy
        lm = CodexMCP(mcp_callable=invoker)
        dspy.configure(lm=lm)
        ```
    """
    if not MCP_CLIENT_AVAILABLE:
        raise ImportError(
            "MCP client not available. Install with: uv add mcp\n"
            "Also ensure codex MCP server is installed and configured"
        )

    def invoker(model: str, prompt: str) -> str:
        """Invoke Codex via real MCP client."""
        return sync_ask_codex(
            prompt=prompt,
            model=model,
            server_command=server_command,
            api_key=api_key
        )

    return invoker


def create_gemini_lm(
    server_command: Optional[str] = None,
    api_key: Optional[str] = None
) -> GeminiMCP:
    """
    One-step setup for Gemini LM with real MCP client.

    This is the simplest way to use Gemini with DSPy from standalone scripts.

    Args:
        server_command: Path to gemini-cli (auto-discovers if None)
        api_key: Gemini API key (uses GEMINI_API_KEY env var if None)

    Returns:
        GeminiMCP: Configured LM instance ready for DSPy

    Raises:
        ImportError: If MCP client not available
        MCPConnectionError: If gemini-cli not found

    Example:
        ```python
        import dspy
        from mathster.mcps import create_gemini_lm

        # One-line setup with auto-discovery
        lm = create_gemini_lm()
        dspy.configure(lm=lm)

        # Use DSPy as normal
        predictor = dspy.ChainOfThought("question -> answer")
        result = predictor(question="What is the Keystone Principle?")
        ```
    """
    invoker = create_gemini_invoker(server_command=server_command, api_key=api_key)
    return GeminiMCP(mcp_callable=invoker)


def create_codex_lm(
    server_command: Optional[str] = None,
    api_key: Optional[str] = None
) -> CodexMCP:
    """
    One-step setup for Codex LM with real MCP client.

    Args:
        server_command: Path to codex server (auto-discovers if None)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)

    Returns:
        CodexMCP: Configured LM instance ready for DSPy

    Raises:
        ImportError: If MCP client not available
        MCPConnectionError: If codex server not found

    Example:
        ```python
        import dspy
        from mathster.mcps import create_codex_lm

        lm = create_codex_lm(server_command="codex-mcp")
        dspy.configure(lm=lm)
        ```
    """
    invoker = create_codex_invoker(server_command=server_command, api_key=api_key)
    return CodexMCP(mcp_callable=invoker)


# Example usage (documentation purposes)
if __name__ == "__main__":
    print(__doc__)
    print("\nExample configuration:")
    print("""
    import dspy
    from mathster.mcps import GeminiMCP

    # Step 1: Define MCP callable (provided by Claude Code)
    def gemini_invoker(model: str, prompt: str) -> str:
        # This would invoke: mcp__gemini-cli__ask-gemini(model=model, prompt=prompt)
        # For testing, return mock response
        return f"Mock response for: {prompt[:50]}..."

    # Step 2: Configure DSPy
    lm = GeminiMCP(mcp_callable=gemini_invoker)
    dspy.configure(lm=lm)

    # Step 3: Use DSPy as normal
    predictor = dspy.ChainOfThought("question -> answer")
    result = predictor(question="What is the Fragile framework?")

    # Step 4: Use optimizers
    from dspy.teleprompt import BootstrapFewShot

    class MathExtractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.ChainOfThought("text -> entities")

        def forward(self, text):
            return self.extract(text=text)

    module = MathExtractor()
    optimizer = BootstrapFewShot(metric=lambda x, y: True)
    optimized = optimizer.compile(module, trainset=[])
    """)
