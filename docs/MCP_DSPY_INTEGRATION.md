# MCP + DSPy Integration Guide

Complete guide for using DSPy with MCP servers in the Fragile/Mathster project.

## Overview

This integration allows DSPy programs to route LLM calls through MCP (Model Context Protocol) servers, enabling:
- Full DSPy optimization features (BootstrapFewShot, MIPRO, etc.)
- Access to Gemini 2.5 Pro via `gemini-cli` MCP server
- Access to OpenAI/Codex via MCP servers
- Flexible deployment (standalone scripts or Claude Code context)

## Architecture

```
DSPy Program/Module
    |
    v
ClaudeCodeLM(dspy.LM) [custom LM class]
    |
    v
MCP Callable (from create_*_invoker)
    |
    v
mathster.mcp_client (GeminiMCPClient/CodexMCPClient)
    |
    v
MCP Server (gemini-cli, codex via stdio transport)
    |
    v
Gemini 2.5 Pro / GPT-5 API
```

## Installation

### 1. Python Dependencies

```bash
# Install MCP SDK and DSPy
uv add mcp dspy-ai

# Or with pip
pip install mcp dspy-ai
```

### 2. MCP Server Installation

#### Gemini CLI (Google Gemini)

```bash
# Install gemini-cli MCP server
npm install -g @google/gemini-cli

# Verify installation
which gemini-cli
gemini-cli --version
```

#### Codex MCP Server (OpenAI)

```bash
# Install codex MCP server (if available)
# Replace with actual package name
npm install -g @anthropic-ai/codex-mcp
```

### 3. API Keys

```bash
# Gemini API key
export GEMINI_API_KEY=your_gemini_api_key

# OpenAI API key (for Codex)
export OPENAI_API_KEY=your_openai_api_key
```

To get API keys:
- Gemini: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys

## Usage

### Quick Start (Recommended)

```python
import dspy
from mathster.mcps import create_gemini_lm

# One-line setup (auto-discovers gemini-cli)
lm = create_gemini_lm()
dspy.configure(lm=lm)

# Use DSPy as normal
predictor = dspy.ChainOfThought("question -> answer")
result = predictor(question="What is the Keystone Principle?")
print(result.answer)
```

### Manual Configuration

```python
import dspy
from mathster.mcps import GeminiMCP, create_gemini_invoker

# Create MCP invoker with explicit server path
invoker = create_gemini_invoker(
    server_command="/path/to/gemini-cli",
    api_key="your_api_key"  # Optional, uses GEMINI_API_KEY if not provided
)

# Create LM
lm = GeminiMCP(mcp_callable=invoker)
dspy.configure(lm=lm)
```

### Using DSPy Optimizers

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from mathster.mcps import create_gemini_lm

# Setup
lm = create_gemini_lm()
dspy.configure(lm=lm)

# Define module
class MathExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought("text -> entities")

    def forward(self, text):
        return self.extract(text=text)

# Create training examples
train_examples = [
    dspy.Example(
        text="Theorem: The Euclidean Gas converges exponentially.",
        entities=["Theorem", "Euclidean Gas", "exponential convergence"]
    ).with_inputs("text")
]

# Optimize with BootstrapFewShot
def accuracy_metric(example, prediction):
    # Define your metric
    return len(prediction.entities) > 0

optimizer = BootstrapFewShot(metric=accuracy_metric)
optimized_extractor = optimizer.compile(
    MathExtractor(),
    trainset=train_examples
)

# Use optimized module
result = optimized_extractor(text="New theorem to extract")
```

### Low-Level MCP Client (Advanced)

```python
import asyncio
from mathster.mcp_client import GeminiMCPClient

async def main():
    # Create client
    client = GeminiMCPClient()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {tools}")

    # Make request
    response = await client.ask(
        prompt="Explain the Keystone Principle in one paragraph",
        model="gemini-2.5-pro"
    )
    print(response)

asyncio.run(main())
```

## Modules

### `src/mathster/mcps.py`

DSPy integration layer.

**Classes:**
- `ClaudeCodeLM(dspy.LM)` - Base LM class for MCP
- `GeminiMCP(ClaudeCodeLM)` - Gemini 2.5 Pro wrapper
- `CodexMCP(ClaudeCodeLM)` - Codex/GPT wrapper

**Functions:**
- `create_gemini_lm()` - One-step Gemini setup
- `create_codex_lm()` - One-step Codex setup
- `create_gemini_invoker()` - Create Gemini callable
- `create_codex_invoker()` - Create Codex callable

### `src/mathster/mcp_client.py`

Low-level MCP client for stdio communication.

**Classes:**
- `BaseMCPClient` - Abstract base for MCP clients
- `GeminiMCPClient` - Gemini MCP client
- `CodexMCPClient` - Codex MCP client
- `MCPConnectionError` - Exception for connection failures

**Functions:**
- `sync_ask_gemini()` - Synchronous Gemini wrapper
- `sync_ask_codex()` - Synchronous Codex wrapper

## Testing

### Run Tests

```bash
# Test DSPy integration only
python scripts/test_mcp.py --test dspy

# Test MCP client only
python scripts/test_mcp.py --test client

# Test full program structure
python scripts/test_mcp.py --test full

# Make live API call (requires API key)
python scripts/test_mcp.py --test live

# Run all tests
python scripts/test_mcp.py
```

### Expected Output

```
============================================================
MCP + DSPy Integration Tests
============================================================
...
✅ DSPy Integration test PASSED
✅ Full Program test PASSED
============================================================
TEST SUMMARY
============================================================
dspy                : ✅ PASS
full                : ✅ PASS
============================================================
🎉 All tests PASSED!
```

## Troubleshooting

### Error: "gemini-cli not found"

**Problem:** MCP server not installed or not in PATH

**Solutions:**
1. Install: `npm install -g @google/gemini-cli`
2. Verify: `which gemini-cli`
3. Add to PATH: `export PATH=$PATH:$(npm bin -g)`
4. Or provide explicit path:
   ```python
   lm = create_gemini_lm(server_command="/full/path/to/gemini-cli")
   ```

### Error: "MCP connection failed"

**Problem:** Server can't start or API key missing

**Solutions:**
1. Set API key: `export GEMINI_API_KEY=your_key`
2. Test server manually: `gemini-cli --help`
3. Check server logs for errors
4. Verify network connectivity

### Error: "ImportError: cannot import name 'StdioServerParameters'"

**Problem:** Wrong MCP SDK version

**Solution:**
```bash
# Uninstall old version
uv remove mcp

# Install latest
uv add mcp

# Or upgrade
uv upgrade mcp
```

### Error: "AdapterParseError: LM response cannot be serialized to JSON"

**Problem:** MCP server returning plain text instead of JSON

**Solution:** DSPy 3.0+ expects structured JSON responses. Ensure:
1. Using compatible DSPy version
2. MCP server supports structured output
3. Or wrap response in JSON format:
   ```python
   import json

   def invoker_wrapper(model: str, prompt: str) -> str:
       raw_response = sync_ask_gemini(prompt, model)
       # Wrap in JSON if needed
       return json.dumps({"answer": raw_response})
   ```

## Examples

### Example 1: Simple Q&A

```python
import dspy
from mathster.mcps import create_gemini_lm

lm = create_gemini_lm()
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")
result = qa(question="What is the Fragile framework?")
print(result.answer)
```

### Example 2: Mathematical Entity Extraction

```python
import dspy
from mathster.mcps import create_gemini_lm

lm = create_gemini_lm()
dspy.configure(lm=lm)

class ExtractMathEntities(dspy.Signature):
    """Extract mathematical entities from text."""
    text = dspy.InputField(desc="Mathematical text")
    definitions = dspy.OutputField(desc="List of definitions found")
    theorems = dspy.OutputField(desc="List of theorems found")

extractor = dspy.Predict(ExtractMathEntities)
result = extractor(text="""
    Definition: A walker is a tuple (x, v, s).
    Theorem: The system converges exponentially.
""")

print("Definitions:", result.definitions)
print("Theorems:", result.theorems)
```

### Example 3: Proof Review with Dual Models

```python
import dspy
from mathster.mcps import create_gemini_lm, create_codex_lm

# Setup both models
gemini = create_gemini_lm()
codex = create_codex_lm()

class ProofReviewer(dspy.Signature):
    """Review mathematical proof for correctness."""
    proof = dspy.InputField()
    issues = dspy.OutputField(desc="List of issues found")
    verdict = dspy.OutputField(desc="Accept or Reject")

# Review with Gemini
dspy.configure(lm=gemini)
gemini_review = dspy.ChainOfThought(ProofReviewer)
gemini_result = gemini_review(proof="Proof: ...")

# Review with Codex
dspy.configure(lm=codex)
codex_review = dspy.ChainOfThought(ProofReviewer)
codex_result = codex_review(proof="Proof: ...")

# Compare results
print("Gemini:", gemini_result.verdict)
print("Codex:", codex_result.verdict)
```

## API Reference

### `create_gemini_lm()`

```python
def create_gemini_lm(
    server_command: Optional[str] = None,
    api_key: Optional[str] = None
) -> GeminiMCP
```

One-step setup for Gemini LM with real MCP client.

**Parameters:**
- `server_command` - Path to gemini-cli (auto-discovers if None)
- `api_key` - Gemini API key (uses GEMINI_API_KEY env var if None)

**Returns:**
- `GeminiMCP` - Configured LM instance ready for DSPy

**Raises:**
- `ImportError` - If MCP client not available
- `MCPConnectionError` - If gemini-cli not found

### `GeminiMCP`

```python
class GeminiMCP(ClaudeCodeLM):
    def __init__(self, mcp_callable: MCPCallable, **kwargs)
```

Convenience wrapper for Gemini 2.5 Pro via MCP.

Enforces use of "gemini-2.5-pro" model per CLAUDE.md guidelines.

### `GeminiMCPClient`

```python
class GeminiMCPClient(BaseMCPClient):
    async def ask(self, prompt: str, model: str = "gemini-2.5-pro") -> str
    async def list_tools(self) -> List[str]
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str
```

Low-level async MCP client for Gemini.

## Best Practices

### 1. Model Selection

Always use `gemini-2.5-pro` for Gemini (per CLAUDE.md):

```python
# Correct
lm = create_gemini_lm()  # Uses gemini-2.5-pro by default

# Incorrect (will raise error with GeminiMCP)
lm = GeminiMCP(mcp_callable=invoker, model="gemini-flash")
```

### 2. Error Handling

Always wrap MCP calls in try/except:

```python
from mathster.mcps import create_gemini_lm, MCPConnectionError

try:
    lm = create_gemini_lm()
    dspy.configure(lm=lm)
except MCPConnectionError as e:
    print(f"MCP setup failed: {e}")
    # Fallback to direct API or exit
```

### 3. API Key Management

Use environment variables, never hardcode:

```python
# Good
lm = create_gemini_lm()  # Uses GEMINI_API_KEY env var

# Acceptable for testing
lm = create_gemini_lm(api_key=os.getenv("GEMINI_API_KEY"))

# Bad - never do this
lm = create_gemini_lm(api_key="hardcoded_key_12345")
```

### 4. Caching for Optimization

DSPy optimizers may make many LLM calls. Consider:
- Setting reasonable training set sizes
- Using caching if available in MCP server
- Monitoring API usage/costs

## Related Documentation

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [CLAUDE.md](../CLAUDE.md) - Project conventions

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Run tests: `python scripts/test_mcp.py`
3. Review source code documentation in modules
4. Open issue on project repository
