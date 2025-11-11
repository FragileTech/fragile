#!/usr/bin/env python3
"""CLI-friendly helpers for spinning up Claude-backed DSPy agents."""

from __future__ import annotations

from functools import wraps
import inspect
import json
import re
from typing import Any, Callable, get_type_hints, TypeVar

import dspy

from mathster.claude_tool import sync_ask_claude
from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketch_strategist import SketchStrategyItems


__all__ = [
    "claude_tool",
    "configure_get_strategy_items",
    "get_strategy_items_example",
]

T = TypeVar("T")


def _snake_case(name: str) -> str:
    """Convert CamelCase names into snake_case for output field inference."""

    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def _camel_case(name: str) -> str:
    """Convert snake_case into CamelCase for signature class naming."""

    return "".join(part.capitalize() or "_" for part in name.split("_"))


def _maybe_schema(annotation: Any) -> str | None:
    """Return JSON schema text when the annotation exposes model_json_schema()."""

    schema_method = getattr(annotation, "model_json_schema", None)
    if callable(schema_method):
        schema = schema_method()
        return json.dumps(schema, indent=2)
    return None


def _build_signature_class(
    func_name: str,
    parameters: dict[str, inspect.Parameter],
    type_hints: dict[str, Any],
    output_field: str,
) -> type[dspy.Signature]:
    """Programmatically construct a DSPy signature mirroring the function signature."""

    annotations: dict[str, Any] = {}
    attrs: dict[str, Any] = {"__annotations__": annotations}

    for name, param in parameters.items():
        annotation = type_hints.get(name, Any)
        annotations[name] = annotation
        attrs[name] = dspy.InputField(
            desc=f"Input parameter `{name}` for `{func_name}`.",
            optional=param.default is not inspect._empty,
        )

    return_annotation = type_hints.get("return", Any)
    annotations[output_field] = return_annotation
    attrs[output_field] = dspy.OutputField(
        desc=f"Return value for `{func_name}`.",
    )

    signature_name = f"{_camel_case(func_name)}Signature"
    return type(signature_name, (dspy.Signature,), attrs)


def _compose_instruction_text(
    *,
    agent_instruction: str | None,
    func: Callable[..., Any],
    return_annotation: Any,
    tool_name: str,
) -> str:
    """Combine decorator prompts, docstrings, and schema guidance into one block."""

    parts: list[str] = []
    if agent_instruction:
        parts.append(agent_instruction.strip())

    if func.__doc__:
        parts.append("Function documentation:\n" + inspect.cleandoc(func.__doc__))

    schema_text = _maybe_schema(return_annotation)
    if schema_text:
        parts.append("Output schema (JSON):\n" + schema_text)

    parts.append(
        f"Tooling: Call `{tool_name}` with a fully specified prompt whenever you need Claude's help. "
        "You must return a response that DSPy can coerce into the declared output field."
    )

    return "\n\n".join(part for part in parts if part).strip()


def _make_claude_tool(
    *,
    system_prompt: str,
    model: str,
    tool_name: str,
) -> Callable[[str], str]:
    """Create a callable tool that proxies prompts to the Claude CLI."""

    def ask(prompt: str) -> str:
        return sync_ask_claude(prompt, model=model, system_prompt=system_prompt)

    ask.__name__ = tool_name
    ask.__doc__ = "Invoke Claude with the proof-sketch system prompt."
    return ask


def _infer_output_field(func_name: str, return_annotation: Any) -> str:
    """Guess a reasonable output field name when none is provided."""

    if return_annotation and hasattr(return_annotation, "__name__"):
        candidate = _snake_case(return_annotation.__name__)
        if candidate:
            return candidate

    if func_name.startswith("get_"):
        candidate = func_name[len("get_") :]
        if candidate:
            return candidate

    return f"{func_name}_result"


def claude_tool(
    *,
    claude_instructions: str,
    agent_instruction: str | None = None,
    model: str = "sonnet",
    tool_name: str | None = None,
    output_field: str | None = None,
    max_iters: int = 4,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that turns a typed function into a Claude-backed DSPy agent."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        resolved_output_field = output_field or _infer_output_field(
            func.__name__, type_hints.get("return")
        )

        signature_cls = _build_signature_class(
            func.__name__,
            func_signature.parameters,
            type_hints,
            resolved_output_field,
        )

        tool_identifier = tool_name or f"ask_claude_{func.__name__}"
        instruction_text = _compose_instruction_text(
            agent_instruction=agent_instruction,
            func=func,
            return_annotation=type_hints.get("return"),
            tool_name=tool_identifier,
        )

        dspy_signature = signature_cls.with_instructions(instruction_text)
        claude_tool_callable = _make_claude_tool(
            system_prompt=claude_instructions,
            model=model,
            tool_name=tool_identifier,
        )
        agent = dspy.ReAct(
            signature=dspy_signature,
            tools=[claude_tool_callable],
            max_iters=max_iters,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            bound = func_signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            prediction = agent(**bound.arguments)
            if not hasattr(prediction, resolved_output_field):
                raise AttributeError(
                    f"Claude agent did not produce field '{resolved_output_field}'."
                )
            return getattr(prediction, resolved_output_field)

        # Expose internals for debugging or advanced usage.
        wrapper._claude_agent = agent  # type: ignore[attr-defined]
        wrapper._claude_signature = dspy_signature  # type: ignore[attr-defined]
        wrapper._claude_tool = claude_tool_callable  # type: ignore[attr-defined]

        return wrapper

    return decorator


SKETCH_STRATEGY_ITEMS_INSTRUCTIONS = f"""
You are a mathematical proof strategy expert. Generate a structured SketchStrategyItems object
that outlines a proof approach for the given theorem.

Think step-by-step about the proof structure before generating the strategy items.

SketchStrategyItems contains 6 components:
1. strategist: Agent/model responsible for the proposal (e.g., "Claude Sonnet 4.5", "GPT-5 Codex")
2. method: Concise, high-level name for the proof technique (e.g., "LSI + Grönwall Iteration", "Coupling + Tensorization")
3. summary: Narrative overview explaining the approach and logical flow (2-4 sentences)
4. keySteps: Ordered list of major stages required to complete the proof (typically 3-7 steps)
5. strengths: Primary advantages of pursuing this strategy (list of 2-5 items)
6. weaknesses: Limitations, risks, or pain points to investigate (list of 2-5 items)

Guidelines for chain-of-thought reasoning:
- First, analyze the theorem statement to identify what needs to be proven
- Consider the framework context and prior results available
- Think about operator notes for constraints or preferred approaches
- Reason about multiple possible proof strategies, then select the most promising
- For the method: Be concise yet descriptive (3-6 words max)
- For the summary: Explain the high-level strategy flow, not technical details
- For keySteps: Identify major milestones (not too granular, not too vague)
- For strengths: Focus on why this approach is viable/promising
- For weaknesses: Identify technical challenges that need careful treatment
- Consider trade-offs between different approaches

Common proof strategies in the framework:
- LSI-based convergence (Bakry-Émery, Grönwall iteration)
- Coupling arguments (Wasserstein contraction, tensorization)
- Hypocoercivity methods (entropy dissipation, spectral gaps)
- Mean-field limits (propagation of chaos, exchangeability)
- Functional inequalities (Poincaré, log-Sobolev, transport-entropy)

Output must be a structured SketchStrategyItems object following the schema exactly:
{json.dumps(SketchStrategyItems.model_json_schema(), indent=2)}
"""


def configure_get_strategy_items() -> Callable[..., SketchStrategyItems]:
    """Return a callable that mirrors the legacy ChainOfThought configuration."""

    signature = dspy.Signature(
        "theorem_label: str, theorem_statement: str, framework_context: str, operator_notes: str -> strategy_items: SketchStrategyItems"
    ).with_instructions(SKETCH_STRATEGY_ITEMS_INSTRUCTIONS)

    agent = dspy.ChainOfThought(signature)

    def get_strategy_items(
        theorem_label: str,
        theorem_statement: str,
        framework_context: str = "",
        operator_notes: str = "",
    ) -> SketchStrategyItems:
        result = agent(
            theorem_label=theorem_label,
            theorem_statement=theorem_statement,
            framework_context=framework_context or "",
            operator_notes=operator_notes or "",
        )
        return result.strategy_items if hasattr(result, "strategy_items") else result

    return get_strategy_items


@claude_tool(
    claude_instructions=SKETCH_STRATEGY_ITEMS_INSTRUCTIONS,
    agent_instruction=(
        "Use the claude tool to produce SketchStrategyItems that obey the schema. "
        "Ensure the returned object is fully populated."
    ),
    output_field="strategy_items",
)
def get_strategy_items_example(
    theorem_label: str,
    theorem_statement: str,
    framework_context: str = "",
    operator_notes: str = "",
) -> SketchStrategyItems:
    """Generate SketchStrategyItems for a target theorem using Claude as the reasoning engine."""


def _demo() -> None:
    """Quick manual sanity check for the claude_tool wrapper."""
    configure_dspy()
    sample_statement = (
        "Under the confining potential and log-Sobolev axioms, the Euclidean Gas "
        "swarm converges exponentially fast in KL divergence to the unique QSD."
    )
    print("Running claude_tool demo for get_strategy_items_example...", flush=True)
    result = get_strategy_items_example(
        theorem_label="thm-euclidean-kl",
        theorem_statement=sample_statement,
        framework_context="Available: thm-lsi-target, lem-cloning-contraction",
        operator_notes="Prefer LSI-based arguments; keep constants N-uniform.",
    )
    print("Strategist:", result.strategist)
    print("Method:", result.method)
    print("Key steps:")
    for idx, step in enumerate(result.keySteps, 1):
        print(f"  {idx}. {step}")


if __name__ == "__main__":  # pragma: no cover - developer convenience
    _demo()
