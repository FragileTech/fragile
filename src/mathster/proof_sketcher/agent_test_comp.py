#!/usr/bin/env python3
"""Example script that demonstrates CompletenessCorrectnessAgent usage."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

try:  # pragma: no cover - shim for local/dev environments without dspy
    import dspy  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback stub (missing or unusable)
    dspy_stub = ModuleType("dspy")

    class _FieldSpec:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Signature:
        @classmethod
        def with_instructions(cls, _instructions: str):
            return cls

    class Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _DummyPredictor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("dspy.Predict stub invoked without patching")

    def InputField(**kwargs: Any) -> _FieldSpec:
        return _FieldSpec(**kwargs)

    def OutputField(**kwargs: Any) -> _FieldSpec:
        return _FieldSpec(**kwargs)

    dspy_stub.Signature = Signature
    dspy_stub.Module = Module
    dspy_stub.InputField = InputField
    dspy_stub.OutputField = OutputField
    dspy_stub.Predict = _DummyPredictor
    dspy_stub.ChainOfThought = _DummyPredictor
    sys.modules["dspy"] = dspy_stub

from mathster.parsing.config import configure_dspy
from mathster.proof_sketcher.sketch_referee_analysis import (
    CompletenessCorrectnessAgent,
    CompletenessCorrectnessReview,
    IdentifiedError,
)


class DemoPredict:
    """Minimal stand-in for dspy.Predict used during the demonstration."""

    def __init__(self, signature=None, *_, **__: Any) -> None:
        self.signature = signature

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        print("[DemoPredict] received fields:", ", ".join(kwargs))
        review = CompletenessCorrectnessReview(
            coversAllClaims=False,
            identifiedErrors=[
                IdentifiedError(
                    location="Step 2",
                    description="Fails to upgrade L^2 convergence to KL without uniform log-Sobolev.",
                    suggestedCorrection="Invoke def-log-sobolev-uniform or restrict to compact support.",
                )
            ],
            score=3,
            confidence=4,
        )
        return SimpleNamespace(completenessAndCorrectness=review)


def run_demo() -> Dict[str, Any]:
    """Instantiate the agent, run a mock review, and return the serialized output."""

    sample_theorem = (
        "Under the Euclidean Gas axioms and the confining potential assumption, "
        "the swarm law converges exponentially fast in KL divergence to the unique QSD."
    )
    sample_sketch = (
        "Step 1: establish tightness via the confinement potential.\n"
        "Step 2: claim KL contraction from Bakry-Émery without showing log-Sobolev constants.\n"
        "Step 3: assert uniqueness directly."
    )

    configure_dspy(model="gemini/gemini-2.5-flash-lite-preview-09-2025")
    print("Testing ProofSketchAgent")
    agent = CompletenessCorrectnessAgent()
    prediction = agent.forward(
        theorem_statement=sample_theorem,
        proof_sketch_text=sample_sketch,
    )

    review = prediction.completenessAndCorrectness
    print("\n=== Completeness/Correctness Review ===")
    print(json.dumps(review.model_dump(), indent=2))
    return review.model_dump()


if __name__ == "__main__":
    run_demo()
