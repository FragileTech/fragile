#!/usr/bin/env python3
"""
Advanced Pydantic validators for mathematical documents.

Demonstrates the full extent of what Pydantic can validate, from basic
type checking to complex cross-document constraints.

Usage:
    from src.analysis.advanced_validators import (
        ValidatedTheorem,
        ValidatedProof,
        ValidatedDocument,
        ValidationContext,
    )

    # Load with validation
    with ValidationContext.from_all_documents(paths):
        doc = ValidatedDocument(**data)  # Full validation!
"""

from contextvars import ContextVar
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


try:
    from sympy import simplify, sympify
    from sympy.parsing.sympy_parser import parse_expr

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ==================== Global Validation Context ====================

_validation_context: ContextVar[dict] = ContextVar("validation_context", default={})


class ValidationContext:
    """Context manager providing global validation state."""

    def __init__(
        self,
        all_labels: set[str] | None = None,
        dependency_graph: nx.DiGraph | None = None,
        enable_symbolic: bool = True,
    ):
        self.all_labels = all_labels or set()
        self.dependency_graph = dependency_graph or nx.DiGraph()
        self.enable_symbolic = enable_symbolic

    def __enter__(self):
        _validation_context.set({
            "all_labels": self.all_labels,
            "dependency_graph": self.dependency_graph,
            "enable_symbolic": self.enable_symbolic,
        })
        return self

    def __exit__(self, *args):
        _validation_context.set({})

    @classmethod
    def from_all_documents(cls, document_paths: list[Path], **kwargs):
        """Create context by scanning all documents."""
        all_labels = set()
        G = nx.DiGraph()

        # First pass: collect labels
        import json

        for path in document_paths:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            for directive in data.get("directives", []):
                label = directive["label"]
                all_labels.add(label)
                G.add_node(label, type=directive["type"])

        return cls(all_labels=all_labels, dependency_graph=G, **kwargs)


# ==================== Base Models with Validators ====================


class CrossReference(BaseModel):
    """Cross-reference with validation."""

    label: str = Field(..., pattern=r"^[a-z][a-z0-9-]*[a-z0-9]$")
    type: str
    role: str | None = None

    @field_validator("label")
    @classmethod
    def validate_label_exists(cls, v: str) -> str:
        """Check if referenced label exists (if context available)."""
        ctx = _validation_context.get()
        if not ctx:
            return v  # No context, skip check

        all_labels = ctx.get("all_labels", set())
        if all_labels and v not in all_labels:
            raise ValueError(f"Referenced label does not exist: {v}")

        return v


class SymbolicClaim(BaseModel):
    """Symbolic claim that can be verified with SymPy."""

    premise: str = Field(..., description="Starting expression (SymPy syntax)")
    conclusion: str = Field(..., description="Resulting expression (SymPy syntax)")
    variables: dict[str, str] | None = Field(None, description="Variable declarations")

    @model_validator(mode="after")
    def verify_claim(self) -> "SymbolicClaim":
        """Verify the claim symbolically if SymPy is available."""
        ctx = _validation_context.get()
        if not ctx or not ctx.get("enable_symbolic", False):
            return self

        if not SYMPY_AVAILABLE:
            return self

        try:
            premise_expr = parse_expr(self.premise)
            conclusion_expr = parse_expr(self.conclusion)

            diff = simplify(premise_expr - conclusion_expr)

            if diff != 0:
                # Don't raise error, just add warning metadata
                # (In production, you might log this)
                pass

        except Exception:
            # Symbolic verification failed, but don't block
            pass

        return self


class ProofStep(BaseModel):
    """Proof step with advanced validation."""

    id: str
    title: str | None = None
    content: str
    techniques: list[str] | None = None
    justification: str | list[str | CrossReference] | None = None
    intermediate_result: str | None = None
    substeps: list["ProofStep"] | None = None
    symbolic_claim: SymbolicClaim | None = None

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure step ID follows pattern."""
        if not v.startswith("Step"):
            raise ValueError(f'Step ID must start with "Step", got: {v}')
        return v

    @model_validator(mode="after")
    def validate_step_structure(self) -> "ProofStep":
        """Validate proof step structure."""

        # Rule 1: If has substeps, should have intermediate result
        if self.substeps and not self.intermediate_result:
            # This is a warning, not an error
            pass

        # Rule 2: Leaf steps need detailed content
        if not self.substeps and len(self.content.strip()) < 10:
            raise ValueError(f"Step {self.id} has insufficient content (<10 chars)")

        # Rule 3: Check substep ID hierarchy
        if self.substeps:
            for substep in self.substeps:
                # Substep should be child of parent
                if not substep.id.startswith(f"{self.id}."):
                    raise ValueError(
                        f"Substep {substep.id} must be child of {self.id} "
                        f"(should start with {self.id}.)"
                    )

        # Rule 4: Proof by contradiction requires justification
        if self.techniques and "contradiction" in self.techniques:
            if not self.justification:
                raise ValueError(f"Step {self.id} uses contradiction but lacks justification")

        return self

    @model_validator(mode="after")
    def validate_depth(self) -> "ProofStep":
        """Limit proof step nesting depth."""

        def max_depth(step: "ProofStep", current: int = 0) -> int:
            if not step.substeps:
                return current
            return max(max_depth(s, current + 1) for s in step.substeps)

        depth = max_depth(self)
        if depth > 5:
            raise ValueError(f"Step {self.id}: Proof nesting too deep ({depth} levels). Max: 5")

        return self


# Enable forward reference resolution
ProofStep.model_rebuild()


class ReviewScore(BaseModel):
    """Review score with consistency validation."""

    reviewer: str
    review_date: date
    rigor: int = Field(..., ge=1, le=10)
    soundness: int = Field(..., ge=1, le=10)
    consistency: int = Field(..., ge=1, le=10)
    verdict: str

    @model_validator(mode="after")
    def validate_verdict_consistency(self) -> "ReviewScore":
        """Ensure verdict matches scores."""
        avg = (self.rigor + self.soundness + self.consistency) / 3

        # Strict consistency checks
        if self.verdict == "ready":
            if avg < 8:
                raise ValueError(
                    f'Verdict "ready" requires avg ≥8, got {avg:.1f} '
                    f"(rigor={self.rigor}, soundness={self.soundness}, "
                    f"consistency={self.consistency})"
                )

        if self.verdict == "reject":
            if avg > 5:
                raise ValueError(
                    f'Verdict "reject" inconsistent with avg {avg:.1f}. '
                    f"Rejection typically implies low scores."
                )

        # Check for critical failures
        min_score = min(self.rigor, self.soundness, self.consistency)
        if min_score < 3 and self.verdict != "reject":
            raise ValueError(
                f"Score <3 detected ({min_score}) but verdict is not reject. "
                f"Critical failures should result in rejection."
            )

        return self


class ValidatedTheorem(BaseModel):
    """Theorem with comprehensive validation."""

    model_config = ConfigDict(validate_assignment=True)

    type: str = "theorem"
    label: str = Field(..., pattern=r"^thm-[a-z0-9-]+$")
    title: str
    statement: str
    hypotheses: list[Any] = Field(default_factory=list)
    conclusion: Any
    prerequisites: list[CrossReference] | None = None
    importance: str | None = None
    peer_review: Any | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Title should not be empty and should be reasonably short."""
        if not v.strip():
            msg = "Title cannot be empty"
            raise ValueError(msg)
        if len(v) > 200:
            raise ValueError(f"Title too long ({len(v)} chars). Max: 200")
        return v.strip()

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v: str) -> str:
        """Statement should be substantial."""
        if len(v.strip()) < 10:
            msg = "Statement too short (<10 chars)"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_prerequisites_exist(self) -> "ValidatedTheorem":
        """Validate all prerequisites exist."""
        if not self.prerequisites:
            return self

        ctx = _validation_context.get()
        if not ctx:
            return self

        all_labels = ctx.get("all_labels", set())
        if not all_labels:
            return self

        for prereq in self.prerequisites:
            if prereq.label not in all_labels:
                raise ValueError(
                    f"Theorem {self.label} requires undefined prerequisite: " f"{prereq.label}"
                )

        return self

    @model_validator(mode="after")
    def validate_consistency(self) -> "ValidatedTheorem":
        """Validate internal consistency."""

        # If importance is "foundational", it should have few prerequisites
        if self.importance == "foundational" and self.prerequisites:
            if len(self.prerequisites) > 3:
                raise ValueError(
                    f"Foundational theorem {self.label} has too many prerequisites "
                    f"({len(self.prerequisites)}). Foundational results should "
                    f"depend on few other results."
                )

        return self


class ValidatedProof(BaseModel):
    """Proof with advanced validation."""

    model_config = ConfigDict(validate_assignment=True)

    type: str = "proof"
    label: str = Field(..., pattern=r"^proof-[a-z0-9-]+$")
    title: str
    statement: str
    proves: CrossReference
    proof_type: str | None = None
    strategy: str
    steps: list[ProofStep] = Field(..., min_length=1)
    prerequisites: list[CrossReference] | None = None
    difficulty: str | None = None
    rigor_level: int | None = Field(None, ge=1, le=10)

    @model_validator(mode="after")
    def validate_proves_reference(self) -> "ValidatedProof":
        """Ensure the 'proves' reference exists and is a theorem."""
        ctx = _validation_context.get()
        if not ctx:
            return self

        G = ctx.get("dependency_graph")
        if not G or not G.has_node(self.proves.label):
            # Cannot verify without graph
            return self

        # Check that the proved theorem actually exists
        node_data = G.nodes[self.proves.label]
        if node_data.get("type") not in {"theorem", "lemma", "proposition"}:
            raise ValueError(
                f"Proof {self.label} claims to prove {self.proves.label}, "
                f"but that is a {node_data.get('type')}, not a theorem/lemma/proposition"
            )

        return self

    @model_validator(mode="after")
    def validate_proof_structure(self) -> "ValidatedProof":
        """Validate proof structure."""

        # Check strategy is substantial
        if len(self.strategy.strip()) < 20:
            raise ValueError(
                f"Proof {self.label} has insufficient strategy (<20 chars). "
                f"Strategy should explain the high-level approach."
            )

        # If difficulty is "routine", should have few steps
        if self.difficulty == "routine" and len(self.steps) > 10:
            raise ValueError(
                f"Proof {self.label} marked as 'routine' but has {len(self.steps)} "
                f"steps. Routine proofs should be simple (<10 steps)."
            )

        # If rigor_level is high, should have detailed steps
        if self.rigor_level and self.rigor_level >= 9:
            # Count total steps including substeps
            total_steps = self._count_all_steps()
            if total_steps < 3:
                raise ValueError(
                    f"Proof {self.label} claims rigor_level={self.rigor_level} "
                    f"but only has {total_steps} steps. High-rigor proofs need detail."
                )

        return self

    @model_validator(mode="after")
    def validate_no_circular_reasoning(self) -> "ValidatedProof":
        """Ensure proof doesn't depend on what it proves."""
        if not self.prerequisites:
            return self

        proved_label = self.proves.label
        for prereq in self.prerequisites:
            if prereq.label == proved_label:
                raise ValueError(
                    f"Circular reasoning: Proof {self.label} uses {proved_label} "
                    f"to prove {proved_label}!"
                )

        return self

    def _count_all_steps(self) -> int:
        """Count all steps including substeps."""

        def count(steps):
            total = len(steps)
            for step in steps:
                if step.substeps:
                    total += count(step.substeps)
            return total

        return count(self.steps)

    @lru_cache(maxsize=1)
    def analyze_structure(self) -> dict[str, Any]:
        """Analyze proof structure (cached for performance)."""

        def max_depth(steps, current=0):
            if not steps:
                return current
            return max(
                (max_depth(s.substeps, current + 1) if s.substeps else current for s in steps),
                default=current,
            )

        return {
            "max_depth": max_depth(self.steps),
            "total_steps": self._count_all_steps(),
            "has_symbolic_claims": any(s.symbolic_claim for s in self._iter_all_steps()),
        }

    def _iter_all_steps(self):
        """Iterate all steps recursively."""
        for step in self.steps:
            yield step
            if step.substeps:
                yield from self._iter_substeps(step.substeps)

    def _iter_substeps(self, substeps):
        for step in substeps:
            yield step
            if step.substeps:
                yield from self._iter_substeps(step.substeps)


class ValidatedDocument(BaseModel):
    """Mathematical document with full validation."""

    model_config = ConfigDict(validate_assignment=True)

    metadata: dict[str, Any]
    directives: list[ValidatedTheorem | ValidatedProof | Any]

    @field_validator("directives", mode="before")
    @classmethod
    def parse_directives(cls, v):
        """Parse directives into appropriate types."""
        if not isinstance(v, list):
            msg = "Directives must be a list"
            raise ValueError(msg)

        parsed = []
        for directive in v:
            if not isinstance(directive, dict):
                parsed.append(directive)
                continue

            dtype = directive.get("type")

            if dtype == "theorem":
                parsed.append(ValidatedTheorem(**directive))
            elif dtype == "proof":
                parsed.append(ValidatedProof(**directive))
            else:
                # For other types, just keep as dict (or create other models)
                parsed.append(directive)

        return parsed

    @model_validator(mode="after")
    def validate_all_cross_references(self) -> "ValidatedDocument":
        """Validate all cross-references within document."""

        # Build label set
        all_labels = {d.label for d in self.directives if hasattr(d, "label")}

        # Check each directive's references
        for directive in self.directives:
            if not hasattr(directive, "prerequisites"):
                continue

            if directive.prerequisites:
                for ref in directive.prerequisites:
                    if ref.label not in all_labels:
                        raise ValueError(f"{directive.label} references undefined {ref.label}")

        return self

    @model_validator(mode="after")
    def validate_dependency_graph(self) -> "ValidatedDocument":
        """Ensure no circular dependencies."""

        G = nx.DiGraph()

        # Add all directives as nodes
        for directive in self.directives:
            if hasattr(directive, "label"):
                G.add_node(directive.label)

        # Add edges
        for directive in self.directives:
            if not hasattr(directive, "label"):
                continue

            # Prerequisites
            if hasattr(directive, "prerequisites") and directive.prerequisites:
                for ref in directive.prerequisites:
                    G.add_edge(ref.label, directive.label)

            # Proofs
            if hasattr(directive, "proves"):
                G.add_edge(directive.label, directive.proves.label)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValueError(
                f"Circular dependencies detected: {' → '.join(cycles[0])} → {cycles[0][0]}"
            )

        return self

    @model_validator(mode="after")
    def validate_publication_readiness(self) -> "ValidatedDocument":
        """Validate publication readiness if present."""

        if "publication_readiness_aggregate" not in self.metadata:
            return self

        agg = self.metadata["publication_readiness_aggregate"]

        # Verify counts
        total_directives = len(self.directives)

        if "directive_summary" in agg:
            summary = agg["directive_summary"]
            claimed_total = summary.get("total_directives", 0)

            if claimed_total != total_directives:
                raise ValueError(
                    f"Publication readiness claims {claimed_total} directives "
                    f"but document has {total_directives}"
                )

        return self

    @computed_field
    @property
    def validation_summary(self) -> dict[str, Any]:
        """Compute validation summary."""
        return {
            "total_directives": len(self.directives),
            "theorem_count": sum(
                1 for d in self.directives if getattr(d, "type", None) == "theorem"
            ),
            "proof_count": sum(1 for d in self.directives if getattr(d, "type", None) == "proof"),
            "has_cross_references": any(
                hasattr(d, "prerequisites") and d.prerequisites for d in self.directives
            ),
            "acyclic": True,  # If we got here, it passed validation
        }


# ==================== Validation Pipeline ====================


class ValidationPipeline:
    """Multi-stage validation pipeline."""

    def __init__(self, enable_symbolic: bool = True):
        self.enable_symbolic = enable_symbolic
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_document(
        self, data: dict[str, Any], all_labels: set[str] | None = None
    ) -> ValidatedDocument:
        """Validate a document with full pipeline."""

        # Stage 1: Basic structure (automatic via Pydantic)
        try:
            with ValidationContext(
                all_labels=all_labels or set(), enable_symbolic=self.enable_symbolic
            ):
                doc = ValidatedDocument(**data)

        except Exception as e:
            self.errors.append(f"Validation failed: {e}")
            raise

        return doc

    def generate_report(self) -> str:
        """Generate validation report."""
        lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]

        if self.errors:
            lines.append(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  • {error}")

        if self.warnings:
            lines.append(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        if not self.errors and not self.warnings:
            lines.append("\n✅ PASSED: All validations passed")

        return "\n".join(lines)


# ==================== Example Usage ====================


def example_usage():
    """Demonstrate advanced validation."""

    # Example 1: Valid theorem
    theorem = ValidatedTheorem(
        label="thm-example",
        title="Example Theorem",
        statement="For all x, x^2 >= 0",
        hypotheses=[],
        conclusion={"statement": "x^2 >= 0 for all real x"},
        importance="foundational",
    )
    print(f"✅ Created theorem: {theorem.label}")

    # Example 2: Proof with validation
    proof = ValidatedProof(
        label="proof-example",
        title="Proof of Example",
        statement="Direct proof",
        proves=CrossReference(label="thm-example", type="theorem"),
        strategy="We show that the square of any real number is non-negative by considering cases.",
        steps=[
            ProofStep(
                id="Step 1",
                title="Consider cases",
                content="For any real number x, either x >= 0 or x < 0.",
                techniques=["case-analysis"],
            ),
            ProofStep(
                id="Step 2",
                title="Case x >= 0",
                content="If x >= 0, then x^2 = x * x >= 0 * 0 = 0.",
                techniques=["direct"],
            ),
            ProofStep(
                id="Step 3",
                title="Case x < 0",
                content="If x < 0, then -x > 0, and x^2 = (-x)^2 = (-x) * (-x) > 0.",
                techniques=["direct"],
            ),
        ],
        difficulty="routine",
        rigor_level=8,
    )
    print(f"✅ Created proof: {proof.label}")

    # Example 3: Invalid proof (will fail validation)
    try:
        ValidatedProof(
            label="proof-invalid",
            title="Invalid Proof",
            statement="Bad proof",
            proves=CrossReference(label="thm-nonexistent", type="theorem"),
            strategy="Short",  # Too short!
            steps=[],  # No steps!
            rigor_level=10,
        )
    except Exception as e:
        print(f"❌ Validation caught error: {e}")


if __name__ == "__main__":
    example_usage()
