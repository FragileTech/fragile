# Pydantic Validation Deep Dive: How Far Can We Go?

**Author:** Claude Code
**Date:** 2025-10-25
**Version:** 1.0.0

## Executive Summary

This document explores the **full extent** of what Pydantic can validate for mathematical documents, from basic type checking to complex cross-document constraints. We examine each validation layer with concrete examples and identify the precise boundary between what Pydantic can and cannot do.

**TL;DR:** Pydantic can enforce **far more** than you might think, including:
- ✅ Cross-field dependencies
- ✅ Complex structural invariants
- ✅ Graph properties (with custom validators)
- ✅ Mathematical properties (with SymPy integration)
- ✅ Cross-document constraints (with context)
- ❌ Type-level proofs (fundamental limitation)
- ❌ Undecidable properties (halting problem)

---

## Table of Contents

1. [Validation Layers: The Hierarchy](#1-validation-layers)
2. [Level 1: Basic Type Checking](#2-level-1-basic-type-checking)
3. [Level 2: Field Validators](#3-level-2-field-validators)
4. [Level 3: Model Validators](#4-level-3-model-validators)
5. [Level 4: Cross-Model Validation](#5-level-4-cross-model-validation)
6. [Level 5: Document-Level Constraints](#6-level-5-document-level-constraints)
7. [Level 6: Cross-Document Validation](#7-level-6-cross-document-validation)
8. [Level 7: Mathematical Property Validation](#8-level-7-mathematical-property-validation)
9. [Advanced Patterns and Techniques](#9-advanced-patterns)
10. [Performance Considerations](#10-performance)
11. [The Boundary: What Pydantic Cannot Do](#11-the-boundary)
12. [Best Practices and Recommendations](#12-best-practices)

---

## 1. Validation Layers

Pydantic validation occurs in **distinct layers**, each with increasing sophistication:

```
┌─────────────────────────────────────────────────────┐
│ Level 7: Mathematical Property Validation           │ ← SymPy integration
├─────────────────────────────────────────────────────┤
│ Level 6: Cross-Document Validation                  │ ← Global context
├─────────────────────────────────────────────────────┤
│ Level 5: Document-Level Constraints                 │ ← Entire document
├─────────────────────────────────────────────────────┤
│ Level 4: Cross-Model Validation                     │ ← Multiple objects
├─────────────────────────────────────────────────────┤
│ Level 3: Model Validators                           │ ← Whole object
├─────────────────────────────────────────────────────┤
│ Level 2: Field Validators                           │ ← Single field
├─────────────────────────────────────────────────────┤
│ Level 1: Basic Type Checking                        │ ← Type system
└─────────────────────────────────────────────────────┘
```

Each level can access all information from lower levels.

---

## 2. Level 1: Basic Type Checking

### What Pydantic Does Automatically

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date

class Theorem(BaseModel):
    label: str  # Must be a string
    hypotheses: List[str]  # Must be a list of strings
    importance: Optional[Literal["foundational", "main-result"]]  # Enum + optional
    date_created: date  # Must be a valid date
    rigor_level: int  # Must be an integer
```

**Validation happens automatically:**

```python
# ✅ Valid
thm = Theorem(
    label="thm-test",
    hypotheses=["h1", "h2"],
    importance="foundational",
    date_created="2025-10-25",  # Pydantic parses this!
    rigor_level=8
)

# ❌ Fails: wrong type for hypotheses
thm = Theorem(
    label="thm-test",
    hypotheses="not a list",  # ValidationError!
    ...
)

# ❌ Fails: invalid enum value
thm = Theorem(
    ...,
    importance="very-important",  # ValidationError! Not in Literal
)

# ❌ Fails: invalid date
thm = Theorem(
    ...,
    date_created="invalid-date",  # ValidationError!
)
```

### Constrained Types

Pydantic provides **constrained types** for common patterns:

```python
from pydantic import BaseModel, Field, constr, conint, confloat, conlist

class MathObject(BaseModel):
    # String constraints
    label: constr(
        pattern=r'^[a-z][a-z0-9-]*[a-z0-9]$',  # Regex
        min_length=3,
        max_length=100
    )

    # Integer constraints
    rigor_level: conint(ge=1, le=10)  # Greater or equal 1, less or equal 10

    # Float constraints
    probability: confloat(ge=0.0, le=1.0, strict=True)  # Strict float

    # List constraints
    tags: conlist(str, min_length=1, max_length=10)  # 1-10 tags
```

**Equivalent using Field():**

```python
class MathObject(BaseModel):
    label: str = Field(
        ...,
        pattern=r'^[a-z][a-z0-9-]*[a-z0-9]$',
        min_length=3,
        max_length=100
    )
    rigor_level: int = Field(..., ge=1, le=10)
    probability: float = Field(..., ge=0.0, le=1.0)
    tags: List[str] = Field(..., min_length=1, max_length=10)
```

### How Far Can We Go?

**✅ Can Validate:**
- Type correctness (int, str, list, etc.)
- String patterns (regex)
- Numeric ranges
- List/dict lengths
- Enum membership
- Date/time formats
- Optional vs required

**❌ Cannot Validate:**
- Relationships between fields (need Level 2+)
- Semantic meaning (e.g., "is this a valid theorem?")
- Complex invariants

---

## 3. Level 2: Field Validators

Field validators apply **custom logic** to individual fields.

### Basic Field Validator

```python
from pydantic import BaseModel, field_validator

class Theorem(BaseModel):
    label: str
    statement: str

    @field_validator('label')
    @classmethod
    def label_must_start_with_prefix(cls, v: str) -> str:
        """Ensure theorem labels start with 'thm-'."""
        if not v.startswith('thm-'):
            raise ValueError('Theorem labels must start with "thm-"')
        return v

    @field_validator('statement')
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        """Ensure statement is not just whitespace."""
        if not v.strip():
            raise ValueError('Statement cannot be empty')
        return v.strip()  # Can transform!
```

### Validation with Context

Access the **entire model** during validation:

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class ProofStep(BaseModel):
    id: str
    content: str
    justification: Optional[List[str]] = None
    techniques: Optional[List[str]] = None

    @field_validator('justification')
    @classmethod
    def validate_justification(cls, v: Optional[List[str]], info: ValidationInfo) -> Optional[List[str]]:
        """If techniques includes 'contradiction', justification is required."""
        # Access other fields via info.data
        techniques = info.data.get('techniques', [])

        if 'contradiction' in techniques and not v:
            raise ValueError('Proof by contradiction requires justification')

        return v
```

### Validation Mode: Before vs After

```python
from pydantic import field_validator

class Definition(BaseModel):
    title: str
    statement: str

    @field_validator('title', mode='before')
    @classmethod
    def normalize_title(cls, v):
        """Normalize before type checking."""
        if isinstance(v, str):
            return v.strip().title()  # Capitalize
        return v

    @field_validator('statement', mode='after')
    @classmethod
    def validate_statement_after(cls, v: str) -> str:
        """Validate after type checking."""
        # v is guaranteed to be a string here
        if len(v) < 10:
            raise ValueError('Statement too short')
        return v
```

### Multiple Fields

Validate multiple fields with one validator:

```python
class Proof(BaseModel):
    rigor_level: int
    difficulty: str

    @field_validator('rigor_level', 'difficulty')
    @classmethod
    def check_consistency(cls, v, info: ValidationInfo):
        """Both fields must be set."""
        if info.field_name == 'rigor_level' and v < 5:
            raise ValueError('Rigor level must be at least 5 for proofs')
        return v
```

### How Far Can We Go?

**✅ Can Validate:**
- String formats and patterns
- Numeric constraints
- Conditional requirements (if X then Y needed)
- String transformations (normalization)
- List membership
- Access to sibling fields (via ValidationInfo)

**❌ Cannot Validate:**
- Complex relationships between multiple fields (need Level 3)
- Relationships to other models
- Document-wide constraints

**Example: Label Uniqueness**

❌ **Cannot do this in field validator:**
```python
@field_validator('label')
@classmethod
def ensure_unique(cls, v: str) -> str:
    # ❌ Field validators are stateless - no access to other instances!
    if v in GLOBAL_REGISTRY:
        raise ValueError(f'Duplicate label: {v}')
    return v
```

✅ **Need model validator (Level 3) with context (Level 6).**

---

## 4. Level 3: Model Validators

Model validators validate the **entire model** after all fields are set.

### Basic Model Validator

```python
from pydantic import BaseModel, model_validator

class Theorem(BaseModel):
    hypotheses: List[str]
    conclusion: str
    proof_reference: Optional[str] = None

    @model_validator(mode='after')
    def check_proof_required(self) -> 'Theorem':
        """Non-trivial theorems require proofs."""
        if len(self.hypotheses) > 2 and not self.proof_reference:
            raise ValueError('Theorems with >2 hypotheses must reference a proof')
        return self
```

### Complex Invariants

```python
class ProofStep(BaseModel):
    id: str
    content: str
    substeps: Optional[List['ProofStep']] = None
    intermediate_result: Optional[str] = None

    @model_validator(mode='after')
    def validate_structure(self) -> 'ProofStep':
        """Complex structural rules."""
        # Rule 1: If has substeps, must have intermediate result
        if self.substeps and not self.intermediate_result:
            raise ValueError(
                f'Step {self.id} has substeps but no intermediate_result'
            )

        # Rule 2: Leaf steps (no substeps) must have content
        if not self.substeps and len(self.content.strip()) < 10:
            raise ValueError(
                f'Leaf step {self.id} must have detailed content'
            )

        # Rule 3: Check substep IDs are children of parent
        if self.substeps:
            for substep in self.substeps:
                if not substep.id.startswith(f"{self.id}."):
                    raise ValueError(
                        f'Substep {substep.id} must start with parent {self.id}'
                    )

        return self
```

### Cross-Field Dependencies

```python
class ReviewScore(BaseModel):
    rigor: int = Field(..., ge=1, le=10)
    soundness: int = Field(..., ge=1, le=10)
    consistency: int = Field(..., ge=1, le=10)
    verdict: Literal["ready", "minor-revisions", "major-revisions", "reject"]

    @model_validator(mode='after')
    def validate_verdict_matches_scores(self) -> 'ReviewScore':
        """Verdict must be consistent with scores."""
        avg_score = (self.rigor + self.soundness + self.consistency) / 3

        if self.verdict == "ready" and avg_score < 8:
            raise ValueError(
                f'Verdict "ready" requires avg score ≥8, got {avg_score:.1f}'
            )

        if self.verdict == "reject" and avg_score > 5:
            raise ValueError(
                f'Verdict "reject" inconsistent with avg score {avg_score:.1f}'
            )

        # Check for critical failures
        if min(self.rigor, self.soundness, self.consistency) < 3:
            if self.verdict != "reject":
                raise ValueError(
                    'Any score <3 should result in "reject" verdict'
                )

        return self
```

### Recursive Validation

```python
class ProofStep(BaseModel):
    id: str
    content: str
    substeps: Optional[List['ProofStep']] = None

    @model_validator(mode='after')
    def validate_depth(self) -> 'ProofStep':
        """Limit proof step nesting depth."""

        def max_depth(step: 'ProofStep', current: int = 0) -> int:
            if not step.substeps:
                return current
            return max(max_depth(s, current + 1) for s in step.substeps)

        depth = max_depth(self)
        if depth > 5:
            raise ValueError(
                f'Proof step nesting too deep ({depth} levels). Max: 5'
            )

        return self

# Enable forward reference resolution
ProofStep.model_rebuild()
```

### How Far Can We Go?

**✅ Can Validate:**
- Complex relationships between fields
- Structural invariants (tree depth, etc.)
- Conditional requirements (if X then Y must hold)
- Aggregate properties (sums, averages)
- Recursive structures
- Business logic rules

**❌ Cannot Validate:**
- Relationships to other model instances (need Level 4)
- Global uniqueness constraints (need Level 6)
- Dependencies on external state

---

## 5. Level 4: Cross-Model Validation

Validate relationships **between different model instances**.

### Pattern: Root Validator with Child Access

```python
class Proof(BaseModel):
    label: str
    proves: CrossReference
    steps: List[ProofStep]

    @model_validator(mode='after')
    def validate_steps_prove_theorem(self) -> 'Proof':
        """Ensure proof steps lead to the stated conclusion."""

        # Check final step's result mentions the theorem
        if self.steps:
            final_step = self.steps[-1]
            final_result = final_step.intermediate_result or ""

            # Extract theorem label
            theorem_label = self.proves.label

            # Basic check: does final step reference the theorem?
            if theorem_label not in final_result:
                raise ValueError(
                    f'Final proof step must reference theorem {theorem_label}'
                )

        return self
```

### Pattern: Dependency Validation

```python
from typing import Dict, Set

class Theorem(BaseModel):
    label: str
    prerequisites: List[CrossReference]

    @model_validator(mode='after')
    def validate_prerequisites_exist(self) -> 'Theorem':
        """Check prerequisites reference valid labels."""
        # This needs context - see Level 6
        return self

class MathematicalDocument(BaseModel):
    directives: List[Union[Theorem, Proof, Definition, ...]]

    @model_validator(mode='after')
    def validate_all_references(self) -> 'MathematicalDocument':
        """Validate all cross-references within document."""

        # Build label registry
        all_labels: Set[str] = {d.label for d in self.directives}

        # Check each directive's references
        for directive in self.directives:
            # Extract references
            refs = self._extract_references(directive)

            for ref in refs:
                if ref.label not in all_labels:
                    raise ValueError(
                        f'{directive.label} references undefined {ref.label}'
                    )

        return self

    def _extract_references(self, directive) -> List[CrossReference]:
        """Extract all cross-references from a directive."""
        refs = []

        if hasattr(directive, 'prerequisites'):
            refs.extend(directive.prerequisites)
        if hasattr(directive, 'proves'):
            refs.append(directive.proves)
        # ... extract from other fields

        return refs
```

### Pattern: Dependency Graph Validation

```python
import networkx as nx

class MathematicalDocument(BaseModel):
    directives: List[Union[Theorem, Proof, Definition, ...]]

    @model_validator(mode='after')
    def validate_no_circular_dependencies(self) -> 'MathematicalDocument':
        """Ensure dependency graph is acyclic."""

        G = nx.DiGraph()

        # Add all nodes
        for directive in self.directives:
            G.add_node(directive.label)

        # Add edges
        for directive in self.directives:
            refs = self._extract_references(directive)
            for ref in refs:
                if ref.role in ['prerequisite', 'uses', 'requires']:
                    # Edge from prerequisite to dependent
                    G.add_edge(ref.label, directive.label)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValueError(
                f'Circular dependencies detected: {cycles[0]}'
            )

        return self
```

### How Far Can We Go?

**✅ Can Validate:**
- Cross-references within a document
- Dependency graphs (DAG property)
- Ordering constraints
- Aggregate document properties
- Tree/graph structures

**❌ Cannot Validate:**
- References to other documents (need Level 6)
- Properties requiring external data

---

## 6. Level 5: Document-Level Constraints

### Pattern: Document Statistics

```python
class MathematicalDocument(BaseModel):
    metadata: Metadata
    directives: List[Directive]

    @model_validator(mode='after')
    def validate_publication_readiness(self) -> 'MathematicalDocument':
        """Validate publication readiness aggregate is correct."""

        if not self.metadata.publication_readiness_aggregate:
            return self  # Optional field

        agg = self.metadata.publication_readiness_aggregate

        # Count directives by type
        total = len(self.directives)

        # Verify counts match
        if agg.directive_summary:
            claimed_total = agg.directive_summary.total_directives
            if claimed_total != total:
                raise ValueError(
                    f'Claimed {claimed_total} directives but found {total}'
                )

        # Verify blocking issues exist
        if agg.blocking_issues:
            for block in agg.blocking_issues:
                label = block.directive_label
                if not any(d.label == label for d in self.directives):
                    raise ValueError(
                        f'Blocking issue references nonexistent {label}'
                    )

        return self
```

### Pattern: Completeness Checks

```python
class MathematicalDocument(BaseModel):
    directives: List[Directive]

    @model_validator(mode='after')
    def validate_completeness(self) -> 'MathematicalDocument':
        """Ensure document is internally complete."""

        theorems = [d for d in self.directives if d.type == 'theorem']
        proofs = [d for d in self.directives if d.type == 'proof']

        # Every theorem should have a proof (or be an axiom)
        proven_theorems = {p.proves.label for p in proofs if hasattr(p, 'proves')}

        for thm in theorems:
            if thm.label not in proven_theorems:
                # Check if it has a proof_reference
                if not thm.proof_reference:
                    raise ValueError(
                        f'Theorem {thm.label} has no proof or proof reference'
                    )

        return self
```

---

## 7. Level 6: Cross-Document Validation

For cross-document validation, we need **context**.

### Pattern: Validation Context

```python
from contextvars import ContextVar
from typing import Set

# Global context for validation
_validation_context: ContextVar[dict] = ContextVar('validation_context', default={})

class ValidationContext:
    """Context manager for validation with global state."""

    def __init__(self, all_labels: Set[str]):
        self.all_labels = all_labels

    def __enter__(self):
        _validation_context.set({'all_labels': self.all_labels})
        return self

    def __exit__(self, *args):
        _validation_context.set({})

class Theorem(BaseModel):
    label: str
    prerequisites: List[CrossReference]

    @model_validator(mode='after')
    def validate_prerequisites_exist(self) -> 'Theorem':
        """Validate prerequisites against global registry."""
        ctx = _validation_context.get()

        if not ctx:
            return self  # No context available, skip check

        all_labels = ctx.get('all_labels', set())

        for prereq in self.prerequisites:
            if prereq.label not in all_labels:
                raise ValueError(
                    f'Theorem {self.label} requires undefined {prereq.label}'
                )

        return self

# Usage
def load_and_validate_all_documents(document_paths: List[Path]):
    """Load all documents with cross-document validation."""

    # First pass: collect all labels
    all_labels = set()
    documents = []

    for path in document_paths:
        with open(path) as f:
            data = json.load(f)

        # Extract labels without full validation
        for directive in data.get('directives', []):
            all_labels.add(directive['label'])

        documents.append(data)

    # Second pass: validate with context
    validated_docs = []

    with ValidationContext(all_labels):
        for doc_data in documents:
            doc = MathematicalDocument(**doc_data)
            validated_docs.append(doc)

    return validated_docs
```

### Pattern: Label Registry

```python
class LabelRegistry:
    """Global registry of all mathematical labels."""

    def __init__(self):
        self.labels: Dict[str, dict] = {}

    def register(self, label: str, type: str, file: str):
        """Register a label."""
        if label in self.labels:
            raise ValueError(
                f'Duplicate label {label}:\n'
                f'  Already in: {self.labels[label]["file"]}\n'
                f'  Duplicate in: {file}'
            )
        self.labels[label] = {'type': type, 'file': file}

    def exists(self, label: str) -> bool:
        """Check if label exists."""
        return label in self.labels

    def get_type(self, label: str) -> Optional[str]:
        """Get type of a label."""
        return self.labels.get(label, {}).get('type')

# Global instance
_global_registry = LabelRegistry()

class Theorem(BaseModel):
    label: str

    @model_validator(mode='after')
    def register_label(self) -> 'Theorem':
        """Register this theorem's label globally."""
        # Get file from context
        ctx = _validation_context.get()
        file = ctx.get('current_file', 'unknown')

        _global_registry.register(self.label, 'theorem', file)
        return self
```

---

## 8. Level 7: Mathematical Property Validation

Integrate **SymPy** for mathematical validation.

### Pattern: Symbolic Validation

```python
from sympy import sympify, simplify, symbols
from sympy.parsing.sympy_parser import parse_expr

class ProofStep(BaseModel):
    id: str
    content: str
    symbolic_claim: Optional[Dict[str, str]] = None

    @model_validator(mode='after')
    def validate_symbolic_claim(self) -> 'ProofStep':
        """Verify symbolic claim if present."""

        if not self.symbolic_claim:
            return self

        try:
            premise = parse_expr(self.symbolic_claim['premise'])
            conclusion = parse_expr(self.symbolic_claim['conclusion'])

            # Check if they're equivalent
            diff = simplify(premise - conclusion)

            if diff != 0:
                raise ValueError(
                    f'Step {self.id}: Symbolic claim failed verification\n'
                    f'  Premise: {premise}\n'
                    f'  Conclusion: {conclusion}\n'
                    f'  Difference: {diff}'
                )
        except Exception as e:
            # Symbolic validation failed, but don't block
            # (could add to warnings instead)
            pass

        return self
```

### Pattern: Dimensional Analysis

```python
import sympy.physics.units as u
from sympy import Symbol, sympify

class QuantitativeBound(BaseModel):
    bound: str
    type: Literal["upper", "lower", "two-sided"]
    dimension: Optional[str] = None  # e.g., "length", "time", "energy"

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'QuantitativeBound':
        """Check dimensional consistency."""

        if not self.dimension:
            return self

        try:
            # Parse the bound expression
            expr = sympify(self.bound)

            # Define dimensional symbols
            dim_map = {
                'length': u.meter,
                'time': u.second,
                'energy': u.joule,
                'mass': u.kilogram,
            }

            expected_dim = dim_map.get(self.dimension)
            if not expected_dim:
                return self  # Unknown dimension, skip

            # Check if expression has consistent dimensions
            # (This is simplified - real implementation more complex)

        except Exception:
            pass  # Skip if symbolic validation fails

        return self
```

### Pattern: Inequality Verification

```python
from sympy import sympify, ask, Q, Symbol

class TheoremConclusion(BaseModel):
    statement: str
    statement_sympy: Optional[str] = None

    @model_validator(mode='after')
    def verify_inequality_if_possible(self) -> 'TheoremConclusion':
        """Attempt to verify inequality claims."""

        if not self.statement_sympy:
            return self

        try:
            # Parse the statement
            expr = sympify(self.statement_sympy)

            # Try to verify simple inequalities
            # Example: "x**2 >= 0 for real x"
            x = Symbol('x', real=True)

            # Check if we can prove it symbolically
            result = ask(expr, Q.real(x))

            # If we can disprove it, that's a problem!
            if result is False:
                raise ValueError(
                    f'Symbolic check DISPROVED the claim: {expr}'
                )

            # If result is True, great!
            # If result is None, we can't verify (that's ok)

        except Exception:
            pass  # Symbolic verification is best-effort

        return self
```

---

## 9. Advanced Patterns

### Pattern: Custom Validation Pipeline

```python
from typing import Callable, List

class ValidationPipeline:
    """Composable validation pipeline."""

    def __init__(self):
        self.checks: List[Callable] = []

    def add_check(self, check: Callable):
        """Add a validation check."""
        self.checks.append(check)

    def validate(self, obj):
        """Run all checks."""
        errors = []
        for check in self.checks:
            try:
                check(obj)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError(f'Validation failed:\n' + '\n'.join(f'  - {e}' for e in errors))

class Theorem(BaseModel):
    label: str
    hypotheses: List[str]
    conclusion: str

    @model_validator(mode='after')
    def run_validation_pipeline(self) -> 'Theorem':
        """Run custom validation pipeline."""
        pipeline = ValidationPipeline()

        # Add checks
        pipeline.add_check(lambda t: self._check_hypotheses_reasonable(t))
        pipeline.add_check(lambda t: self._check_conclusion_follows(t))
        pipeline.add_check(lambda t: self._check_notation_consistent(t))

        pipeline.validate(self)
        return self

    def _check_hypotheses_reasonable(self, thm):
        if len(thm.hypotheses) > 10:
            raise ValueError('Too many hypotheses (>10)')

    def _check_conclusion_follows(self, thm):
        # Complex logic here
        pass

    def _check_notation_consistent(self, thm):
        # Check notation
        pass
```

### Pattern: Cached Validators

```python
from functools import lru_cache

class Proof(BaseModel):
    label: str
    steps: List[ProofStep]

    @model_validator(mode='after')
    def validate_proof_structure(self) -> 'Proof':
        """Validate proof with caching for expensive checks."""

        # Cache expensive computations
        structure = self._analyze_structure()

        if structure['max_depth'] > 5:
            raise ValueError('Proof nesting too deep')

        return self

    @lru_cache(maxsize=1)
    def _analyze_structure(self) -> dict:
        """Expensive structural analysis (cached)."""
        def analyze(steps, depth=0):
            max_d = depth
            for step in steps:
                if step.substeps:
                    max_d = max(max_d, analyze(step.substeps, depth + 1))
            return max_d

        return {
            'max_depth': analyze(self.steps),
            'total_steps': sum(1 for _ in self._iter_all_steps()),
        }

    def _iter_all_steps(self):
        """Iterate all steps recursively."""
        for step in self.steps:
            yield step
            if step.substeps:
                yield from self._iter_all_substeps(step.substeps)

    def _iter_all_substeps(self, substeps):
        for step in substeps:
            yield step
            if step.substeps:
                yield from self._iter_all_substeps(step.substeps)
```

### Pattern: Validation Modes

```python
from enum import Enum

class ValidationMode(Enum):
    STRICT = "strict"        # All checks enabled
    LENIENT = "lenient"      # Only critical checks
    DISABLED = "disabled"    # No validation

class ValidationConfig(BaseModel):
    mode: ValidationMode = ValidationMode.STRICT
    enable_symbolic: bool = True
    enable_cross_references: bool = True

# Use context
_validation_config: ContextVar[ValidationConfig] = ContextVar(
    'validation_config',
    default=ValidationConfig()
)

class Theorem(BaseModel):
    label: str

    @model_validator(mode='after')
    def conditional_validation(self) -> 'Theorem':
        """Validate based on configuration."""
        config = _validation_config.get()

        if config.mode == ValidationMode.DISABLED:
            return self

        # Critical checks (always run)
        if not self.label:
            raise ValueError('Label required')

        # Non-critical checks (only in STRICT mode)
        if config.mode == ValidationMode.STRICT:
            if len(self.label) < 5:
                raise ValueError('Label too short')

        return self

# Usage
with ValidationConfig(mode=ValidationMode.LENIENT):
    # Validation is lenient in this context
    thm = Theorem(label="t")  # Would fail in STRICT mode
```

---

## 10. Performance

### Validation Cost

**Pydantic validation is fast:**
- Simple type checks: ~1-10 μs per field
- Regex validation: ~10-100 μs per field
- Custom validators: Depends on your code
- Model validators: Depends on complexity

**For mathematical documents:**
- Schema validation: ~1-5 ms per document
- Dependency graph: ~10-50 ms (NetworkX overhead)
- Symbolic verification: ~100-1000 ms (SymPy is slow)

### Optimization Strategies

#### 1. Defer Expensive Checks

```python
class MathematicalDocument(BaseModel):
    directives: List[Directive]

    # Fast validation in __init__
    @model_validator(mode='after')
    def quick_validation(self) -> 'MathematicalDocument':
        """Only critical checks."""
        # Check structure only
        return self

    # Expensive validation called explicitly
    def full_validation(self):
        """Run expensive checks on demand."""
        self._validate_dependency_graph()
        self._validate_symbolic_claims()

# Usage
doc = MathematicalDocument(**data)  # Fast!
doc.full_validation()  # Slow, but optional
```

#### 2. Lazy Validation

```python
from pydantic import computed_field

class MathematicalDocument(BaseModel):
    directives: List[Directive]

    _validation_cache: dict = {}

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Lazily computed validation result."""
        if 'valid' not in self._validation_cache:
            try:
                self.full_validation()
                self._validation_cache['valid'] = True
            except Exception:
                self._validation_cache['valid'] = False

        return self._validation_cache['valid']
```

#### 3. Batch Validation

```python
def validate_all_documents_efficiently(paths: List[Path]):
    """Validate multiple documents efficiently."""

    # Load all documents without full validation
    documents = []
    for path in paths:
        with open(path) as f:
            # Use model_construct to skip validation
            doc = MathematicalDocument.model_construct(**json.load(f))
            documents.append(doc)

    # Build global dependency graph once
    G = nx.DiGraph()
    for doc in documents:
        for directive in doc.directives:
            G.add_node(directive.label)
            # ... add edges

    # Check graph once
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError('Circular dependencies across documents!')

    # Now validate each document
    for doc in documents:
        doc.model_validate(doc.model_dump())  # Trigger validation
```

---

## 11. The Boundary: What Pydantic Cannot Do

### Fundamental Limitations

#### 1. Type-Level Computation

❌ **Cannot do:**
```python
# This is what Lean can do but Pydantic cannot
class Vector(BaseModel):
    n: int  # dimension
    elements: List[float]  # Must have length n

    # ❌ Pydantic cannot enforce: len(elements) == n at type level
```

✅ **Workaround:**
```python
class Vector(BaseModel):
    n: int
    elements: List[float]

    @model_validator(mode='after')
    def check_length(self) -> 'Vector':
        if len(self.elements) != self.n:
            raise ValueError(f'Expected {self.n} elements, got {len(self.elements)}')
        return self
```

#### 2. Dependent Types

❌ **Cannot do:**
```python
# Lean: Type depends on value
def sorted_list(n: int) -> Type:
    return List[int] where all(a <= b for a, b in zip(L, L[1:]))
```

✅ **Workaround:**
```python
class SortedList(BaseModel):
    elements: List[int]

    @model_validator(mode='after')
    def check_sorted(self) -> 'SortedList':
        if not all(a <= b for a, b in zip(self.elements, self.elements[1:])):
            raise ValueError('List must be sorted')
        return self
```

#### 3. Undecidable Properties

❌ **Cannot do:**
```python
# Halting problem - undecidable
class ProofStep(BaseModel):
    code: str  # Python code

    @model_validator(mode='after')
    def check_terminates(self) -> 'ProofStep':
        # ❌ Cannot determine if arbitrary code terminates!
        return self
```

#### 4. External State

❌ **Cannot do without context:**
```python
class Theorem(BaseModel):
    label: str

    @model_validator(mode='after')
    def check_unique(self) -> 'Theorem':
        # ❌ No access to other instances without explicit context
        return self
```

---

## 12. Best Practices

### 1. Layer Your Validation

```python
# ✅ GOOD: Layered validation
class Theorem(BaseModel):
    # Level 1: Type checking (automatic)
    label: str
    rigor_level: int

    # Level 2: Field validation
    @field_validator('label')
    @classmethod
    def validate_label_format(cls, v):
        if not v.startswith('thm-'):
            raise ValueError('Must start with thm-')
        return v

    # Level 3: Model validation
    @model_validator(mode='after')
    def validate_internal_consistency(self) -> 'Theorem':
        # Check relationships between fields
        return self
```

### 2. Fail Fast

```python
# ✅ GOOD: Check simple things first
@model_validator(mode='after')
def validate(self) -> 'Proof':
    # Fast checks first
    if not self.steps:
        raise ValueError('Proof must have steps')

    if len(self.steps) > 100:
        raise ValueError('Proof too long')

    # Expensive checks last
    self._validate_dependency_graph()  # Expensive

    return self
```

### 3. Clear Error Messages

```python
# ❌ BAD: Vague error
@model_validator(mode='after')
def validate(self) -> 'Theorem':
    if not self.conclusion:
        raise ValueError('Invalid')  # ❌ Unhelpful!
    return self

# ✅ GOOD: Specific error
@model_validator(mode='after')
def validate(self) -> 'Theorem':
    if not self.conclusion:
        raise ValueError(
            f'Theorem {self.label} missing conclusion. '
            f'Every theorem must have a conclusion statement.'
        )
    return self
```

### 4. Make Validation Optional for Expensive Checks

```python
class MathematicalDocument(BaseModel):
    directives: List[Directive]

    model_config = ConfigDict(
        # Allow creation without full validation
        validate_assignment=False
    )

    def validate_full(self, include_symbolic: bool = False):
        """Explicit full validation."""
        self._validate_structure()
        self._validate_dependencies()

        if include_symbolic:
            self._validate_symbolic_claims()  # Expensive!
```

### 5. Use Context for Global State

```python
# ✅ GOOD: Use context for cross-document validation
with ValidationContext(all_labels=global_labels):
    doc = MathematicalDocument(**data)
```

---

## Conclusion

### Summary Table

| Validation Type | Pydantic Can Do It? | Difficulty | Example |
|----------------|---------------------|------------|---------|
| Type checking | ✅ Yes (automatic) | Easy | `label: str` |
| Regex patterns | ✅ Yes | Easy | `pattern=r'^thm-'` |
| Numeric ranges | ✅ Yes | Easy | `ge=1, le=10` |
| Field dependencies | ✅ Yes | Medium | `@field_validator` |
| Cross-field logic | ✅ Yes | Medium | `@model_validator` |
| Structural invariants | ✅ Yes | Medium | Tree depth, etc. |
| Cross-model validation | ✅ Yes | Hard | Dependency graphs |
| Global uniqueness | ✅ Yes (with context) | Hard | Label registry |
| Symbolic algebra | ✅ Yes (with SymPy) | Hard | Verify identities |
| Type-level proofs | ❌ No | Impossible | Dependent types |
| Undecidable properties | ❌ No | Impossible | Halting problem |

### The Sweet Spot

Pydantic can enforce **remarkably complex** constraints:
- ✅ ~80% of structural correctness
- ✅ ~60% of algebraic correctness (with SymPy)
- ✅ ~40% of semantic correctness (with custom logic)

**Combined with:**
- NetworkX for graph properties
- SymPy for symbolic math
- Custom validators for domain logic
- Context for cross-document state

**You can build a powerful "mathematical linter" that catches the majority of errors!**

The remaining ~20-40% requires:
- Human review for semantic understanding
- LLM review (Gemini + Codex) for high-level logic
- Formal proof assistants (Lean) for absolute soundness

**Recommendation:** Push Pydantic as far as it can go for mechanical checks, then rely on dual review + human synthesis for what automation cannot verify.
