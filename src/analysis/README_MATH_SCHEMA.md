# Mathematical Documentation Schema - Pydantic Module

## Overview

This module provides type-safe Pydantic models that mirror `math_schema.json`, enabling programmatic creation, validation, and manipulation of mathematical documents for the Fragile Gas framework.

**Files:**
- `math_schema.py` - Pydantic BaseModel definitions
- `math_schema_example.py` - Example usage and demonstrations
- `render_math_json.py` - Renders JSON documents to markdown
- `validate_math_schema.py` - JSON Schema validation utility

## Installation

No additional dependencies required beyond the project's existing requirements:
- `pydantic>=2.0` (already in project)
- Standard library only

## Quick Start

### Create a Document Programmatically

```python
from src.analysis.math_schema import (
    MathematicalDocument,
    Definition,
    DefinedObject,
    Metadata,
)
from datetime import date

# Create metadata
metadata = Metadata(
    title="My Theorem",
    document_id="my_theorem",
    version="1.0.0",
    authors=["Your Name"],
    date_created=date.today(),
)

# Create a definition
my_def = Definition(
    type="definition",
    label="def-my-object",
    title="My Object",
    statement="Definition of my mathematical object",
    defined_objects=[
        DefinedObject(
            name="My Object",
            symbol="O",
            mathematical_definition="O := \\{x \\in X : P(x)\\}",
            type="set",
        )
    ],
)

# Assemble document
doc = MathematicalDocument(
    metadata=metadata,
    directives=[my_def],
)

# Save to JSON
with open("my_document.json", "w") as f:
    f.write(doc.model_dump_json(indent=2, exclude_none=True))
```

### Validate Existing JSON

```python
import json
from src.analysis.math_schema import MathematicalDocument

# Load and validate
with open("existing_document.json") as f:
    data = json.load(f)

doc = MathematicalDocument(**data)
# ✅ If this succeeds, the document is valid!

print(f"Loaded: {doc.metadata.title}")
print(f"Directives: {len(doc.directives)}")
```

### Access Directive Data

```python
# Filter directives by type
theorems = [d for d in doc.directives if d.type == "theorem"]
proofs = [d for d in doc.directives if d.type == "proof"]

# Access specific fields with type safety
for thm in theorems:
    print(f"{thm.title}: {thm.conclusion.statement}")
    if thm.importance == "foundational":
        print("  ⭐ Foundational result!")
```

## Features

### ✅ Type Safety

All fields are fully typed with appropriate constraints:

```python
from src.analysis.math_schema import Theorem, ReviewScore

# ✅ This works
score = ReviewScore(
    reviewer="gemini-2.5-pro",
    review_date=date(2025, 10, 25),
    rigor=9,
    soundness=8,
    consistency=9,
    verdict="minor-revisions",
)

# ❌ This fails validation
score = ReviewScore(
    reviewer="invalid-reviewer",  # Must be from enum
    review_date=date(2025, 10, 25),
    rigor=15,  # Must be 1-10
    soundness=8,
    consistency=9,
    verdict="maybe-ready",  # Must be from enum
)
# ValidationError!
```

### ✅ Automatic Validation

Pydantic enforces all constraints from `math_schema.json`:

- **Label format**: Must be kebab-case (`^[a-z][a-z0-9-]*[a-z0-9]$`)
- **Version format**: Must be semantic versioning (`1.0.0`)
- **Numeric constraints**: Scores 1-10, completeness 0-100
- **Enum validation**: All enums enforced (verdict, severity, stage, etc.)
- **Required fields**: Cannot omit required fields

### ✅ IDE Autocomplete

With Pydantic models, your IDE provides:
- Field name completion
- Type hints
- Inline documentation
- Error detection before runtime

### ✅ JSON Schema Generation

Generate JSON Schema from Pydantic models:

```python
from src.analysis.math_schema import MathematicalDocument

schema = MathematicalDocument.model_json_schema()
# Compatible with math_schema.json!
```

## Complete Type Reference

### Core Types

- `Label` - Kebab-case identifier
- `MathExpression` - LaTeX string
- `DirectiveType` - Literal union of all directive types
- `Verdict` - Publication readiness verdict
- `Severity` - Issue severity level

### Mathematical Structures

- `CrossReference` - Reference to another directive
- `MathematicalProperty` - Property with optional quantitative info
- `MathematicalAssumption` - Hypothesis in theorem/lemma
- `ComputationalVerification` - Computational proof verification

### Review & Development Tracking

- `ReviewIssue` - Single issue from review
- `ReviewScore` - Score from single reviewer (Gemini/Codex)
- `DualReviewAnalysis` - Combined Gemini + Codex analysis
- `DevelopmentStatus` - Maturity tracking (sketch → verified → published)
- `SketchProofLinkage` - Link proof to source sketch

### Directive Types

All 13 directive types are available:

- `Definition` - Mathematical definitions
- `Axiom` - Framework axioms
- `Theorem` - Main theorems
- `Lemma` - Supporting lemmas
- `Proposition` - Propositions
- `Corollary` - Corollaries
- `Proof` - Proofs with hierarchical steps
- `Algorithm` - Algorithm specifications
- `Remark` - Remarks and notes
- `Observation` - Observations
- `Conjecture` - Conjectures
- `Example` - Worked examples
- `Property` - Mathematical properties

### Document Structure

- `Metadata` - Document metadata (title, version, authors, readiness)
- `DependencyGraph` - Relationship graph between directives
- `MathematicalDocument` - Complete document with metadata + directives

## Advanced Usage

### Hierarchical Proof Steps

```python
from src.analysis.math_schema import Proof, ProofStep

proof = Proof(
    type="proof",
    label="proof-my-theorem",
    title="My Proof",
    statement="Detailed proof",
    proves=CrossReference(label="thm-my-theorem", type="theorem"),
    strategy="We use induction...",
    steps=[
        ProofStep(
            id="Step 1",
            title="Base Case",
            content="For n=1, ...",
            substeps=[
                ProofStep(
                    id="Step 1.1",
                    content="Verify initial condition",
                    techniques=["direct-computation"],
                ),
                ProofStep(
                    id="Step 1.2",
                    content="Show property holds",
                    justification="By definition",
                ),
            ],
        ),
        ProofStep(
            id="Step 2",
            title="Inductive Step",
            content="Assume for n=k, prove for n=k+1...",
        ),
    ],
)
```

### Publication Readiness Tracking

```python
from src.analysis.math_schema import (
    DualReviewAnalysis,
    ReviewScore,
    ReviewIssue,
)

# Record Gemini review
gemini_review = ReviewScore(
    reviewer="gemini-2.5-pro",
    review_date=date(2025, 10, 25),
    rigor=8,
    soundness=9,
    consistency=8,
    verdict="minor-revisions",
    issues_identified=[
        ReviewIssue(
            severity="minor",
            title="Missing intermediate step",
            location={"section": "§3.2", "proof_step": "Step 2"},
            problem="Gap in reasoning from Eq. (7) to Eq. (8)",
            suggested_fix="Add explicit calculation showing continuity",
        )
    ],
)

# Record Codex review
codex_review = ReviewScore(
    reviewer="codex",
    review_date=date(2025, 10, 25),
    rigor=9,
    soundness=9,
    consistency=9,
    verdict="ready",
)

# Create dual review analysis
dual_review = DualReviewAnalysis(
    gemini_review=gemini_review,
    codex_review=codex_review,
    final_verdict="minor-revisions",
)

# Add to theorem
my_theorem.peer_review = dual_review
```

### Sketch-to-Proof Expansion Tracking

```python
from src.analysis.math_schema import (
    SketchProofLinkage,
    SourceSketch,
    ExpansionHistoryEntry,
    DevelopmentStage,
)

linkage = SketchProofLinkage(
    source_sketch=SourceSketch(
        file_path="sketcher/sketch_thm_convergence.md",
        label="sketch-thm-convergence",
        date_created=date(2025, 10, 20),
        agent="gemini",
    ),
    expansion_history=[
        ExpansionHistoryEntry(
            expansion_date=date(2025, 10, 21),
            stage="partial",
            agent="claude",
            description="Expanded outline to detailed proof structure",
        ),
        ExpansionHistoryEntry(
            expansion_date=date(2025, 10, 22),
            stage="complete",
            agent="claude",
            description="Filled in all proof steps with justifications",
        ),
    ],
    sketch_coverage={"coverage_percentage": 95, "uncovered_items": ["Lemma 3.2 proof"]},
)

my_proof.sketch_linkage = linkage
```

## Workflow Integration

### 1. Create Documents from Code

Use Pydantic models to generate mathematical documents from algorithm implementations:

```python
# After implementing an algorithm, generate its documentation
def document_my_algorithm(algorithm_impl):
    from src.analysis.math_schema import Algorithm, AlgorithmStep

    return Algorithm(
        type="algorithm",
        label=f"alg-{algorithm_impl.__name__}",
        title=algorithm_impl.__name__.replace("_", " ").title(),
        statement=algorithm_impl.__doc__,
        inputs=[...],  # Parse from signature
        outputs=[...],
        steps=[...],  # Extract from implementation
    )
```

### 2. Validate Documents in CI/CD

```python
# test_document_validity.py
import json
from pathlib import Path
from src.analysis.math_schema import MathematicalDocument

def test_all_documents_valid():
    """Ensure all JSON documents validate against schema."""
    docs_dir = Path("docs/source/json")

    for json_file in docs_dir.glob("**/*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # This will raise ValidationError if invalid
        doc = MathematicalDocument(**data)
        assert doc.metadata.title, f"{json_file} missing title"
```

### 3. Generate Reports

```python
def generate_readiness_report(doc: MathematicalDocument):
    """Generate publication readiness report."""

    if not doc.metadata.publication_readiness_aggregate:
        return "Not reviewed yet"

    agg = doc.metadata.publication_readiness_aggregate

    report = f"""
    Publication Readiness Report
    ============================
    Document: {doc.metadata.title}

    Overall Verdict: {agg.overall_verdict}
    Average Rigor: {agg.aggregate_scores.rigor:.1f}/10

    Directive Summary:
    - Total: {agg.directive_summary.total_directives}
    - Ready: {agg.directive_summary.ready_count}
    - Need revisions: {agg.directive_summary.minor_revisions_count}

    Blocking Issues: {len(agg.blocking_issues or [])}
    """
    return report
```

## Common Patterns

### Loading and Modifying Documents

```python
# Load existing document
with open("theorem.json") as f:
    doc = MathematicalDocument(**json.load(f))

# Modify in place
doc.metadata.version = "1.1.0"
doc.metadata.date_modified = date.today()

# Add a new directive
new_lemma = Lemma(...)
doc.directives.append(new_lemma)

# Save back
with open("theorem.json", "w") as f:
    f.write(doc.model_dump_json(indent=2, exclude_none=True))
```

### Filtering and Querying

```python
# Find all foundational theorems
foundational = [
    d for d in doc.directives
    if d.type == "theorem" and d.importance == "foundational"
]

# Find all directives needing review
needs_review = [
    d for d in doc.directives
    if not hasattr(d, 'peer_review') or d.peer_review is None
]

# Find all proofs with low rigor
low_rigor_proofs = [
    d for d in doc.directives
    if d.type == "proof" and (d.rigor_level or 0) < 7
]
```

## Tips & Best Practices

### 1. Use `exclude_none=True` for Clean JSON

```python
# ✅ Clean output - omits all None fields
doc.model_dump_json(indent=2, exclude_none=True)

# ❌ Cluttered - includes all optional fields as null
doc.model_dump_json(indent=2)
```

### 2. Validate Early and Often

```python
# Validate as you build
def add_theorem_with_validation(doc, theorem_data):
    # This validates immediately
    thm = Theorem(**theorem_data)

    # Now we know it's valid
    doc.directives.append(thm)
    return doc
```

### 3. Use Type Hints

```python
from typing import List
from src.analysis.math_schema import Theorem, Proof

def extract_theorems(doc: MathematicalDocument) -> List[Theorem]:
    """Type hints enable IDE autocomplete and type checking."""
    return [d for d in doc.directives if isinstance(d, Theorem)]
```

### 4. Leverage Union Types

```python
from src.analysis.math_schema import Directive

def process_directive(directive: Directive):
    """Directive is a Union of all directive types."""
    match directive.type:
        case "theorem":
            # directive is narrowed to Theorem
            print(directive.conclusion.statement)
        case "proof":
            # directive is narrowed to Proof
            print(f"Proves: {directive.proves.label}")
```

## Troubleshooting

### ValidationError: Field required

```python
# ❌ Missing required field
Definition(
    type="definition",
    label="def-test",
    title="Test",
    # Missing: statement, defined_objects
)

# ✅ Include all required fields
Definition(
    type="definition",
    label="def-test",
    title="Test",
    statement="Test definition",
    defined_objects=[...],  # Required!
)
```

### ValidationError: Input should be a valid enum

```python
# ❌ Invalid enum value
ReviewScore(
    reviewer="gpt-4",  # Not in Reviewer enum!
    ...
)

# ✅ Use valid enum value
ReviewScore(
    reviewer="gemini-2.5-pro",  # Valid!
    ...
)
```

### ValidationError: String should match pattern

```python
# ❌ Invalid label format
Definition(
    label="Def_Test",  # Underscores not allowed!
    ...
)

# ✅ Use kebab-case
Definition(
    label="def-test",  # Valid kebab-case
    ...
)
```

## See Also

- `math_schema.json` - The source JSON Schema
- `render_math_json.py` - Markdown rendering
- `math_schema_example.py` - Complete working examples
- [Pydantic Documentation](https://docs.pydantic.dev/)
