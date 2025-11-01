# Mathematical Document Processing Pipeline

**Comprehensive guide to the Extract-then-Enrich pipeline for transforming mathematical documents into structured, validated, machine-readable formats.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Module Organization](#module-organization)
5. [Key Classes and Data Flow](#key-classes-and-data-flow)
6. [Complete Workflow](#complete-workflow)
7. [API Reference](#api-reference)
8. [Design Patterns](#design-patterns)
9. [Examples](#examples)

---

## Overview

The `mathster` module implements a **two-stage Extract-then-Enrich pipeline** for processing mathematical documents written in Jupyter Book MyST markdown format. The system transforms human-readable mathematical text into structured, validated, machine-readable entities suitable for:

- **Automated proof validation**
- **Dependency graph construction**
- **Symbolic computation integration**
- **Knowledge base construction**
- **Document consistency checking**

### Design Philosophy

The pipeline separates two fundamentally different tasks:

1. **Stage 1 (Extraction)**: LLM performs **shallow transcription** — verbatim copying of mathematical entities with minimal interpretation
2. **Stage 2 (Enrichment)**: Python orchestrator performs **semantic processing** — cross-referencing, validation, and assembly into final models

This separation enables:
- **Robustness**: Structural parsing is reliable (Python), semantic understanding is flexible (LLM)
- **Debuggability**: Each stage produces inspectable intermediate outputs
- **Scalability**: Sections can be processed in parallel
- **Maintainability**: Clear separation of concerns

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MATHEMATICAL DOCUMENT                       │
│                  (Jupyter Book MyST Markdown)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  STAGE 0: PREPARATION         │
                │  - Split by sections          │
                │  - Extract directive hints    │
                │  - Hybrid parsing             │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  STAGE 1: RAW EXTRACTION      │
                │  - LLM: Shallow transcription │
                │  - Goal: Completeness         │
                │  - Output: StagingDocument    │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  STAGE 2: ENRICHMENT          │
                │  - Python: Orchestration      │
                │  - LLM: Focused semantic      │
                │  - Cross-reference resolution │
                │  - Validation                 │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  FINAL: ASSEMBLY              │
                │  - MathematicalDocument       │
                │  - Registry integration       │
                │  - Graph construction         │
                └───────────────────────────────┘
```

---

## Pipeline Stages

### Stage 0: Document Preparation

**Purpose**: Split document and extract structural information

**Modules**: `tools/directive_parser.py`, `tools/line_finder.py`

**Process**:
1. **Section Splitting**: Document is split by headings (H1-H6) into `DocumentSection` objects
2. **Directive Extraction**: Python regex extracts Jupyter Book directives:
   - `{prf:definition}`, `{prf:theorem}`, `{prf:lemma}`, `{prf:proof}`, etc.
   - Extracts: directive type, label, line range, content
3. **Hint Generation**: Create `DirectiveHint` objects to guide LLM extraction

**Key Functions**:
```python
split_into_sections(markdown_text: str) -> List[DocumentSection]
extract_jupyter_directives(markdown_text: str) -> List[DirectiveHint]
format_directive_hints_for_llm(directives: List[DirectiveHint]) -> str
```

**Output**: List of `DocumentSection` with embedded `DirectiveHint` metadata

---

### Stage 1: Raw Extraction

**Purpose**: LLM performs verbatim transcription of mathematical entities

**Modules**: `staging_types.py`, `prompts/extraction.py`, `llm/llm_interface.py`

**Process**:
1. For each section, format directive hints + content
2. Call LLM with `MAIN_EXTRACTION_PROMPT`
3. LLM returns JSON with raw entities:
   - `RawDefinition`: Verbatim definition text
   - `RawTheorem`: Statement with context
   - `RawProof`: Proof text with citations
   - `RawAxiom`, `RawEquation`, `RawParameter`, `RawRemark`, `RawCitation`
4. Validate JSON against Pydantic models
5. Return `StagingDocument` per section

**Key Principles**:
- **Verbatim extraction**: Preserve exact LaTeX, notation, wording
- **No interpretation**: Don't simplify or rephrase
- **Complete context**: Extract text before/after for semantic understanding
- **Descriptive labels**: Assign `thm-exponential-convergence`, `def-walker-state`, etc. for tracking

**Models**:
```python
class RawTheorem(BaseModel):
    label: str                        # "thm-exponential-convergence"
    label_text: str                   # "Theorem 3.1"
    statement_type: Literal[...]      # "theorem", "lemma", etc.
    context_before: Optional[str]     # Preceding paragraph
    full_statement_text: str          # Complete statement (verbatim)
    conclusion_formula_latex: Optional[str]
    equation_label: Optional[str]     # "(3.1)" if numbered
    explicit_definition_references: List[str]
```

**Output**: `StagingDocument` containing all raw entities from the section

---

### Stage 2: Semantic Enrichment

**Purpose**: Transform raw entities into structured, validated final models

**Modules**: `orchestration.py`, `prompts/enrichment.py`, `llm/pipeline_orchestration.py`

**Process**:

1. **Resolution Context Setup**:
   ```python
   ctx = ResolutionContext()
   ctx.add_staging_document(staging_doc)
   ```
   - Builds reverse lookups: label_text → label, term → definition, etc.

2. **Entity-by-Entity Enrichment**:
   For each raw entity:

   a. **Decomposition** (if needed):
      - LLM decomposes theorem into assumptions + conclusion
      - LLM parses LaTeX into SymPy-compatible form

   b. **Cross-Reference Resolution**:
      - Resolve "Theorem 2.1" → `thm-keystone` (final label)
      - Resolve "Walker State" → `def-walker-state`
      - Link proofs to theorems

   c. **Semantic Parsing**:
      - Convert LaTeX to dual representation (LaTeX + SymPy)
      - Extract input/output objects
      - Identify properties required/established

   d. **Validation**:
      - Check label uniqueness
      - Verify referential integrity
      - Validate against Pydantic models

   e. **Assembly**:
      - Create final model (`TheoremBox`, `DefinitionBox`, `ProofBox`, etc.)
      - Assign permanent labels

3. **Error Handling**:
   ```python
   try:
       enriched = enrich_theorem(raw_thm, ctx)
   except EnrichmentError as e:
       log_error(e.error_type, e.entity_id, e.context)
   ```

**Models**:
```python
class TheoremBox(BaseModel):
    label: str                        # "thm-keystone"
    name: str                         # "Keystone Principle"
    statement: str                    # Full statement
    assumptions: List[str]            # Decomposed assumptions
    conclusion: str                   # Decomposed conclusion
    input_objects: List[str]          # Object IDs required
    output_type: TheoremOutputType    # PROPERTY, RELATION, etc.
    properties_required: Dict[str, List[str]]  # obj → [prop1, prop2]
    properties_established: List[Property]     # Properties proven
    proof: Optional[ProofBox]         # Attached proof
    source: SourceLocation            # File, line range
```

**Output**: Enriched entities organized in `EnrichedEntities` container

---

### Final: Assembly

**Purpose**: Create complete `MathematicalDocument` with both raw and enriched data

**Modules**: `llm/document_container.py`

**Process**:
1. Merge section staging documents
2. Organize enriched entities by type
3. Compute statistics
4. Create `MathematicalDocument`

**Model**:
```python
class MathematicalDocument(BaseModel):
    document_id: str                  # Unique identifier
    chapter: Optional[str]            # "1_euclidean_gas"
    staging: StagingDocument          # All raw entities
    enriched: EnrichedEntities        # All enriched entities

    @property
    def total_raw_entities(self) -> int

    @property
    def total_enriched_entities(self) -> int

    @property
    def enrichment_rate(self) -> float  # Percentage successfully enriched
```

---

## Module Organization

The `mathster` package contains **44 Python files** organized as follows:

```
src/fragile/proofs/
├── __init__.py                   # Re-exports all public APIs
├── staging_types.py              # Raw extraction models (Stage 1)
├── orchestration.py              # Enrichment orchestration (Stage 2)
├── schema_generator.py           # JSON schema generation for LLM
├── error_tracking.py             # Error logging and recovery
│
├── core/                         # Final structured models (13 files)
│   ├── __init__.py
│   ├── math_types.py             # MathematicalObject, Property, TheoremBox
│   ├── proof_system.py           # ProofBox, ProofStep, property-level dataflow
│   ├── enriched_types.py         # EquationBox, ParameterBox, RemarkBox
│   ├── pipeline_types.py         # Pipeline-specific types
│   ├── proof_integration.py      # Theorem ↔ Proof integration
│   ├── article_system.py         # SourceLocation, article metadata
│   ├── bibliography.py           # Citation and bibliography management
│   ├── review_system.py          # LLM review integration
│   └── review_helpers.py         # Review utilities
│
├── llm/                          # LLM integration (3 files)
│   ├── __init__.py
│   ├── llm_interface.py          # Anthropic API interface
│   ├── pipeline_orchestration.py # Stage 1 & 2 orchestration
│   └── document_container.py     # MathematicalDocument, EnrichedEntities
│
├── prompts/                      # LLM prompts (2 files)
│   ├── __init__.py
│   ├── extraction.py             # Stage 1: MAIN_EXTRACTION_PROMPT
│   └── enrichment.py             # Stage 2: DECOMPOSE_THEOREM_PROMPT, etc.
│
├── tools/                        # Document parsing (3 files)
│   ├── __init__.py
│   ├── directive_parser.py       # Jupyter Book directive extraction
│   └── line_finder.py            # Source location utilities
│
├── registry/                     # Entity storage (4 files)
│   ├── __init__.py
│   ├── registry.py               # MathematicalRegistry (central index)
│   ├── storage.py                # JSON serialization/deserialization
│   ├── reference_system.py       # Reference resolution, tag queries
│   ├── article_registry.py       # Article-level registry
│   └── review_registry.py        # Review tracking
│
├── relationships/                # Graph system (1 file)
│   ├── __init__.py
│   └── graphs.py                 # Relationship graphs, equivalence classes
│
├── sympy/                        # Symbolic computation (5 files)
│   ├── __init__.py
│   ├── dual_representation.py    # LaTeX/SymPy dual representation
│   ├── expressions.py            # SymPy expression handling
│   ├── validation.py             # Symbolic validation
│   ├── proof_integration.py      # SymPy ↔ Proof integration
│   └── object_extensions.py      # MathematicalObject ↔ SymPy
│
└── validators/                   # Validation (1 file)
    ├── __init__.py
    └── framework_checker.py      # Framework consistency validation
```

---

## Key Classes and Data Flow

### Stage 1: Raw Models

**Purpose**: Direct transcription targets for LLM

```python
RawDefinition       # Definition text + term
RawTheorem          # Statement + context
RawProof            # Proof text + citations
RawAxiom            # Foundational axiom
RawEquation         # Equation + label
RawParameter        # Parameter definition
RawRemark           # Remark, note, example
RawCitation         # Bibliographic reference

StagingDocument     # Container for all raw entities
```

**Characteristics**:
- Frozen (immutable)
- String-heavy (minimal interpretation)
- Descriptive labels (`thm-exponential-convergence`, `def-walker-state`)
- Full context preservation

---

### Stage 2: Orchestration Types

**Purpose**: Coordinate enrichment process

```python
class ResolutionContext(BaseModel):
    """Cross-referencing knowledge base during enrichment."""

    # Entity storage (label → entity)
    definitions: Dict[str, RawDefinition]
    theorems: Dict[str, RawTheorem]
    proofs: Dict[str, RawProof]
    # ... other entity types

    # Reverse lookups (for fast resolution)
    label_text_to_theorem: Dict[str, str]     # "Theorem 2.1" → "thm-exponential-convergence"
    term_to_definition: Dict[str, str]        # "walker" → "def-walker-state"

    # Methods
    def resolve_theorem_reference(self, label_text: str) -> Optional[str]
    def resolve_definition_reference(self, term: str) -> Optional[str]
    def find_proof_for_theorem(self, theorem_label_text: str) -> Optional[RawProof]
```

```python
class EnrichmentError(Exception):
    """Exception for enrichment failures."""
    error_type: ErrorType              # PARSE_FAILURE, REFERENCE_UNRESOLVED, etc.
    entity_id: str                     # "thm-exponential-convergence"
    raw_data: Dict[str, Any]           # Preserved for retry
    context: Dict[str, Any]            # Additional debug info
```

---

### Final Models: Core Types

**Mathematical Entities**:

```python
class MathematicalObject(BaseModel):
    """Core mathematical object (set, function, operator, etc.)."""
    label: str                         # "obj-discrete-system"
    name: str                          # "Discrete System"
    expression: str                    # LaTeX expression
    object_type: ObjectType            # SET, FUNCTION, MEASURE, etc.
    current_properties: List[Property] # Properties established for this object
    tags: List[str]                    # ["euclidean-gas", "discrete"]
```

```python
class Property(BaseModel):
    """Property assigned to an object by a theorem."""
    label: str                         # "prop-lipschitz"
    expression: str                    # "L_φ ≤ 1"
    object_label: str                  # "obj-discrete-system"
    established_by: str                # "thm-keystone"
    timestamp: Optional[int]           # Pipeline execution step
```

```python
class TheoremBox(BaseModel):
    """Theorem, lemma, proposition, or corollary."""
    label: str                         # "thm-keystone"
    name: str                          # "Keystone Principle"
    statement: str                     # Full statement
    assumptions: List[str]             # Decomposed assumptions
    conclusion: str                    # Decomposed conclusion
    input_objects: List[str]           # ["obj-discrete-system"]
    output_type: TheoremOutputType     # PROPERTY, RELATION, CONVERGENCE, etc.
    properties_required: Dict[str, List[str]]  # Object → properties needed
    properties_established: List[Property]     # Properties proven
    proof: Optional[ProofBox]          # Attached proof (new in v2.0)
    source: SourceLocation             # File, line range
```

---

### Proof System

**Purpose**: Hierarchical, recursive proof representation with property-level dataflow

```python
class ProofBox(BaseModel):
    """Complete proof with recursive step structure."""
    proof_id: str                      # "proof-thm-keystone"
    theorem: TheoremReference          # Back-reference to theorem
    inputs: List[ProofInput]           # Objects + properties required
    outputs: List[ProofOutput]         # Objects + properties established
    strategy: str                      # High-level approach
    steps: List[ProofStep]             # Recursive step structure
    status: ProofStatus                # SKETCHED, EXPANDED, VERIFIED
```

```python
class ProofStep(BaseModel):
    """Single step in a proof."""
    step_number: int
    description: str                   # What this step does
    step_type: ProofStepType           # DIRECT_DERIVATION, SUB_PROOF, LEMMA_APPLICATION
    content: Union[DirectDerivation, ProofBox, LemmaApplication]
    status: ProofStepStatus            # SKETCHED, EXPANDED, VERIFIED
```

**Property-Level Dataflow**:
```python
class ProofInput(BaseModel):
    """Input specification: object + specific properties needed."""
    object_id: str
    required_properties: List[PropertyReference]
    required_assumptions: List[AssumptionReference]

class ProofOutput(BaseModel):
    """Output specification: object + properties established."""
    object_id: str
    properties_established: List[PropertyReference]
```

---

### Registry and Storage

```python
class MathematicalRegistry:
    """Central index for all mathematical entities."""

    # Collections
    _objects: Dict[str, MathematicalObject]
    _theorems: Dict[str, TheoremBox]
    _proofs: Dict[str, ProofBox]
    _axioms: Dict[str, Axiom]
    # ... other types

    # Indices
    _index.id_to_object: Dict[str, Any]          # Primary index
    _index.tag_to_ids: Dict[str, Set[str]]       # Tag queries
    _index.type_to_ids: Dict[str, Set[str]]      # Type queries

    # Methods
    def add(self, obj: Any) -> None
    def get(self, id: str) -> Optional[Any]
    def query_by_tags(self, query: TagQuery) -> List[Any]
    def validate_references(self) -> ValidationResult
```

**Persistence**:
```python
save_registry_to_directory(registry, path)
load_registry_from_directory(path) -> MathematicalRegistry
```

---

### Relationships and Graphs

```python
class Relationship(BaseModel):
    """Connection between mathematical objects."""
    label: str
    relationship_type: RelationType    # EQUIVALENCE, EMBEDDING, APPROXIMATION, etc.
    source_object: str                 # Object ID
    target_object: str                 # Object ID
    bidirectional: bool
    established_by: str                # Theorem ID
    expression: str                    # Mathematical expression

# Graph construction
graph = build_relationship_graph_from_registry(registry)
equivalence_classes = EquivalenceClassifier(graph).compute_equivalence_classes()
lineage = ObjectLineage(graph, "obj-discrete-system")
```

---

## Complete Workflow

### Basic Usage

```python
from mathster import process_document_from_file

# Process a single document
math_doc = process_document_from_file(
    file_path="docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
    model="claude-sonnet-4"
)

# Inspect results
print(math_doc.get_summary())
# Output:
#   Document ID: fragile_gas_framework
#   Raw Entities: 45
#   Enriched Entities: 42
#   Enrichment Rate: 93.3%
#
#   Raw:
#     - Definitions: 12
#     - Theorems: 8
#     - Proofs: 6
#     ...
#
#   Enriched:
#     - Definitions: 12
#     - Theorems: 7
#     - Proofs: 6
#     ...
```

---

### Advanced: Manual Stage Control

```python
from mathster import (
    split_into_sections,
    process_section,
    merge_sections,
    enrich_and_assemble,
)

# Read document
with open("path/to/document.md") as f:
    markdown_text = f.read()

# Stage 0: Split
sections = split_into_sections(markdown_text)
print(f"Found {len(sections)} sections")

# Stage 1: Extract (per section)
staging_docs = []
for section in sections:
    staging_doc = process_section(section, model="claude-sonnet-4")
    staging_docs.append(staging_doc)

# Merge sections
merged_staging = merge_sections(staging_docs)

# Stage 2: Enrich
math_doc = enrich_and_assemble(merged_staging)

# Inspect
print(f"Total entities: {math_doc.total_enriched_entities}")
```

---

### Integration with Registry

```python
from mathster import (
    MathematicalRegistry,
    save_registry_to_directory,
    load_registry_from_directory,
)

# Create registry
registry = MathematicalRegistry()

# Add entities from document
registry.add_all(math_doc.enriched.definitions.values())
registry.add_all(math_doc.enriched.theorems.values())
registry.add_all(math_doc.enriched.proofs.values())

# Save to disk
save_registry_to_directory(registry, "path/to/storage")

# Load later
loaded_registry = load_registry_from_directory("path/to/storage")

# Query
lipschitz_objects = registry.query_by_tags(TagQuery(tags=["lipschitz"]))
```

---

### Graph Analysis

```python
from mathster import (
    build_relationship_graph_from_registry,
    EquivalenceClassifier,
)

# Build graph
graph = build_relationship_graph_from_registry(registry)

# Analyze equivalence classes
classifier = EquivalenceClassifier(graph)
classes = classifier.compute_equivalence_classes()

for cls in classes:
    print(f"Equivalence class: {cls.canonical_object}")
    print(f"  Members: {cls.equivalent_objects}")
    print(f"  Establishing theorems: {cls.establishing_theorems}")
```

---

## API Reference

### Pipeline Entry Points

```python
# High-level API
process_document_from_file(
    file_path: str,
    model: str = "claude-sonnet-4",
    chapter: Optional[str] = None
) -> MathematicalDocument

process_multiple_documents(
    file_paths: List[str],
    chapter: str,
    model: str = "claude-sonnet-4"
) -> Dict[str, MathematicalDocument]
```

### Stage 0: Preparation

```python
split_into_sections(
    markdown_text: str,
    file_path: Optional[str] = None
) -> List[DocumentSection]

extract_jupyter_directives(
    markdown_text: str,
    section_id: str = "main"
) -> List[DirectiveHint]

format_directive_hints_for_llm(
    directives: List[DirectiveHint]
) -> str
```

### Stage 1: Extraction

```python
process_section(
    section: DocumentSection,
    prompt_template: str = MAIN_EXTRACTION_PROMPT,
    model: str = "claude-sonnet-4",
    **llm_kwargs
) -> StagingDocument

process_sections_parallel(
    sections: List[DocumentSection],
    **kwargs
) -> List[StagingDocument]

merge_sections(
    staging_docs: List[StagingDocument]
) -> StagingDocument
```

### Stage 2: Enrichment

```python
enrich_and_assemble(
    staging_doc: StagingDocument,
    model: str = "claude-sonnet-4"
) -> MathematicalDocument

# Low-level enrichment (for custom workflows)
ResolutionContext.add_staging_document(staging_doc)
ResolutionContext.resolve_theorem_reference(label_text: str) -> Optional[str]
```

### Registry Operations

```python
registry = MathematicalRegistry()

registry.add(obj: Any) -> None
registry.add_all(objects: List[Any]) -> None
registry.get(id: str) -> Optional[Any]
registry.remove(id: str) -> None

registry.query_by_tags(query: TagQuery) -> List[Any]
registry.query_relationships(source_id: str) -> List[Relationship]

save_registry_to_directory(registry, path: str)
load_registry_from_directory(path: str) -> MathematicalRegistry
```

### Graph Analysis

```python
build_relationship_graph_from_registry(
    registry: MathematicalRegistry
) -> RelationshipGraph

EquivalenceClassifier(graph).compute_equivalence_classes() -> List[EquivalenceClass]
ObjectLineage(graph, object_id: str).get_ancestors() -> Set[str]
```

---

## Design Patterns

### 1. Lean-Compatible Immutability

All models use `ConfigDict(frozen=True)` to ensure immutability:

```python
class TheoremBox(BaseModel):
    model_config = ConfigDict(frozen=True)
    # ... fields
```

This maps to Lean's structure immutability and enables:
- Safe concurrent processing
- Cacheable computation results
- Referential transparency

---

### 2. Two-Stage Pipeline

**Rationale**: Separate transcription (simple, verbose) from semantic understanding (complex, focused)

**Benefits**:
- LLM performs better on simple, well-defined tasks
- Structural validation happens in Python (reliable)
- Semantic validation happens in focused LLM calls (flexible)
- Debugging is easier (inspect intermediate outputs)

---

### 3. Property-Level Dataflow

Theorems and proofs track **properties of objects**, not just objects:

```python
# Not just: "Theorem uses object X"
input_objects: List[str]

# But: "Theorem requires properties P1, P2 of object X"
properties_required: Dict[str, List[str]]  # X → [P1, P2]
```

This enables:
- Precise dependency tracking
- Conditional theorem application
- Automated proof validation

---

### 4. Dual LaTeX/SymPy Representation

Mathematical statements are represented in both formats:

```python
class DualStatement(BaseModel):
    latex: str                         # Human-readable
    sympy_expr: str                    # Machine-parseable
    free_symbols: List[str]
    assumptions: Dict[str, str]
    natural_language: str
```

This enables:
- Human readability (LaTeX)
- Symbolic computation (SymPy)
- Automated validation
- Cross-format consistency checking

---

### 5. Recursive Proof System

Proofs are hierarchical:

```python
ProofBox
  ├─ ProofStep (DIRECT_DERIVATION)
  ├─ ProofStep (SUB_PROOF)
  │   └─ ProofBox (nested)
  │       ├─ ProofStep ...
  │       └─ ProofStep ...
  └─ ProofStep (LEMMA_APPLICATION)
```

This enables:
- Top-down proof development (sketch → expand)
- Bottom-up validation (verify leaves → root)
- Modular proof composition

---

### 6. Error Recovery and Tracking

```python
try:
    enriched = enrich_entity(raw_entity, ctx)
except EnrichmentError as e:
    # Log error with full context
    error_logger.log(e)
    # Preserve raw data for retry
    failed_entities.append((raw_entity, e))
```

This enables:
- Partial success (some entities fail, others succeed)
- Detailed error reporting
- Retry strategies
- Manual intervention where needed

---

## Examples

See `examples/` directory for complete working examples:

### Complete Workflows

**`complete_workflow.py`**
End-to-end demonstration: Create objects → Registry → Storage → Graph analysis

**`extract_then_enrich_pipeline.py`**
Document processing: Split → Extract → Enrich → Assemble

**`article_workflow_example.py`**
Multi-document article processing with chapter organization

### Specialized Workflows

**`theorem_proof_workflow.py`**
Theorem-proof integration: Create theorem → Generate proof sketch → Attach proof → Validate

**`proof_system_example.py`**
Recursive proof system: Property-level dataflow, sub-proofs, validation

**`review_workflow_example.py`**
LLM review integration: Dual review (Gemini + Codex), consensus analysis

### Integration Examples

**`registry_test.py`**
Registry operations: Add, query, tag-based search, reference resolution

**`relationship_test.py`**
Relationship system: Create relationships, build graphs, equivalence classes

**`sympy_integration_example.py`**
SymPy integration: Dual statements, symbolic validation, expression manipulation

**`graph_analysis.py`**
Graph algorithms: Dependency analysis, lineage tracking, cycle detection

---

## Running Examples

```bash
# Navigate to project root
cd /path/to/fragile

# Run complete workflow
python examples/complete_workflow.py

# Process a document
python examples/extract_then_enrich_pipeline.py \
    --file docs/source/1_euclidean_gas/01_fragile_gas_framework.md

# Batch processing
python examples/extract_then_enrich_pipeline.py \
    --directory docs/source/1_euclidean_gas \
    --pattern "*.md"
```

---

## Additional Resources

- **CLAUDE.md**: Project-level instructions for Claude Code
- **GEMINI.md**: Instructions for Gemini collaboration
- **LEAN_EMULATION_GUIDE.md**: Lean-compatible design patterns
- **PYDANTIC_VALIDATION_DEEP_DIVE.md**: Pydantic validation details

---

## Contributing

When adding new entity types or pipeline stages:

1. **Stage 1**: Add raw model to `staging_types.py`
2. **Stage 2**: Add enrichment logic to `prompts/enrichment.py`
3. **Final Model**: Add to `core/math_types.py` or `core/enriched_types.py`
4. **Registry**: Update `registry.py` to handle new type
5. **Tests**: Add to `tests/proofs/`
6. **Examples**: Add usage example to `examples/`

---

## License

See project root LICENSE file.
