# Relationship System Examples

This directory contains comprehensive examples demonstrating the Relationship System functionality.

---

## Quick Start

**Recommended**: Start with `complete_workflow.py` to see the entire system in action.

```bash
python examples/complete_workflow.py
```

---

## Examples Overview

### 1. `relationship_test.py` - Basic Relationships

**What it demonstrates**:
- Creating mathematical objects
- Creating relationships (Equivalence, Embedding)
- Relationship properties (error bounds)
- Type-dependent directionality
- ID validation

**Key concepts**:
- `MathematicalObject`
- `Relationship` with 8 types
- `RelationshipProperty`
- `TheoremBox`

**Run it**:
```bash
python examples/relationship_test.py
```

**Output**: ✅ Relationship model validation

---

### 2. `registry_test.py` - Registry Operations

**What it demonstrates**:
- Adding objects to registry
- Querying by tags (any/all/none)
- Querying by type
- Querying relationships
- Referential integrity validation
- Registry statistics

**Key concepts**:
- `MathematicalRegistry`
- `TagQuery` (simple queries)
- `CombinedTagQuery` (advanced queries)
- Relationship queries

**Run it**:
```bash
python examples/registry_test.py
```

**Output**: ✅ 6 objects, 11 tags, referential integrity valid

---

### 3. `storage_example.py` - Storage Layer

**What it demonstrates**:
- Saving registry to JSON files
- Directory structure organization
- Loading registry from storage
- Verifying loaded objects
- Statistics comparison

**Key concepts**:
- `save_registry_to_directory()`
- `load_registry_from_directory()`
- Directory structure: `objects/`, `relationships/`, `theorems/`
- Index file with metadata

**Run it**:
```bash
python examples/storage_example.py
```

**Output**: ✅ 7 JSON files created, statistics match perfectly

---

### 4. `graph_analysis.py` - Graph Algorithms

**What it demonstrates**:
- Building relationship graph
- Graph connectivity (BFS)
- Path finding (shortest path)
- Object lineage tracing (DFS)
- Equivalence class computation (Union-Find)
- Framework flow analysis
- Theorem dependencies

**Key concepts**:
- `RelationshipGraph`
- `ObjectLineage`
- `EquivalenceClassifier`
- `FrameworkFlow`
- Graph algorithms (BFS, DFS, Union-Find)

**Run it**:
```bash
python examples/graph_analysis.py
```

**Output**: ✅ 5 nodes, 4 edges, 3 equivalence classes, 2 framework layers

---

### 5. `complete_workflow.py` - End-to-End Integration ⭐

**What it demonstrates**:
- Complete workflow from creation to analysis
- Tag-based organization
- Saving to JSON storage
- Loading from storage
- Building relationship graph
- Computing equivalence classes
- Tracing object lineage
- Analyzing framework flow
- Generating statistics

**Key concepts**:
- All features integrated
- Recommended workflow
- Production-ready example

**Run it**:
```bash
python examples/complete_workflow.py
```

**Output**: ✅ Complete framework with 5 objects, 4 relationships, 4 theorems

---

### 6. `complete_integration_example.py` - Full Integration ⭐⭐⭐

**What it demonstrates**:
- TheoremBox + ProofBox + Relationship System integration
- Automatic proof input/output creation from theorem
- Proof validation against theorem claims
- Relationship extraction from proofs
- Registry and graph integration
- ProofEngine management
- Complete end-to-end workflow

**Key concepts**:
- `validate_proof_for_theorem()` - Validate proof matches theorem
- `create_proof_inputs_from_theorem()` - Auto-generate proof inputs
- `create_proof_outputs_from_theorem()` - Auto-generate proof outputs
- `extract_relationships_from_proof()` - Extract relationships
- `get_proof_statistics()` - Analyze proof structure
- Complete integration workflow

**Run it**:
```bash
python examples/complete_integration_example.py
```

**Output**: ✅ Complete workflow from theorem → proof → validation → integration

**Workflow summary**:
```
1. Create mathematical objects with properties
2. Define theorem with property requirements
3. Write compositional proof using integration helpers
4. Validate proof against theorem (automatic checking)
5. Extract relationships from proof
6. Add to registry and build relationship graph
7. Query and analyze the integrated framework
```

---

### 7. `proof_system_example.py` - Compositional Proofs ⭐

**What it demonstrates**:
- Property-level granularity (object has 10 properties, proof uses 2)
- Hierarchical/recursive proof architecture (ProofBox contains sub-proofs)
- Three expansion modes (DirectDerivation, SubProof, LemmaApplication)
- Dataflow validation (properties flow correctly through steps)
- Integration with relationship system
- Graph representation for visualization
- ProofEngine for managing proof expansion

**Key concepts**:
- `PropertyReference` - Reference to specific property of an object
- `ProofInput`/`ProofOutput` - Explicit input/output signatures with properties
- `ProofStep` - Atomic transformation (3 types)
- `ProofBox` - Recursive proof container
- `ProofEngine` - Orchestration layer
- LLM proving pipeline integration

**Run it**:
```bash
python examples/proof_system_example.py
```

**Output**: ✅ Mean Field Limit proof with 3 steps, 1 sub-proof, dataflow validation

**Example proof structure**:
```
📦 Mean Field Limit (proof-thm-mean-field-limit)
   Steps: 3
   ✓ Step 1: sub_proof
    📦 PDE Well-Posedness (proof-pde-wellposedness)
       Steps: 2
       ○ Step 1: lemma_application
       ✓ Step 2: direct_derivation
   ○ Step 2: lemma_application
   ✓ Step 3: direct_derivation
```

---

## Learning Path

### Beginner
1. **START HERE**: `complete_integration_example.py` - See the full integrated system
2. Then `complete_workflow.py` to understand the relationship system
3. Read through `relationship_test.py` to understand basic relationships
4. Explore `registry_test.py` to learn querying

### Intermediate
1. Study `proof_system_example.py` for compositional proofs
2. Study `storage_example.py` for persistence
3. Explore `graph_analysis.py` for graph algorithms
4. Modify examples to create your own framework

### Advanced
1. Read the source code in `src/fragile/proof_integration.py`
2. Read the source code in `src/fragile/proof_system.py`
3. Understand property-level dataflow and recursive proof expansion
4. Read the source code in `src/fragile/relationship_graphs.py`
5. Implement custom proof strategies and validation rules
6. Extend the system with new features (LLM integration, Lean export, etc.)

---

## Common Patterns

### Creating Objects
```python
from fragile.pipeline_types import MathematicalObject, ObjectType

obj = MathematicalObject(
    label="obj-my-object",
    name="My Mathematical Object",
    mathematical_expression="x ∈ ℝ^d",
    object_type=ObjectType.SET,
    tags=["my-framework", "discrete"]
)
```

### Creating Relationships
```python
from fragile.pipeline_types import Relationship, RelationType

rel = Relationship(
    label="rel-source-target-equivalence",
    relationship_type=RelationType.EQUIVALENCE,
    source_object="obj-source",
    target_object="obj-target",
    bidirectional=True,
    established_by="thm-my-theorem",
    expression="A ≡ B"
)
```

### Querying
```python
from fragile.reference_system import TagQuery, CombinedTagQuery

# Simple query
query = TagQuery(tags=["discrete"], mode="all")
result = registry.query_by_tag(query)

# Combined query
query = CombinedTagQuery(
    must_have=["core"],
    any_of=["euclidean-gas", "adaptive-gas"],
    must_not_have=["deprecated"]
)
result = registry.query_by_tags(query)
```

### Storage
```python
from fragile.storage import save_registry_to_directory, load_registry_from_directory

# Save
save_registry_to_directory(registry, "data/objects/")

# Load
registry = load_registry_from_directory(MathematicalRegistry, "data/objects/")
```

### Graph Analysis
```python
from fragile.relationship_graphs import (
    build_relationship_graph_from_registry,
    EquivalenceClassifier,
    ObjectLineage
)

# Build graph
graph = build_relationship_graph_from_registry(registry)

# Find path
path = graph.find_path("obj-source", "obj-target")

# Equivalence classes
classifier = EquivalenceClassifier(graph)
eq_classes = classifier.compute_equivalence_classes()

# Lineage
lineage = ObjectLineage(graph)
descendants = lineage.get_descendants("obj-source")
```

### Compositional Proofs
```python
from fragile.proof_system import (
    PropertyReference,
    ProofInput,
    ProofOutput,
    ProofStep,
    ProofBox,
    ProofEngine,
    DirectDerivation,
    SubProofReference,
    LemmaApplication,
    ProofStepType,
    ProofStepStatus
)

# Define property-level inputs
prop_lipschitz = PropertyReference(
    object_id="obj-discrete-system",
    property_id="prop-lipschitz",
    property_statement="U is Lipschitz: |∇U(x)| ≤ L_U"
)

# Create proof inputs (specify exact properties needed)
proof_inputs = [
    ProofInput(
        object_id="obj-discrete-system",
        required_properties=[prop_lipschitz],
        required_assumptions=[]
    )
]

# Create proof step with explicit dataflow
step = ProofStep(
    step_id="step-1",
    description="Establish well-posedness",
    inputs=proof_inputs,
    outputs=[ProofOutput(...)],
    step_type=ProofStepType.DIRECT_DERIVATION,
    derivation=DirectDerivation(
        mathematical_content="$$...$$",
        techniques=["gronwall", "contraction"]
    ),
    status=ProofStepStatus.EXPANDED
)

# Create proof box (can contain sub-proofs recursively)
proof = ProofBox(
    proof_id="proof-thm-mean-field",
    label="Mean Field Limit",
    proves="thm-mean-field-limit",
    inputs=proof_inputs,
    outputs=[...],
    strategy="Three-step strategy: ...",
    steps=[step],
    sub_proofs={}
)

# Validate dataflow
errors = proof.validate_dataflow()
if not errors:
    print("✓ Dataflow valid")

# Use ProofEngine to manage expansion
engine = ProofEngine()
engine.register_proof(proof)
requests = engine.get_expansion_requests(proof.proof_id)
```

---

## Expected Output

All examples produce detailed output showing:
- ✅ Success indicators
- Object counts and statistics
- Query results
- Graph analysis results
- Framework insights

Example output from `complete_workflow.py`:
```
✓ Created framework: 5 objects, 4 relationships, 4 theorems
✓ Saved to JSON: 14 files
✓ Loaded from storage
✓ Built relationship graph: 5 nodes, 4 edges
✓ Computed equivalence classes: discrete ≡ continuous
✓ Traced object lineage: 4 descendants
✓ Analyzed framework flow: 2 dependency layers
✓ Referential integrity: 100% valid
```

---

## Storage Locations

Examples create temporary storage directories:
- `examples/test_storage/` (from `storage_example.py`)
- `examples/complete_workflow_storage/` (from `complete_workflow.py`)

These can be safely deleted after running examples.

---

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/fragile
python examples/complete_workflow.py
```

### Validation Errors
Check ID naming conventions:
- Use `obj-` prefix for objects
- Use `rel-source-target-type` format for relationships
- Use lowercase kebab-case

### Missing References
Run referential integrity check:
```python
missing = registry.validate_referential_integrity()
if missing:
    print(f"Missing references: {missing}")
```

---

## Further Reading

- **`../RELATIONSHIP_SYSTEM_GUIDE.md`** - Complete usage guide
- **`../ID_LABELING_SYSTEM.md`** - ID naming conventions
- **`../RELATIONSHIP_SYSTEM_COMPLETE.md`** - Implementation summary
- **`../src/fragile/proof_system.py`** - Compositional proof system source (700+ lines)
- **`../math_schema.json`** - Mathematical proof schema reference

---

## Contributing

When adding new examples:
1. Follow the existing example structure
2. Add comprehensive comments
3. Include expected output in docstring
4. Test before committing
5. Update this README

---

**Happy exploring!** 🚀
