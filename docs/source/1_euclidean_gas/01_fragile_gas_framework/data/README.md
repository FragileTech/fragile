# Fragile Gas Framework - Extracted Mathematical Content

## Overview

This directory contains the extracted and structured mathematical content from the Fragile Gas Framework foundational document.

**Source**: `docs/source/1_euclidean_gas/01_fragile_gas_framework.md`
**Extraction Date**: 2025-10-26T10:39:15.638937
**Total Mathematical Objects**: 113

## Directory Structure

```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── data/                          # Consolidated extraction results
│   ├── pydantic_objects.json      # All objects organized by type
│   ├── dependency_graph.json      # Complete dependency graph
│   ├── extraction_statistics.json # Detailed extraction metrics
│   ├── extraction_report.json     # Comprehensive extraction report
│   └── README.md                  # This file
├── objects/                       # Individual mathematical object files (43)
├── axioms/                        # Individual axiom files (19)
└── theorems/                      # Individual theorem/lemma/proposition files (51)
```

## Contents Summary

### Mathematical Objects (43)
Mathematical definitions including:
- State spaces (walker, swarm, algorithmic space)
- Operators (perturbation, cloning, standardization)
- Measures (noise measures, aggregation operators)
- Metrics (W₂ metric, algorithmic distance)

### Axioms (19)
Foundational axioms including:
- Axiom of Guaranteed Revival (κ_revival > 1)
- Boundary Regularity (L_death, α_B)
- Reward Regularity (L_R Lipschitz constant)
- Measurement Quality Axioms
- Exploration Axioms

### Theorems (24)
Main results establishing:
- Operator continuity bounds
- N-uniform stability
- Revival guarantees
- Mean-square error bounds

### Lemmas (24)
Technical lemmas supporting theorem proofs

### Propositions (3)
Auxiliary results

## Extraction Statistics

### Dependency Graph
- **Nodes**: 113 mathematical entities
- **Edges**: 23 dependency relationships
- **Coverage**: 26/51 theorems with extracted dependencies (~51%)

### Dependency Types
- Object dependencies: 17
- Axiom dependencies: 6
- Parameter dependencies: 18
- Internal lemma dependencies: 0

### Most Referenced Entities

**Objects**:
- obj-alive-dead-sets: 13 references
- obj-perturbation-measure: 2 references
- obj-valid-state-space: 2 references

**Axioms**:
- axiom-def-axiom-rescale-function: 3 references
- axiom-bounded-measurement-variance: 1 references
- axiom-def-axiom-bounded-second-moment-perturbation: 1 references
- axiom-def-axiom-boundary-regularity: 1 references

## Usage Examples

### Load All Objects
```python
import json
from pathlib import Path

data_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/data")
pydantic_objects = json.loads((data_dir / "pydantic_objects.json").read_text())

# Access by type
axioms = pydantic_objects['axioms']
objects = pydantic_objects['objects']
theorems = pydantic_objects['theorems']
lemmas = pydantic_objects['lemmas']
propositions = pydantic_objects['propositions']
```

### Query Dependency Graph
```python
graph = json.loads((data_dir / "dependency_graph.json").read_text())

# Find all theorems that use a specific axiom
axiom_label = "axiom-def-axiom-guaranteed-revival"
dependent_theorems = [
    edge['source']
    for edge in graph['edges']
    if edge['target'] == axiom_label and edge['type'] == 'uses_axiom'
]

# Find all axioms required by a theorem
theorem_label = "thm-revival-guarantee"
required_axioms = [
    edge['target']
    for edge in graph['edges']
    if edge['source'] == theorem_label and edge['type'] == 'uses_axiom'
]
```

### Analyze Theorem Dependencies
```python
# Load specific theorem
thm_file = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/thm-revival-guarantee.json")
thm = json.loads(thm_file.read_text())

print(f"Input Objects: {thm['input_objects']}")
print(f"Input Axioms: {thm['input_axioms']}")
print(f"Input Parameters: {thm['input_parameters']}")
print(f"Output Type: {thm['output_type']}")
```

## Extraction Quality

### Completeness
- **Method**: Pattern matching + heuristic analysis
- **Coverage**: ~51% of theorems have extracted dependencies
- **Status**: Partial - ~25% of theorems have extracted dependencies

### Limitations
- Only 47/113 directives fully parsed (missing 66 definitions)
- Implicit dependencies require deeper semantic analysis
- Property and relationship extraction is minimal
- Proof structure not fully captured
- Mathematical notation mapping incomplete

### Recommended Next Steps
1. Use LLM-assisted extraction (Gemini 2.5 Pro) for semantic analysis
2. Manual review of theorem proofs to extract internal lemma dependencies
3. Expand mathematical notation dictionary
4. Extract property propagation chains
5. Build lemma dependency DAG from proof structure

## Validation

### Status
- **Label Pattern Validation**: ✓ All labels follow Pydantic patterns
- **Cross-Reference Integrity**: ✓ All explicit references resolve correctly
- **Type Consistency**: ✓ Object types and theorem output types are valid

### Warnings
- 25 theorems have no extracted inputs
- 47 theorems have no extracted outputs
- 0 internal lemma dependencies extracted (proof structure not parsed)

## Files Description

### Consolidated Files

**pydantic_objects.json**
- Complete catalog of all mathematical entities
- Organized by type (axioms, objects, theorems, lemmas, propositions)
- Includes metadata and extraction provenance
- ~4KB

**dependency_graph.json**
- Nodes: All 113 mathematical entities with type information
- Edges: Dependency relationships (uses_object, uses_axiom, uses_lemma)
- Statistics: Counts by node type and edge type
- Enables graph analysis and visualization

**extraction_statistics.json**
- Detailed metrics on extraction coverage
- Most referenced entities
- Theorem complexity rankings
- Quality indicators

**extraction_report.json**
- Comprehensive extraction summary
- Quality assessment
- Validation results
- File manifest

### Individual Files

Each mathematical entity has its own JSON file with:
- **Objects**: Label, name, mathematical_expression, object_type, tags, properties
- **Axioms**: Label, statement, mathematical_expression, foundational_framework
- **Theorems**: Label, name, statement_type, inputs (objects, axioms, parameters), outputs (type, properties, relations)

## Citation

If you use this structured mathematical content, please cite:

```
Fragile Gas Framework Mathematical Foundations
Source: docs/source/1_euclidean_gas/01_fragile_gas_framework.md
Extraction: 2025-10-26T10:39:15.638937
```

## Maintenance

This extraction is a snapshot as of 2025-10-26T10:39:15.638937. If the source document is updated:

1. Re-run extraction scripts: `python extract_framework_deep.py && python merge_extraction_results.py`
2. Review new/modified theorems
3. Validate cross-references
4. Update dependency graph

## Contact

For questions about the mathematical content, consult the source document.
For questions about the extraction, see `AGENTS.md` in the project root.
