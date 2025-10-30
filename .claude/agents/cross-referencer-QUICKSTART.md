# Cross-Referencer Agent - Quick Start Guide

## Overview

The **Cross-Referencer** agent discovers and links relationships between mathematical objects extracted by the document-parser.

**Stage**: 1.5 (between document-parser and document-refiner)
**Input**: Document-parser output (JSON files)
**Output**: Enhanced JSONs + relationships/ directory

---

## Quick Invocation

### Method 1: Direct Task Tool (Parallel-Ready)

```
@agent-cross-referencer analyze docs/source/1_euclidean_gas/01_fragile_gas_framework
```

### Method 2: Manual Loading

```
Load the cross-referencer agent from .claude/agents/cross-referencer.md

Analyze: docs/source/1_euclidean_gas/01_fragile_gas_framework
```

### Method 3: Python CLI

```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework
```

---

## Modes

### Fast Mode (Explicit Refs Only)
```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework \
  --no-llm
```
**Time**: ~2-3 seconds
**Output**: Only explicit {prf:ref} relationships

### Full Mode (With LLM)
```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework
```
**Time**: ~5-10 minutes (48 theorems)
**Output**: Explicit + implicit relationships

### High-Confidence Mode (Dual AI)
```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework \
  --dual-ai
```
**Time**: ~10-15 minutes
**Output**: Cross-validated relationships (Gemini + Codex)

---

## Parallel Processing

Process multiple documents simultaneously:

```
Launch 3 cross-referencers in parallel:

1. @agent-cross-referencer analyze docs/source/1_euclidean_gas/01_fragile_gas_framework
2. @agent-cross-referencer analyze docs/source/1_euclidean_gas/02_euclidean_gas
3. @agent-cross-referencer analyze docs/source/1_euclidean_gas/03_cloning
```

---

## What It Does

✅ Processes explicit cross-references from {prf:ref} tags
✅ Discovers implicit dependencies via LLM analysis
✅ Traces mathematical symbols to definitions
✅ Identifies assumed axioms and parameters
✅ Fills input_objects, input_axioms, input_parameters
✅ Constructs typed Relationship objects
✅ Validates all relationships against registry
✅ Generates comprehensive reports

---

## Output Structure

```
docs/source/1_euclidean_gas/01_fragile_gas_framework/
├── data/                               # (Input - from document-parser)
│   └── extraction_inventory.json
├── objects/                            # (Input)
│   └── *.json
├── theorems/                           # (Modified - filled dependencies)
│   └── *.json
├── axioms/                             # (Input)
│   └── *.json
└── relationships/                      # (NEW - created by cross-referencer)
    ├── rel-*.json                      # Individual relationships
    ├── index.json                      # Summary statistics
    └── REPORT.md                       # Human-readable report
```

---

## Example Output

### Enhanced Theorem JSON (Modified)
```json
{
  "label": "thm-standardization-structural-error",
  "name": "Bounding the Expected Squared Structural Error",
  "input_objects": [
    "obj-swarm",
    "obj-structural-error",
    "obj-alive-set"
  ],
  "input_axioms": [
    "axiom-bounded-domain"
  ],
  "input_parameters": [
    "n_c",
    "C_S_direct",
    "C_S_indirect"
  ],
  "attributes_required": {
    "obj-swarm": ["attr-finite", "attr-partitioned"]
  }
}
```

### Relationship JSON (New)
```json
{
  "label": "rel-discrete-swarm-continuous-measure-approximation",
  "relationship_type": "APPROXIMATION",
  "bidirectional": false,
  "source_object": "obj-discrete-swarm",
  "target_object": "obj-continuous-measure",
  "established_by": "thm-mean-field-convergence",
  "expression": "Discrete swarm approximates continuous with O(N^{-1/d}) error",
  "attributes": [
    {
      "label": "error-rate",
      "expression": "O(N^{-1/d})",
      "description": "Approximation error rate"
    }
  ],
  "tags": ["mean-field", "discrete-continuous"]
}
```

### REPORT.md (New)
```markdown
# Cross-Reference Analysis Report

**Generated**: 2025-10-27T20:00:00

## Statistics

- **Theorems Processed**: 48
- **Explicit Refs**: 12
- **Implicit Deps Discovered**: 156
- **Relationships Created**: 142
- **Input Objects Filled**: 103
- **Input Axioms Filled**: 42
- **Validation Errors**: 0

## Relationships by Type

- **OTHER**: 106
- **APPROXIMATION**: 23
- **EMBEDDING**: 8
- **EQUIVALENCE**: 5
```

---

## Prerequisites

**Required**: Document must be processed by document-parser first
**Input Files**: data/, objects/, theorems/, axioms/
**Python Module**: `fragile.agents.cross_reference_analyzer`

---

## Typical Workflow

```
# Step 1: Parse document
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/01_fragile_gas_framework.md

# Step 2: Analyze relationships
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/01_fragile_gas_framework

# Step 3: Inspect results
cat docs/source/1_euclidean_gas/01_fragile_gas_framework/relationships/REPORT.md
```

---

## Relationship Types

| Type | Direction | Example |
|------|-----------|---------|
| EQUIVALENCE | ↔️ | discrete ≡ continuous |
| EMBEDDING | → | particles ↪ fluid |
| APPROXIMATION | → | discrete ≈ continuous + O(ε) |
| REDUCTION | → | PDE → ODE |
| EXTENSION | → | Adaptive extends Euclidean |
| GENERALIZATION | → | Wasserstein generalizes L² |
| SPECIALIZATION | → | Gaussian ⊂ sub-Gaussian |
| OTHER | → | Generic "uses" |

---

## Troubleshooting

### No relationships created
**Cause**: No cross-refs in document
**Solution**: Use --no-llm to verify, check extraction_inventory.json

### LLM hallucinates labels
**Cause**: Insufficient framework context
**Solution**: Provide --glossary docs/glossary.md

### Validation errors
**Cause**: Referenced objects not in registry
**Solution**: Check document-parser output completeness

---

## Integration

```
document-parser → cross-referencer → document-refiner
     ↓                    ↓                  ↓
  [empty]         [relationships]      [validated]
```

The cross-referencer fills the gap between extraction and refinement, providing essential dependency information for downstream processing.

---

**Full Documentation**: `.claude/agents/cross-referencer.md`
**Status**: ✅ Ready for production use
