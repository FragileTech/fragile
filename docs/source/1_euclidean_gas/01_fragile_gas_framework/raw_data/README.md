# Raw Data: Extracted Mathematical Entities

This directory contains all mathematical entities extracted from `01_fragile_gas_framework.md` using the document-parser agent.

## Structure

Each subdirectory contains JSON files for a specific entity type:

```
raw_data/
├── axioms/          - Axiomatic requirements and assumptions
├── citations/       - References to external documents
├── corollaries/     - Direct consequences of theorems
├── definitions/     - Core mathematical definitions
├── equations/       - Standalone equations
├── lemmas/          - Supporting results and intermediate steps
├── objects/         - Mathematical objects, operators, constants
├── parameters/      - Framework parameters
├── proofs/          - Complete proof contents
├── propositions/    - Properties and auxiliary results
├── remarks/         - Pedagogical notes and explanations
└── theorems/        - Main mathematical results
```

## Entity Counts

- **Axioms**: 21 files
- **Citations**: 1 file
- **Corollaries**: 3 files
- **Definitions**: 31 files
- **Equations**: 0 files
- **Lemmas**: 45 files
- **Objects**: 39 files
- **Parameters**: 8 files
- **Proofs**: 15 files
- **Propositions**: 3 files
- **Remarks**: 9 files
- **Theorems**: 30 files
- **TOTAL**: 205 entities

## File Naming Conventions

### Labeled Entities
Files with explicit labels from the source document:
- `axiom-<label>.json` - Axioms
- `def-<label>.json` - Definitions
- `thm-<label>.json` - Theorems
- `lem-<label>.json` - Lemmas
- `prop-<label>.json` - Propositions
- `cor-<label>.json` - Corollaries
- `obj-<label>.json` - Mathematical objects
- `param-<label>.json` - Parameters
- `rem-<label>.json` - Remarks

### Unlabeled Entities
Files extracted from unlabeled sections:
- `raw-<type>-NNN.json` - Sequential numbering for unlabeled entities
- `unlabeled-<type>-NNN.json` - Explicitly marked as unlabeled

### Sub-entities
Supporting results within larger proofs:
- `sub-lem-<label>.json` - Sub-lemmas
- `proof-<parent-label>.json` - Proofs of parent theorems

## JSON Schema

Each entity file follows the staging schema defined in `src/fragile/proofs/staging_types.py`.

Basic structure:
```json
{
  "label": "entity-label",
  "type": "axiom|definition|theorem|...",
  "name": "Human-readable name",
  "content": "LaTeX/markdown mathematical content",
  "section": "Section number",
  "source_file": "Source document",
  "source_lines": "Line range",
  "tags": ["tag1", "tag2"],
  "dependencies": ["other-label-1", "other-label-2"]
}
```

## Processing Stages

This is **Stage 1: Raw Extraction** output.

### Stage 1: Raw Extraction (✅ Complete)
- Verbatim transcription from source document
- Basic entity type identification
- Line number tracking
- Minimal validation

### Stage 2: Document Refinement (⏳ Pending)
Agent: `document-refiner`
- Semantic enrichment
- Pydantic schema validation
- Type inference
- Property extraction
- Cross-reference resolution

### Stage 3: Cross-Referencing (⏳ Pending)
Agent: `cross-referencer`
- Dependency graph construction
- Relationship validation
- Input/output typing
- Circular dependency detection

### Stage 4: Proof System Integration (⏳ Pending)
- Registry loading
- Theorem-proof mapping
- Validation against framework
- Export to proof system

## Usage

### Loading Entities

```python
from pathlib import Path
import json

def load_entity(entity_file: Path):
    with open(entity_file) as f:
        return json.load(f)

# Load all theorems
theorems_dir = Path("raw_data/theorems")
theorems = [load_entity(f) for f in theorems_dir.glob("*.json")]
```

### Validation

```python
from fragile.proofs.staging_types import RawEntity

def validate_entity(entity_file: Path):
    with open(entity_file) as f:
        data = json.load(f)
    return RawEntity(**data)  # Pydantic validation
```

### Statistics

For extraction statistics and reports, see the `../statistics/` directory.

## Source Document

- **File**: `docs/source/1_euclidean_gas/01_fragile_gas_framework.md`
- **Sections**: 22 sections (0-21)
- **Lines**: ~5,300 lines
- **Extraction Date**: 2025-10-27

## Tools

Extraction and consolidation performed using:
- `document-parser` agent (parallel execution, 22 agents)
- `src/fragile/proofs/tools/consolidate_raw_data.py`
- `src/fragile/proofs/tools/consolidate_extraction.py`

For consolidation details, see `../CONSOLIDATION_SUMMARY.md`.

## Notes

- All mathematical content preserved verbatim
- LaTeX formatting maintained
- MyST directive labels preserved
- Proof content extracted when present
- Unlabeled entities receive temporary IDs
- Duplicate handling: renamed with `_dupN` suffix
- No semantic interpretation at this stage

## Next Steps

To continue processing:

1. Run document-refiner agent on each entity
2. Run cross-referencer to build dependency graph
3. Load into proof system registry
4. Validate theorem-proof consistency
5. Generate dependency visualizations

---

**Extraction Method**: Parallel document-parser agents
**Status**: Stage 1 Complete
**Last Updated**: 2025-10-27
