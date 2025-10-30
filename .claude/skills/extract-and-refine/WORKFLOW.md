# Extract-and-Refine - Complete Workflow

## Prerequisites

- ✅ MyST markdown documents with Jupyter Book directives
- ✅ Python environment with fragile package
- ✅ Gemini 2.5 Pro API key (optional, for LLM features)

---

## Stage 1: Document Parsing

### Step 1.1: Verify Document Format

**Purpose**: Ensure document has valid MyST directives
**Time**: 30 seconds

```bash
# Check for valid directives
grep -n "^:::{prf:" docs/source/1_euclidean_gas/03_cloning.md | head -10
```

**Expected output**:
```
125::::{prf:definition} Single Swarm Space
150::::{prf:theorem} Keystone Principle
```

**Verify**: Exactly 3 colons (`:::`) at start of lines

---

### Step 1.2: Run Parser (Fast Mode)

**Purpose**: Extract entities without LLM overhead
**Time**: ~15 seconds

```bash
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/03_cloning.md \
  --no-llm
```

**Expected console output**:
```
🚀 MathDocumentParser Starting
📄 Processing: 03_cloning.md
  Phase 1: Extracting MyST directives...
    ✓ Found 119 directives
  Phase 2: Creating mathematical objects...
    ✓ Created 36 objects
  Phase 3: Creating theorems...
    ✓ Created 59 theorems
  ...
✅ Processing complete!
```

**Output files**:
```
docs/source/1_euclidean_gas/03_cloning/
├── raw_data/
│   ├── objects/
│   │   └── obj-*.json (36 files)
│   ├── axioms/
│   │   └── axiom-*.json (6 files)
│   └── theorems/
│       └── thm-*.json, lem-*.json (59 files)
└── data/
    ├── extraction_inventory.json
    └── statistics.json
```

---

### Step 1.3: Validate Extraction

**Purpose**: Check for errors before proceeding
**Time**: 10 seconds

```bash
# Check statistics
cat docs/source/1_euclidean_gas/03_cloning/data/statistics.json
```

**Expected**:
```json
{
  "objects_created": 36,
  "theorems_created": 59,
  "proofs_created": 0,
  "relationships_created": 0,
  "validation_errors": 0,    # ← Should be 0
  "validation_warnings": 0
}
```

**If validation_errors > 0**:
- Check `validation_errors.json` for details
- Fix issues in source markdown
- Re-run parser

---

## Stage 1.5: Cross-Referencing

### Step 2.1: Run Cross-Referencer (Fast Mode)

**Purpose**: Link explicit cross-references without LLM
**Time**: ~2-3 seconds

```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/03_cloning \
  --no-llm
```

**Expected output**:
```
📊 Cross-Reference Analysis
  Source: docs/source/1_euclidean_gas/03_cloning
  Mode: explicit only (no LLM)

✓ Processed 12 explicit references
✓ Created 12 relationships
✓ Updated 59 theorems with dependencies

Report: docs/source/1_euclidean_gas/03_cloning/relationships/REPORT.md
```

**Output files**:
```
docs/source/1_euclidean_gas/03_cloning/
├── relationships/           # NEW
│   ├── rel-*.json (12 files)
│   ├── index.json
│   └── REPORT.md
└── theorems/                # MODIFIED
    └── thm-*.json (now with input_objects filled)
```

---

### Step 2.2: (Optional) Enable LLM for Implicit Dependencies

**Purpose**: Discover implicit dependencies via AI analysis
**Time**: ~5-10 minutes

```bash
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/03_cloning
  # (no --no-llm flag)
```

**What it does**:
- Analyzes theorem statements with Gemini 2.5 Pro
- Traces mathematical symbols to definitions
- Infers required axioms and parameters
- Fills `input_axioms`, `input_parameters` fields

**Expected output**:
```
✓ Processed 12 explicit references
✓ Discovered 156 implicit dependencies
✓ Created 142 relationships
```

---

### Step 2.3: Review Cross-Reference Report

**Purpose**: Verify relationship quality
**Time**: 2 minutes

```bash
cat docs/source/1_euclidean_gas/03_cloning/relationships/REPORT.md
```

**Expected content**:
```markdown
# Cross-Reference Analysis Report

## Statistics
- Theorems Processed: 59
- Explicit Refs: 12
- Implicit Deps: 156
- Relationships Created: 142
- Validation Errors: 0

## Relationships by Type
- OTHER: 106
- APPROXIMATION: 23
- EMBEDDING: 8
- EQUIVALENCE: 5
```

**Spot-check a relationship**:
```bash
cat docs/source/1_euclidean_gas/03_cloning/relationships/rel-*.json | head -1 | python -m json.tool
```

---

## Stage 2: Refinement & Validation

### Step 3.1: Run Document Refiner

**Purpose**: Validate entities against strict schemas and enrich
**Time**: ~10-30 minutes (depends on entity count)

**In Claude Code**:
```
Load the document-refiner agent from .claude/agents/document-refiner.md

Refine: docs/source/1_euclidean_gas/03_cloning
```

**What the agent does**:
1. Loads all entities from `raw_data/`
2. Validates against Pydantic schemas
3. Enriches with semantic information via Gemini
4. Fills missing required fields
5. Exports to `refined_data/` with validation reports

**Expected console output** (from agent):
```
🔧 Document Refiner Starting
  Source: docs/source/1_euclidean_gas/03_cloning

Phase 1: Loading raw entities...
  ✓ Loaded 36 objects
  ✓ Loaded 6 axioms
  ✓ Loaded 59 theorems

Phase 2: Validating against schemas...
  ⚠️  3 validation warnings (non-critical)
  ✓ 0 validation errors

Phase 3: Enriching entities...
  [Gemini 2.5 Pro analysis...]
  ✓ Enriched 95 entities

Phase 4: Exporting refined data...
  ✓ Exported to refined_data/

✅ Refinement complete!
```

---

### Step 3.2: Verify Refined Output

**Purpose**: Check quality of refined entities
**Time**: 2 minutes

```bash
# Check statistics
cat docs/source/1_euclidean_gas/03_cloning/reports/statistics/refined_statistics.json
```

**Expected**:
```json
{
  "objects_refined": 36,
  "axioms_refined": 6,
  "theorems_refined": 59,
  "parameters_refined": 2,
  "validation_errors": 0,    # ← Should be 0
  "warnings": 3              # ← Acceptable
}
```

**List refined files**:
```bash
find docs/source/1_euclidean_gas/03_cloning/refined_data -name "*.json" | wc -l
# Expected: 103 files (36 + 6 + 59 + 2)
```

**Spot-check a refined object**:
```bash
cat docs/source/1_euclidean_gas/03_cloning/refined_data/objects/obj-euclidean-gas.json | python -m json.tool
```

**Expected structure**:
```json
{
  "label": "obj-euclidean-gas",
  "name": "Euclidean Gas",
  "object_type": "SPACE",
  "description": "...",
  "mathematical_expression": "...",
  "source": {
    "document_id": "03_cloning",
    "file_path": "docs/source/1_euclidean_gas/03_cloning.md",
    "chapter": "1_euclidean_gas",
    "line_start": 125,
    "line_end": 145
  },
  "tags": ["langevin", "cloning", "space"],
  "properties_added": [...]
}
```

---

## Final Verification

### Checklist

- [ ] `raw_data/` directory exists with JSON files
- [ ] `data/statistics.json` shows `validation_errors: 0`
- [ ] `relationships/` directory exists with REPORT.md
- [ ] `relationships/REPORT.md` shows expected relationship counts
- [ ] `refined_data/` directory exists with JSON files
- [ ] `reports/statistics/refined_statistics.json` shows `validation_errors: 0`
- [ ] Spot-checked refined entities have complete `source` fields
- [ ] All labels follow kebab-case convention

---

## Integration Points

### To Registry Building

After refinement, use the automated registry pipeline:

```bash
# Build all registries automatically
python -m fragile.proofs.tools.build_all_registries --docs-root docs/source
```

The automated pipeline:
- Discovers all documents with `refined_data/` (including 03_cloning)
- Transforms to pipeline format automatically
- Builds per-document and combined registries

See [../registry-management/WORKFLOW.md](../registry-management/WORKFLOW.md) for registry details.

### To Proof Development

After extraction, theorems available for sketching:

```bash
# List available theorems
ls docs/source/1_euclidean_gas/03_cloning/refined_data/theorems/

# Use proof-sketcher on specific theorem
# (See ../proof-validation/WORKFLOW.md)
```

---

## Time Summary

| Stage | Fast Mode | With LLM | Notes |
|-------|-----------|----------|-------|
| Parsing | ~15 sec | ~3-5 min | Use fast for iteration |
| Cross-ref | ~3 sec | ~5-10 min | Fast finds explicit refs only |
| Refinement | N/A | ~10-30 min | Always uses LLM |
| **Total** | **~20 sec** | **~20-45 min** | Fast for validation, LLM for production |

---

**Next**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
