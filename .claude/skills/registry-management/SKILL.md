---
name: registry-management
description: Build and manage mathematical entity registries with document-agnostic automation. Use for building per-document and combined registries, automatic discovery of all documents, transforming data formats, or preparing data for dashboard visualization.
---

# Registry Management - Complete Workflow Guide

This directory contains tools for building `MathematicalRegistry` instances with fully automated document discovery.

## Document-Agnostic Architecture

**Key Principle**: Zero hardcoded document names. Add new document → Create refined_data → Run build_all → Everything else happens automatically!

### Automatic Discovery

**Chapter Pattern**: `^\d+_\w+$` (e.g., `1_euclidean_gas`, `2_geometric_gas`, `10_advanced_topics`)

**Document Discovery**: Scans for `refined_data/` directories within chapters
- Pattern: `docs/source/{chapter}/{document}/refined_data/`
- No configuration needed - just create the directory structure

**Registry Structure**:
```
registries/
├── per_document/           # Individual document registries
│   ├── 01_fragile_gas_framework/
│   │   ├── refined/
│   │   ├── pipeline/
│   │   └── raw/
│   ├── 02_euclidean_gas/
│   │   ├── refined/
│   │   └── pipeline/
│   └── {any_document}/     # Auto-discovered!
│       ├── refined/
│       └── pipeline/
└── combined/               # Aggregated registries
    ├── refined/           # All refined entities
    ├── pipeline/          # All pipeline entities
    └── raw/               # All raw entities
```

## Overview - Pipeline Flow

```
Raw Data (document-parser output)
  ↓ [build_raw_registry.py]
Per-Document Raw Registry

Enriched Data (document-refiner output)
  ↓ [validation.py - REQUIRED]
Validated Refined Data
  ↓ [build_refined_registry.py]
Per-Document Refined Registry

Validated Refined Data
  ↓ [enriched_to_math_types.py]
Pipeline Data (math_types format)
  ↓ [build_pipeline_registry.py]
Per-Document Pipeline Registry

Per-Document Registries
  ↓ [aggregate_registries.py]
Combined Registry (All Documents)

Combined Registries
  ↓ [Dashboard Auto-Discovery]
Interactive Visualization
```

**Single Command Builds Everything**:
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```
This automatically:
1. Discovers all documents with `refined_data/`
2. Transforms `refined_data` → `pipeline_data`
3. Builds per-document registries (refined + pipeline)
4. Aggregates into combined registries

## Tool Descriptions

### Validation Tools (Required Before Registry Building)

Before building registries, always validate refined data:

```bash
# Complete validation (recommended)
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/{chapter}/{document}/refined_data/ \
  --mode complete \
  --output-report validation_report.md
```

**Validation ensures**:
- All entities pass schema validation
- Cross-references are valid
- Dependencies are complete
- Framework consistency

**If validation fails**, use completion workflow:
```bash
# Find incomplete entities
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/{chapter}/{document}/refined_data/

# Generate completion plan
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir docs/source/{chapter}/{document}/refined_data/
```

See [validate-refinement](../validate-refinement/) and [complete-partial-refinement](../complete-partial-refinement/) skills for detailed workflows.

**Only proceed to registry building after validation shows 0 errors.**

---

### 0. `build_all_registries.py` ⭐ **RECOMMENDED**

**Purpose**: Master orchestration script - builds everything automatically

**Input**: `--docs-root` path to docs/source directory
**Output**: Complete registry structure (per-document + combined)

**Features**:
- **Automatic document discovery** (no hardcoded names)
- Transforms `refined_data` → `pipeline_data` (if needed)
- Builds per-document registries (refined + pipeline)
- Aggregates all registries into combined registries
- **Single command does everything**

**Usage**:
```bash
# Build everything automatically (RECOMMENDED)
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

# Custom output location
python -m fragile.mathster.tools.build_all_registries \
  --docs-root docs/source \
  --output-root registries

# Skip aggregation (only build per-document)
python -m fragile.mathster.tools.build_all_registries \
  --docs-root docs/source \
  --skip-aggregate
```

**Output**:
- `registries/per_document/{document}/{refined,pipeline}/`
- `registries/combined/{refined,pipeline}/`
- Statistics report

**When to use**:
- ✅ **Primary workflow** (always start here)
- ✅ Adding new documents
- ✅ Rebuilding all registries
- ✅ Setting up dashboard data

---

### 0.5. `aggregate_registries.py`

**Purpose**: Aggregate per-document registries into combined registry

**Input**: `--per-document-root`, `--type` (refined or pipeline)
**Output**: Combined registry with all documents merged

**Features**:
- Automatic document discovery
- Duplicate detection and handling
- Statistics reporting

**Usage**:
```bash
# Aggregate refined registries
python -m fragile.mathster.tools.aggregate_registries \
  --type refined \
  --per-document-root registries/per_document \
  --output registries/combined/refined

# Aggregate pipeline registries
python -m fragile.mathster.tools.aggregate_registries \
  --type pipeline \
  --per-document-root registries/per_document \
  --output registries/combined/pipeline
```

**When to use**:
- ✅ After building multiple per-document registries manually
- ⚠️ Usually not needed - `build_all_registries.py` calls this automatically

---

### 1. `registry_builders_common.py`

**Purpose**: Shared utilities for all registry builders

**Contents**:
- Path and location helpers
- Label normalization
- Preprocessing functions for attributes, relationships, and edges

**Import example**:
```python
from fragile.proofs.tools.registry_builders_common import (
    extract_location_from_path,
    create_source_location,
    ensure_object_label_prefix,
    preprocess_attributes_added,
    preprocess_relations_established,
)
```

### 2. `enriched_to_math_types.py`

**Purpose**: Transform enriched JSON → pipeline-ready math_types

**Input**: `refined_data/` directories (document-refiner output)
**Output**: `pipeline_data/` directories (math_types format)

**Transformations**:
- Enriched Object → `MathematicalObject`
- Enriched Axiom → `Axiom`
- Enriched Theorem → `TheoremBox`
- Enriched Parameter → `Parameter`
- `ParameterBox` → `Parameter` (uses type_conversions)
- `EquationBox`, `RemarkBox` → semantic linking

**Usage**:
```bash
# Transform Chapter 1 refined data
python -m fragile.mathster.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \
  --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

# Transform only specific entity types
python -m fragile.mathster.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/02_euclidean_gas/refined_data \
  --output docs/source/1_euclidean_gas/02_euclidean_gas/pipeline_data \
  --types objects theorems
```

**Output**:
- Transformed JSON files in `pipeline_data/`
- `transformation_report.json` with statistics

### 3. `build_raw_registry.py`

**Purpose**: Build registry from raw extraction data

**Input**: `raw_data/` directories (document-parser output)
**Output**: `MathematicalRegistry`

**Features**:
- Minimal preprocessing
- Direct path-based location inference
- Handles raw formats with validation

**Usage**:
```bash
# Build from all raw_data directories
python -m fragile.mathster.tools.build_raw_registry \
  --docs-root docs/source \
  --output raw_registry

# Custom docs root
python -m fragile.mathster.tools.build_raw_registry \
  --docs-root /path/to/docs/source \
  --output my_raw_registry
```

**When to use**:
- ✅ Testing extraction pipeline output
- ✅ Quick registry from raw data
- ❌ Production use (use refined or pipeline instead)

### 4. `build_refined_registry.py`

**Purpose**: Build registry from enriched data

**Input**: `refined_data/` directories (document-refiner output)
**Output**: `MathematicalRegistry`

**Features**:
- Automatic transformation during loading
- Full preprocessing (attributes, relationships)
- Comprehensive validation

**Usage**:
```bash
# Build from all refined_data directories
python -m fragile.mathster.tools.build_refined_registry \
  --docs-root docs/source \
  --output refined_registry

# Custom docs root
python -m fragile.mathster.tools.build_refined_registry \
  --docs-root /path/to/docs/source \
  --output my_refined_registry
```

**When to use**:
- ✅ Loading directly from refined_data
- ✅ Skipping transformation step
- ✅ Standard workflow

### 5. `build_pipeline_registry.py`

**Purpose**: Build registry from pre-transformed pipeline data

**Input**: `pipeline_data/` directories (enriched_to_math_types output)
**Output**: `MathematicalRegistry`

**Features**:
- **Fastest loading** (no transformation)
- Direct deserialization
- Pre-validated data

**Usage**:
```bash
# Build from Chapter 1 pipeline data
python -m fragile.mathster.tools.build_pipeline_registry \
  --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
  --output pipeline_registry

# Build from multiple chapters (run separately)
python -m fragile.mathster.tools.build_pipeline_registry \
  --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
  --output chapter1_registry
```

**When to use**:
- ✅ **Optimal workflow** (with transformation cache)
- ✅ Fastest loading performance
- ✅ Production use

## Complete Workflows

### Workflow 0: Automated (Recommended) ⭐

**Step 1: Validate refined data (REQUIRED)**
```bash
# Validate all refined data before building
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/{chapter}/{document}/refined_data/ \
  --mode complete \
  --output-report validation_report.md
```

**Expected**: 0 validation errors. If errors exist, use [complete-partial-refinement](../complete-partial-refinement/) skill to fix.

**Step 2: Build registries**
```bash
# Single command builds everything automatically
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

**What happens**:
1. Discovers all documents with `refined_data/` (no configuration!)
2. Transforms `refined_data` → `pipeline_data` (if needed)
3. Builds per-document registries (refined + pipeline)
4. Aggregates into combined registries

**Output**:
- `registries/per_document/{document}/{refined,pipeline}/`
- `registries/combined/{refined,pipeline}/`

**Pros**: Fully automated, scalable, complete
**Cons**: Requires validation pass before use

---

### Workflow 1: Dashboard Visualization

```bash
# After building registries, visualize them
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

**Features**:
- **Auto-discovery**: Dashboard finds all available registries
- **Dynamic dropdown**: Shows all per-document and combined registries
- **Interactive**: Browse entities, search, visualize relationships

**Data Sources** (auto-discovered):
- Per-document registries: `{document_name} (Pipeline/Refined)`
- Combined registries: `Combined Pipeline Registry`, `Combined Refined Registry`

---

### Workflow 2: Adding New Documents

**Steps**:
1. Extract and refine new document
   ```bash
   # Run extract-and-refine workflow
   # Creates: docs/source/{chapter}/{new_document}/refined_data/
   ```

2. Validate refined data (REQUIRED)
   ```bash
   python -m fragile.mathster.tools.validation \
     --refined-dir docs/source/{chapter}/{new_document}/refined_data/ \
     --mode complete
   ```
   **Expected**: 0 errors. If validation fails, complete entities before continuing.

3. Build all registries (automatic discovery!)
   ```bash
   python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
   ```

**Result**: New document automatically included in:
- `registries/per_document/{new_document}/`
- `registries/combined/` (merged with existing documents)
- Dashboard dropdown (no code changes needed!)

**Example**: Adding `02_euclidean_gas`
```bash
# After creating refined_data/, validate first:
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/02_euclidean_gas/refined_data/ \
  --mode complete

# If validation passes, build registries:
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

# The document is automatically:
# - Discovered (no hardcoded names)
# - Transformed (refined → pipeline)
# - Built (per-document registries)
# - Aggregated (combined registries)
# - Available in dashboard
```

---

### Workflow 3: Manual Per-Document (Advanced)

**Use when**: Building specific document only (not typical)

```bash
# Build single document manually
python -m fragile.mathster.tools.build_refined_registry \
  --docs-root docs/source/{chapter} \
  --output registries/per_document/{document}/refined
```

**Note**: For most use cases, use Workflow 0 (Automated) instead

---

## File Structure

```
src/fragile/proofs/tools/
├── build_all_registries.py           # ⭐ Master orchestrator (420 lines)
├── aggregate_registries.py           # Registry aggregation (285 lines)
├── registry_builders_common.py       # Shared utilities (300 lines)
├── enriched_to_math_types.py         # Transformation layer (600 lines)
├── build_raw_registry.py             # Raw → Registry (450 lines)
├── build_refined_registry.py         # Refined → Registry (450 lines)
├── build_pipeline_registry.py        # Pipeline → Registry (300 lines)
├── WORKFLOW.md                        # Comprehensive workflow guide
└── combine_all_chapters.py           # DEPRECATED (keep for reference)

src/fragile/proofs/
└── proof_pipeline_dashboard.py       # Interactive visualization dashboard
```

**Total**: ~2800 lines (fully document-agnostic)

---

## Output Registry Structure

All builders save to the same format:

```
output_registry/
├── objects/
│   └── obj-*.json
├── axioms/
│   └── axiom-*.json
├── parameters/
│   └── param-*.json
├── properties/
│   └── prop-*.json
├── relationships/
│   └── rel-*.json
├── theorems/
│   └── thm-*.json
└── index.json
```

**Loading the registry**:
```python
from fragile.proofs import load_registry_from_directory, MathematicalRegistry

registry = load_registry_from_directory(MathematicalRegistry, 'output_registry')

# Use registry
objects = registry.get_all_objects()
theorems = registry.get_all_theorems()
relationships = registry.get_all_relationships()
```

---

## Performance Comparison

| Workflow | Speed | Quality | Automation | Use Case |
|----------|-------|---------|------------|----------|
| **Workflow 0 (Automated)** ⭐ | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Best | 🤖 Full | **Primary workflow** |
| Manual per-document | ⚡ Slow | ⭐⭐⭐ Good | ❌ None | Advanced use only |
| Raw data build | ⚡⚡ Fast | ⭐⭐ Medium | 🤖 Partial | Testing extraction |

**Recommendation**: Always use **Workflow 0 (Automated)** - it's faster, better, and fully automated.

---

## Dashboard Visualization

**Location**: `src/fragile/proofs/proof_pipeline_dashboard.py`

### Auto-Discovery Features

The dashboard automatically discovers all available registries at runtime:

**How it works**:
1. Scans `registries/per_document/` for all documents
2. Checks for `refined/` and `pipeline/` subdirectories
3. Scans `registries/combined/` for combined registries
4. Dynamically populates data source dropdown

**Identifier Format**: `per_doc_{document_name}_{type}`
- Example: `per_doc_01_fragile_gas_framework_pipeline`
- Example: `per_doc_02_euclidean_gas_refined`
- Combined: `combined_pipeline`, `combined_refined`

### Launch Dashboard

```bash
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

### Data Source Dropdown (Auto-Generated)

The dropdown shows all discovered registries:
- **Per-Document**: `{document_name} (Pipeline)`, `{document_name} (Refined)`
- **Combined**: `Combined Pipeline Registry`, `Combined Refined Registry`
- **Custom**: User-specified path

**No code changes needed when adding documents!**

### Usage

1. Launch dashboard: `panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show`
2. Select data source from dropdown
3. Click "Reload Data"
4. Browse entities, search, visualize relationships

---

## Validation and Error Handling

### Pre-Build Validation (Required)

Always validate refined data before building registries:

```bash
# Step 1: Validate (REQUIRED)
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/{chapter}/{document}/refined_data/ \
  --mode complete \
  --output-report validation_report.md

# Step 2: Build registries (only if validation passes)
if [ $? -eq 0 ]; then
  python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
else
  echo "❌ Validation failed - fix errors before building registries"
  echo "See validation_report.md for details"
fi
```

**Why validate first?**
- Prevents downstream errors in registry building
- Ensures data quality and completeness
- Identifies missing cross-references early
- Verifies framework consistency

See [validate-refinement](../validate-refinement/) skill for comprehensive validation workflow.

### Build-Time Error Handling

Registry building tools:
- ✅ Continue processing on individual file errors
- ✅ Collect all errors for reporting
- ✅ Report statistics (success/failure counts)
- ✅ Exit with error code if errors occurred

**Error handling example**:
```bash
# Build registry and capture exit code
python -m fragile.mathster.tools.build_refined_registry \
  --docs-root docs/source \
  --output refined_registry

# Check exit code
if [ $? -eq 0 ]; then
  echo "✅ Success"
else
  echo "❌ Errors occurred, check output"
fi
```

---

## Deprecation Notice

**`combine_all_chapters.py`** is now **DEPRECATED**.

**Migration path**: Use `build_all_registries.py` (recommended) or individual builders:
- For automated workflow: Use `build_all_registries.py` ⭐
- For manual raw data: Use `build_raw_registry.py`
- For manual refined data: Use `build_refined_registry.py`
- For manual pipeline data: Use `build_pipeline_registry.py`

The old script is kept for reference but should not be used for new workflows.

---

## Troubleshooting

### Issue: Validation errors on raw data

**Cause**: Raw data has truly raw extraction format
**Solution**: Use `build_refined_registry.py` or transform to pipeline first

### Issue: SourceLocation validation errors

**Cause**: Enriched data has old SourceLocation format
**Solution**: Document-refiner should update SourceLocation fields, or fix manually

### Issue: Missing mathematical_expression field

**Cause**: Field was optional in enriched schema
**Solution**: Add field to enriched data or skip validation temporarily

### Issue: Slow loading

**Cause**: Using refined_registry builder (transforms on-the-fly)
**Solution**: Pre-transform with `enriched_to_math_types.py`, then use `build_pipeline_registry.py`

### Issue: Tool picking up outdated data from unexpected locations

**Cause**: ❌ **FIXED** - Previously used recursive glob which could find scattered old data
**Solution**: Tools now only search for `raw_data`/`refined_data` as **direct children** of section directories. This prevents picking up:
- Old data in nested subdirectories
- Other directories like `data/` (which contain reports, not entity data)
- Scattered outdated files elsewhere in the docs tree

**Technical Details**:
- `build_raw_registry.py`: Only looks for `{section_dir}/raw_data/`
- `build_refined_registry.py`: Only looks for `{section_dir}/refined_data/`
- `build_pipeline_registry.py`: Takes explicit `--pipeline-dir` argument (no search)

**Example**: In `docs/source/1_euclidean_gas/`:
- ✅ Will find: `01_fragile_gas_framework/refined_data/objects/`
- ❌ Will ignore: `02_euclidean_gas/data/` (reports, not entity data)
- ❌ Will ignore: Nested `*/old_data/refined_data/` (not direct child)

---

## Related Documentation

### Validation and Completion Skills
- **[validate-refinement](../validate-refinement/)** - Comprehensive QA workflow (required before registry building)
- **[complete-partial-refinement](../complete-partial-refinement/)** - Systematic completion of incomplete entities
- **[refine-entity-type](../refine-entity-type/)** - Entity-specific refinement guidance

### Extraction and Refinement
- **[extract-and-refine](../extract-and-refine/)** - Complete extraction and refinement workflow

### Code References
- **Pipeline Types**: `src/fragile/proofs/core/math_types.py`
- **Enriched Types**: `src/fragile/proofs/core/enriched_types.py`
- **Type Conversions**: `src/fragile/proofs/core/type_conversions.py`
- **Registry System**: `src/fragile/proofs/registry/`
- **Storage Layer**: `src/fragile/proofs/registry/storage.py`
- **Validation Module**: `src/fragile/proofs/tools/validation/`

---

**Questions?** Check the docstrings in each tool file for detailed implementation notes.
