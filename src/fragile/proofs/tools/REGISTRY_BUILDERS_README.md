# Registry Builders - Complete Workflow Guide

This directory contains tools for building `MathematicalRegistry` instances from different data formats.

## Overview

The registry building system has been split into focused, modular tools:

```
Raw Data (document-parser output)
  ↓ [build_raw_registry.py]
Raw Registry

Enriched Data (document-refiner output)
  ↓ [build_refined_registry.py]
Refined Registry

Enriched Data
  ↓ [enriched_to_math_types.py]
Pipeline Data (math_types format)
  ↓ [build_pipeline_registry.py]
Pipeline Registry (fastest)
```

## Tool Descriptions

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
python -m fragile.proofs.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \
  --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

# Transform only specific entity types
python -m fragile.proofs.tools.enriched_to_math_types \
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
python -m fragile.proofs.tools.build_raw_registry \
  --docs-root docs/source \
  --output raw_registry

# Custom docs root
python -m fragile.proofs.tools.build_raw_registry \
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
python -m fragile.proofs.tools.build_refined_registry \
  --docs-root docs/source \
  --output refined_registry

# Custom docs root
python -m fragile.proofs.tools.build_refined_registry \
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
python -m fragile.proofs.tools.build_pipeline_registry \
  --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
  --output pipeline_registry

# Build from multiple chapters (run separately)
python -m fragile.proofs.tools.build_pipeline_registry \
  --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
  --output chapter1_registry
```

**When to use**:
- ✅ **Optimal workflow** (with transformation cache)
- ✅ Fastest loading performance
- ✅ Production use

## Complete Workflows

### Workflow 1: Quick Testing (Raw Data)

```bash
# Extract → Build (skip refinement)
python -m fragile.proofs.tools.build_raw_registry \
  --docs-root docs/source \
  --output raw_registry
```

**Pros**: Fast, simple
**Cons**: Lower quality, validation errors expected

---

### Workflow 2: Standard (Refined Data)

```bash
# Extract → Refine → Build
python -m fragile.proofs.tools.build_refined_registry \
  --docs-root docs/source \
  --output refined_registry
```

**Pros**: Good quality, automatic transformation
**Cons**: Slower than pipeline workflow

---

### Workflow 3: Optimal (Pipeline Data with Cache)

```bash
# Extract → Refine → Transform → Build
# Step 1: Transform enriched → pipeline (once)
python -m fragile.proofs.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \
  --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

# Step 2: Build registry (fast, repeatable)
python -m fragile.proofs.tools.build_pipeline_registry \
  --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
  --output pipeline_registry
```

**Pros**: Fastest loading, best quality, repeatable
**Cons**: Two-step process

---

### Workflow 4: Multiple Chapters

```bash
# Transform each chapter
for chapter in 01_fragile_gas_framework 02_euclidean_gas; do
  python -m fragile.proofs.tools.enriched_to_math_types \
    --input docs/source/1_euclidean_gas/$chapter/refined_data \
    --output docs/source/1_euclidean_gas/$chapter/pipeline_data
done

# Build unified registry from all chapters
python -m fragile.proofs.tools.build_refined_registry \
  --docs-root docs/source \
  --output unified_registry
```

---

## File Structure

```
src/fragile/proofs/tools/
├── registry_builders_common.py       # Shared utilities (300 lines)
├── enriched_to_math_types.py         # Transformation layer (600 lines)
├── build_raw_registry.py             # Raw → Registry (450 lines)
├── build_refined_registry.py         # Refined → Registry (450 lines)
├── build_pipeline_registry.py        # Pipeline → Registry (300 lines)
├── combine_all_chapters.py           # DEPRECATED (keep for reference)
└── REGISTRY_BUILDERS_README.md       # This file
```

**Total**: ~2100 lines (vs original 914 lines, but much more organized)

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

| Tool | Input Format | Speed | Quality | Use Case |
|------|--------------|-------|---------|----------|
| `build_raw_registry` | raw_data | ⚡⚡ Fast | ⭐⭐ Medium | Testing extraction |
| `build_refined_registry` | refined_data | ⚡ Moderate | ⭐⭐⭐ Good | Standard workflow |
| `build_pipeline_registry` | pipeline_data | ⚡⚡⚡ Fastest | ⭐⭐⭐⭐ Best | Production use |

**Recommendation**: Use **Workflow 3 (Optimal)** for production pipelines.

---

## Validation and Error Handling

All tools:
- ✅ Continue processing on individual file errors
- ✅ Collect all errors for reporting
- ✅ Report statistics (success/failure counts)
- ✅ Exit with error code if errors occurred

**Error handling example**:
```bash
# Build registry and capture exit code
python -m fragile.proofs.tools.build_refined_registry \
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

**Migration path**:
- For raw data: Use `build_raw_registry.py`
- For refined data: Use `build_refined_registry.py`
- For pipeline data: Use `build_pipeline_registry.py`

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

- **Pipeline Types**: `src/fragile/proofs/core/math_types.py`
- **Enriched Types**: `src/fragile/proofs/core/enriched_types.py`
- **Type Conversions**: `src/fragile/proofs/core/type_conversions.py`
- **Registry System**: `src/fragile/proofs/registry/`
- **Storage Layer**: `src/fragile/proofs/registry/storage.py`

---

## Version History

- **v1.0.1** (2025-10-28): Directory search fix
  - Fixed recursive glob issue that could pick up scattered old data
  - Tools now only search for `raw_data`/`refined_data` as direct children of section directories
  - Prevents accidentally processing outdated or report data

- **v1.0.0** (2025-10-28): Initial split from `combine_all_chapters.py`
  - Created 5 focused tools
  - Added transformation layer
  - Improved modularity and performance

---

**Questions?** Check the docstrings in each tool file for detailed implementation notes.
