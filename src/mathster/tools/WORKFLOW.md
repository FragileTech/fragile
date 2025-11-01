# Registry Pipeline Workflow (Document-Agnostic)

This document explains the complete workflow for the document-agnostic registry pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Document Discovery](#document-discovery)
4. [Pipeline Stages](#pipeline-stages)
5. [Registry Structure](#registry-structure)
6. [Adding New Documents](#adding-new-documents)
7. [Individual Tools](#individual-tools)
8. [Dashboard Usage](#dashboard-usage)
9. [Troubleshooting](#troubleshooting)

## Overview

The registry pipeline is **fully document-agnostic** - it automatically discovers and processes all documents without requiring any hardcoded configuration. The key principle is:

> **Add new document → Create refined_data → Run build_all → Everything else happens automatically!**

### Key Features

✓ **Zero configuration**: No hardcoded document names anywhere
✓ **Auto-discovery**: Automatically finds all chapters and documents
✓ **Single command**: One command builds everything
✓ **Dynamic dashboard**: Adapts to available registries automatically
✓ **Scalable**: Add unlimited documents without code changes

## Quick Start

### Build Everything Automatically

```bash
# From project root
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

This single command:
1. Discovers all documents with `refined_data/`
2. Transforms `refined_data` → `pipeline_data` (if needed)
3. Builds per-document registries
4. Aggregates into combined registries

### Visualize in Dashboard

```bash
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

Dashboard automatically discovers and displays all available registries.

## Document Discovery

### Chapter Discovery Pattern

Chapters are discovered using the pattern: `{digit(s)}_{name}`

**Examples:**
- `1_euclidean_gas` ✓
- `2_geometric_gas` ✓
- `10_advanced_topics` ✓
- `chapter_one` ✗ (no digit prefix)

### Document Discovery Pattern

Within each chapter, documents are discovered by:
1. Looking for numbered subdirectories
2. Checking for `refined_data/` directory inside
3. Verifying `refined_data/` contains entity subdirectories (objects/, axioms/, theorems/)

**Example Structure:**
```
docs/source/
├── 1_euclidean_gas/
│   ├── 01_fragile_gas_framework/
│   │   └── refined_data/  ✓ Discovered!
│   │       ├── objects/
│   │       ├── axioms/
│   │       └── theorems/
│   ├── 02_euclidean_gas/
│   │   └── refined_data/  ✓ Discovered!
│   └── some_notes.md  ✗ Skipped (not a directory)
└── 2_geometric_gas/
    └── 03_geometric_gas/
        └── refined_data/  ✓ Discovered!
```

## Pipeline Stages

### Stage 1: Raw Data Extraction
**Source:** MyST markdown documents
**Output:** `docs/source/{chapter}/{document}/raw_data/`
**Tool:** `document-parser` agent

Extracts mathematical entities directly from markdown.

### Stage 2: Data Refinement
**Source:** `raw_data/`
**Output:** `docs/source/{chapter}/{document}/refined_data/`
**Tool:** `document-refiner` agent

Enriches entities with:
- Semantic analysis
- Cross-references
- Dependency tracking
- Framework consistency checks

### Stage 3: Pipeline Transformation
**Source:** `refined_data/`
**Output:** `docs/source/{chapter}/{document}/pipeline_data/`
**Tool:** `enriched_to_math_types`

Transforms enriched JSON to math_types format for registry.

### Stage 4: Per-Document Registry Build
**Source:** `refined_data/` or `pipeline_data/`
**Output:** `registries/per_document/{document}/{refined,pipeline}/`
**Tools:** `build_refined_registry`, `build_pipeline_registry`

Creates registries for individual documents.

### Stage 5: Combined Registry Aggregation
**Source:** `registries/per_document/*/`
**Output:** `registries/combined/{refined,pipeline}/`
**Tool:** `aggregate_registries`

Merges all per-document registries into combined registries.

## Registry Structure

```
registries/
├── combined/
│   ├── refined/
│   │   ├── index.json
│   │   ├── objects/
│   │   ├── axioms/
│   │   └── theorems/
│   └── pipeline/
│       ├── index.json
│       ├── objects/
│       ├── axioms/
│       └── theorems/
└── per_document/
    ├── 01_fragile_gas_framework/
    │   ├── refined/
    │   │   ├── index.json
    │   │   ├── objects/
    │   │   ├── axioms/
    │   │   └── theorems/
    │   └── pipeline/
    │       ├── index.json
    │       ├── objects/
    │       ├── axioms/
    │       └── theorems/
    └── 02_euclidean_gas/
        ├── refined/
        └── pipeline/
```

## Adding New Documents

### Complete Example: Adding 02_euclidean_gas

#### Step 1: Extract and Refine

```bash
# Extract from markdown (document-parser agent)
# Creates: docs/source/1_euclidean_gas/02_euclidean_gas/raw_data/

# Refine extracted data (document-refiner agent)
# Creates: docs/source/1_euclidean_gas/02_euclidean_gas/refined_data/
```

#### Step 2: Build All Registries

```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

**What happens automatically:**
1. Discovers `02_euclidean_gas` (no configuration needed!)
2. Transforms `refined_data` → `pipeline_data`
3. Builds `registries/per_document/02_euclidean_gas/refined/`
4. Builds `registries/per_document/02_euclidean_gas/pipeline/`
5. Aggregates all documents into `registries/combined/`

#### Step 3: Visualize

```bash
panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show
```

Dashboard automatically shows:
- 01_fragile_gas_framework (Pipeline/Refined)
- **02_euclidean_gas (Pipeline/Refined)** ← NEW!
- Combined Pipeline Registry
- Combined Refined Registry

**No code changes needed!**

## Individual Tools

### build_all_registries.py (Master Orchestrator)

Runs the complete pipeline automatically.

```bash
# Build everything
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

# Custom output location
python -m fragile.mathster.tools.build_all_registries \
    --docs-root docs/source \
    --output-root my_registries

# Skip combined registry aggregation
python -m fragile.mathster.tools.build_all_registries \
    --docs-root docs/source \
    --skip-aggregate
```

### build_refined_registry.py

Builds registry from refined_data (automatically transforms during build).

```bash
# Build for all chapters
python -m fragile.mathster.tools.build_refined_registry \
    --docs-root docs/source \
    --output registries/per_document/01_fragile_gas_framework/refined

# Build for specific chapter
python -m fragile.mathster.tools.build_refined_registry \
    --docs-root docs/source/1_euclidean_gas \
    --output registries/per_document/01_fragile_gas_framework/refined
```

### build_pipeline_registry.py

Builds registry from pipeline_data (pre-validated format).

```bash
python -m fragile.mathster.tools.build_pipeline_registry \
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
    --output registries/per_document/01_fragile_gas_framework/pipeline
```

### enriched_to_math_types.py

Transforms refined_data to pipeline_data format.

```bash
python -m fragile.mathster.tools.enriched_to_math_types \
    --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \
    --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data
```

### aggregate_registries.py

Aggregates per-document registries into combined registries.

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

## Dashboard Usage

### Auto-Discovery

The dashboard automatically:
1. Scans `registries/per_document/` for available documents
2. Checks `registries/combined/` for combined registries
3. Dynamically populates the data source dropdown

No manual configuration needed!

### Data Source Options

The dropdown is generated at runtime and shows:
- **Per-Document Registries**: `{document_name} (Pipeline/Refined)`
- **Combined Registries**: `Combined Pipeline Registry`, `Combined Refined Registry`
- **Custom Path**: User-specified path

### Switching Documents

Simply select from the dropdown and click "Reload Data". The dashboard will:
1. Load the selected registry
2. Update statistics
3. Regenerate graph
4. Update filters (axiom frameworks, documents, etc.)

## Troubleshooting

### No Documents Found

**Symptom**: `build_all_registries` reports 0 documents discovered

**Diagnosis**:
```bash
# Check directory structure
ls -la docs/source/1_euclidean_gas/

# Check for refined_data
ls -la docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/
```

**Solution**: Ensure:
- Chapter directory matches pattern `{digit}_{name}`
- Document has `refined_data/` directory
- `refined_data/` contains entity subdirectories (objects/, axioms/, theorems/)

### Dashboard Shows No Registries

**Symptom**: Dashboard dropdown is empty (except "Custom Path")

**Diagnosis**:
```bash
# Check if registries exist
ls -la registries/per_document/

# Check if index.json exists
ls -la registries/per_document/01_fragile_gas_framework/pipeline/
```

**Solution**: Run `build_all_registries` to create registries:
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

### Registry Load Fails

**Symptom**: Dashboard shows "Error loading registry"

**Diagnosis**: Check console output for specific error

**Common Causes:**
1. Corrupted JSON files
2. Missing required fields
3. Invalid Pydantic model data

**Solution**:
```bash
# Rebuild the registry
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

### Duplicate Entities in Combined Registry

**Symptom**: `aggregate_registries` warns about duplicates

**Diagnosis**: Same entity label exists in multiple documents

**Solution**: This is expected behavior. The aggregator:
1. Keeps first occurrence
2. Skips duplicates from other documents
3. Reports count of duplicates

To fix:
- Ensure entities have unique labels across documents
- Or accept that duplicates will be skipped (first wins)

## Best Practices

### 1. Always Use build_all_registries

Instead of running individual tools, use the master script:
```bash
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

This ensures:
- All documents are processed
- Transformations are up to date
- Per-document and combined registries are in sync

### 2. Document Naming Convention

Follow the pattern for automatic discovery:
- Chapters: `{digit}_{lowercase_with_underscores}`
- Documents: `{two_digit_number}_{lowercase_with_underscores}`

Examples:
- ✓ `1_euclidean_gas/01_fragile_gas_framework/`
- ✓ `2_geometric_gas/03_geometric_gas/`
- ✗ `ChapterOne/Introduction/` (won't be discovered)

### 3. Clean Builds

If you encounter issues, do a clean rebuild:
```bash
# Remove old registries
rm -rf registries/per_document/* registries/combined/*

# Rebuild everything
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source
```

### 4. Dashboard Reload

After rebuilding registries, reload the dashboard:
1. Click "Reload Data" button
2. Or restart the dashboard: `panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show`

## Advanced Usage

### Processing Subset of Documents

To process only specific documents, temporarily move others out of `docs/source/`:
```bash
# Move unwanted documents temporarily
mv docs/source/1_euclidean_gas/02_euclidean_gas /tmp/

# Run build_all (only processes 01_fragile_gas_framework now)
python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

# Restore document
mv /tmp/02_euclidean_gas docs/source/1_euclidean_gas/
```

### Custom Registry Locations

All tools support custom output paths:
```bash
python -m fragile.mathster.tools.build_all_registries \
    --docs-root docs/source \
    --output-root custom_registries
```

Then point dashboard to custom location:
- Select "Custom Path" in dropdown
- Enter path: `custom_registries/per_document/01_fragile_gas_framework/pipeline`
- Click "Reload Data"

## See Also

- [REGISTRY_COMPLETION_REPORT.txt](../../../../REGISTRY_COMPLETION_REPORT.txt) - Current pipeline status
- [README.md](../README.md) - Proofs module overview
- Individual tool `--help` for detailed options

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review console output for specific errors
3. Verify directory structure matches expected pattern
4. Try clean rebuild

The document-agnostic design means: if it works for one document, it works for all documents. No special configuration needed!
