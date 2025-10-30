# Registry Builder Directory Search Fix

**Date**: 2025-10-28
**Version**: v1.0.1

## Problem

The original registry builders used `rglob()` to recursively search for `raw_data` and `refined_data` directories:

```python
# ❌ OLD (INCORRECT)
for subdir in chapter_path.rglob("raw_data"):
    # This could pick up ANY raw_data directory anywhere in the tree
```

This caused several issues:
1. Could pick up old/outdated data in nested subdirectories
2. Could accidentally process report directories with similar names
3. No control over which specific directories to process
4. Risk of processing inconsistent or scattered data

## Solution

Changed to only search for `raw_data`/`refined_data` as **direct children** of section directories:

```python
# ✅ NEW (CORRECT)
for section_dir in chapter_path.iterdir():
    if not section_dir.is_dir():
        continue

    # Check for raw_data as immediate child of section directory
    raw_data_dir = section_dir / "raw_data"
    if not raw_data_dir.exists():
        continue
```

## Files Modified

1. **`/home/guillem/fragile/src/fragile/proofs/tools/build_raw_registry.py`**
   - Updated `find_raw_data_directories()` method
   - Updated `process_chapter()` to handle multiple sections correctly

2. **`/home/guillem/fragile/src/fragile/proofs/tools/build_refined_registry.py`**
   - Updated `find_refined_data_directories()` method
   - Updated `process_chapter()` to handle multiple sections correctly

3. **`/home/guillem/fragile/src/fragile/proofs/tools/REGISTRY_BUILDERS_README.md`**
   - Added troubleshooting section documenting the fix
   - Updated version history

## Expected Directory Structure

The tools now correctly handle this structure:

```
docs/source/
└── 1_euclidean_gas/                    # Chapter
    ├── 01_fragile_gas_framework/       # Section
    │   ├── raw_data/                   # ✅ Found (direct child)
    │   │   ├── objects/
    │   │   ├── axioms/
    │   │   └── theorems/
    │   └── refined_data/               # ✅ Found (direct child)
    │       ├── objects/
    │       └── theorems/
    ├── 02_euclidean_gas/               # Section
    │   ├── data/                       # ❌ Ignored (not "refined_data")
    │   ├── objects/                    # ❌ Ignored (not inside raw_data/)
    │   └── theorems/                   # ❌ Ignored (not inside raw_data/)
    └── 03_cloning/                     # Section
        └── old_backup/
            └── refined_data/           # ❌ Ignored (not direct child of section)
```

## Verification

### Test 1: Only finds correct directories

```bash
$ python -m fragile.proofs.tools.build_refined_registry --docs-root docs/source --output test_registry

Found 2 chapters:
  - 1_euclidean_gas
  - 2_geometric_gas

Processing Chapter: 1_euclidean_gas
  Processing objects from: 1_euclidean_gas/01_fragile_gas_framework/refined_data/objects
  Processing theorems from: 1_euclidean_gas/01_fragile_gas_framework/refined_data/theorems

Processing Chapter: 2_geometric_gas
  ⚠️  No refined_data directories found  # ✅ Correctly ignored data/ directories
```

### Test 2: Does NOT pick up other directories

The following directories are correctly ignored:
- ❌ `docs/source/1_euclidean_gas/02_euclidean_gas/data/` (reports, not refined_data)
- ❌ `docs/source/2_geometric_gas/11_geometric_gas/data/` (reports, not refined_data)
- ❌ `docs/source/2_geometric_gas/13_geometric_gas_c3_regularity/data/` (reports)
- ❌ Any nested `refined_data/` directories that aren't direct children

## Benefits

1. **Predictable behavior**: Only processes explicitly structured directories
2. **No accidental data**: Won't pick up scattered old files
3. **Performance**: Only scans one level (no recursive search)
4. **Maintainability**: Clear contract for where data should be placed
5. **Safety**: Prevents processing outdated or report data

## Recommendations

### For Document Parsers/Refiners

Always create data in the expected locations:
```
{chapter_dir}/{section_dir}/raw_data/{entity_type}/
{chapter_dir}/{section_dir}/refined_data/{entity_type}/
```

### For Reports

Place reports in the centralized reports directory:
```
{chapter_dir}/{section_dir}/reports/{report_type}/
```

**Do NOT** create ad-hoc `data/` directories for reports.

## Related

- See `REGISTRY_BUILDERS_README.md` for full documentation
- See agent definitions in `.claude/agents/` for correct report output paths
- All agents now use centralized `reports/` directory structure
