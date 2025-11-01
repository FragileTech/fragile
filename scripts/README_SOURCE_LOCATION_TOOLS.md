# Source Location Enrichment Tools - Quick Reference

## Quick Commands

### Check Current Status
```bash
# Full validation report
python scripts/validate_all_source_locations.py --all-documents docs/source/

# Detailed per-file diagnostic
python scripts/comprehensive_diagnostic_report.py \
    --directory docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data \
    --markdown docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
    --output DIAGNOSTIC.txt
```

### Fix Issues

```bash
# Fix missing line_range (directive matching)
python scripts/enrich_line_ranges_from_directives.py \
    --document docs/source/1_euclidean_gas/01_fragile_gas_framework

# Fix missing line_range (text matching)
python scripts/aggressive_text_matching.py \
    --document docs/source/1_euclidean_gas/01_fragile_gas_framework

# Fix missing line_range (special cases: parameters, mathster, remarks)
python scripts/handle_special_cases.py \
    --document docs/source/1_euclidean_gas/01_fragile_gas_framework

# Extract sections from line_range
python scripts/extract_sections_from_markdown.py \
    --markdown docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
    --raw-data docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data

# Normalize section format (remove §, "Section", etc.)
python scripts/normalize_section_format.py \
    --batch docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data

# Add missing directive_label fields
python scripts/add_missing_directive_labels.py \
    --directory docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data
```

## Tool Overview

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `validate_all_source_locations.py` | Validate strict requirements | Check current status |
| `comprehensive_diagnostic_report.py` | Per-file detailed diagnostic | Understand what's missing |
| `enrich_line_ranges_from_directives.py` | Find via `:label:` directives | First pass for line_range |
| `aggressive_text_matching.py` | Find via text content | Second pass |
| `handle_special_cases.py` | Parameters, proofs, remarks | Third pass |
| `extract_sections_from_markdown.py` | Extract section from line_range | After finding line_range |
| `normalize_section_format.py` | Clean up section format | After extraction |
| `add_missing_directive_labels.py` | Auto-add directive_label | Final cleanup |
| `batch_fix_source_locations.py` | All-in-one batch fix | Initial bulk fix |

## Validation Checks

### What Gets Validated

- ✅ `document_id` format: `[0-9]{2}_[a-z_]+`
- ✅ `section` format: `X.Y.Z` (only digits and dots)
- ✅ `line_range` format: `[int, int]` with valid bounds
- ✅ `directive_label` presence (when entity has label)
- ✅ No § symbols in section
- ✅ No out-of-bounds line ranges

### Pass Criteria

All 194 valid files meet:
1. Valid line_range: `[int, int]` with start >= 1, start <= end, within file bounds
2. Valid section: X.Y.Z format (no symbols, no text)
3. Valid directive_label: present if entity has label
4. All field types correct

## Current Status (01_fragile_gas_framework)

- **194/217 (89.4%) valid** ✅
- **22 missing line_range** (cannot trace to source)
- **1 missing section** (has line_range, can extract)

### Quality Metrics - ALL PASSED ✅

- ✅ Zero § symbols in source_location.section
- ✅ All line_range are valid [int, int] tuples
- ✅ All sections in X.Y.Z format
- ✅ Zero out-of-bounds line_range

## Typical Workflow

### For New Document

1. **Initial enrichment:**
   ```bash
   python scripts/batch_fix_source_locations.py --document DIR
   ```

2. **Find line_range (3 passes):**
   ```bash
   python scripts/enrich_line_ranges_from_directives.py --document DIR
   python scripts/aggressive_text_matching.py --document DIR
   python scripts/handle_special_cases.py --document DIR
   ```

3. **Extract and normalize:**
   ```bash
   python scripts/extract_sections_from_markdown.py --markdown FILE.md --raw-data DIR/raw_data
   python scripts/normalize_section_format.py --batch DIR/raw_data
   python scripts/add_missing_directive_labels.py --directory DIR/raw_data
   ```

4. **Validate:**
   ```bash
   python scripts/validate_all_source_locations.py --all-documents docs/source/
   ```

### For Quick Fixes

```bash
# Just normalize sections
python scripts/normalize_section_format.py --batch DIR

# Just add directive_labels
python scripts/add_missing_directive_labels.py --directory DIR

# Just validate
python scripts/validate_all_source_locations.py --all-documents docs/source/
```

## Error Types

| Error | Severity | Meaning | Fix |
|-------|----------|---------|-----|
| Missing line_range | CRITICAL | Cannot trace to source | Use enrichment tools |
| Invalid section format | HIGH | Has §, text, or wrong format | Use normalize_section_format.py |
| Out of bounds | HIGH | line_range exceeds file length | Manual correction needed |
| Missing section | MEDIUM | Has line_range, no section | Use extract_sections_from_markdown.py |
| Missing directive_label | LOW | Optional field missing | Use add_missing_directive_labels.py |

## Output Files

- `FINAL_SUMMARY_REPORT.md` - Overall statistics and achievements
- `COMPREHENSIVE_DIAGNOSTIC_REPORT.txt` - Per-file detailed issues
- `SOURCE_LOCATION_ENRICHMENT_SUMMARY.md` - Original summary with tools list
- `fix_report_first_run.json` - Initial batch fix statistics

## Next Steps

To achieve 100% coverage:

1. **Manual review (22 files):** Find locations in markdown manually
2. **Extract section (1 file):** Run extract_sections_from_markdown.py
3. **Update Pydantic model:** Add validators to SourceLocation class
