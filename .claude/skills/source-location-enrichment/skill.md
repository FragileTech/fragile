# Source Location Enrichment Skill

Enrich raw or refined mathematical entities with precise source locations (line ranges) linking them to source markdown documents.

## Purpose

Add precise line-range tracking to mathematical entities extracted from markdown documents. This enables:
- Exact traceability from programmatic objects back to source documentation
- Validation of extractions against original text
- Bidirectional navigation between code and docs
- Source location grounding for all entities in registries

## When to Use This Skill

Use this skill when:
1. **Raw data lacks source locations**: After extraction, raw JSON files don't have line_range info
2. **Validating enrichment**: Verify that source locations point to correct content
3. **Re-enrichment needed**: After markdown restructuring or refactoring
4. **Manual entity creation**: Adding source locations to manually created entities
5. **Fixing incomplete enrichment**: Some entities missing line ranges after pipeline run
6. **Debugging extraction**: Verify that extracted text matches source location

## Available Tools

### Core Utilities

1. **`source_location_enricher.py`** - Main enrichment utility
   - Location: `src/fragile/proofs/tools/source_location_enricher.py`
   - Functions:
     - `enrich_single_entity()` - Enrich one JSON file
     - `enrich_directory()` - Enrich all entities in raw_data/
     - `batch_enrich_all_documents()` - Process entire corpus
   - Uses text matching to find line ranges automatically

2. **`find_source_location.py`** - Interactive CLI tool
   - Location: `src/tools/find_source_location.py`
   - Commands:
     - `find-text` - Find text snippet
     - `find-directive` - Find by Jupyter Book directive label
     - `find-equation` - Find LaTeX equation
     - `find-section` - Find section by heading
     - `batch` - Process batch queries from CSV
   - User-friendly for manual lookups

3. **`line_finder.py`** - Low-level utilities
   - Location: `src/fragile/proofs/tools/line_finder.py`
   - Functions used by enricher internally
   - Can be called directly for custom searches

4. **`source_helpers.py`** - SourceLocation builders
   - Location: `src/fragile/proofs/utils/source_helpers.py`
   - SourceLocationBuilder methods:
     - `from_markdown_location()` - Most precise (line range)
     - `from_jupyter_directive()` - Precise (directive label)
     - `from_section()` - Less precise (section only)
     - `from_raw_entity()` - Auto-fallback strategy
     - `with_fallback()` - Flexible fallback
     - `minimal()` - Least precise (document only)

## Workflow

### 1. Detect What Needs Enrichment

First, identify which entities are missing source locations or have incomplete data:

```bash
# Check raw_data for missing source locations
find docs/source -name "*.json" -path "*/raw_data/*" | while read file; do
    if ! grep -q "source_location" "$file"; then
        echo "Missing: $file"
    fi
done

# Or check a specific document
python -c "
import json
from pathlib import Path

raw_dir = Path('docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data')
missing = []
for json_file in raw_dir.glob('*/*.json'):
    data = json.loads(json_file.read_text())
    if 'source_location' not in data:
        missing.append(json_file.name)

print(f'Missing source_location: {len(missing)} files')
for f in missing[:10]:
    print(f'  - {f}')
"
```

### 2. Choose Enrichment Strategy

**Option A: Batch Process All Documents**

Use when processing the entire corpus or re-enriching after markdown changes:

```bash
python src/fragile/proofs/tools/source_location_enricher.py batch \
    docs/source/ \
    --types theorems definitions axioms
```

**Option B: Single Document**

Use when working on one document:

```bash
python src/fragile/proofs/tools/source_location_enricher.py directory \
    docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data \
    docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
    01_fragile_gas_framework
```

**Option C: Single Entity**

Use for one-off enrichment or manual fixes:

```bash
python src/fragile/proofs/tools/source_location_enricher.py single \
    docs/source/.../raw_data/theorems/thm-keystone.json \
    docs/source/.../03_cloning.md \
    03_cloning
```

**Option D: Manual Lookup**

Use the interactive CLI tool when you need to find a specific location manually:

```bash
# Find text snippet
python src/tools/find_source_location.py find-text \
    docs/source/1_euclidean_gas/03_cloning.md \
    "The Keystone Principle states" \
    --document-id 03_cloning

# Find by directive label
python src/tools/find_source_location.py find-directive \
    docs/source/1_euclidean_gas/03_cloning.md \
    thm-keystone \
    --document-id 03_cloning
```

### 3. Validation

After enrichment, validate that source locations are correct:

```python
# Validation script
import json
from pathlib import Path
from fragile.proofs.tools.line_finder import extract_lines, validate_line_range

def validate_enriched_entity(json_file: Path, markdown_file: Path):
    """Validate that source location points to correct text."""
    entity = json.loads(json_file.read_text())
    markdown_content = markdown_file.read_text()

    if "source_location" not in entity:
        print(f"❌ {json_file.name}: No source_location")
        return False

    loc = entity["source_location"]

    # Check line range
    if "line_range" in loc and loc["line_range"]:
        start, end = loc["line_range"]
        max_lines = len(markdown_content.splitlines())

        if not validate_line_range((start, end), max_lines):
            print(f"❌ {json_file.name}: Invalid line range {start}-{end}")
            return False

        # Extract text and check if it matches entity content
        extracted = extract_lines(markdown_content, (start, end))

        # Check if key text is present
        if "full_statement_text" in entity:
            key_text = entity["full_statement_text"][:100]
            if key_text.lower() not in extracted.lower():
                print(f"⚠️  {json_file.name}: Text mismatch at lines {start}-{end}")
                return False

        print(f"✓ {json_file.name}: Valid line range {start}-{end}")
        return True
    else:
        print(f"⚠️  {json_file.name}: No line range (fallback mode)")
        return True

# Run validation on a directory
raw_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data")
markdown_file = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework.md")

for entity_type in ["theorems", "definitions", "axioms"]:
    entity_dir = raw_dir / entity_type
    if entity_dir.exists():
        print(f"\n{entity_type.upper()}:")
        for json_file in entity_dir.glob("*.json"):
            validate_enriched_entity(json_file, markdown_file)
```

### 4. Report Coverage Statistics

Generate a report of enrichment coverage:

```python
import json
from pathlib import Path
from collections import defaultdict

def report_enrichment_coverage(docs_source_dir: Path):
    """Generate enrichment coverage report."""
    stats = defaultdict(lambda: {"total": 0, "with_line_range": 0, "with_directive": 0, "section_only": 0, "minimal": 0})

    for raw_dir in docs_source_dir.glob("*/*/raw_data"):
        document_id = raw_dir.parent.name

        for entity_type_dir in raw_dir.iterdir():
            if not entity_type_dir.is_dir():
                continue

            entity_type = entity_type_dir.name

            for json_file in entity_type_dir.glob("*.json"):
                stats[document_id]["total"] += 1

                data = json.loads(json_file.read_text())

                if "source_location" not in data:
                    stats[document_id]["minimal"] += 1
                    continue

                loc = data["source_location"]

                if "line_range" in loc and loc["line_range"]:
                    stats[document_id]["with_line_range"] += 1
                elif "directive_label" in loc and loc["directive_label"]:
                    stats[document_id]["with_directive"] += 1
                elif "section" in loc and loc["section"]:
                    stats[document_id]["section_only"] += 1
                else:
                    stats[document_id]["minimal"] += 1

    # Print report
    print("\nSOURCE LOCATION ENRICHMENT COVERAGE REPORT")
    print("="*70)
    print(f"{'Document':<30} {'Total':<8} {'Line Range':<12} {'Directive':<12} {'Section':<10} {'Minimal':<10}")
    print("-"*70)

    total_all = sum(s["total"] for s in stats.values())
    total_line_range = sum(s["with_line_range"] for s in stats.values())

    for doc_id in sorted(stats.keys()):
        s = stats[doc_id]
        print(f"{doc_id:<30} {s['total']:<8} {s['with_line_range']:<12} {s['with_directive']:<12} {s['section_only']:<10} {s['minimal']:<10}")

    print("-"*70)
    coverage = 100 * total_line_range / total_all if total_all > 0 else 0
    print(f"{'TOTAL':<30} {total_all:<8} {total_line_range:<12} - Line range coverage: {coverage:.1f}%")

# Run report
report_enrichment_coverage(Path("docs/source"))
```

## Common Issues and Troubleshooting

### Issue 1: Text Not Found

**Symptom**: Enricher reports "text not found" for entity

**Causes**:
- Text was modified since extraction
- Search text is too long (truncation issues)
- Special characters or LaTeX formatting mismatch

**Solutions**:
1. Use shorter search snippet (enricher already truncates to 200 chars)
2. Try directive label matching instead
3. Manual lookup with interactive CLI tool
4. Fall back to section-level precision

### Issue 2: Multiple Matches

**Symptom**: Text appears multiple times in document

**Causes**:
- Common terms or phrases
- Repeated theorem statements

**Solutions**:
1. Add more context to search text
2. Use directive label if available
3. Manual disambiguation with CLI tool's `find-all-occurrences`

### Issue 3: Line Range Out of Bounds

**Symptom**: Validation fails with "line range out of bounds"

**Causes**:
- Markdown file was modified after enrichment
- Incorrect line number calculation

**Solutions**:
1. Re-run enrichment on the current markdown
2. Check if file path is correct
3. Validate line range with `validate_line_range()`

### Issue 4: Performance on Large Corpus

**Symptom**: Batch enrichment is slow

**Optimization**:
- Process in parallel (add threading to enricher)
- Cache markdown file reads
- Skip already-enriched entities (check for existing source_location)

## Integration with Other Skills

- **extract-and-refine**: Source location enrichment runs as Stage 1.75 (after extraction, before refinement)
- **validate-refinement**: Validation checks include source location completeness
- **framework-consistency**: Source locations enable verification against source documents
- **registry-management**: Registry builders use enriched source locations for glossary generation

## Expected Output

After successful enrichment, each raw JSON file will have a `source_location` field:

```json
{
  "temp_id": "raw-thm-1",
  "label_text": "Theorem 3.2 (Keystone Principle)",
  "statement_type": "theorem",
  "full_statement_text": "Let v > 0 and assume the walker velocity...",
  "source_section": "§3.2",
  "source_location": {
    "document_id": "03_cloning",
    "file_path": "docs/source/1_euclidean_gas/03_cloning.md",
    "line_range": [142, 158],
    "directive_label": "thm-keystone",
    "section": "§3.2",
    "url_fragment": "#thm-keystone"
  }
}
```

Precision levels (in order of preference):
1. **Line range** (most precise): `line_range: [142, 158]`
2. **Directive label** (precise): `directive_label: "thm-keystone"`, `url_fragment: "#thm-keystone"`
3. **Section** (less precise): `section: "§3.2"`
4. **Minimal** (least precise): Just `document_id` and `file_path`

## Success Criteria

✓ All entities have `source_location` field
✓ >90% have line_range (most precise level)
✓ Line ranges validated against markdown files
✓ Text at line range matches entity content
✓ No "out of bounds" line ranges
✓ Fallback strategies work for edge cases

## Notes

- Source location enrichment is **deterministic** (no LLM involved)
- Can be re-run safely (idempotent)
- Does not modify markdown files (read-only)
- Fast (processes ~100 entities/second)
- Designed for both pipeline automation and manual use
