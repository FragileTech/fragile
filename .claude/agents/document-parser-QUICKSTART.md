# Document Parser Agent - Quick Start Guide

## Simplest Usage (Copy-Paste Ready)

### Single Document Parsing

Just paste this into Claude:

```
Load the document-parser agent from .claude/agents/document-parser.md
and parse:

Document: docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

### Quick Validation (No LLM, Fast)

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/03_cloning.md
No LLM: true
```

### Complete Directory

```
Load document-parser agent.

Parse directory: docs/source/1_euclidean_gas/
Mode: sketch
```

---

## Direct Python Invocation (Recommended)

For faster execution, you can call the parser directly via Python:

### Single Document
```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md
```

### With Options
```bash
# Fast mode (no LLM)
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm

# Sketch mode only
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --mode sketch

# Custom output directory
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --output-dir custom/path/
```

### Directory Processing
```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/
```

---

## What Happens

When you run the parser, it executes 8 phases:

1. **Extract MyST Directives** (~2 sec)
   - Finds all `{prf:definition}`, `{prf:theorem}`, etc.
   - Reports counts: "Found 119 directives: {definition: 36, ...}"

2. **Create Mathematical Objects** (~3 sec)
   - Transforms definitions → MathematicalObject instances
   - Auto-detects types (SET, FUNCTION, MEASURE, etc.)
   - Normalizes labels (lowercase, fix special chars)

3. **Create Theorems** (~3 sec)
   - Transforms theorems/lemmas → TheoremBox instances
   - Extracts axioms → Axiom instances
   - Validates label format

4. **Extract Relationships** (~10-30 sec if LLM enabled)
   - Parses `{prf:ref}` directives
   - Uses Gemini 2.5 Pro to infer implicit dependencies (if LLM enabled)

5. **Create Proof Sketches** (~5 sec)
   - Parses `{prf:proof}` blocks
   - Creates ProofBox structures with SKETCHED steps

6. **Expand Proofs** (~30-120 sec if LLM enabled)
   - Uses Gemini 2.5 Pro to expand proof steps
   - Fills mathematical derivations

7. **Validate** (~1 sec)
   - Checks schema compliance
   - Reports: "Validation complete: 0 errors, 0 warnings"

8. **Export to JSON** (~1 sec)
   - Writes extraction_inventory.json (complete catalog)
   - Writes statistics.json (summary metrics)

**Total Time**:
- No LLM: ~15 seconds
- With LLM (sketch): ~30-60 seconds
- With LLM (both): ~2-5 minutes

---

## Expected Output

### Console Output

```
🚀 MathDocumentParser Starting
   Source: docs/source/1_euclidean_gas/03_cloning.md
   Mode: both
   Output: docs/source/1_euclidean_gas/03_cloning/data

📄 Processing: 03_cloning.md
  Phase 1: Extracting MyST directives...
    ✓ Found 119 directives
      {'definition': 36, 'axiom': 6, 'proposition': 12, 'lemma': 32, ...}
  Phase 2: Creating mathematical objects...
    ✓ Created 36 objects
  Phase 3: Creating theorems...
    ✓ Created 59 theorems
  Phase 4: Extracting relationships...
    ✓ Created 0 relationships
  Phase 5: Creating proof sketches...
    ✓ Created 0 sketch proofs
  Phase 7: Validating...
    ✓ Validation complete
      Errors: 0
      Warnings: 0
  Phase 8: Exporting to JSON...
    ✓ Exported to docs/source/1_euclidean_gas/03_cloning/data

✅ Processing complete!
   Objects: 36
   Theorems: 59
   Proofs: 0
   Relationships: 0
```

### Files Created

```bash
docs/source/1_euclidean_gas/03_cloning/data/
├── extraction_inventory.json  # Complete directive catalog (71KB)
└── statistics.json            # Summary metrics (160B)
```

### Check Results

```bash
# View statistics
cat docs/source/1_euclidean_gas/03_cloning/data/statistics.json
```

**Expected Output:**
```json
{
  "objects_created": 36,
  "theorems_created": 59,
  "proofs_created": 0,
  "relationships_created": 0,
  "validation_errors": 0,
  "validation_warnings": 0
}
```

**Success Criteria**: `validation_errors: 0`

### Validate Extracted Objects (Pydantic)

After extraction, validate objects against Pydantic schemas:

```bash
# Quick validation script
python -c "
import json
from fragile.proofs import MathematicalObject
from pydantic import ValidationError

# Load extracted data
with open('docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json') as f:
    data = json.load(f)

# Validate first object
if data['directives']:
    obj_data = data['directives'][0]
    try:
        obj = MathematicalObject.model_validate(obj_data)
        print(f'✓ Validated: {obj.object_id}')
    except ValidationError as e:
        print(f'✗ Validation failed:')
        for err in e.errors():
            print(f'  {err[\"loc\"]}: {err[\"msg\"]}')
"
```

**Expected Output**:
```
✓ Validated: obj-single-swarm-space
```

**If validation fails**:
```
✗ Validation failed:
  ('object_type',): Input should be 'set', 'function', 'measure', ...
```

Check `validation_errors.json` for full details.

---

## Common Use Cases

### Use Case 1: Quick Document Structure Analysis

**Goal**: See what's in a document without LLM overhead

```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/06_convergence.md --no-llm
```

**Result**: 15-second extraction, see directive counts and types

### Use Case 2: Full Document Compilation

**Goal**: Create complete structured representation for downstream processing

```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --mode both
```

**Result**: 3-5 minute processing, all content extracted and validated

### Use Case 3: Batch Chapter Processing

**Goal**: Process entire chapter directory

```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/
```

**Result**: Each document gets own `data/` subdirectory with JSON files

### Use Case 4: Incremental Document Updates

**Goal**: Re-parse single modified document

```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm
```

**Result**: Fast 15-second re-extraction, updated JSON files

---

## Real Example

Let's parse the cloning document:

```bash
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm
```

**What happens**:
1. Parser reads 03_cloning.md (~400KB)
2. Finds 119 MyST directives using regex
3. Creates 36 MathematicalObject instances (from definitions)
4. Creates 59 TheoremBox/Axiom instances (from theorems, lemmas, axioms)
5. Validates all instances against Pydantic schema
6. Exports to `docs/source/1_euclidean_gas/03_cloning/data/`

**Time**: ~15 seconds

**Result**:
```json
{
  "objects_created": 36,
  "theorems_created": 59,
  "validation_errors": 0
}
```

---

## Checking Extracted Content

### View First 5 Directives

```bash
cat docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json | python -m json.tool | head -100
```

### View Directive Types

```bash
cat docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json | python -m json.tool | grep '"type"'
```

### Count Theorems

```bash
cat docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json | python -m json.tool | grep '"type": "theorem"' | wc -l
```

### Find Specific Theorem

```bash
cat docs/source/1_euclidean_gas/03_cloning/data/extraction_inventory.json | python -m json.tool | grep -A 10 '"label": "thm-keystone-principle"'
```

---

## Integration Examples

### Example 1: Parse → Analyze Structure

```bash
# Step 1: Parse document
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/09_kl_convergence.md --no-llm

# Step 2: View statistics
cat docs/source/1_euclidean_gas/09_kl_convergence/data/statistics.json

# Step 3: Extract theorem labels
grep -o '"label": "thm-[^"]*"' docs/source/1_euclidean_gas/09_kl_convergence/data/extraction_inventory.json
```

### Example 2: Parse → Sketch Proofs

```bash
# Step 1: Parse to get theorem labels
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/09_kl_convergence.md --no-llm

# Step 2: Use proof-sketcher on extracted theorems
# (Load proof-sketcher agent)
# Sketch proof for: thm-kl-convergence-euclidean
```

### Example 3: Batch Processing

```bash
# Process all documents in chapter
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/ --no-llm

# View aggregated statistics
for dir in docs/source/1_euclidean_gas/*/data; do
  echo "=== $(dirname $dir) ==="
  cat "$dir/statistics.json"
done
```

---

## Command Line Options Reference

```
python -m fragile.agents.math_document_parser [SOURCE] [OPTIONS]

Arguments:
  SOURCE              Path to document or directory (required)

Options:
  --mode MODE         Processing mode: sketch | expand | both (default: both)
  --no-llm            Disable LLM processing (faster)
  --output-dir DIR    Custom output directory (default: auto-detected)
  -h, --help          Show help message
```

### Examples

```bash
# Minimal (default mode=both, LLM enabled)
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md

# Fast validation (no LLM)
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm

# Sketch only (no proof expansion)
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --mode sketch

# Custom output
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --output-dir /tmp/analysis/

# Directory processing
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/
```

---

## Parallel Processing Example

Process 3 documents simultaneously:

```bash
# Terminal 1
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm &

# Terminal 2
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/04_convergence.md --no-llm &

# Terminal 3
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/05_mean_field.md --no-llm &

# Wait for all to complete
wait
```

All 3 will complete in ~15 seconds (parallel execution).

---

## Tips

1. **Start with --no-llm** for quick validation (~15 sec vs ~3 min)

2. **Check validation_errors first** in statistics.json:
   ```bash
   cat docs/source/.../data/statistics.json | grep validation_errors
   ```
   Should be: `"validation_errors": 0`

3. **Use extraction_inventory.json** to understand document structure:
   - See all directive types
   - Check cross-references
   - Verify labels are correct

4. **For large documents** (>500KB), always use `--no-llm` initially

5. **Processing directories**: Use `--mode sketch` or `--no-llm` for speed

6. **Debugging**: Check line_range in extraction_inventory.json to locate directives in source

---

## Next Steps After Parsing

1. **Review statistics.json**: Verify counts match expectations

2. **Check extraction_inventory.json**: Browse extracted content

3. **Validate results**: Ensure `validation_errors: 0`

4. **Use downstream agents**:
   - **Proof Sketcher**: Generate proof strategies for extracted theorems
   - **Theorem Prover**: Expand proof sketches to full proofs
   - **Math Reviewer**: Review mathematical rigor

5. **Query registry**: Use MathematicalRegistry to programmatically access extracted objects

---

## Troubleshooting Quick Fixes

### Problem: Pydantic ValidationError

**Quick Fix**: Check which field is invalid
```bash
python -c "
from fragile.proofs import MathematicalObject, ObjectType
print('Valid ObjectType values:', list(ObjectType))
print('Valid label pattern: ^obj-[a-z0-9-]+$')
print('Example valid object_id: obj-euclidean-gas')
"
```

**Common Issues**:
- Invalid enum: Use `ObjectType.SPACE` not `"Space"` (case-sensitive)
- Invalid label: Must be lowercase kebab-case (e.g., `obj-my-object`)
- Missing field: Check Pydantic schema with `.model_json_schema()`

### Problem: Found 0 directives

**Quick Fix**: Check directive syntax in document
```bash
grep -n "^:::" docs/source/1_euclidean_gas/03_cloning.md | head -20
```
Should show `:::` (3 colons), not `::::` (4 colons)

### Problem: Validation errors

**Quick Fix**: Check error messages in console output
```bash
python -m fragile.agents.math_document_parser docs/source/... 2>&1 | grep "Failed to create"
```
Parser auto-normalizes most label issues, but review failures.

### Problem: Slow processing

**Quick Fix**: Disable LLM
```bash
python -m fragile.agents.math_document_parser docs/source/... --no-llm
```
Reduces time from ~3 min to ~15 sec.

---

## Comparison: Agent vs Manual Extraction

| Feature | Manual Extraction | Document Parser |
|---------|------------------|----------------|
| **Time** | Hours per document | 15 sec - 3 min |
| **Coverage** | Easy to miss directives | All directives found |
| **Validation** | Manual schema checks | Automatic Pydantic validation |
| **Normalization** | Manual label fixes | Auto-normalized labels |
| **Cross-refs** | Manual tracking | Automatic extraction |
| **Export** | Custom scripts | Structured JSON |
| **Updates** | Re-do manually | Re-run parser |

---

That's it! Just copy-paste the simple commands above to get started.

For more details, see:
- Full agent definition: `.claude/agents/document-parser.md`
- Complete docs: `.claude/agents/document-parser-README.md`
- Framework context: `CLAUDE.md`
