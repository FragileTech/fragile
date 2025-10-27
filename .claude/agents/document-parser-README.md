# Document Parser Agent - Documentation

**Version**: 1.0
**Created**: 2025-10-26
**Purpose**: Extract and validate all mathematical content from MyST markdown documents

---

## Overview

The **Document Parser** is an autonomous agent that extracts all mathematical content from Jupyter Book MyST markdown documents and transforms it into structured, validated JSON following the `fragile.proofs` type system.

### Key Distinction

| Feature | Document Parser | Proof Sketcher | Math Reviewer |
|---------|----------------|----------------|---------------|
| **Input** | MyST markdown documents | Theorem statements | Existing proofs |
| **Goal** | Extract & structure | Create proof strategies | Find errors |
| **Output** | JSON (objects, theorems, proofs) | Proof sketches | Issue reports |
| **Verification** | Schema validation | Framework consistency | Logical rigor |
| **Use Case** | Document compilation | Proof planning | Quality control |
| **When to Use** | Initial document processing | Before writing proofs | After writing proofs |

---

## Core Capabilities

### MyST Directive Extraction
- Parses all `{prf:definition}`, `{prf:theorem}`, `{prf:lemma}`, `{prf:proposition}`, `{prf:axiom}`, `{prf:proof}` blocks
- Extracts titles, labels, content, and mathematical expressions
- Builds complete directive inventory with line ranges

### Mathematical Object Creation
- Transforms definitions → `MathematicalObject` instances
- Infers object types (SET, FUNCTION, MEASURE, SPACE, DISTRIBUTION, etc.)
- Extracts tags from content (euclidean-gas, discrete, continuous, etc.)
- Auto-normalizes labels (lowercase, replace special chars)

### Theorem Extraction
- Converts theorems/lemmas/propositions → `TheoremBox` instances
- Extracts axioms → `Axiom` instances
- Infers theorem output types (PROPERTY, EQUIVALENCE, CONVERGENCE, etc.)
- Validates label format and cross-references

### Relationship Inference (Hybrid)
- **Explicit**: Extracts cross-references from `{prf:ref}` directives
- **LLM-Assisted** (optional): Uses Gemini 2.5 Pro to infer implicit dependencies
- Creates `Relationship` instances with bidirectionality tracking

### Proof Parsing
- Parses `{prf:proof}` blocks into `ProofBox` structures
- Creates SKETCHED proof steps
- LLM expansion to EXPANDED steps (optional)
- Validates proof dataflow consistency

### Validation & Export
- Pydantic schema validation for all types
- Cross-reference integrity checking
- JSON export to structured directories
- MathematicalRegistry population

---

## How to Use

### Method 1: Single Document

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

### Method 2: Directory Processing

```
Load document-parser agent.

Parse directory: docs/source/1_euclidean_gas/
```

### Method 3: Quick Validation (No LLM)

```
Load document-parser agent.

Parse: docs/source/2_geometric_gas/11_geometric_gas.md
No LLM: true
```

### Method 4: Custom Output Directory

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/05_mean_field.md
Output: custom/analysis/path/
Mode: sketch
```

---

## Input Parameters

### Required
- **source**: Path to document or directory
  - Example: `docs/source/1_euclidean_gas/03_cloning.md`
  - Example: `docs/source/1_euclidean_gas/` (directory)

### Optional
- **mode**: Processing mode
  - `sketch`: Parse objects/theorems, create proof sketches (no expansion)
  - `expand`: Full proof expansion with LLM (slower)
  - `both`: Sketch + expand (default)

- **no_llm**: Disable LLM processing (faster, but misses relationship inference)
  - `true`: Skip LLM phases (Phases 4, 6)
  - `false`: Use LLM (default)

- **output_dir**: Custom output directory
  - Default: Auto-detected from source path (`docs/source/N_chapter/document/data/`)
  - Custom: Any valid directory path

---

## Output Format

### Directory Structure

After processing `docs/source/1_euclidean_gas/03_cloning.md`:

```
docs/source/1_euclidean_gas/
├── 03_cloning.md              # Original document
└── 03_cloning/                # Auto-created output directory
    └── data/                  # Structured JSON outputs
        ├── extraction_inventory.json  # Complete directive catalog (large)
        └── statistics.json            # Summary metrics
```

### extraction_inventory.json Structure

```json
{
  "source_file": "docs/source/1_euclidean_gas/03_cloning.md",
  "total_directives": 119,
  "counts_by_type": {
    "definition": 36,
    "axiom": 6,
    "proposition": 12,
    "lemma": 32,
    "corollary": 6,
    "theorem": 15,
    "remark": 12
  },
  "directives": [
    {
      "type": "definition",
      "label": "def-single-swarm-space",
      "title": "Single-Walker and Swarm State Spaces",
      "content": "1. A **walker** is a tuple $(x, s)$...",
      "math_expression_count": 9,
      "first_math": "S := \\left( (x_1, v_1, s_1), ... \\right)",
      "cross_refs": [],
      "line_range": [108, 130]
    }
  ]
}
```

**Fields Explained**:
- `type`: Directive type (definition, theorem, lemma, axiom, proof, etc.)
- `label`: Normalized label (lowercase, special chars replaced)
- `title`: Human-readable title
- `content`: Full content (truncated to 200 chars in JSON)
- `math_expression_count`: Number of LaTeX expressions found
- `first_math`: Primary mathematical expression (useful for quick reference)
- `cross_refs`: List of labels referenced via `{prf:ref}`
- `line_range`: [start, end] line numbers in source document

### statistics.json Structure

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

**Metrics Explained**:
- `objects_created`: Number of `MathematicalObject` instances (from definitions)
- `theorems_created`: Number of `TheoremBox` + `Axiom` instances
- `proofs_created`: Number of `ProofBox` instances (if proof phases active)
- `relationships_created`: Number of `Relationship` instances (if LLM enabled)
- `validation_errors`: **Should be 0** - schema validation failures
- `validation_warnings`: Non-critical issues (deprecated patterns, etc.)

---

## Processing Phases

The Document Parser executes in 8 distinct phases:

### Phase 1: MyST Directive Extraction (~2 seconds)
- Parses document with regex for `{prf:...}` blocks
- Creates `DocumentInventory` with full indexing
- Reports counts by type

**Output**: "Found 119 directives: {definition: 36, axiom: 6, ...}"

### Phase 2: Mathematical Object Creation (~3 seconds)
- Transforms `{prf:definition}` → `MathematicalObject`
- Infers object types from keywords
- Extracts tags from content
- Normalizes labels
- Validates against Pydantic schema

**Output**: "Created 36 objects"

### Phase 3: Theorem Creation (~3 seconds)
- Transforms theorems/lemmas/propositions → `TheoremBox`
- Extracts axioms → `Axiom`
- Infers output types
- Validates labels and cross-refs

**Output**: "Created 59 theorems"

### Phase 4: Relationship Extraction (~10-30 seconds if LLM enabled)
- **Explicit**: Parses `{prf:ref}` directives
- **LLM** (if enabled): Gemini 2.5 Pro infers implicit dependencies
- Creates `Relationship` instances

**Output**: "Created N relationships"

### Phase 5: Proof Sketch Creation (~5 seconds)
- Parses `{prf:proof}` directives
- Creates `ProofBox` with SKETCHED steps
- Maps inputs/outputs to properties

**Output**: "Created N sketch proofs"

### Phase 6: Proof Expansion (~30-120 seconds if LLM enabled)
- Uses Gemini 2.5 Pro to expand SKETCHED → EXPANDED
- Fills mathematical derivations
- Adds techniques and references

**Output**: "Expanded N proofs"

### Phase 7: Validation (~1 second)
- Checks all Pydantic constraints
- Validates label uniqueness
- Checks cross-reference integrity
- Reports errors and warnings

**Output**: "Validation complete: 0 errors, 0 warnings"

### Phase 8: Export to JSON (~1 second)
- Exports to output directory
- Creates `extraction_inventory.json` and `statistics.json`
- Updates MathematicalRegistry

**Output**: "Exported to docs/source/.../data/"

**Total Time**:
- **No LLM**: ~15-20 seconds
- **With LLM (mode=sketch)**: ~30-60 seconds
- **With LLM (mode=both)**: ~2-5 minutes

---

## Example Usage Scenarios

### Scenario 1: Initial Document Analysis

**Goal**: Understand structure of new mathematical document

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/06_convergence.md
No LLM: true
```

**Result**:
- Quick extraction (~15 seconds)
- See what directives exist
- Get counts of theorems, definitions, etc.
- Review `extraction_inventory.json` for content overview

### Scenario 2: Full Document Compilation

**Goal**: Create complete structured representation for downstream processing

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/03_cloning.md
Mode: both
```

**Result**:
- Full processing (~3-5 minutes)
- All objects, theorems, relationships extracted
- Proofs sketched and expanded
- Ready for proof-sketcher or theorem-prover agents

### Scenario 3: Batch Chapter Processing

**Goal**: Process entire chapter at once

```
Load document-parser agent.

Parse directory: docs/source/1_euclidean_gas/
Mode: sketch
```

**Result**:
- Processes all `.md` files in directory
- Each document gets own `data/` subdirectory
- Aggregated statistics available

### Scenario 4: Incremental Updates

**Goal**: Re-parse single modified document

```
Load document-parser agent.

Parse: docs/source/1_euclidean_gas/03_cloning.md
No LLM: true
```

**Result**:
- Fast re-extraction (~15 seconds)
- Updated JSON files
- Registry automatically updated with changes

---

## Schema Compatibility and Validation

### Pydantic Schema Validation

The Document Parser validates **ALL** extracted objects against Pydantic models from `src/fragile/proofs/` to ensure 100% data pipeline compatibility.

**Available Schemas**:
- `src/fragile/proofs/llm_schemas.json` - Full schema (76 models)
- `src/fragile/proofs/llm_proof.json` - Rigorous proof schema (32 models)
- `src/fragile/proofs/llm_sketch.json` - Proof sketch schema (23 models)

**Validation Process**:
1. After Phase 2: Validate `MathematicalObject` instances
2. After Phase 3: Validate `TheoremBox` and `Axiom` instances
3. After Phase 4: Validate `Relationship` instances
4. After Phase 5-6: Validate `ProofBox` instances
5. Phase 7: Final validation of ALL instances

**Example Validation**:
```python
from fragile.proofs import MathematicalObject, ObjectType
from pydantic import ValidationError

# Validate extracted object
try:
    obj = MathematicalObject.model_validate({
        "object_id": "obj-euclidean-gas",
        "object_type": ObjectType.SPACE,
        "description": "Euclidean gas state space",
        "mathematical_definition": "...",
        "tags": ["euclidean-gas"],
        "properties": []
    })
    print(f"✓ Valid: {obj.object_id}")
except ValidationError as e:
    print(f"✗ Invalid: {e}")
```

**Validation Output**:
- `statistics.json`: Includes `validation_details` with per-model counts
- `validation_errors.json`: Created if any validation failures occur
- Detailed field-level error messages for debugging

---

## Validation and Error Handling

### Automatic Label Normalization

The parser automatically fixes common label issues:

**Issue**: Label contains uppercase
```
Label: "lem-V-coercive"
```
**Fix**: Converted to lowercase
```
Normalized: "lem-v-coercive"
```

**Issue**: Label contains colons or underscores
```
Label: "axiom-ax:domain-regularity"
Label: "lem-v_varx-implies-variance"
```
**Fix**: Special chars replaced with hyphen
```
Normalized: "axiom-ax-domain-regularity"
Normalized: "lem-v-varx-implies-variance"
```

**Issue**: Missing prefix
```
Label: "keystone-principle"
```
**Fix**: Prefix added based on type
```
Normalized: "thm-keystone-principle" (if theorem)
Normalized: "obj-keystone-principle" (if definition)
```

### Validation Checklist

After parsing, the agent verifies:

- ✅ **Label Format**: All labels match pattern (e.g., `^obj-[a-z0-9-]+$`)
- ✅ **Label Uniqueness**: No duplicate labels within document
- ✅ **Cross-Reference Integrity**: All `{prf:ref}` point to valid labels
- ✅ **Type Consistency**: Objects classified correctly
- ✅ **Math Expression Extraction**: LaTeX properly identified
- ✅ **Pydantic Schema**: All instances pass validation

**Success Criteria**: `validation_errors: 0` in `statistics.json`

---

## Integration with Other Agents

### Workflow 1: Parse → Sketch → Prove → Review

```bash
# Step 1: Parse document
Load document-parser agent.
Parse: docs/source/1_euclidean_gas/09_kl_convergence.md

# Step 2: Generate proof sketch
Load proof-sketcher agent.
Sketch proof for: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md

# Step 3: Expand proof
Load theorem-prover agent.
Expand proof sketch for: thm-kl-convergence-euclidean

# Step 4: Review proof
Load math-reviewer agent.
Review proof: docs/source/1_euclidean_gas/proofs/proof_thm_kl_convergence.md
```

### Workflow 2: Batch Processing → Analysis

```bash
# Step 1: Parse entire chapter
Load document-parser agent.
Parse directory: docs/source/1_euclidean_gas/

# Step 2: Analyze structure
# Read all statistics.json files
# Identify documents with most theorems
# Identify documents with missing proofs

# Step 3: Prioritize proof development
# Use proof-sketcher on high-priority theorems
```

### Workflow 3: Incremental Document Updates

```bash
# User edits: docs/source/1_euclidean_gas/03_cloning.md
# Adds new theorem: thm-new-result

# Step 1: Re-parse document
Load document-parser agent.
Parse: docs/source/1_euclidean_gas/03_cloning.md
No LLM: true

# Step 2: Verify new theorem extracted
# Check statistics.json: theorems_created increased?
cat docs/source/1_euclidean_gas/03_cloning/data/statistics.json

# Step 3: Sketch new theorem
Load proof-sketcher agent.
Sketch proof for: thm-new-result
```

---

## Advanced Usage

### Custom Object Type Hints

Guide object type inference by including keywords in content:

```markdown
:::{prf:definition} Heat Kernel
:label: def-heat-kernel

We define the **heat kernel** as a **function** $p_t(x, y): \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ that satisfies...
:::
```

Parser recognizes "function" → `ObjectType.FUNCTION`

**Supported Keywords**:
- `function`, `map`, `operator` → `ObjectType.FUNCTION`
- `measure`, `probability` → `ObjectType.MEASURE`
- `space`, `manifold` → `ObjectType.SPACE`
- `distribution`, `density` → `ObjectType.DISTRIBUTION`
- `metric`, `distance` → `ObjectType.METRIC`
- `process`, `evolution` → `ObjectType.PROCESS`
- Default: `ObjectType.SET`

### Explicit Relationship Declarations

Use cross-references to create explicit relationships:

```markdown
:::{prf:theorem} Main Result
:label: thm-main

Using {prf:ref}`lem-technical-bound` and {prf:ref}`prop-compactness`, we conclude...
:::
```

Parser creates:
- `Relationship(source="thm-main", target="lem-technical-bound")`
- `Relationship(source="thm-main", target="prop-compactness")`

### Proof Sketch Structure

Structure proofs for better parsing:

```markdown
:::{prf:proof} of {prf:ref}`thm-main`
:label: proof-thm-main

**Strategy**: Use contraction mapping argument.

**Step 1**: Establish uniform bound using {prf:ref}`lem-bound`.

**Step 2**: Show contraction via Lipschitz property.

**Step 3**: Apply Banach fixed point theorem.
:::
```

Parser extracts:
- Strategy: "Use contraction mapping argument"
- Steps: 3 steps with descriptions
- Dependencies: `lem-bound`

---

## Performance Optimization

### File Size Guidelines

| Document Size | Recommended Mode | Expected Time |
|---------------|------------------|---------------|
| <100KB | `mode=both` | ~1-2 min |
| 100KB-500KB | `mode=sketch` or `--no-llm` | ~30-60 sec |
| >500KB | `--no-llm` | ~15-30 sec |

### Parallel Processing

Process multiple documents simultaneously:

```
Launch 3 document-parser agents in parallel:

Agent 1: Parse docs/source/1_euclidean_gas/03_cloning.md
Agent 2: Parse docs/source/1_euclidean_gas/04_convergence.md
Agent 3: Parse docs/source/1_euclidean_gas/05_mean_field.md

All with mode=sketch, no-llm=true for speed
```

### Memory Usage

- **Single document**: ~50-100MB peak memory
- **Directory**: ~100-200MB peak memory (processes sequentially)
- **Parallel instances**: ~100MB per instance

Safe to run 5-10 instances in parallel on 8GB RAM.

---

## Troubleshooting

### Problem: Pydantic ValidationError

**Symptom**: `ValidationError: 1 validation error for MathematicalObject`

**Common Causes**:
1. **Invalid enum value**: Using wrong value for ObjectType, TheoremOutputType, etc.
   ```python
   # Check valid values
   from fragile.proofs import ObjectType
   print(list(ObjectType))
   ```

2. **Missing required field**: Required field not included in extracted data
   ```python
   # Check schema for required fields
   from fragile.proofs import MathematicalObject
   print(MathematicalObject.model_json_schema())
   ```

3. **Label pattern violation**: Label doesn't follow naming convention
   ```python
   # Fix: Use lowercase kebab-case
   "obj-euclidean-gas-discrete"  # ✓ Valid
   "obj-Euclidean_Gas:Discrete"  # ✗ Invalid
   ```

**Solution**: Check `validation_errors.json` for detailed error messages

### Problem: No Directives Found

**Symptom**: `Found 0 directives`

**Cause**: Document doesn't use MyST format or uses incorrect syntax

**Solution**:
1. Check directive format: should use `:::` (3 colons), not `::::` (4 colons)
2. Verify labels are on separate lines:
   ```markdown
   :::{prf:definition} Title
   :label: def-my-label
   ```

### Problem: Validation Errors

**Symptom**: `validation_errors: 7` in statistics.json

**Cause**: Labels contain invalid characters or don't match pattern

**Solution**:
- Parser auto-normalizes labels, but check for edge cases
- Review error messages in console output
- Manually inspect `extraction_inventory.json` for problematic labels

### Problem: Wrong Object Types

**Symptom**: All objects classified as `SET`

**Cause**: Content lacks type-specific keywords

**Solution**:
- Add type hints in definition content (e.g., "function", "measure")
- Or manually specify after parsing

### Problem: Missing Cross-References

**Symptom**: `relationships_created: 0` when references clearly exist

**Cause**: LLM disabled or cross-refs not in `{prf:ref}` format

**Solution**:
- Run without `--no-llm` flag
- Use proper MyST syntax: `{prf:ref}\`label\``

### Problem: Slow Processing

**Symptom**: Takes >5 minutes for single document

**Cause**: LLM phases active on large document

**Solution**:
- Use `--no-llm` for faster processing
- Use `mode=sketch` to skip expansion
- Process in parallel if handling multiple documents

---

## Best Practices

1. **Always check validation_errors**: Should be zero after successful parse

2. **Use --no-llm for quick validation**: Speeds up testing of document structure

3. **Review extraction_inventory.json**: Verify all directives were captured correctly

4. **Check cross_refs field**: Identifies theorem dependencies for proof ordering

5. **Monitor line_range**: Helps locate issues in source documents

6. **Use mode=sketch first**: Test parsing before expensive LLM expansion

7. **Process directories incrementally**: Parse one document at a time to identify issues early

8. **Commit JSON to git**: Track changes to mathematical content structure over time

9. **Use MathematicalRegistry**: Query extracted objects programmatically

10. **Integrate with CI/CD**: Run parser on every commit to detect broken cross-refs

---

## Model Configuration

### LLM Usage

The Document Parser uses Gemini 2.5 Pro for:

**Phase 4**: Relationship Inference
- Analyzes theorem statements
- Infers implicit dependencies
- Creates Relationship instances

**Phase 6**: Proof Expansion
- Expands SKETCHED steps to EXPANDED
- Fills mathematical derivations
- Adds techniques and justifications

**Model**: `gemini-2.5-pro` (pinned)

**Disable LLM**: Use `--no-llm` flag to skip Phases 4 and 6

---

## Limitations

### What Document Parser CANNOT Do

❌ **Prove theorems**: Only extracts and structures, doesn't generate proofs
❌ **Understand LaTeX errors**: Requires valid LaTeX syntax
❌ **Handle non-MyST formats**: Only works with Jupyter Book MyST markdown
❌ **Fix malformed directives**: Requires proper `:::` syntax
❌ **Infer unstated relationships without LLM**: Needs explicit `{prf:ref}` or LLM

### What Document Parser CAN Do

✅ **Extract all directive types**: Definitions, theorems, lemmas, axioms, proofs, remarks
✅ **Validate schema compliance**: Ensures all objects follow `fragile.proofs` types
✅ **Normalize labels**: Auto-fixes common label format issues
✅ **Infer object types**: Classifies objects based on content keywords
✅ **Extract cross-references**: Builds dependency graph from `{prf:ref}`
✅ **Track line ranges**: Links JSON output to source document locations
✅ **Process directories**: Batch processing of multiple documents

---

## Future Enhancements

Planned improvements:

- **Phase 4 (Complete)**: Full LLM-based relationship inference with Gemini 2.5 Pro
- **Phase 5 (Complete)**: Detailed proof sketch extraction from proof blocks
- **Phase 6 (Complete)**: LLM-based proof expansion to publication standard
- **Phase 9 (Future)**: Automated proof validation using Lean4 export
- **Phase 10 (Future)**: Interactive visualization of mathematical dependency graph
- **Performance**: Parallel document processing with multiprocessing
- **Formats**: Support for LaTeX, PDF, and Jupyter notebooks
- **Export**: Generate Lean4, Coq, or Isabelle formal proofs from structured JSON

---

## Support

For issues or questions:

1. Check this README
2. See QUICKSTART guide for copy-paste examples
3. Consult `document-parser.md` for agent internals
4. Check CLAUDE.md § Mathematical Proofing
5. Open issue: https://github.com/anthropics/claude-code/issues

---

**Version**: 1.0
**Last Updated**: 2025-10-26
**Maintainer**: Fragile Framework Team
