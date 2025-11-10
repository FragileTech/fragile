# Sketch-JSON Agent System Documentation

## Overview

The **sketch-json agent system** provides automated generation of structured proof sketch JSON files through dual AI validation (Gemini + Codex) and autonomous strategy selection (Opus reasoning).

### Components

1. **sketch-json** (sonnet) - Main orchestration agent
2. **strategy-selector** (opus) - Pure reasoning agent for strategy evaluation
3. **validate_sketch.py** - Python utility for JSON schema validation

---

## Quick Start

### Basic Usage

Invoke the sketch-json agent with a mathematical entity label:

```bash
# In Claude Code interface
sketch-json thm-geometry-guarantees-variance
```

The agent will:
1. ✅ Search for the label in mathster registry
2. ✅ Gather mathematical context and dependencies
3. ✅ Generate two independent proof strategies (Gemini + Codex)
4. ✅ Select optimal strategy using strategy-selector (opus reasoning)
5. ✅ Validate against JSON schema
6. ✅ Verify framework dependencies
7. ✅ Write JSON file to `docs/source/[chapter]/[document_id]/sketches/sketch-[label].json`

---

## Agent Details

### 1. sketch-json Agent

**Purpose**: Orchestrate the complete proof sketch generation workflow

**Model**: Sonnet (balanced reasoning + tool management)

**Tools Available**:
- Read, Grep, Glob - Document navigation
- Bash - Command execution (mathster search, directory creation)
- Write - File output
- mcp__gemini-cli__ask-gemini - Gemini 2.5 Pro consultation
- mcp__codex__codex - GPT-5 consultation
- Task - Invoke strategy-selector agent

**Input Format**:
```
Generate proof sketch for: [label]
```

**Output**:
```json
{
  "label": "thm-example",
  "entity_type": "theorem",
  "document_id": "03_cloning",
  "title": "Theorem Title",
  "statement": "Mathematical statement...",
  "strategy": {
    "strategist": "Selected strategist",
    "method": "Proof method name",
    "summary": "Core insight...",
    "keySteps": ["Step 1", "Step 2", ...],
    "strengths": ["Advantage 1", ...],
    "weaknesses": ["Challenge 1", ...],
    "frameworkDependencies": {
      "theorems": [...],
      "lemmas": [...],
      "axioms": [...],
      "definitions": [...]
    },
    "technicalDeepDives": [...],
    "confidenceScore": "High|Medium|Low"
  },
  "_metadata": {
    "generated_at": "ISO timestamp",
    "agent_version": "sketch-json v1.0",
    "validation": {
      "schema_valid": true,
      "framework_verified": true,
      "missing_fields": [],
      "unverified_dependencies": []
    },
    "selection_process": {
      "gemini_confidence": "High",
      "codex_confidence": "Medium",
      "selected_strategist": "STRATEGY A",
      "selector_confidence": "HIGH",
      "selector_justification": "..."
    }
  }
}
```

---

### 2. strategy-selector Agent

**Purpose**: Deep reasoning to select optimal proof strategy

**Model**: Opus (maximum reasoning capability)

**Tools**: None (pure reasoning agent)

**Input**: Two proof strategies + theorem context

**Evaluation Criteria**:
1. **Framework Consistency**: All dependencies exist and are correctly applied
2. **Logical Soundness**: No circular reasoning, valid step connections
3. **Technical Feasibility**: All operations are performable
4. **Completeness**: Addresses all theorem parts
5. **Clarity**: Actionable steps for expansion

**Output Format**:
```markdown
# Strategy Selection Report

## DECISION: STRATEGY A | STRATEGY B | HYBRID

**Selected Strategist**: Gemini 2.5 Pro
**Confidence Level**: HIGH

## SELECTION JUSTIFICATION
[Detailed reasoning...]

## SELECTED STRATEGY (JSON)
{...}
```

**Decision Rules** (Priority Order):
1. Framework validity (invalid dependencies → reject)
2. Logical soundness (circular reasoning → reject)
3. Completeness (missing parts → penalize)
4. Technical feasibility (fewer lemmas → prefer)
5. Clarity (clearer steps → prefer)
6. Confidence scores (higher → prefer)
7. Consensus bonus (both agree → high confidence)

---

### 3. validate_sketch.py Utility

**Purpose**: Standalone JSON schema validation

**Location**: `src/mathster/agent_schemas/validate_sketch.py`

**CLI Usage**:
```bash
# Validate a sketch file
python src/mathster/proof_sketcher/validate_sketch.py sketch-thm-example.json

# Output
✅ Validation PASSED: sketch-thm-example.json
# or
❌ Validation FAILED: sketch-thm-example.json
Errors:
  - Validation error at keySteps: [] is too short
```

**Python API**:
```python
from mathster.agent_schemas.validate_sketch import (
    validate_sketch_strategy,
    get_missing_required_fields,
    fill_missing_fields,
)

# Validate a strategy
strategy = {...}
is_valid, errors = validate_sketch_strategy(strategy)

if not is_valid:
    print(f"Errors: {errors}")

# Check missing fields
missing = get_missing_required_fields(strategy)
print(f"Missing: {missing}")

# Fill missing fields with defaults
filled, filled_fields = fill_missing_fields(strategy)
print(f"Filled fields: {filled_fields}")
```

---

## Workflow Examples

### Example 1: Complete Success

**Command**:
```
sketch-json thm-geometry-guarantees-variance
```

**Execution Trace**:
```
✅ Label found: theorem in document 03_cloning
✅ Document located: 1_euclidean_gas/03_cloning.md
✅ Output directory created: docs/source/1_euclidean_gas/03_cloning/sketches
✅ Resolved 3 dependencies: 2 definitions, 1 axiom
🚀 Submitting to Gemini 2.5 Pro...
🚀 Submitting to GPT-5 (high reasoning)...
✅ Gemini strategy received (confidence: High)
✅ Codex strategy received (confidence: Medium)
🔍 Invoking strategy-selector agent (opus)...
✅ Selection complete: STRATEGY A (Gemini)
   - Confidence: HIGH
✅ Schema validation: PASSED
✅ Framework verification: 11/12 dependencies verified
✅ JSON written to: docs/source/1_euclidean_gas/03_cloning/sketches/sketch-thm-geometry-guarantees-variance.json
```

**Output File**: `docs/source/1_euclidean_gas/03_cloning/sketches/sketch-thm-geometry-guarantees-variance.json`

---

### Example 2: Label Not Found

**Command**:
```
sketch-json thm-nonexistent-theorem
```

**Output**:
```
❌ LABEL NOT FOUND: thm-nonexistent-theorem

Attempted searches:
- Preprocess registry: NOT FOUND
- Directives registry: NOT FOUND

Troubleshooting:
Run the following command to find similar labels:
  grep -r "nonexistent" unified_registry/

User Action Required:
- Verify label spelling
- Check if entity exists in framework
```

---

### Example 3: Partial Validation

**Command**:
```
sketch-json lem-drift-bound
```

**Output**:
```
✅ Proof sketch generated successfully

Strategy Selected: HYBRID (Gemini primary + Codex techniques)
Confidence: MEDIUM

Validation Status:
- Schema Valid: ✅
- Framework Verified: ⚠️
- Missing Fields: 0
- Unverified Dependencies: 2

⚠️ 2 dependencies could not be verified:
  - def-new-concept (not in glossary.md)
  - lem-helper-result (not in registry)

Recommendation: Manually verify these dependencies before expanding proof.

Output File: docs/source/1_euclidean_gas/06_convergence/sketches/sketch-lem-drift-bound.json
```

---

## Output Directory Structure

```
docs/source/
├── 1_euclidean_gas/
│   ├── 03_cloning.md
│   ├── 03_cloning/
│   │   ├── registry/
│   │   └── sketches/                          # Created by sketch-json
│   │       ├── sketch-thm-geometry-guarantees-variance.json
│   │       ├── sketch-lem-pairing-property.json
│   │       └── ...
│   ├── 07_mean_field.md
│   ├── 07_mean_field/
│   │   └── sketches/
│   │       ├── sketch-thm-mean-field-limit.json
│   │       └── ...
│   └── ...
├── 2_geometric_gas/
│   ├── 11_geometric_gas/
│   │   └── sketches/
│   │       └── ...
│   └── ...
└── ...
```

---

## Integration with Proof Pipeline

The sketch-json agent is designed to integrate with the broader proof development pipeline:

### Pipeline Stages

1. **Extraction** → Raw mathematical entities from markdown
2. **Refinement** → Structured entities with dependencies
3. **Sketch Generation** ← **sketch-json agent** (this system)
4. **Proof Expansion** → Full detailed proof (theorem-prover agent)
5. **Verification** → Dual review (math-reviewer agent)

### Workflow Integration

```bash
# Stage 3: Generate proof sketch
sketch-json thm-main-result

# Stage 4: Expand to full proof
theorem-prover docs/source/.../sketches/sketch-thm-main-result.json

# Stage 5: Verify proof
math-reviewer docs/source/.../proofs/proof-thm-main-result.md
```

---

## JSON Schema Reference

**File**: `src/mathster/agent_schemas/sketch_strategy.json`

**Required Fields**:
- `strategist` (string): AI model name
- `method` (string): Proof technique name
- `summary` (string): Core insight narrative
- `keySteps` (array): Main logical stages (min 1)
- `strengths` (array): Advantages
- `weaknesses` (array): Technical difficulties
- `frameworkDependencies` (object): Required framework results
  - `theorems` (array)
  - `lemmas` (array)
  - `axioms` (array)
  - `definitions` (array)
- `confidenceScore` (enum): "High" | "Medium" | "Low"

**Optional Fields**:
- `technicalDeepDives` (array): Analysis of challenging parts

**Dependency Object Structure**:
```json
{
  "label": "thm-example",
  "document": "document_id",
  "purpose": "What this provides for the proof",
  "usedInSteps": ["Step 1", "Step 3"]
}
```

---

## Troubleshooting

### Issue 1: "Label not found"

**Cause**: Label doesn't exist in registry

**Solution**:
1. Check spelling: `grep -r "your-label" unified_registry/`
2. Verify in glossary: `grep "your-label" docs/glossary.md`
3. Run extraction if entity exists in source but not registry

---

### Issue 2: "Invalid JSON from strategist"

**Cause**: Gemini or Codex returned malformed JSON

**Solution**: Agent automatically:
1. Attempts to fix common issues (markdown wrappers, trailing commas)
2. Falls back to single-strategist analysis
3. Continues with lower confidence

**User Action**: Review raw responses in metadata

---

### Issue 3: "Directory creation failed"

**Cause**: File system permissions or disk space

**Solution**:
1. Check permissions: `ls -ld docs/source/[chapter]/[document_id]`
2. Check disk space: `df -h`
3. Agent displays JSON inline as fallback

---

### Issue 4: "Many unverified dependencies"

**Cause**: Strategy cites labels not in glossary/registry

**Solution**:
1. Check if dependencies are forward references (cite later documents)
2. Verify dependencies manually: `uv run mathster search [label]`
3. Update glossary if dependencies are valid but missing

**Status**: Non-critical - sketch still usable, needs manual review

---

## Advanced Usage

### Batch Processing

Process multiple labels sequentially:

```bash
# In a loop
for label in thm-main-1 thm-main-2 thm-main-3; do
    sketch-json $label
done
```

### Custom Validation

Validate existing sketch files:

```bash
# Validate single file
python src/mathster/proof_sketcher/validate_sketch.py \
    docs/source/.../sketches/sketch-thm-example.json

# Validate all sketches in directory
for f in docs/source/.../sketches/sketch-*.json; do
    python src/mathster/proof_sketcher/validate_sketch.py "$f"
done
```

### Programmatic Access

```python
from pathlib import Path
import json
from mathster.agent_schemas.validate_sketch import validate_file

# Load and validate sketch
sketch_path = Path("docs/source/.../sketches/sketch-thm-example.json")
is_valid, errors = validate_file(sketch_path)

if is_valid:
    sketch = json.loads(sketch_path.read_text())
    print(f"Method: {sketch['strategy']['method']}")
    print(f"Confidence: {sketch['strategy']['confidenceScore']}")
else:
    print(f"Validation failed: {errors}")
```

---

## Performance Characteristics

### Time Estimates

- Label resolution: ~10 seconds
- Context gathering: ~30 seconds
- Dual strategy generation: ~2-3 minutes (parallel)
- Strategy selection: ~1-2 minutes (opus reasoning)
- Validation: ~20 seconds
- **Total**: ~4-6 minutes per sketch

### Optimization

- Strategies are generated **in parallel** (single message, two tool calls)
- Mathster searches are **cached** within session
- Source document reads are **minimized** (prefer preprocess registry)

---

## Testing

### Unit Tests

```bash
# Run validation utility tests
uv run pytest tests/test_validate_sketch.py -v

# Expected output
tests/test_validate_sketch.py::test_load_schema PASSED
tests/test_validate_sketch.py::test_valid_strategy PASSED
tests/test_validate_sketch.py::test_missing_required_fields PASSED
tests/test_validate_sketch.py::test_fill_missing_fields PASSED
tests/test_validate_sketch.py::test_invalid_strategy_validation PASSED
tests/test_validate_sketch.py::test_empty_key_steps PASSED

6 passed in 1.39s
```

### Integration Tests

```bash
# Test with real theorem
sketch-json thm-geometry-guarantees-variance

# Verify output exists
ls -lh docs/source/1_euclidean_gas/03_cloning/sketches/sketch-thm-geometry-guarantees-variance.json

# Validate output
python src/mathster/proof_sketcher/validate_sketch.py \
    docs/source/1_euclidean_gas/03_cloning/sketches/sketch-thm-geometry-guarantees-variance.json
```

---

## Future Enhancements

Potential improvements:

1. **Batch mode**: Process multiple labels in single invocation
2. **Incremental updates**: Update existing sketches when framework changes
3. **Confidence calibration**: Track accuracy of AI confidence scores
4. **Strategy library**: Build corpus of successful strategies for similar theorems
5. **Interactive mode**: Allow user to guide strategy selection
6. **Dependency graph**: Visualize framework dependencies in sketch

---

## Support and Feedback

For issues or questions:
1. Check troubleshooting section above
2. Review agent specifications:
   - `.claude/agents/sketch-json.md`
   - `.claude/agents/strategy-selector.md`
3. Examine validation utility: `src/mathster/agent_schemas/validate_sketch.py`
4. Review test cases: `tests/test_validate_sketch.py`

---

**Version**: 1.0
**Last Updated**: 2025-11-10
**Agents**: sketch-json (sonnet), strategy-selector (opus)
**Schema**: src/mathster/agent_schemas/sketch_strategy.json
