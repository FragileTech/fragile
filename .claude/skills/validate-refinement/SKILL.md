---
name: validate-refinement
description: Comprehensive validation workflow to verify refinement quality, completeness, and framework consistency. Use after completing refinement workflow, before building registries, or when debugging refinement issues.
---

# Validate Refinement Skill

## Purpose

Comprehensive validation workflow to verify that refined mathematical entities meet quality standards, schema requirements, and framework consistency before proceeding to registry building and pipeline transformation.

**Input**: Refined JSON files from document-refiner (`refined_data/`)
**Output**: Validation reports (markdown/JSON) with errors, warnings, and recommendations
**Scope**: Quality assurance and consistency checking across all entity types

---

## When to Use This Skill

Use this skill when you need to:

- **After completing refinement**: Verify quality before building registries
- **Before pipeline transformation**: Ensure refined data is ready for Stage 3
- **Debugging refinement issues**: Identify validation errors systematically
- **Quality assurance**: Verify publication-ready standards
- **Finding incomplete entities**: Discover missing fields or dependencies

---

## Validation Modes

### 1. Schema Validation (Fast - ~10 seconds)

Validates entities against Pydantic schemas only. Best for quick checks during development.

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode schema
```

**Checks**:
- ✅ Pydantic schema compliance (TheoremBox, Axiom, MathematicalObject, etc.)
- ✅ Required fields present and non-empty
- ✅ Field types correct (strings, lists, dicts)
- ✅ Label format (kebab-case with correct prefix)

### 2. Relationship Validation (Medium - ~30 seconds)

Validates cross-references and dependencies between entities.

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode relationships
```

**Checks**:
- ✅ `input_objects` references exist in registry
- ✅ `input_axioms` references exist in registry
- ✅ `input_parameters` references exist in registry
- ✅ `properties_required` references valid object attributes
- ✅ Proof → Theorem back-references valid
- ✅ Circular dependency detection

### 3. Framework Validation (Slow - ~2-3 minutes with Gemini)

Validates consistency with Fragile framework standards using LLM.

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode framework \
  --glossary docs/glossary.md
```

**Checks**:
- ✅ Notation consistency with `docs/glossary.md`
- ✅ Axiom usage correctness
- ✅ Definition alignment with framework
- ⚠️ **Note**: Requires Gemini MCP integration (runs basic checks in standalone mode)

### 4. Complete Validation (Comprehensive - ~3-5 minutes)

Runs all three validation modes in sequence.

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete \
  --output-report validation_report.md \
  --glossary docs/glossary.md
```

**Best for**: Final quality check before registry building

---

## Entity-Specific Validation

Validate only specific entity types for faster iteration:

```bash
# Validate theorems only
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --entity-types theorems \
  --mode schema

# Validate axioms and objects
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --entity-types axioms objects \
  --mode complete
```

**Available entity types**: `theorems`, `axioms`, `objects`, `parameters`, `proofs`, `remarks`, `equations`

---

## Strict Mode

Treat warnings as errors (fails validation if any warnings):

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete \
  --strict
```

**Use when**: Enforcing highest quality standards for publication

---

## Validation Report Format

### Markdown Report

```markdown
# Validation Report

**Generated**: 2025-10-28 14:32:01
**Directory**: `docs/source/1_euclidean_gas/03_cloning/refined_data/`
**Mode**: `complete`

---

## Summary

| Metric | Value |
|--------|-------|
| **Status** | ❌ **INVALID** |
| **Entities Validated** | 87 |
| **Errors** | 12 |
| **Warnings** | 23 |

## Validation Errors

Found **12** validation errors that must be fixed:

### thm-keystone-principle.json

- 🔴 **Field**: `statement` - Required field missing
- 🟠 **Field**: `input_axioms` - Referenced axiom 'ax-undefined' not found in registry

### obj-euclidean-gas.json

- 🟠 **Field**: `tags` - No tags specified

## Recommendations

❌ **Validation failed.** Please fix the errors above before proceeding.

**Recommended Actions:**
1. **Fix 2 critical errors first** (these prevent loading)
2. **Fix 10 validation errors** (these violate schema requirements)
3. **Review 23 warnings** (these suggest improvements)

**Useful Commands:**
```bash
# Find incomplete entities
python -m fragile.proofs.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/

# Complete partial refinements
python -m fragile.proofs.tools.complete_refinement \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/
```
```

### JSON Report

```json
{
  "timestamp": "2025-10-28T14:32:01.123456",
  "refined_dir": "docs/source/1_euclidean_gas/03_cloning/refined_data/",
  "mode": "complete",
  "summary": {
    "is_valid": false,
    "entity_count": 87,
    "error_count": 12,
    "warning_count": 23,
    "critical_error_count": 2
  },
  "errors": [
    {
      "file": "thm-keystone-principle.json",
      "field": "statement",
      "message": "Required field missing",
      "severity": "error"
    }
  ],
  "warnings": [
    {
      "file": "obj-euclidean-gas.json",
      "field": "tags",
      "message": "No tags specified",
      "suggestion": "Add descriptive tags for discoverability"
    }
  ]
}
```

---

## Integration with Other Skills

### From extract-and-refine

After Stage 2 refinement, validate quality:

```
Load extract-and-refine skill.
# ... run Stage 1 and Stage 2 ...

Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/
```

### To complete-partial-refinement

If validation finds incomplete entities:

```
Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/

# If errors found...
Load complete-partial-refinement skill.
Complete: docs/source/1_euclidean_gas/03_cloning/refined_data/
```

### To registry-management

After validation passes:

```
Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/

# If validation passes...
Load registry-management skill.
Transform and build registries: docs/source/1_euclidean_gas/03_cloning/
```

---

## Best Practices

1. **Validate often**: Run schema validation after each entity type refinement
2. **Complete validation before registry**: Always run complete validation before Stage 3
3. **Fix critical errors first**: These prevent loading and must be addressed
4. **Review warnings**: While non-blocking, they indicate quality improvements
5. **Use strict mode for publication**: Enforce highest standards
6. **Save reports**: Keep validation reports for documentation

---

## Entity-Specific Validation Rules

### Theorems (TheoremBox)
- **Required**: `label`, `name`, `statement`
- **Label format**: `thm-*`, `lem-*`, `prop-*`, or `cor-*`
- **Dependencies**: Should have `input_objects` and/or `input_axioms`
- **Properties**: If `input_objects` specified, should have `properties_required`
- **Output type**: Should be one of: property, bound, convergence, existence, uniqueness, equivalence, characterization

### Axioms (Axiom)
- **Required**: `label`, `name`, `statement`
- **Label format**: `ax-*`
- **Framework**: Should have `foundational_framework` and `core_assumption`
- **Parameters**: Should list parameters used

### Objects (MathematicalObject)
- **Required**: `label`, `name`, `mathematical_expression`
- **Label format**: `obj-*`
- **Type**: Should be one of: SPACE, OPERATOR, MEASURE, FUNCTION, SET, METRIC, DISTRIBUTION, PROCESS, ALGORITHM, CONSTANT
- **Attributes**: Should have non-empty `current_attributes`

### Parameters (Parameter/ParameterBox)
- **Required**: `symbol`
- **Domain**: Should specify mathematical domain
- **Scope** (enriched only): global, local, or universal
- **Constraints**: Should add if parameter has restrictions

### Proofs (ProofBox)
- **Required**: `proof_id`, `theorem`
- **Theorem linkage**: Must reference valid theorem
- **Steps**: Should have non-empty steps list with `step_number`, `content`, `justification`
- **Status**: Should be unproven, sketched, expanded, or verified

### Remarks (RemarkBox)
- **Required**: `label`, `content`
- **Label format**: `remark-*`
- **Type**: note, observation, intuition, example, warning, or historical
- **Related**: Should link to relevant entities

### Equations (EquationBox)
- **Required**: `label`, `latex_content`
- **Label format**: `eq-*`
- **Type**: definition, identity, evolution, constraint, or property
- **Symbols**: Should track `introduces_symbols` or `references_symbols`

---

## Troubleshooting

### Error: "Directory does not exist"

**Cause**: Invalid `--refined-dir` path

**Fix**:
```bash
# Check directory exists
ls -la docs/source/1_euclidean_gas/03_cloning/refined_data/

# Use absolute path
python -m fragile.mathster.tools.validation \
  --refined-dir /home/user/fragile/docs/source/1_euclidean_gas/03_cloning/refined_data/
```

### Warning: "No files matching pattern"

**Cause**: Entity type directory empty or doesn't exist

**Fix**: This is informational - entity type may not have been refined yet

### Error: "Invalid JSON"

**Cause**: Malformed JSON file

**Fix**:
```bash
# Find invalid JSON files
find docs/source/1_euclidean_gas/03_cloning/refined_data/ -name "*.json" -exec python -m json.tool {} \; 2>&1 | grep -B1 "Expecting"

# Fix JSON syntax
# Open file and fix syntax errors (trailing commas, missing quotes, etc.)
```

### Many "Required field missing" errors

**Cause**: Incomplete refinement

**Fix**: Use `complete-partial-refinement` skill to fill missing fields

### Many "Referenced ... not found in registry" errors

**Cause**: Incomplete entity extraction or incorrect labels

**Fix**:
1. Check if referenced entity exists in `raw_data/` but not `refined_data/`
2. If exists in raw_data, refine it
3. If doesn't exist, correct the reference label

---

## Success Criteria

Validation is successful when:

- ✅ **0 critical errors**: All files load without JSON/schema errors
- ✅ **0 validation errors**: All required fields present and valid
- ✅ **<5% completeness warnings**: Optional fields mostly populated
- ✅ **0 broken references**: All cross-references resolve
- ✅ **Report generated**: Detailed report available for review

After successful validation:
- Proceed to Stage 3 transformation
- Build entity registries
- Generate documentation

---

## Exit Codes

- **0**: Validation passed (is_valid == True)
- **1**: Validation failed (is_valid == False)

Use in scripts:
```bash
if python -m fragile.mathster.tools.validation --refined-dir PATH --mode schema; then
    echo "Validation passed!"
    # Proceed with next steps...
else
    echo "Validation failed!"
    exit 1
fi
```
