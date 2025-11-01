# Validate Refinement - Quick Start

Copy-paste commands for validating refined mathematical entities.

---

## Quick Schema Check (10 seconds)

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode schema
```

**Use**: Quick validation during development

---

## Complete Validation with Report (3-5 minutes)

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --output-report validation_report.md \
  --glossary docs/glossary.md
```

**Use**: Final quality check before registry building

---

## Validate Specific Entity Types

```bash
# Theorems only
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --entity-types theorems \
  --mode schema

# Axioms and objects
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --entity-types axioms objects \
  --mode complete
```

---

## Strict Mode (Warnings as Errors)

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --strict
```

**Use**: Publication-ready quality enforcement

---

## Relationship Validation Only

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode relationships
```

**Use**: Check cross-references and dependencies

---

## Generate JSON Report

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --output-json validation_report.json
```

**Use**: Machine-readable validation results

---

## Common Workflows

### After Completing Refinement

```bash
# 1. Quick schema check
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode schema

# 2. If no errors, run complete validation
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete \
  --output-report validation_report.md
```

### Before Registry Building

```bash
# Complete validation with strict mode
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete \
  --strict \
  --output-report final_validation.md
```

### Debugging Refinement Issues

```bash
# 1. Schema validation to identify basic errors
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode schema

# 2. Relationship validation to find broken references
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode relationships

# 3. Fix errors, then re-validate
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete
```

---

## Interpreting Results

### ✅ Validation Passed

```
======================================================================
Validation Result: ✅ VALID
  Entities: 87
  Errors: 0
  Warnings: 5
======================================================================
```

**Next Steps**: Proceed to transformation and registry building

---

### ❌ Validation Failed

```
======================================================================
Validation Result: ❌ INVALID
  Entities: 87
  Errors: 12
  Warnings: 23
  Critical Errors: 2
======================================================================
```

**Next Steps**: Fix errors, then re-validate

**Fix critical errors first** (prevent loading):
- Invalid JSON syntax
- Missing required files
- Schema violations

**Then fix validation errors** (schema requirements):
- Missing required fields
- Broken references
- Invalid field types

**Review warnings** (quality improvements):
- Missing optional fields
- Inconsistent naming
- Suboptimal classifications

---

## Time Estimates

| Validation Mode | Time | Use Case |
|----------------|------|----------|
| **Schema** | ~10 sec | Quick dev checks |
| **Relationships** | ~30 sec | Dependency checks |
| **Framework** | ~2-3 min | Notation consistency |
| **Complete** | ~3-5 min | Final QA before registries |

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| **0** | Validation passed | Proceed to next stage |
| **1** | Validation failed | Fix errors and re-run |

---

## Shell Integration

```bash
# Store result in variable
if python -m fragile.mathster.tools.validation \
    --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
    --mode complete; then
    echo "✅ Validation passed - proceeding to transformation"
    python -m fragile.mathster.tools.enriched_to_math_types \
      --input docs/source/1_euclidean_gas/03_cloning/refined_data/ \
      --output docs/source/1_euclidean_gas/03_cloning/pipeline_data/
else
    echo "❌ Validation failed - please fix errors"
    exit 1
fi
```

---

## All Entity Types

**Available for `--entity-types`**:
- `theorems` (includes lemmas, propositions, corollaries)
- `axioms`
- `objects` (mathematical objects from definitions)
- `parameters`
- `proofs`
- `remarks`
- `equations`

---

## Report Locations

### Default Output (Console)
Validation results print to stdout

### Markdown Report
```bash
--output-report PATH.md
```
Human-readable, includes recommendations

### JSON Report
```bash
--output-json PATH.json
```
Machine-readable, for automation

---

## Pro Tips

1. **Validate incrementally**: After refining each entity type
2. **Use entity-specific validation**: Faster iteration during development
3. **Complete validation before commits**: Ensure quality
4. **Save reports**: Documentation and tracking
5. **Script validation**: Integrate into CI/CD pipelines

---

## Next Steps After Validation

### If Validation Passes

```bash
# Transform to pipeline format
python -m fragile.mathster.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --output docs/source/1_euclidean_gas/03_cloning/pipeline_data/

# Build registries
# Use registry-management skill
```

### If Validation Fails

```bash
# Find incomplete entities
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/

# Complete partial refinements
python -m fragile.mathster.tools.complete_refinement \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/

# Re-validate
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete
```
