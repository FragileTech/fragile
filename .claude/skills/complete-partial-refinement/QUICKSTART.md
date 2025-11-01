# Complete Partial Refinement - Quick Start

Copy-paste commands for completing partially refined entities.

---

## Quick Completion Workflow (3 steps)

### Step 1: Find Incomplete Entities (10 seconds)

```bash
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output incomplete_entities.json
```

**Output**: `incomplete_entities.json` with list of all incomplete entities

---

### Step 2: Generate Completion Plan (10 seconds)

```bash
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output completion_plan.json \
  --output-instructions completion_instructions.md
```

**Output**:
- `completion_plan.json` - Structured plan with Gemini prompts
- `completion_instructions.md` - Step-by-step instructions

---

### Step 3: Execute Completion (Claude Code Required)

**NOTE**: This step requires Claude Code with `mcp__gemini-cli__ask-gemini` access.

Load the `completion_plan.json` and for each entity:

1. **Call Gemini**:
   ```
   Model: gemini-2.5-pro
   Prompt: <from completion_plan.gemini_prompt>
   ```

2. **Parse response** (extract JSON from markdown code block)

3. **Update entity file** with filled fields

4. **Validate** updated entity

5. **Log** completion status

---

## Alternative: Full Automated Script

```bash
# Run all steps at once (requires manual Gemini calls)
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output incomplete_entities.json

python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output completion_plan.json

# Then execute completion_plan.json using Claude Code
# (See completion_instructions.md for details)
```

---

## After Completion: Re-validate

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --output-report post_completion_validation.md
```

**Expected**: 0 validation errors

---

## Example Output

### find_incomplete_entities.py

```
Scanning refined data directory: docs/source/.../refined_data/

  Scanning theorems: 34 files... 34 incomplete
  Scanning axioms: 20 files... 20 incomplete
  Scanning objects: 31 files... 31 incomplete

======================================================================
SUMMARY
======================================================================
Total entities scanned: 85
Incomplete entities: 85
Completion rate: 0.0%

Top missing fields:
  - statement: 34 entities
  - tags: 30 entities
  - properties_required: 28 entities

Incomplete entities report saved to: incomplete_entities.json
```

### complete_refinement.py

```
======================================================================
COMPLETION PLAN GENERATION
======================================================================
Incomplete entities: 85
Completion rate: 0.0%

THEOREMS (34 incomplete)
----------------------------------------------------------------------
  thm-keystone-principle
    Priority: high
    Missing: statement, input_axioms

  thm-convergence-main
    Priority: high
    Missing: statement

...

✅ Completion plan saved to: completion_plan.json
✅ Instructions saved to: completion_instructions.md

======================================================================
NEXT STEPS
======================================================================
1. Review completion_plan.json to understand what will be filled
2. Use Claude Code to execute the completion workflow
3. Claude Code will call Gemini for each entity to fill missing fields
4. Review completed entities for accuracy
5. Re-run validation to verify all issues resolved
```

---

## Priority Processing

Entities are ordered by priority:

| Priority | Description | Example |
|----------|-------------|---------|
| **Critical** | JSON syntax errors | Prevents loading |
| **High** | Missing required fields | `statement`, `name` missing |
| **Medium** | Warnings only | Missing `tags`, optional fields |

Process critical first, then high, then medium.

---

## Time Estimates

| Entities | Find | Generate | Execute (Gemini) | Validate | Total |
|----------|------|----------|------------------|----------|-------|
| 10 | 5 sec | 5 sec | 50 sec | 10 sec | **~1 min** |
| 50 | 10 sec | 10 sec | 4 min | 15 sec | **~5 min** |
| 100 | 15 sec | 15 sec | 8 min | 20 sec | **~9 min** |

*Note: Execution time depends on Gemini API response time (~5 sec per entity)*

---

## Common Workflows

### Workflow 1: After Validation Fails

```bash
# 1. Validation found incomplete entities
python -m fragile.mathster.tools.validation \
  --refined-dir PATH \
  --mode schema
# Result: 85 errors

# 2. Find what's incomplete
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir PATH \
  --output incomplete_entities.json

# 3. Generate completion plan
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir PATH

# 4. Execute with Claude Code
# (Use Gemini to fill fields)

# 5. Re-validate
python -m fragile.mathster.tools.validation \
  --refined-dir PATH \
  --mode complete
# Expected: 0 errors
```

### Workflow 2: Updating Old Refined Data

```bash
# Old refined data may not have new schema fields

# 1. Find incomplete
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir old_refined_data/ \
  --output incomplete.json

# 2. Generate plan
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete.json \
  --refined-dir old_refined_data/

# 3. Execute completion
# (Claude Code + Gemini)

# 4. Validate against new schema
python -m fragile.mathster.tools.validation \
  --refined-dir old_refined_data/ \
  --mode complete
```

### Workflow 3: Completing Specific Entity Type

```bash
# Only complete theorems

# 1. Find incomplete (all types)
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir PATH \
  --output incomplete.json

# 2. Generate plan
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete.json \
  --refined-dir PATH

# 3. Filter completion_plan.json to theorems only
cat completion_plan.json | jq '.completion_tasks.theorems' > theorems_plan.json

# 4. Execute completion for theorems only
# (Claude Code + Gemini)

# 5. Validate theorems
python -m fragile.mathster.tools.validation \
  --refined-dir PATH \
  --entity-types theorems \
  --mode complete
```

---

## Troubleshooting Quick Fixes

### Error: "No incomplete entities found"

**Cause**: All complete

**Fix**: Nothing to do - proceed to next stage

---

### Warning: "Many entities need manual review"

**Cause**: Gemini couldn't fill fields confidently

**Fix**: Expected - schedule manual review time

---

### Error: "Validation still fails after completion"

**Cause**: Gemini filled fields incorrectly

**Fix**:
```bash
# Check validation report
cat post_completion_validation.md

# Identify which entities still have errors
# Manually edit those entity files
# Re-validate
python -m fragile.mathster.tools.validation \
  --refined-dir PATH \
  --mode complete
```

---

## Shell Script: Complete Automation

Save as `complete_refinement_workflow.sh`:

```bash
#!/bin/bash
set -e

REFINED_DIR="$1"

if [ -z "$REFINED_DIR" ]; then
    echo "Usage: $0 <refined_dir>"
    exit 1
fi

echo "Step 1: Finding incomplete entities..."
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir "$REFINED_DIR" \
  --output incomplete_entities.json

echo ""
echo "Step 2: Generating completion plan..."
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir "$REFINED_DIR" \
  --output completion_plan.json

echo ""
echo "Step 3: Please execute completion_plan.json using Claude Code"
echo "  1. Load completion_plan.json"
echo "  2. For each entity, call Gemini with the prompt"
echo "  3. Update entity files with filled fields"
echo "  4. Validate each updated entity"
echo ""
echo "Step 4: After completion, re-validate:"
echo "  python -m fragile.proofs.tools.validation \\"
echo "    --refined-dir $REFINED_DIR \\"
echo "    --mode complete"
```

**Usage**:
```bash
chmod +x complete_refinement_workflow.sh
./complete_refinement_workflow.sh docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/
```

---

## Next Steps After Completion

### If Validation Passes (0 errors)

```bash
# Proceed to transformation
python -m fragile.mathster.tools.enriched_to_math_types \
  --input docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --output docs/source/1_euclidean_gas/03_cloning/pipeline_data/

# Build registries
# Use registry-management skill
```

### If Validation Still Has Errors

```bash
# Review error report
cat post_completion_validation.md

# Identify problem entities
# Manually edit entity files
# Re-validate

python -m fragile.mathster.tools.validation \
  --refined-dir PATH \
  --mode complete
```

---

## Pro Tips

1. **Process in batches**: For 100+ entities, process 50 at a time
2. **Validate incrementally**: After every 20 entities, run validation
3. **Keep logs**: Save Gemini responses for audit trail
4. **Review high-priority first**: Critical and high priority need immediate attention
5. **Schedule manual review time**: ~30 sec per entity needing review
6. **Test on small sample first**: Process 5 entities, validate, then scale up

---

## Success Checklist

After completion, verify:

- ✅ `find_incomplete_entities` returns 0 incomplete entities
- ✅ Validation shows 0 errors
- ✅ Validation warnings < 5%
- ✅ Completion log shows high success rate (>90%)
- ✅ Manual review completed for flagged entities
- ✅ Completion report generated
- ✅ Post-completion validation report saved

---

## Quick Reference

```bash
# Find incomplete
python -m fragile.mathster.tools.find_incomplete_entities --refined-dir PATH

# Generate plan
python -m fragile.mathster.tools.complete_refinement --incomplete-file incomplete_entities.json --refined-dir PATH

# Execute: Use Claude Code with Gemini

# Re-validate
python -m fragile.mathster.tools.validation --refined-dir PATH --mode complete
```
