---
name: complete-partial-refinement
description: Systematic workflow for completing partially refined entities by discovering missing fields, generating completion plans with Gemini prompts, and batch-filling incomplete data. Use after validation finds incomplete entities or when updating old refined data to current standards.
---

# Complete Partial Refinement Skill

## Purpose

Systematic workflow for completing partially refined mathematical entities by:
1. Discovering entities with missing/incomplete fields
2. Generating intelligent completion plans with Gemini prompts
3. Batch-filling missing fields using LLM assistance
4. Validating completed entities

**Input**: Partially complete `refined_data/` with validation errors
**Output**: Complete `refined_data/` with all required fields filled
**Scope**: Systematic completion of incomplete refinements

---

## When to Use This Skill

Use this skill when you need to:

- **After validation fails**: Fix incomplete entities identified by validate-refinement
- **Updating old refined data**: Bring entities up to current schema standards
- **Filling missing fields in bulk**: Complete many entities systematically
- **Interrupted refinement workflows**: Resume and complete partial work
- **Schema migrations**: Update entities to new schema versions

---

## Complete Workflow

### Step 1: Identify Incomplete Entities

Use the `find_incomplete_entities.py` tool to discover entities with missing or incomplete fields:

```bash
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --output incomplete_entities.json
```

**Output**: `incomplete_entities.json` with:
- List of all incomplete entities
- Missing fields for each entity
- Validation errors and warnings
- Statistics by entity type

**Example output**:
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
  - core_assumption: 20 entities
```

### Step 2: Generate Completion Plan

Use the `complete_refinement.py` tool to generate a completion plan with Gemini prompts:

```bash
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --output completion_plan.json \
  --output-instructions completion_instructions.md
```

**Output**:
- `completion_plan.json` - Structured completion plan with:
  - Gemini prompts for each entity
  - Priority ordering (critical → high → medium)
  - Current entity data for reference
  - Missing fields list
- `completion_instructions.md` - Step-by-step instructions for Claude Code

**Example completion plan entry**:
```json
{
  "label": "thm-keystone-principle",
  "file": "theorems/thm-keystone-principle.json",
  "entity_type": "theorem",
  "priority": "high",
  "missing_fields": ["statement", "input_axioms"],
  "gemini_prompt": "I need to complete the following theorem entity:\n\n**Label**: thm-keystone-principle\n**Name**: Keystone Principle\n\n**Missing fields to fill:**\n- `statement`: Full mathematical statement/theorem\n- `input_axioms`: List of axiom labels this requires\n\nPlease provide the missing fields in JSON format..."
}
```

### Step 3: Execute Completion (Claude Code)

**IMPORTANT**: This step requires Claude Code with access to `mcp__gemini-cli__ask-gemini`.

For each entity in the completion plan:

#### 3a. Load Completion Task

```python
import json
from pathlib import Path

# Load completion plan
with open("completion_plan.json") as f:
    plan = json.load(f)

# Process by priority
for entity_type, tasks in plan["completion_tasks"].items():
    for task in sorted(tasks, key=lambda t: {"critical": 0, "high": 1, "medium": 2}[t["priority"]]):
        # Process this task...
```

#### 3b. Call Gemini to Fill Fields

Use `mcp__gemini-cli__ask-gemini` with model `gemini-2.5-pro`:

```python
# Get Gemini prompt from task
gemini_prompt = task["gemini_prompt"]

# Call Gemini (Claude Code only)
response = mcp__gemini-cli__ask-gemini(
    model="gemini-2.5-pro",
    prompt=gemini_prompt
)

# Parse JSON response
import re
json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
if json_match:
    filled_fields = json.loads(json_match.group(1))
```

#### 3c. Update Entity File

```python
# Load current entity data
file_path = refined_dir / task["file"]
with open(file_path) as f:
    entity_data = json.load(f)

# Update with filled fields
entity_data.update(filled_fields)

# Save updated entity
with open(file_path, 'w') as f:
    json.dump(entity_data, f, indent=2, ensure_ascii=False)
```

#### 3d. Validate Updated Entity

```python
from fragile.proofs.tools.validation import TheoremValidator

# Validate
validator = TheoremValidator()
result = validator.validate_entity(entity_data, file_path)

if result.is_valid:
    print(f"✅ {task['label']}: Completed successfully")
else:
    print(f"⚠️ {task['label']}: Validation failed after completion")
    print(f"  Errors: {len(result.errors)}")
    # May need manual review
```

### Step 4: Track Completion Status

Maintain a completion log:

```python
completion_log = {
    "completed": [],
    "failed_validation": [],
    "needs_manual_review": [],
    "errors": []
}

# After each entity
if result.is_valid:
    completion_log["completed"].append(task["label"])
elif result.errors:
    completion_log["failed_validation"].append({
        "label": task["label"],
        "errors": [str(e) for e in result.errors]
    })
```

### Step 5: Manual Review

Review entities that need attention:

```bash
# Check completion log
cat completion_log.json | jq '.failed_validation'

# Manually inspect problematic entities
# Edit files directly if Gemini's suggestions were incorrect
```

### Step 6: Re-validate All Entities

After completing all entities, run full validation:

```bash
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/03_cloning/refined_data/ \
  --mode complete \
  --output-report post_completion_validation.md
```

**Success criteria**:
- ✅ 0 validation errors
- ✅ <5% warnings (optional fields)
- ✅ All required fields populated

### Step 7: Generate Completion Report

Document what was completed:

```markdown
# Completion Report

**Date**: 2025-10-28
**Refined Directory**: docs/source/1_euclidean_gas/03_cloning/refined_data/

## Summary

- **Total incomplete entities**: 85
- **Successfully completed**: 80
- **Failed validation**: 3
- **Requires manual review**: 2

## Statistics by Entity Type

| Entity Type | Incomplete | Completed | Failed | Success Rate |
|-------------|------------|-----------|--------|--------------|
| Theorems    | 34         | 32        | 2      | 94%          |
| Axioms      | 20         | 20        | 0      | 100%         |
| Objects     | 31         | 28        | 3      | 90%          |

## Fields Filled

- `statement`: 34 entities
- `tags`: 80 entities
- `input_axioms`: 28 entities
- `properties_required`: 25 entities

## Entities Requiring Manual Review

1. **thm-complex-theorem** - Gemini couldn't infer input_axioms
2. **obj-ambiguous-object** - Object type classification unclear

## Next Steps

1. Manually review 2 entities requiring attention
2. Re-run validation after manual fixes
3. Proceed to transformation and registry building
```

---

## Priority Ordering

The completion plan orders entities by priority:

### Critical Priority
- **Triggers**: JSON syntax errors, schema violations
- **Impact**: Prevents loading entity file
- **Action**: Fix immediately before other entities

### High Priority
- **Triggers**: Missing required fields (statement, name, mathematical_expression)
- **Impact**: Entity unusable without these fields
- **Action**: Complete before medium priority

### Medium Priority
- **Triggers**: Warnings only (missing tags, missing optional fields)
- **Impact**: Reduced quality but entity still functional
- **Action**: Complete for publication-ready quality

---

## Entity-Specific Completion

### Theorems (TheoremBox)

**Commonly missing**:
- `statement`: Full mathematical statement
- `input_axioms`: Axiom dependencies
- `input_objects`: Object dependencies
- `properties_required`: Required object properties
- `tags`: Discoverability tags

**Gemini prompt focus**: "Based on theorem name and existing content, what axioms and objects does this theorem require?"

### Axioms (Axiom)

**Commonly missing**:
- `foundational_framework`: Framework name
- `core_assumption`: Core assumption description
- `parameters`: Parameter list

**Gemini prompt focus**: "What framework does this axiom belong to? What is its core assumption?"

### Objects (MathematicalObject)

**Commonly missing**:
- `object_type`: Type classification (SPACE, OPERATOR, etc.)
- `current_attributes`: Property list
- `tags`: Discoverability tags

**Gemini prompt focus**: "Classify this mathematical object and list its key attributes"

### Parameters (Parameter/ParameterBox)

**Commonly missing**:
- `domain`: Mathematical domain
- `constraints`: Restrictions (e.g., γ > 0)
- `scope`: global/local/universal

**Gemini prompt focus**: "What is the domain and constraints for this parameter?"

---

## Batch Processing Strategy

### Small Batches (< 10 entities)
Process sequentially with manual review after each entity.

### Medium Batches (10-50 entities)
Process in groups of 10, validate batch, review errors, continue.

### Large Batches (50+ entities)
1. Process all automatically
2. Collect all validation failures
3. Review and fix failures in batch
4. Re-validate entire set

---

## Common Issues and Solutions

### Issue: Gemini Doesn't Have Enough Context

**Symptom**: Gemini returns "I don't have enough information to complete this field"

**Solution**:
1. Check if source document information is in entity data
2. Manually look up entity in source markdown
3. Add context to Gemini prompt:
   ```
   Additional context from source document:
   [relevant excerpt from markdown]

   Now please fill the missing fields...
   ```

### Issue: Gemini Suggestions Don't Validate

**Symptom**: Entity fails validation after Gemini fills fields

**Solution**:
1. Review validation errors
2. Check if Gemini used wrong field format (e.g., titlecase instead of lowercase)
3. Manually correct field format
4. Re-validate

### Issue: Missing Field Can't Be Inferred

**Symptom**: Field genuinely requires domain knowledge not in entity data

**Solution**:
1. Mark entity for manual review
2. Consult framework documents (docs/glossary.md, source markdown)
3. Manually fill field based on research
4. Validate

### Issue: Circular Dependencies

**Symptom**: Completing entity A requires entity B, but B isn't refined yet

**Solution**:
1. Add entity B to refinement queue
2. Process B first
3. Then complete A with proper reference

---

## Integration with Other Skills

### From validate-refinement

After validation finds incomplete entities:

```
Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/

# If validation fails with incomplete entities...
Load complete-partial-refinement skill.
Complete: docs/source/1_euclidean_gas/03_cloning/refined_data/
```

### To validate-refinement

After completing entities, re-validate:

```
Load complete-partial-refinement skill.
Complete: docs/source/1_euclidean_gas/03_cloning/refined_data/

# After completion...
Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/
```

### To registry-management

After all validations pass:

```
Load complete-partial-refinement skill.
Complete: docs/source/1_euclidean_gas/03_cloning/refined_data/

Load validate-refinement skill.
Validate: docs/source/1_euclidean_gas/03_cloning/refined_data/

# If validation passes...
Load registry-management skill.
Transform and build registries: docs/source/1_euclidean_gas/03_cloning/
```

---

## Best Practices

1. **Always start with find_incomplete_entities**: Don't guess what's incomplete
2. **Process by priority**: Critical → High → Medium
3. **Validate incrementally**: After every 10 entities, validate batch
4. **Review Gemini suggestions**: Don't blindly accept, verify correctness
5. **Keep completion log**: Track what was completed and what failed
6. **Manual review flagged entities**: Some fields genuinely require human judgment
7. **Re-validate at end**: Ensure all issues resolved
8. **Document completion**: Generate completion report for records

---

## Success Criteria

Completion is successful when:

- ✅ **0 critical errors**: All entities load without JSON/schema errors
- ✅ **0 required field errors**: All required fields populated
- ✅ **<5% warnings**: Optional fields mostly complete
- ✅ **Completion log generated**: Full audit trail
- ✅ **Post-completion validation passes**: Final validation successful
- ✅ **Completion report created**: Documentation complete

---

## Time Estimates

| Task | Time (per entity) | Batch Time (50 entities) |
|------|-------------------|--------------------------|
| **Find incomplete** | ~1 sec | ~10 sec |
| **Generate plan** | ~1 sec | ~10 sec |
| **Call Gemini** | ~5 sec | ~4 min |
| **Update & validate** | ~2 sec | ~2 min |
| **Manual review** | ~30 sec | ~25 min |
| **Total** | ~40 sec | **~30 min** |

**Note**: Actual time varies based on entity complexity and Gemini response time

---

## Example: Complete 85 Incomplete Entities

**Scenario**: Found 85 incomplete entities (34 theorems, 20 axioms, 31 objects)

### Step-by-Step Execution

```bash
# 1. Find incomplete (10 seconds)
python -m fragile.mathster.tools.find_incomplete_entities \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output incomplete_entities.json

# 2. Generate completion plan (10 seconds)
python -m fragile.mathster.tools.complete_refinement \
  --incomplete-file incomplete_entities.json \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --output completion_plan.json

# 3. Execute completion (Claude Code - 7 minutes for 85 entities)
# Use Claude Code to process completion_plan.json
# Call Gemini for each entity (~5 sec per entity)

# 4. Re-validate (20 seconds)
python -m fragile.mathster.tools.validation \
  --refined-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data/ \
  --mode complete \
  --output-report post_completion_validation.md

# 5. Review (5-10 minutes for 85 entities)
# Manual review of any failed validations

# Total time: ~15-20 minutes for 85 entities
```

---

## Troubleshooting

### Error: "No incomplete entities found"

**Cause**: All entities are complete

**Action**: No completion needed - proceed to next stage

### Error: "Gemini rate limit exceeded"

**Cause**: Too many API calls in short time

**Solution**:
- Process in smaller batches
- Add delays between Gemini calls
- Resume from completion log after cooldown

### Error: "Field still missing after completion"

**Cause**: Gemini didn't fill the field

**Solution**:
1. Check Gemini response for explanation
2. Manually add field based on domain knowledge
3. Re-validate

### Warning: "Many entities need manual review"

**Cause**: Gemini couldn't confidently fill fields

**Solution**:
- Expected for complex entities
- Schedule time for manual review
- Consult framework documents for guidance

---

## Automation Tips

### Script for Batch Completion

```python
#!/usr/bin/env python3
"""Automated completion workflow."""

import json
import subprocess
from pathlib import Path

def complete_refinement_workflow(refined_dir: Path):
    # Step 1: Find incomplete
    subprocess.run([
        "python", "-m", "fragile.mathster.tools.find_incomplete_entities",
        "--refined-dir", str(refined_dir),
        "--output", "incomplete_entities.json"
    ])

    # Step 2: Generate plan
    subprocess.run([
        "python", "-m", "fragile.mathster.tools.complete_refinement",
        "--incomplete-file", "incomplete_entities.json",
        "--refined-dir", str(refined_dir),
        "--output", "completion_plan.json"
    ])

    # Step 3: Execute completion (Claude Code required)
    print("Please execute completion_plan.json using Claude Code")
    print("After completion, re-run this script with --validate flag")

if __name__ == "__main__":
    refined_dir = Path("docs/source/1_euclidean_gas/03_cloning/refined_data/")
    complete_refinement_workflow(refined_dir)
```

---

## Exit Codes

- **0**: Success - entities completed (or none incomplete)
- **1**: Error - tool execution failed
- **2**: Manual review required - some entities need attention
