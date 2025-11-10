# Referencer Agent - Quickstart Guide

## What It Does

The **Referencer Agent** adds **backward cross-references** to your mathematical documents to ensure all entities are properly connected.

**Key Point**: This agent **only adds backward references** - it does NOT add forward references.

**Example**:
- **Forward reference** (already exists): Theorem T says "Using {prf:ref}`def-walker-state`"
- **Backward reference** (added by referencer): Definition D says "This is used in {prf:ref}`thm-main`"

## Quick Start

### 1. Check Baseline Connectivity

```bash
uv run mathster connectivity docs/source/1_euclidean_gas/03_cloning.md
```

Look for **isolated labels** - these need backward references.

### 2. Run Referencer Agent

```python
# Via Task tool in Claude Code
Task(
    subagent_type="referencer",
    description="Add backward refs to cloning doc",
    prompt="Add backward references to docs/source/1_euclidean_gas/03_cloning.md"
)
```

Or if CLI integration is added:
```bash
uv run mathster reference docs/source/1_euclidean_gas/03_cloning.md
```

### 3. Review Changes

Check the generated report:
```bash
cat reports/referencer_03_cloning_<timestamp>.md
```

Verify connectivity improved:
```bash
uv run mathster connectivity docs/source/1_euclidean_gas/03_cloning.md
```

### 4. Build and Review

```bash
make build-docs
```

Open `docs/_build/html/1_euclidean_gas/03_cloning.html` and verify:
- ✅ Backward references render as clickable links
- ✅ All non-remark entities have appropriate "Used by" mentions
- ✅ Corollaries reference parent theorems

### 5. Commit Changes

```bash
git add docs/source/1_euclidean_gas/03_cloning.md
git commit -m "Add backward references to cloning document

- Reduced isolated labels from 42 to 3
- Added 89 backward references
- All references validated by Gemini 2.5 Pro"
```

---

## What Gets Modified

### ✅ Modified
- **Markdown files**: In-place edits with backward references
- Example:
  ```markdown
  :::{prf:definition} Walker State
  :label: def-walker-state

  A walker is characterized by position and velocity.

  This definition is central to {prf:ref}`thm-keystone-principle` and {prf:ref}`thm-convergence`.
  :::
  ```

### ❌ NOT Modified
- Registry JSON files (unchanged)
- Forward references (not added)
- Proof blocks (don't get backward refs)

---

## Key Rules

1. **Backward references only**: From foundational concepts to results that use them
2. **Mathematical validation**: All references validated by Gemini 2.5 Pro
3. **No duplicates**: Each entity referenced once per directive
4. **Corollaries must reference parent theorems**: Always enforced
5. **Exhaustive but valid**: Lists all valid references, filters invalid ones

---

## Expected Results

### Before
```markdown
# Connectivity Report
- Isolated labels: 42
- Bidirectional: 168

Isolated entities:
- axiom-lipschitz-fields
- def-walker-state
- def-cloning-operator
...
```

### After
```markdown
# Connectivity Report
- Isolated labels: 3 (all remarks)
- Bidirectional: 220

Remaining isolated:
- rem-technical-note
- rem-implementation-detail
```

**Typical improvement**: 80-95% reduction in isolated non-remark labels.

---

## Common Options

### Dry Run (Report Only)
```bash
uv run mathster reference docs/source/.../document.md --dry-run
```
- Generates connectivity report
- Shows what would be changed
- **Does not modify files**

### Skip LLM Validation (Faster but Less Accurate)
```bash
uv run mathster reference docs/source/.../document.md --no-validate-llm
```
- Skips Gemini validation
- Relies only on registry + text patterns
- **Not recommended for production**

### Custom Batch Size
```bash
uv run mathster reference docs/source/.../document.md --batch-size 10
```
- Validates 10 references per LLM call instead of 20
- Useful if hitting rate limits

---

## When to Use

### ✅ Use Referencer When:
- Document has many isolated labels (definitions, axioms)
- Definitions are used but not acknowledged
- Corollaries lack parent theorem references
- Navigation from concepts to results is poor

### ❌ Don't Use Referencer When:
- Forward references need updating (use cross-referencer)
- Registry metadata needs filling (use cross-referencer)
- Document is intentionally minimal (e.g., introduction)

---

## Troubleshooting

### Issue: LLM validation is slow
**Solution**: Be patient - batches of 20 refs take ~10-15s each. For 100 refs = ~1 minute of validation.

### Issue: Too many backward references for one entity
**Example**: def-walker-state referenced by 30 entities
**Solution**: Agent groups them by type:
```markdown
This is used throughout:
- **Theorems**: {prf:ref}`thm-1`, {prf:ref}`thm-2`, ...
- **Lemmas**: {prf:ref}`lem-1`, {prf:ref}`lem-2`, ...
```

### Issue: Edit failed (old_string not unique)
**Solution**: Agent logs failed edits and continues. Check report for errors.

### Issue: Connectivity didn't improve
**Possible causes**:
- All references were invalid (check validation stats in report)
- Document already well-connected
- Registry data is stale (run cross-referencer first)

---

## Integration with Other Agents

### Recommended Pipeline

1. **document-parser** → Extract entities
2. **cross-referencer** → Add forward reference metadata
3. **document-refiner** → Enrich entity data
4. **referencer** ← **You are here** (add backward references to markdown)

**Note**: Referencer can work standalone with just markdown + connectivity analysis.

---

## Example Report Excerpt

```markdown
# Referencer Report: 03_cloning
Date: 2025-01-10 14:30:00

## Summary
- Backward references added: 89
- Isolated labels reduced: 42 → 3 (93% reduction)
- LLM validation batches: 5
- Invalid references filtered: 2

## Modifications by Entity Type
- Definitions: 28 entities, 45 refs added
- Axioms: 4 entities, 12 refs added
- Theorems: 15 entities, 18 refs added

## Validation
- Valid: 89 (97.8%)
- Invalid: 2 (2.2%)
```

---

## Next Steps After Running

1. **Review report**: Check validation statistics and filtered references
2. **Build docs**: `make build-docs` to verify MyST syntax
3. **Visual inspection**: Open HTML and check rendered references
4. **Commit**: If satisfied, commit changes to git
5. **Iterate**: Run on other documents if needed

---

**Quick Reference**:
- Purpose: Add backward cross-references
- Input: Single markdown file
- Output: Modified markdown + connectivity report
- Time: 5-15 minutes (depends on document size)
- Validation: Gemini 2.5 Pro (batched)
- Modification: In-place edits

For detailed specification, see `referencer.md`.
