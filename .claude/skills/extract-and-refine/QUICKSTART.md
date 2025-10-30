# Extract-and-Refine - Quick Start

## TL;DR

Extract mathematical entities from markdown → validated JSON in 3 commands.

```bash
# 1. Parse document (~15 sec)
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/01_fragile_gas_framework.md --no-llm

# 2. Cross-reference (~3 sec)
python -m fragile.agents.cross_reference_analyzer docs/source/1_euclidean_gas/01_fragile_gas_framework

# 3. Refine (load agent, ~10-30 min)
# Load document-refiner agent: Refine docs/source/1_euclidean_gas/01_fragile_gas_framework
```

**Output**: `refined_data/` directory with validated JSON entities

---

## Common Use Cases

### Use Case 1: Quick Validation (No LLM, 15 sec)

```bash
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/03_cloning.md \
  --no-llm
```

**Output**: `raw_data/` + `data/statistics.json`
**Check**: `cat docs/source/1_euclidean_gas/03_cloning/data/statistics.json | grep validation_errors`
**Expected**: `"validation_errors": 0`

---

### Use Case 2: Full Pipeline (With LLM, 20-45 min)

```bash
# Step 1: Parse with LLM (3-5 min)
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/03_cloning.md

# Step 2: Cross-reference with LLM (5-10 min)
python -m fragile.agents.cross_reference_analyzer \
  docs/source/1_euclidean_gas/03_cloning

# Step 3: Refine (10-30 min)
# In Claude Code:
# Load document-refiner agent.
# Refine: docs/source/1_euclidean_gas/03_cloning
```

**Output**: Complete `refined_data/` directory
**Check**: `ls -lh docs/source/1_euclidean_gas/03_cloning/refined_data/objects/`

---

### Use Case 3: Batch Processing (Directory)

```bash
# Parse all documents in directory
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/ \
  --no-llm

# Each document gets own subdirectory
find docs/source/1_euclidean_gas -name "statistics.json"
```

---

### Use Case 4: Incremental Re-extraction

```bash
# After editing document, quickly re-parse
python -m fragile.agents.math_document_parser \
  docs/source/1_euclidean_gas/03_cloning.md \
  --no-llm

# Check what changed
cat docs/source/1_euclidean_gas/03_cloning/data/statistics.json
```

---

## Verification Commands

### After Parsing

```bash
# Check statistics
cat docs/source/.../data/statistics.json

# Expected output:
# {
#   "objects_created": 36,
#   "theorems_created": 59,
#   "validation_errors": 0
# }

# List extracted objects
ls -1 docs/source/.../raw_data/objects/ | wc -l
```

### After Cross-Referencing

```bash
# Check relationships report
cat docs/source/.../relationships/REPORT.md

# Count relationships
ls -1 docs/source/.../relationships/rel-*.json | wc -l

# Check enhanced theorems have dependencies
grep "input_objects" docs/source/.../theorems/thm-*.json | head -5
```

### After Refinement

```bash
# Check refined statistics
cat docs/source/.../reports/statistics/refined_statistics.json

# List refined entities
find docs/source/.../refined_data -name "*.json" | wc -l

# Spot-check a refined object
cat docs/source/.../refined_data/objects/obj-euclidean-gas.json | python -m json.tool
```

---

## Quick Troubleshooting

### Problem: "Found 0 directives"

```bash
# Check directive syntax in document
grep -n "^:::" docs/source/.../document.md | head -20

# Should show ::: (3 colons), not :::: (4 colons)
```

### Problem: Validation errors

```bash
# See which files failed
python -m fragile.agents.math_document_parser docs/source/.../document.md --no-llm 2>&1 | grep "Failed to create"

# Check specific error
cat docs/source/.../data/validation_errors.json | python -m json.tool
```

### Problem: Cross-referencer finds no relationships

```bash
# Verify parser ran successfully
ls -lh docs/source/.../raw_data/

# If empty, run parser first:
python -m fragile.agents.math_document_parser docs/source/.../document.md --no-llm
```

---

## Parallel Processing

Process multiple documents simultaneously:

```bash
# Terminal 1
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/03_cloning.md --no-llm &

# Terminal 2
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/04_convergence.md --no-llm &

# Terminal 3
python -m fragile.agents.math_document_parser docs/source/1_euclidean_gas/05_mean_field.md --no-llm &

# Wait for all
wait
```

All 3 complete in ~15 seconds (parallel execution).

---

## Next Steps

1. **Check output**: Verify `refined_data/` directory created
2. **Review entities**: Spot-check a few JSON files
3. **Build registry**: Use [registry-management](../registry-management/QUICKSTART.md) skill
4. **Develop proofs**: Use [proof-validation](../proof-validation/QUICKSTART.md) skill

---

**Full Documentation**: [SKILL.md](./SKILL.md)
**Step-by-Step**: [WORKFLOW.md](./WORKFLOW.md)
**Issues**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
**Examples**: [examples/](./examples/)
