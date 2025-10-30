# Cross-Reference Analysis Quick Start

## TL;DR

Process 69 theorem/lemma entities to fill dependency fields using Gemini 2.5 Pro.

**Status**: Infrastructure ready, batches generated, ready to process.

---

## Quick Commands

### Check Status
```bash
# See total entities and batches
cd /home/guillem/fragile
python scripts/gemini_batch_processor.py \
  docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data
```

Output:
```
Total entities: 81
Batch size: 10
Total batches: 9
```

### Process One Entity (Manual)

```bash
# 1. Read prompt
cat /tmp/gemini_batches/batch_000/lem-boundary-heat-kernel.txt

# 2. In Claude Code, query Gemini:
mcp__gemini-cli__ask-gemini(
  model="gemini-2.5-pro",
  prompt="<paste prompt here>"
)

# 3. Save JSON response to:
/tmp/gemini_batches/batch_000/lem-boundary-heat-kernel.result.json
```

### Process Entire Batch

```bash
# After processing all 10 entities in batch_000, consolidate:
{
  "batch_num": 0,
  "processed": 10,
  "results": {
    "lem-boundary-heat-kernel": { <Gemini JSON> },
    "lem-boundary-uniform-ball": { <Gemini JSON> },
    ...
  }
}

# Save as: /tmp/gemini_batches/batch_000_results.json

# Apply to entity files:
python scripts/apply_cross_references.py \
  /tmp/gemini_batches/batch_000_results.json \
  docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data
```

---

## Batch List

| Batch | Entities | First Entity | Last Entity |
|-------|----------|--------------|-------------|
| 000 | 10 | cor-chain-rule-sigma-reg-var | lem-cubic-patch-derivative-bounds |
| 001 | 6 | lem-cubic-patch-derivative | lem-normalization-difference-bound |
| 002 | 10 | sub-lem-perturbation-positional-bound-reproof | lem-sigma-reg-derivative-bounds |
| 003 | 9 | lem-single-walker-own-status-error | lem-validation-uniform-ball |
| 004 | 6 | sub-lem-bound-sum-total-cloning-probs | unlabeled-lemma-72 |
| 005 | 9 | thm-revival-guarantee | thm-expected-raw-distance-bound |
| 006 | 10 | thm-expected-raw-distance-k1 | thm-perturbation-operator-continuity-reproof |
| 007 | 9 | thm-post-perturbation-status-update-continuity | thm-z-score-norm-bound |

**Total**: 69 entities with valid statements

---

## Expected JSON Format

Each Gemini response should be:

```json
{
  "input_objects": [
    "obj-algorithmic-space-generic",
    "obj-alg-distance",
    "obj-perturbation-measure"
  ],
  "input_axioms": [
    "axiom-raw-value-mean-square-continuity"
  ],
  "input_parameters": [
    "param-kappa-variance"
  ],
  "output_type": "Lipschitz",
  "relations_established": [
    "Establishes Lipschitz continuity of operator X with constant C",
    "Bounds error by quadratic function of status changes"
  ]
}
```

---

## Validation Checklist

For each result:
- [ ] All `input_objects` labels exist in `refined_data/objects/`
- [ ] All `input_axioms` labels exist in `raw_data/axioms/`
- [ ] All `input_parameters` labels exist in `raw_data/parameters/`
- [ ] `output_type` is one of: Bound, Property, Existence, Continuity, Lipschitz, Convergence, Equivalence, Other
- [ ] `relations_established` are specific (not "this theorem establishes...")

---

## Available Framework Labels

### Objects (37)
```
obj-algorithmic-space-generic, obj-alg-distance, obj-perturbation-measure,
obj-cloning-measure, obj-swarm-aggregation-operator-axiomatic,
obj-standardization-lower-bound, obj-revival-state, obj-raw-value-operator,
obj-perturbation-constants, obj-cloning-probability-lipschitz-constants,
obj-smoothed-gaussian-measure, obj-cemetery-state-measure,
obj-boundary-regularity-constants, obj-aggregator-lipschitz-constants,
obj-distance-positional-measures, obj-canonical-fragile-swarm, ...
```

### Axioms (5)
```
axiom-raw-value-mean-square-continuity, axiom-rescale-function,
axiom-bounded-measurement-variance, def-axiom-rescale-function,
def-axiom-bounded-second-moment-perturbation
```

### Parameters (2)
```
param-kappa-variance, param-F-V-ms
```

---

## Files & Locations

### Batches
```
/tmp/gemini_batches/
├── batch_000/
│   ├── batch_info.json
│   ├── lem-boundary-heat-kernel.txt
│   ├── lem-boundary-uniform-ball.txt
│   └── ...
├── batch_001/
│   └── ...
└── ...
```

### Results (to be created)
```
/tmp/gemini_batches/
├── batch_000_results.json
├── batch_001_results.json
└── ...
```

### Entity Files (updated in place)
```
docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/
├── theorems/*.json
├── lemmas/*.json
├── propositions/*.json
└── corollaries/*.json
```

### Scripts
```
/home/guillem/fragile/scripts/
├── cross_reference_raw_data.py      # Pattern matching (already run)
├── gemini_batch_processor.py        # Batch generation (already run)
└── apply_cross_references.py        # Apply results
```

---

## Time Budget

- **Batch 000** (10 entities): ~20-30 minutes
- **All batches** (69 entities): ~3-4 hours

---

## Example Session

```bash
# Start with batch 000
cd /home/guillem/fragile

# Check batch contents
cat /tmp/gemini_batches/batch_000/batch_info.json

# Process first entity
cat /tmp/gemini_batches/batch_000/lem-boundary-heat-kernel.txt

# (In Claude Code, query Gemini with the prompt content)
# Save the JSON response

# After processing all 10, consolidate and apply
python scripts/apply_cross_references.py \
  /tmp/gemini_batches/batch_000_results.json \
  docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data

# Check results
grep -A 5 '"input_objects"' \
  docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/lemmas/lem-boundary-heat-kernel.json
```

---

## Success Criteria

After processing all batches:
- ✅ 69 entities have filled dependency fields
- ✅ ~200-300 total object dependencies identified
- ✅ ~30-50 axiom dependencies identified
- ✅ All output_types categorized
- ✅ ~150-200 specific relations documented
- ✅ Zero validation errors (all labels exist)

---

## Documentation

Full details in:
- **CROSS_REFERENCE_SUMMARY.md** - Current status and infrastructure
- **CROSS_REFERENCE_ANALYSIS_COMPLETE.md** - Complete report
- **CROSS_REFERENCE_WORKFLOW.md** - Detailed workflow guide
- **CROSS_REFERENCE_REPORT.md** - Pattern-matching results

---

**Ready to Process**: ✅
**Next Action**: Process batch_000 (10 entities)
