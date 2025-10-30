# Cross-Reference Analysis Workflow

## Overview

This document describes the workflow for cross-referencing all theorem-like entities in the Fragile Gas Framework using Gemini 2.5 Pro.

## Entities to Process

- **Total Entities**: 81
  - Theorems: 30
  - Lemmas: 45
  - Propositions: 3
  - Corollaries: 3

## Batches Generated

All prompts have been generated in `/tmp/gemini_batches/`:
- `batch_000`: 10 entities (corollaries + first lemmas)
- `batch_001`: 6 entities
- `batch_002-007`: ~9-10 entities each
- `batch_008`: 0 entities (all processed in earlier batches)

Total: **69 entities with valid statements** (12 entities skipped due to missing statements)

## Fields to Fill

For each entity JSON file, we need to fill:

1. **input_objects**: List of `obj-*` labels
2. **input_axioms**: List of `axiom-*` labels
3. **input_parameters**: List of `param-*` labels
4. **output_type**: One of: Bound | Property | Existence | Continuity | Lipschitz | Convergence | Equivalence | Other
5. **relations_established**: List of strings describing specific relationships

## Available Entities in Framework

### Objects (37 total)
Key objects include:
- `obj-algorithmic-space-generic`
- `obj-alg-distance`
- `obj-perturbation-measure`
- `obj-cloning-measure`
- `obj-swarm-aggregation-operator-axiomatic`
- `obj-standardization-lower-bound`
- `obj-revival-state`
- ... (see `refined_data/objects/` for complete list)

### Axioms (5 total)
- `axiom-raw-value-mean-square-continuity`
- `axiom-rescale-function`
- `axiom-bounded-measurement-variance`
- `def-axiom-rescale-function`
- `def-axiom-bounded-second-moment-perturbation`

### Parameters (2 total)
- `param-kappa-variance`: Maximum Measurement Variance
- `param-F-V-ms`: Expected Squared Value Error Bound

## Processing Workflow

### Option 1: Manual Processing (Recommended for Quality)

For each batch:

1. Read prompt from `/tmp/gemini_batches/batch_NNN/<label>.txt`
2. Query Gemini 2.5 Pro with the prompt
3. Parse JSON response
4. Validate labels against available entities
5. Save result to `/tmp/gemini_batches/batch_NNN/<label>.result.json`
6. Apply results using `scripts/apply_cross_references.py`

### Option 2: Automated Batch Processing

**NOTE**: Requires manual review of results before applying.

1. Process entire batch with Gemini
2. Collect all responses
3. Run validation checks
4. Review and approve
5. Apply to entity files

## Sample Processing

### Example: lem-boundary-heat-kernel

**Statement**: Heat-kernel death probability is Lipschitz with constant ≲ 1/σ

**Gemini Analysis**:
```json
{
  "input_objects": [
    "obj-algorithmic-space-generic",
    "obj-alg-distance",
    "obj-smoothed-gaussian-measure",
    "obj-cemetery-state-measure",
    "obj-boundary-regularity-constants",
    "obj-perturbation-constants"
  ],
  "input_axioms": [],
  "input_parameters": [],
  "output_type": "Lipschitz",
  "relations_established": [
    "Establishes that the heat-kernel-smoothed probability of entering an invalid set (P_sigma(x)) is Lipschitz continuous.",
    "Bounds the Lipschitz constant of the death probability by a term proportional to Per(E)/σ.",
    "Relates the Lipschitz constant of the death probability (L_death) in the algorithmic metric to the perimeter of the invalid set and the Lipschitz constant of the coordinate map (L_phi)."
  ]
}
```

**Application**: Updates `raw_data/lemmas/lem-boundary-heat-kernel.json` with these fields.

## Scripts Created

1. **`scripts/cross_reference_raw_data.py`**: Basic pattern-matching cross-referencer
2. **`scripts/llm_cross_reference.py`**: LLM prompt generator
3. **`scripts/batch_cross_reference.py`**: Batch prompt generator
4. **`scripts/gemini_batch_processor.py`**: Full batch processing system
5. **`scripts/apply_cross_references.py`**: Apply JSON results to entity files

## Next Steps

To complete the cross-reference analysis:

1. **Process Batch 000** (10 entities) - Demonstrate workflow
2. **Review Results** - Validate accuracy
3. **Process Remaining Batches** (batches 001-007)
4. **Generate Final Report** - Statistics and validation

## Quality Control

Each result should be:
- ✓ Using only labels from available entities
- ✓ Including ALL mathematical objects mentioned
- ✓ Correctly categorizing output_type
- ✓ Providing specific relations_established (not generic)
- ✓ Validated against framework context

## Estimated Time

- Per entity (with review): ~2-3 minutes
- Total for 69 entities: ~2-3 hours
- Batch 000 (10 entities): ~20-30 minutes

## Status

- [x] Entity registry built (39 objects, 21 axioms, 8 parameters)
- [x] All prompts generated (69 valid entities)
- [x] Batch structure created
- [x] Sample processing demonstrated (lem-boundary-heat-kernel)
- [ ] Process batch_000 (10 entities)
- [ ] Process batches 001-007 (59 entities)
- [ ] Generate final cross-reference report
- [ ] Validate all dependencies exist
