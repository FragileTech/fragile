# Section 5 Extraction Report
**Document**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md`
**Section**: 5. Algorithmic Noise Measures (lines 1112-1211)
**Date**: 2025-10-27

## Summary

Successfully extracted mathematical entities from Section 5 (Algorithmic Noise Measures).

### Entities Extracted

#### Definitions (2 existing, verified)
1. **Perturbation Measure** (`def-perturbation-measure`)
   - File: `objects/obj-perturbation-measure.json` (already exists)
   - Content: Valid Noise Measure for perturbation step

2. **Cloning Measure** (`def-cloning-measure`)
   - File: `objects/obj-cloning-measure.json` (already exists)
   - Content: Valid Noise Measure for cloning step

#### Lemmas (4 newly created)
1. **Validation of the Heat Kernel** (`lem-validation-heat-kernel`)
   - File: `theorems/lem-validation-heat-kernel.json` ✓ CREATED
   - Content: Proof that heat kernel satisfies required axioms
   - Status: expanded (proof provided in document)

2. **Validation of the Uniform Ball Measure** (`lem-validation-uniform-ball`)
   - File: `theorems/lem-validation-uniform-ball.json` ✓ CREATED
   - Content: Proof that uniform ball measure satisfies required axioms
   - Status: expanded (proof provided in document)

3. **Uniform-ball death probability is Lipschitz** (`lem-boundary-uniform-ball`)
   - File: `theorems/lem-boundary-uniform-ball.json` ✓ CREATED
   - Content: BV/perimeter Lipschitz bound for uniform-ball death probability
   - Status: expanded (proof provided in document)

4. **Heat-kernel death probability is Lipschitz** (`lem-boundary-heat-kernel`)
   - File: `theorems/lem-boundary-heat-kernel.json` ✓ CREATED
   - Content: Heat-kernel Lipschitz bound via BV smoothing
   - Status: expanded (proof provided in document)

#### Remarks (2 identified, not yet extracted)
1. **Explicit Constants for Standardization Bounds** (line 1135)
   - Numbered as "11.3.8 Remark" (appears to be mislabeled)
   - Content: Reference table of constants for standardization bounds

2. **Projection choice** (line 1191)
   - Embedded within proof of lem-boundary-uniform-ball
   - Content: Note about taking φ=Id for simplification

## Files Created

### New Files (4)
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/lem-validation-heat-kernel.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/lem-validation-uniform-ball.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/lem-boundary-uniform-ball.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/lem-boundary-heat-kernel.json`

### Existing Files (2)
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/obj-perturbation-measure.json`
- `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/obj-cloning-measure.json`

## Extraction Statistics

| Entity Type | Count | Status |
|-------------|-------|--------|
| Definitions | 2 | Already existed |
| Lemmas | 4 | Newly created |
| Remarks | 2 | Identified (not extracted) |
| **Total** | **8** | **6 JSON files** |

## Dependencies Established

### Lemma DAG Edges
- `lem-validation-uniform-ball` → `lem-boundary-uniform-ball`
- `lem-validation-heat-kernel` → `lem-boundary-heat-kernel`

### Input Dependencies
All 4 lemmas depend on:
- `obj-perturbation-measure` (object)
- `def-axiom-bounded-second-moment-perturbation` (axiom)
- `def-axiom-boundary-regularity` (axiom)

## Notes

1. **Section Numbering Issue**: The section uses "4.1" and "4.2" subsection numbers but is labeled as "Section 5". This appears to be a documentation inconsistency.

2. **Remark 11.3.8**: This remark has an unusual numbering scheme (11.3.8) that doesn't match the section structure. It may have been copied from another section.

3. **Proof Status**: All 4 lemmas have proofs provided in the document, so their status is set to "expanded" rather than "unproven".

4. **Remarks Not Extracted**: The 2 remarks were identified but not extracted to JSON files. The remarks directory exists but format unclear from existing examples.

## Verification

To verify the extraction:
```bash
# Check created files
ls -lh /home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/lem-*.json | grep -E "(heat-kernel|uniform-ball|validation)"

# Count entities
echo "Theorems: $(ls /home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/theorems/*.json | wc -l)"
echo "Objects: $(ls /home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/*.json | wc -l)"
```
