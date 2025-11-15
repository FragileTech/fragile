# Cross-Reference Analysis: 02_euclidean_gas.md

**Document**: `docs/source/1_euclidean_gas/02_euclidean_gas.md`
**Date**: 2025-11-12
**Status**: WELL-REFERENCED (Existing: 63 framework references)

## Summary

The document `02_euclidean_gas.md` is already well-connected to the framework document `01_fragile_gas_framework.md` with **63 existing cross-references**. This analysis identifies opportunities for additional strategic enrichment.

---

## Current State: Existing Framework References (30 shown)

The document already references:
- ✓ Core structural concepts: `def-walker`, `def-swarm-and-state-space`, `def-alive-dead-sets`
- ✓ All viability axioms: `axiom-guaranteed-revival`, `axiom-boundary-regularity`, `axiom-boundary-smoothness`
- ✓ All environmental axioms: `axiom-environmental-richness`, `axiom-reward-regularity`
- ✓ Key algorithmic axioms: `axiom-sufficient-amplification`, `axiom-non-degenerate-noise`, `axiom-geometric-consistency`
- ✓ Measurement pipeline: `def-statistical-properties-measurement`, `def-canonical-logistic-rescale-function-example`
- ✓ Operators: `def-perturbation-measure`, `def-status-update-operator`
- ✓ Algorithm specification: `def-fragile-gas-algorithm`, `def-fragile-swarm-instantiation`

---

## Enrichment Opportunities

### High-Value Additions (Strategic Locations)

**1. Section 0 (TLDR) - Line 9**
- **Current**: "rigorously verifies that the Euclidean Gas satisfies all framework axioms ({prf:ref}`def-fragile-gas-algorithm`)"
- **Enrichment Opportunity**: Add reference to instantiation concept
- **Suggested**: "...all framework axioms ({prf:ref}`def-fragile-gas-algorithm`, {prf:ref}`def-fragile-swarm-instantiation`)"

**2. Section 1.1 (Goal and Scope) - Line 19**
- **Current**: "Each walker $w_i = (x_i ({prf:ref}`def-walker`), v_i, s_i)$"
- **Issue**: Reference placement is awkward (inside tuple notation)
- **Suggested**: "Each walker $w_i = (x_i, v_i, s_i)$ ({prf:ref}`def-walker`)"

**3. Section 3.1 (Algorithm) - Line 160**
- **Current**: "If all walkers are dead (no alive indices in $\mathcal A_t$) return the cemetery state ({prf:ref}`def-cemetery-state`)"
- **Current reference**: Uses `def-cemetery-state`
- **Issue**: Label should be `def-cemetery-state-measure` (from glossary)
- **Action**: Verify correct label

**4. Section 3.3 (Position-Velocity Foundations) - Line 373**
- **Current**: "**Walker state** $w_i=(x_i,v_i,s_i)\in\mathcal X\times\mathbb R^d\times\{0,1\}$"
- **Enrichment**: Add reference to walker definition
- **Suggested**: "**Walker state** $w_i=(x_i,v_i,s_i)$ ({prf:ref}`def-walker`)"

**5. Section 4.1 (Viability Axioms) - Line 548**
- **Current**: "1. **Guaranteed Revival.**"
- **Already has**: Multiple references in surrounding context
- **Status**: Well-referenced

**6. Section 4.3 (Algorithmic Axioms) - Line 1230**
- **Current**: "4. **Non-degenerate noise ({prf:ref}`def-axiom-non-degenerate-noise`).**"
- **Status**: Well-referenced

---

## Within-Document References (Phase 1)

### Mathematical Entities Defined in 02_euclidean_gas.md

The document defines 28 new mathematical entities:
1. `alg-euclidean-gas` (Algorithm, line 155)
2. `lem-squashing-properties-generic` (Lemma, line 420)
3. `lem-projection-lipschitz` (Lemma, line 449)
4. `lem-sasaki-kinetic-lipschitz` (Lemma, line 557)
... (25 more)

### Backward Within-Document Cross-References

**Strategy**: Add references from later lemmas/theorems back to earlier foundational lemmas.

**Example Pattern**:
- Line 596: `lem-sasaki-kinetic-lipschitz` references `lem-squashing-properties-generic` ✓ (Already present)
- Line 630: `lem-euclidean-boundary-holder` references `lem-sasaki-kinetic-lipschitz` ✓ (Already present)
- Line 1182: `lem-sasaki-total-squared-error-stable` references `lem-sasaki-single-walker-positional-error` ✓ (Already present)

**Assessment**: Within-document backward references are **already well-established**.

---

## Recommendations

### Priority 1: Fix Awkward Reference Placements

1. **Line 19**: Move walker reference outside tuple notation
   ```markdown
   - Current: $w_i = (x_i ({prf:ref}`def-walker`), v_i, s_i)$
   - Fixed: $w_i = (x_i, v_i, s_i)$ ({prf:ref}`def-walker`)
   ```

### Priority 2: Add Missing Strategic References

2. **Line 9 (TLDR)**: Add instantiation reference
   ```markdown
   ({prf:ref}`def-fragile-gas-algorithm`, {prf:ref}`def-fragile-swarm-instantiation`)
   ```

3. **Line 373 (Walker State Definition)**: Add backward reference
   ```markdown
   **Walker state** $w_i=(x_i,v_i,s_i)$ ({prf:ref}`def-walker`)
   ```

### Priority 3: Verify Label Consistency

4. **Line 160**: Check if `def-cemetery-state` should be `def-cemetery-state-measure`
   - Glossary shows: `def-cemetery-state-measure` (canonical label)
   - Document uses: `def-cemetery-state` (may be incorrect)
   - **Action**: Verify against 01_fragile_gas_framework.md and correct if needed

---

## Statistics

### Current Cross-Reference Density
- **Total framework references**: 63
- **Document length**: 2480 lines
- **Density**: 2.5% (1 reference per 39 lines)
- **Assessment**: Well-referenced (healthy density for technical mathematics)

### Unlinked Mentions (Without {prf:ref})
- "walker": 30 mentions (many are in math notation/variables, not concept references)
- "swarm": 22 mentions (similar - mostly in "$\mathcal S$" notation)
- "axiom": 21 mentions (mostly in prose, selective referencing appropriate)
- "alive set": 7 mentions (mostly in notation "$\mathcal A_t$")

**Assessment**: These unlinked mentions are primarily in mathematical notation or passing references where adding `{prf:ref}` would be cluttered. The document appropriately uses references for **first introductions** and **significant conceptual invocations**.

---

## Conclusion

**The document `02_euclidean_gas.md` is already well-cross-referenced** with 63 backward references to framework concepts. The main opportunities are:

1. **Cosmetic improvements**: Fix awkward reference placements (Priority 1)
2. **Strategic additions**: Add 2-3 high-value references (Priority 2)
3. **Label verification**: Ensure consistency with glossary labels (Priority 3)

**Estimated Impact**: Modest improvements (3-5 additional references)

**Backward-Only Constraint**: ✓ All existing and proposed references point to earlier material (01_fragile_gas_framework.md or earlier sections of 02_euclidean_gas.md)

---

## Next Steps

1. Apply Priority 1 fixes (awkward placements)
2. Add Priority 2 strategic references
3. Verify Priority 3 label consistency
4. Generate final enrichment report with statistics

**Total Expected Additions**: ~5 modifications
**Total Expected References After Enrichment**: ~68

