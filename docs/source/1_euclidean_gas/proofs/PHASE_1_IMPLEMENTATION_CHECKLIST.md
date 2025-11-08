# Phase 1 Implementation Checklist

**Date**: 2025-11-07
**Status**: In Progress
**Target**: Fix Issues #1, #2, #3 from dual review

---

## Deliverables Status

### Completed ✅

- [x] **Issue #2 Fix**: Revised Section 2.4 (Revival Entropy Expansion)
  - Location: `proof_discrete_kl_convergence_REVISED.md`, lines 1360-1422
  - Removed invalid HWI bound
  - Replaced with direct entropy analysis
  - Added new Lemma 2.4 (lem-revival-entropy-expansion-revised)

- [x] **Issue #2 Fix**: Updated Section 2.5 (Combined Cloning Bound)
  - Location: `proof_discrete_kl_convergence_REVISED.md`, lines 1426-1468
  - Updated Lemma 2.5 (lem-discrete-cloning-entropy-bound-revised)
  - Changed: C_HWI W_2 term → C_revival τ term
  - Updated physical interpretation

- [x] **Issue #1 Fix**: Complete Section 3.7 Rewrite
  - Created: `SECTION_3_7_REVISED.md` (standalone document)
  - Implements multi-step Grönwall analysis
  - Establishes O(τ) residual behavior
  - Ready for integration

- [x] **Documentation**: Phase 1 Revisions Summary
  - Created: `PHASE_1_REVISIONS_SUMMARY.md`
  - Complete specification of all three fixes
  - Mathematical justification
  - References and verification

- [x] **Documentation**: Revised Section 3.7
  - Created: `SECTION_3_7_REVISED.md`
  - Complete replacement for original Section 3.7
  - Integration instructions included

- [x] **Documentation**: Implementation Checklist
  - Created: `PHASE_1_IMPLEMENTATION_CHECKLIST.md` (this document)

### Pending ⏳

- [ ] **Issue #3 Fix**: Add Lemma 3.6 (Wasserstein bound from Foster-Lyapunov)
  - Target location: Before Section 3.7 in proof_discrete_kl_convergence_REVISED.md
  - Insert as new subsection 3.6
  - Renumber existing 3.6 → 3.7, 3.7 → 3.8, etc.
  - See `PHASE_1_REVISIONS_SUMMARY.md` for complete lemma text

- [ ] **Integration**: Replace Section 3.7 in main proof
  - Delete: Lines 1660-2080 in proof_discrete_kl_convergence_REVISED.md
  - Insert: Content from `SECTION_3_7_REVISED.md`
  - Update: Cross-references to new theorem labels

- [ ] **Update**: Main Theorem Statement (Theorem 4.1)
  - Add O(τ) residual term to theorem statement
  - Update LSI constant definition (C_LSI = 1/β_net, NOT 1/(β_net - C_clone))
  - Add note about continuous-time limit

- [ ] **Update**: Section 4.1 (Discrete-to-Continuous Conversion)
  - Use revised Lyapunov bound from Theorem 3.7 (revised)
  - Include O(τ) residual in conversion

- [ ] **Update**: Section 6.2 (Explicit Constants)
  - Update C_clone definition: C_clone = C_kill + C_revival
  - Remove C_HWI from C_clone definition
  - Add C_revival = O(β V_fit,max N log N)
  - Update β definition: β_net = c_kin γ (NOT c_kin γ - C_clone)

- [ ] **Update**: Section 6.3 (Verification Checklist)
  - Mark Issue #1 as FIXED
  - Mark Issue #2 as FIXED
  - Mark Issue #3 as FIXED
  - Update publication readiness score

- [ ] **Verification**: Cross-Reference Audit
  - Check all {prf:ref} directives resolve correctly
  - Verify theorem numbering consistency
  - Update any broken references

- [ ] **Testing**: Jupyter Book Build
  - Run `make build-docs` to verify all MyST markdown compiles
  - Check for any rendering errors
  - Verify mathematical notation displays correctly

---

## File Modifications Required

### Files to Edit

1. **`proof_discrete_kl_convergence_REVISED.md`** (Main proof document):
   - [x] Section 2.4 (lines 1360-1422) - DONE
   - [x] Section 2.5 (lines 1426-1468) - DONE
   - [ ] Insert new Section 3.6 (Lemma: Wasserstein bound) - TODO
   - [ ] Replace Section 3.7 (lines ~1660-2080) with SECTION_3_7_REVISED.md - TODO
   - [ ] Update Theorem 4.1 (Section 4.2) - TODO
   - [ ] Update Section 4.1 (Discrete-to-continuous conversion) - TODO
   - [ ] Update Section 6.2 (Explicit constants) - TODO
   - [ ] Update Section 6.3 (Verification checklist) - TODO

### Files Created (Reference Documents)

2. **`PHASE_1_REVISIONS_SUMMARY.md`** ✅
   - Complete specification of fixes
   - Mathematical justification
   - References

3. **`SECTION_3_7_REVISED.md`** ✅
   - Complete Section 3.7 rewrite
   - Integration instructions

4. **`PHASE_1_IMPLEMENTATION_CHECKLIST.md`** ✅
   - This document
   - Progress tracking

---

## Detailed Task Breakdown

### Task 1: Insert Lemma 3.6 (Issue #3 Fix)

**What to do**:
1. Open `proof_discrete_kl_convergence_REVISED.md`
2. Find Section 3.6 (currently "Optimal Coupling")
3. Renumber 3.6 → 3.7
4. Insert NEW Section 3.6 with content from `PHASE_1_REVISIONS_SUMMARY.md`, Issue #3 section

**Content to insert** (from summary):
```markdown
### 3.6 Time-Dependent Wasserstein Bound (NEW - Issue #3 Fix)

[Complete lemma text from PHASE_1_REVISIONS_SUMMARY.md, lines 249-280]
```

**Estimated time**: 30 minutes

---

### Task 2: Replace Section 3.7

**What to do**:
1. Open `proof_discrete_kl_convergence_REVISED.md`
2. Locate Section 3.7 (after renumbering from Task 1, this will be ~3.8)
3. Delete entire section (approximately lines 1660-2080)
4. Insert complete content from `SECTION_3_7_REVISED.md`
5. Renumber as Section 3.7

**Estimated time**: 1 hour (careful editing + verification)

---

### Task 3: Update Main Theorem (Theorem 4.1)

**What to do**:
1. Locate Theorem 4.1 in Section 4
2. Replace theorem statement with revised version from `PHASE_1_REVISIONS_SUMMARY.md`

**Old statement**:
```latex
D_KL(μ_t || π_QSD) ≤ e^{-t/C_LSI} D_KL(μ_0 || π_QSD) + O(τ)

with C_LSI = 1/β and β = c_kin γ - C_clone  [WRONG]
```

**New statement**:
```latex
D_KL(μ_t || π_QSD) ≤ e^{-β_net t} D_KL(μ_0 || π_QSD) + C_clone/(c_kin γ)

with β_net = c_kin γ and C_clone = C_kill + C_revival
```

**Estimated time**: 1 hour (update proof, add notes about O(τ) residual)

---

### Task 4: Update Section 4.1 (Discrete-to-Continuous)

**What to do**:
1. Locate Lemma 4.1
2. Update to use revised Lyapunov bound from Theorem 3.7
3. Add explicit handling of O(τ) residual term
4. Verify conversion to continuous time includes residual

**Estimated time**: 1 hour

---

### Task 5: Update Section 6.2 (Constants)

**What to do**:
1. Locate Table of Constants
2. Update β definition: Remove subtraction, set β_net = c_kin γ
3. Update C_clone definition: C_clone = C_kill + C_revival
4. Add new entry for C_revival = O(β V_fit,max N log N)
5. Remove C_HWI from cloning expansion (only used in Wasserstein bound now)
6. Update C_LSI = 1/β_net (not 1/(β_net - C_clone))

**Estimated time**: 1 hour

---

### Task 6: Update Verification Checklist (Section 6.3)

**What to do**:
1. Mark Issues #1, #2, #3 as FIXED
2. Update rigor score (9.4/10 → revised score accounting for fixes)
3. Add note: "Phase 1 revisions completed, Issues #4-10 remain"
4. Update publication readiness assessment

**Estimated time**: 30 minutes

---

### Task 7: Cross-Reference Audit

**What to do**:
1. Search for all instances of:
   - `{prf:ref}`thm-lyapunov-contraction`` → Update to `thm-lyapunov-contraction-revised`
   - `{prf:ref}`lem-discrete-cloning-entropy-bound`` → Update to `lem-discrete-cloning-entropy-bound-revised`
   - Any other broken references from section renumbering
2. Run global search-replace to fix
3. Verify all cross-references resolve

**Estimated time**: 1 hour

---

### Task 8: Jupyter Book Build Test

**What to do**:
1. Run `make build-docs` from repository root
2. Check build output for errors
3. Open HTML output and verify:
   - All sections render correctly
   - Mathematical notation displays properly
   - Cross-references link correctly
   - No broken MyST directives
4. Fix any rendering issues

**Estimated time**: 1 hour (including fixes)

---

## Total Estimated Time

**Completed work**: 4 hours ✅
**Remaining work**: 7.5 hours ⏳

**Total Phase 1 effort**: ~11.5 hours

---

## Success Criteria

Phase 1 is complete when ALL of the following are satisfied:

- [x] Issue #2 fixed: No invalid HWI entropy bound
- [ ] Issue #1 fixed: No invalid rate subtraction (β = c_kin γ - C_clone removed)
- [ ] Issue #3 fixed: Time-dependent Wasserstein bound established with explicit formula
- [ ] All theorem statements mathematically rigorous
- [ ] All cross-references resolve correctly
- [ ] Jupyter Book builds without errors
- [ ] Proof remains internally consistent

**Current status**: 2/7 criteria met (29%)

---

## Next Steps After Phase 1

Once Phase 1 is complete:

1. **Submit to Dual Review** for verification of fixes
2. **Proceed to Phase 2**:
   - Issue #4: BAOAB backward error analysis (CRITICAL)
   - Issue #5: Log-concavity of π_QSD (CRITICAL)
   - Issues #6-10: Minor technical gaps (MAJOR)
3. **Final polish** (Section 6.6 tasks)
4. **Publication submission**

---

## Notes

**Key Mathematical Changes**:
- Original incorrect: β = c_kin γ - C_clone (subtracts additive from multiplicative)
- Revised correct: β_net = c_kin γ, with C_clone in additive residual
- Result: Exponential convergence to O(τ) neighborhood, NOT exact convergence

**Files Architecture**:
- `proof_discrete_kl_convergence_REVISED.md`: Main proof (work in progress)
- `PHASE_1_REVISIONS_SUMMARY.md`: Complete specification (reference)
- `SECTION_3_7_REVISED.md`: Complete Section 3.7 (ready to insert)
- `PHASE_1_IMPLEMENTATION_CHECKLIST.md`: This checklist (tracking)

**Backup**:
- Original proof preserved as `proof_discrete_kl_convergence.md` (unchanged)
- Can compare original vs revised using diff tools

---

## Contact

For questions or clarifications:
- Review dual review summary: `docs/source/1_euclidean_gas/proofs/reviewer/review_20251107_1341_proof_discrete_kl_convergence.md`
- Consult Phase 1 summary: `PHASE_1_REVISIONS_SUMMARY.md`

---

**Last Updated**: 2025-11-07
**Phase**: 1 (Issues #1-3)
**Status**: In Progress (29% complete)

---

**END OF CHECKLIST**
