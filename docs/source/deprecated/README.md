# Deprecated W₂ Contraction Proof Documents

**⚠️ DO NOT USE THESE DOCUMENTS ⚠️**

These documents contain errors, incomplete analyses, or have been superseded by the complete proof.

## Use This Instead

**Complete, Correct Proof:** [../03_wasserstein_contraction_complete.md](../03_wasserstein_contraction_complete.md)

## Documents in This Folder

### Fundamentally Flawed (Do Not Reference)

- **03_A_wasserstein_contraction.md** - Incomplete case analysis, missing critical lemmas
- **03_B_companion_contraction.md** - WRONG independence assumption for companion selection
- **03_D_mixed_fitness_case.md** - Partial analysis only, incomplete

### Partial Work (Consolidated into Complete Proof)

- **03_C_wasserstein_single_pair.md** - Single-pair lemma structure (partial, now in Section 5)
- **03_E_case_b_contraction.md** - Case B attempt with SCALING ERROR (fixed in Section 4)
- **03_F_outlier_alignment.md** - Lemma statement with proof skeleton only (full proof in Section 2)

## Historical Context

These documents represent the iterative development process that led to the final proof. They are preserved for:
- Historical reference
- Understanding the evolution of the proof strategy
- Documenting errors that were identified and corrected

The key breakthroughs documented in [../00_W2_PROOF_PROGRESS_SUMMARY.md](../00_W2_PROOF_PROGRESS_SUMMARY.md) explain how the correct proof structure was discovered.

## What Was Wrong

### Error 1: Independence Assumption (03_B)
- **Wrong:** Assumed companion selections $c_x$ and $c_y$ are independent
- **Why Wrong:** The synchronous coupling uses the SAME matching $M$ for both swarms
- **Corrected In:** Section 1 of complete proof (proper synchronous coupling)

### Error 2: Scaling Mismatch (03_E)
- **Wrong:** Tried to prove $D_{ii} - D_{ji} \geq \alpha(D_{ii} + D_{jj})$
- **Why Wrong:** LHS scales as $L \cdot R_H$, RHS scales as $L^2$ (dimensional mismatch)
- **Corrected In:** Section 4 of complete proof (uses Outlier Alignment)

### Error 3: Incomplete Analysis (03_A, 03_D)
- **Wrong:** Missing critical lemmas, partial case analysis
- **Corrected In:** Complete proof has all 8 sections with rigorous proofs

## For Reviewers

If you are reviewing the W₂ contraction proof:
1. **IGNORE** all documents in this folder
2. **READ** [../03_wasserstein_contraction_complete.md](../03_wasserstein_contraction_complete.md)
3. **CHECK** [../00_W2_PROOF_COMPLETION_SUMMARY.md](../00_W2_PROOF_COMPLETION_SUMMARY.md) for summary

The complete proof has been verified for:
- Mathematical rigor
- Internal consistency
- Proper citations to framework documents
- N-uniformity of all constants
- Publication readiness

---

**Status:** These documents are DEPRECATED and should not be cited or used.
**Replacement:** [../03_wasserstein_contraction_complete.md](../03_wasserstein_contraction_complete.md)
