# Deprecated Documents

**⚠️ DO NOT USE THESE DOCUMENTS ⚠️**

This directory contains **intermediate working versions** and **superseded documents** that have been replaced by the final, complete proofs.

---

## Status and Summary Documents (00_X series)

These are **working/status documents** from the W₂ contraction proof development. They track progress, identify issues, and document resolutions.

### Progress Tracking (Historical)
- `00_W2_PROOF_PROGRESS_SUMMARY.md` - Archived progress summary (breakthrough session)
- `00_NEXT_SESSION_PLAN.md` - Archived session plan (completed)
- `00_FINAL_STATUS_SUMMARY.md` - Status summary from W₂ completion

### Issue Management
- `00_GEMINI_REVIEW_RESPONSE.md` - Response to Gemini's critical review
- `00_FORMALIZATION_ROADMAP.md` - Roadmap for addressing Gemini's issues
- `00_W2_FORMALIZATION_COMPLETE.md` - Confirmation of formalization completion

### Completion Summaries
- `00_W2_PROOF_COMPLETION_SUMMARY.md` - W₂ proof completion summary

**Why deprecated**: These are **status documents**, not proofs. They track the development process but don't contain the actual mathematical content.

**Superseded by**: `../03_wasserstein_contraction_complete.md` - The complete, final W₂ contraction proof

---

## Mean-Field LSI Working Versions (10_X series)

These documents represent the **iterative development** of the mean-field proof before all gaps were resolved.

### Early Attempts (Before Gap Resolutions)
- `10_A_lemma5.2_revised.md` - First revision attempt
- `10_B_lemma5.2_complete.md` - Early "complete" version (had gaps)
- `10_C_lemma5.2_meanfield.md` - Initial mean-field approach
- `10_D_step3_cloning_bounds.md` - Framework for cloning bounds
- `10_E_lemma5.2_corrected.md` - Corrected for noise double-counting issue

### Status Documents (Mid-Development)
- `10_F_status_summary.md` - Mid-session status
- `10_FINAL_STATUS.md` - Status before discovering final issues
- `10_K_FINAL_ASSESSMENT.md` - Assessment showing remaining gaps

### Individual Gap Working Documents
- `10_G_gap2_fitness_potential.md` - Fitness-potential anti-correlation work
- `10_H_gap3_entropy_variance.md` - Entropy variance bound attempt

### Late Iterations (Before Final Resolution)
- `10_I_lemma5.2_complete_final.md` - Near-final version
- `10_J_lemma5.2_final_corrected.md` - Corrected symmetrization
- `10_L_lemma5.2_publication_ready.md` - Attempted publication version (still had critical issues)

**Why deprecated**: All these documents contained unresolved mathematical gaps:
- **Gap #1 (CRITICAL)**: No valid inequality for $(e^{-x} - 1)x$
- **Gap #3 (MAJOR)**: Misapplication of Shannon's entropy power inequality

**What was wrong**:
- Gap #1: Attempted to use pointwise inequality that doesn't exist
- Gap #3: Applied EPI to cross-entropy instead of true entropy
- Both had incorrect/incomplete symmetrization arguments

**Superseded by**:
- `../10_M_meanfield_sketch.md` - Updated with all resolutions
- `../10_O_gap1_resolution_report.md` - Gap #1 resolved via **permutation symmetry**
- `../10_P_gap3_resolution_report.md` - Gap #3 resolved via **de Bruijn + LSI**
- `../10_Q_complete_resolution_summary.md` - Overall summary
- `../10_R_meanfield_lsi_hybrid.md` - Complete hybrid proof
- `../10_S_meanfield_lsi_standalone.md` - Complete standalone proof

---

## Wasserstein Contraction Working Versions (03_X series)

These documents represent **exploratory work** on Wasserstein contraction for the cloning operator.

### Fundamentally Flawed (Do Not Reference)
- `03_A_wasserstein_contraction.md` - Incomplete case analysis, missing critical lemmas
- `03_B_companion_contraction.md` - **WRONG independence assumption** for companion selection
- `03_D_mixed_fitness_case.md` - Partial analysis only, incomplete

### Partial Work (Consolidated into Complete Proof)
- `03_C_wasserstein_single_pair.md` - Single-pair lemma structure (partial)
- `03_E_case_b_contraction.md` - Case B attempt with **SCALING ERROR**
- `03_F_outlier_alignment.md` - Lemma statement with proof skeleton only

**What was wrong**:
- **Error 1**: Assumed companion selections are independent (wrong for synchronous coupling)
- **Error 2**: Dimensional mismatch - tried to prove inequality with incompatible scaling
- **Error 3**: Incomplete case analysis, missing critical lemmas

**Superseded by**: Section 5.2 (lines 920-1040) of `../10_kl_convergence.md` (displacement convexity approach)

---

## Historical Value

These documents are valuable for:
1. **Understanding the development process** - shows how the proof evolved
2. **Learning from mistakes** - documents the errors that were corrected
3. **Alternative approaches** - some ideas may be useful for future work
4. **Collaboration record** - shows the AI-human collaboration process

---

## Final Documentation (in `../`)

The **current, authoritative documents** are:

### Main Convergence Document
- `../10_kl_convergence.md` - Complete LSI/KL convergence proof using **displacement convexity**

### Mean-Field Approach (Complementary)
- `../10_M_meanfield_sketch.md` - Mean-field sketch with resolved gaps
- `../10_R_meanfield_lsi_hybrid.md` - **Hybrid proof** (efficient, references existing results)
- `../10_S_meanfield_lsi_standalone.md` - **Standalone proof** (self-contained, pedagogical)

### Resolution Reports
- `../10_O_gap1_resolution_report.md` - Gap #1: **Permutation symmetry** resolution
- `../10_P_gap3_resolution_report.md` - Gap #3: **De Bruijn + LSI** resolution
- `../10_Q_complete_resolution_summary.md` - Overall resolution summary

### AI Engineering
- `../10_N_lemma5.2_ai_engineering_report.md` - Practical AI engineering perspective

---

## Key Breakthroughs

The final proofs were achieved through:

1. **Gap #1 Resolution**: Using **permutation symmetry** (Theorem 2.1 from `14_symmetries_adaptive_gas.md`)
   - Enables symmetrization that transforms $(e^{-x} - 1)x$ into tractable sinh expression
   - Global inequality without pointwise bounds

2. **Gap #3 Resolution**: Using **de Bruijn identity + Log-Sobolev Inequality**
   - Treats Gaussian noise as heat flow
   - LSI from log-concavity provides exponential contraction

3. **Three Complete Proofs**: All rigorous and publication-ready
   - Displacement convexity (geometric/global)
   - Mean-field hybrid (efficient)
   - Mean-field standalone (self-contained)

---

## Mean-Field Convergence Working Versions (13_X series)

**13_stage1_kinetic_dominance.md** - Initial Stage 1 attempt with **CRITICAL FLAW**

**What was wrong**:
- **Fundamental error**: Assumed $\rho_\infty$ is invariant for kinetic operator $\mathcal{L}_{\text{kin}}$ alone
- **Reality**: $\rho_\infty$ satisfies $\mathcal{L}(\rho_\infty) = 0$ for FULL generator, so $\mathcal{L}_{\text{kin}}(\rho_\infty) = -\mathcal{L}_{\text{jump}}(\rho_\infty) \neq 0$
- This invalidates the hypocoercivity analysis performed on kinetic operator in isolation
- Integration by parts produces uncontrolled remainder terms

**Superseded by**: `../mean_field_convergence/11_stage1_entropy_production.md` - Corrected analysis using full generator

**Why the correction matters**: The corrected proof analyzes the **complete entropy production** for $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$ and uses NESS (Non-Equilibrium Steady State) hypocoercivity framework from Dolbeault et al. (2015).

---

## For Reviewers

If you are reviewing the LSI/KL convergence proofs:

1. **IGNORE** all documents in this folder
2. **READ** one of:
   - `../kl_convergence/10_kl_convergence.md` (displacement convexity - primary)
   - `../kl_convergence/10_R_meanfield_lsi_hybrid.md` (mean-field - hybrid)
   - `../kl_convergence/10_S_meanfield_lsi_standalone.md` (mean-field - standalone)
3. **CHECK** `../kl_convergence/10_Q_complete_resolution_summary.md` for overview
4. For continuous-time mean-field convergence, see `../mean_field_convergence/` folder

---

## Notes

- **Do not cite** these deprecated documents in publications
- **Do not update** these documents - they are frozen for historical reference
- **Do cite** the final documents listed above
- If you need to reference the development process, cite the resolution reports (10_O, 10_P, 10_Q)

---

**Status:** These documents are DEPRECATED and should not be cited or used.

**Last updated:** 2025-10-09
