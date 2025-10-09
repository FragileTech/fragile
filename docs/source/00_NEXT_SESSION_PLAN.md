# ARCHIVED: Next Session Plan - W₂ Contraction Proof Completion

**⚠️ ARCHIVED - PLAN COMPLETED ⚠️**

**Status:** This plan has been EXECUTED. The W₂ contraction proof is complete in [03_wasserstein_contraction_complete.md](03_wasserstein_contraction_complete.md)

**Original Date:** Session at ~120K tokens
**Completion Date:** Following session

**Purpose:** Historical record of the task breakdown used to complete the proof.

---

# Original Plan (Historical)

**Current Session Status:** ~120K tokens used, Gemini MCP responses coming back empty (possible rate limit)

**Progress This Session:** Major conceptual breakthroughs achieved ✅

---

## Completed This Session

1. ✅ **Identified correct synchronous coupling mechanism**
2. ✅ **Diagnosed and corrected fundamental scaling error**
3. ✅ **Discovered Outlier Alignment is derivable** (not a new axiom!)
4. ✅ **Established complete proof structure** for both Cases A & B
5. ✅ **Identified all required citations** from 03_cloning.md
6. ✅ **Created Outlier Alignment Lemma document** (03_F_outlier_alignment.md)
7. ✅ **Documented comprehensive progress** (00_W2_PROOF_PROGRESS_SUMMARY.md)

---

## Next Session: Start Here

### Task 1: Complete Outlier Alignment Lemma Proof (2-3 Gemini iterations)

**File:** `03_F_outlier_alignment.md`

**Current status:** Lemma statement complete, proof skeleton outlined (Steps 1-6)

**What's needed:** Fill in rigorous proofs for each step

**Key steps to complete:**

1. **Step 1 (Fitness Valley):**
   - Formalize "stably separated swarms"
   - Prove valley exists between them
   - Use entropy production from 14_symmetries_adaptive_gas.md
   - Show: if no valley, cloning would cause merger

2. **Step 2 (Survival Probability):**
   - From cloning operator definition (Chapter 9, 03_cloning.md)
   - Quantify: $p_{\text{survive},i} \propto V_{\text{fit},i}^{\alpha}$

3. **Step 3 (Define Wrong Side):**
   - Misaligned set: $M_1 = \{x \mid \langle x - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle < 0\}$

4. **Step 4 (Low Fitness on Wrong Side):**
   - Outlier on wrong side is near fitness valley
   - Bound $f(x_{1,i})$ from above

5. **Step 5 (Probability Bound):**
   - Show: $\mathbb{P}(\text{survive} \mid \text{wrong side}) \to 0$ as $L \to \infty$

6. **Step 6 (Conclude η):**
   - Extract quantitative bound with constant η
   - Verify N-uniformity

**Approach:** Work with Gemini step-by-step, completing one step at a time

---

### Task 2: Complete Case A Contraction (1-2 Gemini iterations)

**File:** Create `03_G_case_a_contraction.md`

**Structure (from 03_C_wasserstein_single_pair.md):**
- Both swarms have same lower-fitness walker
- Jitter CANCELS in Clone-Clone case (key insight!)
- Derive uniform contraction factor γ_A < 1

**Required:**
- Cite companion concentration (Lemma 6.5.1)
- Cite outlier principle (Corollary 6.4.4, Theorem 7.5.2.4)
- Rigorous bounds on all cases (PP, CC, CP, PC)

---

### Task 3: Complete Case B Contraction (2-3 Gemini iterations)

**File:** Revise `03_E_case_b_contraction.md`

**Critical fix:** Use corrected target inequality
- **OLD (wrong):** $D_{ii} - D_{ji} \geq \alpha(D_{ii} + D_{jj})$ (scaling mismatch!)
- **NEW (correct):** $D_{ii} - D_{ji} \geq \alpha_B \|x_{1,i} - x_{1,j}\|^2$

**Uses:** Outlier Alignment Lemma (from Task 1)

**Derive:** Uniform contraction factor γ_B < 1

---

### Task 4: Unified Single-Pair Lemma (1 iteration)

**File:** Create `03_H_single_pair_unified.md`

**Combines:** Cases A & B into single lemma

**Statement:**
$$\mathbb{E}[\|x'_{1,i} - x'_{2,i}\|^2 + \|x'_{1,j} - x'_{2,j}\|^2 \mid M, S_1, S_2] \leq \gamma_{\text{pair}} (\|x_{1,i} - x_{2,i}\|^2 + \|x_{1,j} - x_{2,j}\|^2) + C_{\text{pair}}$$

where $\gamma_{\text{pair}} = \max(\gamma_A, \gamma_B) < 1$

---

### Task 5: Sum Over Matching (1-2 iterations)

**Sum over all pairs in matching M:**

$$\mathbb{E}[W_2^2(S'_1, S'_2) \mid M] \leq \gamma_{\text{pair}} W_2^2(S_1, S_2) + N \cdot C_{\text{pair}}$$

**Key:** Linearity of expectation

---

### Task 6: Integrate Over Matching Distribution (2-3 iterations)

**Take expectation over M ~ P(M|S₁):**

$$\mathbb{E}[W_2^2(S'_1, S'_2)] \leq \gamma_{\text{pair}} W_2^2(S_1, S_2) + N \cdot C_{\text{pair}}$$

**Handle:** Asymmetric coupling (matching depends only on S₁)

---

### Task 7: Final W₂ Theorem (1 iteration)

**File:** Create `03_I_wasserstein_final_theorem.md`

**Statement:**
$$\mathbb{E}[W_2^2(\mu_{S_1'}, \mu_{S_2'})] \leq (1 - \kappa_W) W_2^2(\mu_{S_1}, \mu_{S_2}) + C_W$$

with explicit:
- $\kappa_W = (1 - \gamma_{\text{pair}}) > 0$
- $C_W = N \cdot C_{\text{pair}}$
- N-uniformity verification

---

### Task 8: Fix References in 10_kl_convergence.md

**Current issue:** Lemma 4.3 incorrectly cites 04_convergence.md (kinetic operator)

**Fix:** Update to cite final W₂ theorem from Task 7

---

## Estimated Timeline

**Total Gemini iterations needed:** ~10-15

**Session breakdown:**
- Session 1 (fresh start): Tasks 1-3 (~6-8 iterations)
- Session 2 (if needed): Tasks 4-7 (~4-7 iterations)
- Final cleanup: Task 8 (1 iteration)

---

## Key Documents Reference

**Read these first:**
- `00_W2_PROOF_PROGRESS_SUMMARY.md` - Complete session summary
- `03_F_outlier_alignment.md` - Outlier Alignment Lemma (needs proof)
- `03_C_wasserstein_single_pair.md` - Single-pair structure
- `03_E_case_b_contraction.md` - Case B attempt (needs scaling fix)

**Framework citations:**
- `03_cloning.md` - Keystone Principles (Chapters 6-8)
- `14_symmetries_adaptive_gas.md` - Entropy production, H-theorem

---

## Critical Reminders

1. **Do NOT modify 03_cloning.md** - add new results in separate documents
2. **Scaling principle:** Use intra-swarm distance $\|x_{1,i} - x_{1,j}\|^2$, NOT $D_{ii} + D_{jj}$
3. **Outlier Alignment is derivable** - prove it, don't assume it
4. **Linear contraction required** - not affine (needed for LSI)
5. **N-uniformity throughout** - verify all constants

---

## Success Criteria

✅ Outlier Alignment Lemma with complete rigorous proof
✅ Case A with uniform γ_A < 1
✅ Case B with uniform γ_B < 1 (corrected scaling)
✅ Final W₂ theorem: $\mathbb{E}[W_2^2] \leq (1-\kappa_W)W_2^2 + C_W$
✅ All proofs meet publication standards
✅ 10_kl_convergence.md references fixed

**Then:** Can proceed to log-concavity issue (final LSI barrier)
