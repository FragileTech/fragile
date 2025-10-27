# Review Summary: thm-corrected-kl-convergence

**Date:** 2025-10-25
**Reviewer:** Codex (GPT-5) via MCP
**Review Protocol:** Single-strategist (Gemini MCP currently unavailable)
**Confidence Level:** MEDIUM (single reviewer, flagged per protocol)

---

## Executive Summary

**Overall Severity:** CRITICAL

The proof sketch has a **solid high-level strategy** but contains **one critical misapplication of LSI theory**, several major consistency and assumption gaps, and minor notation/definition issues. The dominance condition and rate constants need precise derivations and dependencies.

**Key Concerns:**
1. LSI for the QSD is cited from hypocoercivity (DMS 2015) without satisfying assumptions and is likely inapplicable as stated
2. Inconsistent definitions and constants for α_net/α_kin and relative Fisher/LSI factors
3. Dominance condition scaling and λ_LSI dependence are unsupported on unbounded domains

**Publication Readiness:** MAJOR REVISIONS REQUIRED

**Mathematical Rigor:** 5/10 (key LSI step incorrectly sourced)
**Logical Soundness:** 6/10 (coherent strategy but critical gaps)
**Computational Correctness:** 5/10 (factor-of-two errors, invalid scaling)

---

## Critical Issues (Must Fix)

### Issue #1: Misapplication of DMS (2015) for LSI
**Severity:** CRITICAL
**Location:** Lines 112-118

**Problem:** The sketch claims the QSD admits an LSI "under R1-R6" by citing Dolbeault-Mouhot-Schmeiser (2015) NESS hypocoercivity. However, DMS (2015) develops **decay estimates** for linear kinetic equations, not an LSI for a general nonreversible QSD measure.

**Why This Matters:** An LSI is a property of a **measure** (typically requiring curvature/log-concavity), not a direct consequence of hypocoercive decay of the semigroup. Without a valid LSI for ρ∞, the entire α_kin derivation and Grönwall rate collapse.

**Recommended Fix:**
- Replace citation with valid route:
  - (a) Prove LSI via log-concavity or Bakry-Émery criteria if log ρ∞ has uniform convexity
  - (b) Derive Poincaré inequality first, then upgrade to LSI via Holley-Stroock perturbation
  - (c) Use known LSI for Gaussian-like conditional velocity measures, bootstrap to joint measure via hypocoercive decomposition

**Literature to Add:**
- Arnold-Markowich-Toscani-Unterreiter (2001)
- Hérau (2007)
- Arnold-Erb (2014)
- Bakry-Émery/Gross LSI criteria

---

## Major Issues

### Issue #2: α_kin/α_net Definition Inconsistency
**Severity:** MAJOR
**Location:** Lines 51, 130-132, 194-200

**Problem:**
- α_kin defined as C_hypo/C_LSI (line 51)
- Step 4 yields α_net = σ²λ_LSI - σ²C_Fisher^coup - C_KL^coup - A_jump (line 195)
- These are **inconsistent**

**Factor Mismatch:** Fisher-to-KL inequality states I_v(ρ) ≥ 2λ_LSI D_KL - ..., but source lemma implies I_v(ρ) ≥ λ_LSI D_KL - C (not 2λ_LSI).

**Impact:** May overstate α_net by factor ~2; invalidates claimed sufficient condition.

**Recommended Fix:** Unify definitions. α_kin should be defined via d/dt D_KL ≤ -(σ²/2)I_v + ... and I_v ≥ λ_LSI D_KL - const, giving α_kin = σ²λ_LSI - σ²C_Fisher^coup - C_KL^coup.

---

### Issue #3: Invalid Scaling Formula for λ_LSI
**Severity:** MAJOR
**Location:** Lines 217-221

**Problem:** λ_LSI claimed to scale as min(σ²/diam(Ω)², γ, κ_conf). For mean-field kinetic PDE, Ω = ℝ^{2d} (unbounded), so diam(Ω) = ∞ and the bound yields **0**, contradicting requirement λ_LSI > 0.

**Impact:** Undermines Step 5 dominance verification.

**Recommended Fix:** Remove diam(Ω) term. Provide LSI proof with lower bound depending on dimension, γ, κ_conf, and bounds on ∇² log ρ∞. If only Poincaré available, adjust entire rate chain.

---

### Issue #4: Hypocoercivity Applicability to Nonlinear/Jump Generator
**Severity:** MAJOR
**Location:** Lines 136-170, 214-237

**Problem:** NESS hypocoercivity bounds (designed for **linear**, mass-conserving kinetic equations) are used without stating assumptions to ensure closability, boundedness, or smallness of jump perturbation.

**Gap:** Nonlinearity (McKean-Vlasov) and non-conservativity (killing/revival) change operator class and can break functional coercivity estimates.

**Recommended Fix:** State explicit assumptions:
- Boundedness/Lipschitz of mean-field coefficients in ρ∞-weighted spaces
- Jump term treated as bounded perturbation with constants entering A_jump, B_jump
- Smallness condition translating to α_net > 0
- Bootstrap lemma ensuring modified entropy method closes

---

### Issue #9: Dominance Condition Statement vs. α_net
**Severity:** MAJOR
**Location:** Lines 28-41, 202-208

**Problem:** Theorem states σ²γκ_conf > C_0 max(...) as hypothesis, but α_net > 0 (lines 194-196) is the **true condition**. The scaling link (Step 5) is not demonstrated.

**Impact:** Risk of overstating hypothesis sufficiency without proof.

**Recommended Fix:** State theorem with hypothesis "α_net > 0" as primary condition. Present σ²γκ_conf > C_0 max(...) as an **explicit sufficient condition** once constants are derived.

---

## Minor Issues

### Issue #5: Adjoint/Generator Sign Convention Mismatch
**Severity:** MINOR
**Location:** Lines 145-152, 73-83

**Problem:** L*_kin written as generator on functions (not adjoint on densities). Mixing conventions risks sign errors.

**Fix:** State both operators precisely: L acting on functions and L† acting on densities. Align Step 1 and Step 3 conventions.

---

### Issue #6: Undefined Symbol m_d(ρ)
**Severity:** MINOR
**Location:** Line 26

**Problem:** m_d(ρ) used in jump operator but not defined locally.

**Fix:** Add definition: "m_d(ρ) = 1 - ∫ ρ" (consistent with source document).

---

### Issue #7: Relative Fisher Information Weight Inconsistency
**Severity:** MINOR
**Location:** Lines 114-116

**Problem:** Sketch defines I_v(ρ‖ρ∞) with weight ρ∞, while source defines it with weight ρ.

**Fix:** Adopt source convention (weight ρ). Adjust Lemma 3.3 and Step 4 constants.

---

### Issue #8: Vague Constants (C_0 = O(1), B_jump unspecified)
**Severity:** MINOR
**Location:** Lines 41-53, 188-200

**Problem:** Annals-level requires explicit dependencies.

**Fix:** Provide definitions: C_Fisher^coup(C_Δ), C_KL^coup(C_x, C_v, ‖a‖∞, ε), B_jump(λ_revive, κ_kill, M_∞).

---

### Issue #10: "Residual Neighborhood" Phrasing
**Severity:** SUGGESTION
**Location:** Lines 55-56, 206-210

**Problem:** Informal phrasing.

**Fix:** State: "limsup_{t→∞} D_KL(ρ_t‖ρ∞) ≤ C_total/α_net" with precise specification of when C_total = B_jump vs. includes coupling offsets.

---

## Required Proofs Checklist

- [ ] **Prove LSI (or Poincaré) for ρ∞** under explicit assumptions; avoid relying on DMS 2015 for LSI
- [ ] **Fisher-to-KL inequality** with correct constants and weight; align with lem-fisher-bound
- [ ] **Hypocoercivity coupling bound** (existence/regularity of a and coercivity) with explicit constants
- [ ] **Jump expansion bound dependencies** B_jump(·) and A_jump(·) fully specified
- [ ] **Nonlinearity and jump perturbation control:** show modified Lyapunov method closes
- [ ] **Circularity check:** ensure LSI not derived using hypocoercivity that itself later uses LSI

---

## Prioritized Action Plan

1. **Fix LSI step** (CRITICAL): Replace DMS citation with valid LSI proof route or downgrade to Poincaré and adjust rates
2. **Correct α_kin/α_net definitions** (MAJOR): Unify relative Fisher convention and constant factors
3. **Restate theorem hypothesis** (MAJOR): Use "α_net > 0" as primary; present σ²γκ_conf > ... as proven sufficiency
4. **Specify constants with dependencies** (MAJOR): C_Fisher^coup, C_KL^coup, B_jump
5. **Clarify operator conventions** (MINOR): Adjoints/signs, define m_d(ρ)
6. **Expand Step 3 details** (MAJOR): Auxiliary function a and coercivity with explicit closure assumptions

---

## Answers to Specific Questions

**Q1: Is the Stage 0, 0.5, 1, 2, 3 decomposition sound?**

**A:** Sound conceptually. **Recommend splitting Stage 2** explicitly into:
- **(2a) LSI for ρ∞** (measure property)
- **(2b) Hypocoercive coupling bounds** (dynamical)

**Avoid circularity:** Do not use hypocoercivity to justify LSI and then LSI to justify hypocoercivity.

---

**Q2: Does NESS hypocoercivity framework (DMS 2015) apply directly?**

**A:** **No, not directly.** DMS 2015 does not yield LSI for the QSD. It can inform decay estimates for the kinetic part, but the nonlinearity and jumps must be treated as **bounded/small perturbations** in a properly weighted space with additional assumptions.

---

**Q3: Is "residual neighborhood" vs. "convergence to QSD" rigorous?**

**A:** Rigorous if stated as **limsup bound:**

limsup_{t→∞} D_KL(ρ_t‖ρ∞) ≤ C_total/α_net

Exact convergence to QSD requires B_jump = C_0^coup = 0 (or their ratio → 0).

---

**Q4: Should dominance condition be α_net > 0 or σ²γκ_conf > C_0 max(...)?**

**A:** State the **primary condition as α_net > 0**. The inequality σ²γκ_conf > C_0 max(λ_revive/M_∞, κ̄_kill) should be presented as a **proven sufficient condition** once constants are derived; do not state it as the main hypothesis without proof of equivalence.

---

## Time and Difficulty Assessment

**Codex Assessment:**
- **Difficulty:** HIGH to VERY HIGH is justified (nonlinearity + jump perturbations)
- **Time Estimate:** 40-60 hours is **optimistic** for Annals-level rigor

**More Realistic Range:** 120-200 hours
- 40-80 hours: Establish LSI/Poincaré rigor for ρ∞
- 40-60 hours: Close hypocoercive coupling with explicit constants
- 20-40 hours: Jump perturbation integration and final assembly
- 20+ hours: Verification and polishing

---

## Additional Literature Suggestions

**Hypocoercivity/LSI:**
- Arnold, Markowich, Toscani, Unterreiter (2001)
- Hérau (2007)
- Arnold-Erb (2014)
- Bakry-Émery/Gross for LSI criteria
- Villani (2009) for hypocoercivity architecture

**QSD Framework:**
- Champagnat-Villemonais (2016) [already cited]
- Cattiaux et al. on QSD for diffusions with killing

---

## Conclusion

The proof sketch demonstrates **excellent strategic vision** and a **well-organized multi-stage architecture**. However, the **critical reliance on an incorrect LSI claim** (Issue #1) must be addressed before the proof can proceed. Once the LSI/Poincaré foundation is established with correct constants and the major consistency issues are resolved, this has the potential to become a **rigorous, publication-quality result**.

**Recommended Next Steps:**
1. Address Issue #1 (LSI for QSD) as top priority
2. Revise constant definitions and unify conventions (Issues #2, #3, #7, #8)
3. Add explicit assumptions for nonlinear/jump perturbation control (Issue #4)
4. Restate theorem with α_net > 0 as primary hypothesis (Issue #9)
5. Expand Step 3 technical details with closure proof

**Single-Reviewer Caveat:** This review was conducted by Codex only due to Gemini MCP issues. **Recommend second review** once critical issues are addressed to increase confidence in the proof strategy.

---

**Review Status:** COMPLETE
**Sketch Status:** REQUIRES MAJOR REVISION before expansion to full proof
