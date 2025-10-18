# Round 2 Corrections Summary

**Date**: 2025-10-18

**Status**: ✅ ALL 5 CRITICAL/MAJOR ISSUES ADDRESSED

---

## Issues Fixed

### ✅ Issue #1: Kramers Step Still Non-Rigorous (CRITICAL)

**Codex's Problem**: "The proof invokes the classical Kramers escape rate without rigorous justification. As written, the main result of Part II remains unproven."

**Our Fix**:
1. Changed theorem title from "Main Result" to **"Conditional Result"**
2. Added explicit assumption: "**assuming Eyring-Kramers metastability theory applies**"
3. Replaced weak note with detailed `{important}` block that:
   - Cites standard reference: Bovier et al., *J. Eur. Math. Soc.* (2004)
   - Lists exact hypotheses to verify (non-degenerate minima, saddles, spectral gap, dimensional reduction)
   - Provides verification strategy using framework Langevin dynamics
   - Offers alternatives (conditional statement, LSI approach, numerical verification)
   - **Explicitly states**: "As written, this theorem is NOT fully rigorous."

**Result**: Honest assessment that theorem is conditional pending verification

---

### ✅ Issue #2: Local Cumulant Bound Misstated (MAJOR)

**Codex's Problem**: "Theorem claims $C^m N^{-(m-1)}$ with $C$ independent of $m$, but Brydges-Imbrie gives $(m-1)! m^{m-2}$ growth. Factorial growth cannot be absorbed into constant."

**Our Fix**:
1. Updated theorem statement to include explicit factorial:
   $$|\text{Cum}_{\text{local}}| \leq (m-1)! \cdot m^{m-2} \cdot C^m N^{-(m-1)}$$

2. Added note explaining why this still works:
   "For the moment method (summing over $m \leq M = o(N^{1/2})$), this factorial growth is acceptable because the contribution decays as $\left(\frac{4m^2}{eN}\right)^m$ (super-exponential for $N \gg m^2$)."

3. Kept Step 7's detailed Stirling analysis which was already correct

**Result**: Theorem statement now matches proof exactly

---

### ✅ Issue #3: Non-Local Cumulant Scaling Oversight (MAJOR)

**Codex's Problem**: "Proof derives antichain size $N^{(d-1)/d}$ but stated bound omits this factor. Should include $N^{2(d-1)/d}$ explicitly."

**Our Fix**:
1. Updated theorem bound to explicit power:
   $$|\text{Cum}_{\text{nonlocal}}| \leq C^m N^{-m/2 + 2(d-1)/d} \cdot \exp(-cR/\xi)$$

2. Added decay analysis showing this still vanishes:
   - For $d=2$: $-m/2 + 1$, decays for $m \geq 3$
   - For $d=3$: $-m/2 + 4/3$, decays for $m \geq 3$
   - For all $d \geq 2$: polynomial part decays for $m \geq 3$

3. Updated Step 5 to show complete calculation with antichain factor

**Result**: Antichain scaling now explicit, decay still proven

---

### ✅ Issue #4: Broken LSI Reference (MAJOR)

**Codex's Problem**: "The label `thm-qsd-lsi` does not exist in `15_geometric_gas_lsi_proof.md`; the proven result is labeled `thm-adaptive-lsi-main`."

**Our Fix**:
1. Searched framework document for correct label
2. Updated reference from `{prf:ref}`thm-qsd-lsi`` to `{prf:ref}`thm-adaptive-lsi-main``

**Result**: Reference now resolvable

---

### ✅ Issue #5: Overstated Zeta Connection (MAJOR)

**Codex's Problem**: "Manuscript asserts 'Information Graph statistics ≡ Riemann zeta zero statistics' and claims this is rigorously proven, yet the argument relies on unproven Montgomery-Odlyzko conjecture."

**Our Fix**:
1. Changed "Our result + Montgomery-Odlyzko **implies**" to "Our result + Montgomery-Odlyzko **conjecture** (if proven) **would imply**"

2. Added `{important}` conditional box:
   ```
   **Conditional Result**: This equivalence is **conditional on the Montgomery-Odlyzko conjecture**

   1. ✅ **Proven**: Information Graph exhibits GUE statistics
   2. ✅ **Conjecture**: Zeta zeros exhibit GUE pair correlation (numerically verified)
   3. **Conditional conclusion**: IF Montgomery-Odlyzko holds, THEN algorithmic vacuum = zeta statistics
   ```

3. Changed "This is the first rigorously proven example" to "This would be the first example... **pending proof of Montgomery-Odlyzko**"

**Result**: Clearly labeled as conditional on unproven conjecture

---

## Summary of Changes

**Theorem Statements**:
- QSD Localization: Now **"Conditional Result"** with explicit Kramers assumption
- Local Cumulant: Now includes $(m-1)! m^{m-2}$ factor
- Non-Local Cumulant: Now includes $N^{2(d-1)/d}$ antichain factor

**Citations**:
- LSI reference fixed: `thm-qsd-lsi` → `thm-adaptive-lsi-main`
- Kramers theory: Added Bovier et al. (2004) citation with verification requirements

**Conditional Statements**:
- Kramers-based localization: Explicitly conditional
- Montgomery-Odlyzko connection: Explicitly conditional

---

## What is NOW Rigorously Proven

1. ✅ **GUE Universality** (Part I): Information Graph exhibits Wigner semicircle law
   - With correct cumulant bounds including factorial growth and antichain factors
   - Using rigorously proven antichain-surface correspondence from framework

2. ✅ **Well Structure** (Part II, partial): Z-reward creates multi-well potential at zero locations
   - Potential minima proven at $|t_n|$
   - Barrier structure analyzed

3. ✅ **Cluster Formation** (Part II, partial): QSD localization implies clustered Information Graph
   - Proven as consequence IF localization holds

---

## What Remains Conditional

1. ⚠️ **QSD Localization at Zeros**: Conditional on Eyring-Kramers metastability theory
   - Requires verification of hypotheses or completion of LSI-based alternative proof

2. ⚠️ **Zeta Statistics Connection**: Conditional on Montgomery-Odlyzko conjecture
   - Our contribution is GUE proof; zeta side is conjectural

---

## Publication Assessment

**Before Round 2 Fixes**: NOT publication-ready (Codex assessment)

**After Round 2 Fixes**:
- ✅ All quantitative errors corrected (cumulant bounds)
- ✅ All references fixed
- ✅ All conditional statements labeled honestly
- ✅ Clear guidance on what verification remains

**Suitable for**:
- Communications in Mathematical Physics (with conditional statements)
- Journal of Statistical Physics (appropriate for conditional results with clear caveats)

**NOT suitable for**:
- Annals of Mathematics (too many conditional steps)

---

## Next Steps

**Option A - Complete Proofs** (2-3 weeks):
1. Verify Eyring-Kramers hypotheses for Fragile Gas using framework dynamics
2. Complete LSI-based alternative to Kramers (more technical but fully rigorous)
3. Submit for Round 3 review

**Option B - Submit As-Is** (1 week):
1. Parts I-II are honest about what's proven vs. conditional
2. Target JSP or CMP with explicit scope
3. Future work: complete Kramers verification

**Option C - Split Publications**:
1. **Paper 1**: GUE Universality (fully proven) → CMP
2. **Paper 2**: Z-Localization (conditional on Kramers) → JSP with explicit caveat
3. **Paper 3**: Density-Spectrum Mechanism → SIAM J. Appl. Math.

---

**Recommendation**: Option C (split publications) - each paper fully rigorous within its scope

---

*End of Round 2 Corrections Summary*
