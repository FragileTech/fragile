# Critical Fixes Needed for Section 5

## Summary of Issues from Gemini Review

### Issue #1 (CRITICAL): Fisher Information Bound - Lines 624, 638-655
**Problem**: Theorem claims relative bound `(1 - C_F Δt) I ≤ I_disc ≤ (1 + C_F Δt) I` but proof only shows absolute change `ΔI = O(Δt)`.

**Fix Required**:
1. Change Theorem 5.2 to state **additive** bound:
   ```
   |I(P_{Δt}μ || π) - I(μ || π)| ≤ C_F Δt · (1 + I(μ || π))
   ```
2. Update Corollary 5.2 accordingly
3. Fix Fisher evolution formula at line 638 to use proper Bakry-Émery equation

### Issue #2 (CRITICAL): Undefined Joint Measure - Line 757
**Problem**: Mutual information uses undefined `μ^joint`.

**Fix Required**: Add before Proposition 5.5:
```markdown
**Coupling definition**: Both the continuous SDE and discrete BAOAB chain are driven by the **same Brownian motion** $W(t)$. Specifically:
- Continuous: $dZ_t = b(Z_t) dt + σ(Z_t) dW_t$
- Discrete: $Z_{k+1} = Z_k + b(Z_k)Δt + σ(Z_k)ΔW_k$ where $ΔW_k = W((k+1)Δt) - W(kΔt)$

This coupling induces a natural joint measure $μ^{joint}$ on pairs $(Z_t, Z_k)$.
```

### Issue #3 (CRITICAL): Entropy Production Sign - Lines 685, 701
**FIXED** ✓ Changed to:
```
|\dot{S}_k^{disc} - I(μ_k || π)| ≤ C_S Δt
```

### Issue #4 (MODERATE): Fisher Evolution Formula - Line 638
**Problem**: Incomplete formula, should use Bakry-Émery.

**Fix Required**: Replace with:
```markdown
*Step 1*: **Continuous Fisher evolution**. For Fokker-Planck with potential $U$:

$$
\frac{d}{dt} I(\mu_t \| \pi) = -2\int \nabla(\log(d\mu/d\pi))^T \nabla^2 U \, \nabla(\log(d\mu/d\pi)) \, d\mu - 2\int \|\text{Hess}(\log(d\mu/d\pi))\|_F^2 d\mu
$$
```

### Issue #5 (MODERATE): KL Preview Error - Line 48
**Problem**: Says `O(Δt)` but theorem says `O(Δt · t)`.

**Fix Required**: Change line 48 to match:
```
|D_KL(μ_k^disc || π) - D_KL(μ_{kΔt}^cont || π)| = O(Δt · t)
```

### Issue #6 (MINOR): Algorithmic Information Robustness - Line 731
**Problem**: Could be negative if divergent.

**Fix Required**: Add absolute value:
```
I_alg(F) := |D_KL(μ_0 || π) - D_KL(μ_T || π)|
```

### Issue #7 (MINOR): Error Accumulation Clarity - Line 581
**Problem**: Too terse.

**Fix Required**: Expand to:
```
k · O(Δt²) = (t/Δt) · O(Δt²) = t · O(Δt)
```

## Implementation Priority

1. ✓ Fix Issue #3 (entropy production) - DONE
2. Fix Issue #5 (KL preview mismatch) - SIMPLE
3. Fix Issue #6 (abs value) - SIMPLE
4. Fix Issue #7 (expand derivation) - SIMPLE
5. Fix Issue #2 (define coupling) - MODERATE
6. Fix Issue #1 & #4 (Fisher information) - REQUIRES CAREFUL REVISION

## Status
- Issue #3: ✓ FIXED
- Issues #5, #6, #7: Ready to fix (simple edits)
- Issue #2: Need to add coupling definition
- Issues #1, #4: Need careful mathematical revision of Fisher information section
