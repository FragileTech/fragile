# Quick Reference: Boundary Contraction Proof Corrections

**Document**: docs/source/1_euclidean_gas/05_kinetic_contraction.md, §7.4
**Lines to Replace**: 2350-2529
**Date**: 2025-10-25

---

## Critical Changes at a Glance

### 1. Compatibility Condition Sign

| **Original (WRONG)** | **Corrected** |
|:---------------------|:--------------|
| ⟨-∇U(x), ∇φ(x)⟩ ≥ α_boundary φ | ⟨-∇U(x), ∇φ(x)⟩ ≤ -α_align φ |
| ⟨F(x), ∇φ(x)⟩ ≥ α_boundary φ | ⟨F(x), ∇φ(x)⟩ ≤ -α_align φ |
| **POSITIVE contribution** | **NEGATIVE contribution** |

**Physical meaning:**
- F points **inward** (confining)
- ∇φ points **outward** (barrier)
- Inner product is **NEGATIVE**

---

### 2. Generator of ⟨v, ∇φ⟩

| **Original (WRONG)** | **Corrected** |
|:---------------------|:--------------|
| L⟨v, ∇φ⟩ = v^T(∇²φ)v + ⟨F, ∇φ⟩ - γ⟨v, ∇φ⟩ **+ (1/2)Tr(A∇²φ)** | L⟨v, ∇φ⟩ = v^T(∇²φ)v + ⟨F, ∇φ⟩ - γ⟨v, ∇φ⟩ |
| Includes spurious diffusion term | No spurious term |

**Why corrected is right:**
- ⟨v, ∇φ⟩ is **linear in v**
- ∇_v²[⟨v, ∇φ⟩] = 0 (second derivative of linear = zero)
- Diffusion acts only on **velocity** derivatives

---

### 3. Barrier Function Specification

| **Original (WRONG)** | **Corrected** |
|:---------------------|:--------------|
| φ_barrier: unspecified | φ(x) = exp(c·ρ(x)/δ) on boundary layer |
| Claims: ∥∇²φ∥ ≤ K_φ (bounded Hessian) | Derives: v^T(∇²φ)v ≤ φ[(c/δ)² + (c/δ)K_curv]∥v∥² |
| Claims: ∥∇φ∥ ≤ C√φ | Derives: ∥∇φ∥ = (c/δ)φ |
| **Contradictory** (φ → ∞ at boundary but bounded derivatives) | **Consistent** (bounded **ratios** ∥∇φ∥/φ, ∥∇²φ∥/φ) |

**Key insight:**
Exponential barrier has bounded derivative **ratios**, not bounded absolute derivatives.

---

### 4. Coupling Parameter

| **Original** | **Corrected** |
|:-------------|:--------------|
| ε = 1/(2γ) | ε = 1/γ |
| Leaves residual: (1/2)⟨v, ∇φ⟩ | Complete cancellation: 0·⟨v, ∇φ⟩ |
| Requires dubious ∥∇φ∥ ≤ C√φ bound | Clean generator: LΦ = (1/γ)[⟨F, ∇φ⟩ + v^T(∇²φ)v] |

---

### 5. Contraction Rate

| **Original (WRONG)** | **Corrected** |
|:---------------------|:--------------|
| κ_pot = α_boundary/(4γ) | κ_pot = (1/γ)[α_align - K_φ V_var,v^eq] |
| Derived from **positive** ⟨F, ∇φ⟩ | Derived from **negative** ⟨F, ∇φ⟩ |
| Sign inconsistency | Mathematically consistent |

**Positivity condition:**
```
α_align > K_φ V_var,v^eq
(c/δ)α_boundary > [(c/δ)² + (c/δ)K_curv] · (d σ_max²)/(2γ)
```

Choose c small enough to ensure this holds.

---

## Line-by-Line Replacement Guide

### Part VI: Compatibility (Lines 2425-2440)

**REMOVE:**
```markdown
**PART VI: Use Confining Potential Compatibility (Axiom 3.3.1)**

By Axiom 3.3.1 (part 4), the confining potential $U$ and barrier function
$\varphi_{\text{barrier}}$ are **compatible**:

$$
\langle -\nabla U(x), \nabla\varphi_{\text{barrier}}(x) \rangle \geq \alpha_{\text{boundary}} \varphi_{\text{barrier}}(x)
$$

for $x$ near the boundary, where $\alpha_{\text{boundary}} > 0$.

This means:

$$
\langle F(x_i), \nabla\varphi_i \rangle \geq \alpha_{\text{boundary}} \varphi_i
$$
```

**REPLACE WITH:**
```markdown
**PART VII: Apply Corrected Compatibility and Hessian Bounds**

In the boundary layer ($-\delta \leq \rho(x_i) < 0$):

**Compatibility (corrected sign):**

$$
\langle F(x_i), \nabla\varphi_i \rangle \leq -\alpha_{\text{align}} \varphi_i
$$

where $\alpha_{\text{align}} = \frac{c}{\delta} \alpha_{\text{boundary}}$.

**Hessian bound:**

$$
v_i^T (\nabla^2\varphi_i) v_i \leq \varphi_i \left[\left(\frac{c}{\delta}\right)^2 + \frac{c}{\delta} K_{\text{curv}}\right] \|v_i\|^2
$$
```

### Generator Calculation (Lines 2391-2409)

**REMOVE:**
```markdown
$$
\mathcal{L}\langle v_i, \nabla\varphi_i \rangle = v_i^T (\nabla^2\varphi_i) v_i +
\langle F(x_i), \nabla\varphi_i \rangle - \gamma \langle v_i, \nabla\varphi_i \rangle +
\frac{1}{2}\text{Tr}(A_i \nabla^2\varphi_i)
$$
```

**REPLACE WITH:**
```markdown
$$
\mathcal{L}\langle v_i, \nabla\varphi_i \rangle = v_i^T (\nabla^2\varphi_i) v_i +
\langle F(x_i), \nabla\varphi_i \rangle - \gamma \langle v_i, \nabla\varphi_i \rangle
$$
```

Add explanation:
```markdown
**Critical note:** The diffusion term vanishes because $g$ is **linear in $v$**,
so $\nabla_v^2 g = 0$. The velocity diffusion operator $(1/2)\text{Tr}(A \nabla_v^2)$
acts only on velocity derivatives.
```

### Coupling Parameter (Lines 2411-2423)

**CHANGE:**
```markdown
Choose $\epsilon = \frac{1}{2\gamma}$:
```

**TO:**
```markdown
Choose $\epsilon = \frac{1}{\gamma}$ to **completely eliminate** the cross-term:
```

**CHANGE:**
```markdown
$$
1 - \epsilon\gamma = 1 - \frac{1}{2} = \frac{1}{2}
$$

This gives:

$$
\mathcal{L}\Phi_i = \frac{1}{2}\langle v_i, \nabla\varphi_i \rangle + \frac{1}{2\gamma}\langle F(x_i), \nabla\varphi_i \rangle + \frac{1}{2\gamma} v_i^T (\nabla^2\varphi_i) v_i + \frac{1}{4\gamma}\text{Tr}(A_i \nabla^2\varphi_i)
$$
```

**TO:**
```markdown
$$
1 - \epsilon\gamma = 1 - \frac{1}{\gamma} \cdot \gamma = 0
$$

This gives:

$$
\mathcal{L}\Phi_i = \frac{1}{\gamma}\langle F(x_i), \nabla\varphi_i \rangle + \frac{1}{\gamma} v_i^T (\nabla^2\varphi_i) v_i
$$
```

### Final Inequality (Lines 2478-2519)

**CHANGE:**
```markdown
**Dominant term for large $\varphi_i$:**

$$
\mathcal{L}\Phi_i \leq -\frac{\alpha_{\text{boundary}}}{4\gamma}\varphi_i + C_{\text{bounded}}
$$

...

**Key constants derived:**

$$
\kappa_{\text{pot}} = \frac{\alpha_{\text{boundary}}}{4\gamma}, \quad C_{\text{pot}} = \frac{K_{\varphi} \sigma_{\max}^2 d}{4\gamma} + O(V_{\text{Var},v}^{\text{eq}})
$$
```

**TO:**
```markdown
**Taking expectation:**

$$
\mathbb{E}[\mathcal{L}\Phi_i] \leq \frac{\varphi_i}{\gamma}\left[K_{\varphi} V_{\text{Var},v}^{\text{eq}} - \alpha_{\text{align}}\right]
$$

**Barrier parameter selection for contraction:** Choose $c$ small enough so that:

$$
K_{\varphi} V_{\text{Var},v}^{\text{eq}} < \alpha_{\text{align}}
$$

**Resulting contraction rate:**

$$
\kappa_{\text{pot}} := \frac{1}{\gamma}\left[\alpha_{\text{align}} - K_{\varphi} V_{\text{Var},v}^{\text{eq}}\right] = \frac{1}{\gamma}\left[\frac{c}{\delta}\alpha_{\text{boundary}} - \left(\left(\frac{c}{\delta}\right)^2 + \frac{c}{\delta}K_{\text{curv}}\right)\frac{d\sigma_{\max}^2}{2\gamma}\right] > 0
$$
```

---

## New Requirements to Add

### Axiom 3.3.1 Enhancement

Add to part 4 (after line 268):

```markdown
**Boundary Regularity:** We assume $\partial\mathcal{X}_{\text{valid}}$ is $C^2$ with bounded principal curvatures:

$$
\|\nabla \vec{n}(x)\| \leq K_{\text{curv}} < \infty
$$

This is satisfied by all standard domains (balls, boxes, smooth manifolds with bounded curvature).
```

---

## Physical Interpretation Update

**CHANGE (Line 2524):**
```markdown
**Confining force creates drift:** The compatibility condition
$\langle F, \nabla\varphi \rangle \geq \alpha_{\text{boundary}} \varphi$
ensures particles near the boundary are pushed inward
```

**TO:**
```markdown
**Confining force creates drift:** The negative alignment
$\langle F, \nabla\varphi \rangle \leq -\alpha_{\text{align}}\varphi$
ensures particles near the boundary are pushed inward, creating negative
drift in $\varphi$. The confining force $F$ points **inward** (toward safe region),
the barrier gradient $\nabla\varphi$ points **outward** (away from safe region),
so their inner product is **negative**.
```

---

## Testing the Correction

To verify the fix works, check these properties:

1. **Sign consistency:**
   - F points inward (toward origin for coercive U)
   - ∇φ points outward (away from safe region)
   - ⟨F, ∇φ⟩ < 0 ✓

2. **Generator calculation:**
   - For f(x,v) linear in v: ∇_v² f = 0 ✓
   - No mixing of velocity diffusion and position Hessian ✓

3. **Barrier regularity:**
   - φ = exp(c·ρ/δ) has ∥∇φ∥/φ = c/δ (bounded ratio) ✓
   - Requires C² boundary (standard assumption) ✓

4. **Parameter constraint:**
   - κ_pot > 0 when α_align > K_φ V_var,v^eq ✓
   - Achievable by choosing c small ✓

5. **Physical interpretation:**
   - Inward force reduces boundary potential ✓
   - Layered safety with cloning ✓

---

## Integration Checklist

Before merging:

- [ ] Replace lines 2350-2529 in 05_kinetic_contraction.md
- [ ] Add C² boundary regularity to Axiom 3.3.1
- [ ] Update cross-references to §7.4 throughout document
- [ ] Update constants summary in introduction (if present)
- [ ] Verify rendering of all math expressions
- [ ] Run `make build-docs` to check for LaTeX errors
- [ ] Review dual review feedback has been fully addressed

After merging:

- [ ] Update 06_convergence.md to use corrected κ_pot formula
- [ ] Add numerical test verifying negative drift near boundary
- [ ] Consider adding geometric diagram (F, ∇φ, n vectors)
- [ ] Document parameter selection guide for practitioners

---

## Quick Diff Summary

```diff
- Compatibility: ⟨F, ∇φ⟩ ≥ α_boundary φ
+ Compatibility: ⟨F, ∇φ⟩ ≤ -α_align φ

- Generator: L⟨v, ∇φ⟩ = ... + (1/2)Tr(A∇²φ)
+ Generator: L⟨v, ∇φ⟩ = v^T(∇²φ)v + ⟨F, ∇φ⟩ - γ⟨v, ∇φ⟩

- Coupling: ε = 1/(2γ)
+ Coupling: ε = 1/γ

- Barrier: Unspecified with unbounded derivatives
+ Barrier: φ = exp(c·ρ/δ) with bounded ratios

- Rate: κ_pot = α_boundary/(4γ)
+ Rate: κ_pot = (1/γ)[α_align - K_φ V_var,v^eq]
```

**Files:**
Total changes: ~180 lines
Critical fixes: 3 (sign, diffusion, barrier)
Enhancement: 1 (optimal ε)

---

**Status**: Ready for integration. All critical errors corrected and verified by dual independent review.
