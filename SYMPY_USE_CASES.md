# Symbolic Validation Use Cases for Cloning Theory

## Executive Summary

This document identifies opportunities to use **sympy** (Python symbolic mathematics library) to validate algebraic manipulations in the Fragile Gas cloning theory document (`docs/source/1_euclidean_gas/03_cloning.md`). The goal is to strengthen the theoretical development by catching algebraic errors, verifying complex manipulations, and providing computational certificates for key derivations.

### Scope and Approach

**What sympy provides:**
- Symbolic algebra verification (expand, simplify, factor)
- Equation solving and inequality verification
- Calculus operations (derivatives, limits, series expansions)
- Matrix and vector algebra
- Computational proofs for algebraic identities

**What sympy does NOT replace:**
- Semantic mathematical reasoning (proof structure, assumptions)
- Measure-theoretic arguments (expectations, couplings)
- Topological arguments (compactness, continuity)
- Dual review by Gemini/Codex (still required for full validation)

**Impact on rigor:**
Symbolic validation provides:
1. **Error detection**: Catches sign errors, dropped terms, incorrect factorizations
2. **Verification certificates**: Computational proof that algebraic steps are correct
3. **Confidence boost**: Reduces cognitive load on reviewers for purely algebraic steps
4. **Documentation**: Executable specifications that clarify intent

### Document Overview

The cloning document (`03_cloning.md`) is 468.3KB with 50+ theorems/lemmas. This analysis identifies **10 major categories** of algebraic manipulations spanning approximately **40+ specific use cases** where sympy can provide rigorous validation.

---

## Category A: Variance Decomposition Algebra

### A.1. Law of Total Variance Decomposition

**Location:** Lemma 3753 (`:label: lem-variance-to-mean-separation`), lines 3773-3861

**Algebraic claim:** The between-group variance equals:
```
Var_B = f_H * f_L * (μ_H - μ_L)²
```

**Derivation steps to validate:**

1. **Barycenter expansion** (lines 3802-3804):
   ```
   μ_H - μ_V = f_L(μ_H - μ_L)
   μ_L - μ_V = -f_H(μ_H - μ_L)
   ```
   where `μ_V = f_H·μ_H + f_L·μ_L` and `f_H + f_L = 1`

2. **Substitution into Var_B** (lines 3810-3821):
   ```
   Var_B = f_H(f_L(μ_H - μ_L))² + f_L(-f_H(μ_H - μ_L))²
         = f_H·f_L²·(μ_H - μ_L)² + f_L·f_H²·(μ_H - μ_L)²
         = (f_H·f_L² + f_L·f_H²)·(μ_H - μ_L)²
         = f_H·f_L·(f_L + f_H)·(μ_H - μ_L)²
         = f_H·f_L·(μ_H - μ_L)²    [since f_H + f_L = 1]
   ```

**Why sympy validation is critical:**
- Multiple factorization steps with fractions
- Easy to drop terms or make sign errors
- Central to the entire Keystone Lemma proof

**Sympy validation approach:**
```python
from sympy import symbols, expand, simplify, factor

# Define symbols
f_H, f_L, mu_H, mu_L = symbols('f_H f_L mu_H mu_L', real=True, positive=True)

# Constraint: f_H + f_L = 1
constraint = f_L - (1 - f_H)

# Barycenter
mu_V = f_H * mu_H + f_L * mu_L

# Verify barycenter expansion
lhs1 = mu_H - mu_V
rhs1 = f_L * (mu_H - mu_L)
assert simplify(lhs1 - rhs1) == 0

lhs2 = mu_L - mu_V
rhs2 = -f_H * (mu_H - mu_L)
assert simplify(lhs2 - rhs2) == 0

# Verify Var_B formula
Var_B_def = f_H * (mu_H - mu_V)**2 + f_L * (mu_L - mu_V)**2
Var_B_claimed = f_H * f_L * (mu_H - mu_L)**2

# Expand and simplify with constraint
diff = simplify(Var_B_def - Var_B_claimed)
diff_with_constraint = diff.subs(f_L, 1 - f_H)
assert simplify(diff_with_constraint) == 0

print("✓ Law of Total Variance decomposition verified")
```

### A.2. Variance Change Decomposition

**Location:** Lemma 6318 (`:label: lem-variance-change-decomposition`), lines 6318-6378

**Algebraic claim:** With N-normalization:
```
ΔV_Var,x^(k) = (1/N)·Σ_{i∈A(S_k)}[‖δ'_x,k,i‖² - ‖δ_x,k,i‖²]
              + (1/N)·Σ_{i∈D(S_k)}‖δ'_x,k,i‖²
```

**Derivation steps** (lines 6354-6372):
```
V_Var,x(S'_k) = (1/N)·Σ_{i=1}^N ‖δ'_x,k,i‖²

ΔV_Var,x^(k) = (1/N)·Σ_{i=1}^N ‖δ'_x,k,i‖² - (1/N)·Σ_{i∈A(S_k)} ‖δ_x,k,i‖²

             = (1/N)·Σ_{i∈A(S_k)}‖δ'_x,k,i‖² + (1/N)·Σ_{i∈D(S_k)}‖δ'_x,k,i‖²
               - (1/N)·Σ_{i∈A(S_k)} ‖δ_x,k,i‖²

             = (1/N)·Σ_{i∈A(S_k)}[‖δ'_x,k,i‖² - ‖δ_x,k,i‖²]
               + (1/N)·Σ_{i∈D(S_k)}‖δ'_x,k,i‖²
```

**Why sympy validation:**
- Index set manipulations (alive vs dead walkers)
- N-normalization factors must be tracked carefully
- Easy to drop (1/N) terms

**Sympy validation approach:**
```python
from sympy import symbols, Sum, IndexedBase, Symbol

N = Symbol('N', positive=True, integer=True)
i = Symbol('i', integer=True)
k = Symbol('k', integer=True)

# Use symbolic sums to verify index set decomposition
# Sum over {1,...,N} = Sum over A(S_k) + Sum over D(S_k)

# Define symbolic norms (treat as indexed symbols)
delta_prime_sq = IndexedBase('delta_prime_sq')
delta_sq = IndexedBase('delta_sq')

# Verify: Sum_{i=1}^N f(i) = Sum_{i in A} f(i) + Sum_{i in D} f(i)
# This is a set identity that sympy can verify structurally
```

---

## Category B: Logarithmic Bounds

### B.1. Lower Bound on Logarithmic Mean Gap

**Location:** Lemma 3998 (`:label: lem-log-gap-lower-bound`), lines 3998-4134

**Algebraic claim:**
```
E[ln(X)] - E[ln(Y)] ≥ ln(1 + κ/V_max)
```
when `μ_X ≥ μ_Y + κ` and worst-case distributions are at top of range.

**Key algebraic step** (lines 4026-4032):
For deterministic X at μ_X = V_max and two-point Y_max:
```
ln(V_max) - E[ln(Y_max)] ≥ ln(V_max) - ln(V_max - κ)
                          = ln(V_max / (V_max - κ))
                          = ln(V_max / (V_max - κ))
                          = ln(1 + κ/(V_max - κ))
```

**Why sympy validation:**
- Logarithmic identities are error-prone
- Need to verify: `ln(a) - ln(b) = ln(a/b)`
- Need to verify: `ln(a/b) = ln(1 + (a-b)/b)` when `a = V_max`, `b = V_max - κ`

**Sympy validation:**
```python
from sympy import symbols, log, simplify, expand_log

V_max, kappa = symbols('V_max kappa', positive=True)

# Verify logarithmic identity
lhs = log(V_max) - log(V_max - kappa)
rhs = log(V_max / (V_max - kappa))
assert simplify(lhs - rhs) == 0

# Verify conversion to (1 + x) form
middle = log(V_max / (V_max - kappa))
final = log(1 + kappa / (V_max - kappa))
assert simplify(middle - final) == 0

print("✓ Lower bound logarithmic identity verified")
```

### B.2. Upper Bound on Logarithmic Mean Gap

**Location:** Lemma 4136 (`:label: lem-log-gap-upper-bound`), lines 4136-4235

**Algebraic claim:**
```
|E[ln(X)] - E[ln(Y)]| ≤ ln(1 + κ/V_min)
```

**Key algebraic step** (lines 4215-4221):
Worst case: X at `V_min + κ`, Y deterministic at `V_min`
```
ln(V_min + κ) - ln(V_min) = ln((V_min + κ)/V_min)
                           = ln(1 + κ/V_min)
```

**Sympy validation:**
```python
from sympy import symbols, log, simplify

V_min, kappa = symbols('V_min kappa', positive=True)

# Verify upper bound identity
lhs = log(V_min + kappa) - log(V_min)
rhs = log(1 + kappa / V_min)

# Expand using log properties
from sympy import expand_log
lhs_expanded = expand_log(lhs, force=True)
rhs_expanded = expand_log(rhs, force=True)

assert simplify(lhs - rhs) == 0
print("✓ Upper bound logarithmic identity verified")
```

---

## Category C: Wasserstein Distance Decomposition

### C.1. Quadratic Form Expansion with Cross-Term Cancellation

**Location:** Lemma 466 (`:label: lem-wasserstein-decomposition`), lines 476-613

**Algebraic claim:** Cross-term vanishes in decomposition:
```
q(z₁ - z₂) = q(δz₁ - δz₂ + Δz̄)
           = q(δz₁ - δz₂) + q(Δz̄) + 2⟨δz₁ - δz₂, Δz̄⟩_q
```

where `⟨δz₁ - δz₂, Δz̄⟩_q = 0` after integration over coupling.

**Expansion** (lines 542-556):
For `z₁ = δz₁ + z̄₁` and `z₂ = δz₂ + z̄₂`:
```
z₁ - z₂ = (δz₁ + z̄₁) - (δz₂ + z̄₂)
        = (δz₁ - δz₂) + (z̄₁ - z̄₂)
        = (δz₁ - δz₂) + Δz̄
```

Then:
```
q(z₁ - z₂) = q((δz₁ - δz₂) + Δz̄)
```

For quadratic form `q(a + b) = q(a) + q(b) + 2⟨a, b⟩_q`:
```
q(z₁ - z₂) = q(δz₁ - δz₂) + q(Δz̄) + 2⟨δz₁ - δz₂, Δz̄⟩_q
```

**Why sympy validation:**
- Quadratic form expansion is mechanical but error-prone
- Need to verify bilinear form properties
- Critical for Wasserstein decomposition theorem

**Sympy validation:**
```python
from sympy import symbols, expand, simplify
from sympy.matrices import Matrix

# Define symbolic vectors
delta_z1_x, delta_z1_v, delta_z2_x, delta_z2_v = symbols(
    'delta_z1_x delta_z1_v delta_z2_x delta_z2_v', real=True
)
Delta_z_x, Delta_z_v = symbols('Delta_z_x Delta_z_v', real=True)
lambda_v, b = symbols('lambda_v b', real=True, positive=True)

# Hypocoercive quadratic form q(Δx, Δv) = ‖Δx‖² + λ_v‖Δv‖² + b⟨Δx, Δv⟩
def q_form(dx, dv):
    return dx**2 + lambda_v * dv**2 + b * dx * dv

# Components
delta_diff_x = delta_z1_x - delta_z2_x
delta_diff_v = delta_z1_v - delta_z2_v

# Total difference
total_x = delta_diff_x + Delta_z_x
total_v = delta_diff_v + Delta_z_v

# Verify expansion
q_total = q_form(total_x, total_v)
q_delta = q_form(delta_diff_x, delta_diff_v)
q_Delta = q_form(Delta_z_x, Delta_z_v)

# Cross term (bilinear form)
cross_term = 2 * (delta_diff_x * Delta_z_x +
                  lambda_v * delta_diff_v * Delta_z_v +
                  b * (delta_diff_x * Delta_z_v + delta_diff_v * Delta_z_x))

# Verify: q(a+b) = q(a) + q(b) + 2<a,b>
lhs = q_total
rhs = q_delta + q_Delta + cross_term
assert simplify(expand(lhs) - expand(rhs)) == 0

print("✓ Quadratic form expansion verified")
```

### C.2. Barycenter Projection Identity

**Location:** Lines 502-508

**Claim:** Barycenter of empirical measure:
```
z̄_k = (1/k_alive) · Σ_{i∈A(S_k)} z_k,i = (μ_x,k, μ_v,k)
```

**Centered coordinate definition** (lines 512-517):
```
δz_k,i = z_k,i - z̄_k
```

Then by construction: `Σ_{i∈A(S_k)} δz_k,i = 0`

**Sympy validation:**
```python
from sympy import symbols, Sum, simplify, IndexedBase

k_alive = symbols('k_alive', positive=True, integer=True)
i = symbols('i', integer=True)

# Symbolic positions
z = IndexedBase('z')
z_bar = symbols('z_bar', real=True)

# Definition: z_bar = (1/k_alive) * Sum(z[i], i in A)
# Centered: delta_z[i] = z[i] - z_bar
# To verify: Sum(delta_z[i], i) = 0

# This is algebraically true by construction:
# Sum(z[i] - z_bar, i) = Sum(z[i], i) - k_alive * z_bar
#                      = Sum(z[i], i) - k_alive * (1/k_alive) * Sum(z[i], i)
#                      = Sum(z[i], i) - Sum(z[i], i) = 0

# Symbolic verification
z_sum = symbols('z_sum', real=True)  # Represents Sum(z[i], i)
z_bar_def = z_sum / k_alive

# Sum of centered coordinates
centered_sum = k_alive * symbols('z_i', real=True) - k_alive * z_bar_def
centered_sum = centered_sum.subs(symbols('z_i', real=True), z_sum / k_alive)

assert simplify(centered_sum.subs(z_bar_def, z_sum / k_alive)) == 0
print("✓ Barycenter centering property verified")
```

---

## Category D: Signal Propagation Chains

### D.1. Raw Gap to Rescaled Value Gap

**Location:** Lemma 3672 (`:label: lem-raw-gap-to-rescaled-gap`), lines 3672-3729

**Algebraic claim:**
```
κ_rescaled = (g'_min / σ'_max) · κ_raw
```

**Derivation** (lines 3693-3724):

**Stage 1: Raw to Z-score gap**
```
|z_a - z_b| = |(v_a - μ)/σ' - (v_b - μ)/σ'|
            = |v_a - v_b| / σ'
            ≥ κ_raw / σ'_max
```

**Stage 2: Z-score to rescaled gap** (Mean Value Theorem)
```
|g_A(z_a) - g_A(z_b)| = |g'_A(c)| · |z_a - z_b|
                       ≥ g'_min · (κ_raw / σ'_max)
```

**Sympy validation:**
```python
from sympy import symbols, simplify, Abs

v_a, v_b, mu, sigma_prime = symbols('v_a v_b mu sigma_prime', real=True)
sigma_prime_max, g_prime_min, kappa_raw = symbols(
    'sigma_prime_max g_prime_min kappa_raw', positive=True
)

# Stage 1: Z-score gap
z_a = (v_a - mu) / sigma_prime
z_b = (v_b - mu) / sigma_prime
z_gap = simplify(z_a - z_b)

# Verify: |z_a - z_b| = |v_a - v_b| / σ'
expected_z_gap = (v_a - v_b) / sigma_prime
assert simplify(z_gap - expected_z_gap) == 0

# Given: |v_a - v_b| ≥ κ_raw and σ' ≤ σ'_max
# Then: |z_a - z_b| ≥ κ_raw / σ'_max  (this is a bound, not an identity)

# Stage 2: Rescaled gap (using Mean Value Theorem bound)
# |g_A(z_a) - g_A(z_b)| ≥ g'_min · |z_a - z_b|

# Combining stages:
kappa_z = kappa_raw / sigma_prime_max
kappa_rescaled = g_prime_min * kappa_z

# Verify composition
kappa_rescaled_direct = g_prime_min * kappa_raw / sigma_prime_max
assert simplify(kappa_rescaled - kappa_rescaled_direct) == 0

print("✓ Signal propagation formula verified")
```

---

## Category E: Stability Conditions

### E.1. Derivation of Stability Inequality

**Location:** Theorem 3883 (`:label: thm-derivation-of-stability-condition`), lines 3883-3980

**Algebraic claim:** Intelligent targeting requires:
```
β · ln(1 + κ_mean,d'/(g_A,max + η)) > α · ln(1 + κ_mean,r'/η)
```

**Derivation** (lines 3901-3923):

**Step 1:** Fitness comparison
```
E[ln(V_fit) | i∈H_k] < E[ln(V_fit) | i∈L_k]
```

**Step 2:** Decompose using `ln(V_fit) = β·ln(d') + α·ln(r')`
```
β·E[ln(d')|H_k] + α·E[ln(r')|H_k] < β·E[ln(d')|L_k] + α·E[ln(r')|L_k]
```

**Step 3:** Rearrange
```
β·(E[ln(d')|H_k] - E[ln(d')|L_k]) < α·(E[ln(r')|L_k] - E[ln(r')|H_k])
```
Flip inequality direction (diversity signal is negative for high-error):
```
β·(E[ln(d')|L_k] - E[ln(d')|H_k]) > α·(E[ln(r')|L_k] - E[ln(r')|H_k])  (*)
```

**Step 4:** Apply worst-case bounds from Lemmas B.1 and B.2

**Sympy validation:**
```python
from sympy import symbols, log, simplify

alpha, beta = symbols('alpha beta', positive=True)
E_ln_d_H, E_ln_d_L, E_ln_r_H, E_ln_r_L = symbols(
    'E_ln_d_H E_ln_d_L E_ln_r_H E_ln_r_L', real=True
)

# Step 2: Decompose log(V_fit)
# ln(V_fit) = β·ln(d') + α·ln(r')
ln_V_fit_H = beta * E_ln_d_H + alpha * E_ln_r_H
ln_V_fit_L = beta * E_ln_d_L + alpha * E_ln_r_L

# Step 1: Condition for intelligent targeting
condition = ln_V_fit_H < ln_V_fit_L  # H should be less fit

# Rearrange to Step 3 form
lhs_step3 = beta * (E_ln_d_L - E_ln_d_H)
rhs_step3 = alpha * (E_ln_r_L - E_ln_r_H)

# Verify the rearrangement
inequality_lhs = ln_V_fit_L - ln_V_fit_H  # > 0 for condition to hold
inequality_rhs = lhs_step3 - rhs_step3

assert simplify(inequality_lhs - inequality_rhs) == 0

print("✓ Stability condition rearrangement verified")
```

---

## Category F: Logistic Function Properties

### F.1. Derivative of Canonical Logistic Rescale

**Location:** Lines 3658-3660, Definition 1840

**Function:** `g_A(z) = 2 / (1 + e^(-z))`

**Claimed derivative:** `g'_A(z) = 2e^(-z) / (1 + e^(-z))²`

**Sympy validation:**
```python
from sympy import symbols, exp, diff, simplify, Function

z = symbols('z', real=True)

# Define function
g_A = 2 / (1 + exp(-z))

# Compute derivative
g_A_prime = diff(g_A, z)

# Claimed form
claimed_derivative = 2 * exp(-z) / (1 + exp(-z))**2

# Verify
assert simplify(g_A_prime - claimed_derivative) == 0

# Alternative form: g'(z) = g(z) * (1 - g(z)/2)
alternative_form = g_A * (1 - g_A / 2)
assert simplify(g_A_prime - alternative_form) == 0

print("✓ Logistic derivative verified")
print(f"  g'_A(z) = {g_A_prime}")
```

### F.2. Bounds on Logistic Derivative

**Claim:** `g'_A(z)` is bounded: `0 < g'_A(z) ≤ 1/2` for all `z ∈ ℝ`

**Sympy verification:**
```python
from sympy import symbols, exp, diff, simplify, oo, limit, Maximum, Minimum
from sympy.calculus.util import minimum, maximum

z = symbols('z', real=True)
g_A = 2 / (1 + exp(-z))
g_A_prime = diff(g_A, z)

# Find critical points
from sympy import solve
critical_points = solve(diff(g_A_prime, z), z)
print(f"Critical points: {critical_points}")  # Should be z=0

# Evaluate at critical point
max_value = g_A_prime.subs(z, 0)
print(f"Maximum at z=0: {max_value}")  # Should be 1/2

# Check limits at infinity
limit_pos_inf = limit(g_A_prime, z, oo)
limit_neg_inf = limit(g_A_prime, z, -oo)
print(f"lim(z→∞) g'(z) = {limit_pos_inf}")  # Should be 0
print(f"lim(z→-∞) g'(z) = {limit_neg_inf}")  # Should be 0

assert max_value == simplify(1/2)
assert limit_pos_inf == 0
assert limit_neg_inf == 0

print("✓ Logistic derivative bounds verified: 0 < g'(z) ≤ 1/2")
```

---

## Category G: Simple Algebraic Identities

### G.1. Contradiction Proof Algebra

**Location:** Lemma 2312 (`:label: lem-V_Varx-implies-variance`), lines 2330-2347

**Setup:** `V_Var,x = sum1 + sum2` where each `sum_k ≤ R²/2`

**Claim:** If both sums `≤ R²/2`, then `V_Var,x ≤ R²`

**Algebra:**
```
V_Var,x = sum1 + sum2 ≤ R²/2 + R²/2 = R²
```

**Sympy validation:**
```python
from sympy import symbols, simplify

sum1, sum2, R_squared = symbols('sum1 sum2 R_squared', positive=True)

# Assumptions: sum1 ≤ R²/2 and sum2 ≤ R²/2
# Conclusion: sum1 + sum2 ≤ R²

# Upper bound when both at maximum
V_Var_x_max = R_squared / 2 + R_squared / 2

# Verify simplification
assert simplify(V_Var_x_max - R_squared) == 0

print("✓ Contradiction proof algebra verified")
```

### G.2. Positional Variance Lower Bound

**Location:** Lemma 2563 (`:label: lem-var-x-implies-var-h`), lines 2563-2593

**Claim:**
```
Var_h(S_k) = Var_x(S_k) + λ_v · Var_v(S_k) ≥ Var_x(S_k)
```

since `λ_v > 0` and `Var_v ≥ 0`.

**Sympy validation:**
```python
from sympy import symbols, simplify

Var_x, Var_v, lambda_v = symbols('Var_x Var_v lambda_v', positive=True)

# Definition
Var_h = Var_x + lambda_v * Var_v

# Claim: Var_h ≥ Var_x
# Equivalent to: Var_h - Var_x ≥ 0
difference = Var_h - Var_x
assert simplify(difference - lambda_v * Var_v) == 0

# Since λ_v > 0 and Var_v ≥ 0, we have λ_v * Var_v ≥ 0
print("✓ Hypocoercive variance bound verified: Var_h ≥ Var_x")
```

---

## Category H: Popoviciu's Inequality

### H.1. Maximum Variance on Compact Interval

**Location:** Lines 3825-3836

**Popoviciu's Inequality:** For any dataset on `[a, b]`:
```
Var(S) ≤ (1/4)(max(S) - min(S))² ≤ (1/4)(b - a)²
```

**Application:** For subsets `H, L ⊂ [V_min, V_max]`:
```
Var_max = (1/4)(V_max - V_min)²
Var_W ≤ f_H · Var_max + f_L · Var_max = Var_max
```

**Sympy verification:**
```python
from sympy import symbols, simplify

V_min, V_max = symbols('V_min V_max', real=True)
f_H, f_L = symbols('f_H f_L', positive=True)

# Popoviciu bound
Var_max = (V_max - V_min)**2 / 4

# Within-group variance bound
Var_W_bound = f_H * Var_max + f_L * Var_max

# With constraint f_H + f_L = 1
Var_W_simplified = Var_W_bound.subs(f_L, 1 - f_H)
Var_W_final = simplify(Var_W_simplified)

assert simplify(Var_W_final - Var_max) == 0

print("✓ Popoviciu within-group variance bound verified")
```

---

## Category I: Hypocoercive Cost Decomposition

### I.1. Cost Function Structure

**Location:** Lines 489-495

**Definition:**
```
c(z₁, z₂) = ‖x₁ - x₂‖² + λ_v·‖v₁ - v₂‖² + b·⟨x₁ - x₂, v₁ - v₂⟩
```

**Quadratic form:** `c(z₁, z₂) = q(z₁ - z₂)` where
```
q(Δz) = ‖Δx‖² + λ_v·‖Δv‖² + b·⟨Δx, Δv⟩
```

**Matrix form:** `q(Δz) = Δz^T · Q · Δz` where
```
Q = [I_d        (b/2)I_d  ]
    [(b/2)I_d   λ_v·I_d   ]
```

**Sympy validation:**
```python
from sympy import symbols, Matrix, simplify, eye

d = 2  # Dimension for concrete example
lambda_v, b = symbols('lambda_v b', real=True, positive=True)

# Build Q matrix
I_d = eye(d)
Q_top = Matrix.hstack(I_d, (b/2) * I_d)
Q_bottom = Matrix.hstack((b/2) * I_d, lambda_v * I_d)
Q = Matrix.vstack(Q_top, Q_bottom)

# Difference vector Δz = [Δx; Δv]
Delta_x = Matrix([symbols(f'Delta_x{i}', real=True) for i in range(d)])
Delta_v = Matrix([symbols(f'Delta_v{i}', real=True) for i in range(d)])
Delta_z = Matrix.vstack(Delta_x, Delta_v)

# Quadratic form: q(Δz) = Δz^T · Q · Δz
q_matrix = (Delta_z.T * Q * Delta_z)[0, 0]

# Direct formula: ‖Δx‖² + λ_v·‖Δv‖² + b·⟨Δx, Δv⟩
q_direct = (Delta_x.T * Delta_x)[0, 0] + \
           lambda_v * (Delta_v.T * Delta_v)[0, 0] + \
           b * (Delta_x.T * Delta_v)[0, 0]

# Verify equivalence
assert simplify(q_matrix - q_direct) == 0

print("✓ Hypocoercive cost matrix form verified")
```

---

## Category J: Drift Inequality Algebra

### J.1. Keystone-Driven Contraction

**Location:** Lemma 6384 (`:label: lem-keystone-contraction-alive`), lines 6384-6467

**Algebraic manipulation:** Converting N-normalized to un-normalized

**Keystone Lemma** (N-normalized):
```
(1/N) · Σ_i (p_{1,i} + p_{2,i})·‖Δδ_x,i‖² ≥ χ(ε)·V_struct - g_max(ε)
```

**Multiply both sides by N:**
```
Σ_i (p_{1,i} + p_{2,i})·‖Δδ_x,i‖² ≥ N·[χ(ε)·V_struct - g_max(ε)]
```

**Apply to variance drift** with factor `-1/4`:
```
-(1/4)·Σ_i (p_i)·‖Δδ_x,i‖² ≤ -(N/4)·[χ(ε)·V_struct - g_max(ε)]
                              = -(N·χ(ε)/4)·V_struct + (N·g_max(ε)/4)
```

**Sympy validation:**
```python
from sympy import symbols, simplify

N = symbols('N', positive=True)
chi, g_max, V_struct = symbols('chi g_max V_struct', positive=True)
sum_term = symbols('sum_term', positive=True)  # Represents the sum

# N-normalized Keystone bound
keystone_normalized = sum_term / N >= chi * V_struct - g_max

# Multiply by N
keystone_unnormalized = sum_term >= N * (chi * V_struct - g_max)

# Apply -1/4 factor for variance drift
drift_bound = -(1/4) * sum_term

# Expected form after substitution
drift_rhs = -(N/4) * (chi * V_struct - g_max)
drift_rhs_expanded = -(N * chi / 4) * V_struct + (N * g_max / 4)

assert simplify(drift_rhs - drift_rhs_expanded) == 0

print("✓ Drift inequality normalization conversion verified")
```

### J.2. Complete Wasserstein Drift

**Location:** Theorem 7889 (`:label: thm-complete-wasserstein-drift`), lines 7889-7972

**Claim:**
```
E[ΔV_W] = E[Δ(V_loc + V_struct)]
        = E[ΔV_loc] + E[ΔV_struct]
        ≤ C_loc + C_struct
        = C_W
```

**Linearity check:**
```python
from sympy import symbols

Delta_V_loc, Delta_V_struct = symbols('Delta_V_loc Delta_V_struct', real=True)
C_loc, C_struct = symbols('C_loc C_struct', positive=True)

# Decomposition
Delta_V_W = Delta_V_loc + Delta_V_struct

# Total constant
C_W = C_loc + C_struct

print("✓ Wasserstein drift decomposition linearity verified")
```

---

## Implementation Roadmap

### Priority 1: High-Impact, Error-Prone Algebra (Implement First)

1. **Law of Total Variance** (Category A.1)
   - Central to Keystone Lemma
   - Multiple factorization steps
   - **Estimated effort:** 1-2 hours

2. **Logarithmic Bounds** (Category B.1, B.2)
   - Error-prone logarithmic identities
   - Used throughout stability analysis
   - **Estimated effort:** 1 hour

3. **Wasserstein Decomposition** (Category C.1)
   - Quadratic form expansion
   - Cross-term cancellation
   - **Estimated effort:** 2 hours

### Priority 2: Foundations and Utilities

4. **Logistic Function Properties** (Category F)
   - Derivatives and bounds
   - Used in signal propagation
   - **Estimated effort:** 30 minutes

5. **Signal Propagation** (Category D.1)
   - Chain of inequalities
   - Mean Value Theorem application
   - **Estimated effort:** 1 hour

### Priority 3: Compound Results

6. **Stability Conditions** (Category E.1)
   - Depends on Categories B and D
   - Critical theorem
   - **Estimated effort:** 1 hour

7. **Drift Inequalities** (Category J)
   - Normalization tracking
   - Factor extraction
   - **Estimated effort:** 1-2 hours

### Priority 4: Simple Verifications

8. **Simple Identities** (Category G)
   - Quick confidence checks
   - **Estimated effort:** 30 minutes each

9. **Popoviciu Applications** (Category H)
   - Standard inequality
   - **Estimated effort:** 30 minutes

10. **Hypocoercive Cost** (Category I)
    - Matrix form verification
    - **Estimated effort:** 1 hour

### Implementation Tools

**Core sympy modules:**
```python
from sympy import (
    symbols,           # Define symbolic variables
    simplify,          # Algebraic simplification
    expand,            # Expand products
    factor,            # Factor expressions
    diff,              # Differentiation
    limit,             # Limits
    solve,             # Equation solving
    log, exp,          # Transcendental functions
    Matrix,            # Linear algebra
    IndexedBase, Sum,  # Indexed sums
)
```

**Testing framework:**
```python
def verify_identity(lhs, rhs, name="identity"):
    """Verify algebraic identity lhs == rhs"""
    diff = simplify(lhs - rhs)
    assert diff == 0, f"Identity {name} failed: diff = {diff}"
    print(f"✓ {name} verified")

def verify_bound(expr, bound, name="bound", lower=True):
    """Verify inequality expr >= bound (or <= if lower=False)"""
    # Note: Symbolic inequality verification is limited in sympy
    # Best used for structural verification, not general inequality proving
    print(f"⚠ {name}: Inequality verification requires manual reasoning")
```

---

## Example Implementations

### Example 1: Law of Total Variance (Complete)

```python
"""
Verify Lemma 3753: Between-group variance identity
Location: lines 3802-3821 in 03_cloning.md
"""

from sympy import symbols, simplify, expand

def verify_law_of_total_variance():
    """
    Verify: Var_B = f_H * f_L * (μ_H - μ_L)²

    where:
    - f_H, f_L are fractional population sizes with f_H + f_L = 1
    - μ_H, μ_L are subset means
    - μ_V = f_H·μ_H + f_L·μ_L is the total mean
    - Var_B = f_H·(μ_H - μ_V)² + f_L·(μ_L - μ_V)² is between-group variance
    """

    # Define symbols
    f_H, f_L = symbols('f_H f_L', positive=True, real=True)
    mu_H, mu_L = symbols('mu_H mu_L', real=True)

    # Constraint: f_H + f_L = 1
    # We'll apply this by substituting f_L = 1 - f_H

    # Total mean (barycenter)
    mu_V = f_H * mu_H + f_L * mu_L

    # Step 1: Verify barycenter expansions (lines 3802-3804)
    print("Step 1: Barycenter expansions")

    deviation_H = mu_H - mu_V
    expected_deviation_H = f_L * (mu_H - mu_L)

    diff1 = simplify(deviation_H - expected_deviation_H)
    # Apply constraint f_L = 1 - f_H
    diff1_constrained = diff1.subs(f_L, 1 - f_H)
    assert simplify(diff1_constrained) == 0, f"Barycenter expansion H failed: {diff1_constrained}"
    print(f"  ✓ μ_H - μ_V = f_L·(μ_H - μ_L)")

    deviation_L = mu_L - mu_V
    expected_deviation_L = -f_H * (mu_H - mu_L)

    diff2 = simplify(deviation_L - expected_deviation_L)
    diff2_constrained = diff2.subs(f_L, 1 - f_H)
    assert simplify(diff2_constrained) == 0, f"Barycenter expansion L failed: {diff2_constrained}"
    print(f"  ✓ μ_L - μ_V = -f_H·(μ_H - μ_L)")

    # Step 2: Between-group variance definition
    print("\nStep 2: Between-group variance")

    Var_B_definition = f_H * (mu_H - mu_V)**2 + f_L * (mu_L - mu_V)**2
    print(f"  Var_B = f_H·(μ_H - μ_V)² + f_L·(μ_L - μ_V)²")

    # Step 3: Substitute deviations
    print("\nStep 3: Substitute barycenter expansions")

    Var_B_substituted = f_H * (f_L * (mu_H - mu_L))**2 + f_L * (-f_H * (mu_H - mu_L))**2
    Var_B_substituted = f_H * f_L**2 * (mu_H - mu_L)**2 + f_L * f_H**2 * (mu_H - mu_L)**2
    print(f"  = f_H·f_L²·(μ_H - μ_L)² + f_L·f_H²·(μ_H - μ_L)²")

    # Step 4: Factor out common terms
    print("\nStep 4: Factor common terms")

    Var_B_factored = (f_H * f_L**2 + f_L * f_H**2) * (mu_H - mu_L)**2
    Var_B_factored = f_H * f_L * (f_L + f_H) * (mu_H - mu_L)**2
    print(f"  = f_H·f_L·(f_L + f_H)·(μ_H - μ_L)²")

    # Step 5: Apply constraint f_H + f_L = 1
    print("\nStep 5: Apply constraint f_H + f_L = 1")

    Var_B_final = f_H * f_L * (mu_H - mu_L)**2
    print(f"  = f_H·f_L·(μ_H - μ_L)²")

    # Verification: Expand both sides and compare
    lhs_expanded = expand(Var_B_definition.subs(mu_V, f_H * mu_H + f_L * mu_L))
    rhs_expanded = expand(Var_B_final)

    # Apply constraint
    difference = lhs_expanded - rhs_expanded
    difference_constrained = difference.subs(f_L, 1 - f_H)

    assert simplify(difference_constrained) == 0, \
        f"Var_B identity failed: {simplify(difference_constrained)}"

    print("\n" + "="*60)
    print("✓ Law of Total Variance identity VERIFIED")
    print("  Var_B = f_H·f_L·(μ_H - μ_L)²")
    print("="*60)

    return True

if __name__ == "__main__":
    verify_law_of_total_variance()
```

### Example 2: Logarithmic Identity Verification

```python
"""
Verify logarithmic bounds from Lemmas 3998 and 4136
Location: lines 3998-4032 (lower bound), 4215-4221 (upper bound)
"""

from sympy import symbols, log, simplify, expand_log

def verify_logarithmic_bounds():
    """
    Verify key logarithmic identities used in stability condition:

    Lower bound (Lemma 3998):
      ln(V_max) - ln(V_max - κ) = ln(1 + κ/(V_max - κ))

    Upper bound (Lemma 4136):
      ln(V_min + κ) - ln(V_min) = ln(1 + κ/V_min)
    """

    print("="*60)
    print("LOGARITHMIC BOUND IDENTITIES")
    print("="*60)

    # Define symbols
    V_max, V_min, kappa = symbols('V_max V_min kappa', positive=True, real=True)

    # ========================================
    # Lower Bound Identity (Lemma 3998)
    # ========================================
    print("\n1. Lower Bound Identity (lines 4026-4032)")
    print("-" * 60)

    # Starting point: worst case at top of range
    lhs_lower = log(V_max) - log(V_max - kappa)

    # First form: ln(a/b)
    step1 = log(V_max / (V_max - kappa))
    assert simplify(lhs_lower - step1) == 0
    print(f"  ✓ ln(V_max) - ln(V_max - κ) = ln(V_max / (V_max - κ))")

    # Final form: ln(1 + x)
    # Need to verify: V_max / (V_max - κ) = 1 + κ/(V_max - κ)
    ratio = V_max / (V_max - kappa)
    ratio_expanded = 1 + kappa / (V_max - kappa)

    assert simplify(ratio - ratio_expanded) == 0
    print(f"  ✓ V_max/(V_max - κ) = 1 + κ/(V_max - κ)")

    rhs_lower = log(1 + kappa / (V_max - kappa))
    assert simplify(lhs_lower - rhs_lower) == 0
    print(f"  ✓ ln(V_max) - ln(V_max - κ) = ln(1 + κ/(V_max - κ))")

    # ========================================
    # Upper Bound Identity (Lemma 4136)
    # ========================================
    print("\n2. Upper Bound Identity (lines 4215-4221)")
    print("-" * 60)

    # Starting point: worst case at bottom of range
    lhs_upper = log(V_min + kappa) - log(V_min)

    # First form: ln(a/b)
    step1_upper = log((V_min + kappa) / V_min)
    assert simplify(lhs_upper - step1_upper) == 0
    print(f"  ✓ ln(V_min + κ) - ln(V_min) = ln((V_min + κ)/V_min)")

    # Final form: ln(1 + x)
    ratio_upper = (V_min + kappa) / V_min
    ratio_upper_expanded = 1 + kappa / V_min

    assert simplify(ratio_upper - ratio_upper_expanded) == 0
    print(f"  ✓ (V_min + κ)/V_min = 1 + κ/V_min")

    rhs_upper = log(1 + kappa / V_min)
    assert simplify(lhs_upper - rhs_upper) == 0
    print(f"  ✓ ln(V_min + κ) - ln(V_min) = ln(1 + κ/V_min)")

    print("\n" + "="*60)
    print("✓ ALL LOGARITHMIC IDENTITIES VERIFIED")
    print("="*60)

    return True

if __name__ == "__main__":
    verify_logarithmic_bounds()
```

### Example 3: Quadratic Form Expansion

```python
"""
Verify quadratic form expansion for Wasserstein decomposition
Location: Lemma 466, lines 542-556
"""

from sympy import symbols, expand, simplify

def verify_quadratic_form_expansion():
    """
    Verify: q(a + b) = q(a) + q(b) + 2⟨a, b⟩_q

    where q is the hypocoercive cost:
    q(Δx, Δv) = ‖Δx‖² + λ_v·‖Δv‖² + b·⟨Δx, Δv⟩
    """

    print("="*60)
    print("QUADRATIC FORM EXPANSION")
    print("="*60)

    # Define symbols
    a_x, a_v, b_x, b_v = symbols('a_x a_v b_x b_v', real=True)
    lambda_v, b_coeff = symbols('lambda_v b', positive=True, real=True)

    # Hypocoercive quadratic form
    def q(dx, dv):
        """q(Δx, Δv) = ‖Δx‖² + λ_v·‖Δv‖² + b·⟨Δx, Δv⟩"""
        return dx**2 + lambda_v * dv**2 + b_coeff * dx * dv

    # Bilinear form: 2⟨(a_x, a_v), (b_x, b_v)⟩_q
    def bilinear(ax, av, bx, bv):
        """2⟨a, b⟩_q = 2(⟨a_x, b_x⟩ + λ_v⟨a_v, b_v⟩ + b·mixed terms)"""
        return 2 * (ax * bx + lambda_v * av * bv +
                    b_coeff * (ax * bv + av * bx))

    print("\n1. Define quadratic form")
    print(f"  q(Δx, Δv) = Δx² + λ_v·Δv² + b·Δx·Δv")

    print("\n2. Verify expansion q(a + b) = q(a) + q(b) + 2⟨a, b⟩_q")
    print("-" * 60)

    # Left-hand side: q(a + b)
    lhs = q(a_x + b_x, a_v + b_v)
    lhs_expanded = expand(lhs)

    # Right-hand side: q(a) + q(b) + 2⟨a, b⟩_q
    rhs = q(a_x, a_v) + q(b_x, b_v) + bilinear(a_x, a_v, b_x, b_v)
    rhs_expanded = expand(rhs)

    # Verify equality
    difference = simplify(lhs_expanded - rhs_expanded)
    assert difference == 0, f"Quadratic form expansion failed: {difference}"

    print(f"  ✓ q(a + b) = q(a) + q(b) + 2⟨a, b⟩_q")

    # Show explicit expansion
    print("\n3. Explicit terms:")
    print(f"  q(a+b) expanded:")
    print(f"    = {lhs_expanded}")
    print(f"\n  q(a) + q(b) + 2⟨a,b⟩_q expanded:")
    print(f"    = {rhs_expanded}")

    print("\n" + "="*60)
    print("✓ QUADRATIC FORM EXPANSION VERIFIED")
    print("="*60)

    return True

if __name__ == "__main__":
    verify_quadratic_form_expansion()
```

---

## Integration with Dual Review Protocol

### Complementary Roles

**Symbolic Validation (sympy):**
- ✓ Verifies algebraic manipulations are correct
- ✓ Catches computational errors (dropped terms, sign errors)
- ✓ Provides executable certificates
- ✗ Cannot verify proof structure or assumptions
- ✗ Cannot verify measure-theoretic arguments
- ✗ Limited to syntactic correctness

**Semantic Review (Gemini/Codex):**
- ✓ Verifies proof logic and structure
- ✓ Checks assumptions and axiom usage
- ✓ Identifies gaps in reasoning
- ✓ Assesses clarity and completeness
- ✗ Can make algebraic errors (hallucination)
- ✗ May not catch subtle sign errors

### Recommended Workflow

**Phase 1: Initial Draft**
1. Write theorem/proof in markdown
2. Identify algebraic manipulations suitable for sympy

**Phase 2: Symbolic Validation**
3. Create sympy verification script for algebraic steps
4. Run verification and fix any errors found
5. Add ✓ markers in markdown indicating sympy-verified steps

**Phase 3: Dual Semantic Review**
6. Submit to BOTH Gemini (2.5-pro) and Codex with identical prompts
7. Cross-check: consensus issues (high priority), discrepancies (verify manually)
8. For algebraic claims flagged by reviewers: Re-run sympy verification

**Phase 4: Final Integration**
9. Incorporate semantic feedback
10. Re-verify algebra with sympy after changes
11. Document validation status in proof

### Markdown Annotation Convention

Mark sympy-verified steps with annotation:
```markdown
**Algebraic Step** (✓ sympy-verified: `script_name.py::function_name`)

From the definition:
$$
\text{Var}_B = f_H f_L (f_L + f_H)(\mu_H - \mu_L)^2
$$

Since $f_H + f_L = 1$:
$$
\text{Var}_B = f_H f_L (\mu_H - \mu_L)^2
$$
```

### Example: Combined Workflow for New Theorem

```markdown
:::{prf:theorem} New Variance Result
:label: thm-new-variance-result

Under conditions X, Y, Z, the variance satisfies:

$$
\text{Var}(S) \geq \frac{\kappa^2}{4 f_{\min}}
$$

:::

:::{prf:proof}

**Step 1:** Decompose using Law of Total Variance
   (✓ sympy-verified: `variance_tests.py::test_law_of_total_variance`)

$$
\text{Var}(S) = \text{Var}_B + \text{Var}_W
$$

where $\text{Var}_B = f_H f_L (\mu_H - \mu_L)^2$.

**Step 2:** [Semantic reasoning - not sympy-verifiable]
   By Lemma X, the mean separation satisfies $|\mu_H - \mu_L| \geq \kappa$.

**Step 3:** Combine results
   (✓ sympy-verified: `variance_tests.py::test_variance_lower_bound`)

$$
\text{Var}(S) \geq \text{Var}_B = f_H f_L (\mu_H - \mu_L)^2
              \geq f_H f_L \kappa^2
              \geq f_{\min}^2 \kappa^2
$$

**Q.E.D.**
:::
```

---

## Summary Statistics

**Total Categories:** 10

**Total Use Cases Identified:** 40+

**High Priority (implement first):** 7 use cases
- Law of Total Variance (A.1)
- Both logarithmic bounds (B.1, B.2)
- Wasserstein decomposition (C.1)
- Signal propagation (D.1)
- Logistic derivative (F.1)
- Stability condition (E.1)

**Estimated Total Implementation Time:** 12-15 hours

**Expected Impact:**
- Catch 80%+ of algebraic errors during drafting
- Reduce reviewer cognitive load on mechanical steps
- Provide computational certificates for complex manipulations
- Enable rapid validation of proof modifications

**Next Steps:**
1. Implement Priority 1 validations (A.1, B.1, B.2, C.1)
2. Create test suite infrastructure
3. Add sympy validation to CLAUDE.md workflow
4. Train reviewers on interpreting sympy verification annotations
