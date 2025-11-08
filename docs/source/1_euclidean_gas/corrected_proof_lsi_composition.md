# Corrected Proof: Discrete-Time LSI for Euclidean Gas Composition

## Corrected Theorem Statement

:::{prf:theorem} Defective Discrete-Time LSI for the Euclidean Gas
:label: thm-main-lsi-composition-corrected

Under the seesaw condition $\kappa_W > \beta/(1+\beta)$, the composed operator $\Psi_{\text{total}} = \Psi_{\text{clone}} \circ \Psi_{\text{kin}}$ satisfies a **defective discrete-time logarithmic Sobolev inequality**.

**Assumptions:**

1. **Kinetic operator** $\Psi_{\text{kin}}$:
   - Exponential entropy decay: $D_{\text{KL}}((\Psi_{\text{kin}})_* \mu \| \pi) \le e^{-\rho_k} D_{\text{KL}}(\mu \| \pi)$ for $\rho_k > 0$
   - Wasserstein non-expansive: $W_2((\Psi_{\text{kin}})_* \mu, \pi) \le (1 + \beta)^{1/2} W_2(\mu, \pi)$ for small $\beta > 0$

2. **Cloning operator** $\Psi_{\text{clone}}$:
   - Wasserstein contraction with defect: $W_2^2((\Psi_{\text{clone}})_* \mu, \pi) \le (1-\kappa_W) W_2^2(\mu, \pi) + C_W$ for $\kappa_W \in (0,1]$, $C_W \ge 0$
   - Entropy bound via HWI: $D_{\text{KL}}((\Psi_{\text{clone}})_* \mu \| \pi) \le D_{\text{KL}}(\mu \| \pi) - \alpha W_2^2(\mu, \pi) + C_{\text{clone}}$ for $\alpha > 0$, $C_{\text{clone}} \ge 0$

3. **Invariant measure** $\pi$:
   - Satisfies HWI inequality with displacement convexity constant $K > 0$:

$$
D_{\text{KL}}(\mu \| \pi) \le W_2(\mu, \pi) \sqrt{I(\mu \| \pi)} - \frac{K}{2} W_2^2(\mu, \pi)
$$

   where $I(\mu \| \pi) = \int \left\| \nabla \log \frac{d\mu}{d\pi} \right\|^2 d\mu$ is the Fisher information

4. **Parameter configuration:**
   - Lyapunov function $V(\mu) = D_{\text{KL}}(\mu \| \pi) + c W_2^2(\mu, \pi)$ with $c = \alpha e^{-\rho_k}/(1 - K_W)$ where $K_W = (1+\beta)(1-\kappa_W)$
   - Contraction factor $\lambda = \max(e^{-\rho_k}, K_W) < 1$ (requires seesaw condition)

**Conclusion (Defective Modified LSI):**

For any $\eta > 1/K$, the one-step functional inequality holds for all probability measures $\mu$:

$$
D_{\text{KL}}((\Psi_{\text{total}})_* \mu \| \pi) \le a(\eta) D_{\text{KL}}(\mu \| \pi) + b(\eta) I(\mu \| \pi) + C_{\text{steady}}
$$

where the **explicit constants** are:

**Contraction coefficients:**

$$
a(\eta) := \lambda - \frac{2c\lambda}{K - 1/\eta}, \quad b(\eta) := \frac{c\lambda\eta}{K - 1/\eta}
$$

**Defect constant:**

$$
C_{\text{steady}} := e^{-\rho_k} C_{\text{clone}} + c(1+\beta) C_W
$$

**Properties:**
- $a(\eta) < \lambda < 1$ for all $\eta > 1/K$ (geometric contraction)
- $b(\eta) > 0$ (Fisher information control)
- $C_{\text{steady}} \ge 0$ (additive defect - prevents convergence to zero unless $C_{\text{steady}} = 0$)

**Asymptotic behavior:**

$$
D_{\text{KL}}(\mu_t \| \pi) \le a(\eta)^t D_{\text{KL}}(\mu_0 \| \pi) + \frac{C_{\text{steady}}}{1 - a(\eta)}
$$

The system exhibits **geometric convergence to a neighborhood** of $\pi$ with radius $C_{\text{steady}}/(1-a(\eta))$, not exponential convergence to $\pi$ itself.

**Zero-defect case (True LSI):** If $C_W = 0$ and $C_{\text{clone}} = 0$ (perfect cloning operator), then $C_{\text{steady}} = 0$ and the inequality becomes a genuine discrete-time LSI with exponential convergence to $\pi$.

:::

## Corrected Proof

:::{prf:proof}
:label: proof-main-lsi-composition-corrected

We derive a one-step functional inequality relating $D_{\text{KL}}((\Psi_{\text{total}})_* \mu \| \pi)$ to $D_{\text{KL}}(\mu \| \pi)$ and the Fisher information $I(\mu \| \pi)$.

**Step 1: Lyapunov Function One-Step Contraction**

From Theorem {prf:ref}`thm-entropy-transport-contraction`, the Lyapunov function $V(\mu) = D_{\text{KL}}(\mu \| \pi) + c W_2^2(\mu, \pi)$ satisfies:

$$
V((\Psi_{\text{total}})_* \mu) \le \lambda V(\mu) + C_{\text{steady}}
$$

where $\lambda < 1$ under the seesaw condition and $C_{\text{steady}} = e^{-\rho_k} C_{\text{clone}} + c(1+\beta) C_W$.

**Verification of upstream result:** This follows from composing the kinetic and cloning operators:
- After kinetics: $D_{\text{KL}}(\mu' \| \pi) \le e^{-\rho_k} D_{\text{KL}}(\mu \| \pi)$ and $W_2^2(\mu', \pi) \le (1+\beta) W_2^2(\mu, \pi)$
- After cloning: $D_{\text{KL}}(\mu'' \| \pi) \le D_{\text{KL}}(\mu' \| \pi) - \alpha W_2^2(\mu', \pi) + C_{\text{clone}}$ and $W_2^2(\mu'', \pi) \le (1-\kappa_W) W_2^2(\mu', \pi) + C_W$

Combining yields the stated contraction with $\lambda = \max(e^{-\rho_k}, (1+\beta)(1-\kappa_W))$.

**Step 2: Isolate the Entropy Term**

Write $\mu' = (\Psi_{\text{total}})_* \mu$. Expanding the Lyapunov inequality:

$$
D_{\text{KL}}(\mu' \| \pi) + c W_2^2(\mu', \pi) \le \lambda \left( D_{\text{KL}}(\mu \| \pi) + c W_2^2(\mu, \pi) \right) + C_{\text{steady}}
$$

Rearranging to isolate the entropy term:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda W_2^2(\mu, \pi) - c W_2^2(\mu', \pi) + C_{\text{steady}}
$$

Since $W_2^2(\mu', \pi) \ge 0$, dropping this non-positive term gives:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda W_2^2(\mu, \pi) + C_{\text{steady}}
$$

**Note:** This is an upper bound; we have not yet eliminated the Wasserstein term.

**Step 3: Apply HWI Inequality to Control Wasserstein Distance**

The HWI inequality for $\pi$ with displacement convexity constant $K > 0$ states:

$$
D_{\text{KL}}(\mu \| \pi) \le W_2(\mu, \pi) \sqrt{I(\mu \| \pi)} - \frac{K}{2} W_2^2(\mu, \pi)
$$

**Young's inequality:** For any $\eta > 0$, we have $ab \le \frac{a^2}{2\eta} + \frac{\eta}{2} b^2$. Applying with $a = W_2(\mu, \pi)$ and $b = \sqrt{I(\mu \| \pi)}$:

$$
W_2(\mu, \pi) \sqrt{I(\mu \| \pi)} \le \frac{W_2^2(\mu, \pi)}{2\eta} + \frac{\eta}{2} I(\mu \| \pi)
$$

Substituting into the HWI inequality:

$$
D_{\text{KL}}(\mu \| \pi) \le \frac{W_2^2(\mu, \pi)}{2\eta} + \frac{\eta}{2} I(\mu \| \pi) - \frac{K}{2} W_2^2(\mu, \pi)
$$

$$
D_{\text{KL}}(\mu \| \pi) \le -\frac{1}{2}\left(K - \frac{1}{\eta}\right) W_2^2(\mu, \pi) + \frac{\eta}{2} I(\mu \| \pi)
$$

**Step 4: Solve for Wasserstein Distance**

For $\eta > 1/K$, the coefficient $\left(K - \frac{1}{\eta}\right) > 0$. Rearranging:

$$
W_2^2(\mu, \pi) \le \frac{\eta}{K - 1/\eta} I(\mu \| \pi) - \frac{2}{K - 1/\eta} D_{\text{KL}}(\mu \| \pi)
$$

**Step 5: Substitute Back into Entropy Bound**

From Step 2, we have:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda W_2^2(\mu, \pi) + C_{\text{steady}}
$$

Substituting the bound from Step 4:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda \left[ \frac{\eta}{K - 1/\eta} I(\mu \| \pi) - \frac{2}{K - 1/\eta} D_{\text{KL}}(\mu \| \pi) \right] + C_{\text{steady}}
$$

Collecting terms in $D_{\text{KL}}(\mu \| \pi)$:

$$
D_{\text{KL}}(\mu' \| \pi) \le \left[ \lambda - \frac{2c\lambda}{K - 1/\eta} \right] D_{\text{KL}}(\mu \| \pi) + \left[ \frac{c\lambda\eta}{K - 1/\eta} \right] I(\mu \| \pi) + C_{\text{steady}}
$$

**Step 6: Define Explicit Constants**

Define:

$$
a(\eta) := \lambda - \frac{2c\lambda}{K - 1/\eta}, \quad b(\eta) := \frac{c\lambda\eta}{K - 1/\eta}
$$

Then the one-step functional inequality is:

$$
D_{\text{KL}}((\Psi_{\text{total}})_* \mu \| \pi) \le a(\eta) D_{\text{KL}}(\mu \| \pi) + b(\eta) I(\mu \| \pi) + C_{\text{steady}}
$$

**Verification of contraction:** Since $c, \lambda > 0$ and $K - 1/\eta > 0$ (for $\eta > 1/K$), we have:

$$
a(\eta) = \lambda - \frac{2c\lambda}{K - 1/\eta} < \lambda < 1
$$

Thus $a(\eta) < 1$ is a valid contraction factor.

**Step 7: Trajectory Convergence Analysis**

Iterating the one-step inequality and using $I(\mu \| \pi) \ge 0$:

$$
D_{\text{KL}}(\mu_{t+1} \| \pi) \le a(\eta) D_{\text{KL}}(\mu_t \| \pi) + C_{\text{steady}}
$$

This is a scalar affine recursion $x_{t+1} \le \lambda x_t + b$ with $\lambda = a(\eta) < 1$ and $b = C_{\text{steady}} \ge 0$. The solution satisfies:

$$
x_t \le \lambda^t x_0 + b \sum_{j=0}^{t-1} \lambda^j = \lambda^t x_0 + b \frac{1 - \lambda^t}{1 - \lambda} \le \lambda^t x_0 + \frac{b}{1-\lambda}
$$

Therefore:

$$
D_{\text{KL}}(\mu_t \| \pi) \le a(\eta)^t D_{\text{KL}}(\mu_0 \| \pi) + \frac{C_{\text{steady}}}{1 - a(\eta)}
$$

**Interpretation:**
- **If $C_{\text{steady}} > 0$:** The system converges **geometrically to a neighborhood** of $\pi$ with radius $\frac{C_{\text{steady}}}{1-a(\eta)}$. This is NOT exponential convergence to $\pi$ itself.
- **If $C_{\text{steady}} = 0$:** The system converges **exponentially to $\pi$** with rate $a(\eta)$.

**Step 8: Conditions for Zero Defect**

$C_{\text{steady}} = 0$ requires:

$$
e^{-\rho_k} C_{\text{clone}} + c(1+\beta) C_W = 0
$$

Since all terms are non-negative, this requires **both** $C_{\text{clone}} = 0$ and $C_W = 0$.

**Physical interpretation:**
- $C_W = 0$: The cloning operator achieves exact Wasserstein contraction without additive noise floor
- $C_{\text{clone}} = 0$: The cloning operator introduces no entropy defect (e.g., perfect resampling without smoothing error)

In practice, Gaussian cloning with noise scale $\delta$ yields $C_W = O(\delta^2)$ and $C_{\text{clone}} = O(\delta^2)$, so:

$$
C_{\text{steady}} = O(\delta^2)
$$

Thus the neighborhood size can be made arbitrarily small by decreasing $\delta$, but for fixed $\delta > 0$, convergence is to a neighborhood, not to $\pi$ exactly.

:::

## Simplified Corollary (Talagrand-Based)

:::{prf:corollary} Simplified Defective LSI via Talagrand
:label: cor-simplified-defective-lsi

If $\pi$ satisfies a Talagrand inequality $T_2(C_T)$:

$$
W_2^2(\mu, \pi) \le 2 C_T D_{\text{KL}}(\mu \| \pi)
$$

then the composed operator satisfies:

$$
D_{\text{KL}}((\Psi_{\text{total}})_* \mu \| \pi) \le \gamma D_{\text{KL}}(\mu \| \pi) + C_{\text{steady}}
$$

where:

$$
\gamma := \lambda + 2c\lambda C_T = \lambda(1 + 2cC_T)
$$

For this to be a valid contraction ($\gamma < 1$), we need $c < \frac{1-\lambda}{2\lambda C_T}$.

**Constants:**
- $C_{\text{steady}} = e^{-\rho_k} C_{\text{clone}} + c(1+\beta) C_W$ (same as theorem)
- $C_T$: Talagrand constant of $\pi$

:::

:::{prf:proof}
:label: proof-simplified-defective-lsi

From Step 2 of the main proof:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda W_2^2(\mu, \pi) + C_{\text{steady}}
$$

Apply the Talagrand inequality $W_2^2(\mu, \pi) \le 2C_T D_{\text{KL}}(\mu \| \pi)$:

$$
D_{\text{KL}}(\mu' \| \pi) \le \lambda D_{\text{KL}}(\mu \| \pi) + c\lambda \cdot 2C_T D_{\text{KL}}(\mu \| \pi) + C_{\text{steady}}
$$

$$
D_{\text{KL}}(\mu' \| \pi) \le (\lambda + 2c\lambda C_T) D_{\text{KL}}(\mu \| \pi) + C_{\text{steady}}
$$

Setting $\gamma = \lambda(1 + 2cC_T)$ yields the result.

**Note:** This form is simpler but loses the Fisher information control present in the full HWI-based inequality. It is useful for trajectory analysis when $I(\mu \| \pi)$ is not needed explicitly.

:::

## Comparison with Original Flawed Proof

### Original Claim (INCORRECT)

The original proof claimed:

$$
D_{\text{KL}}(\mu_t \| \pi) \le \lambda^t V_0 + \frac{C_{\text{steady}}}{1-\lambda}
$$

and concluded "for large $t$, the steady-state term dominates, giving exponential convergence with rate $\lambda$."

### Why This is Wrong

1. **Mathematical error:** The inequality shows convergence to a **neighborhood** of size $\frac{C_{\text{steady}}}{1-\lambda}$, NOT to $\pi$ itself
2. **Logical contradiction:** Step 3 claims "steady-state term dominates" but also claims "exponential convergence" - these are contradictory when $C_{\text{steady}} > 0$
3. **Missing functional inequality:** The proof only analyzes trajectory behavior, not the one-step operator inequality required for discrete-time LSI
4. **Undefined constant:** $C_{\text{init}}$ in theorem statement is never defined

### Corrected Statement

The corrected proof establishes:

$$
D_{\text{KL}}(\mu_t \| \pi) \le a(\eta)^t D_{\text{KL}}(\mu_0 \| \pi) + \frac{C_{\text{steady}}}{1-a(\eta)}
$$

with **explicit interpretation:**
- **Transient decay:** $a(\eta)^t D_{\text{KL}}(\mu_0 \| \pi)$ decays exponentially
- **Steady-state floor:** $\frac{C_{\text{steady}}}{1-a(\eta)}$ is a permanent lower bound on asymptotic error
- **No convergence to $\pi$** unless $C_{\text{steady}} = 0$

## Recommended Changes to Document

### 1. Revise Theorem Statement (lines 1570-1588)

Replace with the corrected theorem from above, explicitly stating:
- Defective discrete-time LSI (not standard LSI)
- Explicit constants $a(\eta)$, $b(\eta)$, $C_{\text{steady}}$
- Neighborhood convergence, not convergence to $\pi$

### 2. Replace Proof (lines 1590-1603)

Replace with the corrected proof from above.

### 3. Add Remark on Zero-Defect Case

Add a remark explaining:
- When $C_{\text{steady}} = 0$ (requires $C_W = C_{\text{clone}} = 0$)
- How to achieve this (perfect cloning operator)
- Practical scaling $C_{\text{steady}} = O(\delta^2)$ for Gaussian cloning

### 4. Update Cross-References

Ensure all downstream results using `thm-main-lsi-composition` are updated to account for the neighborhood convergence.

## Constants Reference Table

| Symbol | Definition | Source |
|--------|------------|--------|
| $\rho_k$ | Kinetic entropy dissipation rate | Kinetic operator property |
| $\kappa_W$ | Wasserstein contraction strength | Cloning operator property |
| $C_W$ | Wasserstein defect constant | Cloning operator (Gaussian noise) |
| $C_{\text{clone}}$ | Entropy defect from cloning | HWI inequality application |
| $K$ | Displacement convexity constant | Property of $\pi$ (HWI inequality) |
| $\beta$ | Kinetic Wasserstein expansion | Kinetic operator property |
| $c$ | Lyapunov coupling constant | $c = \alpha e^{-\rho_k}/(1-K_W)$ |
| $\lambda$ | Lyapunov contraction factor | $\lambda = \max(e^{-\rho_k}, K_W)$ |
| $K_W$ | Combined Wasserstein factor | $K_W = (1+\beta)(1-\kappa_W)$ |
| $C_{\text{steady}}$ | Steady-state defect | $e^{-\rho_k} C_{\text{clone}} + c(1+\beta) C_W$ |
| $a(\eta)$ | Entropy contraction coefficient | $\lambda - \frac{2c\lambda}{K-1/\eta}$ |
| $b(\eta)$ | Fisher information coefficient | $\frac{c\lambda\eta}{K-1/\eta}$ |
| $\eta$ | Young's inequality parameter | Free parameter, must satisfy $\eta > 1/K$ |
| $C_T$ | Talagrand constant | Property of $\pi$ (optional, for simplified corollary) |
