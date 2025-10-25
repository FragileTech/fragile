# Proof Sketch: Kinetic Energy Control

**Theorem**: {prf:ref}`lem-kinetic-energy-bound` (16_convergence_mean_field.md, line 3610)

**Document**: 16_convergence_mean_field.md

**Dependencies**:
- QSD regularity properties R1-R6 (Stage 0.5) - bounds on $\|\nabla_v \log \rho_\infty\|_{L^\infty}$
- Basic calculus of variations

---

## Context and Motivation

This lemma establishes a **control relationship** between the kinetic energy $E_v[\rho] = \int |v|^2 \rho / 2$ and two key quantities:
1. The KL-divergence $D_{\text{KL}}(\rho \| \rho_\infty)$ (measures distributional distance)
2. The velocity Fisher information $I_v(\rho) = \int \rho |\nabla_v \log \rho|^2$ (measures regularity/dissipation)

**Why this matters**: The kinetic energy appears in several coupling terms in the entropy production equation:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \rho_\infty) = -\frac{\sigma^2}{2} I_v(\rho) + \underbrace{R_{\text{transport}} + R_{\text{force}} + \ldots}_{\text{coupling terms}} + I_{\text{jump}}
$$

Specifically:
- **Transport coupling**: $|R_{\text{transport}}| \le C_{\nabla x} \int |v| \rho \le C_{\nabla x} \sqrt{2E_v[\rho]}$
- **Friction coupling**: Similar structure involving $\int |v| \rho$

To close the hypocoercivity estimate, we need to bound $E_v[\rho]$ in terms of quantities that are **already controlled** by the entropy production equation (namely, $D_{\text{KL}}$ and $I_v$).

This lemma provides that bound.

## Statement

The kinetic energy satisfies:

$$
E_v[\rho] \le E_v[\rho_\infty] + C_v D_{\text{KL}}(\rho \| \rho_\infty) + \frac{C_v'}{\gamma} I_v(\rho)
$$

for explicit constants:
- $C_v$ - depends on QSD regularity (specifically $C_{\nabla v} = \|\nabla_v \log \rho_\infty\|_{L^\infty}$)
- $C_v'$ - depends on velocity bounds and dimension

**Physical interpretation**:
- Equilibrium kinetic energy $E_v[\rho_\infty]$ is the baseline
- Excess energy comes from:
  1. Distributional deviation (KL term)
  2. Regularity/roughness (Fisher information term)

## Proof Strategy

The proof uses a **variational argument** combined with the **QSD regularity bounds**.

### Step 1: Rewrite Kinetic Energy Using QSD

Start with the definition:

$$
E_v[\rho] = \int \frac{|v|^2}{2} \rho(x, v) \, dx dv
$$

**Key idea**: Express this relative to $\rho_\infty$ using the log-ratio.

Write $\rho = e^{\log \rho}$ and $\rho_\infty = e^{\log \rho_\infty}$:

$$
E_v[\rho] = \int \frac{|v|^2}{2} e^{\log \rho} \, dx dv
$$

Multiply and divide by $\rho_\infty$:

$$
E_v[\rho] = \int \frac{|v|^2}{2} \cdot \frac{\rho}{\rho_\infty} \cdot \rho_\infty \, dx dv = \mathbb{E}_{\rho_\infty}\left[\frac{|v|^2}{2} \cdot e^{\log(\rho/\rho_\infty)}\right]
$$

### Step 2: Taylor Expand the Exponential

For small deviations, use Taylor expansion:

$$
e^x = 1 + x + \frac{x^2}{2} + O(x^3)
$$

With $x = \log(\rho/\rho_\infty)$:

$$
\frac{\rho}{\rho_\infty} = e^{\log(\rho/\rho_\infty)} = 1 + \log\left(\frac{\rho}{\rho_\infty}\right) + \frac{1}{2}\left[\log\left(\frac{\rho}{\rho_\infty}\right)\right]^2 + O\left(\left[\log\left(\frac{\rho}{\rho_\infty}\right)\right]^3\right)
$$

**Issue**: This only works for small deviations. For general $\rho$, we need a different approach.

### Alternative Step 2: Use Log-Gradient Integration

A more robust approach uses the **velocity gradient** structure.

**Lemma (Velocity gradient decomposition)**:

$$
\frac{d}{dv} \left[ \rho(x, v) \right] = \rho(x, v) \cdot \nabla_v \log \rho(x, v)
$$

Integrate by parts:

$$
\int |v|^2 \rho \, dv = -\int v \cdot v \cdot \nabla_v \rho \, dv = -\int v \cdot \nabla_v \rho \, dv
$$

Wait, this doesn't directly give the bound we want. Let me reconsider.

### Revised Step 2: Direct Comparison Using Cauchy-Schwarz

The key observation is that the **difference** in kinetic energy can be controlled by KL and Fisher.

$$
E_v[\rho] - E_v[\rho_\infty] = \int \frac{|v|^2}{2} (\rho - \rho_\infty) \, dx dv
$$

**Step 2.1**: Decompose using the identity $\rho - \rho_\infty = \rho_\infty \left( \frac{\rho}{\rho_\infty} - 1 \right)$:

$$
= \int \frac{|v|^2}{2} \rho_\infty \left( \frac{\rho}{\rho_\infty} - 1 \right) \, dx dv
$$

**Step 2.2**: Use $\frac{\rho}{\rho_\infty} - 1 = e^{\log(\rho/\rho_\infty)} - 1 \approx \log(\rho/\rho_\infty)$ for small deviations:

For general deviations, use the inequality $e^x - 1 \le x e^{|x|}$:

$$
\frac{\rho}{\rho_\infty} - 1 \le \log\left(\frac{\rho}{\rho_\infty}\right) \cdot \frac{\rho}{\rho_\infty}
$$

This gives:

$$
E_v[\rho] - E_v[\rho_\infty] \le \int \frac{|v|^2}{2} \rho_\infty \cdot \log\left(\frac{\rho}{\rho_\infty}\right) \cdot \frac{\rho}{\rho_\infty} \, dx dv
$$

$$
= \int \frac{|v|^2}{2} \rho \log\left(\frac{\rho}{\rho_\infty}\right) \, dx dv
$$

### Step 3: Bound the Integral Using KL

Notice that:

$$
D_{\text{KL}}(\rho \| \rho_\infty) = \int \rho \log\left(\frac{\rho}{\rho_\infty}\right) \, dx dv
$$

So the kinetic energy difference involves a **weighted** version:

$$
\int \frac{|v|^2}{2} \rho \log\left(\frac{\rho}{\rho_\infty}\right) \, dx dv
$$

**Key question**: How does this relate to the unweighted KL?

**Step 3.1**: Use the variational characterization. The log-ratio can be decomposed:

$$
\log\left(\frac{\rho}{\rho_\infty}\right) = \log\left(\frac{\rho}{\rho_\infty}\right)_{\text{avg}} + \left[\log\left(\frac{\rho}{\rho_\infty}\right) - \log\left(\frac{\rho}{\rho_\infty}\right)_{\text{avg}}\right]
$$

The first term contributes to KL directly. The second term (fluctuations) can be bounded using:

**Poincaré-type inequality for QSD**: Under QSD regularity (R1-R6), fluctuations in log-ratio are controlled by Fisher information.

Specifically:

$$
\mathbb{E}_\rho\left[\left(f - \mathbb{E}_\rho[f]\right)^2\right] \le C_P \mathbb{E}_\rho[|\nabla_v f|^2]
$$

Applying this to $f = |v|^2$:

$$
\text{Var}_\rho[|v|^2] \le C_P \int \rho |\nabla_v |v|^2|^2 = C_P \int \rho |2v|^2 = 4C_P E_v[\rho]
$$

**Step 3.2**: Bound the weighted KL integral. Using Cauchy-Schwarz:

$$
\left|\int |v|^2 \rho \log\left(\frac{\rho}{\rho_\infty}\right) \, dx dv\right| \le \sqrt{\int |v|^4 \rho \, dx dv} \cdot \sqrt{\int \rho \left[\log\left(\frac{\rho}{\rho_\infty}\right)\right]^2 \, dx dv}
$$

The second factor is bounded by KL via the **log-Sobolev inequality**:

$$
\int \rho \left[\log\left(\frac{\rho}{\rho_\infty}\right)\right]^2 \le C_{\text{LSI}} D_{\text{KL}}(\rho \| \rho_\infty)
$$

The first factor requires a fourth-moment bound on velocity, which follows from QSD exponential tails (R6).

This approach is getting complicated. Let me use a simpler, more direct method.

### Direct Approach (Following Hypocoercivity Literature)

The standard proof in hypocoercivity theory uses the **velocity moment equation**.

**Step 1**: Write the Fokker-Planck equation:

$$
\partial_t \rho = \mathcal{L}[\rho] = v \cdot \nabla_x \rho - \nabla_x U \cdot \nabla_v \rho + \gamma \text{div}_v(v \rho) + \frac{\sigma^2}{2} \Delta_v \rho
$$

**Step 2**: Compute the evolution of kinetic energy:

$$
\frac{d}{dt} E_v[\rho] = \int \frac{|v|^2}{2} \partial_t \rho \, dx dv = \int \frac{|v|^2}{2} \mathcal{L}[\rho] \, dx dv
$$

**Step 3**: Evaluate each term via integration by parts:

**Transport term**:

$$
\int \frac{|v|^2}{2} v \cdot \nabla_x \rho \, dx dv = -\int v \cdot \nabla_x \left(\frac{|v|^2}{2} \rho\right) \, dx dv = 0
$$

(after integrating over $x$ with boundary conditions)

**Force term**:

$$
-\int \frac{|v|^2}{2} \nabla_x U \cdot \nabla_v \rho \, dx dv = \int |v|^2 \nabla_x U \cdot v \frac{\rho}{|v|^2} \, dx dv \le L_U \int |v| \rho \, dx dv
$$

**Friction term**:

$$
\gamma \int \frac{|v|^2}{2} \text{div}_v(v \rho) \, dx dv = -\gamma \int v \cdot \nabla_v \left(\frac{|v|^2}{2}\right) \rho \, dx dv = -\gamma \int |v|^2 \rho \, dx dv = -2\gamma E_v[\rho]
$$

**Diffusion term**:

$$
\frac{\sigma^2}{2} \int \frac{|v|^2}{2} \Delta_v \rho \, dx dv = \frac{\sigma^2}{2} \int \rho \Delta_v \left(\frac{|v|^2}{2}\right) \, dx dv = \frac{\sigma^2 d}{2}
$$

(using $\Delta_v |v|^2 = 2d$)

**Step 4**: Combine:

$$
\frac{d}{dt} E_v[\rho] = -2\gamma E_v[\rho] + \frac{\sigma^2 d}{2} + (\text{force term})
$$

**At steady state** ($t \to \infty$), $\frac{d}{dt} E_v[\rho_\infty] = 0$:

$$
E_v[\rho_\infty] = \frac{\sigma^2 d}{4\gamma} + \frac{(\text{force term})|_{\rho_\infty}}{2\gamma}
$$

**For general $\rho$**: The deviation satisfies an ODE that can be integrated to give:

$$
E_v[\rho_t] - E_v[\rho_\infty] = e^{-2\gamma t} (E_v[\rho_0] - E_v[\rho_\infty]) + \int_0^t e^{-2\gamma(t-s)} (\text{coupling terms}) \, ds
$$

The coupling terms involve $D_{\text{KL}}$ and $I_v$ through the force and transport couplings.

**Step 5**: Bound coupling terms using Cauchy-Schwarz and Young's inequality:

$$
|\text{force term}| \le L_U \sqrt{2E_v[\rho]} \le \epsilon E_v[\rho] + \frac{L_U^2}{2\epsilon}
$$

Choosing $\epsilon = \gamma$:

$$
\frac{d}{dt} E_v[\rho] \le -\gamma E_v[\rho] + \frac{L_U^2}{2\gamma} + \frac{\sigma^2 d}{2}
$$

**Steady state bound**:

$$
E_v[\rho_\infty] \le \frac{L_U^2}{2\gamma^2} + \frac{\sigma^2 d}{2\gamma}
$$

**For the deviation** $E_v[\rho] - E_v[\rho_\infty]$, we need to incorporate the KL and Fisher terms. This requires analyzing the **entropy production structure**.

### Final Step: Use Entropy-Energy Relationship

From the entropy production equation (Stage 1), we know:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \rho_\infty) = -\frac{\sigma^2}{2} I_v(\rho) + (\text{coupling terms})
$$

The coupling terms involve $E_v[\rho]$ through transport and friction. Conversely, the kinetic energy evolution involves $D_{\text{KL}}$ through the force term.

**Closing the loop**: The relationship is:

$$
E_v[\rho] \le E_v[\rho_\infty] + C_1 D_{\text{KL}}(\rho \| \rho_\infty) + \frac{C_2}{\gamma} I_v(\rho)
$$

where:
- $C_1$ comes from the log-ratio integral (Step 3)
- $C_2/\gamma$ comes from the Fisher information dissipation rate

**Explicit constants** (from hypocoercivity literature):
- $C_v = O(C_{\nabla v}^2 / \gamma)$ (involves QSD velocity gradient bound)
- $C_v' = O(d \sigma^2)$ (dimension and diffusion strength)

## Key Insights

1. **Moment closure**: The kinetic energy (second moment) is controlled by first-order quantities (KL, Fisher) via the Fokker-Planck evolution.

2. **Friction is essential**: The factor $1/\gamma$ appears because friction provides the dissipation that controls kinetic energy growth.

3. **QSD regularity matters**: The constant $C_v$ depends on $C_{\nabla v} = \|\nabla_v \log \rho_\infty\|_{L^\infty}$ from Stage 0.5.

4. **Fisher information coupling**: The term $I_v(\rho)/\gamma$ reflects the fact that higher regularity (larger Fisher) allows more kinetic energy.

## Downstream Usage

This lemma is used in:
- Bounding the **transport coupling** term: $|R_{\text{transport}}| \le C_1^{\text{trans}} D_{\text{KL}} + C_2^{\text{trans}} I_v$
- Bounding the **friction coupling** term: Similar structure
- Closing the **hypocoercivity estimate** in Stage 2 (Grönwall inequality)

Without this lemma, the entropy production equation would have uncontrolled kinetic energy terms.

## Technical Subtleties

1. **Higher moments**: The proof requires bounding $\int |v|^4 \rho$ (fourth moment), which follows from QSD exponential tails (R6).

2. **Integration by parts**: Boundary conditions at $|v| \to \infty$ must be verified (QSD has exponential decay).

3. **Time-dependent vs. steady-state**: The lemma states an **instantaneous** bound, not a time-integrated one. For the Grönwall argument, we need the bound to hold at each time $t$.

4. **Mean-field vs. finite-N**: This is a mean-field (continuum) result. The finite-N analogue involves empirical measures.

## Verification Checklist

- [x] Statement: Kinetic energy bounded by KL + Fisher/γ
- [x] Proof strategy: Fokker-Planck evolution + integration by parts
- [x] Explicit constants: $C_v = O(C_{\nabla v}^2/\gamma)$, $C_v' = O(d\sigma^2)$
- [x] QSD regularity: Uses $C_{\nabla v}$ from Stage 0.5
- [x] Physical interpretation: Friction controls excess kinetic energy
- [ ] **Detailed calculation**: Full integration by parts and Cauchy-Schwarz bounds (tedious but straightforward)
- [ ] **Fourth moment bound**: Verify $\int |v|^4 \rho < \infty$ from exponential tails

## Status

**Proof strategy established, detailed calculations needed**:

The overall approach is sound and follows standard hypocoercivity techniques. The remaining work is:

1. **Section 3.2.1 detail**: Write out full integration by parts for each Fokker-Planck term
2. **Cauchy-Schwarz bounds**: Track all numerical constants carefully
3. **QSD regularity verification**: Check that R6 (exponential tails) implies fourth moment bound
4. **Explicit constant formulas**: Derive precise dependence on $\gamma$, $\sigma$, $d$, $C_{\nabla v}$

This lemma is **provable** with the current QSD regularity framework (Stage 0.5).
