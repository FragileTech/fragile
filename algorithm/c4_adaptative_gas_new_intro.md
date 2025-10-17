# $C^4$ Regularity and Higher-Order Stability Analysis of the ρ-Localized Adaptive Gas

## 0. TLDR

**C⁴ Regularity with O(ρ⁻³) Scaling (Simplified Model)**: For a **simplified fitness model** where the measurement function depends only on walker position (not companion selection), the fitness potential $V_{\text{fit}}[f_k, \rho](x_i)$ is four times continuously differentiable with k-uniform and N-uniform bounds. Remarkably, the fourth derivative has the **same ρ-scaling** as the third derivative ($O(\rho^{-3})$), not the naive $O(\rho^{-4})$, due to the centered moment structure of localized statistics. This establishes Hessian Lipschitz continuity with constant $K_{V,3}(\rho) = O(\rho^{-3})$.

**Telescoping Identities Ensure k-Uniformity**: The normalized localization weights satisfy $\sum_{j \in A_k} \nabla^4 w_{ij}(\rho) = 0$ identically for all $x_i$. This fourth-order telescoping identity is the cornerstone of k-uniform bounds: when computing fourth derivatives of localized moments, the leading-order term vanishes, preventing linear growth in the walker count $k$. This extends the C³ telescoping mechanism to fourth order and ensures all bounds are independent of swarm size $N$.

**Higher-Order Functional Inequalities (Conditional on Convexity)**: C⁴ regularity is a necessary prerequisite for advanced functional inequalities, though additional geometric conditions are required. When the fitness potential satisfies **uniform convexity** ($\nabla^2 V_{\text{fit}} \geq \lambda_\rho I$), C⁴ smoothness enables Brascamp-Lieb inequalities for sharp concentration bounds (see {prf:ref}`cor-brascamp-lieb`) and validates Bakry-Émery Γ₂-calculus for hypercontractivity (see {prf:ref}`prop-bakry-emery-gamma2`). These functional inequalities provide optimal constants for convergence to the quasi-stationary distribution (QSD) and sharper entropy production estimates beyond the Log-Sobolev framework.

**Time-Step Constraint Remains O(ρ³)**: C⁴ regularity validates higher-order splitting schemes (e.g., BABAB achieving $O(\Delta t^4)$ weak error), but does **not** relax the time-step stability constraint. Both BAOAB (2nd order) and fourth-order integrators have the same constraint: $\Delta t \lesssim \rho^3$, determined by $\Delta t \le \rho^{3/2}/\sqrt{K_{V,3}(\rho)} \sim \rho^{3/2}/\rho^{-3/2} = \rho^3$. C⁴ regularity improves only the **error constant** (multiplicative factor), not the **scaling** with $\rho$.

## 1. Introduction

### 1.1. Goal and Scope

The goal of this document is to establish **C⁴ regularity** (four times continuous differentiability) of the ρ-localized fitness potential in the Adaptive Gas framework. The central object of study is the fitness potential:

$$
V_{\text{fit}}[f_k, \rho](x_i) = g_A(Z_\rho[f_k, d, x_i])
$$

where $Z_\rho$ is the regularized Z-score measuring a walker's standardized diversity relative to its ρ-local neighborhood, and $g_A: \mathbb{R} \to [0, A]$ is the smooth rescale function.

This analysis extends the C³ regularity established in [13_adaptative_gas_c3_regularity.md](13_adaptative_gas_c3_regularity.md), which proved that $V_{\text{fit}} \in C^3$ with k-uniform and N-uniform third-derivative bounds $\|\nabla^3 V_{\text{fit}}\| \le K_{V,3}(\rho) = O(\rho^{-3})$. The main result of the present document is:

**Main Theorem** ({prf:ref}`thm-c4-regularity`): Under C⁴ regularity assumptions on the measurement function $d$, localization kernel $K_\rho$, rescale function $g_A$, and regularized standard deviation $\sigma'_{\text{reg}}$, the fitness potential is C⁴ with:

$$
\|\nabla^4_{x_i} V_{\text{fit}}[f_k, \rho](x_i)\| \le K_{V,4}(\rho) = O(\rho^{-3}) < \infty
$$

for all alive walker counts $k \in \{1, \ldots, N\}$ and all swarm sizes $N \ge 1$.

:::{warning} Scope Limitation: Simplified Fitness Model
:label: warn-scope-simplified-model

This document analyzes a **simplified fitness potential model** where the measurement function $d: \mathcal{X} \to \mathbb{R}$ depends only on a walker's position $x_i$.

In the full Adaptive Gas framework (see [11_adaptative_gas.md](11_adaptative_gas.md)), the diversity measurement $d_i = d_{\text{alg}}(i, c(i))$ depends on the **entire swarm state** through companion selection $c(i)$. Extending the C⁴ analysis to the full swarm-dependent measurement is deferred to future work, as it requires:

1. Additional combinatorial arguments for companion selection derivatives
2. Analysis of how companion reassignment couples walker derivatives
3. Verification that the telescoping mechanism survives these couplings

**Implication:** The C⁴ regularity result proven here applies to position-dependent fitness models but not yet to the full algorithmic distance-based diversity measurement. The simplified model is still of significant theoretical and practical interest, as it captures the core localization mechanism and validates the mathematical foundations for the general case.
:::

**Relationship to framework documents**: This document builds on:
- **C³ regularity** ([13_adaptative_gas_c3_regularity.md](13_adaptative_gas_c3_regularity.md)): Establishes the telescoping mechanism and centered moment structure for third derivatives
- **Adaptive Gas specification** ([11_adaptative_gas.md](11_adaptative_gas.md)): Defines the fitness pipeline and localization scheme
- **Framework axioms** ([01_fragile_gas_framework.md](../docs/source/01_fragile_gas_framework.md)): Provides the foundational regularity conditions

### 1.2. Why C⁴ Regularity Matters: Advanced Stability and Functional Inequalities

The C³ regularity analysis was sufficient to validate the BAOAB Langevin integrator, which requires $V \in C^3$ for its $O(\Delta t^2)$ weak error guarantee. Why, then, invest effort in establishing C⁴ regularity? The answer lies in **higher-order functional analysis** and **optimal convergence theory**.

**Hessian Lipschitz Continuity and Optimization**: C⁴ regularity implies that the Hessian $\nabla^2 V_{\text{fit}}$ is Lipschitz continuous with constant $L_{\text{Hess}} \le K_{V,3}(\rho)$. This property is fundamental for:

- **Newton-Raphson convergence**: Guarantees local quadratic convergence of second-order optimization methods
- **Cubic regularization**: Validates trust-region algorithms with optimal $O(\epsilon^{-3/2})$ complexity for finding $\epsilon$-critical points
- **Gradient flow regularity**: Ensures smooth trajectories in the continuous-time limit of gradient descent

**Brascamp-Lieb and Log-Sobolev Inequalities (Conditional Results)**: The Brascamp-Lieb inequality provides optimal variance bounds for probability measures. For the QSD $\pi_{\text{QSD}} \propto \exp(-\beta V_{\text{fit}})$, establishing this inequality requires:

1. **C⁴ regularity of the potential** (this document's contribution)
2. **Uniform convexity**: $\nabla^2 V_{\text{fit}}(x) \geq \lambda_\rho I$ for all $x \in \mathcal{X}$ and some $\lambda_\rho > 0$

When both conditions hold (see {prf:ref}`cor-brascamp-lieb`), the resulting inequality:

$$
\text{Var}_{\pi_{\text{QSD}}}[f] \le \frac{1}{\lambda_{\min}(\nabla^2 V_{\text{fit}})} \|\nabla f\|^2_{L^2(\pi_{\text{QSD}})}
$$

provides **sharp concentration bounds** with optimal constants. The C⁴ regularity ensures that the Hessian eigenvalues are well-defined and Lipschitz continuous, which is essential for the inequality's proof. This strengthens the Log-Sobolev inequalities used in the mean-field convergence analysis (see [11_mean_field_convergence](../docs/source/11_mean_field_convergence/)).

**Bakry-Émery Γ₂-Calculus (Conditional Results)**: The Bakry-Émery formalism uses the Γ₂ operator to characterize **hypercontractivity** and **entropy dissipation** for Markov semigroups. The Γ₂ operator involves second derivatives of the generator $\mathcal{L} = \Delta - \nabla V_{\text{fit}} \cdot \nabla$, which in turn require fourth derivatives of the potential. C⁴ regularity ensures Γ₂ is well-defined.

However, to conclude hypercontractivity, one must also verify the **Bakry-Émery curvature condition** (see {prf:ref}`prop-bakry-emery-gamma2`):

$$
\Gamma_2(f, f) \geq \lambda_{\text{BE}} \Gamma(f, f) \quad \text{for some } \lambda_{\text{BE}} > 0
$$

which is equivalent to uniform convexity of the potential for the Langevin generator. When this condition holds, C⁴ regularity enables:

- Sharp exponential convergence rates for the QSD
- Optimal Sobolev and Poincaré constants
- Refined entropy production estimates beyond the LSI framework

**Higher-Order Integrators (Conditional)**: Higher-order splitting schemes like BABAB (4th order) or Runge-Kutta methods can achieve $O(\Delta t^4)$ weak error when the potential has sufficient smoothness (typically $V \in C^5$ or $C^6$). While C⁴ regularity is one step toward validating these integrators, it does not yet provide the full regularity needed. More importantly, the **time-step constraint** does not improve with C⁴ regularity (see below).

:::{important}
**Practical Impact on Time-Step Selection**: C⁴ regularity does **NOT** relax the time-step constraint for the existing BAOAB integrator. The stability bound is determined by the **third** derivative, not the fourth:

$$
\Delta t \le \frac{\rho^{3/2}}{\sqrt{K_{V,3}(\rho)}} \sim \frac{\rho^{3/2}}{\sqrt{\rho^{-3}}} = \rho^{3/2} \cdot \rho^{3/2} = \rho^3
$$

Both C³ and C⁴ analyses yield $K_{V,3}(\rho) = K_{V,4}(\rho) = O(\rho^{-3})$ (same scaling), so the time-step bound $\Delta t \lesssim \rho^3$ applies to both 2nd and 4th order methods. The fourth derivative affects only the **error constant** (multiplicative factor in the $O(\Delta t^4)$ weak error), not the **scaling** with $\rho$.

See {prf:ref}`prop-timestep-c4` and {prf:ref}`cor-baoab-validity` for detailed derivations.
:::

### 1.3. Overview of the Proof Strategy and Document Structure

The proof of C⁴ regularity follows the same structural approach as the C³ analysis, extended to fourth order with additional combinatorial complexity from the Faà di Bruno formula. The strategy is to differentiate the fitness pipeline step-by-step, carefully tracking how derivatives propagate through compositions and quotients.

**The Five-Step Fitness Pipeline**:

1. **Localization weights** $w_{ij}(\rho) = K_\rho(x_i, x_j) / \sum_{j'} K_\rho(x_i, x_{j'})$ (Gaussian kernel quotient)
2. **Localized mean** $\mu_\rho[f_k, d, x_i] = \sum_j w_{ij}(\rho) d(x_j)$
3. **Localized variance** $\sigma^2_\rho[f_k, d, x_i] = \sum_j w_{ij}(\rho) [d(x_j) - \mu_\rho]^2$
4. **Regularized Z-score** $Z_\rho = (d(x_i) - \mu_\rho) / \sigma'_{\text{reg}}(\sigma^2_\rho)$
5. **Fitness potential** $V_{\text{fit}} = g_A(Z_\rho)$

Each step requires careful application of differentiation rules (product, quotient, chain) to fourth order, using:

- **Faà di Bruno formula** for fourth-order chain rule
- **Leibniz formula** for fourth-order product rule
- **Quotient rule extension** to fourth order (12 terms)
- **Telescoping identities** to ensure k-uniformity

**Key Structural Insight**: The **centered moment construction** is crucial. When computing $\nabla^4 \mu_\rho = \nabla^4(\sum_j w_{ij} d(x_j))$, the leading term $\sum_j (\nabla^4 w_{ij}) d(x_j)$ would naively grow with $k$. However, by writing:

$$
\sum_j (\nabla^4 w_{ij}) d(x_j) = \sum_j (\nabla^4 w_{ij}) [d(x_j) - d(x_i)] + \underbrace{\left(\sum_j \nabla^4 w_{ij}\right)}_{=0 \text{ (telescoping)}} d(x_i)
$$

the constant term vanishes due to normalization. The remaining centered sum $\sum_j (\nabla^4 w_{ij}) [d(x_j) - d(x_i)]$ involves differences that scale with $\|x_j - x_i\|$, introducing a factor of $O(\rho)$ from Gaussian localization. This gives:

$$
C_{\mu,\nabla^4}(\rho) = O(\rho^{-4}) \cdot O(\rho) = O(\rho^{-3})
$$

instead of the naive $O(\rho^{-4})$. This "one-order improvement" propagates through the entire pipeline, ensuring that $K_{V,4}(\rho) = O(\rho^{-3})$ (same scaling as $K_{V,3}(\rho)$).

The diagram below illustrates the logical flow of the proof. We systematically establish fourth-derivative bounds for each stage of the fitness pipeline (Sections 4-7), culminating in the main C⁴ regularity theorem (Section 8). Sections 9-11 analyze the implications for stability, functional inequalities, and ρ-scaling.

```mermaid
graph TD
    subgraph "Part I: Foundations & Assumptions (Sec 2-3)"
        A["<b>Sec 2: Mathematical Framework</b><br>Fitness pipeline, differential operators,<br>Faà di Bruno formula"]:::stateStyle
        B["<b>Sec 3: C⁴ Regularity Assumptions</b><br>C⁴ measurement d(x_i) for <b>simplified model</b>,<br>Gaussian kernel K_ρ, rescale g_A, regularized sqrt σ'_reg"]:::axiomStyle
    end

    subgraph "Part II: Fourth-Derivative Bounds for Pipeline Components (Sec 4-7)"
        C["<b>Sec 4: Localization Weights</b><br>Fourth derivative ∇⁴w_ij = O(ρ⁻⁴)<br><b>Telescoping: Σ_j ∇⁴w_ij = 0</b>"]:::lemmaStyle
        D["<b>Sec 5: Localized Moments</b><br>∇⁴μ_ρ and ∇⁴σ²_ρ via telescoping<br><b>Result: O(ρ⁻³) not O(ρ⁻⁴)</b>"]:::lemmaStyle
        E["<b>Sec 6: Regularized Std Dev</b><br>Chain rule for σ'_reg(σ²_ρ)<br>Fourth derivative bound"]:::lemmaStyle
        F["<b>Sec 7: Z-Score</b><br>Fourth-order quotient rule (12 terms)<br><b>K_Z,4(ρ) = O(ρ⁻³)</b>"]:::lemmaStyle
    end

    subgraph "Part III: Main Result & Implications (Sec 8-11)"
        G["<b>Sec 8: Main C⁴ Theorem</b><br>V_fit = g_A(Z_ρ) via Faà di Bruno<br><b>K_V,4(ρ) = O(ρ⁻³)</b>"]:::theoremStyle
        H["<b>Sec 9: Stability Corollaries</b><br>Hessian Lipschitz, Brascamp-Lieb <b>(conditional)</b>,<br>Bakry-Émery Γ₂ <b>(conditional)</b>, integrators"]:::theoremStyle
        I["<b>Sec 10: ρ-Scaling Analysis</b><br>Time-step constraint Δt ≲ ρ³<br>Numerical recommendations"]:::theoremStyle
        J["<b>Sec 11: Comparison with C³</b><br>Regularity hierarchy, when C⁴ needed,<br>practical guidance"]:::stateStyle
    end

    subgraph "Part IV: Conclusion (Sec 12)"
        K["<b>Sec 12: Conclusion</b><br>Summary, open questions,<br>C^∞ conjecture"]:::externalStyle
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K

    C -.."Telescoping ensures k-uniformity for"..- D
    D -.."Centered moments give O(ρ⁻³) to"..- F
    F -.."Dominant term in Z-score fourth derivative"..- G
    G -.."Enables functional inequalities <b>(with convexity)</b> in"..- H

    classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
    classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
    classDef externalStyle fill:#5f4a8c,stroke:#a48fd4,stroke-width:2px,stroke-dasharray: 3 3,color:#f0e8f6
```

**Document Roadmap**:

- **Sections 2-3**: Establish the mathematical framework, notation, and regularity assumptions for all components (simplified model)
- **Section 4**: Prove fourth-derivative bounds for localization weights and the crucial telescoping identity
- **Section 5**: Derive fourth derivatives of localized mean and variance using the centered moment technique
- **Section 6**: Apply the chain rule to the regularized standard deviation composition
- **Section 7**: Compute the fourth derivative of the Z-score using the extended quotient rule
- **Section 8**: Compose all results via Faà di Bruno to prove the main C⁴ regularity theorem
- **Section 9**: Analyze implications for Hessian Lipschitz continuity, functional inequalities (conditional on convexity), and integrators
- **Section 10**: Study ρ-scaling, time-step constraints ($\Delta t \lesssim \rho^3$), and numerical recommendations
- **Section 11**: Compare with C³ analysis and provide practical guidance on when C⁴ is beneficial
- **Section 12**: Conclude with open questions, the C^∞ regularity conjecture, and future work on the full swarm-dependent model
