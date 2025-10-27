# Proof Sketch: Drift Condition with Quadratic Lyapunov

**Theorem Label**: `lem-drift-condition-corrected`

**Source Document**: `/home/guillem/fragile/docs/source/2_geometric_gas/16_convergence_mean_field.md` (lines 2087-2236)

**Date**: 2025-10-25

**Sketcher**: Claude Code (Proof Sketcher Agent)

**Review Status**: Single-strategist (GPT-5/Codex only; Gemini MCP unavailable)

---

## 1. Theorem Statement

:::{prf:lemma} Drift Condition with Quadratic Lyapunov
:label: lem-drift-condition-corrected

Under Assumptions A1 (confinement) and A3 (friction), there exist constants $a, b, c > 0$ such that the quadratic Lyapunov function:

$$
V(x,v) = a|x|^2 + 2b x \cdot v + c|v|^2
$$

satisfies a drift condition with respect to the **adjoint operator** $\mathcal{L}^*$:

$$
\mathcal{L}^*[V] \le -\beta V + C
$$

for some $\beta > 0$ and $C < \infty$.
:::

---

## 2. Context and Dependencies

### 2.1. Framework Assumptions

**Assumption A1 (Confinement)**:
- The potential $U: \mathcal{X} \to \mathbb{R}$ satisfies:
  - $U(x) \to +\infty$ as $x \to \partial \mathcal{X}$ or $|x| \to \infty$
  - $\nabla^2 U(x) \ge \kappa_{\text{conf}} I_d$ for some $\kappa_{\text{conf}} > 0$ (strong convexity)

**Assumption A3 (Bounded parameters)**:
- Friction coefficient: $\gamma > 0$
- Diffusion coefficient: $\sigma^2 > 0$
- Revival rate: $0 < \lambda_{\text{revive}} < \infty$

### 2.2. Kinetic Operator

The kinetic SDE is:

$$
dx_t = v_t \, dt, \quad dv_t = -\nabla_x U(x_t) \, dt - \gamma v_t \, dt + \sigma \, dW_t
$$

The **adjoint operator** (SDE generator) is:

$$
\mathcal{L}^* = v \cdot \nabla_x - \nabla_x U(x) \cdot \nabla_v - \gamma v \cdot \nabla_v + \frac{\sigma^2}{2} \Delta_v
$$

### 2.3. Motivation

The simple Lyapunov function $V = |x|^2 + |v|^2$ does **not** satisfy a drift condition due to the cross-term $x \cdot v$ arising from the transport term $v \cdot \nabla_x$. A properly designed **quadratic form** is needed to absorb this coupling and achieve coercivity.

---

## 3. Proof Strategy

The proof establishes the drift condition through a **five-step parameter optimization**:

### Step 1: Compute $\mathcal{L}^*[V]$ term-by-term

Apply the adjoint operator to the quadratic form $V(x,v) = a|x|^2 + 2b x \cdot v + c|v|^2$:

1. **Transport term**: $v \cdot \nabla_x V = 2av \cdot x + 2b|v|^2$
2. **Force term**: $-\nabla_x U \cdot \nabla_v V = -2b \nabla_x U \cdot x - 2c \nabla_x U \cdot v$
3. **Friction term**: $-\gamma v \cdot \nabla_v V = -2\gamma b v \cdot x - 2\gamma c |v|^2$
4. **Diffusion term**: $\frac{\sigma^2}{2} \Delta_v V = \sigma^2 c d$

Combine to obtain:

$$
\mathcal{L}^*[V] = 2(a - \gamma b) v \cdot x - 2b \nabla_x U \cdot x - 2c \nabla_x U \cdot v + (2b - 2\gamma c)|v|^2 + \sigma^2 c d
$$

### Step 2: Apply strong convexity bounds

Use the strong convexity of $U$ from Assumption A1:

$$
\nabla_x U \cdot x \ge \kappa_{\text{conf}} |x|^2 - C_1
$$

For the mixed term, use Cauchy-Schwarz:

$$
|\nabla_x U \cdot v| \le |\nabla_x U| |v| \le (\kappa_{\text{conf}}|x| + C_2)|v|
$$

where we've used that $|\nabla_x U| \le \kappa_{\text{conf}}|x| + C_2$ (linear growth from strong convexity).

### Step 3: Handle cross-terms with Young's inequality

Apply Young's inequality with parameters $\delta_1, \delta_2 > 0$:

$$
|v \cdot x| \le \frac{|v|^2}{2\delta_1} + \frac{\delta_1|x|^2}{2}, \quad |x||v| \le \frac{|v|^2}{2\delta_2} + \frac{\delta_2|x|^2}{2}
$$

Substitute into the expression from Step 1-2 to collect coefficients of $|x|^2$ and $|v|^2$.

### Step 4: Choose parameters $(a, b, c, \delta_1, \delta_2)$ to maximize negative drift

The key insight is to:
- Normalize $c = 1$ (WLOG)
- Choose $a, b$ small enough that the matrix $M = \begin{pmatrix} a & b \\ b & 1 \end{pmatrix}$ is positive definite
- Optimize $\delta_1, \delta_2$ to ensure both $|x|^2$ and $|v|^2$ coefficients are negative

The source document attempts the choice:
- $b = \varepsilon$ (small parameter)
- $a = \kappa_{\text{conf}}\varepsilon$
- $\delta_1, \delta_2$ chosen to balance terms

**Critical issue**: The calculation in the source document lines 2194-2219 shows that naive parameter choices lead to a **friction condition** $\gamma > \frac{4\kappa_{\text{conf}}}{9}$, which may not always hold.

### Step 5: Relate negative drift to $V$

Once negative coefficients $\beta_x, \beta_v > 0$ are established for $|x|^2$ and $|v|^2$ respectively, use the spectral bounds of the matrix $M$:

$$
\lambda_{\min}(M) (|x|^2 + |v|^2) \le V(x,v) \le \lambda_{\max}(M) (|x|^2 + |v|^2)
$$

to obtain:

$$
\mathcal{L}^*[V] \le -\beta_x|x|^2 - \beta_v|v|^2 + C \le -\frac{\min(\beta_x, \beta_v)}{\lambda_{\max}(M)} V + C
$$

setting $\beta := \frac{\min(\beta_x, \beta_v)}{\lambda_{\max}(M)} > 0$.

---

## 4. Technical Lemmas Required

### Lemma 4.1: Strong convexity growth bounds

**Statement**: Under Assumption A1, there exist constants $C_1, C_2 < \infty$ such that:

$$
\nabla_x U \cdot x \ge \kappa_{\text{conf}} |x|^2 - C_1, \quad |\nabla_x U| \le \kappa_{\text{conf}}|x| + C_2
$$

**Proof sketch**: Use $\nabla^2 U \ge \kappa_{\text{conf}} I$ and integrate along radial lines. The first inequality follows from integrating $\nabla_x U = \int_0^1 \nabla^2 U(tx) x \, dt$. The second uses triangle inequality and boundedness of $U$ on compact sets.

### Lemma 4.2: Positive definiteness of quadratic form

**Statement**: For $\varepsilon < \min\left(\frac{1}{\kappa_{\text{conf}}}, 1\right)$, the matrix:

$$
M = \begin{pmatrix} \kappa_{\text{conf}}\varepsilon & \varepsilon \\ \varepsilon & 1 \end{pmatrix}
$$

is positive definite with eigenvalues:

$$
\lambda_{\min} = \frac{1}{2}\left(1 + \kappa_{\text{conf}}\varepsilon - \sqrt{(1 - \kappa_{\text{conf}}\varepsilon)^2 + 4\varepsilon^2}\right) > 0
$$

$$
\lambda_{\max} = \frac{1}{2}\left(1 + \kappa_{\text{conf}}\varepsilon + \sqrt{(1 - \kappa_{\text{conf}}\varepsilon)^2 + 4\varepsilon^2}\right)
$$

**Proof sketch**: Compute $\det(M) = \kappa_{\text{conf}}\varepsilon - \varepsilon^2 > 0$ for small $\varepsilon$. The trace is $1 + \kappa_{\text{conf}}\varepsilon > 0$, so both eigenvalues are positive.

### Lemma 4.3: Parameter regime for negative drift

**Statement**: There exists a choice of $(a, b, c, \delta_1, \delta_2)$ such that the coefficients of $|x|^2$ and $|v|^2$ in $\mathcal{L}^*[V]$ are both negative, provided:

$$
\gamma > \gamma_{\text{crit}}(\kappa_{\text{conf}}, \sigma^2, d)
$$

for some explicit critical friction $\gamma_{\text{crit}}$.

**Proof sketch**: This requires careful optimization. The source document's calculation suggests $\gamma > \frac{4\kappa_{\text{conf}}}{9}$, but a more sophisticated choice may remove or weaken this condition. The diffusion term $\sigma^2 c d$ provides a constant offset, so the drift condition can be satisfied for any $\gamma > 0$ by absorbing the offset into $C$.

**KEY TECHNICAL GAP**: The existing calculation assumes a specific parameter choice. A complete proof should either:
1. Optimize over all $(a, b, c, \delta_1, \delta_2)$ to find the minimal friction requirement, OR
2. Prove that the drift condition holds for **all** $\gamma > 0$ by exploiting the diffusion term

---

## 5. Critical Issues and Gaps

### Issue 1: Friction condition dependence

**Problem**: The current proof sketch requires $\gamma > \frac{4\kappa_{\text{conf}}}{9}$, which is a **non-trivial constraint** not stated in Assumption A3.

**Severity**: HIGH - This is a gap between the theorem statement and the proof.

**Possible resolutions**:
1. **Add hypothesis**: Explicitly require $\gamma > \gamma_{\text{crit}}$ in the theorem statement
2. **Refine parameter choice**: Find a better $(a, b, c, \delta_1, \delta_2)$ that works for all $\gamma > 0$
3. **Use diffusion offset**: Show that the constant $C$ can absorb any positive drift, so only **eventual** drift negativity is needed

### Issue 2: Parameter optimization not complete

**Problem**: The source document tries one specific parameter choice, which fails to give optimal bounds. The space of admissible $(a, b, c, \delta_1, \delta_2)$ is five-dimensional.

**Severity**: MEDIUM - The proof is not rigorous without systematic optimization.

**Resolution**: Formulate as a **semidefinite programming problem**: maximize $\min(\beta_x, \beta_v)$ subject to $M \succ 0$. This is computationally tractable and would give explicit formulas.

### Issue 3: Dimension dependence of constant $C$

**Problem**: The constant $C$ includes terms like $\sigma^2 c d$ (linear in dimension $d$). High-dimensional behavior is not analyzed.

**Severity**: LOW - The drift condition holds, but $C$ may grow with $d$.

**Resolution**: Track all $d$-dependent terms explicitly. This is important for applications to high-dimensional optimization.

---

## 6. Difficulty Assessment

**Overall Difficulty**: **MEDIUM-HIGH**

- **Algebraic complexity**: HIGH - The term-by-term expansion and parameter optimization involve substantial algebra
- **Technical tools**: MEDIUM - Uses standard Lyapunov theory and Young's inequality
- **Conceptual novelty**: LOW - Drift conditions for Langevin dynamics are classical
- **Gap severity**: HIGH - The friction condition is a serious issue that must be resolved

**Expansion Time Estimate**: **8-12 hours**

- 2-3 hours: Rigorous term-by-term calculation with dimension tracking
- 3-4 hours: Parameter optimization (analytical or SDP formulation)
- 2-3 hours: Resolving the friction condition issue
- 1-2 hours: Writing up with full justifications and citations

---

## 7. Recommended Proof Strategy for Full Expansion

### Approach A: Explicit parameter optimization (Recommended)

1. **Normalize**: Set $c = 1$ WLOG
2. **Parameterize**: Write $a = \alpha \varepsilon$, $b = \varepsilon$ for small $\varepsilon > 0$ and $\alpha > 0$
3. **Expand**: Compute coefficients of $|x|^2$, $|v|^2$ as functions of $(\alpha, \varepsilon, \delta_1, \delta_2, \gamma, \kappa_{\text{conf}}, \sigma^2, d)$
4. **Optimize**: Solve for $(\alpha, \delta_1, \delta_2)$ to maximize $\min(\beta_x, \beta_v)$ subject to $M \succ 0$
5. **Verify**: Check that optimal choice works for all $\gamma > 0$ (or find minimal $\gamma_{\text{crit}}$)

**Advantages**: Systematic, yields explicit formulas, reveals precise parameter dependence

**Disadvantages**: Algebraically intensive

### Approach B: Variational reformulation

1. **Define energy**: $E(x,v) = \frac{1}{2}|v|^2 + U(x)$
2. **Perturbed Lyapunov**: Use $V_\theta(x,v) = E(x,v) + \theta x \cdot v$ for small $\theta > 0$
3. **Known results**: Cite classical hypocoercivity theory (Villani 2009, Dolbeault et al. 2015)
4. **Relate to quadratic form**: Show $V_\theta$ is equivalent to the quadratic form

**Advantages**: Connects to established literature, conceptually cleaner

**Disadvantages**: May not give explicit constants, requires strong convexity globally

### Approach C: Split analysis (if friction condition is needed)

1. **Case 1**: If $\gamma \ge \gamma_{\text{crit}}(\kappa_{\text{conf}})$, use the direct calculation
2. **Case 2**: If $\gamma < \gamma_{\text{crit}}$, use a **modified Lyapunov** that emphasizes velocity dissipation
3. **Union**: Show at least one case always works

**Advantages**: Handles all parameter regimes

**Disadvantages**: More complex proof structure

---

## 8. Connection to Broader Framework

### Role in convergence proof

This lemma is **Stage 0.5** in the mean-field KL-convergence roadmap. The drift condition establishes:

$$
\mathbb{E}[V(x_t, v_t) \mid x_0, v_0] \le e^{-\beta t} V(x_0, v_0) + \frac{C}{\beta}
$$

which implies:
1. **Exponential concentration**: $\rho_\infty(x,v) \le C_\rho e^{-\alpha(|x|^2 + |v|^2)}$ for some $\alpha > 0$
2. **Moment bounds**: $\int (|x|^2 + |v|^2)^k \rho_\infty \, dx dv < \infty$ for all $k \ge 1$
3. **LSI prerequisite**: Exponential tails are needed to prove the Log-Sobolev inequality in Stage 2

### Related results

- **Euclidean Gas framework**: Document `06_convergence.md` proves Foster-Lyapunov drift for the kinetic operator in the finite-$N$ setting
- **Hypocoercivity theory**: Villani (2009) establishes drift conditions for Langevin dynamics with equilibrium Maxwell-Boltzmann target
- **NESS hypocoercivity**: Dolbeault, Mouhot, Schmeiser (2015) extend to non-equilibrium stationary states

**Key difference**: Our $\rho_\infty$ is a **quasi-stationary distribution** with killing/revival, not an equilibrium. The drift condition must work with the **adjoint operator** $\mathcal{L}^*$, not assuming $\mathcal{L}(\rho_\infty) = 0$ for the kinetic part.

---

## 9. Literature and Citations

### Primary references

1. **Villani, C.** (2009). "Hypocoercivity". *Memoirs of the American Mathematical Society*, 202(950).
   - Chapter 3: Quadratic Lyapunov functionals for kinetic equations

2. **Dolbeault, J., Mouhot, C., Schmeiser, C.** (2015). "Hypocoercivity for linear kinetic equations conserving mass". *Transactions of the AMS*, 367(6), 3807-3828.
   - Section 2: Modified entropy functionals for NESS

3. **Bakry, D., Gentil, I., Ledoux, M.** (2014). *Analysis and Geometry of Markov Diffusion Operators*. Springer.
   - Chapter 5: Lyapunov functions and exponential convergence

### Secondary references

4. **Hairer, M., Mattingly, J.C.** (2011). "Yet another look at Harris' ergodic theorem for Markov chains". *Seminar on Stochastic Analysis, Random Fields and Applications VI*, 109-117.
   - Lyapunov drift conditions for ergodicity

5. **Cattiaux, P., Guillin, A.** (2014). "Functional inequalities for heavy tailed distributions and application to isoperimetry". *Electronic Journal of Probability*, 19, 1-30.
   - Drift conditions with polynomial tails

---

## 10. Verification Checklist for Full Proof

- [ ] All terms in $\mathcal{L}^*[V]$ computed correctly with explicit dimension dependence
- [ ] Strong convexity bounds (Lemma 4.1) proven rigorously
- [ ] Positive definiteness of $M$ verified for chosen parameters (Lemma 4.2)
- [ ] Parameter choice $(a, b, c, \delta_1, \delta_2)$ explicitly constructed
- [ ] Negative drift coefficients $\beta_x, \beta_v > 0$ verified algebraically
- [ ] Friction condition $\gamma > \gamma_{\text{crit}}$ either proven unnecessary or added as hypothesis
- [ ] Constant $C$ bounded explicitly in terms of $(\kappa_{\text{conf}}, \gamma, \sigma^2, d, C_1, C_2)$
- [ ] Spectral bounds on $M$ used correctly to relate $|x|^2 + |v|^2$ to $V$
- [ ] Final inequality $\mathcal{L}^*[V] \le -\beta V + C$ derived with explicit $\beta, C$
- [ ] All constants tracked through the proof for numerical verifiability

---

## 11. Confidence Assessment

**Confidence in theorem statement**: HIGH
- The need for a quadratic Lyapunov is well-motivated
- The adjoint operator formulation is correct
- The drift condition structure is standard

**Confidence in current proof approach**: MEDIUM
- The term-by-term calculation is sound
- The parameter choice is plausible but not optimal
- The friction condition issue must be resolved

**Confidence in eventual provability**: MEDIUM-HIGH
- The result is likely true for sufficiently large $\gamma$ or with modified parameters
- Standard hypocoercivity techniques should apply
- May require citing deeper functional analytic results (e.g., from Villani 2009)

**Red flags**:
- ⚠️ The friction condition $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ is not in Assumption A3
- ⚠️ The parameter optimization is incomplete (only one choice attempted)
- ⚠️ The source document shows multiple failed attempts (lines 2189-2194), suggesting subtlety

---

## 12. Recommendation

**For full proof development**:
1. **First**: Attempt **Approach B** (variational reformulation) to connect with Villani's hypocoercivity framework
2. **If blocked**: Fall back to **Approach A** (explicit optimization) with systematic SDP formulation
3. **If friction condition is unavoidable**: Either add it as a hypothesis or prove it's always satisfiable for physical parameters

**For pipeline continuation**:
- **Proceed with caution**: The drift condition is **assumed** in later stages (exponential concentration → LSI → KL-convergence)
- Flag the friction condition issue for the Math Reviewer
- Consider this lemma as **conditionally proven** pending resolution of the parameter choice

---

## 13. GPT-5 Review Request

**Review prompt** (to be submitted to Codex):

> Review the following proof sketch for mathematical rigor and completeness. Specifically assess:
>
> 1. **Correctness**: Are the term-by-term calculations of $\mathcal{L}^*[V]$ correct?
> 2. **Completeness**: Is the parameter optimization strategy sound, or are there better approaches?
> 3. **Friction condition**: Is the constraint $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ necessary, or can parameters be chosen to eliminate it?
> 4. **Literature**: Are there classical results (Villani, Dolbeault et al.) that directly imply this lemma?
> 5. **Gaps**: Identify any logical gaps or unstated assumptions in the proof strategy.
> 6. **Expansion plan**: Evaluate the recommended proof strategy (Approaches A/B/C) for feasibility.
>
> The theorem statement is:
>
> **Lemma**: Under Assumptions A1 (strong convexity $\nabla^2 U \ge \kappa_{\text{conf}} I$) and A3 (parameters $\gamma, \sigma^2 > 0$), there exist $a, b, c > 0$ such that $V(x,v) = a|x|^2 + 2b x \cdot v + c|v|^2$ satisfies:
> $$
> \mathcal{L}^*[V] \le -\beta V + C
> $$
> where $\mathcal{L}^* = v \cdot \nabla_x - \nabla_x U \cdot \nabla_v - \gamma v \cdot \nabla_v + \frac{\sigma^2}{2} \Delta_v$ and $\beta, C > 0$ are constants.
>
> Provide specific recommendations for completing the full proof rigorously.

**Expected review confidence**: MEDIUM (single-strategist, no Gemini cross-check)

---

**End of Proof Sketch**

**Next Steps**:
1. Submit sketch to GPT-5/Codex for review
2. Address review feedback
3. Decide on proof approach (A, B, or C)
4. Expand to full proof with explicit constants
5. Submit full proof to dual review (GPT-5 + Gemini when available)
