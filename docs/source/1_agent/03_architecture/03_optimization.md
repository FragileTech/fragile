(sec-optimizer-thermodynamic-governor)=
# Optimization: Thermodynamic Governor and Stable Updates

## TLDR

- We formalize the optimizer as a controlled dynamical system with a Lyapunov objective, consistent with the Universal
  Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`).
- We derive trust-region ("Mach limit") clipping, alignment damping, SNR gating, varentropy cooling, and thermal
  conduction from explicit assumptions and show the resulting descent or stability guarantees.
- Guarantees are conditional: they hold under stated smoothness, bounded-noise, and metric-approximation assumptions.
  No claim is made about global optimality.

## Roadmap

1. Formal setup and assumptions.
2. Preconditioned descent and trust-region (Mach limit) guarantee.
3. Varentropy brake as rigorous annealing control.
4. Alignment damping and oscillation suppression.
5. SNR gating for stochastic stability.
6. Thermal conduction across param-group learning rates.
7. Summary of guarantees and implementation surrogates.

## 1. Formal Setup: Controlled Optimization

We optimize a Lyapunov objective that combines task loss with constraint penalties as in the Governor framework
({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`). Let

$$
\mathcal{L}_{\text{task}}(\theta) \in \mathbb{R},
\qquad
C_k(\theta) \le 0 \text{ for } k=1,\ldots,K,
$$

and define the Lyapunov objective

$$
\mathcal{V}(\theta)
:= \mathcal{L}_{\text{task}}(\theta)
+ \sum_{k=1}^K \frac{\mu_k}{2}\max(0, C_k(\theta))^2.
$$

We write the controlled update as a preconditioned step

$$
\theta_{t+1} = \theta_t - \eta_t M_t g_t,
\qquad
g_t := \nabla_\theta \mathcal{V}(\theta_t),
$$

where $M_t$ is a symmetric positive definite (SPD) preconditioner. In practice, $M_t$ is a bounded surrogate that
implements the **trust-region regulation** implied by the state-space metric $G$ (see
{ref}`sec-second-order-sensitivity-value-defines-a-local-metric`), without equating parameter-space and state-space
geometry. Its surrogate construction is justified by the approximation doctrine
({ref}`sec-infeasible-implementation-replacements`).

We treat the learning rate as a temperature-scaled control input,

$$
\eta_t = \eta_0\,\tau_t,
$$

where $\tau_t$ is a dimensionless temperature scale driven by the Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`), and $\eta_0$ is the base
step size.

### Assumptions (Explicit)

We state the assumptions required for the guarantees in this chapter.

**A1 (Smoothness).** $\mathcal{V}$ is $L$-smooth: for all $x,y$,
$\mathcal{V}(y) \le \mathcal{V}(x) + \nabla\mathcal{V}(x)^\top (y-x) + \frac{L}{2}\|y-x\|^2$.

**A2 (Preconditioner bounds).** $M_t$ is SPD with eigenvalues in $[m_{\min}, m_{\max}]$ for all $t$, with
$0 < m_{\min} \le m_{\max} < \infty$. For the stochastic gate, $M_t$ is measurable with respect to the
past information $\sigma(\theta_t,\hat g_{t-1},\hat g_{t-2},\ldots)$ and is therefore independent of the current
noise conditional on $\theta_t$.

**A3 (Stochastic gradients).** The observed gradient is $\hat g_t = g_t + \xi_t$ with
$\mathbb{E}[\xi_t \mid \theta_t] = 0$ and $\mathbb{E}[\|\xi_t\|^2 \mid \theta_t] \le \sigma^2$.

**A4 (Temperature mapping).** The temperature scale $\tau_t$ controls the effective step size, and changes in $\tau_t$
are slow relative to parameter updates (timescale separation).

**A5 (Conduction ordering).** Parameter groups are ordered along the data-flow chain of a single physical subsystem
(Encoder $\to$ Latent Dynamics $\to$ Decoder). Cross-system conduction is disallowed.

These are standard in stability analyses and make the guarantees precise.

## 2. Preconditioned Descent and Trust-Region (Mach Limit)

:::{prf:definition} Preconditioned Update
:label: def-preconditioned-update
The preconditioned update is

$$
\theta_{t+1} = \theta_t - \eta_t M_t g_t,
$$
with $M_t$ SPD and $g_t = \nabla\mathcal{V}(\theta_t)$.
:::

:::{prf:theorem} Preconditioned Descent (Sufficient Condition)
:label: thm-preconditioned-descent
Under A1--A2, if

$$
0 < \eta_t < \frac{2 m_{\min}}{L m_{\max}^2},
$$
then the update in Definition {prf:ref}`def-preconditioned-update` satisfies

$$
\mathcal{V}(\theta_{t+1}) \le \mathcal{V}(\theta_t) - \left(\eta_t m_{\min} - \frac{L}{2}\eta_t^2 m_{\max}^2\right)
\|g_t\|^2.
$$
In particular, $\mathcal{V}$ decreases whenever $g_t \ne 0$. If
$\eta_t = 2 m_{\min} / (L m_{\max}^2)$, the right-hand side yields nonincrease.
:::

:::{prf:proof}
By $L$-smoothness (A1), for $d_t := \eta_t M_t g_t$,

$$
\mathcal{V}(\theta_t - d_t)
\le \mathcal{V}(\theta_t) - g_t^\top d_t + \frac{L}{2}\|d_t\|^2.
$$
Since $M_t$ is SPD with eigenvalues in $[m_{\min}, m_{\max}]$ (A2), we have
$g_t^\top M_t g_t \ge m_{\min}\|g_t\|^2$ and
$\|M_t g_t\|^2 \le m_{\max}^2\|g_t\|^2$. Substituting yields

$$
\mathcal{V}(\theta_{t+1})
\le \mathcal{V}(\theta_t) - \eta_t m_{\min}\|g_t\|^2 + \frac{L}{2}\eta_t^2 m_{\max}^2\|g_t\|^2.
$$
The right-hand side is strictly smaller than $\mathcal{V}(\theta_t)$ whenever
$\eta_t < 2 m_{\min} / (L m_{\max}^2)$ and yields nonincrease at equality. \qedhere
:::

:::{prf:definition} Relative Trust Region (Mach Limit)
:label: def-relative-trust-region
A relative trust region is the constraint

$$
\|\theta_{t+1} - \theta_t\| \le \kappa\,(\|\theta_t\| + \epsilon_\theta),
$$
with $\kappa \in (0,1)$ and $\epsilon_\theta \ge 0$. The case $\epsilon_\theta = 0$ recovers the strict relative form,
while $\epsilon_\theta > 0$ avoids degeneracy at $\|\theta_t\| = 0$.
:::

:::{prf:lemma} Trust-Region Scaling Preserves Descent
:label: lem-trust-region-scaling
Let $d_t = \eta_t M_t g_t$ with $\eta_t$ satisfying Theorem {prf:ref}`thm-preconditioned-descent`. Define the scaled step

$$
\tilde d_t = s d_t, \qquad s := \min\left(1, \frac{\kappa(\|\theta_t\| + \epsilon_\theta)}{\|d_t\|} \right).
$$
Then $\mathcal{V}(\theta_t - \tilde d_t) \le \mathcal{V}(\theta_t)$.
:::

:::{prf:proof}
If $d_t = 0$, then $\theta_{t+1} = \theta_t$ and the inequality holds trivially. Otherwise, for $s \in (0,1]$,
$\mathcal{V}(\theta_t - s d_t)$ satisfies the smoothness bound

$$
\mathcal{V}(\theta_t - s d_t)
\le \mathcal{V}(\theta_t) - s g_t^\top d_t + \frac{L}{2}s^2 \|d_t\|^2.
$$
The right-hand side minus $\mathcal{V}(\theta_t)$ equals
$s\,(-g_t^\top d_t+\tfrac{L}{2}s\|d_t\|^2)$. The bracket is nondecreasing in $s$ and is nonpositive at
$s=1$ by Theorem {prf:ref}`thm-preconditioned-descent`, hence nonpositive for all $s\in[0,1]$. \qedhere
:::

*Interpretation.* The Mach limit is a trust-region enforcement consistent with the Zeno/step-size constraint
({ref}`sec-using-scaling-exponents-to-gate-updates-and-tune-step-sizes`). It is rigorous provided the descent
condition holds and the scaling is applied by $s \le 1$.

## 3. Varentropy Brake: Rigorous Annealing Control

The Varentropy Brake is stated and proved in {prf:ref}`cor-varentropy-brake` and
{ref}`sec-appendix-e-proof-of-corollary-varentropy-brake`. We restate a discrete-time version.

:::{prf:proposition} Discrete Varentropy Brake
:label: prop-varentropy-brake-discrete
Let $T_t > 0$ be the cognitive temperature and $V_H(\theta_t)$ the varentropy. Define

$$
T_{t+1} = T_t\left(1 - \frac{\eta_T}{1 + \gamma V_H(\theta_t)}\right),
$$
with $\eta_T \in (0,1)$ and $\gamma > 0$. If a relaxation constant $C>0$ is available and
$\eta_T\le 2C\sqrt{\gamma}$, then

1. $0 < T_{t+1} \le T_t$ (temperature is positive and nonincreasing), and
2. $|T_{t+1} - T_t| \le C T_t / \sqrt{V_H(\theta_t)}$ whenever $V_H(\theta_t)>0$ (cooling is slowed when
   $V_H$ is large).
:::

:::{prf:proof}
Since $\eta_T \in (0,1)$ and $V_H \ge 0$, the multiplier lies in $(0,1]$, proving positivity and monotone
nonincrease. The update gives the first bound, and the condition $\eta_T\le2C\sqrt\gamma$ together with
$1+\gamma V_H\ge2\sqrt{\gamma V_H}$ gives the second. \qedhere
:::

*Implementation note.* When we identify $\eta_t = \eta_0 \tau_t$ with $\tau_t \propto T_t$, this gives an
admissible annealing schedule when the displayed $C$-bound holds and A4's slow-change condition has been
checked. Governor-initiated reheating is outside this monotone-brake proposition.

## 4. Alignment Damping: Oscillation Suppression

Alignment damping is a step-size shrinkage rule. It preserves the local descent bound when the underlying
update is already a descent step; it does not by itself prove a momentum-oscillation bound.

:::{prf:definition} Gradient-Momentum Alignment
:label: def-gradient-alignment
Let $g_t$ be the gradient and $m_t$ a momentum estimate. Define the alignment score

$$
a_t := g_t^\top m_t.
$$
:::

:::{prf:proposition} Alignment-Triggered Step Damping
:label: prop-alignment-step-damping
Let $a_t := g_t^\top m_t$ and fix a damping factor $\rho \in (0,1]$. Define

$$
\eta_t^{+} :=
\begin{cases}
\rho\,\eta_t & \text{if } a_t < 0, \\
\eta_t & \text{otherwise}.
\end{cases}
$$
If $\eta_t$ satisfies Theorem {prf:ref}`thm-preconditioned-descent`, then the damped step with $\eta_t^{+}$ also
decreases $\mathcal{V}$.
:::

:::{prf:proof}
The update with $\eta_t^{+}$ is a scaled version of the step with $\eta_t$, with scaling factor in $(0,1]$.
By the same smoothness inequality used in the proof of Theorem {prf:ref}`thm-preconditioned-descent`, any reduction
in step size preserves nonincreasing $\mathcal{V}$ as long as the original step was in the descent regime.
Equivalently, this is a special case of the scaling argument in Lemma {prf:ref}`lem-trust-region-scaling`. \qedhere
:::

*Interpretation.* When the alignment trigger fires, reducing $\eta_t$ preserves the descent bound of the
underlying gradient step. A separate momentum analysis is required to claim oscillation suppression.

## 5. SNR Gate: Stochastic Stability

We derive a conservative learning-rate gate from the stochastic update model (A3), where the update uses
$\hat g_t$ instead of $g_t$.

:::{prf:proposition} SNR-Gated Step Size
:label: prop-snr-gate
Under A1--A3 and the stochastic update $\theta_{t+1} = \theta_t - \eta_t M_t \hat g_t$, assume $g_t \ne 0$. The
expected Lyapunov change satisfies

$$
\mathbb{E}[\mathcal{V}(\theta_{t+1})\mid\theta_t]
\le \mathcal{V}(\theta_t)
- \eta_t m_{\min}\|g_t\|^2
+ \frac{L}{2}\eta_t^2 m_{\max}^2\left(\|g_t\|^2 + \sigma^2\right).
$$
Consequently, a sufficient condition for expected descent is

$$
\eta_t
\le
\frac{2 m_{\min}}{L m_{\max}^2\left(1 + \sigma^2 / \|g_t\|^2\right)}
= \frac{2 m_{\min}}{L m_{\max}^2}\cdot\frac{\mathrm{SNR}}{1+\mathrm{SNR}},
$$
with $\mathrm{SNR} := \|g_t\|^2 / \sigma^2$. If $g_t = 0$, the sufficient bound reduces to $\eta_t = 0$.
:::

:::{prf:proof}
Apply the smoothness bound with $d_t = \eta_t M_t \hat g_t$ and take conditional expectations. The measurability
clause in A2 makes $M_t$ independent of the current noise, so
$\mathbb{E}[g_t^\top M_t\xi_t\mid\theta_t]=0$.
Use $\mathbb{E}[\hat g_t] = g_t$ and
$\mathbb{E}[\|\hat g_t\|^2] = \|g_t\|^2 + \mathbb{E}[\|\xi_t\|^2] \le \|g_t\|^2 + \sigma^2$.
Then apply the eigenvalue bounds from A2 and solve for $\eta_t$ to make the coefficient negative. \qedhere
:::

*Interpretation.* When SNR is low, the gate shrinks $\eta_t$ to protect expected descent. This is a rigorous
stochastic analogue of the "cool when signal is weak" rule.

## 6. Thermal Conduction Across Param Groups

Thermal conduction smooths learning rates across adjacent blocks inside a single physical system (A5). We implement
conduction in log space to preserve positivity.

:::{prf:definition} Log-LR Conduction Update
:label: def-log-lr-conduction
Let $\eta_i > 0$ be per-group learning rates and $x_i := \log(\eta_i)$. Define

$$
 x_i^{+} = x_i + \frac{k}{2}(x_{i-1} - 2 x_i + x_{i+1}),
$$
with Neumann boundary conditions $x_0 = x_1$, $x_{n+1} = x_n$ and conductivity $k \in [0,1]$.
:::

:::{prf:proposition} Conduction Contracts LR Disparities
:label: prop-conduction-contracts
Let $L$ be the path-graph Laplacian on $n$ groups and define the energy

$$
E(x) := \frac{1}{2} x^\top L x = \frac{1}{2}\sum_{i=1}^{n-1}(x_{i+1} - x_i)^2.
$$
The update in Definition {prf:ref}`def-log-lr-conduction` is gradient descent on $E$ with step size $k/2$, and for
$k \in [0,1]$ it satisfies $E(x^{+}) \le E(x)$.
:::

:::{prf:proof}
We have $\nabla E(x) = L x$ for the path graph. The update is
$x^{+} = x - (k/2) L x$, i.e. gradient descent with step size $k/2$.
Because $L$ is symmetric positive semidefinite with eigenvalues bounded by $\lambda_{\max} \le 4$ for a path graph,
step size $k/2 \le 1/2$ ensures $(1 - (k/2)\lambda)^2 \le 1$ for all eigenvalues. Therefore
$E(x^{+}) = \tfrac{1}{2} x^\top (I - (k/2)L)^\top L (I - (k/2)L)x \le E(x)$. \qedhere
:::

*Interpretation.* Conduction is a provably contractive smoothing of log learning rates. It enforces coherence without
mixing separate physical subsystems.

## 7. Combined Guarantees (Under Assumptions)

:::{prf:theorem} Thermodynamic Governor Stability (Conditional)
:label: thm-optimizer-conditional-stability
Under A1--A5, with updates that apply in the order: conduction on log learning rates, SNR/alignment gates,
the preconditioned step, and trust-region clipping last, and with $M_t$ block-diagonal across parameter groups,
assume:
1. preconditioned descent (Theorem {prf:ref}`thm-preconditioned-descent`),
2. trust-region scaling (Lemma {prf:ref}`lem-trust-region-scaling`),
3. alignment-triggered step damping (Proposition {prf:ref}`prop-alignment-step-damping`),
4. varentropy brake (Proposition {prf:ref}`prop-varentropy-brake-discrete`),
5. SNR gating (Proposition {prf:ref}`prop-snr-gate`), and
6. log-LR conduction (Proposition {prf:ref}`prop-conduction-contracts`),

the optimizer produces a nonincreasing Lyapunov objective in the deterministic case. The expected-descent claim
under noise applies only to the unscaled stochastic step in Proposition {prf:ref}`prop-snr-gate`; trust-region
clipping and alignment damping are covered there only in the deterministic regime. Learning rates remain positive
and coherent across adjacent groups, and the temperature schedule obeys the adiabatic constraint during annealing
phases with no Governor-initiated heating.
:::

:::{prf:remark}
:label: rem-agent-optimization-conditional
These guarantees are conditional and local. They ensure stability and controlled descent, not global optimality.
Violations of A1--A5 (e.g., non-smooth losses, unbounded noise, misordered groups) void the guarantees.
:::

## 8. Implementation Surrogates (Rigorous, but Practical)

The theory uses metric-aware updates and diagnostic signals. Practical implementations may use the following
surrogates, which are explicitly permitted in {ref}`sec-infeasible-implementation-replacements`:

- Use a **lagged** Adam second-moment statistic
  $M_t=\operatorname{diag}(1/(\sqrt{\hat v_{t-1}}+\epsilon))$, clipped to $[m_{\min},m_{\max}]$, as a
  diagonal proxy for the trust-region regulator. The usual current-step Adam direction lies outside the
  measurability class analysed by Proposition {prf:ref}`prop-snr-gate`.
- Estimate $\mathrm{SNR}$ with running averages of gradient norms.
- Replace exact varentropy with a windowed variance of the chosen metric signal (loss or grad RMS).
- Implement conduction on param groups ordered by data-flow to satisfy A5.

Each surrogate preserves the theoretical intent: bounded step sizes, adiabatic annealing, step-size damping,
noise-aware cooling, and cross-layer coherence.

## 9. Summary Checklist

1. Verify A1--A5 for your setting (or document violations).
2. Enforce trust-region scaling when relative step sizes exceed $\kappa$.
3. Apply varentropy brake to $T_c$ (and thus $\eta_t$) for safe annealing.
4. Use alignment-triggered step damping as a descent-preserving shrinkage rule; verify any momentum claim separately.
5. Gate $\eta_t$ by SNR to maintain expected descent.
6. Apply log-LR conduction only within a single physical subsystem.

This chapter connects the Governor theory to local optimizer guarantees under the stated assumptions and scope.
