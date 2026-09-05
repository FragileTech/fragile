# QFT Calibration Report: Standard Model Parameter Mapping

:::{div} feynman-prose

Calibration starts with two lists of numbers: physical couplings at a stated
scale, and statistics measured from a specified simulator run. A declared
parameter dictionary connects the lists. Solving that dictionary gives candidate
simulation parameters; testing the resulting correlations tells us whether the
identification describes the run.

This report records the algebra used by the calibration script. The
[Standard Model chapter](04_standard_model.md) supplies the representation and
normalization conditions, and the [calibration guide](09_qft_calibration.md)
specifies the active observables. In particular, the same-frame `lr_fraction`
and `lr_coupling_mag` masks give zero channels, and the routine named
`compute_su2_gauge_link` supplies a scalar
phase. Neither is an independent measurement of a non-Abelian coupling.

:::

(sec-qft-calibration-scope)=
## Scope and inputs

:::{div} feynman-prose

Choose a common physical scale and normalization for
$\alpha_{\rm em}$, $\sin^2\theta_W$, and $\alpha_s$. Record their uncertainties
and provenance. The simulator inputs are the dimension $d\ge2$, a mass scale
$m>0$, an action parameter $\hbar_{\rm eff}>0$, and the two pair statistics
$\mathcal N_1$ and $\langle K_{\rm visc}^2\rangle$. Specify the pair-sampling
law: uniform pairs, selected companions, and recorded time-history pairs can
have different averages.

The formulas below use numerical parameters in fixed reference units, with
$c=\hbar_0=1$ and hats suppressed. The reference action unit $\hbar_0$ and the
adjustable parameter $\hbar_{\rm eff}$ have different roles. Restore reference
factors using {prf:ref}`def-sm-coupling-definition` and the dimensionless
proxies in {prf:ref}`thm-su2-coupling-constant` and
{prf:ref}`thm-u1-coupling-constant` before interpreting dimensional output.
For the strong-coupling target, the proposed color identification uses $d=3$;
other dimensions define other proxy models.

:::

(sec-qft-calibration-couplings-report)=
## Couplings in the chosen convention

:::{prf:definition} Physical target convention
:label: def-qft-report-target-convention

In rationalized natural units, use

$$
e_{\rm em}=\sqrt{4\pi\alpha_{\rm em}},\qquad
 g_2=\frac{e_{\rm em}}{\sin\theta_W},\qquad
 g_1=\frac{e_{\rm em}}{\cos\theta_W},\qquad
 g_3=\sqrt{4\pi\alpha_s}.
$$

Here $g_1$ is the hypercharge coupling $g_Y$, not the alternative
$\sqrt{5/3}\,g_Y$ convention. These relations specify the electroweak target
normalization discussed in {prf:ref}`prop-sm-unification`.
:::

:::{div} feynman-prose

A normalization matters numerically. Replacing $g_Y$ by a conventionally
rescaled coupling while leaving the inversion formula unchanged would change
an inferred interaction range. Fix the convention before fitting anything.

:::

(sec-qft-calibration-inversion)=
## Parameter inversion in the declared model

:::{prf:definition} Calibration dictionary
:label: def-qft-report-dictionary

Identify the three proxies of {prf:ref}`def-sm-coupling-definition` with the
chosen positive targets. Impose also the fitness proxy, action relation, and
weak-coupling proxy below, all in the reference units just specified:

$$
\begin{aligned}
 g_2^2&=\frac{2\hbar_{\rm eff}}{\epsilon_c^2}
             \frac{C_2(2)}{C_2(d)},
 &C_2(n)&=\frac{n^2-1}{2n},\\
 g_1^2&=\frac{\hbar_{\rm eff}\mathcal N_1}{\epsilon_d^2},
 &g_3^2&=\frac{\nu^2}{\hbar_{\rm eff}^2}
        \frac{d(d^2-1)}{12}\langle K_{\rm visc}^2\rangle,\\
 e_{\rm em}^2&=\frac m{\epsilon_F},
 &\hbar_{\rm eff}&=\frac{m\epsilon_c^2}{2\tau},
 &g_2^2&=\frac{m\tau\rho^2}{\epsilon_c^2}.
\end{aligned}
$$

The fitness scale $\epsilon_F$ here includes its declared energy conversion.
The simultaneous identification of the two weak proxies is an additional
calibration constraint. Gauge symmetry alone does not impose it.
:::

:::{prf:proposition} Positive solution with fixed pair statistics
:label: prop-qft-report-inversion

For positive fixed inputs and $d\ge2$, the dictionary has the positive solution

$$
\begin{aligned}
 \epsilon_c&=\sqrt{\frac{2\hbar_{\rm eff}C_2(2)}{C_2(d)g_2^2}},
 &\epsilon_d&=\sqrt{\frac{\hbar_{\rm eff}\mathcal N_1}{g_1^2}},\\
 \nu&=\frac{\hbar_{\rm eff}g_3}
 {\sqrt{\frac{d(d^2-1)}{12}\langle K_{\rm visc}^2\rangle}},
 &\epsilon_F&=\frac m{e_{\rm em}^2},\\
 \tau&=\frac{m\epsilon_c^2}{2\hbar_{\rm eff}},
 &\rho&=\sqrt{\frac{2\hbar_{\rm eff}}{m^2}}\,g_2.
\end{aligned}
$$
:::

:::{prf:proof}
Solve the first four dictionary equations for their positive unknowns.
The action equation then gives $\tau$. Substituting it into the last weak
proxy yields $g_2^2=m^2\rho^2/(2\hbar_{\rm eff})$, giving $\rho$.
Every denominator and radicand is positive under the stated assumptions.
:::

:::{div} feynman-prose

This is an exact algebraic solution when the two measured statistics are held
fixed. A new run generally changes those statistics. The complete calibration
therefore asks for a self-consistent run as well as a solution of the displayed
equations. The Casimir calculation in {prf:ref}`thm-sm-g2-coupling` fixes the
chosen proxy factor; {prf:ref}`thm-sm-g1-coupling` bounds the diversity average;
and {prf:ref}`thm-sm-g3-coupling` shows which velocity cross terms enter the
actual viscous-force statistic.

:::

(sec-qft-calibration-qsd)=
## Estimating the pair statistics

:::{div} feynman-prose

Estimate $\mathcal N_1$ and $\langle K_{\rm visc}^2\rangle$ from `RunHistory`
using the sampling law in the dictionary. A QSD interpretation additionally
uses the survival conditioning and relaxation bounds for that run. Correlated
frames require an uncertainty estimate that accounts for temporal dependence.

An iteration can alternate parameter inversion and new simulations, updating
the two averages each time. Its convergence needs to be checked; positivity
of the algebraic solution does not make this feedback iteration contractive.
An order-one value supplied to the script is an initial guess, not a measured
QSD statistic.

:::

(sec-qft-calibration-continuum)=
## Resolution changes and time consistency

:::{prf:proposition} Constants preserved by the square-root rescaling
:label: prop-qft-report-resolution-scaling

At fixed $m$ and $\gamma$, the family

$$
 \epsilon_c(\tau)=\epsilon_c^{(0)}\sqrt{\tau/\tau_0},\qquad
 \rho(\tau)=\rho^{(0)}\sqrt{\tau/\tau_0}
$$

preserves $\hbar_{\rm eff}=m\epsilon_c^2/(2\tau)$ and
$\rho/\epsilon_c$. Writing $s=\tau/\tau_0$, it multiplies the Casimir range
proxy $2\hbar_{\rm eff}C_2(2)/(C_2(d)\epsilon_c^2)$ by $s^{-1}$ and the
weak proxy $m\tau\rho^2/\epsilon_c^2$ by $s$.
:::

:::{prf:proof}
Under this rescaling $\epsilon_c^2$ and $\rho^2$ each acquire the factor
$s$, while $\tau$ acquires $s$. Substitution gives all four claims.
:::

:::{div} feynman-prose

Thus a family that preserves the action parameter need not preserve the fitted
couplings. If the two weak proxies agree at $s=1$, this rescaling separates
them at other positive $s$. Recalibrate the chosen model when changing resolution.

Convergence to a continuous-time dynamics uses a different calculation:
{prf:ref}`thm-correct-continuum-limit` proves the accumulated error bound from
a local transition error and a stability estimate for the actual transition
family. Fixed per-step cloning probabilities require their own consistency
calculation. A dimensional scaling relation alone supplies none of these
transition estimates.

:::

(sec-qft-calibration-hierarchy)=
## Energy scales, decay rates, and ratios

:::{prf:definition} Scale diagnostics
:label: def-qft-report-scale-diagnostics

Retaining the action and speed factors, use the energy scales

$$
 E_c=\frac{\hbar_{\rm eff}c}{\epsilon_c},\qquad
 E_\rho=\frac{\hbar_{\rm eff}c}{\rho},\qquad
 E_{\rm gap}=\hbar_{\rm eff}\lambda_{\rm gap}.
$$

Their associated masses are $E/c^2$. The recorded shorthand
$m_{\rm clone}=1/\epsilon_c$, $m_{\rm MF}=1/\rho$ applies when the calibrated
units set $\hbar_{\rm eff}=c=1$. A spatial correlation length
$\xi=c/\lambda_{\rm gap}=\hbar_{\rm eff}c/E_{\rm gap}$ additionally requires
the temporal-to-spatial correlation identification in
{prf:ref}`def-correlation-length`; then $\xi=1/m_{\rm gap}$ in those same units.

Dimensionless diagnostics include

$$
 \sigma_{\rm sep}=\frac{\epsilon_c}{\rho},\qquad
 \eta_{\rm time}=\tau\lambda_{\rm gap},\qquad
 \kappa=\frac{c}{\rho\lambda_{\rm gap}}.
$$

The last expression is $1/(\rho\hbar_{\rm eff}\lambda_{\rm gap})$ only in
the shorthand units $\hbar_{\rm eff}=c=1$.
:::

:::{div} feynman-prose

These scales help choose a resolved simulation and compare lengths with decay
times. A channel mass still comes from that channel's spectral or correlation
measurement. The analytical ordering statement is
{prf:ref}`thm-mass-scales`, and cancellation of a common action calibration is
{prf:ref}`thm-dimensionless-ratios`. Neither assigns a nonzero mass to an
identically zero channel.

:::

(sec-qft-calibration-workflow-report)=
## Practical workflow and script usage

:::{div} feynman-prose

1. Supply physical couplings at a common scale and choose the reference units.
2. Estimate the two pair statistics for the specified run and sampling law.
3. Run the parameter inversion script with those inputs. For example, this
   command uses illustrative order-one statistics:

```bash
uv run python src/experiments/calibrate_fractal_gas_qft.py \
  --d 3 \
  --m-gev 1.0 \
  --hbar-eff 1.0 \
  --qsd-n1 1.0 \
  --qsd-kvisc2 1.0
```

4. Simulate the resulting parameters and remeasure the pair statistics.
5. Test nonzero active correlators with their masks, normalization, lag units,
   fit windows, and uncertainty recorded. Compare quantities in the same
   physical normalization.

The [empirical validation chapter](06_empirical_validation.md) describes the
statistical checks, and the [calibration guide](09_qft_calibration.md) explains
which parameter changes require a new run. Agreement with target numbers is
evidence for the tested identification at that resolution; the continuum and
field-realization hypotheses remain part of that identification.

:::
