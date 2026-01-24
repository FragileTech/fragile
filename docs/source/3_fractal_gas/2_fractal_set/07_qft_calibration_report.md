# QFT Calibration Report: Standard Model Parameter Mapping

This report maps measured Standard Model constants to algorithmic parameters in the
Fractal Gas QFT simulator. Every formula below is tied to Volume 3 statements in
`docs/source/3_fractal_gas/2_fractal_set` and does not introduce new assumptions.

(sec-qft-calibration-scope)=
## Scope and Inputs

Calibration requires two kinds of inputs:

- Measured constants at a chosen scale (for example, the $M_Z$ scale):
  - $\alpha_{\text{em}}$ (fine structure constant)
  - $\sin^2\theta_W$ (Weinberg angle)
  - $\alpha_s$ (strong coupling)
- Algorithmic choices and QSD statistics:
  - Dimension $d$ (the emergent $SU(d)$ color dimension)
  - $m$ (walker mass scale in natural units)
  - $\hbar_{\text{eff}}$ (effective Planck constant, {prf:ref}`thm-effective-planck-constant`)
  - $\mathcal{N}_1(T, d)$ and $\langle K_{\text{visc}}^2 \rangle_{\text{QSD}}$ (QSD statistics,
    {prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g3-coupling`)
  - Optional: $\lambda_{\text{gap}}$ and $\gamma$ for gap/hierarchy checks

The result is a consistent set of algorithmic parameters
$(\epsilon_d, \epsilon_c, \epsilon_F, \nu, \rho, \tau)$ that reproduces the
measured couplings through the Volume 3 coupling theorems.

(sec-qft-calibration-couplings)=
## Gauge Couplings from Measured Constants

At a chosen physical scale, convert measured constants into gauge couplings:

$$
e_{\text{em}} = \sqrt{4\pi\,\alpha_{\text{em}}}
$$

$$
g_2 = \frac{e_{\text{em}}}{\sin\theta_W}, \quad g_1 = \frac{e_{\text{em}}}{\cos\theta_W}
$$

$$
g_3 = \sqrt{4\pi\,\alpha_s}
$$

The Weinberg relation is consistent with {prf:ref}`prop-sm-unification`.

(sec-qft-calibration-inversion)=
## Algorithmic Parameter Inversion

Use the coupling theorems to solve for algorithmic parameters.

1. **Cloning interaction range** (from $SU(2)$ coupling, {prf:ref}`thm-sm-g2-coupling`):

$$
\epsilon_c = \sqrt{\frac{2\,\hbar_{\text{eff}}\,C_2(2)}{C_2(d)\,g_2^2}}, \quad
C_2(N) = \frac{N^2 - 1}{2N}
$$

2. **Diversity interaction range** (from $U(1)$ coupling, {prf:ref}`thm-sm-g1-coupling`):

$$
\epsilon_d = \sqrt{\frac{\hbar_{\text{eff}}\,\mathcal{N}_1(T, d)}{g_1^2}}
$$

This depends on the QSD normalization $\mathcal{N}_1(T, d)$.

3. **Viscous coupling** (from $SU(d)$ color coupling, {prf:ref}`thm-sm-g3-coupling`):

$$
\nu = \frac{\hbar_{\text{eff}}\,g_3}{\sqrt{\frac{d(d^2-1)}{12}\,\langle K_{\text{visc}}^2\rangle_{\text{QSD}}}}
$$

4. **Fitness coupling scale** (from $U(1)$ fitness coupling, {prf:ref}`thm-u1-coupling-constant` and
{prf:ref}`def-fine-structure-constant-ym`):

$$
\epsilon_F = \frac{m}{e_{\text{em}}^2}
$$

5. **Effective Planck constant constraint** ({prf:ref}`thm-effective-planck-constant`):

$$
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}
$$

6. **Localization scale** (from $SU(2)$ coupling normalization,
{prf:ref}`thm-su2-coupling-constant`):

$$
\rho = \sqrt{\frac{2\,\hbar_{\text{eff}}}{m^2}}\,g_2
$$

This is equivalent to solving $g_{\text{weak}}^2 = m\tau\rho^2/\epsilon_c^2$ with
$g_{\text{weak}}=g_2$.

(sec-qft-calibration-qsd)=
## QSD-Dependent Normalizations

Two quantities must be estimated from QSD statistics:

- $\mathcal{N}_1(T, d)$ in {prf:ref}`thm-sm-g1-coupling`
- $\langle K_{\text{visc}}^2 \rangle_{\text{QSD}}$ in {prf:ref}`thm-sm-g3-coupling`

These are expectations under the QSD of the Fractal Gas and should be computed
from saved `RunHistory` data. Because $\mathcal{N}_1$ depends on $\epsilon_d$,
calibration is typically iterative: initialize $\mathcal{N}_1$ with an order-one
estimate, solve for $\epsilon_d$, measure $\mathcal{N}_1$ from the resulting
QSD, then update.

(sec-qft-calibration-continuum)=
## Continuum Consistency and Scaling

To maintain correct physics across resolutions, apply the continuum scaling
prescription from {prf:ref}`thm-correct-continuum-limit`:

$$
\epsilon_c(\tau) = \epsilon_c^{(0)}\sqrt{\tau/\tau_0}, \quad
\rho(\tau) = \rho^{(0)}\sqrt{\tau/\tau_0}, \quad
\gamma = \text{fixed}
$$

Keep $\hbar_{\text{eff}} = m\epsilon_c^2/(2\tau)$ fixed while changing $\tau$.

(sec-qft-calibration-hierarchy)=
## Mass Scales, Gap, and Ratios

Use {prf:ref}`thm-mass-scales` and {prf:ref}`def-correlation-length` to validate
the mass hierarchy and correlation structure:

$$
m_{\text{clone}} = 1/\epsilon_c, \quad
m_{\text{MF}} = 1/\rho, \quad
m_{\text{gap}} = \hbar_{\text{eff}}\lambda_{\text{gap}}, \quad
\xi = 1/m_{\text{gap}}
$$

Dimensionless ratios from {prf:ref}`thm-dimensionless-ratios` provide diagnostics:

$$
\sigma_{\text{sep}} = \epsilon_c/\rho, \quad
\eta_{\text{time}} = \tau\lambda_{\text{gap}}, \quad
\kappa = \frac{1}{\rho\,\hbar_{\text{eff}}\lambda_{\text{gap}}}
$$

(sec-qft-calibration-workflow)=
## Practical Workflow and Script Usage

1. Choose a physical scale and import measured values of
   $(\alpha_{\text{em}}, \sin^2\theta_W, \alpha_s)$.
2. Set $(d, m, \hbar_{\text{eff}})$ to define the algorithmic unit system.
3. Estimate $\mathcal{N}_1(T, d)$ and $\langle K_{\text{visc}}^2\rangle_{\text{QSD}}$
   from simulation data.
4. Run the calibration script:

```bash
python src/experiments/calibrate_fractal_gas_qft.py \
  --d 3 \
  --m-gev 1.0 \
  --hbar-eff 1.0 \
  --qsd-n1 1.0 \
  --qsd-kvisc2 1.0
```

5. Use the output parameters $(\epsilon_c, \epsilon_d, \epsilon_F, \nu, \rho, \tau)$
   to configure the QFT simulator.

This workflow is compatible with the validation pipeline in
`docs/source/3_fractal_gas/2_fractal_set/06_empirical_validation.md`.
