# Proof: Exponential Convergence to QSD (Curvature Route)

:::{prf:corollary} Exponential Convergence to QSD
:label: proof-cor-exponential-qsd-companion-dependent-full

Assume the Log-Sobolev Inequality from {prf:ref}`thm-lsi-companion-dependent-full`. Then the Geometric Gas converges exponentially to its QSD in $L^2$:

$$
\|\rho_t - \nu_{\mathrm{QSD}}\|_{L^2(\nu_{\mathrm{QSD}})} \le e^{-\lambda_{\mathrm{gap}} t} \|\rho_0 - \nu_{\mathrm{QSD}}\|_{L^2(\nu_{\mathrm{QSD}})},
$$

with $\lambda_{\mathrm{gap}} \ge \alpha$ and $\alpha$ the LSI constant.
:::

:::{prf:proof}
An LSI with constant $\alpha>0$ implies a Poincare inequality with spectral gap $\lambda_{\mathrm{gap}} \ge \alpha$. The Poincare inequality yields exponential $L^2$ convergence to the invariant measure by standard semigroup theory for Markov generators. Applying this to the Geometric Gas semigroup gives the stated bound. \(\square\)
:::

:::{note}
If a curvature bound is available, the Bakry-Emery route provides explicit constants for $\alpha$ and hence for $\lambda_{\mathrm{gap}}$. This refinement is optional and does not affect the qualitative convergence statement.
:::
