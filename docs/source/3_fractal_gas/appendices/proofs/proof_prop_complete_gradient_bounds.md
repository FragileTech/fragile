# Proof: Complete Gradient and Laplacian Bounds (Compact Setting)

:::{prf:proposition} Complete Gradient and Laplacian Bounds
:label: proof-prop-complete-gradient-bounds

Assume the effective alive domain $\Omega = \mathcal{X} \times \mathbb{R}^d$ is compactified
by the confining envelope of the framework (so $\mathcal{X}$ is compact, or all statements are
restricted to a compact $\Omega_{\text{eff}}$ on which the QSD is supported). If the QSD density
$\rho_\infty$ is strictly positive and $C^3$ on $\Omega_{\text{eff}}$, then there exist
constants $C_x, C_\Delta < \infty$ such that

$$
\|\nabla_x \log \rho_\infty\|_{L^\infty(\Omega_{\text{eff}})} \le C_x,
\qquad
\|\Delta_v \log \rho_\infty\|_{L^\infty(\Omega_{\text{eff}})} \le C_\Delta.
$$
:::

:::{prf:proof}
Strict positivity and $C^3$ regularity imply $\log \rho_\infty \in C^3(\Omega_{\text{eff}})$.
On a compact set, every continuous derivative attains its maximum, so the $L^\infty$ norms of
$\nabla_x \log \rho_\infty$ and $\Delta_v \log \rho_\infty$ are finite. \(\square\)
:::
