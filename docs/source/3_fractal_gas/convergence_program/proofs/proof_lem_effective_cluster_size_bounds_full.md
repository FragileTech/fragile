# Proof: Bounds on Effective Cluster Size

:::{prf:lemma} Bounds on Effective Cluster Size (Mean-Field)
:label: proof-lem-effective-cluster-size-bounds-full

Let $\{\psi_m\}_{m=1}^M$ be the smooth partition-of-unity cluster functions and define the
mean-field cluster mass

$$
k_{m,\mathrm{mf}}^{\mathrm{eff}} := \int_{\mathcal{Y}} \psi_m(y)\, \rho_{\mathrm{QSD}}(y)\, dy,
$$

where $\rho_{\mathrm{QSD}}$ satisfies Theorem {prf:ref}`assump-uniform-density-full`.
Then

$$
k_{m,\mathrm{mf}}^{\mathrm{eff}} \le \rho_{\max} \, \mathrm{Vol}(B(y_m, 2\varepsilon_c))
= C_{\mathrm{vol}}\, \rho_{\max}\, \varepsilon_c^{2d},
$$

and the mean-field masses conserve total mass:

$$
\sum_{m=1}^M k_{m,\mathrm{mf}}^{\mathrm{eff}} = 1.
$$

For finite $N$, the empirical effective counts
$k_m^{\mathrm{eff}} := \sum_{j \in \mathcal{A}} \psi_m(x_j, v_j)$ satisfy
$\mathbb{E}[k_m^{\mathrm{eff}}]/k \to k_{m,\mathrm{mf}}^{\mathrm{eff}}$ by propagation of chaos,
so the same bound holds in expectation.
:::

:::{prf:proof}
Because $\psi_m$ is supported on the ball $B(y_m, 2\varepsilon_c)$, we have

$$
0 \le \psi_m(x_j, v_j) \le \mathbb{1}_{(x_j, v_j) \in B(y_m, 2\varepsilon_c)}.
$$

Hence, in the mean-field limit,

$$
k_{m,\mathrm{mf}}^{\mathrm{eff}}
= \int_{\mathcal{Y}} \psi_m(y)\, \rho_{\mathrm{QSD}}(y)\, dy
\le \rho_{\max} \, \mathrm{Vol}(B(y_m, 2\varepsilon_c)).
$$

The phase-space dimension is $2d$, so

$$
\mathrm{Vol}(B(y_m, 2\varepsilon_c)) = \frac{\pi^d}{d!} (2\varepsilon_c)^{2d} = C_{\mathrm{vol}}\, \varepsilon_c^{2d}.
$$

For conservation, use the partition-of-unity property $\sum_{m=1}^M \psi_m(x, v) = 1$ and the
normalization of $\rho_{\mathrm{QSD}}$:

$$
\sum_{m=1}^M k_{m,\mathrm{mf}}^{\mathrm{eff}}
= \int_{\mathcal{Y}} \sum_{m=1}^M \psi_m(y)\, \rho_{\mathrm{QSD}}(y)\, dy
= \int_{\mathcal{Y}} \rho_{\mathrm{QSD}}(y)\, dy
= 1.
$$

\(\square\)
:::

:::{note}
For finite $N$, concentration upgrades (beyond expectation) require additional mixing assumptions
(e.g., LSI-based concentration for empirical measures).
:::
