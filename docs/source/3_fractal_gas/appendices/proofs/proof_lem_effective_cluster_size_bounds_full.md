# Proof: Bounds on Effective Cluster Size

:::{prf:lemma} Bounds on Effective Cluster Size
:label: proof-lem-effective-cluster-size-bounds-full

Let $\{\psi_m\}_{m=1}^M$ be the smooth partition-of-unity cluster functions and define

$$
k_m^{\mathrm{eff}} := \sum_{j \in \mathcal{A}} \psi_m(x_j, v_j).
$$

Assume the framework density bound (Theorem {prf:ref}`assump-uniform-density-full`) in algorithmic coordinates: for any phase-space ball $B$,

$$
\#\{j \in \mathcal{A}: (x_j, v_j) \in B\} \le \rho_{\max} \, \mathrm{Vol}(B).
$$

Then

$$
k_m^{\mathrm{eff}} \le \rho_{\max} \, \mathrm{Vol}(B(y_m, 2\varepsilon_c)) = C_{\mathrm{vol}}\, \rho_{\max}\, \varepsilon_c^{2d},
$$

and the effective populations conserve total mass:

$$
\sum_{m=1}^M k_m^{\mathrm{eff}} = k.
$$
:::

:::{prf:proof}
Because $\psi_m$ is supported on the ball $B(y_m, 2\varepsilon_c)$, we have

$$
0 \le \psi_m(x_j, v_j) \le \mathbb{1}_{(x_j, v_j) \in B(y_m, 2\varepsilon_c)}.
$$

Hence

$$
k_m^{\mathrm{eff}} = \sum_{j \in \mathcal{A}} \psi_m(x_j, v_j)
\le \#\{j \in \mathcal{A}: (x_j, v_j) \in B(y_m, 2\varepsilon_c)\}
\le \rho_{\max} \, \mathrm{Vol}(B(y_m, 2\varepsilon_c)).
$$

The phase-space dimension is $2d$, so

$$
\mathrm{Vol}(B(y_m, 2\varepsilon_c)) = \frac{\pi^d}{d!} (2\varepsilon_c)^{2d} = C_{\mathrm{vol}}\, \varepsilon_c^{2d}.
$$

For conservation, use the partition-of-unity property $\sum_{m=1}^M \psi_m(x, v) = 1$:

$$
\sum_{m=1}^M k_m^{\mathrm{eff}}
= \sum_{m=1}^M \sum_{j \in \mathcal{A}} \psi_m(x_j, v_j)
= \sum_{j \in \mathcal{A}} \sum_{m=1}^M \psi_m(x_j, v_j)
= \sum_{j \in \mathcal{A}} 1
= k.
$$

\(\square\)
:::

:::{note}
If the density bound is interpreted at the mean-field level, the same estimate applies to the expected effective population. Concentration upgrades require additional mixing assumptions (e.g., LSI-based concentration for empirical measures).
:::
