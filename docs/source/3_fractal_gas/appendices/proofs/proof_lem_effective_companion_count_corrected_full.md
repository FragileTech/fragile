# Proof: Effective Companion Count

:::{prf:lemma} Effective Companion Count
:label: proof-lem-effective-companion-count-corrected-full

Assume the uniform density bound from {prf:ref}`assump-uniform-density-full`. For any walker $i$, define the effective companion count

$$
k_{\mathrm{eff}}(i) := \sum_{\ell \in \mathcal{A} \setminus \{i\}} \mathbb{1}_{d_{\mathrm{alg}}(i,\ell) \le R_{\mathrm{eff}}}.
$$

Then

$$
k_{\mathrm{eff}}(i) \le \rho_{\max} \, C_{\mathrm{vol}} \, R_{\mathrm{eff}}^{2d},
$$

and with $R_{\mathrm{eff}} = O(\varepsilon_c \sqrt{\log k})$ (Corollary {prf:ref}`cor-effective-interaction-radius-full`),

$$
k_{\mathrm{eff}}(i) = O\bigl(\varepsilon_c^{2d} (\log k)^d\bigr).
$$
:::

:::{prf:proof}
Let

$$
B_i := \{(x,v): d_{\mathrm{alg}}((x,v),(x_i,v_i)) \le R_{\mathrm{eff}}\}.
$$

Then

$$
k_{\mathrm{eff}}(i) = \#\{\ell \in \mathcal{A} \setminus \{i\}: (x_\ell, v_\ell) \in B_i\}.
$$

By the density bound assumption,

$$
\#\{\ell \in \mathcal{A}: (x_\ell, v_\ell) \in B_i\} \le \rho_{\max} \, \mathrm{Vol}(B_i) = \rho_{\max} \, C_{\mathrm{vol}} \, R_{\mathrm{eff}}^{2d}.
$$

This yields the first inequality. The second follows by substituting $R_{\mathrm{eff}} = O(\varepsilon_c \sqrt{\log k})$.
\(\square\)
:::

:::{note}
If the density bound is interpreted in expectation (mean-field level), the same estimate holds for $\mathbb{E}[k_{\mathrm{eff}}(i)]$; concentration upgrades require additional mixing assumptions.
:::
