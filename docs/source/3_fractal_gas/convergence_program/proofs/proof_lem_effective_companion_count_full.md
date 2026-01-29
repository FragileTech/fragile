# Proof: Effective Companion Count (Finite-$N$ Heuristic)

:::{prf:lemma} Effective Companion Count (Finite-$N$ Heuristic)
:label: proof-lem-effective-companion-count-full

This lemma provides a **finite-$N$ heuristic** estimate and is **not used** in the mean-field
$C^\infty$ proof (which uses kernel-mass bounds instead). Assume the uniform density bound from
{prf:ref}`assump-uniform-density-full`. For any walker $i$, define the effective companion count

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
In the mean-field proof, $k_{\mathrm{eff}}$ is controlled by kernel-mass integrals against
$\rho_{\mathrm{QSD}}$ (Lemma {prf:ref}`lem-mean-field-kernel-mass-bound`). The estimate here is
retained only for finite-$N$ intuition.
:::
