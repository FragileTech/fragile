# Proof: Effective Interaction Radius (Finite-$N$ Heuristic)

:::{prf:corollary} Effective Interaction Radius (Finite-$N$ Heuristic)
:label: proof-cor-effective-interaction-radius-full

This corollary is a **finite-$N$ heuristic** and is **not used** in the mean-field $C^\infty$
proof (which uses kernel-mass bounds instead). Assume the softmax tail bound from
{prf:ref}`lem-softmax-tail-corrected-full` and let $k = |\mathcal{A}| \ge 2$. Define

$$
R_{\mathrm{eff}} := \sqrt{R_{\max}^2 + 2\varepsilon_c^2 \log(k^2)} = \varepsilon_c \sqrt{C_{\mathrm{comp}}^2 + 2\log(k^2)},
$$

where $R_{\max} = C_{\mathrm{comp}}\, \varepsilon_c$. Then

$$
\mathbb{P}(d_{\mathrm{alg}}(i, c(i)) > R_{\mathrm{eff}}) \le \frac{1}{k}.
$$
:::

:::{prf:proof}
From {prf:ref}`lem-softmax-tail-corrected-full`,

$$
\mathbb{P}(d_{\mathrm{alg}}(i, c(i)) > R) \le k \exp\left(-\frac{R^2 - R_{\max}^2}{2\varepsilon_c^2}\right).
$$

Set the right-hand side equal to $1/k$ and solve for $R$:

$$
\exp\left(-\frac{R_{\mathrm{eff}}^2 - R_{\max}^2}{2\varepsilon_c^2}\right) = k^{-2},
$$

so

$$
R_{\mathrm{eff}}^2 = R_{\max}^2 + 2\varepsilon_c^2 \log(k^2).
$$

Substituting this into the tail bound gives the claimed inequality. \(\square\)
:::

:::{note}
For fixed $\varepsilon_c$, the growth is $R_{\mathrm{eff}} = O(\varepsilon_c \sqrt{\log k})$.
Practical numeric estimates can be added as a separate implementation note if desired.
:::
