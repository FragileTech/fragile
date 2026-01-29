# Proof: Macroscopic Transport Inequality

:::{prf:lemma} Macroscopic Transport (Poincare Form)
:label: proof-lem-macro-transport

Let $\rho_{\text{QSD}}(x,v)$ have position marginal $\rho_x(x)$ and conditional velocity
covariance $\Sigma_v(x) := \int v v^\top \rho_{\text{QSD}}(v|x)\,dv$. Assume:

1. **Position Poincare**: $\rho_x$ satisfies
   
   $$
   \|a\|^2_{L^2(\rho_x)} \le \frac{1}{\kappa_x}\|\nabla_x a\|^2_{L^2(\rho_x)}
   \quad \text{for all } a \text{ with } \int a\,\rho_x = 0.
   $$
2. **Uniform covariance**: $\Sigma_v(x) \succeq c_v I_d$ for all $x$.
3. **Centered velocities**: $\int v\,\rho_{\text{QSD}}(v|x)\,dv = 0$ for all $x$.

Then for any $h \in H^1(\rho_{\text{QSD}})$ with $\int h\,\rho_{\text{QSD}} = 1$, letting
$a := \Pi h - 1$ (velocity average), we have

$$
\|a\|^2_{L^2(\rho_x)} \le \frac{1}{\kappa_x c_v}\,
\|v \cdot \nabla_x a\|^2_{L^2(\rho_{\text{QSD}})}.
$$
:::

:::{prf:proof}
By the Poincare inequality for $\rho_x$ and $\int a\,\rho_x = 0$,

$$
\|a\|^2_{L^2(\rho_x)} \le \frac{1}{\kappa_x}\|\nabla_x a\|^2_{L^2(\rho_x)}.
$$

For each $x$, the covariance bound implies

$$
|\nabla_x a(x)|^2 \le \frac{1}{c_v} \int |v\cdot\nabla_x a(x)|^2 \rho_{\text{QSD}}(v|x)\,dv.
$$

Integrating over $x$ yields

$$
\|\nabla_x a\|^2_{L^2(\rho_x)}
\le \frac{1}{c_v} \|v\cdot\nabla_x a\|^2_{L^2(\rho_{\text{QSD}})}.
$$

Combining the two inequalities gives the claim. \(\square\)
:::
