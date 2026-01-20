# Proof: Telescoping Identity for Localization-Weight Derivatives

:::{prf:lemma} Telescoping Identity for Derivatives
:label: proof-lem-telescoping-derivatives

Let $\mathcal{A}$ be the alive set with $k = |\mathcal{A}| \ge 1$. Define the localization weights

$$
w_{ij}(\rho) := \frac{K_\rho(x_i, x_j)}{Z_i(\rho)}, \qquad Z_i(\rho) := \sum_{\ell \in \mathcal{A}} K_\rho(x_i, x_\ell),
$$

where $K_\rho$ is smooth in its first argument and strictly positive. Then for every derivative order $m \ge 1$,

$$
\sum_{j \in \mathcal{A}} \nabla_{x_i}^m w_{ij}(\rho) = 0.
$$
:::

:::{prf:proof}
Because $K_\rho(x_i, x_j) > 0$ and $k \ge 1$, we have $Z_i(\rho) > 0$ for all $x_i$, so each $w_{ij}$ is smooth in $x_i$. By construction,

$$
\sum_{j \in \mathcal{A}} w_{ij}(\rho) = \frac{\sum_{j \in \mathcal{A}} K_\rho(x_i, x_j)}{Z_i(\rho)} = 1
$$

as an identity in $x_i$. Differentiating both sides $m$ times and using linearity of differentiation with a finite sum yields

$$
\sum_{j \in \mathcal{A}} \nabla_{x_i}^m w_{ij}(\rho) = \nabla_{x_i}^m 1 = 0.
$$

This holds for any $m \ge 1$. \(\square\)
:::
