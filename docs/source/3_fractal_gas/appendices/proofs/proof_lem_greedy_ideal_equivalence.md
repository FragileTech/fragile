# Proof: Greedy Pairing Regularity Transfer

:::{prf:lemma} Statistical Equivalence Preserves C^∞ Regularity
:label: proof-lem-greedy-ideal-equivalence

Let $P_{\mathrm{greedy}}(M\mid S)$ denote the sequential stochastic greedy pairing distribution (Definition {prf:ref}`def-greedy-pairing-algorithm` in {doc}`03_cloning`), and define the greedy expected measurement

$$
\bar d_i^{\mathrm{greedy}}(S) := \mathbb{E}_{M \sim P_{\mathrm{greedy}}(\cdot\mid S)}[d_{\mathrm{alg}}(i, M(i))].
$$

Then $\bar d_i^{\mathrm{greedy}}(S)$ is a $C^\infty$ function of the swarm state with the same k-uniform Gevrey-1 derivative bounds as the idealized pairing expectation from Theorem {prf:ref}`thm-diversity-pairing-measurement-regularity`.
:::

:::{prf:proof}
The greedy pairing distribution is a finite sum of products of softmax weights defined from the smooth kernel $\exp(-d_{\mathrm{alg}}^2/(2\varepsilon_d^2))$ and the regularized distance $d_{\mathrm{alg}}$. Each greedy realization probability is a product of finitely many smooth factors, and the expectation $\bar d_i^{\mathrm{greedy}}$ is a finite weighted sum of $d_{\mathrm{alg}}(i, \ell)$ with those smooth weights.

Because $d_{\mathrm{alg}}$ is $C^\infty$ with uniform bounds (regularization $\varepsilon_d > 0$) and each softmax denominator is bounded below by companion availability (Lemma {prf:ref}`lem-companion-availability-enforcement`), repeated application of the product and quotient rules yields $C^\infty$ regularity of $\bar d_i^{\mathrm{greedy}}$. The derivative bounds follow from the same locality and telescoping arguments used in the idealized pairing analysis (notably {prf:ref}`lem-derivative-locality-cinf` and {prf:ref}`lem-telescoping-localization-weights-full`), so the Gevrey-1 constants depend only on $(\varepsilon_d, d, \rho_{\max})$ and are independent of $k$ and $N$.

Thus the greedy mechanism inherits the same k-uniform Gevrey-1 regularity as the idealized pairing. \(\square\)
:::

:::{note}
If an additional statistical equivalence bound $|\mathbb{E}_{\mathrm{greedy}}[d_i\mid S] - \mathbb{E}_{\mathrm{ideal}}[d_i\mid S]| \le C k^{-\beta}$ is established, it can be layered on top of this lemma as a quantitative comparison. The regularity transfer itself does not require that rate.
:::
