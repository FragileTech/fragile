# Proof: Variance-to-Gap Lemma

:::{prf:lemma} Variance-to-Gap (Universal)
:label: proof-lem-variance-to-gap-adaptive

Let $X$ be a real random variable with mean $\mu$ and variance $\sigma^2>0$. Then

$$
\sup_{x \in \operatorname{supp}(X)} |x-\mu| \ge \sigma.
$$

If the support is bounded, the supremum is attained and equals the maximum.
:::

:::{prf:proof}
Let $R := \sup_{x \in \operatorname{supp}(X)} |x-\mu| \in [0,\infty]$. By definition of support,
$|X-\mu| \le R$ almost surely. Hence $\mathbb{E}[(X-\mu)^2] \le R^2$, so $\sigma^2 \le R^2$ and
therefore $\sigma \le R$. If $R<\infty$, continuity of $x \mapsto |x-\mu|$ on the compact support
implies the supremum is attained. \(\square\)
:::
