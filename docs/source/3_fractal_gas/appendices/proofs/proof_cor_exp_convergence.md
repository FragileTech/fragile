# Proof: Exponential Convergence to QSD

:::{prf:corollary} Exponential Convergence
:label: proof-cor-exp-convergence

Assume the adaptive system satisfies the discrete Foster-Lyapunov drift

$$
\mathbb{E}[V_{\text{total}}(S_{k+1}) \mid S_k]
\le (1-\kappa_{\text{total}}) V_{\text{total}}(S_k) + C_{\text{total}}
$$

with constants $\kappa_{\text{total}} \in (0,1)$ and $C_{\text{total}} < \infty$. Then

$$
\mathbb{E}[V_{\text{total}}(S_k)]
\le (1-\kappa_{\text{total}})^k \mathbb{E}[V_{\text{total}}(S_0)]
 + \frac{C_{\text{total}}}{\kappa_{\text{total}}}.
$$

In particular, the Lyapunov level converges exponentially fast to the equilibrium level
$C_{\text{total}}/\kappa_{\text{total}}$.
:::

:::{prf:proof}
Set $W_k := \mathbb{E}[V_{\text{total}}(S_k)]$. Taking total expectation in the drift inequality yields

$$
W_{k+1} \le (1-\kappa_{\text{total}}) W_k + C_{\text{total}}.
$$

Let $W_* := C_{\text{total}}/\kappa_{\text{total}}$ and $\delta_k := W_k - W_*$. Then

$$
\delta_{k+1} \le (1-\kappa_{\text{total}})\delta_k.
$$

By induction, $\delta_k \le (1-\kappa_{\text{total}})^k \delta_0$, hence

$$
W_k \le (1-\kappa_{\text{total}})^k W_0 + \frac{C_{\text{total}}}{\kappa_{\text{total}}}.
$$

The QSD existence/uniqueness result in {doc}`../06_convergence` identifies the unique invariant
law for which the Lyapunov level is $W_*$. Therefore the bound gives exponential convergence to
the QSD in the Lyapunov sense. \(\square\)
:::
