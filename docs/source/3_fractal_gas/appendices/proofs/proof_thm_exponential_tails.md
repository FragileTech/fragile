# Proof: Exponential Tails for the QSD

:::{prf:theorem} Exponential Tails for QSD
:label: proof-thm-exponential-tails

Assume the confining potential and kinetic parameters admit a quadratic Lyapunov function
$V(x,v)$ such that

$$
\mathcal{L}^*[V] \le -\beta V + C
$$

for the adjoint generator $\mathcal{L}^*$, with $\beta>0$ and $C<\infty$. Then the QSD
$\rho_\infty$ satisfies

$$
\rho_\infty(x,v) \le C_0 e^{-\alpha (|x|^2 + |v|^2)}
$$

for some $\alpha, C_0 > 0$.
:::

:::{prf:proof}
Let $W_\theta := e^{\theta V}$ with $\theta>0$ small. A direct chain-rule computation gives

$$
\mathcal{L}^* W_\theta
= \theta W_\theta \mathcal{L}^* V + \frac{\theta^2}{2} W_\theta \|\sigma^\top \nabla_v V\|^2.
$$

Using the quadratic growth of $V$ and the drift bound $\mathcal{L}^* V \le -\beta V + C$,
choose $\theta$ small enough that the $\theta^2$ term is absorbed, yielding

$$
\mathcal{L}^* W_\theta \le -\eta W_\theta + C_\theta
$$

for some $\eta>0$ and $C_\theta<\infty$. Since $\rho_\infty$ is stationary,
$\int \mathcal{L}^* W_\theta \,\rho_\infty = 0$, hence

$$
\int W_\theta \rho_\infty \le \frac{C_\theta}{\eta} < \infty.
$$

Markov's inequality gives exponential decay of the tail probability for $V$.
Finally, the hypoelliptic Harnack inequality and positivity bounds from
{doc}`../11_hk_convergence` localize this integral tail bound to a pointwise
estimate, yielding $\rho_\infty(x,v) \le C_0 e^{-\alpha (|x|^2 + |v|^2)}$
for suitable constants. \(\square\)
:::
