# Proof: Gevrey-1 Classification of the Fitness Potential

:::{prf:corollary} Gevrey-1 Classification (Mean-Field Expected Fitness)
:label: proof-cor-gevrey-1-fitness-potential-full

Assume the C^\infty bound from Theorem {prf:ref}`thm-main-cinf-regularity-fitness-potential-full`
for the **mean-field expected** fitness potential:

$$
\|\nabla^m V_{\mathrm{fit}}\|_\infty \le C_{V,m} \cdot m! \cdot \max(\rho^{-m}, \varepsilon_d^{1-m})
$$

with $C_{V,m} \le C_0 C_1^m$ for some constants $C_0, C_1$ independent of $k$ and $N$. Then the
**mean-field expected** $V_{\mathrm{fit}}$ is Gevrey-1 on every compact set, i.e., there exist
$A,B>0$ such that

$$
\sup_{(x,v) \in K} \|\nabla^m V_{\mathrm{fit}}(x,v)\| \le A B^m m!\quad \text{for all } m\ge 0.
$$
:::

:::{prf:proof}
From the assumed bound,

$$
\|\nabla^m V_{\mathrm{fit}}\|_\infty
\le C_0 C_1^m m! \cdot \max(\rho^{-m}, \varepsilon_d^{1-m}).
$$

Since $\max(\rho^{-m}, \varepsilon_d^{1-m}) \le \max(1,\varepsilon_d)\cdot \max(\rho^{-m}, \varepsilon_d^{-m})$,
we obtain

$$
\|\nabla^m V_{\mathrm{fit}}\|_\infty
\le A \cdot B^m m!,
$$

with $A := C_0 \max(1,\varepsilon_d)$ and $B := C_1 \max(\rho^{-1}, \varepsilon_d^{-1})$. These constants depend only on $(\rho, \varepsilon_d, \varepsilon_c, \eta_{\min}, d)$ and are k- and N-uniform. This is precisely the Gevrey-1 definition.
\(\square\)
:::

:::{note}
Gevrey-1 implies real-analyticity on the interior of the domain. The explicit constants above determine a local radius of analyticity but are not needed for the regularity theory.
:::
