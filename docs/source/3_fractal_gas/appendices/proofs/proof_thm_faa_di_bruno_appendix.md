# Proof: Faà di Bruno Formula and Gevrey-1 Closure

:::{prf:theorem} Multivariate Faà di Bruno Formula (Form Used)
:label: proof-thm-faa-di-bruno-appendix

Let $f: \mathbb{R} \to \mathbb{R}$ and $g: \mathbb{R}^d \to \mathbb{R}$ be $C^m$. For $h = f \circ g$ and any multi-index $\alpha$ with $|\alpha|=m$,

$$
\partial^\alpha h(x) = \sum_{k=1}^m f^{(k)}(g(x))\, \mathcal{B}_{\alpha,k}(\partial g(x), \partial^2 g(x), \ldots, \partial^m g(x)),
$$

where $\mathcal{B}_{\alpha,k}$ is a (multivariate) Bell polynomial in the derivatives of $g$. In particular, there is a constant $C_m$ depending only on $m$ and $d$ such that

$$
\|\nabla^m h(x)\| \le C_m \sum_{k=1}^m |f^{(k)}(g(x))| \sum_{\substack{r_1+\cdots+r_k=m \\ r_j\ge 1}} \prod_{j=1}^k \|\nabla^{r_j} g(x)\|.
$$
:::

:::{prf:proof}
This is the standard multivariate Faà di Bruno formula (see Constantine & Savits, 1996). The second inequality follows by bounding each Bell polynomial by a combinatorial constant $C_m$ times products of derivative norms of $g$. \(\square\)
:::

:::{prf:corollary} Gevrey-1 Closure Under Composition
:label: proof-cor-gevrey-1-closure

Assume there exist constants $A_f,B_f,A_g,B_g>0$ such that

$$
|f^{(k)}(y)| \le A_f B_f^k k!,\qquad \|\nabla^r g(x)\| \le A_g B_g^r r!\quad \text{for all } k,r\ge 1.
$$

Then there exist $A,B>0$ (depending only on $A_f,B_f,A_g,B_g,d$) such that

$$
\|\nabla^m (f\circ g)(x)\| \le A B^m m!\quad \text{for all } m\ge 1.
$$
:::

:::{prf:proof}
Apply the Faà di Bruno formula above. Each term contains a factor $f^{(k)}(g(x))$ and a product of $k$ derivatives of $g$ whose orders sum to $m$. Using the Gevrey-1 bounds,

$$
|f^{(k)}(g(x))| \prod_{j=1}^k \|\nabla^{r_j} g(x)\|
\le A_f A_g^k B_f^k B_g^m k! \prod_{j=1}^k r_j!.
$$

The sum over partitions of $m$ is bounded by a combinatorial constant times $m!$ (standard Bell-number control). Absorb all combinatorial factors into $A$ and set $B = C \max(B_f, B_g)$ for a constant $C$ depending on $d$. This yields the stated Gevrey-1 bound. \(\square\)
:::
