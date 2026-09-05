# C³ Regularity of the Fractal Gas Fitness

(sec-gg-c3-regularity)=
## 1. The fitness functions and their derivative coordinates

:::{div} feynman-prose
Move one walker a little and ask how its fitness changes. Three things can move with it: its distance to a companion, the local averages used to compare walkers, and the probabilities of choosing those companions. Keeping these three effects separate makes the calculation manageable.

The main estimate comes from normalization. A weighted average has a denominator that grows with the number of terms. We differentiate that denominator along with the numerator, so the particle count can cancel exactly. A positive distance floor controls collisions, and a positive standard-deviation floor controls populations whose measurements are almost identical. These are different jobs, performed by different parameters.

The results below give explicit bounds through third order, including bounds independent of the number of walkers on specified bounded-distance families. They cover both a fixed realization of the sampled companions and, with the companion-probability derivatives included, the expected sampled fitness. The final sections explain the force estimates and numerical error arguments that these bounds support.
:::

:::{prf:definition} Configuration strata and derivatives
:label: def-c3-configuration-stratum

Fix an alive set $\mathcal A$ with $k\geq2$ walkers, fixed candidate sets, and an open set $O$ of their continuous coordinates. Write $z_j=(x_j,v_j)$ and

$$
s_{j\ell}^2=\|x_j-x_\ell\|^2+
\lambda_{\mathrm{alg}}\|v_j-v_\ell\|^2,
\qquad d_{j\ell}=\sqrt{s_{j\ell}^2+\delta^2},
\qquad \delta=\epsilon_{\mathrm{dist}}>0.
$$

The derivative $D$ means differentiation in the position block $x_i$, with other positions and all velocities fixed. Its norm is the induced multilinear operator norm. The generic normalized calculus below also applies to a different specified coordinate, after establishing its input bounds.

A bound is **$k$-uniform** when the same constant applies to all alive counts in the stated configuration family. It is **$N$-uniform** when independent of the total population size. Uniform block derivative bounds do not by themselves bound the operator norm of a derivative in all $Nd$ position coordinates.

The regularized $d_{j\ell}$ is a smooth positive dissimilarity: $d_{jj}=\delta$. The unregularized $s_{j\ell}$ is a phase-space metric when $\lambda_{\mathrm{alg}}>0$.
:::

:::{prf:definition} Sampled fitness, averaged measurements, and expected fitness
:label: def-c3-fitness-laws

For a companion assignment $c=(c_j)_{j\in\mathcal A}$, $c_j\ne j$, define sampled diversity measurements $d_j^c=d_{j,c_j}$. The reward measurements are $r_j$. For either measurement channel $m=(m_j)$ set

$$
w_{ij}(\rho)=\frac{\exp[-s_{ij}^2/(2\rho^2)]}
{\sum_{\ell\in\mathcal A}\exp[-s_{i\ell}^2/(2\rho^2)]},
\quad
\mu_i[m]=\sum_jw_{ij}m_j,
\quad V_i[m]=\sum_jw_{ij}(m_j-\mu_i[m])^2,
$$

$$
Z_i[m]=\frac{m_i-\mu_i[m]}{\sqrt{V_i[m]+\sigma_{\min}^2}},
\qquad g_A(z)=\frac{A}{1+e^{-z}},
\qquad
F_i^c=(g_A(Z_i[d^c])+\eta)^{\beta_{\mathrm{fit}}}
      (g_A(Z_i[r])+\eta)^{\alpha_{\mathrm{fit}}}.
$$

Here $\rho,\sigma_{\min},A,\eta>0$; $\eta$ is the channel positivity floor. Global statistics use $w_{ij}=1/k$. Including $\delta^2$ in every localization exponent gives the same weights, because its common factor cancels.

For a selected companion scale $\varepsilon_{\mathrm{comp}}>0$, put

$$
P_{j\ell}=\frac{\exp[-s_{j\ell}^2/(2\varepsilon_{\mathrm{comp}}^2)]}
{\sum_{q\ne j}\exp[-s_{jq}^2/(2\varepsilon_{\mathrm{comp}}^2)]},
\qquad
\bar d_j=\sum_{\ell\ne j}P_{j\ell}d_{j\ell}.
$$

The latent algorithm denotes the diversity companion scale by $\epsilon_d$ and the cloning companion scale by $\epsilon_c$. Neither is the distance floor $\delta=\epsilon_{\mathrm{dist}}$.

There are three distinct functions:

1. $F_i^c$, the sampled fitness with $c$ held fixed during differentiation;
2. $\widetilde F_i$, obtained by substituting $\bar d$ for $d^c$ in the nonlinear fitness formula;
3. $\overline F_i=\sum_c p_c F_i^c$, the expectation of the sampled fitness under the actual assignment law $p_c$.

In general $\widetilde F_i\ne\overline F_i$. Derivatives of $\overline F_i$ include derivatives of $p_c$. Independent softmax rows give $p_c=\prod_jP_{j,c_j}$; a pairing law requires its own joint probabilities. These definitions agree with the fitness distinction in {prf:ref}`def-latent-fractal-gas-fitness` and {prf:ref}`rem-mean-field-fitness-field-latent`.
:::

:::{prf:assumption} Measurement bounds on the configuration family
:label: assump-c3-measurement-companion

For uniform quantitative estimates, assume $s_{j\ell}\leq D_0$ throughout $O$, uniformly over the populations considered. The reward channel is $C^3$ and has specified block derivative bounds $R_0^{\mathrm{rew}},\ldots,R_3^{\mathrm{rew}}$. All regularization parameters are fixed positive numbers. Let

$$
D_\delta=\sqrt{D_0^2+\delta^2},\qquad
M_0^d=D_\delta,\quad M_1^d=1,\quad
M_2^d=2\delta^{-1},\quad M_3^d=6\delta^{-2}.
$$

These $M_r^d$ bound every sampled pair measurement with respect to $x_i$. For a phase block $(x_i,v_i)$ multiply the order-$r$ bounds by $\max(1,\sqrt{\lambda_{\mathrm{alg}}})^r$.

Bounded positions together with a hard velocity cap provide one bounded-distance family. On an unbounded state space the general theorem instead requires the summed derivative bounds of {prf:ref}`def-normalized-measurement-bounds` on the family in question. This chapter does not impose compactness on that algorithm.
:::

:::{prf:proof}
For $f(y)=\sqrt{\|y\|^2+b^2}$, $b\geq\delta$,

$$
Df=\frac y f,\qquad
D^2f=\frac I f-\frac{y^{\otimes2}}{f^3},\qquad
D^3f=-\frac{3\operatorname{Sym}(y\otimes I)}{f^3}
       +\frac{3y^{\otimes3}}{f^5}.
$$

Here and throughout, $\operatorname{Sym}$ is the **average** over tensor permutations; its norm is at most the product of factor norms. Since $\|y\|\leq f$ and $f\geq\delta$, the displayed derivative bounds follow. A frozen pair either has no dependence on $x_i$, or has this form with one argument varying. The linear phase-coordinate transformation gives the additional norm factor.
:::

:::{prf:assumption} Localization kernels
:label: assump-c3-kernel

Localization uses positive $C^3$ raw weights $a_j$ on the chosen stratum. Uniform estimates require finite, population-independent values of $B_1,B_2,B_3$ in {prf:ref}`def-normalized-measurement-bounds`. Gaussian weights on the bounded-distance family satisfy this assumption by {prf:ref}`lem-weight-third-derivative`. Uniform weights satisfy it with $B_r=0$.
:::

:::{prf:assumption} Rescaling and positive powers
:label: assump-c3-rescale

A scalar channel rescale $g$ is $C^3$ on the attained score range, with $|g^{(r)}|\leq G_r$ for $r=1,2,3$. For the algorithmic sigmoid $g_A$, one may take $G_1=G_2=G_3=A/4$. Channel values lie in $[\eta,A+\eta]$, so any fixed real fitness exponent has bounded derivatives through order three on this interval.
:::

:::{prf:proof}
Writing $p=(1+e^{-z})^{-1}$ gives

$$
g_A'=Ap(1-p),\quad
g_A''=Ap(1-p)(1-2p),\quad
g_A'''=Ap(1-p)(1-6p+6p^2).
$$

For $0\leq p\leq1$ the last two factors have absolute value at most one, while $p(1-p)\leq1/4$. Although $g_A'(z)>0$ for finite $z$, its infimum on $\mathbb R$ is zero. A bounded increasing function cannot have a globally positive lower derivative: integrating $g'\geq c>0$ would force linear growth. The forward derivative estimates here require only the upper bounds.
:::

:::{prf:assumption} Standard-deviation regularization
:label: assump-c3-patch

Use $q(V)=\sqrt{V+\sigma_{\min}^2}$ for $V\geq0$, with $\sigma_{\min}>0$. More general regularizers are allowed when $q\in C^3$, $q\geq q_{\min}>0$, and its first three derivatives have specified bounds on the attained variance range.
:::

(sec-normalized-measurement-calculus)=
## 2. Normalization, moments, and exact cancellations

:::{div} feynman-prose
Imagine adding a hundred copies of every walker. An average should stay the same. A derivative estimate that grows by a factor of a hundred has lost the normalization somewhere.

The calculation below keeps the numerator and denominator together. We bound the sum of the magnitudes of the weight derivatives directly. That is stronger than counting how many walkers lie within an informal effective radius, and it works even when many walkers occupy the same location. The cancellation of signed derivatives is useful too, but it must be accompanied by a bound on their absolute sum.
:::

:::{prf:definition} Summed derivative bounds
:label: def-normalized-measurement-bounds

On an open set $O$ of configurations with a fixed alive set, let
$a_j:O\to(0,\infty)$ and $d_j:O\to\mathbb R$ be $C^3$ functions indexed by a
finite set $J$. Put $A=\sum_j a_j$ and $w_j=a_j/A$. Derivatives are Fréchet
derivatives in a specified configuration coordinate, with the induced
multilinear operator norm. Assume the following constants are finite:

$$
B_r=\sup_O\frac{\sum_j\|D^r a_j\|}{A},\quad r=1,2,3,
\qquad M_r=\sup_{O,j}\|D^r d_j\|,\quad r=0,1,2,3.
$$

Here $\|D^0d_j\|=|d_j|$. Define

$$
W_0=1,\qquad W_r=2B_r+\sum_{q=1}^{r-1}\binom rq B_qW_{r-q},
$$

and, for $0\leq r\leq3$,

$$
U_r=\sum_{q=0}^r\binom rq W_qM_{r-q},\qquad
H_r=\sum_{q=0}^r\binom rq M_qM_{r-q},\qquad
S_r=\sum_{q=0}^r\binom rq W_qH_{r-q}.
$$

The constants depend on the region $O$, the derivative coordinate and the
kernel scales. Uniformity over particle counts means that these suprema are
bounded by the same constants over the whole specified family of configurations.
:::

:::{prf:lemma} Normalized weights and localized moments
:label: lem-normalized-weight-derivatives

Under {prf:ref}`def-normalized-measurement-bounds`,

$$
\sum_j\|D^rw_j\|\leq W_r,\qquad
\|D^r\mu\|\leq U_r,\qquad
\left\|D^r\sum_jw_jd_j^2\right\|\leq S_r,
\quad \mu=\sum_jw_jd_j.
$$

In particular
$W_1=2B_1$, $W_2=2B_2+4B_1^2$, and
$W_3=2B_3+12B_1B_2+12B_1^3$.
:::

:::{prf:proof}
The identity $Aw_j=a_j$ gives, after $r$ differentiations and symmetrization,

$$
A D^rw_j=D^ra_j-
\sum_{q=1}^r\binom rq\operatorname{Sym}(D^qA\otimes D^{r-q}w_j).
$$

The operator norm of a symmetrized tensor product is at most the product of
the norms. Sum over $j$, divide by $A>0$, and use
$\|D^qA\|\leq\sum_j\|D^qa_j\|\leq AB_q$.
The $q=r$ term uses $\sum_jw_j=1$ and contributes another $B_r$.
Induction therefore proves the recursion for $W_r$. The Leibniz rule for
$\sum_jw_jd_j$ gives $U_r$. Applying the rule first to $d_j^2$ and then to
its weighted sum gives $H_r$ and $S_r$.
:::

:::{prf:lemma} Gradient of the localized variance
:label: lem-variance-gradient

Let $V=\sum_jw_j(d_j-\mu)^2=\sum_jw_jd_j^2-\mu^2$. Then

$$
DV=\sum_jDw_j(d_j-\mu)^2+2\sum_jw_j(d_j-\mu)Dd_j,
\qquad \|DV\|\leq T_1:=S_1+2U_0U_1.
$$
:::

:::{prf:proof}
Differentiate the centered expression. Terms containing $D\mu$ cancel because
$\sum_jw_j(d_j-\mu)=0$. The displayed norm bound follows alternatively by
differentiating $\sum_jw_jd_j^2-\mu^2$ and using the preceding lemma.
:::

:::{prf:lemma} Hessian and third derivative of the localized variance
:label: lem-variance-hessian

For $r=2,3$,

$$
\|D^rV\|\leq T_r:=S_r+
\sum_{q=0}^r\binom rq U_qU_{r-q}.
$$

In particular $T_2=S_2+2U_0U_2+2U_1^2$ and
$T_3=S_3+2U_0U_3+6U_1U_2$.
:::

:::{prf:proof}
Apply the Leibniz rule to $V=\sum_jw_jd_j^2-\mu^2$ and use
{prf:ref}`lem-normalized-weight-derivatives` for each factor. Finiteness of the
index set justifies differentiation term by term.
:::

(sec-support-lem-telescoping-derivatives)=
### 2.1. Cancellation and Gaussian weight bounds

:::{prf:lemma} Derivatives of a partition of unity
:label: lem-telescoping-derivatives

For $r=1,2,3$,

$$
\sum_jD^rw_j=0,\qquad
\sum_j(D^rw_j)d_j=\sum_j(D^rw_j)(d_j-b)
$$

for any scalar $b$ evaluated at the same configuration. Consequently

$$
\left\|\sum_j(D^rw_j)d_j\right\|
\leq W_r\sup_j|d_j-b|.
$$
:::

:::{prf:proof}
:label: proof-lem-telescoping-derivatives

Differentiate the finite identity $\sum_jw_j=1$ exactly $r$ times. Subtract $b\sum_jD^rw_j=0$, then use the summed norm estimate from {prf:ref}`lem-normalized-weight-derivatives`. The subtraction is an identity at each configuration; it does not differentiate $b$.
:::

:::{prf:lemma} Explicit Gaussian weight derivatives
:label: lem-weight-third-derivative

For $a_j=\exp[-s_{ij}^2/(2\rho^2)]$ and $D=\partial_{x_i}$ on the bounded-distance family, set

$$
L_1=\frac{D_0}{\rho^2},\qquad L_2=\rho^{-2},\qquad
E_1=L_1,\quad E_2=L_2+L_1^2,\quad
E_3=3L_1L_2+L_1^3.
$$

Then $B_r\leq E_r$ and

$$
\sum_j\|Dw_{ij}\|\leq 2L_1,\qquad
\sum_j\|D^2w_{ij}\|\leq2L_2+6L_1^2,
$$

$$
\sum_j\|D^3w_{ij}\|
\leq 18L_1L_2+26L_1^3.
$$

These bounds are independent of $k$ and $N$. They apply to all positive Gaussian weights without truncating their tails.
:::

:::{prf:proof}
:label: proof-lem-weight-third-derivative

For $j\ne i$, $\ell_j=\log a_j$ satisfies $\|D\ell_j\|\leq L_1$, $\|D^2\ell_j\|\leq L_2$, and $D^3\ell_j=0$. For $j=i$ all derivatives vanish. Differentiating $a_j=e^{\ell_j}$ gives

$$
Da_j=a_jD\ell_j,\quad
D^2a_j=a_j\bigl(D^2\ell_j+(D\ell_j)^{\otimes2}\bigr),
$$

$$
D^3a_j=a_j\bigl(3\operatorname{Sym}(D\ell_j\otimes D^2\ell_j)
+(D\ell_j)^{\otimes3}\bigr).
$$

Thus $\|D^ra_j\|\leq a_jE_r$. Summing and dividing by $\sum_ja_j$ proves $B_r\leq E_r$. Substitute these bounds in $W_1,W_2,W_3$. In particular
$2E_3+12E_1E_2+12E_1^3=18L_1L_2+26L_1^3$.
:::

:::{prf:corollary} Uniformity from numerator and denominator bounds
:label: cor-normalized-bounded-distance-uniformity

If a family of raw weights satisfies $a_j\geq a_{\min}>0$ and $\|D^ra_j\|\leq A_r$, then $B_r\leq A_r/a_{\min}$ independently of the number of terms. Gaussian weights at a fixed positive scale on a bounded-distance family meet these hypotheses; {prf:ref}`lem-weight-third-derivative` gives a sharper estimate by retaining the factor $a_j$ in each derivative.
:::

:::{prf:proof}
With $m$ terms, the denominator is at least $ma_{\min}$ and the summed numerator at most $mA_r$. Dividing cancels $m$.
:::

:::{div} feynman-prose
The useful quantity is now visible: a derivative of a raw weight divided by the total raw weight. For a Gaussian, each derivative is the original Gaussian times a polynomial in the displacement. The original weights add to the denominator, leaving a bound on that polynomial. This is the entire particle-count cancellation; no estimate of how many neighbors fit inside a ball is needed.
:::

### 2.2. Third derivatives of the moments

:::{prf:lemma} Third derivative of the localized mean
:label: lem-mean-third-derivative

Under {prf:ref}`def-normalized-measurement-bounds`,

$$
\|D^3\mu\|\leq K_{\mu,3}:=
M_3+3W_1M_2+3W_2M_1+W_3M_0=U_3.
$$

This applies to sampled measurements, expected measurements, and the reward channel once their respective $M_r$ have been established.
:::

:::{prf:proof}
:label: proof-lem-mean-third-derivative

The full product rule is

$$
D^3\mu=\sum_j\left[
 w_jD^3d_j+3\operatorname{Sym}(Dw_j\otimes D^2d_j)
 +3\operatorname{Sym}(D^2w_j\otimes Dd_j)+(D^3w_j)d_j\right].
$$

Use $\sum_jw_j=1$ for the first term and $\sum_j\|D^rw_j\|\leq W_r$ for the other terms. Every measurement may depend on $x_i$; the estimate does not discard those derivatives.
:::

:::{prf:lemma} Third derivative of the localized variance
:label: lem-variance-third-derivative

For $V=\sum_jw_jd_j^2-\mu^2$,

$$
\|D^3V\|\leq T_3=S_3+2U_0U_3+6U_1U_2,
$$

where

$$
H_0=M_0^2,\quad H_1=2M_0M_1,\quad
H_2=2M_0M_2+2M_1^2,\quad
H_3=2M_0M_3+6M_1M_2,
$$

$$
S_3=H_3+3W_1H_2+3W_2H_1+W_3H_0.
$$
:::

:::{prf:proof}
:label: proof-lem-variance-third-derivative

For any scalar $u$,

$$
D^3(u^2)=2uD^3u+6\operatorname{Sym}(Du\otimes D^2u).
$$

There is no $(Du)^{\otimes3}$ term, since the third derivative of the square function is zero. Applying this formula to $d_j$ gives $H_3$; the lower-order product rules give $H_0,H_1,H_2$. The third product rule for $\sum_jw_jd_j^2$ gives $S_3$. Apply the same square formula to $\mu$ and add the norm bounds.
:::

(sec-c3-companion-derivatives)=
## 3. Differentiating companion selection

:::{div} feynman-prose
When walker $i$ moves, a different walker $j$ sees one candidate move. Only one raw Gaussian in row $j$ changes. But the normalization changes too, so every probability in that row can change. This is the precise meaning of locality here.

For walker $i$'s own row, every distance can change. We handle that row by the normalized-average estimate already proved. For the other rows, we retain the probability of selecting $i$ as a factor. Summing that factor over walkers will let us differentiate the expectation of the entire sampled fitness without paying an extra factor of the population size.
:::

:::{prf:lemma} Off-diagonal companion derivatives
:label: lem-derivative-locality-c3

Fix $j\ne i$ and define $q_{ji}=s_{ji}^2/(2\varepsilon_{\mathrm{comp}}^2)$. Then

$$
DP_{j\ell}=P_{j\ell}(P_{ji}-\mathbf1_{\{\ell=i\}})Dq_{ji},
$$

$$
D\bar d_j=P_{ji}\left[Dd_{ji}+(\bar d_j-d_{ji})Dq_{ji}\right]
=P_{ji}\left[1+\frac{d_{ji}(\bar d_j-d_{ji})}
{\varepsilon_{\mathrm{comp}}^2}\right]Dd_{ji}.
$$

Let $E_r^{\mathrm{comp}}$ be the Gaussian bounds of {prf:ref}`lem-weight-third-derivative` at scale $\varepsilon_{\mathrm{comp}}$, and let $C_r$ be the corresponding $W_r$. Then, pointwise,

$$
\sum_{\ell\ne j}\|D^rP_{j\ell}\|\leq P_{ji}C_r,
\qquad r=1,2,3.
$$
:::

:::{prf:proof}
:label: proof-lem-derivative-locality-c3

Only the raw weight $a_{ji}$ varies in row $j$. Differentiate its quotient by the row sum to obtain the first identity. In $D\bar d_j$, the direct measurement derivative occurs only at $\ell=i$. Summing the probability derivatives gives
$P_{ji}(\bar d_j-d_{ji})Dq_{ji}$, proving the second identity. The relation $Dq_{ji}=d_{ji}Dd_{ji}/\varepsilon_{\mathrm{comp}}^2$ gives its alternative form.

At a fixed configuration, the raw derivative ratios satisfy
$B_{r,j}\leq P_{ji}E_r^{\mathrm{comp}}$. The proof of {prf:ref}`lem-normalized-weight-derivatives` is pointwise. Its recurrence bounds the normalized order-$r$ derivative sum by

$$
2P_{ji}E_r^{\mathrm{comp}}+
\sum_{q=1}^{r-1}\binom rq
(P_{ji}E_q^{\mathrm{comp}})(P_{ji}C_{r-q})
\leq P_{ji}C_r,
$$

using $0\leq P_{ji}\leq1$ and induction. This proves the asserted bounds for all three orders.
:::

:::{prf:lemma} Self-row and expected-measurement bounds
:label: lem-self-measurement-derivatives

For the self row,

$$
D\bar d_i=\mathbb E_i[Dd_{i\ell}]
-\operatorname{Cov}_i(d_{i\ell},Dq_{i\ell}),
\qquad q_{i\ell}=s_{i\ell}^2/(2\varepsilon_{\mathrm{comp}}^2),
$$

where expectation is under $P_{i\ell}$. Every row, including $j=i$, satisfies

$$
\|D^r\bar d_j\|\leq
\overline M_r:=\sum_{q=0}^r\binom rq C_qM_{r-q}^d,
\qquad C_0=1,\quad 0\leq r\leq3.
$$

In particular
$\overline M_3=M_3^d+3C_1M_2^d+3C_2M_1^d+C_3M_0^d$.
:::

:::{prf:proof}
:label: proof-lem-self-measurement-derivatives

For the self row,
$DP_{i\ell}=-P_{i\ell}(Dq_{i\ell}-\mathbb E_iDq_{i\ell})$.
Insert this formula into $D\sum_\ell P_{i\ell}d_{i\ell}$ to obtain the covariance identity. Each raw weight has order-$r$ derivative bounded by $a_{i\ell}E_r^{\mathrm{comp}}$. Thus its row derivative sums are at most $C_r$. The same bound applies to the off-diagonal rows by the preceding lemma. Finally differentiate the finite weighted mean through order $r$ and apply the Leibniz rule. This retains all terms involving derivatives of both the measurement and its probability.
:::

:::{prf:lemma} Derivatives of the full independent companion law
:label: lem-c3-joint-companion-law

Assume every other alive walker is a candidate and $s_{j\ell}\leq D_0$. Put

$$
a_*:=e^{-D_0^2/(2\varepsilon_{\mathrm{comp}}^2)},\qquad
S_r^{\mathrm{comp}}:=C_r(1+a_*^{-1}),
$$

$$
J_0=1,\quad J_1=S_1^{\mathrm{comp}},\quad
J_2=S_2^{\mathrm{comp}}+(S_1^{\mathrm{comp}})^2,
$$

$$
J_3=S_3^{\mathrm{comp}}+3S_1^{\mathrm{comp}}S_2^{\mathrm{comp}}
+(S_1^{\mathrm{comp}})^3.
$$

For independent row choices $p_c=\prod_jP_{j,c_j}$,

$$
\sum_c\|D^rp_c\|\leq J_r,\qquad 0\leq r\leq3.
$$

If $\|D^rF_i^c\|\leq F_r$ uniformly in assignments and configurations, then

$$
\left\|D^r\overline F_i\right\|
\leq\sum_{q=0}^r\binom rq J_qF_{r-q}.
$$

All constants are independent of $k$ and $N$.
:::

:::{prf:proof}
Each row denominator is at least $(k-1)a_*$, so
$P_{ji}\leq[(k-1)a_*]^{-1}$. Let
$A_{r,j}=\sum_\ell\|D^rP_{j\ell}\|$. The self row has $A_{r,i}\leq C_r$, and the other rows have $A_{r,j}\leq C_rP_{ji}$. Hence

$$
\sum_jA_{r,j}\leq C_r\left(1+\sum_{j\ne i}P_{ji}\right)
\leq C_r(1+a_*^{-1}).
$$

Differentiate the product defining $p_c$ and then sum its norm over all assignments. Every undifferentiated row sums to one. At order one, the result is at most $\sum_jA_{1,j}$. At order two it is at most

$$
\sum_jA_{2,j}+\sum_{j\ne\ell}A_{1,j}A_{1,\ell}
\leq S_2^{\mathrm{comp}}+(S_1^{\mathrm{comp}})^2.
$$

At order three there are one-row terms, two-row terms with coefficient three, and three-distinct-row terms. Dropping distinctness restrictions in these nonnegative sums bounds them by $S_3^{\mathrm{comp}}$, $3S_1^{\mathrm{comp}}S_2^{\mathrm{comp}}$, and $(S_1^{\mathrm{comp}})^3$. This proves the $J_r$ estimates. The final bound is the Leibniz rule for the finite sum $\sum_cp_cF_i^c$.
:::

:::{prf:remark} Candidate restrictions and pairing laws
:label: rem-c3-pairing-scope

The fixed-assignment bounds and the normalized row calculus apply to any fixed nonempty candidate sets. For count-uniform derivatives of the joint assignment law, the proof above uses a bound on $\sum_{j\ne i}P_{ji}$. Restricted graphs must supply their own bound on this sum. Dependent pairings must supply $\sum_c\|D^rp_c\|$ directly. Smoothness of a finite pairing law follows when each branch probability is $C^3$ on a fixed stratum, but independence cannot be used for that law.
:::

(sec-c3-score-fitness)=
## 4. From variance to the full fitness

:::{div} feynman-prose
Standardization asks how far a measurement sits from its local mean, measured in units of its local spread. If the spread is nearly zero, those units become very small. The parameter $\sigma_{\min}$ puts a lower limit on the unit of measurement.

That lower limit enters more strongly when we differentiate several times. Differentiating an inverse square root once produces a third power in the denominator; three derivatives can produce a seventh power. Keeping these powers explicit tells us why an extremely small regularizer can create large derivatives even when the original fitness values remain bounded.
:::

:::{prf:lemma} Third-order scalar composition
:label: lem-patch-chain-rule

For scalar $V\in C^3$ and $q\in C^3$ on its range,

$$
D^3(q\circ V)=q'''(V)(DV)^{\otimes3}
+3q''(V)\operatorname{Sym}(DV\otimes D^2V)+q'(V)D^3V.
$$

If $|q^{(r)}|\leq L_r$ and $\|D^rV\|\leq T_r$, then
$\|D^3(q\circ V)\|\leq L_3T_1^3+3L_2T_1T_2+L_1T_3$.
:::

:::{prf:proof}
:label: proof-lem-patch-chain-rule

Differentiate $D(q\circ V)=q'(V)DV$ to get
$D^2(q\circ V)=q''(V)(DV)^{\otimes2}+q'(V)D^2V$.
Differentiating the first term yields one term with $q'''$ and two permutations with $q''$. Differentiating the second yields the third permutation with $q''$ and the term with $q'$. Collect the three permutations using the averaged symmetrization convention, then take norms.
:::

:::{prf:lemma} Derivatives of the regularized standard deviation
:label: lem-patch-third-derivative

Let $q(V)=\sqrt{V+\sigma_{\min}^2}$. Then

$$
|q'|\leq\frac1{2\sigma_{\min}},\quad
|q''|\leq\frac1{4\sigma_{\min}^3},\quad
|q'''|\leq\frac3{8\sigma_{\min}^5},
$$

and

$$
\|D^3q(V)\|\leq
\frac{T_3}{2\sigma_{\min}}
+\frac{3T_1T_2}{4\sigma_{\min}^3}
+\frac{3T_1^3}{8\sigma_{\min}^5}.
$$
:::

:::{prf:proof}
:label: proof-lem-patch-third-derivative

Differentiate the scalar square root three times. Its derivatives are
$\tfrac12(V+\sigma_{\min}^2)^{-1/2}$,
$-\tfrac14(V+\sigma_{\min}^2)^{-3/2}$, and
$\tfrac38(V+\sigma_{\min}^2)^{-5/2}$.
Use $V\geq0$ and {prf:ref}`lem-patch-chain-rule`.
:::

:::{prf:lemma} Regularized inverse deviation and Z-score
:label: lem-normalized-zscore-bounds

Let $\sigma_{\min}>0$, $h=(V+\sigma_{\min}^2)^{-1/2}$, and $Z=(d_i-\mu)h$, with the same
measurement derivative bounds for $d_i$. Set

$$
R_0=\sigma_{\min}^{-1},\qquad R_1=\frac{T_1}{2\sigma_{\min}^3},\qquad
R_2=\frac{T_2}{2\sigma_{\min}^3}+\frac{3T_1^2}{4\sigma_{\min}^5},
$$

$$
R_3=\frac{T_3}{2\sigma_{\min}^3}+\frac{9T_1T_2}{4\sigma_{\min}^5}
+\frac{15T_1^3}{8\sigma_{\min}^7},\qquad
P_r=\sum_{q=0}^r\binom rq(M_q+U_q)R_{r-q}.
$$

Then $\|D^rh\|\leq R_r$ and $\|D^rZ\|\leq P_r$ for $0\leq r\leq3$.
:::

:::{prf:proof}
Variance is nonnegative. For $F(t)=(t+\sigma_{\min}^2)^{-1/2}$ on $t\geq0$,

$$
|F'|\leq\frac1{2\sigma_{\min}^3},\quad
|F''|\leq\frac3{4\sigma_{\min}^5},\quad
|F'''|\leq\frac{15}{8\sigma_{\min}^7}.
$$

The chain rule gives
$Dh=F'(V)DV$,
$D^2h=F''(V)(DV)^{\otimes2}+F'(V)D^2V$, and

$$
D^3h=F'''(V)(DV)^{\otimes3}
+3F''(V)\operatorname{Sym}(DV\otimes D^2V)+F'(V)D^3V.
$$

Insert the variance bounds to obtain $R_r$. Finally the Leibniz rule for
$(d_i-\mu)h$ gives $P_r$.
:::

:::{prf:theorem} First derivative of a rescaled channel
:label: thm-c1-regularity

If $g\in C^3$ with $|g^{(r)}|\leq G_r$ on the attained Z-score range, then
$C_{\mathrm{channel}}=g(Z)$ satisfies
$\|DC_{\mathrm{channel}}\|\leq G_1P_1$ under
{prf:ref}`def-normalized-measurement-bounds`.
:::

:::{prf:proof}
The chain rule gives $DC_{\mathrm{channel}}=g'(Z)DZ$. Apply the bounds on $g'$ and
$DZ$ from {prf:ref}`lem-normalized-zscore-bounds`.
:::

:::{prf:theorem} Second and third derivatives of a rescaled channel
:label: thm-c2-regularity

Under the preceding hypotheses,

$$
\|D^2C_{\mathrm{channel}}\|\leq G_2P_1^2+G_1P_2,\qquad
\|D^3C_{\mathrm{channel}}\|\leq G_3P_1^3+3G_2P_1P_2+G_1P_3.
$$

All bounds are uniform over particle counts when the input bounds are uniform.
:::

:::{prf:proof}
Differentiate $g(Z)$ twice and three times. The resulting tensors are
$g''(Z)DZ^{\otimes2}+g'(Z)D^2Z$ and
$g'''(Z)DZ^{\otimes3}+3g''(Z)\operatorname{Sym}(DZ\otimes D^2Z)+g'(Z)D^3Z$.
The norm estimates follow term by term. No lower bound on $g'$ is used.
:::


:::{prf:lemma} Complete third-derivative quotient bound
:label: lem-zscore-third-derivative

The score $Z=(d_i-\mu)/q(V)$ satisfies $\|D^3Z\|\leq P_3$ with the explicit $P_3$ of {prf:ref}`lem-normalized-zscore-bounds` when $q(V)=\sqrt{V+\sigma_{\min}^2}$.

For a general positive denominator $v\geq v_*>0$ and numerator $u$, suppose $\|D^ru\|\leq u_r$, $\|D^rv\|\leq v_r$. Then the complete quotient estimate is

$$
\begin{aligned}
\|D^3(u/v)\|\leq{}&\frac{u_3}{v_*}
+\frac{3u_2v_1+3u_1v_2+u_0v_3}{v_*^2}\\
&+\frac{6u_1v_1^2+6u_0v_1v_2}{v_*^3}
+\frac{6u_0v_1^3}{v_*^4}.
\end{aligned}
$$

For $u=d_i-\mu$, one may take $u_r=M_r+U_r$, including $u_0\leq2M_0$.
:::

:::{prf:proof}
:label: proof-lem-zscore-third-derivative

The first assertion was proved by differentiating the inverse square root directly. For the general quotient, let $b=1/v$. Its derivatives are

$$
Db=-v^{-2}Dv,\quad
D^2b=2v^{-3}(Dv)^{\otimes2}-v^{-2}D^2v,
$$

$$
D^3b=-6v^{-4}(Dv)^{\otimes3}
+6v^{-3}\operatorname{Sym}(Dv\otimes D^2v)-v^{-2}D^3v.
$$

Insert these into
$D^3(ub)=bD^3u+3\operatorname{Sym}(D^2u\otimes Db)
+3\operatorname{Sym}(Du\otimes D^2b)+uD^3b$.
The triangle inequality gives all seven displayed contributions. No reciprocal power is absorbed into a constant declared independent of $v_*$.
:::

:::{div} feynman-prose
We have now bounded one standardized channel. The algorithm multiplies two positive channels after raising them to their fitness exponents. The positivity floor $\eta$ makes that final step smooth even for noninteger exponents. It is separate from $\sigma_{\min}$: one protects the powers, the other protects division by the spread.
:::

:::{prf:theorem} C³ regularity of sampled, surrogate, and expected fitness
:label: thm-c3-regularity

On a fixed configuration stratum, suppose the two measurement channels and the localization weights satisfy {prf:ref}`def-normalized-measurement-bounds`, and the rescale and regularizer satisfy {prf:ref}`assump-c3-rescale` and {prf:ref}`assump-c3-patch` with the square-root choice. Then each fixed-assignment fitness $F_i^c$ is $C^3$.

For a channel $a\in\{d,r\}$ let its score bounds be $P_1^a,P_2^a,P_3^a$ and define

$$
K_0^a=A+\eta,\quad K_1^a=G_1P_1^a,\quad
K_2^a=G_2(P_1^a)^2+G_1P_2^a,
$$

$$
K_3^a=G_3(P_1^a)^3+3G_2P_1^aP_2^a+G_1P_3^a.
$$

For a real exponent $p$, write

$$
Q_{p,\ell}=\max_{\eta\leq t\leq A+\eta}
\left|\frac{d^\ell}{dt^\ell}t^p\right|,
\qquad 0\leq\ell\leq3,
$$

and define the powered-channel bounds

$$
B_0^{a,p}=Q_{p,0},\quad B_1^{a,p}=Q_{p,1}K_1^a,
\quad B_2^{a,p}=Q_{p,2}(K_1^a)^2+Q_{p,1}K_2^a,
$$

$$
B_3^{a,p}=Q_{p,3}(K_1^a)^3
+3Q_{p,2}K_1^aK_2^a+Q_{p,1}K_3^a.
$$

Then

$$
\|D^rF_i^c\|\leq F_r:=
\sum_{q=0}^r\binom rq
B_q^{d,\beta_{\mathrm{fit}}}B_{r-q}^{r,\alpha_{\mathrm{fit}}},
\qquad 0\leq r\leq3.
$$

The same result applies to $\widetilde F_i$ using the expected-measurement bounds $\overline M_r$ of {prf:ref}`lem-self-measurement-derivatives`. For independent complete softmax rows on the bounded-distance family, the actual expectation $\overline F_i$ is $C^3$ and

$$
\|D^r\overline F_i\|\leq
\sum_{q=0}^r\binom rqJ_qF_{r-q}.
$$

The bounded-distance, bounded-reward-derivative hypotheses of {prf:ref}`assump-c3-measurement-companion` make these estimates uniform in $k,N$ and, for the sampled bounds, in $c$. For a general family the conclusion is uniform precisely when the displayed input and joint-law bounds are uniform. For the scalar channel alone, its third-derivative bound is $K_3^a$.
:::

:::{prf:proof}
:label: proof-thm-c3-regularity

Positive raw weights have a positive finite denominator, so their normalized weights are $C^3$. Finite weighted sums and products give $C^3$ moments. The regularized variance stays in the domain $V+\sigma_{\min}^2>0$, so its inverse square root and the score are $C^3$. The derivative estimates for these steps are {prf:ref}`lem-normalized-weight-derivatives`, {prf:ref}`lem-variance-third-derivative`, and {prf:ref}`lem-normalized-zscore-bounds`.

Compose each score with $g_A$ and add $\eta$. The chain rule gives the stated $K_r^a$. Each channel lies in $[\eta,A+\eta]$, where $t^p$ and its first three derivatives are bounded by $Q_{p,\ell}$. A second application of the chain rule gives $B_r^{a,p}$. The Leibniz rule for the product of the two powered channels gives $F_r$.

For frozen sampled distances the measurement bounds are independent of the assignment, even if many walkers choose $i$. Their sum occurs inside the normalized moment estimates. For $\widetilde F_i$, replace those bounds by $\overline M_r$ and repeat the same calculation. For $\overline F_i$, differentiate its finite expectation and use {prf:ref}`lem-c3-joint-companion-law`. Each constant in the bounded-distance case depends only on the diameter, reward bounds, fixed scales, exponents, and regularization floors. None depends on $k$ or $N$.
:::

:::{prf:proposition} Explicit computable third-derivative constants
:label: prop-explicit-k-v-3

For a specified measurement channel, the following finite recursion gives its constant $K_{V,3}^{\mathrm{channel}}$:

$$
\begin{gathered}
(B_1,B_2,B_3),(M_0,M_1,M_2,M_3)
\longmapsto (W_r,U_r,H_r,S_r)_{r\leq3}\\
\longmapsto (T_1,T_2,T_3)
\longmapsto (R_0,R_1,R_2,R_3),(P_0,P_1,P_2,P_3)\\
\longmapsto K_{V,3}^{\mathrm{channel}}
=G_3P_1^3+3G_2P_1P_2+G_1P_3.
\end{gathered}
$$

The definitions appear in {prf:ref}`def-normalized-measurement-bounds` and {prf:ref}`lem-normalized-zscore-bounds`. For the algorithmic full fitness take $K_{V,3}^{\mathrm{sampled}}=F_3$; for its independent-companion expectation take

$$
K_{V,3}^{\mathrm{expected}}=F_3+3J_1F_2+3J_2F_1+J_3F_0.
$$

These are upper bounds, with their full parameter dependence retained. In particular a scalar-channel bound is not substituted for a full-fitness bound.
:::

:::{prf:proof}
Each arrow is the explicit inequality proved above. Composing inequalities with nonnegative coefficients preserves them. The final formula is the order-three product bound from {prf:ref}`lem-c3-joint-companion-law`.
:::

(sec-c3-counts-scaling)=
## 5. Neighbor counts and parameter scaling

:::{div} feynman-prose
A probability density describes a fraction of the population. If a ball contains one percent of that probability, a swarm of a thousand walkers can put about ten walkers there. A swarm of a million can put about ten thousand there. The count has a factor of the population size.

This distinction matters when reading a localization argument. A short-range kernel can make distant walkers contribute little, but it does not place a fixed limit on the number of nearby walkers. The derivative proof above survives that distinction because it works with normalized weights. Here we keep the geometric counts for interpretation and state exactly what they imply.
:::

:::{prf:definition} Effective radii and geometric neighbor counts
:label: def-effective-counts-two-scales

For constants $C_{\mathrm{comp}},C_\rho>0$, define

$$
R_{\mathrm{eff}}^{(\varepsilon_{\mathrm{comp}})}
=\varepsilon_{\mathrm{comp}}
\sqrt{C_{\mathrm{comp}}^2+2\log(k^2)},\qquad
R_{\mathrm{eff}}^{(\rho)}=C_\rho\rho,
$$

$$
k_{\mathrm{eff}}^{(\varepsilon_{\mathrm{comp}})}(i)
=\#\{j\ne i:s_{ij}\leq R_{\mathrm{eff}}^{(\varepsilon_{\mathrm{comp}})}\},
\quad
k_{\mathrm{eff}}^{(\rho)}(i)
=\#\{j\in\mathcal A:s_{ij}\leq R_{\mathrm{eff}}^{(\rho)}\}.
$$

These are configuration-dependent integer counts. Both can be of order $k$, including at fixed positive scales.
:::

:::{prf:notation} Scale superscripts
:label: notation-keff-superscripts

The superscripts distinguish companion selection from localization. The notation $k_{\mathrm{eff}}^{(\epsilon_c)}$ is the companion count when $\varepsilon_{\mathrm{comp}}=\epsilon_c$; the diversity scale uses $\epsilon_d$ instead. A superscript specifies a scale, not a population-uniformity property.
:::

:::{prf:lemma} Density bounds and Gaussian tail mass
:label: lem-c3-count-density-tail

In rescaled phase coordinates $y=(x,\sqrt{\lambda_{\mathrm{alg}}}v)\in\mathbb R^{2d}$, suppose the conditional density of each $Y_j$ given $Y_i$, $j\ne i$, is bounded by $\rho_{\max}$. Then

$$
\mathbb E\#\{j:\|Y_j-Y_i\|\leq R\}
\leq1+(k-1)\rho_{\max}\omega_{2d}R^{2d},
\qquad \omega_{2d}=\frac{\pi^d}{\Gamma(d+1)}.
$$

For a deterministic center $y$, marginal density bounds suffice and the bound is $k\rho_{\max}\omega_{2d}R^{2d}$. If the density is instead taken relative to $dx\,dv$, the ball volume is $\lambda_{\mathrm{alg}}^{-d/2}\omega_{2d}R^{2d}$ when $\lambda_{\mathrm{alg}}>0$.

For a fixed companion row with partition function $Z_i^{\mathrm{comp}}=\sum_{j\ne i}e^{-s_{ij}^2/(2\varepsilon_{\mathrm{comp}}^2)}$,

$$
\mathbb P_i(s_{i,c_i}>R)
\leq\frac{(k-1)e^{-R^2/(2\varepsilon_{\mathrm{comp}}^2)}}
{Z_i^{\mathrm{comp}}}.
$$

If at least one candidate has $s_{ij}\leq C_{\mathrm{comp}}\varepsilon_{\mathrm{comp}}$, then at $R=R_{\mathrm{eff}}^{(\varepsilon_{\mathrm{comp}})}$ the last bound is at most $(k-1)/k^2\leq1/k$.
:::

:::{prf:proof}
Condition on $Y_i$, integrate the conditional density over its radius-$R$ ball, and sum the resulting bounds over $j\ne i$. The center contributes one. For a deterministic center the same argument uses each marginal density directly. The linear change of velocity coordinates gives the stated Jacobian. For the tail estimate, each numerator term outside the ball is at most $e^{-R^2/(2\varepsilon_{\mathrm{comp}}^2)}$, with at most $k-1$ terms. A candidate inside $C_{\mathrm{comp}}\varepsilon_{\mathrm{comp}}$ supplies the denominator lower bound $e^{-C_{\mathrm{comp}}^2/2}$. Substitution of the effective radius proves the final estimate.
:::

:::{prf:remark} What density control establishes
:label: rem-c3-count-uniformity

The count estimate is an expectation bound with a factor $k-1$. Markov's inequality can convert it into a probability bound; it cannot give a deterministic count bound. Bounded one-particle densities alone also do not justify the random-center estimate: if every walker equals the same random variable with bounded density, all $k$ walkers lie in every positive-radius ball centered at a walker.

Gaussian tail mass controls the probability assigned to distant companions. It gives no population-independent bound on the number of nearby companions.
:::

:::{prf:proposition} Bandwidth dependence of the derivative estimates
:label: prop-scaling-kv3

At fixed $D_0$, the Gaussian weight bounds are

$$
W_1\leq\frac{2D_0}{\rho^2},\qquad
W_2\leq\frac2{\rho^2}+\frac{6D_0^2}{\rho^4},\qquad
W_3\leq\frac{18D_0}{\rho^4}+\frac{26D_0^3}{\rho^6}.
$$

For fixed measurement bounds and positive regularizers, their substitution in {prf:ref}`prop-explicit-k-v-3` gives an upper bound $K_{V,3}=O(\rho^{-6})$ as $\rho\downarrow0$. This is a conservative bound, not an asymptotic equality for the derivative.

A stronger scaled-moment hypothesis gives the familiar $O(\rho^{-3})$ bound: if the localization weights satisfy

$$
\sum_jw_{ij}s_{ij}^q\leq C_q\rho^q,\qquad q=1,2,3,
$$

uniformly over the family, then $B_r=O(\rho^{-r})$, $W_r=O(\rho^{-r})$, and $K_{V,3}=O(\rho^{-3})$ for fixed measurement bounds. The moment hypothesis must be established for that family.
:::

:::{prf:proof}
The first display is {prf:ref}`lem-weight-third-derivative`. For $0<\rho\leq1$, it gives $W_r=O(\rho^{-2r})$. Every order-$r$ term in the moment, variance, inverse-deviation, and chain-rule formulas is a product whose derivative orders add to $r$. Hence its power is at most $\rho^{-2r}$. The same observation applies to the final powers and product of the two channels. At $r=3$ this proves the first scaling assertion.

Under the scaled-moment hypothesis, the raw Gaussian derivative identities instead give

$$
B_1\leq\frac{\sum_jw_{ij}s_{ij}}{\rho^2},\quad
B_2\leq\rho^{-2}+\frac{\sum_jw_{ij}s_{ij}^2}{\rho^4},
$$

$$
B_3\leq\frac{3\sum_jw_{ij}s_{ij}}{\rho^4}
+\frac{\sum_jw_{ij}s_{ij}^3}{\rho^6}.
$$

Thus $B_r=O(\rho^{-r})$. The normalized recurrence and the same order-counting argument prove the improved estimate.
:::

:::{prf:example} Normalization on an unbounded family
:label: ex-c3-unbounded-weight-derivative

In one dimension, place the reference walker at $0$ and the remaining $m=k-1>1$ walkers at $R_m=\rho\sqrt{2\log m}$. With localization including the reference itself, its raw weight is $1$ and each other raw weight is $1/m$. At this configuration,

$$
w_{ii}=\frac12,\qquad
\frac{\partial w_{ii}}{\partial x_i}=-\frac{R_m}{4\rho^2},\qquad
\sum_j\left|\frac{\partial w_{ij}}{\partial x_i}\right|
=\frac{R_m}{2\rho^2}.
$$

Indeed the total raw weight is $2$, its derivative is $R_m/\rho^2$, and each nonreference normalized derivative is $R_m/(4m\rho^2)$. The summed derivative diverges as $m\to\infty$ at fixed $\rho$, although $\sum_jDw_{ij}=0$ exactly. This configuration family has no fixed diameter bound.
:::

:::{prf:remark} Global statistics and varying parameters
:label: rem-c3-global-limit

On bounded configurations, $\rho\to\infty$ makes $w_{ij}\to1/k$ and all positive-order weight derivatives tend to zero. The remaining derivatives still include the measurement mean, its variance, and the nonlinear score and rescale. They need not reduce to the third derivative of the individual measurement. Uniform estimates over a parameter schedule require uniform bounds on the displayed constants; positive parameters tending to zero generally do not provide them.
:::

(sec-c3-force-numerics)=
## 6. Force estimates and numerical consequences

:::{div} feynman-prose
A third derivative of the fitness measures how quickly its curvature changes. That gives a direct control on the error made when replacing the force by a local linear approximation. This is a concrete use of the regularity estimate.

A second-order numerical method needs an additional argument. Its successive substeps must cancel the lower-order errors, and the remaining local error must stay bounded on the states the process visits. Smooth fitness helps with those estimates. The splitting, the other coefficients, the test functions, and any boundary operations must enter the calculation as well.
:::

:::{prf:corollary} Regularity of the adaptive force
:label: cor-smooth-perturbation

Let a specified fitness $F$ satisfy $\|D^rF\|\leq K_r$ for $r=1,2,3$ on a convex position region. The adaptive force $b_{\mathrm{adapt}}=\epsilon_F\nabla F$ is $C^2$ and

$$
\|b_{\mathrm{adapt}}\|\leq\epsilon_FK_1,\quad
\|Db_{\mathrm{adapt}}\|\leq\epsilon_FK_2,\quad
\|D^2b_{\mathrm{adapt}}\|\leq\epsilon_FK_3.
$$

Moreover

$$
\|D^2F(x+h)-D^2F(x)\|\leq K_3\|h\|,
$$

$$
\|\nabla F(x+h)-\nabla F(x)-D^2F(x)h\|
\leq\tfrac12K_3\|h\|^2,
$$

whenever the segment lies in the region. A $C^3$ fitness gives a $C^2$ force; a $C^3$ force requires another derivative.
:::

:::{prf:proof}
:label: proof-cor-smooth-perturbation

Differentiate $b_{\mathrm{adapt}}=\epsilon_F\nabla F$ zero, one, and two times. Integrating $D^3F$ along the segment gives the Hessian Lipschitz bound. Integrate that bound once more in the identity

$$
\nabla F(x+h)-\nabla F(x)-D^2F(x)h
=\int_0^1[D^2F(x+th)-D^2F(x)]h\,dt
$$

to obtain the factor $\int_0^1t\,dt=1/2$.
:::

:::{prf:corollary} Regularity of normalized Lyapunov observables and drift transfer
:label: cor-lyapunov-c3

If $\|D^3U\|\leq K_U$ on a position region, the normalized observable

$$
\mathcal V_N=1+\frac1N\sum_iU(x_i)
+\frac{a}{N}\sum_i\|v_i\|^2
+\frac{b}{N^2}\sum_{i,j}\|x_i-x_j\|^2
$$

has $\|D^3\mathcal V_N\|\leq K_U/N$ in the full Euclidean configuration operator norm. The quadratic terms have zero third derivative. This statement concerns this explicitly defined observable; a boundary barrier or an added population-dependent term requires its own regularity bounds.

For a nonnegative observable $\mathcal V$, let $P_h$ be a reference kernel and $Q_h$ a numerical kernel. If

$$
P_h\mathcal V\leq r_h\mathcal V+b_h,\qquad
|(Q_h-P_h)\mathcal V|\leq e_h\mathcal V+c_h,
$$

then

$$
Q_h\mathcal V\leq(r_h+e_h)\mathcal V+b_h+c_h.
$$

The contraction survives when $r_h+e_h<1$.
:::

:::{prf:proof}
:label: proof-cor-lyapunov-c3

The third derivative applied to configuration directions $u,v,w$ is
$N^{-1}\sum_iD^3U(x_i)[u_i,v_i,w_i]$. For unit full-configuration directions,

$$
\sum_i\|u_i\|\|v_i\|\|w_i\|
\leq\|u\|_2\|v\|_2\max_i\|w_i\|\leq1.
$$

This proves the derivative estimate. The drift transfer follows by writing
$Q_h\mathcal V=P_h\mathcal V+(Q_h-P_h)\mathcal V$ and inserting the two hypotheses. Regularity alone supplies neither the reference drift nor the numerical defect bound.
:::

:::{prf:corollary} Second-order weak accuracy from a local splitting estimate
:label: cor-baoab-validity

Let $P_h$ be an exact semigroup step and $Q_h$ a numerical step on test functions, with $P_{nh}=P_h^n$. Fix $T>0$ and a norm $\|\cdot\|$ for which $\|Q_h^j\|\leq e^{cjh}$. Suppose for the propagated test functions $P_h^j\varphi$, $jh\leq T$, a proved local estimate gives

$$
\|(Q_h-P_h)P_h^j\varphi\|\leq C_{T,\varphi}h^3.
$$

Then, for $nh\leq T$,

$$
\|(Q_h^n-P_{nh})\varphi\|
\leq T e^{cT}C_{T,\varphi}h^2
$$

when $c\geq0$. This theorem applies to BAOAB or Boris–BAOAB once their actual substeps satisfy these hypotheses. It is a finite-time weak error result for the specified kernels. Stationary-law error and conditioned QSD error require additional long-time and survival estimates.
:::

:::{prf:proof}
:label: proof-cor-baoab-validity

Use the noncommutative telescoping identity

$$
Q_h^n-P_h^n=\sum_{j=0}^{n-1}
Q_h^{n-1-j}(Q_h-P_h)P_h^j.
$$

Apply it to $\varphi$, then use the local estimate and the bound on powers of $Q_h$. Each term is at most $e^{cT}C_{T,\varphi}h^3$. Summing $n\leq T/h$ terms proves the result.
:::

:::{prf:remark} Inputs required for the split integrator
:label: rem-c3-baoab-inputs

The local $O(h^3)$ estimate in {prf:ref}`cor-baoab-validity` requires a calculation for the actual generator and splitting. Its hypotheses include enough regularity of the coefficients and propagated test functions to justify the expansion, control of the resulting moments, and consistency of force, noise, and time normalization. State-dependent noise, velocity caps, Boris rotations, killing, and cloning must be included whenever they belong to the kernel being compared. A $C^3$ fitness alone does not prove that local estimate or a global weak order.

For the exact Ornstein–Uhlenbeck substep, friction is integrated as $e^{-\gamma h}$; a restriction $h<1/(2\gamma)$ does not follow from that substep. Curvature can restrict other substeps. For example $U(x)=\omega^2x^2/2$ has $D^3U=0$, while the deterministic Verlet part has amplification trace $2-h^2\omega^2$ and determinant one. Its eigenvalues leave the unit circle for $h\omega>2$. Thus a stability rule depending only on $\|D^3U\|$ cannot be valid.

See {prf:ref}`rem-parameter-step-size` for the algorithm's step-size discussion and {doc}`06_convergence` for component drift and killed-kernel estimates.
:::

:::{prf:proposition} A displacement criterion for force linearization
:label: prop-timestep-constraint

Suppose $\|D^3F\|\leq K_3$ and a displacement has $\|\Delta x\|\leq h v_{\max}$ along a segment in the regularity region. To keep the error in the linear approximation to $\epsilon_F\nabla F$ below a chosen force tolerance $\zeta>0$, it suffices that

$$
\frac{\epsilon_FK_3}{2}h^2v_{\max}^2\leq\zeta.
$$

When $\epsilon_FK_3v_{\max}^2>0$, this becomes
$h\leq\sqrt{2\zeta/(\epsilon_FK_3v_{\max}^2)}$.
It is an accuracy criterion for this particular approximation, not a general BAOAB stability or weak-order condition. With additional unbounded position noise, the deterministic displacement premise must be replaced by an appropriate moment or probability bound.
:::

:::{prf:proof}
:label: proof-prop-timestep-constraint

Multiply the gradient remainder in {prf:ref}`cor-smooth-perturbation` by $\epsilon_F$, substitute $\|\Delta x\|\leq hv_{\max}$, and solve the resulting scalar inequality.
:::

:::{prf:corollary} Derivative hierarchy and its uses
:label: cor-regularity-hierarchy

For each fitness covered by {prf:ref}`thm-c3-regularity`, there are established bounds through order three:

| Quantity | Available bound | Direct consequence |
|---|---|---|
| Sampled fitness $F_i^c$ | $F_0$ | Bounded observable |
| Its gradient | $F_1$ | Bounded adaptive force |
| Its Hessian | $F_2$ | Lipschitz adaptive force on convex regions |
| Its third derivative | $F_3$ | Lipschitz Hessian and quadratic force remainder |
| Expected sampled fitness | $\sum_{q=0}^r\binom rqJ_qF_{r-q}$ | Same conclusions under the joint-law hypotheses |

The QSD conclusions in {doc}`06_convergence`, the functional inequalities in {doc}`15_kl_convergence`, and the mean-field results in {doc}`09_propagation_chaos` use their own analytic hypotheses in addition to these coefficient estimates.
:::

:::{prf:proof}
The bounds are those of {prf:ref}`thm-c3-regularity`. Integrating the Hessian or third derivative along a segment gives the stated Lipschitz and remainder consequences, as in {prf:ref}`cor-smooth-perturbation`.
:::

(sec-c3-continuity-implementation)=
## 7. Continuity and implementation conventions

:::{div} feynman-prose
There is one final distinction to keep in mind. The fitness formula can vary smoothly while an algorithm makes a discrete decision. A walker changes its alive status, a companion index is drawn, or a clipping threshold is crossed. The formula between such events and the rule selecting those events are separate mathematical objects.

The theorem describes the formula on a fixed stratum and the finite expectation under a smooth probability law. This tells us exactly which derivatives an implementation can compare with the analytical bounds, and which changes require a transition-kernel argument instead.
:::

:::{prf:theorem} Joint continuity of third derivatives
:label: thm-continuity-third-derivatives

Fix $N$, an alive set, candidate sets, and any discrete branch data used in the measurements. Suppose the input measurements and raw weights are jointly $C^3$ in their continuous state variables and the parameters being varied, with positive kernel scales and regularization floors. Then the block derivatives $D^3F_i^c$ and $D^3\widetilde F_i$ are jointly continuous in the configuration and those parameters.

If the finite assignment probabilities $p_c$ are also jointly $C^3$, the same holds for $D^3\overline F_i$. Each third-derivative map is uniformly continuous on every compact subset of this open stratum. This compact-subset statement is local and does not assume that the full state space is compact.
:::

:::{prf:proof}
:label: proof-thm-continuity-third-derivatives

The raw weight sums are positive continuous functions. On a compact subset of the stated stratum they have positive minima. The positive standard-deviation and channel floors also have positive minima there. Finite sums, products, and division by these positive denominators preserve $C^3$ regularity, as do the square root, sigmoid, and channel powers on their specified domains. The explicit third-derivative formulas are therefore finite sums of products of continuous input derivatives and continuous reciprocal powers.

This proves joint continuity for the fixed-assignment and averaged-measurement formulas. A finite sum $\sum_cp_cF_i^c$ with jointly $C^3$ probabilities has the same property by the product rule. Continuous maps on compact sets are uniformly continuous.
:::

:::{prf:remark} Continuity, Hölder estimates, and higher derivatives
:label: rem-c3-higher-regularity

A continuous third derivative need not be Hölder continuous with any positive exponent. A quantitative $C^{3,\alpha}$ conclusion requires $C^{3,\alpha}$ input bounds; a fourth-derivative bound gives a Lipschitz third derivative on convex regions. If every input is $C^\infty$ and all denominators remain positive, the finite-stratum formulas are $C^\infty$ by repeated composition. Population-uniform bounds at every order and factorial growth estimates require the corresponding normalized derivative estimates developed in {doc}`14_b_geometric_gas_cinf_regularity_full`.
:::

:::{prf:definition} Regularized standard deviation used by the implementation
:label: def-reg-std-implementation

The implementation in `src/fragile/fractalai/core/fitness.py` uses

$$
\sigma_{\mathrm{reg}}(V)=\sqrt{V+\texttt{sigma\_min}^2},
\qquad \texttt{sigma\_min}>0,
$$

for both reward and diversity statistics. The code default is `sigma_min = 1e-8`. The distance computation independently uses `epsilon_dist = 1e-8` inside the pair-distance square root. These are numerical parameter defaults, not lower bounds required by a convergence theorem.

A parameterization $\sigma_{\min}^2=\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2$ describes the same mathematical regularizer after setting `sigma_min` to that square root; it is not a second variance-floor operation in the current formula. For example $\kappa_{\mathrm{var,min}}=10^{-6}$ and $\varepsilon_{\mathrm{std}}=10^{-4}$ give $\sigma_{\min}=\sqrt{1.01\times10^{-6}}\approx1.005\times10^{-3}$.

For $r=1,2,3$, the scalar standard-deviation derivative bounds are respectively

$$
\frac1{2\sigma_{\min}},\qquad
\frac1{4\sigma_{\min}^3},\qquad
\frac3{8\sigma_{\min}^5}.
$$

The inverse standard deviation has the different bounds used in {prf:ref}`lem-normalized-zscore-bounds`.
:::

:::{prf:remark} Derivative settings and discrete branches
:label: rem-c3-implementation-branches

The implementation computes sampled distances and the full multiplicative fitness in {prf:ref}`def-c3-fitness-laws`. Its companion indices and alive mask are held fixed during differentiation. If rewards are supplied as external fixed values, their state derivatives are zero; differentiation through a reward field requires that field's derivative bounds.

When `detach_stats` freezes the mean and standard deviation, automatic differentiation computes the derivative of the formula with those statistics held fixed. It does not compute the total derivative of the state-dependent statistics analyzed in the full normalized calculus. Both derivatives can be studied by stating which inputs are frozen.

The fixed-stratum theorem applies between changes of alive masks, candidate sets, hard cutoff branches, and periodic minimum-image choices. A minimum-image squared distance is only piecewise smooth at an image-switching surface. Hard cloning thresholds, spectral clamps, and velocity caps require separate regularity checks; the $C^3$ fitness theorem does not turn the entire update map into a $C^3$ map.

For a latent metric depending on the state, establish the corresponding raw-weight and measurement derivative ratios in {prf:ref}`def-normalized-measurement-bounds`. The Gaussian quadratic-distance constants in {prf:ref}`lem-weight-third-derivative` apply to the stated fixed phase metric. The algorithm and its distinct latent hypotheses are specified in {doc}`../1_the_algorithm/02_fractal_gas_latent`.
:::
