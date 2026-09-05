# Smooth and Analytic Regularity of the Fractal Gas Fitness

(sec-gg-cinf-regularity)=
## 1. Derivative bounds at every order

:::{div} feynman-prose
The third-derivative calculation showed how normalization controls the dependence on the number of walkers. We now ask whether that calculation can continue indefinitely, and how fast its constants grow.

These are two questions. A function can have derivatives of every order while those derivatives grow too quickly for its Taylor series to recover the function. An analytic function has the stronger bound: the order-$n$ derivative grows at most like a fixed exponential times $n!$. The factorial counts the ways derivatives can be distributed; the exponential determines the radius within which the Taylor series works.

We will keep both quantities visible. The proof follows the actual fitness calculation: normalized weights, measurements, moments, standardization, and the two positive fitness channels. Companion sampling needs an additional calculation. Independent softmax, idealized matching, and sequential greedy pairing have different probability laws, and each law receives its own proof.
:::

:::{prf:definition} Derivative coordinate and fitness convention
:label: def-cinf-objects

Use the fixed configuration strata and the three fitness functions of {prf:ref}`def-c3-configuration-stratum` and {prf:ref}`def-c3-fitness-laws`. In particular,

$$
s_{j\ell}^2=\|x_j-x_\ell\|^2+
\lambda_{\mathrm{alg}}\|v_j-v_\ell\|^2,
\qquad d_{j\ell}=\sqrt{s_{j\ell}^2+\delta^2},
\qquad \delta=\epsilon_{\mathrm{dist}}>0.
$$

The derivative $D$ is the Fréchet derivative in one specified position block $x_i$, with its multilinear operator norm. A phase block or a learned metric requires the corresponding input derivative bounds. Uniformity refers to this specified coordinate, not automatically to the full configuration operator norm.

For a fixed companion assignment $c$, including a singleton self-assignment when the chosen pairing rule uses one, the algorithmic fitness is

$$
F_i^c=(g_A(Z_i[d^c])+\eta)^{\beta_{\mathrm{fit}}}
(g_A(Z_i[r])+\eta)^{\alpha_{\mathrm{fit}}},
\quad d_j^c=d_{j,c_j},\quad g_A(z)=\frac{A}{1+e^{-z}},
$$

$$
Z_i[m]=\frac{m_i-\sum_jw_{ij}m_j}
{\sqrt{\sum_jw_{ij}(m_j-\sum_\ell w_{i\ell}m_\ell)^2+\sigma_{\min}^2}},
\qquad w_{ij}=\frac{e^{-s_{ij}^2/(2\rho^2)}}{\sum_\ell e^{-s_{i\ell}^2/(2\rho^2)}}.
$$

Here $A,\eta,\sigma_{\min},\rho>0$. The diversity companion scale is $\epsilon_d$, the cloning companion scale is $\epsilon_c$, and the distance floor is $\delta$. Global statistics use $w_{ij}=1/k$.

Write $\widetilde F_i$ for the same formula evaluated at expected measurements, and $\overline F_i=\sum_c p_cF_i^c$ for the expectation of the sampled fitness. These generally differ. This chapter first treats finite populations. Mean-field passage is stated separately in {ref}`sec-cinf-mean-field-parameters`.
:::

:::{prf:definition} Smooth bounds and Gevrey-1 bounds
:label: def-cinf-majorants

A family is uniformly $C^\infty$ in the chosen coordinate when every derivative has a finite family-wide bound $M_n$. It has **Gevrey-1 bounds** when

$$
M_n\leq C B^n n!,\qquad n\geq0,
$$

for fixed $C,B<\infty$. Population-uniformity means that these constants are independent of $k,N$ on the specified configuration family.

For a function with derivative bounds $M_n$, use the nonnegative formal series

$$
\mathcal M(t)=\sum_{n\geq0}\frac{M_n}{n!}t^n,
\qquad \mathcal M_+(t)=\mathcal M(t)-M_0.
$$

The notation $\preceq$ means coefficientwise domination. Product estimates become products of these series because the Leibniz coefficient satisfies
$\binom nq q!(n-q)!=n!$. A convergent majorant gives a Gevrey-1 bound: if $\mathcal M(t_*)\leq C_*$ at $t_*>0$, then $M_n\leq C_*n!t_*^{-n}$.
:::

(sec-support-thm-faa-di-bruno-appendix)=
### 1.1. The chain rule and its factorial count

:::{prf:theorem} Faà di Bruno formula
:label: thm-faa-di-bruno-appendix

Let $f$ be scalar-valued and $g$ scalar- or vector-valued, with $n$ derivatives. For directions $h_1,\ldots,h_n$,

$$
D^n(f\circ g)[h_1,\ldots,h_n]
=\sum_{\pi\in\mathcal P_n}
D^{|\pi|}f(g)\left[
D^{|B|}g[h_j:j\in B]:B\in\pi\right],
$$

where $\mathcal P_n$ is the set of partitions of $\{1,\ldots,n\}$. For scalar $g$, a partition with $m_j$ blocks of size $j$ contributes to the norm bound with multiplicity

$$
\frac{n!}{\prod_{j=1}^n m_j!(j!)^{m_j}},
\qquad \sum_jjm_j=n,\quad |\pi|=\sum_jm_j.
$$
:::

:::{prf:proof}
:label: proof-thm-faa-di-bruno-appendix

The first-order formula is the chain rule. Suppose the partition formula holds at order $n$. Differentiating the outer derivative of $f$ creates a new singleton block $\{n+1\}$. Differentiating the factor belonging to a block $B$ inserts $n+1$ into that block. Every partition of $\{1,\ldots,n+1\}$ has exactly one of these forms, determined by the block containing $n+1$. This proves the formula by induction.

To count partitions with prescribed block sizes, first order the $n$ elements, divide them into blocks, and then remove the orders within each block and among equal-sized blocks. This divides $n!$ by $\prod_j(j!)^{m_j}m_j!$, proving the multiplicity.
:::

:::{prf:corollary} Explicit Gevrey-1 composition bound
:label: cor-gevrey-closure

Suppose, for positive-order derivatives,

$$
\|D^qf\|\leq C_f B_f^q q!,\qquad
\|D^jg\|\leq C_g B_g^j j!.
$$

Then, for $n\geq1$,

$$
\|D^n(f\circ g)\|
\leq C_f B_f C_g\,n! B_g^n(1+B_fC_g)^{n-1}.
$$

For $g=(g_1,\ldots,g_q)$, one may take $C_g=\sum_aC_a$ and $B_g=\max_aB_a$ from component bounds. The radius changes under composition; smoothness alone does not give this estimate.
:::

:::{prf:proof}
:label: proof-cor-gevrey-closure

Put $a=B_fC_g$. Substituting the derivative bounds in the partition formula cancels every factor $(j!)^{m_j}$. Partitions with $q$ blocks therefore contribute at most

$$
C_f n! B_g^n a^q
\sum_{\substack{\sum m_j=q\\\sum jm_j=n}}\frac{q!}{\prod_jm_j!}
=C_f n! B_g^n a^q\binom{n-1}{q-1}.
$$

The identity counts ordered compositions of $n$ into $q$ positive parts, first by multiplicities and then by choosing $q-1$ cuts among $n-1$ gaps. Summing over $q$ gives
$\sum_{q=1}^na^q\binom{n-1}{q-1}=a(1+a)^{n-1}$.
For vector $g$, its derivative norm is at most the sum of the component derivative norms, and the same partition proof applies to the multilinear derivatives of $f$.
:::

:::{div} feynman-prose
The cancellation in that proof is why one factorial survives. Counting all partitions first and then multiplying by the largest derivative bound would count the same combinatorial growth twice. Dividing each derivative bound by its factorial keeps track of the cancellations automatically.
:::

### 1.2. Normalization at every order

:::{prf:theorem} All-order normalized derivative calculus
:label: thm-cinf-normalized-majorant

Let $a_j>0$ be $C^\infty$, $A_0=\sum_ja_j$, and $w_j=a_j/A_0$, with a fixed finite index set. Suppose

$$
B_n\geq\sup\frac{\sum_j\|D^na_j\|}{A_0},\qquad n\geq1.
$$

Define $W_0=1$ and

$$
W_n=2B_n+\sum_{q=1}^{n-1}\binom nq B_qW_{n-q}.
$$

Then $\sum_j\|D^nw_j\|\leq W_n$. With
$\mathcal B(t)=\sum_{n\geq1}B_nt^n/n!$, their majorant is

$$
\mathcal W(t)=\sum_{n\geq0}\frac{W_n}{n!}t^n
=\frac{1+\mathcal B(t)}{1-\mathcal B(t)}.
$$

If $B_n\leq b n!R^{-n}$, then with $t_*=R/[2(b+1)]$,

$$
\sum_j\|D^nw_j\|\leq3n!t_*^{-n},\qquad n\geq0.
$$

The same statements hold for positive kernel densities normalized against a fixed measure, replacing sums by integrals, provided differentiation under the integral is justified.
:::

:::{prf:proof}
Differentiate $A_0w_j=a_j$ $n$ times, sum norms over $j$, and divide by $A_0$. The derivatives falling entirely on $a_j$ and entirely on $A_0$ each contribute $B_n$. The remaining terms give the recurrence, exactly as in {prf:ref}`lem-normalized-weight-derivatives`.

After division by $n!$, the convolution recurrence reads
$\mathcal W-1=2\mathcal B+\mathcal B(\mathcal W-1)$, proving the series identity. Under the geometric bound,
$\mathcal B(t_*)\leq b/(2b+1)<1/2$ for $b>0$, so $\mathcal W(t_*)\leq3$. Nonnegative coefficients yield the displayed factorial estimate. If $b=0$, the weights have zero positive-order derivatives. The integral proof uses the same product identity and the triangle inequality under the integral.
:::

:::{prf:lemma} Relative-weight majorants
:label: lem-cinf-relative-majorants

Suppose every raw weight satisfies, at each base configuration,

$$
\sum_{n\geq1}\frac{\|D^na_j\|}{a_j n!}t^n\preceq u(t),
\qquad u(0)=0.
$$

Then its normalized probability has majorant

$$
\sum_{n\geq0}\frac{\|D^nw_j\|}{n!}t^n
\preceq w_j\frac{1+u(t)}{1-u(t)}.
$$

If only one weight $a_i$ varies in a row and its base probability is $p=w_i$, then

$$
\sum_{n\geq0}\frac{\|D^nw_\ell\|}{n!}t^n
\preceq\begin{cases}
 w_\ell/(1-pu),&\ell\ne i,\\
 p(1+u)/(1-pu),&\ell=i.
\end{cases}
$$
:::

:::{prf:proof}
At a base point, write the normalized denominator as its positive base value times $1+e$. Its positive-order derivative series is bounded by $u$, or by $pu$ in the one-varying-weight case. Differentiating $(1+e)^{-1}$, equivalently solving the coefficient recurrence for its reciprocal, gives the nonnegative majorant $\sum_{q\geq0}u^q=(1-u)^{-1}$. Multiply by the numerator majorant $a_j(1+u)$, or by the constant numerator for $\ell\ne i$, and divide by the base denominator. This is a coefficientwise argument about derivatives; convergence is needed only when evaluating the majorants at a positive $t$.
:::

(sec-cinf-distance-weights)=
## 2. Distance regularization and Gaussian weights

:::{div} feynman-prose
The square root in a distance has a singularity when its argument reaches zero. Adding $\delta^2$ moves every real configuration away from that singularity. We can measure the resulting margin and turn it into derivative bounds at every order.

Gaussian weights are simpler than they first appear. Their logarithm is quadratic in the moving walker's position. Its derivatives stop at order two. All higher derivatives of the Gaussian come from exponentiating those two terms, which gives an explicit majorant rather than an unspecified combinatorial constant.
:::

:::{prf:lemma} All-order regularized-distance bounds
:label: lem-dalg-derivative-bounds-full

For $d_{j\ell}=\sqrt{s_{j\ell}^2+\delta^2}$, $\delta>0$, the pair measurement is $C^\infty$. In a varying position argument,

$$
Dd_{j\ell}=\frac{x_j-x_\ell}{d_{j\ell}},\qquad
D^2d_{j\ell}=\frac I{d_{j\ell}}-
\frac{(x_j-x_\ell)^{\otimes2}}{d_{j\ell}^3},
$$

up to the sign of the first derivative when the second argument varies. Thus $\|Dd_{j\ell}\|\leq1$, $\|D^2d_{j\ell}\|\leq\delta^{-1}$, and for every $n\geq1$,

$$
\|D^nd_{j\ell}\|\leq4^n n!\delta^{1-n}.
$$

These positive-order bounds hold without a diameter restriction. On a family with $s_{j\ell}\leq D_0$, one may use the measurement majorant

$$
\mathcal M_d(t)=D_\delta+\frac{4t}{1-4t/\delta},
\qquad D_\delta=\sqrt{D_0^2+\delta^2},\quad 0\leq t<\delta/4.
$$
:::

:::{prf:proof}
:label: proof-lem-dalg-derivative-bounds-full

The first two formulas follow by differentiation. The Hessian eigenvalues lie between zero and $1/d_{j\ell}$, proving its bound.

For the all-order calculation, fix a base displacement $r$ and put $a=d_{j\ell}\geq\delta$. Under an increment $h$ in one position argument,

$$
d(r+h)=a\sqrt{1+\frac{2r\cdot h+\|h\|^2}{a^2}}.
$$

The positive-order derivative coefficients of the inner increment are bounded by $2t/a+t^2/a^2$, since $\|r\|\leq a$. The absolute coefficients of $\sqrt{1+z}-1$ are those of $1-\sqrt{1-z}$. The chain-rule norm majorant for $d(r+h)-a$ is therefore

$$
a\left[1-\sqrt{1-(2t/a+t^2/a^2)}\right].
$$

At $t=a/4$ its value is less than $a$, so every normalized coefficient is at most $a(4/a)^n$. Hence $\|D^nd\|\leq4^nn!a^{1-n}\leq4^nn!\delta^{1-n}$. Summing the latter estimates and using the diameter bound at order zero gives $\mathcal M_d$.
:::

:::{prf:property} Pairwise derivative locality
:label: prop-dalg-locality

The raw distance $d_{j\ell}$ depends only on walkers $j,\ell$. All positive-order derivatives in $x_i$ vanish when $i\notin\{j,\ell\}$. For a frozen assignment, each sampled measurement consequently has the preceding derivative bounds, regardless of how many walkers select $i$.

*Proof.* The formula contains no coordinate of any other walker. If both arguments coincide by a singleton self-assignment, $d_{jj}=\delta$ is constant.
:::

:::{prf:lemma} Gaussian relative derivative majorant
:label: lem-gaussian-kernel-derivatives-full

On a family with $s_{j\ell}\leq D_0$, let
$K_s(j,\ell)=e^{-s_{j\ell}^2/(2s^2)}$ at a fixed positive scale $s$. Define

$$
L_1(s)=D_0/s^2,\qquad L_2(s)=s^{-2},\qquad
u_s(t)=\exp[L_1(s)t+L_2(s)t^2/2]-1.
$$

Then

$$
\frac{\|D^nK_s(j,\ell)\|}{K_s(j,\ell)n!}
\leq[t^n](1+u_s(t)),\qquad n\geq0.
$$

The same bound holds for a raw matching weight when only one edge involving the differentiated walker occurs in that matching. A common factor $e^{-\delta^2/(2s^2)}$ has no effect on normalized probabilities.
:::

:::{prf:proof}
:label: proof-lem-gaussian-kernel-derivatives-full

The logarithm $\ell=\log K_s$ has $\|D\ell\|\leq L_1$, $\|D^2\ell\|\leq L_2$, and $D^n\ell=0$ for $n\geq3$. The partition formula for $e^\ell$ therefore has blocks of sizes one and two only. Dividing by $K_s n!$ gives exactly the coefficients of $\exp(L_1t+L_2t^2/2)$. For a matching, every other edge is independent of $x_i$ and factors out of the derivative ratio.
:::

:::{prf:lemma} Companion availability with its count factor
:label: lem-companion-availability-enforcement

If every other alive walker is a candidate and $s_{j\ell}\leq D_0$, then for $k\geq2$,

$$
(k-1)a_s\leq Z_j(s):=\sum_{\ell\ne j}K_s(j,\ell)\leq k-1,
\qquad a_s=e^{-D_0^2/(2s^2)}.
$$

Thus $P_{j\ell}\leq[(k-1)a_s]^{-1}$. A greedy substep with $m$ available candidates has the bound $Z\geq ma_s$. These are statements on configurations where the candidates exist; extinction or a singleton does not supply a companion.
:::

:::{prf:proof}
:label: proof-lem-companion-availability-enforcement

Every summand lies in $[a_s,1]$. Count the $k-1$ summands, or the $m$ candidates at a greedy substep, and divide the numerator bound $K_s\leq1$ by the denominator lower bound.
:::

:::{prf:lemma} Localization-weight bounds at every order
:label: lem-localization-weight-derivatives-full

For the Gaussian localization weights at scale $\rho$, put

$$
\mathcal H_\rho(t)=\frac{1+u_\rho(t)}{1-u_\rho(t)}.
$$

Then

$$
\sum_j\|D^nw_{ij}\|\leq n![t^n]\mathcal H_\rho(t).
$$

For any $t_\rho>0$ satisfying
$D_0t_\rho/\rho^2+t_\rho^2/(2\rho^2)\leq\log(3/2)$,

$$
\sum_j\|D^nw_{ij}\|\leq3n!t_\rho^{-n}.
$$

The constants are independent of population size. For unbounded families, use the general normalized-ratio hypothesis of {prf:ref}`thm-cinf-normalized-majorant` instead of assuming a diameter bound.
:::

:::{prf:proof}
:label: proof-lem-localization-weight-derivatives-full

The Gaussian relative majorant bounds the summed raw derivative ratios by the same $u_\rho$. Apply {prf:ref}`thm-cinf-normalized-majorant` or sum {prf:ref}`lem-cinf-relative-majorants`. At $t_\rho$ the input majorant is at most $1/2$, so $\mathcal H_\rho(t_\rho)\leq3$. Positivity of its coefficients gives the factorial bound.
:::

:::{prf:lemma} Exact telescoping for all derivative orders
:label: lem-telescoping-localization-weights-full

For $n\geq1$,

$$
\sum_jD^nw_{ij}=0,\qquad
\sum_j(D^nw_{ij})m_j=\sum_j(D^nw_{ij})(m_j-b)
$$

for any scalar $b$ at the same configuration.
:::

:::{prf:proof}
:label: proof-lem-telescoping-localization-weights-full

Differentiate the finite identity $\sum_jw_{ij}=1$ exactly $n$ times. Multiplying the resulting zero sum by $b$ proves the second identity. To estimate its norm one still uses the summed absolute derivative bound; cancellation of a signed sum alone does not bound that norm.
:::

(sec-cinf-companion-laws)=
## 3. Independent sampling and the two matching laws

:::{div} feynman-prose
A matching does more than select a nearby companion. Once two walkers are paired, both leave the candidate pool. That changes the probabilities of all later pairs, so an idealized probability over complete matchings need not reproduce the sequential algorithm.

For regularity, we can follow a single moving walker through the sequence. Until it is paired, only the edge connecting it to the current pivot changes. Once it is removed, the remaining pairing probabilities no longer depend on its position. This gives an induction over the actual sampling history and avoids multiplying a worst-case bound once for every pair.
:::

:::{prf:remark} Three probability laws
:label: note-dual-mechanism-framework

Independent softmax draws, sequential greedy pairing, and the idealized matching distribution are distinct laws. The code exposes independent selection and greedy pairing in `src/fragile/fractalai/core/companion_selection.py`. The proofs below establish regularity for each law directly. A shared kernel and shared regularity class do not identify their distributions or their mean-field equations.
:::

### 3.1. Independent softmax rows

:::{prf:lemma} Relative softmax derivative bounds
:label: lem-softmax-derivative-locality-full

For $P_{j\ell}=K_s(j,\ell)/\sum_{q\ne j}K_s(j,q)$ on the bounded-distance family, let $u=u_s$ and $\mathcal H=(1+u)/(1-u)$. Then

$$
\|D^nP_{j\ell}\|\leq P_{j\ell}n![t^n]\mathcal H(t),
\qquad n\geq0.
$$

Thus probability derivatives retain the factor $P_{j\ell}$, with explicit scale and diameter dependence. For $j\ne i$ there is the stronger summed estimate

$$
\sum_\ell\|D^nP_{j\ell}\|
\leq P_{ji}n![t^n](\mathcal H(t)-1),\qquad n\geq1.
$$
:::

:::{prf:proof}
:label: proof-lem-softmax-derivative-locality-full

Apply {prf:ref}`lem-cinf-relative-majorants` to all weights for the first assertion. For $j\ne i$, only $K_s(j,i)$ varies. With $p=P_{ji}$, summing its two probability majorants gives

$$
\frac{1+pu}{1-pu}=1+\frac{2pu}{1-pu}
\preceq1+p\frac{2u}{1-u}=1+p(\mathcal H-1).
$$

Read the positive-order coefficients to obtain the stated sum bound.
:::

:::{prf:lemma} The softmax Jacobian reduction
:label: lem-softmax-jacobian-reduction

For $j\ne i$, with $\ell_{ji}=\log K_s(j,i)$,

$$
DP_{j\ell}=P_{j\ell}(\mathbf1_{\{\ell=i\}}-P_{ji})D\ell_{ji},
$$

$$
D\bar d_j=P_{ji}Dd_{ji}+P_{ji}(d_{ji}-\bar d_j)D\ell_{ji},
\qquad \bar d_j=\sum_\ell P_{j\ell}d_{j\ell}.
$$

Every normalized probability may change; only one raw weight changes.
:::

:::{prf:proof}
Differentiate the quotient using $D\sum_\ell K_s(j,\ell)=DK_s(j,i)$. In $D\bar d_j$, only $d_{ji}$ has a direct measurement derivative. Summing the probability-derivative terms gives $P_{ji}(d_{ji}-\bar d_j)D\ell_{ji}$.
:::

:::{prf:lemma} Derivatives of an expected companion measurement
:label: lem-derivatives-companion-distance-full

For every row, the complete order-$n$ derivative is

$$
D^n\bar d_j=\sum_\ell\sum_{q=0}^n\binom nq
\operatorname{Sym}(D^qP_{j\ell}\otimes D^{n-q}d_{j\ell}).
$$

This formula describes the expected measurement, while a fixed sampled measurement is differentiated with its companion held fixed.
:::

:::{prf:proof}
:label: proof-lem-derivatives-companion-distance-full

Apply the Leibniz rule to every term of the finite sum $\sum_\ell P_{j\ell}d_{j\ell}$. The symmetrization is the average over permutations, so its norm is bounded by the product of the factor norms.
:::

:::{prf:lemma} Off-diagonal measurement majorant
:label: lem-companion-measurement-derivatives-full

For $j\ne i$ define

$$
\mathcal E(t)=\frac{(1+u_s(t))\mathcal M_{d,+}(t)
+2D_\delta u_s(t)}{1-u_s(t)}.
$$

Then

$$
\|D^n\bar d_j\|\leq P_{ji}n![t^n]\mathcal E(t),
\qquad n\geq1.
$$

In particular these derivatives have population-uniform Gevrey-1 bounds at every positive $t$ where $\mathcal E(t)$ is finite. Both the distance floor and the companion scale occur in the displayed majorant.
:::

:::{prf:proof}
:label: proof-lem-companion-measurement-derivatives-full

At a fixed base point let $p=P_{ji}$, write the relative change in $K_s(j,i)$ as $e$, and the change in $d_{ji}$ as $\Delta d$. All other raw row weights and measurements remain fixed. Algebra gives

$$
\bar d_j(x+h)-\bar d_j(x)
=\frac{p[(1+e)\Delta d+(d_{ji}-\bar d_j)e]}{1+pe}.
$$

The derivative majorants of $e$ and $\Delta d$ are $u_s$ and $\mathcal M_{d,+}$. Also $|d_{ji}-\bar d_j|\leq2D_\delta$. Use the reciprocal majorant $(1-pu_s)^{-1}\preceq(1-u_s)^{-1}$ and the product rule. This proves the coefficientwise estimate without discarding any probability derivative.
:::

:::{prf:lemma} Self-row and uniform expected-measurement bounds
:label: lem-self-measurement-derivatives-full

All expected softmax measurements, including the self row, satisfy

$$
\|D^n\bar d_j\|\leq n![t^n]
\left(\mathcal H_s(t)\mathcal M_d(t)\right),\qquad n\geq0.
$$

For $j=i$ the first derivative also has the covariance form

$$
D\bar d_i=\mathbb E_i[Dd_{i\ell}]
+\operatorname{Cov}_i(d_{i\ell},D\log K_s(i,\ell)).
$$
:::

:::{prf:proof}
:label: proof-lem-self-measurement-derivatives-full

Sum the relative probability bounds of {prf:ref}`lem-softmax-derivative-locality-full`; the undifferentiated probabilities sum to one. Multiply their majorant $\mathcal H_s$ by the measurement majorant using the full Leibniz formula. The covariance identity follows from
$DP_{i\ell}=P_{i\ell}(D\log K_{i\ell}-\mathbb E_iD\log K_{i\ell})$.
:::

:::{prf:theorem} The joint independent-companion law
:label: thm-cinf-independent-history-majorant

For complete independent rows $p_c=\prod_jP_{j,c_j}$, put $a_s=e^{-D_0^2/(2s^2)}$. Then

$$
\sum_c\|D^np_c\|\leq n![t^n]\mathcal J_{\mathrm{ind}}(t),
\qquad
\mathcal J_{\mathrm{ind}}(t)=\mathcal H_s(t)
\exp\left[\frac{2a_s^{-1}u_s(t)}{1-u_s(t)}\right].
$$

The bound is independent of $k,N$. For restricted candidate graphs, replace $a_s^{-1}$ by a proved bound on $\sum_{j\ne i}P_{ji}$.
:::

:::{prf:proof}
The self row has summed derivative majorant $\mathcal H_s$. Each other row has majorant at most $1+2P_{ji}u_s/(1-u_s)$. The product rule for independent rows multiplies these majorants. Since $1+x\preceq e^x$ for a nonnegative series $x$ with zero constant term,

$$
\prod_{j\ne i}\left(1+\frac{2P_{ji}u_s}{1-u_s}\right)
\preceq\exp\left[\frac{2u_s}{1-u_s}\sum_{j\ne i}P_{ji}\right].
$$

The availability estimate gives $\sum_{j\ne i}P_{ji}\leq a_s^{-1}$. Insert it and read coefficients. Undifferentiated rows sum to one, so this product estimate bounds the derivative sum over all assignments, not just individual assignment derivatives.
:::

### 3.2. Idealized matching and sequential greedy pairing

:::{prf:definition} Idealized matching law
:label: def-idealized-pairing-cinf

For even $k$, let $\mathcal M_k$ be the perfect matchings of the alive set. At pairing scale $s>0$, define

$$
W(M)=\prod_{\{j,\ell\}\in M}K_s(j,\ell),\qquad
p_M^{\mathrm{ideal}}=\frac{W(M)}{\sum_{M'}W(M')}.
$$

This is a probability over complete matchings. An odd-population version must state a singleton rule separately.
:::

:::{prf:theorem} Idealized matching-law and measurement derivatives
:label: thm-diversity-pairing-measurement-regularity

For the idealized law,

$$
\sum_M\|D^np_M^{\mathrm{ideal}}\|
\leq n![t^n]\mathcal H_s(t),\qquad n\geq0.
$$

Every expected pair measurement has majorant $\mathcal H_s\mathcal M_d$, and the expected full sampled fitness has majorant $\mathcal H_s\mathcal F$ whenever $\mathcal F$ bounds all fixed-assignment fitness derivatives. These bounds are population-uniform on the bounded-distance family.
:::

:::{prf:proof}
:label: proof-thm-diversity-pairing-measurement-regularity

Exactly one edge of each perfect matching involves walker $i$. Consequently the relative derivative majorant of $W(M)$ is $u_s$, independent of the number of other edges. Normalize the finite family $\{W(M)\}$ using {prf:ref}`lem-cinf-relative-majorants` and sum the resulting bounds over $M$. The probabilities sum to one at order zero, giving $\mathcal H_s$.

For an expected measurement or fitness, apply the product rule to its probability-weighted finite sum. The measurement or fitness may depend on all walkers; its uniform fixed-assignment derivative bound is precisely the additional input used in this step.

The marginal formula

$$
p_{i\ell}^{\mathrm{ideal}}=
\frac{K_s(i,\ell)Z_{\mathcal A\setminus\{i,\ell\}}}
{\sum_{q\ne i}K_s(i,q)Z_{\mathcal A\setminus\{i,q\}}}
$$

is also valid. Its remaining-matching partition functions are independent of $x_i$ but need not be equal. The proof above requires no comparison of those partition functions.
:::

:::{prf:definition} Sequential stochastic greedy pairing
:label: def-diversity-pairing-cinf

Start with the alive set $U=\mathcal A$. While $|U|\geq2$, choose a pivot $j\in U$ by a state-independent rule, sample $\ell\in U\setminus\{j\}$ with probability

$$
P_U(j,\ell)=\frac{K_s(j,\ell)}
{\sum_{q\in U\setminus\{j\}}K_s(j,q)},
$$

record the mutual pair, and remove $j,\ell$. A final singleton maps to itself. A full history records the pivots and sampled partners; its probability is the product of the conditional probabilities along that history. Output matching probabilities are sums over histories with that output.

The implementation's pivot is the first remaining index. A random index order chosen independently of the continuous state also fits this definition. State-dependent pivot selection requires including its own derivatives. This is the algorithm of {prf:ref}`def-greedy-pairing-algorithm`, with its pivot convention made explicit for differentiation.
:::

(sec-support-lem-greedy-ideal-equivalence)=
:::{prf:lemma} Uniform regularity of the actual greedy history law
:label: lem-greedy-ideal-equivalence

For the sequential law just defined, let $p_h$ be the probability of each complete history. Then

$$
\sum_h\|D^np_h\|\leq n![t^n]\mathcal H_s(t),
\qquad \mathcal H_s(t)=\frac{1+u_s(t)}{1-u_s(t)},\quad n\geq0.
$$

The same bound holds after summing histories into output matching probabilities. In particular, if $u_s(t_*)\leq1/2$, the derivative sum is at most $3n!t_*^{-n}$ independently of $k,N$. Expected measurements and expected sampled fitness have majorants $\mathcal H_s\mathcal M_d$ and $\mathcal H_s\mathcal F$, respectively.
:::

:::{prf:proof}
:label: proof-lem-greedy-ideal-equivalence

Every history probability is a finite product of smooth positive normalized weights. We prove its **summed** derivative bound by induction on the number of unpaired walkers, at a fixed base configuration. Write $u=u_s$ and $H=(1+u)/(1-u)$.

If walker $i$ has already been removed, all remaining pairing probabilities are independent of $x_i$. Their summed derivative series is exactly one. The same holds when no further pair can be drawn.

If the next pivot is $i$, every candidate edge can vary, but after the pair is chosen $i$ is removed. The normalized-row majorant is at most $H$, and all continuation probabilities have series one.

Suppose the pivot is $j\ne i$, with $i$ still available, and write $p=P_U(j,i)$ at the base configuration. Only the raw edge $(j,i)$ varies. The branch choosing $i$ has probability majorant $p(1+u)/(1-pu)$ and continuation series one. Every other branch $\ell$ has probability majorant $P_U(j,\ell)/(1-pu)$ and, by induction, continuation majorant at most $H$. Summing all branches gives

$$
\frac{(1-p)H+p(1+u)}{1-pu}=H,
$$

because $(1-u)H=1+u$. All operations involve nonnegative coefficients, so the inequality and the identity hold order by order. State-independent random pivot choices form convex mixtures and preserve the bound.

This completes the induction for the full history law. Summing probabilities of histories with the same output and using the triangle inequality preserves the derivative bound. Multiplying by a uniform fixed-history measurement or fitness majorant proves the expectation claims. Evaluating $H$ at $u(t_*)\leq1/2$ gives $H(t_*)\leq3$ and the stated Gevrey estimate.
:::

:::{div} feynman-prose
The induction closes because choosing the moving walker ends its influence on later probabilities. The same probability that makes one branch sensitive also removes that source of sensitivity. This is why a long sequence of pairing decisions need not produce a derivative bound that grows with the number of pairs.
:::

:::{prf:remark} Common kernels and different distributions
:label: rem-observation-common-kernel-structure

For four walkers with a fixed first pivot $1$, the greedy probability of the matching $\{(1,2),(3,4)\}$ is

$$
\frac{K_{12}}{K_{12}+K_{13}+K_{14}}.
$$

The idealized probability is

$$
\frac{K_{12}K_{34}}
{K_{12}K_{34}+K_{13}K_{24}+K_{14}K_{23}}.
$$

These differ when the three remaining-edge weights differ. Independent softmax has yet another support: choices need not be mutual. Regularity is established for each law by its own majorant, not by equality of laws.
:::

:::{prf:theorem} Comparison of regularity across companion mechanisms
:label: thm-statistical-equivalence-companion-mechanisms

Let $\bar d_j^{(a)}$ and $\bar d_j^{(b)}$ be expectations under two of the laws proved above, with their respective majorants $\mathcal J_a,\mathcal J_b$. Then

$$
|\bar d_j^{(a)}-\bar d_j^{(b)}|\leq D_\delta-\delta,
$$

$$
\|D^n(\bar d_j^{(a)}-\bar d_j^{(b)})\|
\leq n![t^n]\left[(\mathcal J_a+\mathcal J_b)\mathcal M_d\right].
$$

Both expectations are smooth and have population-uniform Gevrey-1 bounds under the stated inputs. Their values, constants, and limiting dynamics can differ. No decay of their difference with $k$ follows from these derivative estimates.
:::

:::{prf:proof}
:label: proof-thm-statistical-equivalence-companion-mechanisms

Every pair measurement, including a singleton, lies in $[\delta,D_\delta]$, giving the first bound. Subtract the finite expectation formulas, differentiate, and apply the two proved probability and measurement majorants. Their triangle inequality gives the second bound. Each majorant converges near zero, so their sum does too.
:::

(sec-cinf-moments-fitness)=
## 4. Moments, standardization, and the complete fitness

:::{div} feynman-prose
We can now put the pieces together. Once the weight and measurement derivatives have convergent majorants, an average is handled by multiplication of two series. Variance needs one more product. The inverse standard deviation is a composition whose radius is controlled by $\sigma_{\min}$.

The construction is finite even though it controls infinitely many derivative orders. At each stage we choose a sufficiently small positive argument for the majorant. The final argument can shrink as regularization parameters shrink, but it does not shrink just because more walkers are added.
:::

:::{prf:definition} Moment derivative sequences
:label: def-cinf-moment-sequences

Let $\sum_j\|D^nw_j\|\leq W_n$, $W_0=1$, and $\|D^nm_j\|\leq M_n$ for all $n\geq0$. Put

$$
U_n=\sum_{q=0}^n\binom nq W_qM_{n-q},\quad
H_n=\sum_{q=0}^n\binom nq M_qM_{n-q},\quad
S_n=\sum_{q=0}^n\binom nq W_qH_{n-q}.
$$

For $n\geq1$, let

$$
T_n=S_n+\sum_{q=0}^n\binom nq U_qU_{n-q}.
$$

Write $\mathcal W,\mathcal M,\mathcal U$ for the corresponding factorial-normalized series. Then

$$
\mathcal U=\mathcal W\mathcal M,\qquad
\mathcal T_+=\sum_{n\geq1}\frac{T_n}{n!}t^n
=\mathcal W\mathcal M^2+\mathcal U^2-2M_0^2.
$$

The subtracted constant leaves a series with zero constant term and nonnegative coefficients. For the variance itself use $0\leq V\leq M_0^2$.
:::

:::{prf:lemma} First derivative of the localized mean
:label: lem-first-derivative-localized-mean-full

For $\mu=\sum_jw_jm_j$,

$$
D\mu=\sum_j(Dw_j)(m_j-\mu)+\sum_jw_jDm_j,
\qquad \|D\mu\|\leq W_1M_0+M_1=U_1.
$$
:::

:::{prf:proof}
:label: proof-lem-first-derivative-localized-mean-full

Differentiate the finite weighted sum. Subtract $\mu\sum_jDw_j=0$ to obtain the centered identity. For the stated bound use its uncentered form: the weight-derivative term is at most $W_1M_0$ and the measurement-derivative average at most $M_1$.
:::

:::{prf:lemma} All derivatives of the localized mean
:label: lem-mth-derivative-localized-mean-full

For every $n\geq0$,

$$
\|D^n\mu\|\leq U_n,
\qquad
\left\|D^n\sum_jw_jm_j^2\right\|\leq S_n.
$$
:::

:::{prf:proof}
:label: proof-lem-mth-derivative-localized-mean-full

The exact product expansion is

$$
D^n\mu=\sum_j\sum_{q=0}^n\binom nq
\operatorname{Sym}(D^qw_j\otimes D^{n-q}m_j).
$$

Take norms and sum the weight derivatives to obtain $U_n$. The Leibniz rule applied first to $m_j^2$ gives $H_n$, and then to its weighted sum gives $S_n$. This proof permits every measurement to depend on the differentiated walker.
:::

:::{prf:theorem} Uniformity of normalized moments
:label: thm-k-uniformity-telescoping-full

If the weight and measurement majorants $\mathcal W,\mathcal M$ converge at some common positive argument, then the localized mean and second moment have population-uniform Gevrey-1 bounds whenever the input majorants are population-uniform. The exact cancellation $\sum_jD^nw_j=0$ permits centering; the summed norm bound $W_n$ provides the quantitative control.
:::

:::{prf:proof}
:label: proof-thm-k-uniformity-telescoping-full

The preceding lemma gives the majorants $\mathcal W\mathcal M$ and $\mathcal W\mathcal M^2$. They are finite at the common argument. Evaluating them there and using nonnegative coefficients gives $Cn!t_*^{-n}$ bounds. No empirical count is replaced by an integral in this argument.
:::

:::{prf:lemma} First derivative of the localized variance
:label: lem-first-derivative-localized-variance-full

For $V=\sum_jw_j(m_j-\mu)^2$,

$$
DV=\sum_j(Dw_j)(m_j-\mu)^2+
2\sum_jw_j(m_j-\mu)Dm_j,
\qquad \|DV\|\leq T_1.
$$
:::

:::{prf:proof}
:label: proof-lem-first-derivative-localized-variance-full

Differentiate the centered expression. The terms containing $D\mu$ vanish because $\sum_jw_j(m_j-\mu)=0$. For the bound, differentiate $V=\sum_jw_jm_j^2-\mu^2$ and apply the moment bounds, giving $S_1+2U_0U_1=T_1$.
:::

:::{prf:theorem} All derivatives of the localized variance
:label: thm-mth-derivative-localized-variance-full

For $n\geq1$, $\|D^nV\|\leq T_n$. If $\mathcal W,\mathcal M$ converge near zero, then so does $\mathcal T_+$, giving population-uniform Gevrey-1 variance bounds.
:::

:::{prf:proof}
:label: proof-thm-mth-derivative-localized-variance-full

The exact identity $V=\sum_jw_jm_j^2-\mu^2$ and the product rule give

$$
D^nV=D^n\sum_jw_jm_j^2-
\sum_{q=0}^n\binom nq
\operatorname{Sym}(D^q\mu\otimes D^{n-q}\mu).
$$

Take norms to obtain $T_n$. Derivatives of a square contain exactly these two-factor terms; there are no higher products of derivatives and no omitted lower-order remainder. Dividing by $n!$ gives the series $\mathcal T_+$ in the definition. Products of the convergent input majorants converge near zero, proving the final assertion.
:::

### 4.1. The square root and inverse square root

:::{prf:lemma} Regularized deviation and its inverse
:label: lem-properties-regularized-std-dev-full

Let $q(V)=\sqrt{V+\sigma_{\min}^2}$, with $V\geq0$ and $\sigma_{\min}>0$. Both $q(V)$ and $q(V)^{-1}$ are $C^\infty$ whenever $V$ is. Their positive-order scalar derivatives obey

$$
\frac{|q^{(\ell)}(V)|}{\ell!}
\leq\left|\binom{1/2}{\ell}\right|\sigma_{\min}^{1-2\ell},
\qquad
\frac{|(q^{-1})^{(\ell)}(V)|}{\ell!}
\leq\frac{\binom{2\ell}{\ell}}{4^\ell}\sigma_{\min}^{-1-2\ell}.
$$

Using the variance increment majorant $\mathcal T_+$, their derivative majorants are

$$
\mathcal Q(t)=\sqrt{M_0^2+\sigma_{\min}^2}
+\sigma_{\min}\left[1-\sqrt{1-\mathcal T_+(t)/\sigma_{\min}^2}\right],
$$

$$
\mathcal R(t)=\frac1{\sigma_{\min}}
\left[1-\mathcal T_+(t)/\sigma_{\min}^2\right]^{-1/2}.
$$

They converge at every sufficiently small positive $t$ with $\mathcal T_+(t)<\sigma_{\min}^2$.
:::

:::{prf:proof}
:label: proof-lem-properties-regularized-std-dev-full

The scalar Taylor coefficients of $s^{1/2}$ and $s^{-1/2}$ at $s=V+\sigma_{\min}^2$ are respectively $\binom{1/2}{\ell}s^{1/2-\ell}$ and $(-1)^\ell\binom{2\ell}{\ell}4^{-\ell}s^{-1/2-\ell}$. For positive order these exponents are negative, so $s\geq\sigma_{\min}^2$ gives the bounds. Their absolute-coefficient series are $1-\sqrt{1-z}$ and $(1-z)^{-1/2}$. Substitute the variance increment majorant using the partition formula. The order-zero deviation is at most $\sqrt{M_0^2+\sigma_{\min}^2}$, and its inverse is at most $\sigma_{\min}^{-1}$.
:::

:::{prf:proposition} An explicit factorial bound for the square-root composition
:label: prop-factorial-sqrt-composition

If $\|D^nV\|\leq C_VB_V^nn!$ for all $n\geq1$, then

$$
\|D^n\sqrt{V+\sigma_{\min}^2}\|
\leq\frac{C_V}{\sigma_{\min}}n!B_V^n
\left(1+\frac{C_V}{\sigma_{\min}^2}\right)^{n-1}.
$$

For the inverse deviation the same bound has leading factor $C_V/\sigma_{\min}^3$. The remaining coefficient generally grows exponentially with order; a bound polynomial in $n$ is not required for Gevrey-1 regularity.
:::

:::{prf:proof}
:label: proof-prop-factorial-sqrt-composition

For $\ell\geq1$, both binomial coefficients in the preceding lemma have absolute normalized value at most one. The square-root outer derivatives therefore satisfy $|q^{(\ell)}|\leq\sigma_{\min}(\sigma_{\min}^{-2})^\ell\ell!$. Apply {prf:ref}`cor-gevrey-closure` with $C_f=\sigma_{\min}$, $B_f=\sigma_{\min}^{-2}$, $C_g=C_V$, and $B_g=B_V$. For the inverse use $C_f=\sigma_{\min}^{-1}$ instead.
:::

:::{prf:theorem} Smooth and Gevrey-1 Z-scores
:label: thm-cinf-regularity-zscore-full

The score $Z=(m_i-\mu)/\sqrt{V+\sigma_{\min}^2}$ is $C^\infty$ for smooth inputs with the positive regularizer. Under the convergent majorant hypotheses,

$$
\|D^nZ\|\leq n![t^n]\mathcal Z(t),\qquad
\mathcal Z(t)=(\mathcal M(t)+\mathcal U(t))\mathcal R(t).
$$

Hence it has population-uniform Gevrey-1 bounds on the stated family.
:::

:::{prf:proof}
:label: proof-thm-cinf-regularity-zscore-full

The numerator is smooth and its derivatives have majorant $\mathcal M+\mathcal U$. The denominator is positive and its inverse has majorant $\mathcal R$. The product rule gives $\mathcal Z$. Choose a common positive argument where these majorants converge and evaluate there. This retains all inverse-regularizer powers; the third-order specialization is {prf:ref}`lem-normalized-zscore-bounds`.
:::

### 4.2. The two channels and their expectations

:::{prf:assumption} Smooth versus analytic channel inputs
:label: assump-rescale-function-cinf-full

For a smoothness conclusion, the reward measurements and rescale function are $C^\infty$ on the chosen stratum. For population-uniform Gevrey-1 bounds, their derivative majorants converge near zero with constants uniform on that family.

The algorithmic sigmoid satisfies globally

$$
|g_A^{(n)}(z)|\leq A n!(2/\pi)^n,\qquad n\geq0,\quad z\in\mathbb R.
$$

A general smooth clipping function need not satisfy a Gevrey-1 bound. The positivity floor $\eta$ keeps arbitrary fixed real channel powers away from their singularity at zero.
:::

:::{prf:proof}
For complex $z=x+iy$ with $|y|\leq\pi/2$, set $a=e^{-x}\geq0$. Then $|1+ae^{-iy}|^2=1+a^2+2a\cos y\geq1$. Thus the analytic sigmoid is bounded by $A$ on the radius-$\pi/2$ disk around any real point. The Cauchy integral formula gives $|g_A^{(n)}|\leq An!(2/\pi)^n$. Its singularities lie outside those disks. This proof establishes the bound for this rescale; smoothness of an arbitrary substitute is a separate property.
:::

:::{prf:theorem} Full sampled-fitness majorant
:label: thm-main-cinf-regularity-fitness-potential-full

For each channel $a\in\{d,r\}$, construct $\mathcal Z_a$ from its measurement majorant. Let $\mathcal Z_{a,+}=\mathcal Z_a-\mathcal Z_a(0)$ and

$$
\mathcal C_a(t)=\eta+
\frac{A}{1-(2/\pi)\mathcal Z_{a,+}(t)}.
$$

For a real exponent $p$, put

$$
C_p=\max\{(\eta/2)^p,(A+3\eta/2)^p\},\qquad
\mathcal B_{a,p}(t)=
\frac{C_p}{1-(2/\eta)(\mathcal C_a(t)-\mathcal C_a(0))}.
$$

Then every fixed-assignment fitness has derivative majorant

$$
\mathcal F(t)=\mathcal B_{d,\beta_{\mathrm{fit}}}(t)
\mathcal B_{r,\alpha_{\mathrm{fit}}}(t),\qquad
\|D^nF_i^c\|\leq n![t^n]\mathcal F(t).
$$

For the bounded-distance family, fixed positive scales and floors, and population-uniform analytic reward bounds, $\mathcal F$ converges near zero independently of $k,N,c$. For general smooth inputs the same finite-stratum formula is $C^\infty$, without an automatic factorial bound.
:::

:::{prf:proof}
:label: proof-thm-main-cinf-regularity-fitness-potential-full

The preceding theorems give the score majorant. Composition with the sigmoid uses its uniform scalar derivative bound and the positive-order score majorant, producing $A/[1-(2/\pi)\mathcal Z_{a,+}]$. Adding $\eta$ gives $\mathcal C_a$.

For a real channel value $c\in[\eta,A+\eta]$, the complex disk of radius $\eta/2$ stays in the right half-plane. The analytic branch of $z^p$ is bounded there by $C_p$, because $\eta/2\leq|z|\leq A+3\eta/2$. Cauchy's formula gives $|(d/dc)^nz^p|\leq C_p n!(2/\eta)^n$. Composing this bound with the positive-order channel majorant gives $\mathcal B_{a,p}$. The product of the two powered channels gives $\mathcal F$.

Every substituted increment series has zero constant term. Starting from convergent weight and measurement series, choose the argument small enough that each displayed denominator remains positive. This gives a common positive radius depending only on the input constants. The sampled-distance bound is uniform in $c$, so the final majorant is too. For smooth inputs without convergent majorants, finite sums, products, and compositions with positive denominators still prove $C^\infty$ regularity.
:::

:::{prf:theorem} Full fitness under each companion law
:label: thm-unified-cinf-regularity-both-mechanisms

Choose any of the three proved laws, with history majorant $\mathcal J$ equal to $\mathcal J_{\mathrm{ind}}$ for independent rows or $\mathcal H_s$ for idealized or greedy matching. Then

$$
\|D^n\overline F_i\|\leq n![t^n](\mathcal J\mathcal F).
$$

The expected-measurement surrogate $\widetilde F_i$ is analyzed separately by substituting the majorant $\mathcal J\mathcal M_d$ for each diversity measurement before constructing the moments. Both constructions are population-uniform Gevrey-1 under their stated input hypotheses, and both are smooth on finite strata when only smooth inputs are assumed.
:::

:::{prf:proof}
:label: proof-thm-unified-cinf-regularity-both-mechanisms

Differentiate the finite sum $\overline F_i=\sum_hp_hF_i^{c(h)}$. At order $n$, the Leibniz rule gives the convolution of the summed history-probability bounds and the uniform fixed-assignment fitness bounds. Its factorial-normalized version is $\mathcal J\mathcal F$. For the surrogate, first apply the same argument to $\bar d_j=\sum_hp_hd_j^{c(h)}$, then pass its bounds through the moments and nonlinear channel formulas. The order of expectation and nonlinear composition is different, which is why the two majorants are constructed separately.
:::

(sec-support-cor-gevrey-1-fitness-potential-full)=
:::{prf:corollary} Real analyticity from the proved factorial bound
:label: cor-gevrey-1-fitness-potential-full

For any function above whose majorant $\mathcal A$ is finite at $t_*>0$,

$$
\|D^nF\|\leq\mathcal A(t_*)n!t_*^{-n}.
$$

On the interior of the fixed stratum this implies real analyticity in the specified coordinate. On a segment remaining in the uniformity region, the degree-$m$ Taylor remainder is at most

$$
\mathcal A(t_*)\left(\frac{\|h\|}{t_*}\right)^{m+1}.
$$
:::

:::{prf:proof}
:label: proof-cor-gevrey-1-fitness-potential-full

Nonnegative coefficients imply $[t^n]\mathcal A\leq\mathcal A(t_*)t_*^{-n}$. Taylor's formula along the segment has remainder bounded by $\sup\|D^{m+1}F\|\|h\|^{m+1}/(m+1)!$, giving the display. When $\|h\|<t_*$ it tends to zero, so the Taylor series represents $F$.
:::

:::{prf:theorem} Complete smoothness and analyticity statement
:label: thm-main-complete-cinf-geometric-gas-full

On a fixed alive/candidate/branch stratum, the full sampled fitness, the expected-measurement surrogate, and the expected sampled fitness under each specified companion law are $C^\infty$ when their continuous inputs are $C^\infty$ and their denominators stay positive.

For fixed positive $\rho,s,\delta,\sigma_{\min},\eta,A$, bounded pair distances, and population-uniform analytic reward bounds, their block derivatives satisfy

$$
\|D^nF\|\leq C_F B_F^n n!,\qquad n\geq0,
$$

with $C_F,B_F$ independent of $k,N$ and the differentiated walker. The relevant constants are obtained by evaluating the explicitly constructed majorants. On an unbounded family, the same conclusion follows from population-uniform convergent normalized derivative-ratio and measurement majorants together with a proved joint-law majorant. No density assumption is needed for the bounded-distance result.
:::

:::{prf:proof}
:label: proof-thm-main-complete-cinf-geometric-gas-full

Combine the distance and normalized-weight estimates, the separate companion-law proofs, and the complete fitness majorant. All operations in the finite formula preserve smoothness. Under the stronger majorant assumptions, choose a common positive $t_*$ where the relevant majorant is finite, and take $C_F$ equal to that value and $B_F=t_*^{-1}$. The preceding corollary gives analyticity. These constants retain the dependence on all kernel scales and regularizers; no scale is omitted from the inner constants by an unproved separation argument.
:::

(sec-cinf-localization-density)=
## 5. Smooth localization, population counts, and density

:::{div} feynman-prose
There are two different reasons for introducing a neighborhood. One is algebraic: nearby particles receive larger kernel weights. The other is probabilistic: a density estimate tells us how many particles we expect to find there. The derivative proof above uses the first fact. It does not need the second. Keeping them separate prevents a population average from quietly becoming a bound on the number of particles in every realization.

A smooth partition is useful when we want to organize those averages by location. Think of overlapping lamps whose brightness adds to one. A point may lie under several lamps, but its total contribution is still counted exactly once.
:::

:::{prf:definition} Auxiliary smooth partition
:label: def-smooth-phase-space-partition-full

Let $O\subset\mathbb R^q$ be open. Choose centers $c_m$ such that the balls $B(c_m,h)$ cover $O$, the family $B(c_m,2h)$ is locally finite, and at most $L$ such enlarged balls meet any point. A subordinate smooth partition consists of functions $\psi_m\geq0$ supported in $B(c_m,2h)$ with

$$
\sum_m\psi_m(y)=1,\qquad y\in O.
$$

This is an auxiliary decomposition for the proof. It is not an additional operation in the particle algorithm. For the phase-space metric with $\lambda>0$, the same construction applies after the linear change of variables $(x,v)\mapsto(x,\sqrt\lambda v)$.
:::

:::{prf:definition} A normalized bump partition
:label: const-mollified-partition-full

Set

$$
\beta(u)=
\begin{cases}
\exp[-1/(1-u)],&0\leq u<1,\\
0,&u\geq1,
\end{cases}
\qquad
b_m(y)=\beta\!\left(\frac{\|y-c_m\|^2}{4h^2}\right),
\qquad
\psi_m(y)=\frac{b_m(y)}{\sum_\ell b_\ell(y)}.
$$

Then $\sum_\ell b_\ell\geq e^{-4/3}$ on $O$, and on $B(c_m,h)$,

$$
\psi_m(y)\geq\frac{e^{-1/3}}{L}.
$$

In particular, a universal lower bound of $1/2$ is not required and generally does not hold when several supports overlap.
:::

:::{prf:lemma} Derivatives of the partition
:label: lem-partition-derivative-bounds-full

The construction is $C^\infty$. For each $n\geq0$ there is a finite constant $C_n=C_n(q,L)$ such that

$$
\sup_{y\in O}\sum_m\|D^n\psi_m(y)\|\leq C_nh^{-n}.
$$

The constants do not depend on the number of centers or particles. This statement does not assert Gevrey-1 growth of $C_n$.
:::

:::{prf:proof}
:label: proof-lem-partition-derivative-bounds-full

Every derivative of $\exp[-1/(1-u)]$ on $u<1$ is that exponential times a polynomial in $(1-u)^{-1}$. All such derivatives tend to zero as $u\uparrow1$. Extension by zero is therefore smooth. The fixed function $z\mapsto\beta(\|z\|^2/4)$ has bounded derivatives of every order, so scaling gives $\|D^nb_m\|\leq c_nh^{-n}$. At any point at most $L$ terms can contribute, hence $\sum_m\|D^nb_m\|\leq Lc_nh^{-n}$.

The covering property gives at least one $b_m\geq e^{-4/3}$; also each $b_m\leq e^{-1}$, giving the stated core lower bound. Apply the finite-order normalized derivative recurrence in {prf:ref}`thm-cinf-normalized-majorant` with denominator floor $e^{-4/3}$. It produces finite constants of the form $C_nh^{-n}$ recursively. Local finiteness justifies the same calculation for a countable family.
:::

:::{prf:remark} Smooth cutoffs and analyticity
:label: rem-cinf-bump-not-analytic

A nonzero compactly supported smooth function on a connected Euclidean domain cannot satisfy uniform Gevrey-1 bounds on every compact subset. Such bounds would imply real analyticity by {prf:ref}`cor-gevrey-1-fitness-potential-full`. An analytic function that vanishes on an open set vanishes on the connected domain: its Taylor series is zero there, and overlapping Taylor neighborhoods continue this identity along any path. The bump partition can organize $C^\infty$ estimates, but the Gevrey-1 proof of the fitness uses the analytic Gaussian and normalization majorants directly.
:::

:::{prf:definition} Soft membership and effective populations
:label: def-soft-cluster-membership-full

For a configuration $y_1,\ldots,y_k$, define $\alpha_{jm}=\psi_m(y_j)$. Thus $0\leq\alpha_{jm}\leq1$ and $\sum_m\alpha_{jm}=1$ for every particle.
:::

:::{prf:definition} Count, empirical fraction, and limiting mass
:label: def-effective-cluster-population-full

The effective count, empirical fraction, and mass under a probability law $\mu$ are respectively

$$
k_{\mathrm{eff},m}=\sum_{j=1}^k\psi_m(y_j),\qquad
\widehat m_m=\frac{k_{\mathrm{eff},m}}k=L_k(\psi_m),\qquad
m_m(\mu)=\int\psi_m\,d\mu.
$$

The first has the scale of a number of particles; the latter two are dimensionless fractions.
:::

(sec-support-lem-effective-cluster-size-bounds-full)=
:::{prf:lemma} Bounds on effective cluster populations
:label: lem-effective-cluster-size-bounds-full

For every configuration,

$$
0\leq k_{\mathrm{eff},m}\leq k,\qquad
\sum_mk_{\mathrm{eff},m}=k.
$$

For fixed centers, if each one-particle marginal has Lebesgue density bounded above by $\varrho_{\max}$, then

$$
\mathbb E k_{\mathrm{eff},m}
\leq k\varrho_{\max}|B(c_m,2h)|,
\qquad
\mathbb E\widehat m_m\leq\varrho_{\max}|B(c_m,2h)|.
$$

The same bounds hold with the right sides truncated at $k$ and $1$. If $L_k\Rightarrow\mu$, then $\widehat m_m\to m_m(\mu)$ in the corresponding mode of weak convergence, because $\psi_m$ is bounded and continuous. The unnormalized count has no such probability-mass limit.
:::

:::{prf:proof}
:label: proof-lem-effective-cluster-size-bounds-full

Sum the nonnegative partition identities over particles. For the expectation, integrate each $\psi_m$ against its marginal density and use $0\leq\psi_m\leq\mathbf1_{B(c_m,2h)}$. Linearity of expectation does not require independence. The weak-convergence statement is the defining bounded-continuous-test property of weak convergence; it also holds in distribution by continuity of $\mu\mapsto\mu(\psi_m)$.
:::

:::{prf:theorem} Density bounds from an identified killed kernel
:label: assump-uniform-density-full

Let $Q$ be a sub-Markov kernel and let $\nu$ be a QSD with $\nu Q=\alpha\nu$, $\alpha>0$. Suppose, relative to a specified Lebesgue measure, the entire kernel $Q^m$ has density $k_m(z,y)$ and

$$
0\leq k_m(z,y)\leq K(y).
$$

Then $\nu$ has density

$$
\varrho_\nu(y)=\alpha^{-m}\int k_m(z,y)\nu(dz)
\leq\alpha^{-m}K(y).
$$

If $k_m(z,y)\geq\ell_E\mathbf1_C(z)$ for $y\in E$ and $\nu(C)>0$, then $\varrho_\nu(y)\geq\alpha^{-m}\ell_E\nu(C)$ on $E$. If all $y$ derivatives of $k_m$ exist and admit locally uniform, $\nu$-integrable majorants in $z$, then $\varrho_\nu$ is smooth.
:::

:::{prf:proof}

Iteration gives $\nu Q^m=\alpha^m\nu$. Tonelli's theorem applied to this identity gives the displayed density formula and both inequalities. Dominated differentiation gives smoothness under the additional derivative hypotheses.

A bound relative to another probability law does not by itself give the assumed Lebesgue kernel bound. Nor may a singular transition branch be omitted when claiming that the entire kernel has a density. These distinctions are developed in {prf:ref}`thm-uniform-density-bound-hk`, {prf:ref}`lem-linfty-full-operator`, and {prf:ref}`lem-qsd-strict-positivity`. Uniformity in population size requires uniform control of the quantities in this formula. A positive global lower bound on a probability density is impossible on a domain of infinite Lebesgue volume.
:::

:::{prf:definition} Kernel masses at the two interaction scales
:label: def-effective-counts-two-scales-cinf

For a probability law $\mu$ on phase space and scale $s>0$, set

$$
z_s(\mu,y)=\int\exp\!\left[-\frac{\|x-x'\|^2+\lambda\|v-v'\|^2}{2s^2}\right]\mu(dx',dv').
$$

The companion scale $s=\varepsilon_d$ and localization scale $s=\rho$ give two different kernel masses. If denoted by $k_{\mathrm{eff}}^{\varepsilon_d}$ or $k_{\mathrm{eff}}^\rho$ in a mean-field formula, these quantities are normalized masses, not finite-particle counts. A common factor $e^{-\delta^2/(2s^2)}$ from the regularized distance can be restored if the unnormalized kernel includes it; it cancels in normalized weights.
:::

:::{prf:remark} Superscripts name a scale, not a normalization
:label: notation-keff-superscripts-cinf

The finite row sum $\sum_{j\ne i}K_s(y_i,y_j)$ has $k-1$ terms. Its empirical version is $(k-1)^{-1}\sum_{j\ne i}K_s(y_i,y_j)$, while $z_s(\mu,y_i)$ is an integral against a probability law. A superscript such as $\rho$ does not remove the factor $k-1$. In particular, kernel mass estimates cannot give a deterministic bound on the number of nonzero Gaussian weights: every weight is positive on a finite configuration.
:::

:::{prf:lemma} Gaussian mass under a bounded phase-space density
:label: lem-mean-field-kernel-mass-bound

If $\mu$ has density at most $\varrho_{\max}$ relative to $dx\,dv$ on $\mathbb R^{2d}$ and $\lambda>0$, then

$$
0<z_s(\mu,y)\leq
\min\!\left\{1,\varrho_{\max}(2\pi s^2)^d\lambda^{-d/2}\right\}.
$$

For a spatial kernel integrated against a spatial density, the Gaussian volume is instead $(2\pi s^2)^{d/2}$. If a density is bounded below by $c>0$ on a set $E$, then $z_s(\mu,y)\geq c\int_EK_s(y,z)\,dz$. This need not give a positive bound uniform over all centers in an unbounded space.
:::

:::{prf:proof}

The kernel is strictly positive and bounded by one. Its integral over phase space factors into two Gaussian integrals. The change of variables $u=\sqrt\lambda(v'-v)$ contributes $\lambda^{-d/2}$, giving $(2\pi s^2)^d\lambda^{-d/2}$. Multiply by the density bound. The lower bound follows by restricting the nonnegative integral to $E$.
:::

:::{prf:lemma} From empirical averages to integrals
:label: lem-sum-to-integral-bound-full

For a fixed center $y$, bounded measurable $f$, and one-particle marginal densities bounded by $\varrho_{\max}$,

$$
\mathbb E\left|\frac1k\sum_{j=1}^kf(Y_j)K_s(y,Y_j)\right|
\leq \varrho_{\max}\|f\|_\infty\int K_s(y,z)\,dz.
$$

The corresponding raw-sum bound has an additional factor $k$. If $L_k\Rightarrow\mu$ and $fK_s(y,\cdot)$ is bounded continuous, the empirical average converges to $\int f(z)K_s(y,z)\mu(dz)$. A random center requires either suitable joint convergence or conditional density bounds; marginal density bounds alone do not justify conditioning on another particle.

If $T$ is an injective $C^1$ change of variables with $|\det DT|\geq J_{\min}>0$ on the relevant region, a density bounded by $\varrho_{\max}$ pushes forward to one bounded by $\varrho_{\max}/J_{\min}$ there.
:::

:::{prf:proof}
:label: proof-lem-sum-to-integral-bound-full

Use the triangle inequality, linearity of expectation, and the marginal density bound in each term. The convergence statement follows from weak convergence. For a random center, the same expectation proof works after conditioning only if the needed conditional densities satisfy the bound. The change-of-variables formula gives

$$
\varrho_{T(Y)}(z)=
\frac{\varrho_Y(T^{-1}z)}{|\det DT(T^{-1}z)|}.
$$

For example, the radial map $T(x)=Cx/(C+\|x\|)$ has determinant $(C/(C+\|x\|))^{d+1}$ away from the origin. Its infimum is zero on the whole unbounded domain. It therefore supplies a density bound only on a region where that determinant has a positive lower bound. Its behavior at the origin must also be checked before invoking higher-order smoothness; a first-order change-of-variables calculation is not an all-orders regularity proof.
:::

:::{prf:lemma} Close pairs under independence
:label: lem-close-pair-probability-full

Let $Y,Y'$ be independent, and suppose $Y'$ has density bounded by $\varrho_{\max}$ on $\mathbb R^q$. Then

$$
\mathbb P(\|Y-Y'\|\leq r)\leq
\min\{1,\varrho_{\max}|B(0,r)|\}.
$$

For $k$ independent such particles, the probability of at least one close unordered pair is at most $\binom{k}{2}\varrho_{\max}|B(0,r)|$, truncated at one. The same argument works with dependent particles if the conditional densities given the center have the stated bound.
:::

:::{prf:proof}

Condition on $Y=y$, integrate the density of $Y'$ over $B(y,r)$, and then average in $y$. The union bound over unordered pairs gives the second statement. Independence is used only to identify the conditional density with the marginal one.
:::

:::{prf:lemma} A row-normalized Gaussian tail
:label: lem-softmax-tail-corrected-full

Write $r_{ij}^2=\|x_i-x_j\|^2+\lambda\|v_i-v_j\|^2$ and $Z_i=\sum_{j\ne i}\exp[-r_{ij}^2/(2s^2)]$. For a complete row with $k\geq2$,

$$
\sum_{j:r_{ij}>R}P_{ij}
\leq\frac{(k-1)e^{-R^2/(2s^2)}}{Z_i}.
$$

If at least one candidate lies within $R_{\max}$, then

$$
\sum_{j:r_{ij}>R}P_{ij}
\leq\min\!\left\{1,(k-1)
 e^{-(R^2-R_{\max}^2)/(2s^2)}\right\}.
$$
:::

:::{prf:proof}

Each tail numerator is at most $e^{-R^2/(2s^2)}$, and there are at most $k-1$ of them. The available nearby candidate gives $Z_i\geq e^{-R_{\max}^2/(2s^2)}$. Divide these bounds. A geometric assumption that some point of a region has high reward does not assert that a current candidate lies there; the displayed nearest-candidate condition must be checked for the row being analyzed.
:::

(sec-support-cor-effective-interaction-radius-full)=
:::{prf:corollary} Radius containing all but a small row mass
:label: cor-effective-interaction-radius-full

Under the nearest-candidate hypothesis, let

$$
R_{\mathrm{eff}}^2=R_{\max}^2+2s^2\log(k^2).
$$

Then the probability assigned outside this radius is at most $(k-1)/k^2\leq1/k$. This controls a probability tail, not the number of candidates inside the radius.
:::

:::{prf:proof}
:label: proof-cor-effective-interaction-radius-full

Substitute $R=R_{\mathrm{eff}}$ in the preceding lemma. The exponential factor is $k^{-2}$.
:::

(sec-support-lem-effective-companion-count-full)=
:::{prf:lemma} Expected number of nearby companions
:label: lem-effective-companion-count-full

Suppose that, conditionally on $Y_i=y$, every other particle has a density at most $\varrho_{\max}$ in $q$-dimensional Euclidean coordinates. For a deterministic radius $R$,

$$
\mathbb E\!\left[\#\{j\ne i:\|Y_j-Y_i\|\leq R\}\mid Y_i\right]
\leq\min\{k-1,(k-1)\varrho_{\max}|B(0,R)|\}.
$$

The expected nearby fraction is obtained by dividing by $k-1$. In phase space with $q=2d$ and metric parameter $\lambda>0$, the ball volume is $\omega_{2d}R^{2d}\lambda^{-d/2}$. When $R_{\max}=O(s)$ uniformly, inserting $R_{\mathrm{eff}}$ gives a bound of order $k s^{2d}(1+\log k)^d$, truncated at $k-1$, rather than a population-independent count.
:::

:::{prf:proof}
:label: proof-lem-effective-companion-count-full

Condition on the center and sum the conditional probabilities of the $k-1$ indicator events. Each is bounded by density times ball volume. The metric volume follows from the same velocity scaling as the Gaussian integral. The radius formula gives the stated dependence on $k$.

There is no deterministic conclusion from this expectation bound. There is also no conclusion from marginal density bounds alone: take $Y_1=\cdots=Y_k=Y$ with $Y$ uniformly distributed on a cube. Each marginal has a bounded density, but every particle has $k-1$ neighbors at any positive radius. This example is precisely why the random-center conditional hypothesis is present.
:::

:::{prf:theorem} Cluster-resolved companion derivatives
:label: thm-cluster-localized-derivative-bounds-full

For an independent companion row $j\ne i$, let $\bar d_j=\sum_\ell P_{j\ell}d_{j\ell}$. The earlier off-diagonal estimate gives

$$
\|D_i^n\bar d_j\|\leq P_{ji}C_n,
\qquad C_n=n![t^n]\mathcal E(t),\quad n\geq1.
$$

In particular one may take $C_1=1+2D_\delta D_0/s^2$. If $Y_i\in\operatorname{supp}\psi_m$ and $Y_j\in\operatorname{supp}\psi_\ell$, set $D_{m\ell}=\|c_m-c_\ell\|$ in the chosen metric coordinates. Then

$$
\|D_i^n\bar d_j\|
\leq C_n\min\!\left\{1,
\frac{\exp[-(D_{m\ell}-4h)_+^2/(2s^2)]}{Z_j}\right\}.
$$

Thus a useful uniform cluster estimate requires a proved lower bound on the actual row denominator. For complete rows on a bounded-distance family, {prf:ref}`lem-companion-availability-enforcement` provides one.
:::

:::{prf:proof}
:label: proof-thm-cluster-localized-derivative-bounds-full

The probability factor is retained in {prf:ref}`lem-companion-measurement-derivatives-full`. The first-order formula from {prf:ref}`lem-softmax-jacobian-reduction` gives $P_{ji}[1+2D_\delta D_0/s^2]$. On the two supports the triangle inequality gives $\|Y_i-Y_j\|\geq(D_{m\ell}-4h)_+$. Insert this bound in $P_{ji}=K_s(Y_j,Y_i)/Z_j$ and also use $P_{ji}\leq1$.

For any pair, the identity $1=\sum_{m,\ell}\psi_m(Y_i)\psi_\ell(Y_j)$ gives an exact decomposition of the estimate by clusters. This multiplies an already differentiated quantity by a partition of unity; it does not discard derivatives of a partition inserted into the original function. Keeping the positive part in the exponent is necessary when the supports overlap.
:::

(sec-cinf-mean-field-parameters)=
## 6. Passing to a continuum field and retaining parameter dependence

:::{div} feynman-prose
A cloud with more particles does not automatically acquire a differentiable continuum limit. The empirical averages must converge, and the functions being averaged must remain controlled while we differentiate them. The normalized calculus makes the second requirement explicit: it asks us to control a denominator and the derivatives of its numerator before taking a ratio.

There is also a useful practical lesson in the analytic radius. A small radius does not mean the field is nonsmooth. It means that a Taylor expansion is reliable over a smaller region. Narrow kernels and small regularizers can produce exactly this situation.
:::

:::{prf:theorem} Differentiating normalized mean-field integrals
:label: thm-cinf-mean-field-integrals

Let $x$ range over an open finite-dimensional parameter domain, let $\mu$ be a fixed probability law, and let $a(x,y)>0$. Suppose every $x$ derivative is locally dominated by an integrable function of $y$. Set

$$
A_\mu(x)=\int a(x,y)\mu(dy),\qquad
w_\mu(x,y)=\frac{a(x,y)}{A_\mu(x)}.
$$

If $A_\mu>0$ and

$$
\frac{\int\|D_x^na(x,y)\|\mu(dy)}{A_\mu(x)}\leq B_n,
$$

then

$$
\int\|D_x^nw_\mu(x,y)\|\mu(dy)\leq W_n,
$$

with exactly the recurrence and majorant of {prf:ref}`thm-cinf-normalized-majorant`. Uniform measurement derivatives give the same localized moment, variance, score, and fitness bounds as in the finite case, with sums replaced by integrals.
:::

:::{prf:proof}

Dominated differentiation gives $D_x^nA_\mu=\int D_x^na\,d\mu$. Differentiate $A_\mu w_\mu=a$, solve for the highest derivative of $w_\mu$, integrate its norm, and use $\int w_\mu\,d\mu=1$. This gives the same recursive inequality as the finite normalized-weight proof. Every subsequent moment formula involves products and integrals already controlled by these bounds, so dominated differentiation and the same Leibniz convolutions apply. For Gevrey-1, require convergent majorants, rather than merely a separate finite bound at each order.
:::

:::{prf:theorem} Smooth convergence of normalized empirical fields
:label: thm-cinf-empirical-field-convergence

Suppose $\mu_k\Rightarrow\mu$. Let $K$ be a compact parameter set with a neighborhood on which, for every fixed $n$, the functions

$$
D_x^na(x,\cdot),\qquad
D_x^n\bigl(a(x,\cdot)m(x,\cdot)\bigr)
$$

are bounded continuous in $y$, uniformly bounded for $x\in K$, and uniformly equicontinuous in $x$ in the supremum norm over $y$. Assume $\inf_{x\in K}A_\mu(x)>0$. Then

$$
\frac{\int a(x,y)m(x,y)\mu_k(dy)}{\int a(x,y)\mu_k(dy)}
\longrightarrow
\frac{\int a(x,y)m(x,y)\mu(dy)}{\int a(x,y)\mu(dy)}
$$

in $C^n(K)$ for every fixed $n$. The same conclusion holds for the subsequent regularized moment and fitness formulas if their constituent empirical integrals meet these hypotheses and the regularizers remain positive.
:::

:::{prf:proof}

For each derivative and each fixed $x$, weak convergence gives convergence of its integral. Uniform equicontinuity supplies a finite $\varepsilon$-net of parameter values whose supremum-norm errors control those at every $x\in K$. The integrals against both probability laws change by at most that supremum-norm error. Convergence at the finitely many net points therefore implies uniform convergence on $K$. Apply this argument to every derivative up to order $n$.

Uniform convergence of the denominators gives a common positive lower bound for all sufficiently large $k$. Starting with the zeroth-order quotient, differentiate the identity $A_kf_k=B_k$. Solving for $D^nf_k$ expresses it through $A_k^{-1}$, derivatives of $A_k,B_k$, and lower derivatives of $f_k$. Induction gives uniform convergence of all derivatives through order $n$. Products and smooth scalar compositions with positive regularizers preserve that convergence.
:::

:::{prf:remark} Identifying the limiting law is a separate step
:label: rem-cinf-field-versus-law-limit

The preceding theorem concerns a specified field evaluated against converging probability laws. For unbounded test functions its boundedness hypotheses can be replaced by explicit uniform-integrability and tail estimates for every derivative used. It does not prove convergence of the empirical particle laws, interchange an infinite-time limit with $N\to\infty$, or identify a continuous-time equation from a fixed-step BAOAB kernel. The existence, uniqueness, contraction, and stationary-limit arguments for a specified nonlinear evolution are proved in {doc}`09_propagation_chaos`; the continuous-time model and its mass balances are specified in {doc}`08_mean_field`.
:::

:::{prf:remark} Which object each proof controls
:label: rem-simplified-vs-full-final

| Object | Additional law calculation | Final analytic bound |
|:--|:--|:--|
| Fitness with a fixed sampled assignment | None; differentiate the selected smooth measurements | $n![t^n]\mathcal F$ |
| Fitness formed from expected measurements | Differentiate the specified companion law before forming moments | Reconstruct $\mathcal F$ using measurement majorant $\mathcal J\mathcal M_d$ |
| Expected sampled fitness, independent rows | Product of all row probabilities, including incoming probability factors | $n![t^n](\mathcal J_{\mathrm{ind}}\mathcal F)$ |
| Expected sampled fitness, idealized matching | Normalize the weights of complete matchings | $n![t^n](\mathcal H_s\mathcal F)$ |
| Expected sampled fitness, sequential greedy pairing | Sum the actual sequential histories and apply the remaining-set induction | $n![t^n](\mathcal H_s\mathcal F)$ |

The coincident upper bounds for the two matching mechanisms do not identify their distributions. An automatic-differentiation calculation that keeps sampled integer indices fixed computes the first row of this table, not the derivative of an expectation over a state-dependent law.
:::

:::{prf:remark} How the scales enter the constants
:label: rem-rho-tradeoffs

The Gaussian majorant at scale $s$ depends on $D_0t/s^2+t^2/(2s^2)$. Thus one explicit admissible radius satisfies

$$
D_0t_s/s^2+t_s^2/(2s^2)\leq\log(3/2).
$$

For fixed $D_0>0$ this choice has $t_s=O(s^2/D_0)$ as $s\downarrow0$. The localization scale $\rho$ enters in the same way. Independent rows additionally retain the incoming-mass factor $a_s^{-1}=\exp[D_0^2/(2s^2)]$. The distance bound uses a radius at most $\delta/4$; the variance composition requires $\mathcal T_+(t)<\sigma_{\min}^2$; real fitness powers require the positive floor $\eta$. These are distinct constraints and must be retained together.

Increasing population size does not worsen these particular bounds under their uniformity hypotheses. Sending a kernel scale or regularizer to zero can worsen them substantially. A density-volume factor cannot be inserted to reverse that conclusion without an additional argument controlling the normalized derivative ratios. The finite-order constants and numerical consistency conditions are recorded in {prf:ref}`prop-scaling-kv3` and {prf:ref}`cor-baoab-validity`.

All-orders smoothness remains a statement on the specified continuous stratum. Alive masks, nearest-image choices, hard caps, underflow fallbacks, and sampled index changes require their own branch analysis. Smoothness alone does not establish second-order weak accuracy of the complete Boris–BAOAB, cloning, and killing step; that requires the actual local-error and stability estimates described in {prf:ref}`rem-c3-implementation-branches` and the $C^3$ chapter.
:::

(sec-cinf-hypoellipticity-entropy)=
## 7. Kinetic regularity and entropy for an identified law

:::{div} feynman-prose
Why is smooth fitness useful to a kinetic equation? Noise acts directly on velocity. Transport then turns a change in velocity into a change in position. The bracket calculation below records this transfer in one line. It explains how a kinetic diffusion can smooth both variables even though its noise matrix vanishes in the position directions.

Entropy convergence asks another question: does the evolving law approach its stationary or survival-conditioned law, and at what rate? Smooth coefficients let us perform the derivative calculation. The logarithmic Sobolev inequality and the full dissipation estimate determine its sign. Those are separate pieces of analysis, and the convergence chapters supply them for their stated laws.
:::

:::{prf:theorem} Interior hypoellipticity of the specified kinetic diffusion
:label: thm-hypoellipticity-companion-dependent-full

On an open subset of $\mathbb R^{2dN}$, consider the continuous-time differential operator

$$
L=v\cdot\nabla_x+b(x,v)\cdot\nabla_v+
\frac{\sigma^2}{2}\Delta_v,\qquad \sigma>0,
$$

with a smooth drift $b$. The continuous fitness fields proved above may be used to construct such a drift wherever their branch and regularity hypotheses hold. The velocity noise fields and their first brackets with the transport drift span the entire tangent space. Consequently $L$, its formal adjoint, and their smooth zeroth-order perturbations are locally hypoelliptic in the interior.

In particular, if a stationary density or eigen-density satisfies $(L^*+c)q=f$ distributionally with smooth $c$ and smooth $f$, then $q$ is smooth in the interior. For a jump-diffusion eigen-equation, the jump term on the other side must be shown smooth, or a separate regularity argument for the full nonlocal operator must be supplied.
:::

:::{prf:proof}
:label: proof-thm-hypoellipticity-companion-dependent-full

Let $a=1,\ldots,dN$, $X_a=\sigma\partial_{v_a}$, and $Y=v\cdot\nabla_x+b\cdot\nabla_v$. Direct differentiation gives

$$
[X_a,Y]=\sigma\partial_{x_a}
+\sigma\sum_\ell(\partial_{v_a}b_\ell)\partial_{v_\ell}.
$$

The $X_a$ span all velocity directions. Subtracting their linear combinations from $[X_a,Y]$ gives every position direction. Thus the bracket-generating hypothesis of [Hörmander's theorem](https://doi.org/10.1007/BF02392081) holds. The formal adjoint has the same second-order fields and the opposite first-order drift, plus a smooth zeroth-order term; its brackets have the same span. Hörmander's local theorem gives the asserted hypoellipticity and regularity of distributional solutions with smooth right-hand sides.

This computation is for the stated differential operator. It does not replace a regularity proof for a discrete split kernel, a nonsmooth velocity cap, or a nonlocal cloning source. The kinetic smoothing hypotheses used for the nonlinear stationary equation are detailed in {prf:ref}`thm-uniqueness-hypoelliptic-regularity`.
:::

:::{prf:theorem} Logarithmic Sobolev inequality for the actual joint law
:label: thm-lsi-companion-dependent-full

Let $\pi_N$ be a specified continuous joint invariant law or QSD satisfying one of the proved structural criteria in {prf:ref}`cor-n-uniform-lsi`: a product reference, a bounded tilt of the entire reference law, uniform joint curvature, or a contractive additive-noise invariant flow. Let its full-gradient LSI constant be $C_N$ in the convention

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq2C_N\int\|\nabla f\|^2\,d\pi_N.
$$

If a positive matrix field $A_N\succeq a_*I$ is used in the gradient form, then

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq\frac{2C_N}{a_*}
\int(\nabla f)^{\mathsf T}A_N\nabla f\,d\pi_N.
$$

Uniform bounds on $C_N$ and $a_*^{-1}$ give a population-uniform inequality. They concern the specified law, not an unidentified Gibbs surrogate.
:::

:::{prf:proof}

The complete product, perturbation, curvature, and invariant-flow arguments are proved in {prf:ref}`cor-n-uniform-lsi`. Apply the criterion that holds for $\pi_N$. Then use $\|\nabla f\|^2\leq a_*^{-1}(\nabla f)^{\mathsf T}A_N\nabla f$ and integrate. This is the metric comparison in {prf:ref}`thm-gg-lsi-main`.
:::

:::{prf:remark} The gradient must see every entropy-bearing variable
:label: rem-cinf-full-gradient-lsi

A velocity-only gradient form vanishes on every function depending only on position. Such a form cannot bound the entropy of a nonconstant position function by an ordinary LSI. Likewise, a continuous gradient vanishes on functions depending only on discrete alive/dead status; a law carrying such strata needs an additional status entropy and form. The law distinctions are defined in {prf:ref}`def-kl-finite-particle-laws`. Kinetic hypocoercivity handles transport and velocity dissipation through a modified functional rather than by asserting a velocity-only LSI for the full law.
:::

:::{prf:corollary} Entropy convergence after the full dissipation estimate
:label: cor-exponential-qsd-companion-dependent-full

Let $\pi_N$ be the invariant law or QSD of a specified continuous evolution, and let $h_t=d\mu_t/d\pi_N$ for its conservative or survival-normalized law. Suppose the preceding full-gradient LSI holds. Define

$$
H(h)=\int h\log h\,d\pi_N,\qquad
I(h)=\int\frac{\|\nabla h\|^2}{h}\,d\pi_N,
$$

$$
I_G(h)=\int\frac{(\nabla h)^{\mathsf T}G_N\nabla h}{h}\,d\pi_N,
\qquad \Phi_G=H+I_G,
\qquad 0\prec G_N\preceq g_{+,N}I.
$$

Assume the actual full evolution, including cloning and killing normalization when present, satisfies

$$
\frac{d}{dt}\Phi_G(h_t)\leq-\delta_N I(h_t),\qquad\delta_N>0.
$$

Then

$$
H(h_t)\leq\Phi_G(h_t)
\leq\exp\!\left[-\frac{\delta_Nt}{C_N/2+g_{+,N}}\right]\Phi_G(h_0).
$$

The rate is uniform in $N$ when $C_N$ and $g_{+,N}$ are uniformly bounded above and $\delta_N$ is uniformly bounded below by a positive number.
:::

:::{prf:proof}
:label: proof-cor-exponential-qsd-companion-dependent-cinf

Apply the LSI to $\sqrt h$: since $\|\nabla\sqrt h\|^2=\|\nabla h\|^2/(4h)$ and $\int h\,d\pi_N=1$, it gives $H(h)\leq C_NI(h)/2$. The matrix bound gives $I_G\leq g_{+,N}I$. Hence $\Phi_G\leq(C_N/2+g_{+,N})I$. Substitute this into the assumed derivative inequality and apply Grönwall. The nonnegativity of $I_G$ gives $H\leq\Phi_G$.

This is the application of {prf:ref}`thm-kl-convergence-euclidean` to the law and field under consideration. Its canonical proofs include the conservative kinetic and common-target cloning cases, as well as the exact normalized killing and boundary entropy identities in {prf:ref}`prop-kl-conditioned-entropy` and {prf:ref}`prop-kl-boundary-entropy`. The separate $L^2$ conclusion under coercivity of the matching form is proved in {prf:ref}`proof-cor-exponential-qsd-companion-dependent-full`. The regularity estimates in this chapter justify derivative operations and supply coefficient bounds; the sign of the full derivative must come from those dynamical estimates.
:::
