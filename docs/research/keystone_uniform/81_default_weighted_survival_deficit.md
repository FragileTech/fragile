# Source-weighted survival and position-budget losses at the harmonic default

(sec-wsd81-register)=
## 1. Actual coupled plans and each own output restriction

:::{prf:definition} Coupled source weights before fresh recipient jitter
:label: def-wsd81-register

Retain the actual finite harmonic count update of
{prf:ref}`def-dsa-survival-record` at the original default in
{prf:ref}`cor-dsa-original-reference`. Let $N\ge2$. Couple two
consistent positive-alive entering arrays and their complete actual
measurement, acceptance, donor, revival, component and original-slot
Haar plans before the independent fresh recipient jitters. No velocity
is copied from a donor. Conditional on this complete coupled plan
$\mathscr A$, use shared independent $J_i\sim N(0,.01I_3)$, and
then shared original independent OU and final-position arrays.
Each individual marginal is its actual transition.

At interpolation parameter $\theta\in[0,1]$, put

$$
X_{\theta,i}=S_{\theta,i}+I_{\theta,i}J_i,\qquad
r_i=\delta S_i+\delta I_iJ_i,\qquad p_i=\delta P_i,
\tag{WSD81.1}
$$

where $S_{\theta,i}\in[-2,2]^3$, $0\le I_{\theta,i}\le1$,
and $P_{\theta,i},p_i,\delta S_i,\delta I_i$ are fixed by
$\mathscr A$. The endpoint indicators are the actual copied/revived
bits; interpolation does not change either endpoint algorithm.
Assume $|P_{\theta,i}|\le4$. Write $\langle\cdot\rangle_N$ for
the normalized row average, and define

$$
K_J=\langle |r|^2+|p|^2\rangle_N,\qquad
K_{\mathscr A}=\langle |\delta S|^2+.03\delta I^2+|p|^2\rangle_N,
\qquad \mathbb E_J K_J=K_{\mathscr A}.
\tag{WSD81.2}
$$

Let $\mathcal S_k$ be the actual next nonextinction event for
endpoint $k\in\{0,1\}$. Let $a_{\rm ret}>0$ be the exact
Gaussian-integral floor (DSA.3), set $\epsilon=1-a_{\rm ret}$,
and write $p_k=\Pr(\mathcal S_k)\ge1-\epsilon^N$.
All restrictions below divide by their own $p_k$. A statement about
a test-vector deficit under $\mathcal S_k$ does not construct a
coupling of the two separately normalized alive laws.
:::

(sec-wsd81-extinction)=
## 2. The missing mixed extinction charge

:::{prf:lemma} An affine source displacement loses at most one return factor
:label: lem-wsd81-source-extinction

For either endpoint's actual nonextinction event,

$$
\mathbb E[K_J\mathbf1_{\mathcal S_k^c}\mid\mathscr A]
\le\epsilon^{N-1}K_{\mathscr A},\qquad
\mathbb E[K_J\mathbf1_{\mathcal S_k^c}]
\le\epsilon^{N-1}\mathbb E K_J.
\tag{WSD81.3}
$$

The weights may correlate with their own recipient jitter, their
entering array, all source choices and component/Haar outcomes.
No bound of this form is asserted for an arbitrary pre-OU random
weight depending on the entire recipient-jitter array.
:::

:::{prf:proof}

For the actual endpoint $k$, use the row safe-return events
$E_{k,i}$ in (DSA.6). Conditional on $\mathscr A$, they are
independent over rows, each has probability at least $a_{\rm ret}$,
and each implies that row's actual terminal alive mark, irrespective
of the globally dependent first count average. These events depend
only on row $i$'s own $J_i,\xi_i,\zeta_i$ and its fixed source.
Hence $\mathcal S_k^c\subseteq\bigcap_j E_{k,j}^c$.

The nonnegative row weight $k_i=|r_i|^2+|p_i|^2$ depends only
on $J_i$ after fixing $\mathscr A$. Discard its own event factor:

$$
\begin{split}
\mathbb E[k_i\mathbf1_{\mathcal S_k^c}\mid\mathscr A]
&\le \mathbb E\left[k_i\prod_{j\ne i}\mathbf1_{E_{k,j}^c}
                         \mid\mathscr A\right]\\
&=\mathbb E_J[k_i\mid\mathscr A]
            \prod_{j\ne i}\Pr(E_{k,j}^c\mid\mathscr A)\\
&\le\epsilon^{N-1}
             (|\delta S_i|^2+.03\delta I_i^2+|p_i|^2).
\end{split}
\tag{WSD81.4}
$$

Average the rows and then integrate the complete plan. The actual
first and second viscous fields can depend on every noise row;
their independence has not been used. No noise is conditioned on
survival before this calculation. All Gaussian outcomes are included.
:::

(sec-wsd81-fixed-vectors)=
## 3. Complete first-provider test vectors remain source controlled

:::{prf:lemma} Pathwise whole-array control of the bare fixed vectors
:label: lem-wsd81-bare-vectors

At the actual interpolated own finite count field let

$$
A_1=I-aL_1,\quad B_1=-L_{\dot k_1}P_\theta,\quad
E=A_1p+aB_1,\quad R=a_xr+bE,\quad Z_b=z_vE-r_Hr,
\tag{WSD81.5}
$$

where $a=.006$, $c=e^{-.04}$, $b=.02(1+c)$,
$a_x=1-.02b$, $z_v=c-.02b$ and $r_H=(1-.02^2)b$.
These are fixed before the fresh OU array, although they depend
on the entire recipient-jitter array. Put
$H_J=\langle |R|^2+|Z_b|^2\rangle_N$. Pathwise,

$$
\frac25K_J\le H_J\le\frac{109}{100}K_J.
\tag{WSD81.6}
$$

Consequently the actual own extinction restriction satisfies

$$
\mathbb E[H_J\mathbf1_{\mathcal S_k^c}]
\le\frac{109}{100}\epsilon^{N-1}\mathbb E K_J.
\tag{WSD81.7}
$$

All inequalities are for normalized whole-array Hilbert norms,
not separate row bounds on $B_1$.
:::

:::{prf:proof}

The actual symmetric count Laplacian has spectrum in $[0,1]$.
Thus $\|A_1\|\le1$ and $\|A_1^{-1}\|\le1/.994$.
The exact centered operator (FFO.6), with the pathwise empirical
mean speed at most four, gives
$\|B_1\|_{2,N}\le8\ell\|r\|_{2,N}\le4.856\|r\|_{2,N}$,
with the strict scalar bound $8\ell<4.856$.
This applies to every realized recipient-jitter array and does
not insert an averaged velocity moment into a local product.

Using $.96<c<.9608$, all scalar factors are positive. With
$u=\|r\|_{2,N}$, $v=\|p\|_{2,N}$,

$$
\begin{pmatrix}\|R\|_{2,N}\\\|Z_b\|_{2,N}\end{pmatrix}
\le
M\begin{pmatrix}u\\v\end{pmatrix},\qquad
M=\begin{pmatrix}
.999216+.039216(.006)(4.856)&.039216\\
.039216+.9608(.006)(4.856)&.9608
\end{pmatrix}.
\tag{WSD81.8}
$$

The rational certificate below proves
$M^{\mathsf T}M\le(109/100)I$, yielding the upper bound.

For the lower bound, set $k=.006(4.856)$ and $d=.994$.
From $p=A_1^{-1}(E-aB_1)$,

$$
\|r\|_{2,N}^2+\|p\|_{2,N}^2
\le u^2+d^{-2}(\|E\|_{2,N}+ku)^2
\le\frac{11}{10}(\|r\|_{2,N}^2+\|E\|_{2,N}^2).
\tag{WSD81.9}
$$

The second comparison is an exact two-by-two positive matrix
check, recorded below. The harmonic matrix has determinant exactly
$a_xz_v+br_H=c$, since $r_H=.02(c+a_x)$. Thus

$$
r=c^{-1}(z_vR-bZ_b),\qquad
E=c^{-1}(r_HR+a_xZ_b).
\tag{WSD81.10}
$$

Its inverse squared Frobenius norm is at most
$(.9608^2+2(.039216)^2+1)/.96^2<21/10$.
Equations (WSD81.9)--(WSD81.10) give $K_J\le(231/100)H_J$,
which implies the weaker lower bound in (WSD81.6).
No commutation of the first count derivative and its spatial
response has been assumed. Finally combine the pathwise upper
bound with (WSD81.3) to prove (WSD81.7).
:::

(sec-wsd81-position-budget)=
## 4. Full-Gaussian weighted empirical position localization

:::{prf:lemma} A position-budget exception carries an exponential source weight
:label: lem-wsd81-position-exception

Let $\mathcal G_{x,\theta}=
\{\langle|X_\theta|^2\rangle_N\le12.25\}$. Then

$$
\mathbb E[K_J\mathbf1_{\mathcal G_{x,\theta}^c}\mid\mathscr A]
\le2e^{-19N/1000}K_{\mathscr A},\qquad
\mathbb E[H_J\mathbf1_{\mathcal G_{x,\theta}^c}]
\le\frac{109}{50}e^{-19N/1000}\mathbb E K_J.
\tag{WSD81.11}
$$

This is a mixed displacement/exception estimate. It uses neither
a fourth displacement moment nor a product of an exception
probability and an arbitrarily correlated displacement.
:::

:::{prf:proof}

Fix $\mathscr A$ and set $\lambda=1/10$, $\sigma^2=1/100$.
For a single row, put $X=S+IJ$, $|S|^2\le12$, $I\in[0,1]$.
Completing the Gaussian square gives

$$
M(S,I):=\mathbb E e^{\lambda|X|^2}
=(1-2\lambda\sigma^2I^2)^{-3/2}
  \exp\!\left(\frac{\lambda|S|^2}{1-2\lambda\sigma^2I^2}\right).
\tag{WSD81.12}
$$

Consequently, with $d_*=499/500$,

$$
\log M(S,I)\le
\frac32\frac{1/500}{d_*}+\frac{6/5}{d_*}
=\frac{1203}{998}<\frac{603}{500}=1.206.
\tag{WSD81.13}
$$

Here $-\log(1-u)\le u/(1-u)$ follows by integrating
$(1-u)^{-1}$ on $[0,u]$; this is an analytic bound on the
entire Gaussian integral.

Under the exponential tilt in (WSD81.12), the same original
$J$ has mean $h=2\lambda\sigma^2IS/(1-2\lambda\sigma^2I^2)$
and covariance $\sigma^2I_3/(1-2\lambda\sigma^2I^2)$.
Therefore, for its actual affine row displacement,

$$
\begin{split}
\frac{\mathbb E[|\delta S+\delta I J|^2e^{\lambda|X|^2}]}{M(S,I)}
&=|\delta S+\delta I h|^2
  +\frac{.03\delta I^2}{1-2\lambda\sigma^2I^2}\\
&\le2|\delta S|^2+
\left[\frac{2(.002)^2(12)}{d_*^2}+\frac{.03}{d_*}\right]\delta I^2\\
&\le2(|\delta S|^2+.03\delta I^2).
\end{split}
\tag{WSD81.14}
$$

The last comparison is rational. The fixed $|p|^2$ term has
tilted expectation exactly $|p|^2$, so each full row score
has tilted expectation at most twice its unweighted expectation.
Independence of the recipient jitters over rows now gives

$$
\mathbb E_J\left[K_Je^{\lambda\sum_i|X_{\theta,i}|^2}
                     \mid\mathscr A\right]
\le2e^{1.206N}K_{\mathscr A}.
\tag{WSD81.15}
$$

The event in (WSD81.11) requires the sum to exceed $12.25N$.
Multiply (WSD81.15) by $e^{-1.225N}$. This yields its first
bound. Use the pathwise $H_J\le1.09K_J$ and integrate to
obtain the second. The local displacement and local position
have been tilted jointly, retaining their covariance and every
large recipient jitter.
:::

(sec-wsd81-surviving-deficit)=
## 5. A cap deficit survives each own finite nonextinction division

:::{prf:theorem} Source-weighted current-survivor cap deficit
:label: thm-wsd81-surviving-deficit

In addition to {prf:ref}`def-wsd81-register`, assume
$\langle|P_\theta|^2\rangle_N\le.56^2$ for every coupled
plan and comparison point. At each comparison point use the
actual complete own second count graph and native-cap derivative
$D_\theta=DC_V(z_\theta)$, with the original uncapped OU array.
Define its whole-array fixed-vector deficit

$$
\mathcal D_J=H_J-
  \langle|D_\theta R|^2+|D_\theta Z_b|^2\rangle_N\ge0.
\tag{WSD81.16}
$$

For each $k$ separately,

$$
\begin{split}
\mathbb E[\mathcal D_J\mid\mathcal S_k]
\ge{}&\frac{41}{200}\mathbb E[H_J\mid\mathcal S_k]\\
&-\frac{109}{100p_k}
 \left[\frac{159}{200}\epsilon^{N-1}
                 +\frac{41}{100}e^{-19N/1000}\right]\mathbb E K_J.
\end{split}
\tag{WSD81.17}
$$

In particular, in the explicit nonempty population-size regime

$$
N\ge N_*:=\max\!\left\{2,
1+\left\lceil\frac{\log100}{-\log(1-a_{\rm ret})}\right\rceil,
\left\lceil\frac{1000\log100}{19}\right\rceil\right\},
\tag{WSD81.18}
$$

the uniform coefficient

$$
\mathbb E[\mathcal D_J\mid\mathcal S_k]
\ge\frac{41}{400}\mathbb E[H_J\mid\mathcal S_k]
\tag{WSD81.19}
$$

holds without a particle floor for this deficit observable.
The constant $41/400$ is independent of $N$. The entering
low-speed class and this derivative deficit are not asserted
invariant or sufficient for alive-law convergence.
:::

:::{prf:proof}

The population-size-independent hypotheses of (GCA74.4) hold
on $\mathcal G_{x,\theta}$: the prepared speed budget is
assumed for every plan, and the position budget is its definition.
Both $R$ and $Z_b$ are fixed before fresh OU. Sum the exact
own-restriction inequality (GCA74.17) for these two vectors,
using normalized row averages. This gives

$$
\mathbb E[\mathcal D_J\mid\mathcal S_k]
\ge\frac{41}{200}\mathbb E[H_J\mid\mathcal S_k]
-\frac{\frac{159}{200}\mathbb E[H_J\mathbf1_{\mathcal S_k^c}]
       +\frac{41}{200}\mathbb E[H_J\mathbf1_{\mathcal G_{x,\theta}^c}]}
      {p_k}.
\tag{WSD81.20}
$$

Apply (WSD81.7) and (WSD81.11) to obtain (WSD81.17).
These mixed estimates are taken under the raw actual Gaussian
proposal before the own restriction. The restricted innovations
have not been replaced by independent Gaussians.

For $N\ge N_*$, both $\epsilon^{N-1}$ and $e^{-19N/1000}$
are at most $1/100$. Also (WSD81.3) and the lower bound in
(WSD81.6) give

$$
\mathbb E[H_J\mid\mathcal S_k]
\ge\frac{2}{5p_k}(1-\epsilon^{N-1})\mathbb E K_J.
\tag{WSD81.21}
$$

The exact rational comparison

$$
\frac{109}{100}\left[\frac{159}{20000}+\frac{41}{10000}\right]
<\frac{41}{400}\frac25\frac{99}{100}
\tag{WSD81.22}
$$

shows that the loss in (WSD81.17) is at most half its retained
deficit. This proves (WSD81.19), including zero displacement.
The source-floor integral is strictly positive, so $N_*$ is
finite. Its conservative value is not a practical certification
of the default at ordinary population sizes.
:::

:::{prf:corollary} Actual low-speed entering arrays supply the preparation budget
:label: cor-wsd81-actual-preparation-budget

If both coupled actual entering capped arrays satisfy
$\langle|v_k|^2\rangle_N\le.56^2$ pathwise, then every
accepted preparation/component/Haar plan supplies the speed
budget in {prf:ref}`thm-wsd81-surviving-deficit`.
This holds at arbitrary configured fitness powers and alive
fractions, with unrestricted retained dead positions.
:::

:::{prf:proof}

The exact original-slot component energy (RVB.1) decreases
the total squared speed for each actual component forest and
Haar realization. The prepared speed is at most four pathwise.
Thus each endpoint prepared array satisfies the declared norm
bound before recipient jitter. Convexity of the squared Hilbert
norm gives the same bound for $P_\theta$ at every comparison
point. Neither donor-velocity copying nor a fitness variance
lower bound is used. Source positions are alive donor or alive
persistent positions, so the source-box hypothesis is retained.
:::

:::{prf:remark} Exact scope of the new normalization repair
:label: rem-wsd81-scope

The new estimates discharge the previously unbounded mixed
extinction and empirical-position-budget charges for the actual
affine source-plan displacements and their complete first-provider
fixed vectors. The $N-1$ return power is justified by dropping
one row's event, rather than multiplying an arbitrary correlated
weight by an extinction probability. The position exception uses
its exact Gaussian tilt, rather than conditioning future noise on
a moment-good event. The prepared low-speed hypothesis is an
actual pathwise entering class; an averaged burn bound alone does
not imply it. If that hypothesis is relaxed, the additional weighted
speed-budget exception in (GCA74.17) remains.

The actual cap-account test $Z_b$ separately has the same mixed
loss control from (WSD81.20), with its own weighted squared norm.
The stronger pair deficit (WSD81.19) does not replace the signed
bare cap cross term or control the full OU-dependent second
force $F_2$. It does not couple two separately surviving empirical
readouts, compare their accepted preparations, control terminal
alive-count mismatch, or prove an unrestricted default delayed
law rate. A population root-alive restriction also has no
$N-1$ power; that distinct marked-law account remains necessary.
:::

(sec-wsd81-certificate)=
## 6. Exact rational comparisons

```python
from fractions import Fraction as F

a, k1, b, c_hi = F('.006'), F('4.856'), F('.039216'), F('.9608')
Ar = F('.999216') + b * a * k1
Cr = b + c_hi * a * k1
upper = F(109, 100)
aa = upper - Ar**2 - Cr**2
dd = upper - b**2 - c_hi**2
off = Ar * b + Cr * c_hi
assert aa > 0 and dd > 0 and aa * dd > off**2

d, k, upper_first = F('.994'), a * k1, F(11, 10)
aa = upper_first - 1 - k**2 / d**2
dd = upper_first - 1 / d**2
off = k / d**2
assert aa > 0 and dd > 0 and aa * dd > off**2
assert (c_hi**2 + 2 * b**2 + 1) / F('.96')**2 < F(21, 10)
assert F(11, 10) * F(21, 10) < F(5, 2)

dstar = F(499, 500)
assert F(1203, 998) < F(603, 500)
assert 2 * F('.002')**2 * 12 / dstar**2 + F('.03') / dstar <= F('.06')
assert F('1.225') - F('1.206') == F(19, 1000)
loss = F(109, 100) * (F(159, 20000) + F(41, 10000))
retained_half = F(41, 400) * F(2, 5) * F(99, 100)
assert loss < retained_half
print('All source-weighted extinction, Gaussian tilt and cap-deficit checks pass.')
```
