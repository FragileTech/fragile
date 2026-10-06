# A weighted global Hamming estimate for actual finite preparation

The finite preparation coupling of
{prf:ref}`lem-slcw-finite-preparation` extends to a cost that charges
unbounded common positions. The calculation below keeps the covariance
between the affected-label count and copied-source energy. It establishes
a preparation estimate, rather than a kinetic reset or full-law theorem.

## 1. The weighted cost and the assumptions

Use the conservative, all-alive finite preparation with bounded raw reward,
the actual sampled fitness normalizers, and the primitive constants
$a_*,c_*,L_f$ from the cited lemma. Require $a_*\le1/4$. Let
$\sigma_J$ be the activated positional jitter standard deviation. Let
$(Y_i,U_i)$ consist of each row's copied or retained position and its
actual frozen collision velocity before positional jitter. In the
canonical kernel $U_i=v_i$: donor velocities are not copied.
Assume the actual component collision contracts this frozen velocity
energy, as does the
canonical shared orthogonal relative-velocity rotation with
$|\alpha_{\rm col}|\le1$. On every actual component $C$,

$$
\sum_{i\in C}|v_i^{\rm prep}|^2\le\sum_{i\in C}|U_i|^2.
\tag{KUWP.1}
$$

The full donor phase weights below deliberately overcharge donor velocity
energy while retaining the exact original-slot collision law.

There is no compact-support assumption on positions or jitter. Set

$$
\begin{aligned}
\rho_N(S,T)&=N^{-1}\#\{i:(x_i,v_i)\ne(x_i',v_i')\},\\
\mathsf M_2(S)&=N^{-1}\sum_i(|x_i|^2+|v_i|^2),\\
\mathsf d_{N,\eta}(S,T)
&=\rho_N(S,T)[1+\eta\mathsf M_2(S)+\eta\mathsf M_2(T)],
\qquad\eta>0.
\end{aligned}
\tag{KUWP.2}
$$

This is a measurable symmetric distance-like cost, not a metric. For example,
with $N=2$, zero velocities and one spatial coordinate, take
$S=(R,0)$, $T=(0,R)$ and $U=(0,0)$. The $S,T$ cost is
$1+\eta R^2$, while the sum of the $S,U$ and $U,T$ costs is
$1+\eta R^2/2$. Thus a transport triangle inequality or a metric-space
Banach theorem cannot be invoked solely from (KUWP.2).

Define the population-independent constants

$$
\begin{aligned}
M&=e^{8c_*},\\
C_H&=M(1+3L_f),\\
C_J&=Mc_*(20+42L_f),\\
C_E&=C_H+C_J,\\
C_{\rm prep}(\eta)
&=\max\{C_H+2\eta d\sigma_J^2C_J,\ C_E\}.
\end{aligned}
\tag{KUWP.3}
$$

Then the actual coupled complete preparations satisfy

$$
\begin{aligned}
\mathbb E\rho_N(S^{\rm prep},T^{\rm prep})
&\le C_H\rho_N(S,T),\\
\mathbb E[\rho_N(S^{\rm prep},T^{\rm prep})
                 \mathsf M_2(S^{\rm prep})]
&\le C_E\rho_N(S,T)\mathsf M_2(S)
      +d\sigma_J^2C_J\rho_N(S,T),\\
\mathbb E\mathsf d_{N,\eta}(S^{\rm prep},T^{\rm prep})
&\le C_{\rm prep}(\eta)\mathsf d_{N,\eta}(S,T).
\end{aligned}
\tag{KUWP.4}
$$

The second inequality also holds with $S$ replaced by $T$. Along the
positive-exponent scaling of the bounded-reward regime,
$c_*=O(\theta)$ and $L_f=O(\theta)$, hence
$C_{\rm prep}(\eta)=1+O(\theta)$ for every fixed $\eta$.
All constants are independent of $N\ge2$.

## 2. Conditional common forests and forcing one edge

Write $A$ for the entering physical mismatch set and $r=|A|/N$.
If $r=0$, couple every draw identically, so every assertion is exact with
zero left side. Suppose $r>0$; then $1/N\le r$.

Freeze both complete measurement arrays. Independently across rows,
maximally couple their outgoing accepted-donor/no-edge tokens. Let $H$
be the set of unequal tokens, let $q_i$ be each token failure probability,
and put $\bar q=N^{-1}\sum_iq_i$. The finite preparation proof gives

$$
q_i\le2a_*\le1/2,\qquad
\mathbb E_{\rm meas}\bar q\le L_fr.
\tag{KUWP.5}
$$

Condition further on $H$ and on both tokens of its rows. Remove their
outgoing edges. The remaining common tokens are independent rows, with
each specified donor probability at most $4c_*/N$. Their accepted graph
is a forest ordered by the first swarm's frozen fitness. The actual
ordered-path estimate gives mean component size at most $M$ for every
specified vertex, including under removal of additional outgoing rows.

Let $\mathcal C(i)$ be its common component. Form the seed list containing
the vertices of $A$ and, for every exceptional row, that row and its two
possible donor labels. Counting repeated labels is harmless. Its length
is at most

$$
K=|A|+3|H|.
$$

Let $F$ be the sum of common-component sizes over that list. Every
potentially unequal prepared row is contained in their union, so
$\rho_N(S^{\rm prep},T^{\rm prep})\le F/N$, and conditionally
$\mathbb EF\le MK$.

For a common row $i\notin H$, force its common accepted token to be
$i\to j$, first removing its outgoing row from the remaining forest.
The other common rows keep their independent conditional laws.
Restoring this one edge merges at most two components. For any specified
seed $b$, its restored component is contained in the union of the
three pre-restoration components of $b,i,j$. Therefore its conditional
mean is at most $3M$, and

$$
\mathbb E[F\mathbf1_{\{i\to j\}}\mid H,\text{exceptional tokens},
                                            \text{measurements}]
\le\frac{4c_*}{N}\,3MK.
\tag{KUWP.6}
$$

If that token has probability zero the inequality is immediate.
This forcing argument does not assert independence between $F$ and an
accepted edge.

## 3. The mixed copying and affected-count moment

In either own swarm use the actual source/frozen-velocity pairs
$(Y_i,U_i)$ and the full deterministic input donor weights
$W_j(S)=|x_j|^2+|v_j|^2$. For the common tokens, sum (KUWP.6) over
recipients and donors with weight $W_j(S)$, and then divide by $N^2$.
This gives, conditionally,

$$
\mathbb E\left[\frac FN\frac1N
       \sum_{i\notin H,\ i\to j}W_j(S)\right]
\le12Mc_*\frac KN\,\mathsf M_2(S),
\tag{KUWP.7}
$$

For exceptional tokens define $J_{ij}$ to be the event that row $i$ has
unequal paired tokens and its own accepted donor is $j$. Conditional on
the measurement arrays, different coupled token rows are independent.
Since $J_{ij}$ implies $i\in H$,

$$
\mathbb E[K\mathbf1_{J_{ij}}\mid\text{measurements}]
\le(|A|+3+3N\bar q)\Pr(J_{ij}\mid\text{measurements}).
\tag{KUWP.8}
$$

The event $J_{ij}$ is a subset of its own accepted-donor event, whose
probability is at most $c_* /(N-1)$. After summing over $i,j$, using
$N/(N-1)\le2$, and applying the conditional bound $\mathbb EF\le MK$,

$$
\mathbb E\left[\frac FN\frac1N
        \sum_{i\in H,\ i\to j}W_j(S)\,
        \middle|\,\text{measurements}\right]
\le2Mc_*(4r+3\bar q)\mathsf M_2(S).
\tag{KUWP.9}
$$

Here $|A|/N+3/N+3\bar q\le4r+3\bar q$. That use of $r>0$
is why the zero-mismatch case was handled separately.

Adding (KUWP.7), averaging (KUWP.5), and using $\mathbb EK/N\le
(1+3L_f)r$ proves

$$
\mathbb E\left[\frac FN\frac1N
                  \sum_{i\to j}W_j(S)\right]
\le C_Jr\mathsf M_2(S).
\tag{KUWP.10}
$$

The same proof with every donor weight replaced by one gives

$$
\mathbb E\left[\frac FN\frac{\#\text{accepted own tokens}}N\right]
\le C_Jr.
\tag{KUWP.11}
$$

These two estimates control the needed graph-energy covariance.
The persistent/copying source identity implies pointwise

$$
\frac1N\sum_i(|Y_i|^2+|U_i|^2)
\le\mathsf M_2(S)+\frac1N\sum_{i\to j}W_j(S).
$$

Own activated jitters have their exact centered independent Gaussian law
conditional on both graphs. Consequently their conditional contribution
to the average squared positional moment is
$d\sigma_J^2\#\text{accepted own tokens}/N$.
The component rotation estimate (KUWP.1) is pointwise and contracts
the original-slot frozen velocities in that source-phase sum. Multiply these
identities by $F/N$ and use (KUWP.10)--(KUWP.11). This proves the second
line of (KUWP.4), with full donor phase energy charged as an upper bound.
The first follows from $\mathbb EF/N\le C_Hr$.
Add the two own mixed-moment bounds and apply the two maxima in (KUWP.3)
to prove its third line.

## 4. The remaining kinetic obligation

This estimate supplies an actual finite-preparation factor approaching
one as selection weakens. It includes changed global normalizers,
exceptional donor endpoints, component collisions and unbounded jitter;
no independent product of count and energy estimates was used.

A complete nonresonant-step active-law argument still needs a kinetic
block contraction in this distance-like cost, or a compatible weighted
Harris theorem with its hypotheses verified. The small-set endpoint reset
and the future output weight share innovation paths; their probabilities
and moments cannot simply be multiplied. The endpoint bridge must retain
each own Gaussian marginal after the second actual preparation.

Dense viscosity adds a further requirement: its two kicks couple all rows,
so the independent-row reference endpoint construction is not a theorem
for the dense kernel. The completed quadratic dense-kinetic result in
`15_dense_block_feedback.md` supplies a different proved component. It
does not by itself supply the needed Hamming reset. Death and future
survival conditioning likewise retain the separate complete-block
requirements stated in `16_survivor_law_block_transfer.md`.

## 5. Why the global weight alone cannot supply a kinetic reset

For the clone-disabled, nonviscous harmonic reference, suppose
$0<a_x<1$. At each update its position satisfies
$x_{n+1}=a_xx_n+Bv_n+tq\xi_n+s\chi_n$, with $|v_n|\le V$.
For every fixed block length $m$, iteration gives

$$
x_m=a_x^mx_0+\mathcal R_m+\mathcal Z_m,
\qquad
|\mathcal R_m|\le BV\sum_{k=0}^{m-1}a_x^k,
\tag{KUWP.12}
$$

where $\mathcal Z_m$ is the linear Gaussian innovation sum, with
variance independent of $N,x_0$. The remainder need not be independent
of this Gaussian sum; the pointwise bound suffices.

Take $T_N=0$ and let $S_N$ have only row one at
$R_Ne_1$, zero velocities, with $R_N=N^{1/4}$. The input global-weight
cost is $N^{-1}(1+\eta N^{-1/2})$ and its averaged energy tends to zero.
By (KUWP.12), the probabilities of the event
$x_{m,1}>a_x^mR_N/2$ in the two row-one output laws tend respectively
to one and zero. Their TV distance therefore tends to one. Every
coupling of the full output arrays has Hamming cost at least that TV
distance divided by $N$, and the weighted cost is no smaller. Hence
the ratio of its optimal output cost to its input cost has lower limit
at least one. There is no strict fixed-block population-uniform
contraction in the cost (KUWP.2) for this reference on arbitrary arrays.

This obstructs precisely the proposed global-weight reference reset.
It does not obstruct the preparation estimate (KUWP.4), or a cost
that separately charges the energy of the mismatched row.

## 6. Preparation in a local-plus-global weighted cost

Use the same actual preparation assumptions. Write
$W_i(S)=|x_i|^2+|v_i|^2$ and
$\mathsf L_A(S)=N^{-1}\sum_{i\in A}W_i(S)$. For fixed
$\eta,g>0$ consider the refined distance-like cost

$$
\begin{aligned}
\mathsf D_{N,\eta,g}(S,T)
&=\frac1N\sum_{i\in A}[1+\eta W_i(S)+\eta W_i(T)]\\
&\quad+g\rho_N(S,T)[\mathsf M_2(S)+\mathsf M_2(T)]\\
&=r+\eta[\mathsf L_A(S)+\mathsf L_A(T)]
       +gr[\mathsf M_2(S)+\mathsf M_2(T)].
\end{aligned}
\tag{KUWP.13}
$$

The local term charges the isolated far mismatch in Section 5. The
global term charges changed-normalizer feedback into far common rows.
Define the following conservative constants from the same primitive register:

$$
\begin{aligned}
C&=4c_*,\qquad J=(2C+C^2)e^{2C},\qquad K_f=1+3L_f,\\
D_r&=2a_*/\kappa_C^2+2L_gC_r^f/\kappa_C,\\
D_m&=2L_gC_s^f/\kappa_C,\\
D_G&=2c_*K_D^f+8c_*/\kappa_D+2D_r+2D_mK_D^f,\\
S_L&=2a_*+4c_*,\qquad S_G=L_f+2D_G,\\
P_L&=C(1+S_L)+2c_*,\\
P_G&=C[S_G+(1+2J)K_f]+D_G,\\
P_J&=P_L+P_G,\\
A_L&=1+S_L+P_L,\qquad A_G=S_G+JK_f+P_G,\\
C_{\rm comb}(\eta,g)
&=\max\{C_H+2d\sigma_J^2(\eta P_J+gC_J),\
          A_L,\ C_E+(\eta/g)A_G\}.
\end{aligned}
\tag{KUWP.14}
$$

Here $K_D^f,C_r^f,C_s^f$ are exactly the finite-preparation constants,
not replacements for the actual empirical normalizers. Then

$$
\mathbb E\mathsf D_{N,\eta,g}(S^{\rm prep},T^{\rm prep})
\le C_{\rm comb}(\eta,g)\mathsf D_{N,\eta,g}(S,T).
\tag{KUWP.15}
$$

For every fixed $\eta,g>0$, $C_{\rm comb}=1+O(\theta)$ in the
bounded-reward weak-selection scaling. In particular the global term
must have a positive coefficient; its coefficient cannot be set to zero
while retaining the displayed proof and its $\eta/g$ comparison.

### 6.1. A specified-vertex forest connection bound

In the conditioned common forest, for distinct specified labels $b,j$,

$$
\Pr(j\in\mathcal C(b))\le J/N.
\tag{KUWP.16}
$$

Indeed their unique undirected connecting path, if present, has a
fitness-increasing arm followed by a fitness-decreasing arm. For length
$\ell\ge1$ with its peak at one of the endpoints, the two orientation
possibilities have at most $N^{\ell-1}/(\ell-1)!$ intermediate choices
each. Its independent outgoing-row edge probability is at most
$(C/N)^\ell$. Summation contributes $2Ce^C/N$.
For an internal peak and arm lengths $k,\ell-k\ge1$, choose the peak
and the separately ordered internal arm vertices. There are at most
$N^{\ell-1}/[(k-1)!(\ell-k-1)!]$ choices. Summation over the two arm
lengths contributes $C^2e^{2C}/N$.
These contributions are bounded by $J/N$ as defined above. Removing
outgoing rows only reduces the permitted paths. Thus for any fixed
nonnegative label weights $W_j$,

$$
\mathbb E\sum_{j\in\mathcal C(b)}W_j
\le W_b+J\frac1N\sum_jW_j.
\tag{KUWP.17}
$$

### 6.2. The energy of exceptional endpoints

Let $\mathcal B$ be the joint bad physical/measurement types after
coupling the sampled companions, with $m=|\mathcal B|/N$.
The original mismatch set is included in $\mathcal B$, while each matched
recipient has bad-measurement probability at most $4r/\kappa_D$.
Consequently

$$
\mathbb Em\le K_D^fr,\qquad
\mathbb E\frac1N\sum_{j\in\mathcal B}W_j(S)
\le\mathsf L_A(S)+4r\mathsf M_2(S)/\kappa_D.
\tag{KUWP.18}
$$

The original per-row token estimate, before averaging rows, also gives
$\mathbb E q_i\le2a_*\mathbf1_{\{i\in A\}}+L_fr$. Thus the
exceptional recipient seeds have expected own weighted mass at most
$2a_*\mathsf L_A(S)+L_fr\mathsf M_2(S)$.

For a good recipient and a good donor label, subtracting the normalized
proposal numerators and gates gives the specified-label bound

$$
|b_{ij}(S)-b_{ij}(T)|
\le\frac{D_rr+D_mm}{N-1}.
\tag{KUWP.19}
$$

For the proposal term, matched physical recipient/donor weights agree
and only the $|A|$ other physical weights can change its denominator.
The denominator difference yields
$2r/[(N-1)\kappa_C^2]$, multiplied by $a_*$.
The gate difference is at most
$2L_g(C_r^fr+C_s^fm)$, multiplied by proposal mass at most
$1/[(N-1)\kappa_C]$. These are exactly $D_r,D_m$.

In a maximal token coupling, the probability that one own donor token
is $j$ and the paired tokens disagree is
$(b_{ij}-b_{ij}')_+$. Bound it by (KUWP.19) on good-good types and by
$c_* /(N-1)$ otherwise. Bad recipients contribute at most
$2c_*m\mathsf M_2(S)$ to the expected normalized donor-seed energy.
Bad donors contribute at most
$2c_*N^{-1}\sum_{j\in\mathcal B}W_j(S)$. Good-good pairs contribute
at most $2(D_rr+D_mm)\mathsf M_2(S)$.
Use (KUWP.18). Each of the two possible donor endpoints therefore has
expected own weighted mass at most

$$
2c_*\mathsf L_A(S)+D_Gr\mathsf M_2(S).
$$

This bound also weights the other swarm's donor label by $W_j(S)$:
the maximal residual and the bad-type bounds were symmetric, and the
label weights here are fixed deterministic nonnegative numbers.
Including the initial seeds, the seed list has expected weighted mass
at most

$$
\mathbb E\frac1N\sum_{b\in\mathrm{seeds}}W_b(S)
\le(1+S_L)\mathsf L_A(S)+S_Gr\mathsf M_2(S).
\tag{KUWP.20}
$$

Let $Z$ be the union of common components meeting that seed list.
By (KUWP.17), (KUWP.20), and $\mathbb EK/N\le K_fr$,

$$
\mathbb E\frac1N\sum_{i\in Z}W_i(S)
\le(1+S_L)\mathsf L_A(S)+(S_G+JK_f)r\mathsf M_2(S).
\tag{KUWP.21}
$$

### 6.3. Copied energy on the affected rows

Condition on the seed list as in Section 2. For a common row, remove
its outgoing edge before forcing $i\to j$. That accepted row belongs
to $Z$ only if a seed meets the removed-forest component of $i$ or $j$.
By (KUWP.16) its conditional affected probability is at most

$$
\sum_{b\in\mathrm{seeds}}
 [\mathbf1_{\{b=i\}}+\mathbf1_{\{b=j\}}+2J/N].
$$

Multiply by the common-edge probability at most $C/N$ and by donor
weight $W_j(S)$, then sum $i,j$ and divide by $N$.
The terms $b=i$ give $C(K/N)\mathsf M_2(S)$;
the terms $b=j$ give $CN^{-1}\sum_{b\in\mathrm{seeds}}W_b(S)$;
the other terms give $2CJ(K/N)\mathsf M_2(S)$.
Average the seed estimates. The common accepted copies in $Z$ have
expected weighted mass at most

$$
C(1+S_L)\mathsf L_A(S)
 +C[S_G+(1+2J)K_f]r\mathsf M_2(S).
$$

An exceptional recipient is itself a seed. Its own copied energy is
bounded directly by the single-own-donor estimate of Section 6.2.
Together these prove

$$
\mathbb E\frac1N\sum_{i\in Z,\ i\to j}W_j(S)
\le P_L\mathsf L_A(S)+P_Gr\mathsf M_2(S).
\tag{KUWP.22}
$$

Setting every $W_j=1$ in this copying calculation gives an activated
jitter count in $Z$ at most $P_Jr$ in expectation.
The set $Z$ is closed under the accepted edges of either swarm:
common edges stay within common components, and both endpoints of
each exceptional edge belong to the seed list. It is therefore a
union of whole actual collision components. The pointwise velocity
energy contraction applies to the original-slot frozen velocities on
$Z$ itself. Its prepared phase energy is at most its entering phase
energy plus the charged donor source phase energy and the conditional
activated-jitter contribution. Equations
(KUWP.21)--(KUWP.22) thus give

$$
\mathbb E\frac1N\sum_{i\in Z}W_i(S^{\rm prep})
\le A_L\mathsf L_A(S)+A_Gr\mathsf M_2(S)
      +d\sigma_J^2P_Jr.
\tag{KUWP.23}
$$

Actual mismatches are a subset of $Z$. Apply this bound to both own
swarm energies and use the count bound $C_Hr$. Apply the global
mixed-moment bound (KUWP.4) to the remaining $g$ term. The total is
bounded by

$$
[C_H+2d\sigma_J^2(\eta P_J+gC_J)]r
 +\eta A_L[\mathsf L_A(S)+\mathsf L_A(T)]
 +(\eta A_G+gC_E)r[\mathsf M_2(S)+\mathsf M_2(T)].
$$

Comparison term by term with (KUWP.13) proves (KUWP.15).
No asserted independence between the affected set and source energies
enters this derivation.

This completes the preparation lemma for the refined cost. Its
nonresonant kinetic weighted-Harris contraction, exact bridge insertion
through the second preparation, and any dense-viscous extension remain
separate obligations.

## 7. A primitive linear envelope for the combined factor

Suppose the positive-exponent register supplies explicit finite constants
$a_0,G_0,C_{r,0},C_{s,0}$ such that, for $0<\theta\le1$,

$$
a_*\le\theta a_0,\qquad L_g\le G_0,\qquad
C_r^f\le\theta C_{r,0},\qquad C_s^f\le\theta C_{s,0}.
$$

The logistic-power derivative register and the fixed bounded raw ranges
give these constants directly. Put $c_0=a_0/\kappa_C$ and define

$$
\begin{aligned}
L_{f,0}&=4c_0+(4c_0+2a_0)K_D^f
                  +2G_0(C_{r,0}+K_D^fC_{s,0}),\\
D_{r,0}&=2a_0/\kappa_C^2+2G_0C_{r,0}/\kappa_C,\\
D_{m,0}&=2G_0C_{s,0}/\kappa_C,\\
D_{G,0}&=2c_0K_D^f+8c_0/\kappa_D+2D_{r,0}+2D_{m,0}K_D^f,\\
S_{L,0}&=2a_0+4c_0,\qquad S_{G,0}=L_{f,0}+2D_{G,0},\\
K_{f,0}&=1+3L_{f,0},\qquad J_{\rm conn,0}=30c_0,\\
P_{L,0}&=4c_0(1+S_{L,0})+2c_0,\\
P_{G,0}&=4c_0[S_{G,0}+(1+2J_{\rm conn,0})K_{f,0}]+D_{G,0},\\
A_{G,0}&=S_{G,0}+J_{\rm conn,0}K_{f,0}+P_{G,0},\\
H_0&=24c_0+9L_{f,0},\qquad
J_0=3c_0(20+42L_{f,0}),\\
K_{\rm comb}&=\max\{H_0+2d\sigma_J^2[\eta(P_{L,0}+P_{G,0})+gJ_0],\
                    S_{L,0}+P_{L,0},\
                    H_0+J_0+(\eta/g)A_{G,0}\}.
\end{aligned}
\tag{KUWP.24}
$$

For $0<\theta\le\min\{1,\kappa_C/(8a_0)\}$, the preparation
condition $a_*\le1/4$ holds and

$$
C_{\rm comb}(\eta,g)\le1+\theta K_{\rm comb}.
\tag{KUWP.25}
$$

Indeed $c_*\le1/8$, so $M\le e<3$ and
$M-1\le8c_*e^{8c_*}\le24\theta c_0$.
Also $L_f\le\theta L_{f,0}$,
$C_H-1\le\theta H_0$, $C_J\le\theta J_0$, and
$J=(8c_*+16c_*^2)e^{8c_*}\le30\theta c_0$.
The remaining envelopes in (KUWP.24) follow by substitution and
$\theta\le1$. Applying them to the three alternatives in (KUWP.14)
proves (KUWP.25).

For the exact nonviscous harmonic kinetic coefficient $q_D<1$ proved
in `18_harmonic_weighted_block.md`, an explicit sufficient active interval is

$$
0<\theta\le\min\left\{1,\frac{\kappa_C}{8a_0},
                 \frac{1-q_D}{2q_DK_{\rm comb}}\right\}.
\tag{KUWP.26}
$$

The actual one-update composed pair coefficient is then at most
$(1+q_D)/2<1$. This statement uses the proved kinetic coefficient
only at $\nu=0$.

For invariant-moment existence one can add the explicit bound
$\theta\le(1-\lambda)/(2\lambda c_0)$, with the kinetic
$\lambda=(1+a_x^2)/2$ and $B_2$ of that note. The own incoming-column
bound and (KUWP.1) give

$$
\mathbb E\mathsf M_2(S^{\rm prep})
\le(1+c_*)\mathsf M_2(S)+d\sigma_J^2a_*.
$$

The actual complete conservative nonviscous kernel therefore satisfies

$$
P_N\mathsf M_2(S)
\le\frac{1+\lambda}{2}\mathsf M_2(S)
        +B_2+\lambda d\sigma_J^2a_0.
\tag{KUWP.27}
$$

This supplies a population-independent moment bound. For fixed $N$,
continuous bounded reward makes the finite-pattern preparation and the
kinetic kernel Feller; the displayed drift gives tight empirical time
averages on the finite-dimensional phase space with capped velocities.
Their invariant subsequential limits have finite averaged second moment.
This existence argument uses confinement, rather than a position cutoff.
