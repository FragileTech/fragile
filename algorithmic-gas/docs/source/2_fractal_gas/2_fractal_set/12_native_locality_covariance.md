# Native spatial locality budgets and verified record covariance

(sec-native-lc-ledger)=
## 1. Complete execution data and calibrated regions

:::{prf:definition} Complete data consumed by the spatial estimates
:label: def-native-lc-ledger

Every statement is indexed by the complete execution record of
{prf:ref}`def-native-complete-execution-record`: every recursive configuration
field, landscape/provider, initial law, arithmetic/randomness convention,
error/termination policy, stage/alignment, mask, weight, geometry recipe and
physical calibration remains explicit. Positive Gaussian estimates use the
existing real-coordinate restriction of {prf:ref}`def-cgd-parameter-register`
and {prf:ref}`def-native-jg-ledger`: current donors, revival before BAOAB,
terminal classification in bounded $D$, original final cap, and no consumed
geometry feedback, curl, history, elite or innovation shift. Both viscosity
normalizations are retained. The donor theorem names its existing independent
one-current-donor Gaussian arm; other modules retain their actual different law.

Write $a=h/2$, $c=e^{-\gamma h}$, $s^2=\sigma_x^2h$,
$R_c=(1+2|\alpha_{\rm col}|)V$ and

$$
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0.\end{cases}
$$

After the original collision and first force evaluation, retain

$$
v_{1i}=v_i^J+a[F(x_i^J)+F_i^{\rm visc}(x^J,v^J)],
\qquad m_i=x_i^J+a(1+c)v_{1i}.
$$

The original B2 input is $X_i=m_i+aq\xi_i$, $z_i=cv_{1i}+q\xi_i$; terminal
position is $X_i^+=X_i+s\zeta_i$. The noises are the original independent
standard Gaussians. All rows are eligible at both kicks, following revival.
The complete interacting preparation, rather than an independent replacement,
determines the means.

The declared length/time units give $(\tau,\mathbf x)=(t_*nh,\ell_*x)$ and
the configured $c_{\rm phys}$ specifies its comparison cone. The older
four-position-coordinate projection (YM.Z81) is a distinct retained readout,
not a substitution for this clock. Position tails are not identified with
quantum commutators. Regional CAR modes keep their original full-law covariance.

Let $B_D=\sup_{y\in D}|y|$. The evaluated force profile $\mathcal F_p$ of
{prf:ref}`lem-native-jg-spatial-gaussian` gives the primitive moment bound

$$
\left(\mathbb E\frac1N\sum_i|v_{1i}|^p\right)^{1/p}
\le V_{1,p}:=R_c+a(\mathcal F_p+2\nu R_c).
$$

A divergent profile gives an infinite certificate. Unmentioned fields remain
in the complete record; an upper bound independent of them does not erase them.
Fixed-seed finite-arithmetic execution has its separate law and comparison.
:::

(sec-native-lc-b2-spatial-decay)=
## 2. Exact second-kick spatial decay with unbounded innovations

:::{prf:theorem} Integrated native B2 pair and remote count-force bound
:label: thm-native-lc-b2-integrated-force

Condition on the actual preparation and B1 output. For $i\ne j$ put

$$
\mu_{ij}=m_j-m_i,\quad d_{ij}=v_{1j}-v_{1i},\quad
R^2=\rho^2+2a^2q^2,\quad Z_0=(\rho^2/R^2)^{d/2},\quad\beta=2aq^2/R^2.
$$

Exactly,

$$
\mathbb E\!\left[e^{-|X_j-X_i|^2/(2\rho^2)}(z_j-z_i)\mid\mathcal F\right]
=Z_0e^{-|\mu_{ij}|^2/(2R^2)}(c d_{ij}-\beta\mu_{ij}).
\tag{LC.1}
$$

The conditional mean of the actual count-normalized B2 force is its original
$\nu/N$ sum of these terms. Its part $\overline F_{i,r}$ from prepared-center
pairs with $|\mu_{ij}|\ge r$ satisfies, for every $r>0$,

$$
\mathbb E\frac1N\sum_i|\overline F_{i,r}|
\le\nu Z_0e^{-r^2/(4R^2)}
 \left[2cV_{1,1}+\beta\sqrt{2/e}\,R\right].
\tag{LC.2}
$$

This population-independent actual stage bound retains the noise-softened
width $R$. For existing records with $h\to0$, $\rho\to0$, bounded
$\nu,V,\alpha_{\rm col},b_O,\gamma\ge0$, and finite uniformly bounded
$\mathcal F_1$, its right side tends to zero at every fixed $r>0$. These
are explicit parameter regimes of the derived bound, not assumed law properties.
Fixed $h,\rho$ retain its finite budget.

For both normalizations, using the actual B2 positions instead, set
$D_i=N$ for count and $D_i=\sum_{j\ne i}K_{ij}$ for row. Pathwise,

$$
\left|\nu\frac{\sum_{j:\,|X_j-X_i|\ge r}K_{ij}(z_j-z_i)}{D_i}\right|
\le\frac{\nu e^{-r^2/(2\rho^2)}}{D_i}
 \sum_{j:\,|X_j-X_i|\ge r}|z_j-z_i|.
\tag{LC.3}
$$

The force is zero for $N=1$. For $N>1$ the real Gaussian row degree is
positive. Its random denominator remains inside (LC.3); (LC.1) divided
by an expected degree is not a row-normalized identity.
:::

:::{prf:proof}
Conditional on the preparation, $Y=X_j-X_i=\mu_{ij}+aq(\xi_j-\xi_i)$,
with $\xi_j-\xi_i\sim N(0,2I_d)$. Completing the Gaussian square gives

$$
\mathbb EK(Y)=Z_0e^{-|\mu_{ij}|^2/(2R^2)},\qquad
\mathbb E[K(Y)(Y-\mu_{ij})]
=-\frac{2a^2q^2}{R^2}\mu_{ij}\mathbb EK(Y).
$$

Insert $z_j-z_i=cd_{ij}+(Y-\mu_{ij})/a$ to prove (LC.1); if $q=0$
the statement follows directly. For $u\ge r$, split the Gaussian factor
into two halves and maximize $u e^{-u^2/(4R^2)}$ to obtain

$$
e^{-u^2/(2R^2)}\le e^{-r^2/(4R^2)},\qquad
u e^{-u^2/(2R^2)}\le\sqrt{2/e}\,R\,e^{-r^2/(4R^2)}.
$$

The normalized double sum of $|d_{ij}|$ is at most
$2N^{-1}\sum_i|v_{1i}|$. Apply the primitive moment bound to prove (LC.2).
In its shrinking regime $q^2\le b_O^2h$, $R\to0$, $Z_0\le1$,
$V_{1,1}$ is bounded, and $\beta R\le\sqrt2q\to0$. Hence the displayed
budget vanishes. Finally $K_{ij}\le e^{-r^2/(2\rho^2)}$ on the actual
remote-pair set; the triangle inequality gives (LC.3).
:::

(sec-native-lc-copying)=
## 3. The original donor and accepted-copy locality budget

:::{prf:lemma} Physical localization of squashed donor sampling
:label: lem-native-lc-donor-localization

Use the configured $\psi_R(x)=x/(1+|x|/R)$ in the independent Gaussian
one-donor cloning module. Freeze the current eligible pool and already
sampled fitness before the independent cloning companion/acceptance draws.
For a living query $i$, all eligible donor positions and $x_i$ lie in $D$.
Put $\kappa_x=(1+B_D/R_x^{\rm feat})^{-2}$ and

$$
\Delta_{ij}^2=
|\psi_{R_x^{\rm feat}}(x_j)-\psi_{R_x^{\rm feat}}(x_i)|^2
+\lambda_{\rm alg}
|\psi_{R_v^{\rm feat}}(v_j)-\psi_{R_v^{\rm feat}}(v_i)|^2.
$$

For the actual allowed index set $\mathcal D_i$, including self/singleton
conventions, its one-draw law is

$$
\pi_{ij}=\frac{e^{-\Delta_{ij}^2/(2\epsilon_C^2)}}
 {\sum_{k\in\mathcal D_i}e^{-\Delta_{ik}^2/(2\epsilon_C^2)}}.
$$

Let $M_i(r)$ count allowed donors with $|x_j-x_i|\ge r$, and
$L_i(\delta)$ those with $\Delta_{ij}\le\delta$. If $L_i(\delta)>0$,

$$
\sum_{j:\,|x_j-x_i|\ge r}\pi_{ij}
\le\min\left\{1,\frac{M_i(r)}{L_i(\delta)}
\exp\!\left[\frac{\delta^2-\kappa_x^2r^2}{2\epsilon_C^2}\right]\right\}.
\tag{LC.4}
$$

Retain the actual acceptance, including its configured step period,

$$
p_{ij}=\mathbf1_{\{n\equiv0\pmod{\mathtt{every}}\}}
\min\left\{1,\left[\frac{V_j-V_i}{s_c(V_i+\epsilon_c)}\right]_+\right\}.
$$

The conditional probability of an accepted remote copy is exactly
$\sum_{\rm remote}\pi_{ij}p_{ij}$ and bounded by (LC.4). If no close
donor exists, only the bound one is supplied. The conditional expected
copy displacement is

$$
C_i=\sum_{j\in\mathcal D_i}\pi_{ij}p_{ij}|x_j-x_i|
\le r+2B_D\sum_{\rm remote}\pi_{ij}p_{ij}.
\tag{LC.5}
$$

For the default uniform current-donor revival, a dead query instead has
remote probability exactly $M_i(r)/M$ and mean jump
$M^{-1}\sum_{j\ {\rm eligible}}|x_j-x_i|$, where $M$ is the current live
count. Its retained dead position need not lie in $D$. Configured weighted
revival uses its original companion law. Living-row Gaussian localization
is not substituted for default uniform revival.
:::

:::{prf:proof}
The tangential/radial eigenvalues of $D\psi_R(x)$ are
$(1+|x|/R)^{-1}$ and $(1+|x|/R)^{-2}$, with derivative identity at zero.
The segment between any two points in $D$ lies in the ball of radius $B_D$.
Integrating its quadratic form and applying Cauchy--Schwarz gives
$|\psi_R(x)-\psi_R(y)|\ge(1+B_D/R)^{-2}|x-y|$.
The velocity contribution is nonnegative. Every remote donor weight is
at most $e^{-\kappa_x^2r^2/(2\epsilon_C^2)}$, whereas the $L_i(\delta)$
close weights lower-bound the denominator. This proves (LC.4).
Conditional on the proposed donor the original gate has probability
$p_{ij}$, proving the exact accepted-copy and jump sums. Split the latter
at distance $r$ and use $|x_j-x_i|\le2B_D$ to prove (LC.5).
The configured default revival draw is uniform over current eligible
sources, giving its separate exact formulas.
:::

(sec-native-lc-clock-transport)=
## 4. Calibrated finite-time transport with copying and original noise

:::{prf:theorem} Native copy register and full-step transport tail
:label: thm-native-lc-clock-transport

Fix a slot $i$ and $T$ updates, stopped when the gas becomes all dead.
Retain the final position after stopping and set later increments to zero.
In update $n$, let $y_i^n$ be its position after literal copying but before
jitter, $L_i^n=y_i^n-x_i^n$ the actual copy/revival jump and
$\chi_i^n$ its actual recipient-jitter mark, including accepted cloning
and forced revival. Define

$$
A_T=\sum_{n<T}|L_i^n|,\quad
\Sigma_h^2=\sigma_J^2+a^2q^2+s^2,\quad
F^*(J)=\sup_{|x|\le B_D+J}|F(x)|,
$$

$$
B_h(J)=a(1+c)\{R_c+a[F^*(J)+2\nu R_c]\}.
$$

For $J,L,u>0$ and finite $F^*(J)$,

$$
\begin{aligned}
\mathbb P\left(\max_{k\le T}|x_i^k-x_i^0|>L+TB_h(J)+u\right)
\le{}&\mathbb P(A_T>L)
+T\,2^{d/2}e^{-J^2/(4\sigma_J^2)}
+2^{d/2}e^{-u^2/(4T\Sigma_h^2)}.
\end{aligned}
\tag{LC.6}
$$

The jitter term is zero when $\sigma_J=0$; the final term is zero when the
total noise variance is zero. Also

$$
\mathbb P(A_T>L)\le L^{-1}\sum_{n<T}\mathbb E C_i^n,
\tag{LC.7}
$$

where $C_i^n$ is the actual donor/gate jump sum of (LC.5), with its original
revival arm for a dead query. Both viscosity normalizations are covered.
In physical coordinates multiply all distances by $\ell_*$ and use horizon
$t_*Th$. If $\ell_*B_h(J)\le c_{\rm phys}t_*h$, (LC.6) bounds tails beyond
that configured kinetic cone, retaining copying and noise.

For the unchanged reference, $R_c=4$, $B_D=2\sqrt3$ and
$F^*(J)=2\sqrt3+J$. At $h=0.04$, $J=0.8$,

$$
B_h(J)<0.1621,\quad \Sigma_h^2<0.010416,\quad
2^{3/2}e^{-J^2/(4\sigma_J^2)}<3.19\times10^{-7}.
\tag{LC.8}
$$

These bounds do not declare a quantum spacelike commutator to be a particle tail.
:::

:::{prf:proof}
The unchanged ordered position update, unaffected by B2 or the velocity cap,
gives

$$
x_i^{n+1}-x_i^n=L_i^n+a(1+c)v_{1i}^n+\eta_i^n,\qquad
\eta_i^n=\chi_i^n\sigma_JZ_i^n+aq\xi_i^n+s\zeta_i^n.
$$

All terms vanish after stopping. The gate precedes clone jitter; conditional
on the complete preceding history and donor/gate choices, that jitter and
the later OU/final-position innovations retain independent Gaussian laws.
Thus $\eta_i^n$ are martingale differences and
$\mathbb E[e^{\theta\cdot\eta_i^n}\mid\mathcal F_n]
\le e^{\Sigma_h^2|\theta|^2/2}$.
Iteration bounds the generating function of $M_k=\sum_{n<k}\eta_i^n$
by $e^{k\Sigma_h^2|\theta|^2/2}$. For an independent standard Gaussian $G$,
the identity
$e^{|v|^2/(4T\Sigma_h^2)}
=\mathbb E_Ge^{G\cdot v/\sqrt{2T\Sigma_h^2}}$
and Tonelli give
$\mathbb E e^{|M_T|^2/(4T\Sigma_h^2)}\le
\mathbb E e^{|G|^2/4}=2^{d/2}$.
Convexity makes this exponential a nonnegative submartingale in $k$.
Stopping at its first threshold crossing gives the last term of (LC.6),
including the maximum over all $k$.

The same Gaussian bound and a union bound control every used jitter by
the second term. Outside that event, pre-jitter positions are current
eligible donors/inputs in $D$, so $|x_i^{J,n}|\le B_D+J$.
The actual component collision has bound $R_c$; B1 viscosity has bound
$2\nu R_c$ under either normalization. Consequently
$|a(1+c)v_{1i}^n|\le B_h(J)$. Summing the exact update and using $A_T\le L$
shows that any larger displacement requires a jitter or martingale-tail
event. This proves (LC.6). Conditional expectation of the original jump
is exactly its original donor/gate sum; Markov's inequality gives (LC.7).
Direct substitution yields (LC.8).
:::

(sec-native-lc-regional-covariance)=
## 5. The fresh regional spatial term and original ancestral covariance

:::{prf:lemma} Conditional localization of the original terminal position descriptor
:label: lem-native-lc-terminal-descriptor-locality

Let $q_O^{\rm pos}$ be the position-and-status projection of the actual
terminal regional record: retain each labeled row in $O$, its terminal
position, and its original mark $\mathbf1_D(X_i^+)$. Consider two actual
prepared arrays with the same original scalar variance
$\tau^2=a^2q^2+s^2>0$. Suppose their centers coincide outside a row set $S$.
Writing $r_i=\operatorname{dist}(m_i,O)$ and
$r_i'=\operatorname{dist}(m_i',O)$, their conditional descriptor laws obey

$$
\left\|\mathcal L(q_O^{\rm pos}\mid m)
-\mathcal L(q_O^{\rm pos}\mid m')\right\|_{\rm TV}
\le\min\left\{1,\,
2^{d/2}\sum_{i\in S}
\left[e^{-r_i^2/(4\tau^2)}+e^{-(r_i')^2/(4\tau^2)}\right]\right\}.
\tag{LC.8a}
$$

Distances and center equalities are evaluated from the actual preparations.
Thus for any existing family with every changed center at distance at least
$r>0$ in both preparations and
$|S|e^{-r^2/(4\tau^2)}\to0$, this original position-descriptor effect
vanishes. Under unit conversion use $(\ell_*r,\ell_*\tau)$.
The claim does not replace the full regional descriptor by its position
projection: original color, faces, history and future selection require
their own terms.
:::

:::{prf:proof}
Couple identical row noises in the two original conditional Gaussian
updates. Rows outside $S$ then have identical terminal positions and marks.
The two projected descriptors agree if all changed rows in both arrays
land outside $O$. A changed row hitting $O$ requires a centered Gaussian
of covariance $\tau^2I$ to have norm at least its mean's distance from $O$.
For $Z\sim N(0,I_d)$,
$\mathbb E e^{|Z|^2/4}=2^{d/2}$, so its hit probability is at most
$2^{d/2}e^{-r_i^2/(4\tau^2)}$. Union bound over both arrays and all changed
rows gives the stated coupling disagreement probability. The defining
coupling bound for total variation proves (LC.8a). This proof uses only
the original terminal Gaussian law and its actual status map.
:::

:::{prf:theorem} Exact native empirical spatial covariance
:label: thm-native-lc-terminal-covariance

At the terminal-position stage retain the original alive mask in
$f_D=\mathbf1_Df$, $g_D=\mathbf1_Dg$, for bounded spatial tests $f,g$.
For all-slot readouts omit that mask. Set
$F=N^{-1}\sum_i f_D(X_i^+)$, $G=N^{-1}\sum_i g_D(X_i^+)$ and

$$
u_i=\int f_D(y)\varphi_\tau(y-m_i)\,dy,\quad
w_i=\int g_D(y)\varphi_\tau(y-m_i)\,dy,\quad
\tau^2=a^2q^2+s^2,\quad M_f=N^{-1}\sum_i u_i,\quad M_g=N^{-1}\sum_i w_i.
$$

Here $\varphi_\tau$ is the normalized Gaussian density. Exactly,

$$
\begin{aligned}
\operatorname{Cov}(\overline F,G)
={}&\operatorname{Cov}(\overline{M_f},M_g)\\
&+\frac1{N^2}\sum_i\mathbb E\left[
\int\overline{f_D(y)}g_D(y)\varphi_\tau(y-m_i)\,dy-\overline{u_i}w_i
\right].
\end{aligned}
\tag{LC.9}
$$

For disjoint supports the integral is zero and

$$
\left|\operatorname{Cov}(\overline F,G)
-\operatorname{Cov}(\overline{M_f},M_g)\right|
\le \|f\|_\infty\|g\|_\infty/N.
\tag{LC.10}
$$

For arbitrary supports replace its right side by twice that bound.
Also $\mathbb E\operatorname{Var}(F\mid\mathcal F)\le\|f\|_\infty^2/N$.
Selection by an event of failure probability $0\le\delta<1$ changes the
unselected covariance by at most $6\|f\|_\infty\|g\|_\infty\delta$.
Thus the already derived native coverage/survival probabilities transfer
this estimate with their original $\delta$.

The original normalized CAR coefficient divides the full covariance in
(LC.9) by the actual standard deviations. The ancestral conditional means
remain in that coefficient. An unnormalized $N^{-1}$ error does not prove
that normalized coefficient vanishes; replacing $F$ by $F-M_f$ would change
the prescribed regional mode.
:::

:::{prf:proof}
Conditional on the complete interacting preparation, terminal positions
are independent nonidentical Gaussians by
{prf:ref}`lem-native-jg-spatial-gaussian`. Expanding the two original
empirical sums cancels every off-diagonal conditional covariance. The
law of total covariance gives (LC.9). Disjoint supports remove the
same-site product identically; bounding the remaining products proves
(LC.10). Bounding both terms gives its arbitrary-support variant.
Each single-site conditional variance is at most $\|f\|_\infty^2$,
and conditional independence proves the variance estimate.

The original law and its conditioning on an event of probability
$1-\delta$ have total-variation distance $\delta$. A bounded expectation
of norm $M$ changes by at most $2M\delta$. Applying this to $\overline FG$
gives $2\|f\|_\infty\|g\|_\infty\delta$; the product of the two means changes
by at most $4\|f\|_\infty\|g\|_\infty\delta$. This proves the factor six,
without moving selection into the independent conditional law.
:::

(sec-native-lc-verified-symmetries)=
## 6. Verified symmetries of the unchanged reference

:::{prf:theorem} Exact reference spatial and color covariance
:label: thm-native-lc-reference-covariance

For the unchanged reference of {prf:ref}`def-cgd-existing-reference`, let
$Q$ be a signed coordinate permutation in dimension three. Transform all
positions and velocities by $Q$, keep clocks, indices, stages and status,
and transform a recorded component matrix by $O\mapsto QOQ^{-1}$.
The actual killed transition satisfies

$$
Q_N(T_Qs,T_QB)=Q_N(s,B).
\tag{LC.11}
$$

Its finite history law with transformed initial law is the pushforward
of the original law, including survival conditioning. An invariant initial
law gives a same-law symmetry. The already proved unique reference QSD
is invariant under all these transformations.

For a coordinate permutation matrix $P$, the original aligned masked
colors and full complex contractions obey

$$
C_i(T_P\omega)=PC_i(\omega),\quad q_{ij}(T_P\omega)=q_{ij}(\omega),\quad
b_{ijk}(T_P\omega)=\det(P)b_{ijk}(\omega),\quad
\Pi_{ijk}(T_P\omega)=\Pi_{ijk}(\omega).
\tag{LC.12}
$$

Under central inversion, exactly

$$
C_i(T_{-I}\omega)=-\overline{C_i(\omega)},\quad
q_{ij}(T_{-I}\omega)=\overline{q_{ij}(\omega)},\quad
b_{ijk}(T_{-I}\omega)=-\overline{b_{ijk}(\omega)},\quad
\Pi_{ijk}(T_{-I}\omega)=\overline{\Pi_{ijk}(\omega)}.
\tag{LC.13}
$$

Original alignment, clone identity, coverage, force-threshold and alive masks
are retained and unchanged. Tests transform in space with unchanged $\tau$.
Passive weights/faces satisfy the corresponding identities when their original
recipe is equivariant, including its tie/error policy. For an invariant
finite history law, a pure determinant average with weights/localization
invariant under an odd permutation therefore has exactly zero mean.

For a same-law specialization, $U_Qf=f\circ T_Q^{-1}$ is a unitary on its
centered full-record space, transports position-regional descriptors to $QO$,
and intertwines the unchanged update channel. Its established exterior lift
gives exact covariance of those regional CAR algebras. This verifies an actual
finite spatial symmetry group; no Poincare action or physical spectrum is
declared by this calculation.
:::

:::{prf:proof}
The reference potential/reward depend on $|x|^2$ and the terminal cube is
invariant under signed permutations. Radial feature squashing, Euclidean
differences, Gaussian donor kernels, diversity and every scalar standardizer
are invariant. Hence the frozen fitness arrays, categorical companion and
gate probabilities are unchanged: couple indices and gates identically.
Literal simultaneous copying commutes with $Q$.

Isotropic clone, OU and final-position Gaussian innovations have unchanged
laws under $Q$. The Haar orthogonal component matrix has unchanged law
under conjugation, since its image is again normalized left/right invariant
measure on the same group. Collision barycenters and relative velocities
transform by $Q$. Both dense viscosity forces transform by $Q$ because all
scalar kernel distances and denominators are invariant. Both potential
evaluations, A stages and the radial cap commute with $Q$. The terminal
cube preserves all statuses and extinction. These facts prove (LC.11)
for the complete stage order and yield the finite history statement by
induction. Conditioning on the unchanged survival event preserves it.
A transformed QSD is another QSD with the same eigenvalue, so the
established uniqueness of {prf:ref}`thm-cgd-finite-n-qsd` proves invariance.

For a permutation, each force component and declared phase velocity component
is permuted together. The scalar $\kappa=m\ell_0/\hbar_{\rm eff}$ and force
norm are unchanged; the original component formula therefore gives $C'=PC$.
Its inner products, determinants and overlap triangles give (LC.12).
For inversion real force and phase velocity both change sign, so
$C'=-\overline C$, proving (LC.13) with the three-column determinant sign.
All specified masks depend on unchanged index/status data or norms.
An invariant weighted determinant average changes sign under an odd
permutation; same-law invariance forces its expectation to equal its negative.

Finally a same-law pushforward preserves inner products and means. It
transports the actual position-descriptor sigma algebras to those of $QO$.
Equation (LC.11) intertwines the native Markov contraction. The existing
CAR exterior lift and channel formula therefore transport these regional
algebras and channels. This does not add a continuous rotation of the cube,
a boost, or a different physical generator.
:::

(sec-native-lc-remaining)=
## 7. Remaining physical identification

:::{prf:proposition} Discharged spatial terms and retained physical residual
:label: prop-native-lc-remaining-register

The new calculations discharge count-B2 conditional spatial-force decay,
physical donor localization, full-step finite-time position transport with
copying, the fresh terminal empirical covariance term, and the verified
reference spatial symmetries. They retain the original ordered update and
uncapped innovations.

For the fixed reference $(h,\rho,\epsilon_C,\sigma_J,\sigma_x)
=(0.04,1,2,0.1,0.1)$ these are finite quantitative budgets; their formulas
do not vanish solely because $N\to\infty$. The remote count-force budget
does vanish in the explicit $h\to0$, $\rho\to0$ regime of (LC.2), for that
force term only, with every parameter change declared.

Physical graded locality of the prescribed regional CAR net still needs
vanishing normalized complete cross-covariance for its actual spacelike
readouts. Equation (LC.9) separates its new $N^{-1}$ fresh spatial contribution
from the original ancestral/selection term. That term, the variance
normalization, the full force/color dependence and the physical transfer
identification must be controlled together. A spatial empirical pairing
is not substituted for a configured gauge channel. Poincare covariance
also needs symmetries and a physical generator beyond the verified finite
group. These missing identifications are not failures inferred from the
CAR criterion alone.
:::
