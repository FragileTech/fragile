# Native descriptor transfer, its derived decay, and the edge correspondence

(sec-npt-complete-record)=
## 1. Complete execution data and the native law

:::{prf:definition} Transfer execution record and explicit reference restriction
:label: def-npt-complete-record

Retain the complete record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, including all nested
configuration fields, landscape/provider functions and their parameters,
initial law, update schedules, random-stream law or fixed seed, arithmetic,
resource/error policies, masks, stage alignment, edge recipes, and physical
calibration. The present reference deductions use the actual real-coordinate
independent-Gaussian canonical instance of
{prf:ref}`def-cgd-parameter-register`. They retain simultaneous cloning,
mandatory revival, the configured donor and fitness rules, its component
collision, both viscous force evaluations, its final position diffusion and
radial velocity cap, and terminal absorption in $D$.

No donor history, elite restoration, consumed geometry feedback, curl, shifted
innovation, intermediate absorption, or extra Markov memory is enabled in
this restriction. The cloning period is one. A different schedule requires
its actual clock residue in the state; it does not inherit a time-homogeneous
kernel on an omitted residue. Fields used only for passive recording remain
in the observation record. Initial distributions other than the named QSD or
stationary law retain their actual finite-window laws.

The actual full killed kernel, its identified QSD and its eigenfunction are

$$
Q_N,\quad \nu_NQ_N=\alpha_N\nu_N,\quad Q_Ne_N=\alpha_Ne_N,
\quad \max e_N=1,\quad m_N=\min e_N>0.
$$

They are constructed in {prf:ref}`thm-cgd-finite-n-qsd`. Set

$$
P_N(s,ds')=\frac{e_N(s')}{\alpha_Ne_N(s)}Q_N(s,ds'),\qquad
\pi_N(ds)=\frac{e_N(s)}{\nu_N(e_N)}\nu_N(ds).
\tag{NPT.1}
$$

These are the conservative Doob transition and its stationary law, rather
than the terminal-survival-conditioned finite history law. The existing
primitive certificate gives

$$
P_N(s,\cdot)\ge\delta_N\mu_N(\cdot),\quad
\mu_N(ds')=\frac{e_N(s')\theta_0(ds')}{\theta_0(e_N)},\quad
\delta_N=\frac{\epsilon_N\theta_0(e_N)}{\alpha_N}>0.
\tag{NPT.2}
$$

For the tested force regime of
{prf:ref}`thm-cgd-primitive-eigenfunction`, all these conclusions hold with

$$
m_N\ge\underline m_F,\qquad
\delta_N\ge\underline\delta_F=\epsilon_F\underline m_F>0,
\tag{NPT.3}
$$

where $\epsilon_F,\underline m_F$ are the explicit primitive expressions in
that theorem. Thus a law-specific mixing or entropy inequality is not added
as an assumption. For the unchanged viscous reference,
$h=.04$, $\lambda=\gamma=b_O=1$, $\nu=.3$, $\rho=1$,
$\sigma_J=\sigma_x=.1$, $V=2$, $\alpha_{\rm col}=.5$,
$D=[-2,2]^3$, all its donor, fitness and reducer parameters remain their
configured values, and both count/row certificates apply. Their coercivity
margins are $.9936$ and $.9876$.

Write $a_{\rm phys}=t_*h>0$ for physical duration of one full sampling update
under the declared clock calibration. An energy reported below is its inverse
physical-time rate multiplied by the configured $\hbar_{\rm eff}>0$. Length
calibration, color phase $\kappa$, localization and edge coefficient
normalization remain in their consumed readouts. No change of calibration
changes $P_N$.
:::

(sec-npt-derived-l2-decay)=
## 2. A derived full-law Hilbert-space decay bound

:::{prf:theorem} Stationary common-part minorization implies a centered Hilbert contraction
:label: thm-npt-minorization-l2

Let $P$ be a Markov kernel with its actual stationary probability $\pi$.
Suppose its already established common part is $P(s,\cdot)\ge\delta\mu$,
where $0<\delta<1$ and $\mu$ is a probability measure. Then, on the complete
centered record space $\mathcal H=L^2_0(\pi;\mathbb C)$,

$$
\|P|_{\mathcal H}\|\le r:=\sqrt{1-\delta}<1.
\tag{NPT.4}
$$

More precisely, for centered $f$,

$$
\|Pf\|_\pi^2
\le(1-\delta)\|f\|_\pi^2
-\delta(1-\delta)\mu(|f|^2)-\delta^2|\mu(f)|^2.
\tag{NPT.5}
$$

Consequently the actual coupled Doob kernel of (NPT.1) satisfies, without
reversibility or independent-walker replacement,

$$
\|C_N^n\|\le r_N^n,
\quad C_N=P_N|_{L^2_0(\pi_N)},\quad
r_N=\sqrt{1-\delta_N}
\le\sqrt{1-\underline\delta_F}.
\tag{NPT.6}
$$

Every actual centered bounded descriptor $f,g$ therefore obeys

$$
\left|\mathbb E_{\pi_N}
 \overline{f(S_0)}g(S_n)\right|
\le r_N^n\|f\|_{\pi_N}\|g\|_{\pi_N}.
\tag{NPT.7}
$$

For $\delta=1$, $P$ is the constant kernel $\mu=\pi$ and $C=0$.
:::

:::{prf:proof}
For $0<\delta<1$ define the actual residual Markov kernel
$R=(P-\delta\mu)/(1-\delta)$. Stationarity gives the measure identity

$$
\pi=\delta\mu+(1-\delta)\pi R.
$$

In particular $\delta\mu\le\pi$, so $\mu(f)$ and $\mu(|f|^2)$ are finite
for every $f\in L^2(\pi)$. Since $\pi(f)=0$,

$$
\pi(Rf)=-\frac\delta{1-\delta}\mu(f).
$$

Expand the square of $Pf=\delta\mu(f)+(1-\delta)Rf$, retain its cross
term, and then apply Jensen only to $R$:

$$
\begin{aligned}
\|Pf\|_\pi^2
&=(1-\delta)^2\pi(|Rf|^2)-\delta^2|\mu(f)|^2\\
&\le(1-\delta)^2\pi R(|f|^2)-\delta^2|\mu(f)|^2\\
&=(1-\delta)\{\pi(|f|^2)-\delta\mu(|f|^2)\}
 -\delta^2|\mu(f)|^2.
\end{aligned}
$$

This proves (NPT.5) and (NPT.4), including for complex $f$. The centering
subspace is invariant because $\pi P=\pi$. Iteration proves (NPT.6).
The actual covariance identity is the native Markov identity on this full
state space, so Cauchy--Schwarz proves (NPT.7). When $\delta=1$, positivity
and total mass one force $P(s,\cdot)=\mu$ and stationarity forces $\pi=\mu$.
:::

:::{prf:corollary} Actual stationary replica sectors and their finite-population decay
:label: cor-npt-native-fock-decay

For the native CAR reconstruction of
{prf:ref}`thm-lqft-record-fock-reconstruction`, the existing transfer is
$\Gamma_-(C_N)$. Its vacuum is the unique fixed vector. On its $k$th exterior
sector and on its nonvacuum even sector,

$$
\|\Lambda^k(C_N^n)\|\le r_N^{kn},\qquad
\|\Gamma_-(C_N)^n|_{\mathcal F_{\rm ev}\ominus\mathbb C\Omega}\|
\le r_N^{2n}=(1-\delta_N)^n.
\tag{NPT.8}
$$

The derived discrete decay rates are

$$
g_{1,N}=-\frac{\log(1-\delta_N)}{2a_{\rm phys}}>0,
\qquad g_{{\rm ev},N}=2g_{1,N}.
\tag{NPT.9}
$$

These are actual correlation/transfer norm rates for the stationary Doob
record. They are not identified with eigenvalue gaps of a self-adjoint
physical Hamiltonian until the native reflection-positive correspondence is
proved. In particular no Hermitian edge operator is substituted for $C_N$.
The primitive version replaces $\delta_N$ by $\underline\delta_F$ and gives
an explicit finite-population certificate. Its displayed $N$ dependence
supplies no uniform-in-population lower bound by itself.
:::

:::{prf:proof}
Restrict the $k$fold tensor operator to antisymmetric tensors. Its norm is at
most $\|C_N^n\|^k$, giving the first estimate. Every nonzero even sector has
$k\ge2$ and $0\le r_N<1$, so its norm is bounded by $r_N^{2n}$, uniformly in
$k$. The direct sum has that same bound. For arbitrary nonvacuum sectors use
$r_N^n$. Hence a fixed vector has vanishing projection to every nonvacuum
sector, while the vacuum is preserved. Taking inverse physical-time logarithms
gives the rates. None of these norm computations establishes self-adjointness.
:::

(sec-npt-transition-records)=
## 3. The correct transfer for attached B2 and geometry records

:::{prf:definition} Native transition-attached descriptor law
:label: def-npt-attached-record

Let $R'$ be the actual passive record produced by one complete update,
including any configured B1/B2 input, terminal geometry, color, masks and
within-step donor/collision marks. The original augmented killed kernel is

$$
\widehat Q(s,ds'\,dr').
$$

Its $s'$ marginal is the unchanged $Q(s,ds')$. The record $r$ from the
previous update does not enter the next update in the declared passive
restriction. Keep precisely the attached fields consumed by the readout;
do not add future information to a present descriptor.

The full augmented Markov state is $\widehat S=(S,R)$. Its actual Doob kernel
and its stationary law are

$$
\begin{aligned}
\widehat P((s,r),ds'\,dr')
 &=\frac{e(s')}{\alpha e(s)}\widehat Q(s,ds'\,dr'),\\
\widehat\pi(ds'\,dr')
 &=\int\pi(ds)\widehat P((s,r),ds'\,dr').
\end{aligned}
\tag{NPT.10}
$$

The last expression is independent of the incoming passive record. Define
the original selected attached-record law

$$
\widehat\nu(ds'\,dr')=
\alpha^{-1}\int\nu(ds)\widehat Q(s,ds'\,dr').
\tag{NPT.11}
$$

Its terminal $S'$ marginal is $\nu$. These are exact laws of the existing
update and its record; no independent preparation replacement is made.
:::

:::{prf:theorem} Two-step common part and descriptor decay for the actual attached records
:label: thm-npt-attached-two-step-decay

For the laws above, $\widehat\pi$ is stationary for $\widehat P$ and

$$
\widehat\pi(ds'\,dr')
=\frac{e(s')}{\nu(e)}\widehat\nu(ds'\,dr').
\tag{NPT.12}
$$

In particular $\widehat\pi\ge m\widehat\nu$. The actual augmented transfer
has a common part after two complete updates:

$$
\widehat P^2((s,r),\cdot)\ge\delta\widehat\mu(\cdot),\qquad
\widehat\mu=\int\mu(ds_1)\widehat P((s_1,r_1),\cdot).
\tag{NPT.13}
$$

Consequently, for $\widehat C=\widehat P|_{L^2_0(\widehat\pi)}$,

$$
\|\widehat C^n\|
\le(1-\delta)^{\lfloor n/2\rfloor/2}.
\tag{NPT.14}
$$

This bound applies to the actual terminal-alive B2 determinant, its polynomial
gauge invariants, terminal metric readouts and other bounded attached
descriptors. A k-sector native replica bound raises its right side to power
$k$. Its large-time inverse-time decay rate is

$$
\widehat g_1=-\frac{\log(1-\delta)}{4a_{\rm phys}}.
\tag{NPT.15}
$$

It does not incorrectly assert that a marginal minorization for $S'$ is
already a one-step joint minorization for $(S',R')$.
:::

:::{prf:proof}
The $S'$ marginal of $\widehat\pi$ is $\pi P=\pi$. The subsequent kernel
uses only $S'$, so one more augmented step leaves its full law unchanged.
For (NPT.12), substitute the definition of $\pi$ in (NPT.10). Its factor
$e(s)$ cancels the denominator of $\widehat P$, leaving

$$
\frac{e(s')}{\alpha\nu(e)}\int\nu(ds)\widehat Q(s,ds'\,dr').
$$

This is the stated law identity, and $e/\nu(e)\ge m$ because
$0<e\le1$. To prove (NPT.13), integrate the first attached transition over
$r_1$ to obtain $P(s,ds_1)$. The next transition ignores $r_1$, so

$$
\widehat P^2((s,r),A)=\int P(s,ds_1)\widehat P((s_1,r_1),A)
\ge\delta\int\mu(ds_1)\widehat P((s_1,r_1),A).
$$

Apply {prf:ref}`thm-npt-minorization-l2` to the stationary kernel
$\widehat P^2$, giving $\|\widehat C^2\|\le\sqrt{1-\delta}$.
Every Markov transition is an $L^2$ contraction by Jensen and stationarity.
Write $n=2\lfloor n/2\rfloor+(n\bmod2)$ and multiply these bounds. Tensor
restriction gives the exterior bounds as before. The limiting rate follows
by division by $na_{\rm phys}$.
:::

(sec-npt-nontrivial-native-gauge-sector)=
## 4. A nonconstant native gauge descriptor and its invariant sector

:::{prf:definition} Primitive zero-determinant cylinder of the unchanged reference
:label: def-npt-reference-zero-cylinder

Retain the reference ledger of {prf:ref}`def-uda-complete-parameters`, with
$N\ge4$, $d=3$, original threshold $\delta_c=10^{-12}>0$ and its actual
terminal-alive B2 determinant $b^{\rm alive}$. Its squared modulus
$F=|b^{\rm alive}|^2\in[0,1]$ is an existing polynomial invariant of the full
complex determinant coordinate, under the declared common $SU(3)$ frame.
No spacetime connection is introduced by this choice.

Use any already tested radii $L_0,r_v$ of
{prf:ref}`def-cgd-minorization-certificate`. Put $t=h/2$,
$c=e^{-\gamma h}$, $q,s>0$ as in the reference and

$$
A_v=1+t(\lambda+2\nu),\qquad
A_x=1+t(1+c)A_v.
$$

The two positive logistic maps have upper bounds
$M_r=A_r+\eta_r$, $M_D=A_D+\eta_D$. Define the explicit fitness variation
coefficients, using the configured global standardizer floors,

$$
C_R=\frac{M_DA_r}{4\sigma_r},\qquad
C_D=\frac{M_rA_D}{4\sigma_D},\qquad
v_-=\eta_r\eta_D>0,\qquad
D_c=s_c(v_-+\epsilon_c)>0.
$$

For the actual reference $M_r=M_D=2.1$, $C_R=C_D=10.5$,
$v_-=.01$, $\epsilon_c=10^{-6}$ and $s_c=1$. Choose the following analysis
radii; they only specify events in the existing unbounded Gaussian draws:

$$
\begin{aligned}
u={}&\min\left\{L_0,r_v,
 \frac{D_c}{8C_D\sqrt{1+\lambda_{\rm alg}}},
 \sqrt{\frac{D_c}{2C_R\lambda}},
 \frac{\delta_c}{8\nu cA_v},\frac{L_D}{4A_x}\right\}>0,\\
G={}&\min\left\{\frac{\delta_c}{8\nu q},
                 \frac{L_D}{4tq}\right\}>0,
\qquad Z=\frac{L_D}{4s}>0.
\end{aligned}
\tag{NPT.16}
$$

Here $\nu=.3>0$ in the denominator is the existing viscosity, whereas the
radius $u$ is the variable defined on the first line. This proof is only
claimed for that positive-viscosity, positive-threshold reference. The raw
input cylinder $A_u$ has every row alive and $|x_i|,|v_i|\le u$; its
$\theta_0$ mass is

$$
a_u=\left[\frac{v_3u^3}{(2L_0)^3}
                  \left(\frac{u}{r_v}\right)^3\right]^N>0.
$$

Let $G_3$ denote the original three-dimensional Gaussian ball probability.
An explicit probability lower bound is

$$
p_{0,N}=\epsilon_Na_u\,2^{-N}
                 G_3(G)^NG_3(Z)^N>0.
\tag{NPT.17}
$$
:::

:::{prf:lemma} The native selected gauge descriptor has positive mass at zero
:label: lem-npt-gauge-zero-mass

For both count and row viscosity at the actual reference,

$$
\widehat\nu_N(F=0)\ge p_{0,N}>0.
\tag{NPT.18}
$$

The event establishing this inequality also has every row terminal alive.
It uses the original acceptance probabilities, collision rule and both
kinetic kicks. No innovation or force is clipped to obtain the event.
:::

:::{prf:proof}
The full QSD eigenmeasure and the original common part give

$$
\nu_N(A_u)=\alpha_N^{-1}\nu_NQ_N(A_u)
\ge\epsilon_N\theta_0(A_u)=\epsilon_Na_u.
$$

On this all-alive input cylinder the squashing maps are nonexpansive, so
every configured squashed distance is at most
$2u\sqrt{1+\lambda_{\rm alg}}$. Adding the existing distance floor and
averaging the existing companions cannot increase the difference between
any two resulting diversity values past this bound. The reward range is
at most $\lambda u^2/2$. With global standardization, subtraction of the
common empirical mean cancels in differences and the divisor is at least
its original floor. A logistic map of amplitude $A$ has derivative at most
$A/4$. Applying the product rule in difference form gives

$$
\max_iV_i-\min_iV_i
\le C_R\lambda u^2/2+2C_Du\sqrt{1+\lambda_{\rm alg}}
\le D_c/2.
$$

Every original living acceptance probability is therefore at most $1/2$,
regardless of the actual companion choice. The existing independent gate
uniforms make the conditional probability of accepting no clones at least
$2^{-N}$. There is no mandatory revival on this all-alive cylinder. With
no accepted clone, the original collision components are singletons and
velocities are unchanged; no clone jitter is consumed by a recipient.

For either normalization, when all input velocities have norm at most $u$,
$|F_i^{\rm visc}|\le2\nu u$. Thus after B1,
$|v_{1i}|\le A_vu$. On the original event $|\xi_i|\le G$ for every row,

$$
|z_i|\le cA_vu+qG\le\frac{\delta_c}{4\nu},\qquad
|X_i^{\rm B2}|\le A_xu+tqG\le L_D/2.
$$

The actual B2 viscous force has norm at most $2\nu\max_i|z_i|\le\delta_c/2$.
Every B2 color is consequently zero by its existing threshold, so $F=0$.
The second force kick and the final velocity cap do not change the terminal
position. On $|\zeta_i|\le Z$, its norm is at most $3L_D/4$, hence every
row is terminal alive in the original box. The fresh independent original
Gaussian draws give probability $G_3(G)^NG_3(Z)^N$. Multiply the event
bounds. The one-step event already implies survival, and division by
$\alpha_N\le1$ can only increase its selected probability. This proves
(NPT.18).
:::

:::{prf:theorem} A nontrivial native gauge-sector transfer with derived decay
:label: thm-npt-native-gauge-sector

For the unchanged reference and its actual passive terminal-alive B2 record,
let $C_{\rm amp}>0$ be the explicit constant of
{prf:ref}`thm-uda-uniform-selected-amplitude`. Then

$$
\operatorname{Var}_{\widehat\nu_N}(F)
\ge p_{0,N}C_{\rm amp}^2>0,\qquad
\operatorname{Var}_{\widehat\pi_N}(F)
\ge m_Np_{0,N}C_{\rm amp}^2
\ge\underline m_Fp_{0,N}C_{\rm amp}^2>0.
\tag{NPT.19}
$$

In particular $f_N=F-\widehat\pi_N(F)$ is a nonzero actual centered
full-record mode. Let $\mathcal H_{G,N}$ be the smallest closed subspace of
$L^2_0(\widehat\pi_N)$ containing this mode and the configured bounded
polynomial gauge descriptors, and invariant under both $\widehat C_N$ and
$\widehat C_N^*$. Equivalently it is the closure of the span of finite words
in these two actual operators applied to those actual centered descriptors.
This is a nonzero reducing native descriptor sector. Its invariance is
under the actual transfer and its adjoint; no local Gauss-law projection or
physical gauge representation is asserted by this operator closure. It obeys

$$
\|\widehat C_N^n|_{\mathcal H_{G,N}}\|
\le(1-\delta_N)^{\lfloor n/2\rfloor/2}.
\tag{NPT.20}
$$

It has no nonzero fixed vectors. Its existing exterior replica algebra and
CAR channels are well defined and have the corresponding native transfer
decay. They preserve all actual descriptor covariances and transitions.
This identifies a genuine invariant record sector and a finite-population
decay estimate. It does not identify that sector with a local Yang--Mills
connection or establish reflection positivity of its actual time transfer.
:::

:::{prf:proof}
The existing determinant columns have norm at most one, so Hadamard's
inequality gives $0\le F\le1$. The proved amplitude bound is
$\widehat\nu_N(F)\ge C_{\rm amp}$. For independent complete copies
$F_1,F_2$ of this same selected observable,

$$
\operatorname{Var}(F)=\tfrac12\mathbb E(F_1-F_2)^2
\ge\widehat\nu_N(F=0)\,\widehat\nu_N(F^2)
\ge p_{0,N}C_{\rm amp}^2.
$$

The first inequality retains the two disjoint events where one copy is zero;
the last follows from Jensen. These independent copies are only the variance
identity, rather than an independent-walker approximation in the algorithm.
For any constant $z$, (NPT.12) implies
$\widehat\pi_N(|F-z|^2)\ge m_N\widehat\nu_N(|F-z|^2)$.
Take the infimum over $z$ to obtain the second variance bound.

The stated operator-word closure is invariant under each operator, because
appending that operator to a finite word gives another finite word and both
operators are bounded. Invariance under the adjoint makes the subspace
reducing. It contains the nonzero mode just proved. Restrict (NPT.14) to
this actual subspace to obtain (NPT.20), which also excludes fixed vectors.
The existing CAR and exterior constructions apply to every closed Hilbert
subspace with its inherited inner product. Because the subspace is reducing,
its transfer is exactly the restriction of the native transfer, rather than
a compression that discards leakage. Its definitions retain every consumed
parameter through the actual law and descriptor maps.
:::

(sec-npt-edge-correspondence)=
## 5. An exact discrepancy for the actual edge zero modes

:::{prf:theorem} Mixing native records cannot reproduce the edge's exact zero-energy excitation modes
:label: thm-npt-native-edge-zero-mismatch

Use the actual native $C_N$ of (NPT.6), a realized existing edge matrix
$h_{\rm e}$, its declared physical duration $a_{\rm e}>0$, and an arbitrary
isometric injection $j:E\to L^2_0(\pi_N)$ of its declared finite modes into
actual centered record modes. No assertion that a particular walker-label
assignment is such an injection is needed: the bound below holds for every
possible verified injection. Let

$$
B=e^{-a_{\rm e}|h_{\rm e}|},\qquad E_0=\ker h_{\rm e}.
$$

If $E_0\ne\{0\}$, then

$$
\|(C_Nj-jB)|_{E_0}\|\ge1-r_N>0,
\qquad
\|(C_N^nj-jB^n)|_{E_0}\|\ge1-r_N^n.
\tag{NPT.21}
$$

The same conclusion for transition-attached actual modes uses the correctly
augmented transfer. Its two-step and general discrepancies satisfy

$$
\|(\widehat C_N^2j-jB^2)|_{E_0}\|\ge1-\sqrt{1-\delta_N},\qquad
\|(\widehat C_N^nj-jB^n)|_{E_0}\|
\ge1-(1-\delta_N)^{\lfloor n/2\rfloor/2}.
\tag{NPT.22}
$$

For the implemented unfloored nonconstant `fitness_ratio` operator at
$m\ge3$ retained alive modes, $\dim E_0=m-2$ by
{prf:ref}`thm-native-gap-fitness-spectrum`; its exact positive
particle--hole transfer therefore cannot equal this native centered
record transfer on its complete excitation space. In the constant-fitness
case $h_{\rm e}=0$, the same discrepancy covers every edge mode.

The obstruction concerns the actual existing finite-edge transfer identity.
It does not rule out the already constructed positive edge dynamics, nor a
separate native physical reconstruction with a different proved correspondence.
Removing the zero modes is not part of this result and does not establish
that correspondence on the remaining active plane.
:::

:::{prf:proof}
For any unit vector $f\in E_0$, $B^nf=f$ and $\|jf\|=1$. The reverse
triangle inequality and the actual native contraction give

$$
\|(C_N^nj-jB^n)f\|
\ge\|jf\|-\|C_N^njf\|
\ge1-r_N^n.
$$

Set $n=1$ and take the operator norm. Repeat with the actual augmented
contraction (NPT.14) to prove (NPT.22). The exact fitness-ratio rank and
zero-mode multiplicity are already computed from the actual executable
recipe, so they apply without a new spectral assumption. Every inequality
holds for every actual isometric $j$, hence does not depend on an invented
mode assignment.
:::

:::{prf:corollary} Quantitative active-mode and parity-sector discrepancy
:label: cor-npt-active-mode-mismatch

For an edge eigenvector with $|h_{\rm e}|f=d f$, $\|f\|=1$, any actual
isometric $j$ obeys

$$
\|(C_N^nj-jB^n)f\|
\ge\bigl(e^{-na_{\rm e}d}-r_N^n\bigr)_+.
\tag{NPT.23}
$$

For the attached record replace $r_N^n$ by the right side of (NPT.14).
For a unit $k$-wedge of edge zero modes, its native-versus-edge discrepancy
is at least $1-r_N^{kn}$, or $1-(1-\delta_N)^{k\lfloor n/2\rfloor/2}$
for attached records. In particular at $m\ge4$, two actual edge zero modes
give an even-sector discrepancy, rather than a discrepancy confined to odd
CAR insertions. This is the exact excitation representation of the filled
Hamiltonian in {prf:ref}`lem-native-pt-particle-hole`.
:::

:::{prf:proof}
Replace $B^nf=f$ in the preceding proof by
$B^nf=e^{-na_{\rm e}d}f$; the norm is nonnegative, yielding the positive
part. For the wedge, the isometry preserves its unit norm and its edge
transfer fixes it. The native exterior contraction has norm at most the
$k$th power of the actual centered transfer norm. The same reverse triangle
inequality proves the claim. A wedge of two zero excitation modes has even
excitation parity and preserves the filled-ground parity under the existing
particle--hole map.
:::


(sec-npt-native-reference-cycle)=
## 6. The actual interacting reference time-reversal cycle

:::{prf:lemma} A target-local density bound retains every unbounded preparation
:label: lem-npt-reference-local-density

Use the actual count-normalized quadratic-force B2 stage, with any finite
prepared A1 position $p$ and B1 velocity $v_1$, and its original Gaussian
innovations $z=cv_1+q\xi$, $Y=p+az+s\zeta$, where $a=h/2$.
Let $m=Nd$, $E=1-a^2\lambda>0$ and

$$
\mathcal K_p(z)=z-a\lambda(p+az)-a\nu L_{p+az}z.
$$

This is the actual uncapped B2 output, with the original Gaussian count
Laplacian. For $W>0$ define the primitive target profiles

$$
\begin{aligned}
E_W&=\frac{W+a^2\lambda\nu\rho/\sqrt e}{1-2a\nu},\\
C_W&=1+2a^2\lambda/e+2aE_W/(\rho\sqrt e),\\
\beta_W&=1-a^2\lambda-2a\nu C_W.
\end{aligned}
\tag{NPT.26}
$$

If $2a\nu<1$, $E-a\nu>0$, and $\beta_W>0$, every $w$ with
$\max_i|w_i|\le W$ has exactly one original preimage under
$\mathcal K_p$, for every finite $p$. At every such preimage
$\sigma_{\min}(D\mathcal K_p)\ge\beta_W$ and its Jacobian determinant is
positive. Thus for the original radial cap
$u_i=Vw_i/(V+|w_i|)$, the conditional joint output density obeys

$$
p_{p,v_1}(Y=y,u)
\le (2\pi qs)^{-m}\beta_W^{-m}
 \prod_{i=1}^N\left(\frac{V}{V-|u_i|}\right)^{d+1}
\tag{NPT.27}
$$

whenever $|u_i|<V$ and its original inverse satisfies
$\max_i|w_i(u)|\le W$. The bound is uniform over all finite preparations,
so it remains true after mixing every original donor, gate, collision and
unbounded clone-jitter pattern. It does not condition away large jitters.
At the unchanged count reference one may take the compact target
$\max_i|u_i|\le1<V=2$, $W=2$. Direct substitution gives
$\beta_W>.9870$.
:::

:::{prf:proof}
Put $X=p+az$ and $B_X=I-a\nu L_X$. At a preimage of $w$ set
$z=a\lambda X+e_z$. Its exact equation is

$$
B_Xe_z=w+a^2\lambda\nu L_XX.
$$

Since $\max_{r\ge0}r e^{-r^2/(2\rho^2)}=\rho/\sqrt e$,
$\|L_XX\|_{\infty,N}\le\rho/\sqrt e$. The diagonal-minus-off-diagonal
row bound for $B_X$ is at least $1-2a\nu$. To verify its inverse bound,
choose a row attaining $\|e_z\|_{\infty,N}$, take the inner product with
its unit direction and use the triangle inequality on the other rows.
It gives $\|B_Xe_z\|_{\infty,N}\ge(1-2a\nu)\|e_z\|_{\infty,N}$.
Consequently $\|e_z\|_{\infty,N}\le E_W$.

For a pair difference $\Delta z_{ij}=a\lambda\Delta X_{ij}+\Delta e_{z,ij}$,
its derivative block is

$$
M_{ij}=K_{ij}\left[I-a\Delta z_{ij}\Delta X_{ij}^{\mathsf T}/\rho^2\right].
$$

The preceding bound and the maxima
$\max_r r^2e^{-r^2/(2\rho^2)}/\rho^2=2/e$ and
$\max_r r e^{-r^2/(2\rho^2)}/\rho^2=1/(\rho\sqrt e)$ give
$\|M_{ij}\|\le C_W$. Pair blocks satisfy $M_{ji}=M_{ij}$.
The block row and column sums of the pair Laplacian derivative are at most
$2C_W$, hence its Euclidean operator norm is at most $2C_W$.
Thus $D\mathcal K_p=EI-a\nu D(L_Xz)$ has smallest singular value at least
$\beta_W$ at every preimage of every tested target.

For uniqueness, continuously replace $\nu$ by $t\nu$, $0\le t\le1$.
The profiles above do not increase when $t$ decreases, so every preimage is
regular throughout this deformation. Uniform properness follows from the
count energy bound $0\le L_X\le I$:

$$
z\cdot\mathcal K_{p,t}(z)
\ge(E-a\nu)|z|^2-a\lambda|p|\,|z|.
$$

Every preimage of a fixed bounded target therefore remains in a common
compact ball. At $t=0$ its unique preimage is
$(w+a\lambda p)/E$. The inverse-function theorem continues it locally in
$t$. Its derivative cannot encounter a singularity, and properness prevents
escape, so continuation reaches $t=1$. Conversely every preimage at any $t$
can be continued backwards to $t=0$ by the same argument. Local uniqueness
of continuation forces it to be this one preimage. The Jacobian determinant
has the positive sign it has at $t=0$.

The original independent joint Gaussian density of $(z,\zeta)$ is bounded
by $(2\pi q)^{-m}$; the $q$ in this expression includes the $z$ scale,
whereas $\zeta$ is standard. The actual transformation
$(z,\zeta)\mapsto(\mathcal K_p(z),Y)$ has block derivative
$\left[\begin{smallmatrix}D\mathcal K_p&0\\aI&sI\end{smallmatrix}\right]$
and determinant $s^m\det D\mathcal K_p$. The original cap inverse is
$w_i=Vu_i/(V-|u_i|)$ with determinant
$(V/(V-|u_i|))^{d+1}$. Apply change of variables and the lower Jacobian
bound to prove (NPT.27). Its right side has no preparation term, so mixing
any original preparation laws preserves it. The reference substitution
uses $a=.02$, $q^2=(1-e^{-.08})/2$, $s=.02$, $\lambda=\rho=1$,
$\nu=.3$ and $W=2$; it gives $E_W<2.025$, $C_W<1.050$,
$\beta_W>.9870$.
:::

:::{prf:theorem} Exact nonzero consensus cycle of the unchanged interacting count reference
:label: thm-npt-reference-consensus-cycle

Use the actual count reference and its unmarked complete core killed kernel
$Q_N$ on position, velocity and terminal-alive strata, $N\ge2$.
For $r\in[0,V)$ let $S_r$ be the all-alive consensus swarm
$x_i=0$, $v_i=re_1$ for every row. Write

$$
E=1-a^2\lambda,\quad B=a(1+c),\quad
D=c-a^2\lambda(1+c),\quad
\kappa_{\rm cap}=\frac{D/q^2-a^2/s^2}{E^2},\quad
g(r)=\frac{Vr}{V-r}.
\tag{NPT.28}
$$

(The symbol $\kappa_{\rm cap}$ here is a kinetic cycle coefficient, not the
configured color phase $\kappa$.) The actual transition has a jointly
continuous positive density on neighborhoods of these states whose target
velocities are bounded by one. For any $0<r<\min\{1/2,V/2\}$,

$$
\log\frac{q_N(S_0,S_r)q_N(S_r,S_{2r})q_N(S_{2r},S_0)}
{q_N(S_r,S_0)q_N(S_{2r},S_r)q_N(S_0,S_{2r})}
=N\kappa_{\rm cap}\frac{2Vr^3}{(V-2r)(V-r)}.
\tag{NPT.29}
$$

At the unchanged reference,
$\kappa_{\rm cap}=23.9921217696\ldots>0$. At $r=1/4$ the log ratio is
$.5712409945\ldots N>0$. The strict cycle therefore persists on an open
product of three actual state neighborhoods; it is not confined to a
zero-measure consensus cylinder. The actual full stationary Doob transition
is not self-adjoint on its complete $L^2(\pi_N)$ record space.

Moreover the same cycle excludes the literal complete-state generalized
detailed-balance identification using
$\Theta(x,v,\mathfrak m)=(x,-v,\mathfrak m)$ and a
$\Theta$-invariant stationary law: this is a necessary tested condition of
that velocity-reversal identification. It does not exclude another proved
physical observable reflection or a gauge-sector/continuum reflection
construction. It does not infer a failure of the native algorithm's gauge
endpoint from nonreversibility of its full sampling transfer.
:::

:::{prf:proof}
**1. Keep every original gate pattern in the density.** At $S_r$, all reward
and squashed diversity values agree for every original companion draw.
The fitnesses are therefore equal and every living cloning acceptance is
exactly zero. There are no dead rows to revive. Thus the configured cloning
and collision maps are exactly the identity, with no recipient jitter,
before the original kinetic update. In a neighborhood of $S_r$ all eligible
rows remain alive. All fitness floors, donor widths and squashing radii are
positive; their finite companion laws, fitness values and original gate
probabilities are continuous there. In particular the probability of any
accepted clone tends to zero as the source tends to $S_r$.

The target-local bound (NPT.27) applies to every accepted pattern, for every
unbounded prepared position and every original Haar rotation. Its mixture
density is therefore bounded by that fixed compact-target bound times the
probability of an accepted clone. This contribution tends to zero uniformly
on the target compact. For the no-accepted pattern, the prepared positions
and B1 velocities depend continuously on the source. Its unique target-local
B2 inverse, guaranteed by the same proof, is smooth in these preparations
and the target. Its density is jointly continuous there. Its original gate
weight is continuous as well and equals one at consensus. More generally
there are finitely many companion and accepted-edge patterns at fixed $N$;
for each pattern the actual source maps with fixed independent jitter and
Haar variables are continuous. Dominated convergence under the
preparation-independent bound gives continuity of every pattern density.
Their finite sum proves joint continuity of the actual unmarked kernel.
This argument retains, rather than truncates, every accepted pattern and
every large jitter.

**2. Compute the exact native density at the consensus points.** Let the
source be $S_u$. Its B1 velocity is $u e_1$, its A1 position is $aue_1$,
and its O velocity is $z_i=cu e_1+q\xi_i$. The B2 output is

$$
w=(EI-a\nu L_{aue_1+az})z-a^2\lambda ue_1.
$$

If the target uncapped velocity is the consensus $g(v)e_1$, then
$(EI-a\nu L_X)z=(g(v)+a^2\lambda u)e_1$. This matrix is invertible,
with constants its $E$ eigenspace, because $E-a\nu>0$. Its inverse maps
the displayed constant to a constant. Thus every relative O innovation is
zero at this target. At that preimage the Gaussian interaction kernel is
one and $L_X=I$ on relative count modes. Its mean-mode derivative is $E$,
and its relative derivative is $E-a\nu$. The relative terminal-position
innovation is likewise zero when every target position is zero.

Choose orthonormal particle coordinates with first vector
$N^{-1/2}(1,\ldots,1)$. This volume-preserving change separates mean and
relative noises. The actual mean output before capping is the Gaussian
vector with mean $\sqrt N(Bu,Du)$ in the $e_1$ coordinate and covariance

$$
\Sigma=\begin{pmatrix}
a^2q^2+s^2&aq^2E\\aq^2E&q^2E^2
\end{pmatrix}.
$$

All other spatial components have zero mean. The exact relative Jacobian
factor is $(E-a\nu)^{d(N-1)}$, independent of $u,v$. The cap contributes
$J_g(v)^N$ with $J_g(v)=(V/(V-v))^{d+1}$. Hence for a positive constant
$A_N$ independent of the two consensus velocities,

$$
q_N(S_u,S_v)=A_NJ_g(v)^N
\exp\left[-\frac N2
(-Bu,g(v)-Du)\Sigma^{-1}(-Bu,g(v)-Du)^{\mathsf T}\right].
\tag{NPT.30}
$$

This formula includes both actual force evaluations, Gaussian viscosity,
the independent terminal diffusion, the original cap and terminal marking.
At these target points every row is inside the original box, so the killed
kernel removes no part of the calculated density.

**3. Retain the actual cap in the cycle.** The off-diagonal input-output
coefficient in the logarithm of (NPT.30) is
$N\kappa_{\rm cap}\,u g(v)$. Indeed

$$
\Sigma^{-1}=
\begin{pmatrix}
s^{-2}&-a/(Es^2)\\
-a/(Es^2)&(a^2q^2+s^2)/(q^2E^2s^2)
\end{pmatrix},\qquad
aD-BE=-a.
$$

Multiplication gives exactly (NPT.28). Terms depending only on the source
or only on the target, including all cap Jacobians, cancel around the two
orientations of the cycle. The remaining term is
$N\kappa_{\rm cap}\{r g(2r)-2r g(r)\}$, which is (NPT.29).
The reference numbers follow by direct substitution. Positive continuity
makes the strict cycle persist on open neighborhoods.

**4. Pass the cycle to the actual stationary Doob record.** The Doob density
is $q_N(s,t)e_N(t)/(\alpha_Ne_N(s))$. Its eigenfunction ratios and survival
factors cancel around a cycle, so the same strict cycle is retained.
The original common part and $e_N\ge m_N$ give positive stationary density
on small all-alive neighborhoods of these states (choose the available
analysis $r_v>2r$ in its primitive certificate). If the actual Doob kernel
were self-adjoint, detailed balance would give the equality of the forward
and reverse cycle products almost everywhere on those neighborhoods.
The continuous strict inequality just established contradicts that equality.

For the velocity-flip identification, require its stated stationary
$\Theta$-invariance and generalized detailed balance. At the consensus
endpoints $\Theta S_r=S_{-r}$. The original central-inversion covariance
of the reference, with uniqueness of its normalized eigenfunction, gives
$e_N(-s)=e_N(s)$ and
$p_N(S_{-v},S_{-u})=p_N(S_v,S_u)$. Its generalized cycle compares
$p_N(s_0,s_1)p_N(s_1,s_2)p_N(s_2,s_0)$ with
$p_N(\Theta s_1,\Theta s_0)p_N(\Theta s_2,\Theta s_1)
 p_N(\Theta s_0,\Theta s_2)$. At the consensus triple the latter is
exactly the reverse product in (NPT.29). Both products are continuous on
the corresponding open neighborhoods. Their strict inequality therefore
persists on an open product and contradicts the almost-everywhere
generalized detailed-balance cycle, rather than merely an equality at
zero-measure points. The tested velocity-flip full-state identification fails. This argument
uses its explicit invariant-law condition and makes no claim about a
physical reflection on a separately derived gauge sector.
:::


:::{prf:corollary} The literal complete-record clock reflection needs a different correspondence
:label: cor-npt-reference-full-record-clock-reflection

For the unchanged count reference at finite $N$, there exist actual bounded
real centered state modes $f,g\in L^2_0(\pi_N)$ for which its stationary
one-step reflected matrix

$$
Q=\begin{pmatrix}
\langle f,C_Nf\rangle&\langle f,C_Ng\rangle\\
\langle g,C_Nf\rangle&\langle g,C_Ng\rangle
\end{pmatrix}
\tag{NPT.31}
$$

is not Hermitian. These are native complete-swarm record modes of
{prf:ref}`def-lqft-record-fock-space`, not assigned walker-label CAR modes.
The literal link reflection evaluating the same mode on the two native
clock slices therefore fails positivity on the complete record space.
This identifies a property of that actually defined full-law reflection
form. It does not test or exclude a smaller physical gauge observable net,
a different proved physical mode/reflection correspondence, or a limit in
which this full-law defect is irrelevant to the determining physical modes.
:::

:::{prf:proof}
The preceding open cycle proves that the actual stationary pair measure
$\pi_N(ds)P_N(s,dt)$ differs from its transpose. Rectangles determine finite
measures, so there are measurable state sets $A,B$ with

$$
\int_A\pi_N(ds)P_N(s,B)\ne
\int_B\pi_N(ds)P_N(s,A).
$$

Take the actual bounded centered modes
$f=\mathbf1_A-\pi_N(A)$ and $g=\mathbf1_B-\pi_N(B)$.
Their off-diagonal inner-product difference is exactly this nonzero flux;
the common centering product cancels. Thus (NPT.31) is real and nonsymmetric.
For $z=(1,i)^{\mathsf T}$, its reflected quadratic form has imaginary part
$Q_{12}-Q_{21}\ne0$, and cannot be nonnegative real. The native stationary
Markov covariance identity identifies each entry with the actual correlation
between consecutive sampling-clock slices. The same one-particle entries
are the already constructed CAR replica correlation coefficients, so no
new physical interpretation of a walker index was imposed.
:::

(sec-npt-positive-native-regime)=
## 7. A completely computed positive native transfer regime

:::{prf:theorem} The existing singleton Brownian branch has a positive native Hamiltonian
:label: thm-npt-native-brownian-hamiltonian

Retain the complete execution record, and specialize the existing Rust
`KineticKind::Brownian` branch to one position row, $d\ge1$, amplitude
$\sigma>0$, time step $h>0$, isotropic Gaussian noise factor $b>0$, terminal
`AbsorbingBox` $D=\prod_{j=1}^d[-L_j,L_j]$ with every $L_j>0$, and no
consumed metric/noise feedback. Set $n_{\rm elite}=0$, donor history absent,
invalid-reward policy with a finite constant reward, all other readout fields
passive, and the actual Brownian-required `position_diffusion=0`,
`velocity_cap=None`, `boundary_schedule=Substeps`. The current-donor
singleton rule samples its only row. The positive fitness of that row equals
its donor fitness, so the original acceptance is zero; there are no accepted
clones, nontrivial collision components or clone jitters before extinction.
The unused velocity field, if any, is fixed as part of this component rather
than endowed with a new artificial invariant distribution.

For the real-coordinate independent-Gaussian execution, the actual killed
position kernel is exactly

$$
Q(x,dy)=k_\tau(x-y)\mathbf1_D(y)\,dy,\qquad
k_\tau(z)=(2\pi\tau^2)^{-d/2}e^{-|z|^2/(2\tau^2)},\quad
\tau^2=\sigma^2b^2h>0.
\tag{NPT.24}
$$

Let $\alpha>0$ and positive $e$ be its top eigenvalue/eigenfunction. They are
uniquely determined up to positive normalization. Then its actual Doob
stationary law is $\pi(dx)=e(x)^2dx/\int_De^2$, and its native transfer
$P=e^{-1}Qe/\alpha$ on $L^2(\pi)$ is self-adjoint, strictly positive,
compact and has simple eigenvalue one. With

$$
D_*^2=4\sum_jL_j^2,\qquad
\delta_B=\exp[-D_*^2/(2\tau^2)]\in(0,1),
$$

its centered spectrum is contained in $[0,1-\delta_B]$. Its nonzero
point eigenvalues are strictly positive, zero has no spectral projection,
and zero remains the accumulation point of this compact infinite-dimensional
transfer. Thus the operator

$$
G_B=-a_{\rm phys}^{-1}\log(P|_{L^2_0(\pi)})
$$

is the self-adjoint positive generator of the actual centered transfer at
every recorded grid time, and

$$
G_B\ge\frac{-\log(1-\delta_B)}{a_{\rm phys}}I>0.
\tag{NPT.25}
$$

Its native CAR replica transfer is exactly
$\Gamma_-(P|_{L^2_0(\pi)})=e^{-a_{\rm phys}d\Gamma(G_B)}$ on the centered Fock space.
It has a unique vacuum, positive energy and both slice and link temporal
reflection positivity. The nonvacuum even sector has gap at least twice
(NPT.25). All these assertions concern the actual same-law Doob record,
not a selected finite-history law, a freely chosen edge exponential or the
interacting viscous reference. Its force-based color is zero, so it supplies
no nontrivial Yang--Mills field.
:::

:::{prf:proof}
The executed Brownian update in `kinetic.rs` adds
$\sigma\sqrt h$ times its Gaussian innovation with factor $b$ to the current
position and then applies the configured absorbing box. The singleton
fitness/gate statement removes no configured nonzero event: its gate is
exactly zero. This proves (NPT.24).

The symmetric continuous kernel makes $Q$ self-adjoint and Hilbert--Schmidt
on $L^2(D,dx)$. For $f\ne0$, extend $f$ by zero to $\mathbb R^d$. Gaussian
convolution and its Fourier transform give

$$
\langle f,Qf\rangle
=(2\pi)^{-d}\int_{\mathbb R^d}
 e^{-\tau^2|\xi|^2/2}|\widehat f(\xi)|^2\,d\xi>0.
$$

One can justify this identity first for smooth functions, then by $L^2$
continuity of convolution. The strictly positive multiplier and injectivity
of the Fourier transform give strict positivity. The continuous strictly
positive kernel on the compact box attains a positive minimum. The top
spectral Rayleigh value is positive and attained by compactness. Replacing
an eigenfunction by its absolute value can only increase its Rayleigh value,
and strictly increases it if positive and negative values occur on sets of
positive measure. Consequently the top eigenfunction can be chosen
nonnegative, and its eigenidentity makes it continuous and everywhere
positive. If two independent top eigenfunctions existed, orthogonality would
produce a sign-changing top eigenfunction, contradicting the same strict
inequality. Hence the top eigenvalue is simple.

The Doob conjugation by multiplication with $e/\|e\|_2$ is a unitary map
from $L^2(\pi)$ to $L^2(D,dx)$. It conjugates $P$ to $Q/\alpha$, proving
self-adjointness, compactness and strict positivity. The eigenidentity gives
$P1=1$; the same conjugation shows $\pi P=\pi$ and simplicity of its unit
eigenvalue.

Write $k_-=(2\pi\tau^2)^{-d/2}e^{-D_*^2/(2\tau^2)}$ and
$k_+=(2\pi\tau^2)^{-d/2}$. Let $I_e=\int_De$. Then
$\alpha\max e\le k_+I_e$ and

$$
P(x,dy)\ge\frac{k_-e(y)}{\alpha\max e}\,dy
\ge\delta_B\frac{e(y)}{I_e}\,dy.
$$

Every nonzero centered eigenfunction is continuous and bounded by its
integral eigenidentity. The common-part decomposition contracts its
oscillation by $1-\delta_B$, so its eigenvalue is at most
$1-\delta_B$. The strictly positive compact spectrum consists of these
positive eigenvalues and its possible limit point zero, with no kernel.
The spectral theorem therefore defines the self-adjoint (possibly
unbounded) logarithm with the lower bound (NPT.25). Its exponent at each
integer grid time is exactly the actual transfer.

Second quantization preserves its exact spectral exponential on finite
wedges and extends by contraction to the Fock direct sum. Its only
zero-energy vector is the vacuum; each nonvacuum excitation costs at least
the bound in (NPT.25). For temporal reflection, its slice matrix entries
are inner products of propagated vectors
$e^{-t_i d\Gamma(G_B)}u_i$ and
$e^{-t_j d\Gamma(G_B)}u_j$, hence form a Gram matrix. A link cut inserts
$e^{-a_{\rm phys}d\Gamma(G_B)}\ge0$ between those vectors, again giving a
positive matrix. The same holds with bounded finite CAR words folded about
the cut, using their Hilbert adjoints and the positive native transfer.
This proves the stated same-law finite-time positivity and gap claims.
:::

(sec-npt-remaining-physical-identification)=
## 8. What the new native estimates discharge

:::{prf:proposition} Remaining correspondence for the interacting native gauge sector
:label: prop-npt-physical-correspondence-register

For the interacting viscous reference, the results above discharge a
primitive-parameter $L^2$ decay bound for its actual complete stationary
Doob record, its correct transition-attached descriptor process, a nonzero
native invariant gauge descriptor sector, and the corresponding full replica
transfer norm decay. They also prove a strict discrepancy with the exact
zero-energy excitation modes of the actual finite fitness-ratio edge
operator for every possible isometric native mode injection.

The physical-time reconstruction still requires the same native gauge-sector
time-reflection forms to be positive, or to have proved vanishing negative
and Hermitian defects in the declared joint limit. The existing conditional
word estimates of {prf:ref}`thm-native-pt-transfer-error` can be used only
after an actual discrepancy estimate is discharged on its actual modes.
The zero-mode discrepancy above prevents that route on the complete existing
fitness-ratio excitation space at fixed $N$; it does not determine its
physical active-sector correspondence.

The exact open cycle further characterizes the unchanged interacting count
reference: its complete Doob time transfer is non-self-adjoint and its literal
full-record link-reflection matrix fails positivity. Standard velocity-flip
generalized detailed balance also fails under the invariant-law requirement
of that identification. The native gauge sector may retain different
reflection properties; none were inferred from this full-state calculation.

The new decay rate is finite-population and depends on every consumed
parameter through the existing primitive certificate. A positive uniform
physical gap needs a population-uniform same-law estimate, the physical
sector and its native correlation correspondence, and the required
semigroup/Hilbert-space limit. The derived inequality does not make an
unknown uniform constant a hypothesis. The explicit Brownian example proves
that the existing algorithm family does contain a genuine positive native
transfer reconstruction, but it is a distinct verified configuration and
has no nonzero gauge field.

Native spatial locality budgets remain those of
{prf:ref}`thm-native-lc-b2-integrated-force` and the subsequent actual donor,
propagation and marked covariance estimates. The reducing descriptor sector
constructed here does not turn those estimates into relativistic
microcausality or spacetime covariance. No finite zero mode has been deleted,
no edge coefficient has been altered, and no imposed positive exponential
has replaced the actual reference transition.
:::

:::{prf:remark} The native projector dictionary now has a direct reflection test
:label: rem-npt-native-projector-test

{prf:ref}`thm-npg-ray-kinetic-inverse` reconstructs actual eligible
kinetics from the existing ray projector and recorded positions on regular
charts. {prf:ref}`thm-npg-reference-projector-cycle` places the
interacting reference cycle on these actual descriptor charts.
{prf:ref}`thm-npg-projector-positive-transfer-defect` constructs an
isometric injection of their modes and proves a strictly positive
discrepancy from every self-adjoint two-mode transfer.

The tested dictionary is the primitive projector refinement of the
executed ray instrument, with geometry and masks retained. The literal
averaged loop report and the distinct common-SU(3) color quotient are
specified separately. The latter has two actual hidden centroid
directions in {prf:ref}`thm-npg-su3-hidden-centroid`.
These tests therefore concern defined native choices and their exact
information content, rather than an assigned physical interpretation
of walker-label CAR modes.
:::
