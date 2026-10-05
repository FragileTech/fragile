# Native smeared color fluctuations with the dense force correction

(sec-ncf-register)=
## 1. The original update and its observed cylinders

:::{prf:definition} Complete native color fluctuation register
:label: def-ncf-register

Retain the entire execution and marked-geometry record of
{prf:ref}`def-nmg-complete-register`. Use its existing real-coordinate
count-normalized quadratic BAOAB transition from the allowed all-alive
consensus input $x_i=v_i=0$. All original positive fitness maps agree
there, every living acceptance is zero, every component is a singleton,
and the original first force is zero. This is an exact one-update regime,
not a premise about the reference stationary law. Donor widths, reward,
diversity, standardization, jitter, collision, quadratic coefficient,
cap, terminal boundary, recording and calibration retain their values.
The complete input and its matched B2 observation have the original law

$$
z_i=qG_i,\quad X_i=tz_i,\quad Y_i=tz_i+sZ_i,
\quad (G_i,Z_i)\ \hbox{independent standard Gaussian rows}.
\tag{NCF.1}
$$

Here $t=h/2$, $q^2=b_O^2(1-e^{-2\gamma h})/(2\gamma)$ and
$s^2=\sigma_x^2h$. The configured force is exactly

$$
F_i^N=\frac1N\sum_j k(z_i,z_j),\qquad
k(z,z')=\nu e^{-t^2|z-z'|^2/(2\rho^2)}(z'-z),
$$
$$
F(z)=\mathbb E k(z,qG)
 =-\nu a_0r_0 e^{-t^2|z|^2/(2B)}z,
\quad B=\rho^2+t^2q^2,\quad
a_0=(\rho^2/B)^{d/2},\quad r_0=\rho^2/B.
\tag{NCF.2}
$$

The included self term is identically zero, preserving the actual
nonself count force. The nonself row normalization has a different
fractional influence and is not assigned (NCF.2).

For actual color/projector claims use $d=3$. Its primitive threshold
is $\delta_c\ge0$ and phase coefficient is the recorded $\kappa_c$.
For each observed cylinder choose a bounded measurable spatial test
$\varphi$, a Hermitian source matrix $T$, and a twice continuously
differentiable scalar test $\chi$ of $|F|^2$, with bounded derivatives,
vanishing for $|F|\le\delta_c+\eta$, $\eta>0$. Define

$$
P(F,z)=c(F,z)c(F,z)^\dagger,\qquad
c(F,z)=\frac F{|F|}\odot e^{i\kappa_c z},
$$
$$
H(z,y,F)=\varphi(y)\chi(|F|^2)\operatorname{tr}[TP(F,z)].
\tag{NCF.3}
$$

Set this test to zero in its vanishing-force region. It is then
$C^2$ in $F$. The original consensus-stage kernel itself gives
$|k(z,z')|\le L_K=\nu\rho/(t\sqrt e)$, by maximizing
$r e^{-t^2r^2/(2\rho^2)}$. Thus every actual force, mean force and
force comparison segment lies in $\overline B(0,L_K)$.
Use the finite evaluated derivative bounds
$L_1=\sup_{|F|\le L_K}\|D_FH\|$ and
$L_2=\sup_{|F|\le L_K}\|D_F^2H\|$.
The test uses the original available
projector: its scalar test does not change the color, threshold,
transition or any unbounded innovation. A terminal alive-only test
keeps $\mathbf1_D(y)$ as a factor of $\varphi$; no derivative in $y$
is needed below. Fixed preceding calibrations retain their recorded
values; varying calibrations must have their own joint limit.
These source and smearing parameters are observation data in the
complete record. A fixed-frame projector cylinder is a covariant
instrument, not silently an element of the common-$SU(3)$ orbit quotient.

For a concrete evaluated derivative profile, take $R_F=L_K$ and let
$C_j=\|\chi^{(j)}\|_\infty$, $r_F=\delta_c+\eta>0$.
With $M=\|\varphi\|_\infty\|T\|_F$, admissible bounds are

$$
L_1\le M(2R_FC_1+2C_0/r_F),\qquad
L_2\le M(4R_F^2C_2+2C_1+8R_FC_1/r_F+8C_0/r_F^2).
$$

Indeed the unit-vector map $F/|F|$ has first derivative norm at
most $1/|F|$ and second derivative norm at most $3/|F|^2$;
its projector has corresponding bounds $2/|F|$ and $8/|F|^2$.
The configured diagonal color phase is unitary and preserves these
force derivative bounds. The product rule gives the displayed profile.
:::

(sec-ncf-expansion)=
## 2. The full force correction at fluctuation scale

:::{prf:lemma} Exact finite-population force-availability threshold
:label: lem-ncf-finite-availability

In the actual consensus transition with $q,t,\rho,\nu>0$, the
largest possible count-force magnitude at any row is

$$
F_{\max,N}=(1-N^{-1})\frac{\nu\rho}{t\sqrt e}.
$$

If $\delta_c\ge F_{\max,N}$ every row is unavailable under the
original strict force-threshold rule. If $\delta_c<F_{\max,N}$,
an available row occurs with strictly positive original probability.
For $N=1$ the force is identically zero. The population threshold
in (NCF.9) is a distinct profile, smaller than $L_K$; it is not silently used
as a pointwise bound for every finite sample.
:::

:::{prf:proof}
Every nonself summand has norm at most $L_K$, proving the upper
bound. For $N\ge2$ set one addressed $z_i=0$ and all the other
rows equal to $(\rho/t)e_1$. Their summands at $i$ all equal
$L_Ke_1$, achieving $F_{\max,N}$.
For a strict smaller threshold continuity gives an open neighborhood
of this genuine innovation array where that row remains available.
The original independent $q$-Gaussian rows give every such open
neighborhood positive probability. This uses neither a bounded-noise
replacement nor an assumed stationary support. The strict threshold
cannot exceed the proved upper bound, including its equality case.
:::

:::{prf:theorem} Native count-field influence expansion
:label: thm-ncf-native-influence

Let $q,t,\rho>0$, $\nu\ge0$, and use the original cylinder above.
Put $W=(z,Y)$ with its actual law (NCF.1), and define

$$
S_N=\frac1N\sum_i H(z_i,Y_i,F_i^N),\quad
S=\mathbb E H(z,Y,F(z)),\quad A(W)=D_FH(z,Y,F(z)),
$$
$$
\mathcal B(z')=
\mathbb E_W\left[A(W)\cdot(k(z,z')-F(z))\right],
\quad
\Psi(W)=H(z,Y,F(z))-S+\mathcal B(z).
\tag{NCF.4}
$$

The complete original statistic satisfies

$$
\sqrt N(S_N-S)=\frac1{\sqrt N}\sum_i\Psi(W_i)+R_N,
\qquad \mathbb E|R_N|\le C/\sqrt N,
\tag{NCF.5}
$$

with a finite explicitly bounded coefficient depending on
$\nu,q,d,L_1,L_2$. In particular $\mathbb E\Psi=0$ and
$\mathbb E\Psi^2<\infty$. The term $\mathcal B$ is the response of
every dense B2 force to the same empirical O array. It is retained
at fluctuation scale; replacing those forces by independent limiting
forces would generally omit this term.
:::

:::{prf:proof}
Put $\epsilon_i=F_i^N-F(z_i)$. Conditional on $z_i$, the other
rows give independent summands; the self term is zero. Thus

$$
\mathbb E(|\epsilon_i|^2\mid z_i)
\le \frac1N\mathbb E_{z'}|k(z_i,z')|^2
                      +\frac1{N^2}|F(z_i)|^2.
$$

Since $|k(z,z')|\le\nu|z'-z|$ and the two rows are independent
centered $q$-Gaussians, integration bounds this by
$4\nu^2dq^2/N$. Taylor's formula in the actual force variable gives

$$
S_N=\frac1N\sum_i H(W_i,F(z_i))
 +\frac1{N^2}\sum_{i,j} a(W_i,W_j)+\mathcal E_N,
\qquad
a(W,W')=A(W)\cdot[k(z,z')-F(z)],
$$
$$
\mathbb E|\mathcal E_N|\le2L_2\nu^2dq^2/N.
\tag{NCF.6}
$$

This estimate includes unbounded original Gaussian rows. It makes no
maximum-over-population replacement.

For independent $W,W'$, exactly
$\mathbb E[a(W,W')\mid W]=0$ and
$\mathbb E[a(W,W')\mid W']=\mathcal B(z')$.
Define $b(W,W')=a(W,W')-\mathcal B(z')$. Both conditional
means of $b$ are zero. For distinct ordered pairs $(i,j),(k,l)$,
the covariance of their $b$ terms vanishes unless the pairs have
the same two labels: if they share only one label, condition on
that row and integrate the other independent rows using the
two zero conditional means. Disjoint pairs are independent.
Consequently

$$
\mathbb E\left|\frac1{N^2}\sum_{i\ne j}b(W_i,W_j)\right|^2
\le\frac{2\mathbb E b(W,W')^2}{N^2}.
\tag{NCF.7}
$$

All these moments are finite: Jensen and the preceding Gaussian
kernel bound give
$\mathbb E a^2\le8L_1^2\nu^2dq^2$,
$\mathbb E\mathcal B^2\le\mathbb E a^2$ and
$\mathbb E b^2\le4\mathbb E a^2$.
The diagonal terms cost at most
$\mathbb E|a(W,W)|/N\le L_1\nu q\sqrt d/N$,
because $k(z,z)=0$ and $\mathbb E|F(z)|\le\nu q\sqrt d$;
the last inequality also follows directly from (NCF.2).
Finally
$N^{-2}\sum_{i\ne j}\mathcal B(z_j)
 =(1-N^{-1})N^{-1}\sum_j\mathcal B(z_j)$.
Its difference from the desired average costs at most
$\mathbb E|\mathcal B|/N$.
Combine these estimates and (NCF.6)--(NCF.7), multiply by
$\sqrt N$, and take

$$
C=2L_2\nu^2dq^2+8L_1\nu q\sqrt d
                         +L_1\nu q\sqrt d
                         +\sqrt8L_1\nu q\sqrt d.
$$

This larger coefficient proves (NCF.5).
The independent-pair identities also give
$\mathbb E\mathcal B=0$, hence $\mathbb E\Psi=0$ and the asserted
finite second moment. No dense common-force correlation was discarded.

There is also a full $L^2$ remainder bound. With the intrinsic bound
$L_K$ above, the conditional centered summands have norm at most
$2L_K$. For independent mean-zero vectors $U_j$, expansion of the
fourth power and Cauchy--Schwarz give

$$
\mathbb E\left|\sum_jU_j\right|^4
\le\sum_j\mathbb E|U_j|^4
              +3\left(\sum_j\mathbb E|U_j|^2\right)^2.
$$

The same conditional sum and the exact missing-self term therefore yield
$\mathbb E(|\epsilon_i|^4\mid z_i)\le520L_K^4/N^2$.
Jensen bounds the squared Taylor remainder by
$L_2^2N^{-1}\sum_i|\epsilon_i|^4/4$.
For (NCF.7), use $|a|\le2L_1L_K$ and Jensen to get
$\mathbb E b^2\le16L_1^2L_K^2$.
The diagonal and omitted-average terms have $L^2$ bounds
$L_1L_K/N$ and $2L_1L_K/N$. Consequently the same remainder has

$$
\|R_N\|_2\le\frac{\sqrt{130}L_2L_K^2+
                              (\sqrt{32}+3)L_1L_K}{\sqrt N}.
\tag{NCF.11}
$$

This is an intrinsic bound for the executed kernel on this preparation,
not a truncation of its original Gaussian innovations.
:::

(sec-ncf-limit)=
## 3. A nontrivial spatially smeared native color limit

:::{prf:theorem} Joint native Gaussian color-field fluctuations
:label: thm-ncf-joint-gaussian-limit

For any finite list of the original cylinders above, the jointly
centered statistics satisfy

$$
\bigl(\sqrt N(S_{N,a}-S_a)\bigr)_{a=1}^m
\ \Rightarrow\ N(0,\Sigma),\qquad
\Sigma_{ab}=\mathbb E[\Psi_a(W)\Psi_b(W)].
\tag{NCF.8}
$$

The actual covariance also satisfies
$N\operatorname{Cov}(S_{N,a},S_{N,b})\to\Sigma_{ab}$, and
$|\mathbb ES_{N,a}-S_a|\le C_a/N$ with the coefficient in (NCF.5).

This covariance is a finite Gaussian integral in the original
$q,t,s,\nu,\rho,\kappa_c,\delta_c$ and the declared cylinder
parameters. All other complete transition parameters remain fixed;
their absence from this one-consensus-update integral follows from
their actual zero accepted gates and zero first force.

For $s>0$, a bounded spatial test $\varphi$ nonconstant modulo
Lebesgue-null sets and a cylinder
whose coefficient

$$
D(z)=\chi(|F(z)|^2)\operatorname{tr}[TP(F(z),z)]
$$

is nonzero on a positive-Gaussian-mass set, its actual variance is
strictly positive. An explicit phase-sensitive choice is
$T_{12}=i/2$, $T_{21}=-i/2$, all other entries zero, with
$\kappa_c\ne0$. It reads $\Im P_{12}$.
Choose a nonzero smooth test $\chi$ supported in a nonempty available
force annulus. Such an annulus exists precisely when

$$
\delta_c<\frac{\nu a_0r_0\sqrt B}{t\sqrt e}
\tag{NCF.9}
$$

for $\nu,q,t,\rho>0$; its actual margin $\eta$ is chosen within that
strict gap. Then $\Sigma_{aa}>0$.
This gives a nonzero native spatially smeared color fluctuation,
including its interacting force correction.

If $\kappa_c=0$, this imaginary cylinder vanishes identically; real
projector cylinders retain their computed covariances. If
$\nu=0$ or $q=0$, the corresponding finite color cylinders vanish
in this preparation. An empty population-force availability region gives
zero limiting influence and covariance; it does not claim literal
finite-$N$ unavailability on every original Gaussian outcome.
The Gaussian covariance formula still characterizes each well-defined
degenerate test. With $s=0$ the conditional-variance argument below
does not give nondegeneracy; the exact $\Sigma$ remains the criterion.
:::

:::{prf:proof}
Apply (NCF.5) to each cylinder. For any real coefficient vector
$\theta$, the independent centered row influence
$\theta\cdot\Psi$ has finite second moment.
Expansion of its characteristic function at zero, with the quadratic
remainder dominated by its second moment, gives

$$
\mathbb E e^{iu\theta\cdot\Psi/\sqrt N}
=1-\frac{u^2\theta^{\mathsf T}\Sigma\theta}{2N}+o(N^{-1}).
$$

Raising this to the $N$th power gives the Gaussian characteristic
function. Applying the same formula to arbitrary finite vectors
identifies (NCF.8); the remainder in (NCF.5) tends to zero in probability.
Its $L^2$ bound (NCF.11) additionally proves the full covariance limit
$N\operatorname{Cov}(S_{N,a},S_{N,b})\to\Sigma_{ab}$.

The correction $\mathcal B$ depends only on $z$. Consequently

$$
\operatorname{Var}(\Psi\mid z)
 =D(z)^2\operatorname{Var}[\varphi(tz+sZ)\mid z].
\tag{NCF.10}
$$

For example the Gaussian spatial test
$\varphi_\ell(y)=e^{-|y|^2/(2\ell^2)}$, $\ell>0$, gives the explicit
primitive lower covariance profile

$$
\Sigma_{aa}\ge\mathbb E_{z\sim N(0,q^2I)}D(z)^2
\left[
\left(\frac{\ell^2}{\ell^2+2s^2}\right)^{d/2}
 e^{-t^2|z|^2/(\ell^2+2s^2)}
-\left(\frac{\ell^2}{\ell^2+s^2}\right)^d
 e^{-t^2|z|^2/(\ell^2+s^2)}
\right].
$$

Completing the original terminal Gaussian square gives the two moments
in brackets. The bracket is positive when $s>0$.

For a bounded spatial function that is nonconstant up to Lebesgue-null
sets, the last variance is positive at every $z$ when $s>0$, because
the original conditional Gaussian has a strictly positive density on
all space. For the phase-sensitive choice,

$$
\Im P_{12}(F(z),z)
 =\frac{z_1z_2}{|z|^2}\sin[\kappa_c(z_1-z_2)].
$$

Every nonempty open radial annulus has a positive-measure subset
where this quantity is nonzero when $\kappa_c\ne0$. The maximum of
the original radial force in (NCF.2) is exactly the right side of
(NCF.9), as already proved in
{prf:ref}`thm-nmg-consensus-color-regime`. It gives the claimed annulus
and primitive nondegeneracy regime. Total variance and (NCF.10) prove
strict positivity without presuming that an omitted force correction
cannot cancel the color fluctuations. The stated degenerate cases
follow from the same actual formulas.
:::

(sec-ncf-selection-scope)=
## 4. Native survival and the remaining dynamical correspondence

:::{prf:corollary} Original one-step survival selection preserves this limit
:label: cor-ncf-survival-limit

For the terminal-box branch containing the consensus input, the raw
terminal rows in (NCF.1) are independent with positive alive
probability $p_D=\Pr(tqG+sZ\in D)>0$. Conditioning the complete output
on at least one alive row changes its law by exactly $(1-p_D)^N$ in
total variation. Every jointly scaled color statistic in (NCF.8)
therefore has the same selected-output limit. An alive-only cylinder
retains its original alive mask inside its spatial test.
:::

:::{prf:proof}
The all-dead probability is $(1-p_D)^N$ by the original independent
terminal Gaussian rows. The distance between a probability and its
conditioning on an event is the complement's probability. Measurable
pushforward contracts total variation, even for $N$-dependent scaling.
Apply this to the complete fluctuation vector. The actual final velocity
cap and B2 second kick do not change that terminal position event.
:::

:::{prf:remark} Exact scope of the new native field limit
:label: rem-ncf-scope

This result identifies a full finite-dimensional smeared projector
fluctuation law for an existing initial-state regime, including the
dense force's first-order empirical response. It adds a field limit
to the previously proved centroid determinant channel and local marked
geometry law. It does not establish the reference stationary field CLT,
a physical gauge-invariant Yang--Mills action, a graph-transport limit
or a multi-step stationary fluctuation process. Those results need the
actual preparation fluctuations, temporal memory and their common
physical identification. Finite arithmetic, hard-boundary or other
innovation conventions retain their original separate scope.
:::
