# Positive reconstruction at the actual second viscous kick

(sec-nkr-register)=
## 1. Original algorithm, recorded input and physical clock

:::{prf:definition} Matched second-kick reconstruction register
:label: def-nkr-register

Retain the entire algorithm and all original source addresses of
{prf:ref}`def-nir-register`. In particular, this is the existing
real-coordinate Rust dense Gaussian **row-normalized** viscous force,
$N=2$, $d=3$, bandwidth $\rho>0$, harmonic force $-\lambda x$, original
unbounded/all-alive boundary and `velocity_cap=None`. There is no
graph/curl, metric, adaptive, elite or history feedback. The original
fitness exponents satisfy $p_r=p_s=0$, so every accepted clone gate is
zero and every collision component is a singleton at every update.
The unused companion, donor, clone-period, jitter, Haar, reward and
standardizer fields retain their configured values.

Use the positive interacting family

$$
\begin{gathered}
t=h/2>0,\quad c=e^{-\gamma h}\in(1/3,1),\quad
u=t^2\lambda=1/2,\quad
\alpha=(1-c)/(2c)\in(0,1),\\
t\nu=(1-\alpha)/2=(3c-1)/(4c)>0,\quad
q^2=b_O^2(1-c^2)/(2\gamma)>0,\quad
s^2=\sigma_x^2h\ge0,\quad p=(1-c)/2.
\end{gathered}
\tag{NKR.1}
$$

The color arm here is the **existing** spectroscopy
`ColorSource::ViscousForce` with
`ColorAlignment::MatchedKick { stage: KickStage::B2 }`.
It consumes the recorded B2 viscous force and that same record's
`force_input_velocity`. This is the post-A2, pre-B2 velocity, rather
than the terminal velocity after the second kick. Its original
threshold is $0\le\delta<\infty$, its phase is the fixed original
`PhaseScale`/`LengthScale::Fixed` value
$0<\kappa=m_{\rm phase}\ell_0/\hbar_{\rm eff}<\infty$, and its actual
matched clone-deletion, coverage, eligible and generation masks remain
present. The masks are identically eligible and un-cloned in this
configured family; force availability still uses its strict threshold.
The lecture's positive effective threshold remains positive when that
different readout is consumed.

The existing recording stride is four. Its physical interval is
$4t_*h$ for the original positive time calibration $t_*>0$.
All intermediate updates and their original Gaussian draws still run
and remain in the complete record. Selecting the B2 arm does not move
an innovation, omit a kick or install an additional stochastic state.

Rust `src/kinetic.rs::recorded_force` divides by the actual eligible
row mass. In real arithmetic the Gaussian mass of the unique nonself
row is positive, so both native viscous kicks equal
$\nu(v_2-v_1)$ and $\nu(v_1-v_2)$ exactly. The Python empty-CSR or
floored-row implementation and a finite-precision Gaussian-mass
underflow branch do not acquire this identity. Their original
arithmetic and error branches retain their separate laws.
:::

(sec-nkr-native-clock)=
## 2. Exact full native Markov clock at the B2 input

:::{prf:theorem} Stationary position and phase-velocity law at the second kick
:label: thm-nkr-stage-law

In the orthonormal centroid and relative coordinates, let $a=1$ and
$a=\alpha$, respectively. Let
$Y_{a,n}=(\widehat X_{a,n},Z_{a,n})$ be the actual post-A2 position
and pre-B2 phase velocity in update $n$. Then its entire native law is

$$
\begin{gathered}
Y_{a,n+1}=\widehat A_aY_{a,n}
              +q b\xi_{a,n+1}+s d\zeta_{a,n},\\
b=\binom t1,\qquad d=\binom p{-ct\lambda},\qquad
K_a=\begin{pmatrix}p&ta(1+c)\\-ct\lambda&ca\end{pmatrix},\qquad
L_a=\begin{pmatrix}1&0\\-t\lambda&a\end{pmatrix},\\
\widehat A_a=K_aL_a
=\begin{pmatrix}
1-u(1+c)(1+a)&ta^2(1+c)\\
-ct\lambda(1+a)&ca^2
\end{pmatrix}.
\end{gathered}
\tag{NKR.2}
$$

Here $\xi_{a,n+1}$ is the original next-update O innovation and
$\zeta_{a,n}$ is the original current-update terminal position
innovation. They are independent of the complete past at the current
B2 input and of each other. The centroid and relative source arrays
are independent orthogonal transforms of the original two row arrays.

The full stage covariance is explicitly

$$
\begin{gathered}
\widehat Q_a=q^2bb^T+s^2dd^T,\qquad
\widehat C_a=
\frac{\widehat Q_a+
             \widehat A_a\widehat Q_a\widehat A_a^T}
     {1-c^2a^4}>0,\\
\operatorname{tr}\widehat A_a=0,\qquad
\widehat A_a^2=-ca^2I,\qquad
\widehat A_a^4=c^2a^4I.
\end{gathered}
\tag{NKR.3}
$$

Consequently the actual full twelve-dimensional B2-input chain has
the unique invariant probability
$\widehat\pi=\mathcal N(0,\widehat C_+\otimes I_3)
             \otimes\mathcal N(0,\widehat C_-\otimes I_3)$.
This is the stage marginal of the already proved full stationary
position/velocity trajectory. It does not assume a stationary relative
position law from a stationary velocity law alone.
:::

:::{prf:proof}
The native first kick, first A drift, deterministic part of O and
second A drift multiply, in that order, to $K_a$. The original O
innovation contributes $q(t,1)^T\xi_{a,n}$ to their output. Thus if
$S_{a,n}$ is the actual entering full state,

$$
Y_{a,n}=K_aS_{a,n}+qb\xi_{a,n},\qquad
S_{a,n+1}=L_aY_{a,n}+s e_1\zeta_{a,n}.
\tag{NKR.4}
$$

The second identity is precisely the original B2 kick followed by
the configured terminal position diffusion. Substitute it into the
first identity at update $n+1$. Since $K_ae_1=d$, this gives
(NKR.2) with the unchanged innovation addresses and chronology.
Both displayed stage variables are already consumed and recorded by
the actual B2 arm.

The terminal-state matrix in {prf:ref}`thm-nir-full-stationarity` is
$A_a=L_aK_a$. Both factors are invertible,
$\det L_a=a>0$ and $\det K_a=ca>0$.
Hence $\widehat A_a=K_aA_aK_a^{-1}$.
Its trace, square and fourth power are exactly (NKR.3), so its
spectral radius is $\sqrt c\,a<1$. Summing the covariance series and
pairing its even and odd terms gives the displayed $\widehat C_a$.
No freely chosen transfer generator has been substituted for the
native transition.

The original O source alone is controllable:

$$
\det[b,\widehat A_ab]
 =-t[1-u(1+a)+a^2]
 =-t(a^2-a/2+1/2)\ne0.
\tag{NKR.5}
$$

Therefore $\widehat Q_a+
\widehat A_a\widehat Q_a\widehat A_a^T$ is positive definite even
when $s=0$. Independence of the original Gaussian sources gives the
claimed product Gaussian invariant law. The explicit stable affine
recurrence converges from every deterministic initial point. If an
arbitrary invariant law existed, its characteristic function would
satisfy the same finite recurrence; as the deterministic matrix power
tends to zero, continuity at zero forces that characteristic function
to the displayed Gaussian one. This also proves uniqueness without
an assumed invariant moment bound.

Finally the unique terminal-state stationary law and (NKR.4) give
$\widehat C_a=K_aC_aK_a^T+q^2bb^T$. Thus this clock is a
marginal of the same original stationary path. It has not changed the
algorithm or conditioned away an original source.
:::

:::{prf:corollary} Explicit original B2 phase-velocity variances
:label: cor-nkr-phase-variance

Write $T_a=(\widehat C_a)_{22}$ and set
$R_a=a^2-(1+a)/2$, $B_a=(1+a)p+ca^2$. Then

$$
T_a=
\frac{q^2(1+c^2R_a^2)+
 s^2c^2t^2\lambda^2(1+B_a^2)}{1-c^2a^4}>0.
\tag{NKR.6}
$$

The actual pair mean $M=(Z_1+Z_2)/2$ and difference
$\Delta=Z_2-Z_1$ are independent centered Gaussians with
$\operatorname{Cov}M=\sigma_M^2I_3$, $\sigma_M^2=T_+/2$, and
$\operatorname{Cov}\Delta=\sigma_\Delta^2I_3$,
$\sigma_\Delta^2=2T_-$. Its exact availability probability is

$$
p_A=P(\nu|\Delta|>\delta)
=\frac{\Gamma(3/2,z_0)}{\Gamma(3/2)}>0,\qquad
z_0=\frac{\delta^2}{2\nu^2\sigma_\Delta^2}.
\tag{NKR.7}
$$

In particular, no assertion that a nonzero force exceeds an arbitrary
positive threshold is needed.
:::

:::{prf:proof}
The $22$ entries of $\widehat Q_a$ and
$\widehat A_a\widehat Q_a\widehat A_a^T$ in (NKR.3) respectively
give the two terms in (NKR.6); the second row applied to $b$ is
$cR_a$ and applied to $d$ is $-ct\lambda B_a$.
The orthogonal centroid/relative Gaussian decomposition gives the
independence and variances. The original three-dimensional Gaussian
radial density gives (NKR.7).
:::

(sec-nkr-transfer)=
## 3. Native positive transfer and the complete recorded B2 color history

:::{prf:theorem} Positive reconstruction of the actual second-kick color arm
:label: thm-nkr-positive-color-history

At the existing stride four the native stage transition is exactly

$$
Y_{+,n+4}=r_+Y_{+,n}+\eta_{+,n},\qquad
Y_{-,n+4}=r_-Y_{-,n}+\eta_{-,n},\qquad
r_+=c^2,\quad r_-=c^2\alpha^4,
\tag{NKR.8}
$$

where the original accumulated Gaussian innovation has covariance
$(1-r_a^2)\widehat C_a\otimes I_3$ and is independent of the current
stage state. Its $L^2(\widehat\pi)$ operator $\widehat P_4$ is a
positive selfadjoint contraction with Hermite spectrum

$$
\widehat P_4 H_{k_+,k_-}
 =r_+^{k_+}r_-^{k_-}H_{k_+,k_-},\qquad
E_{k_+,k_-}=g_+k_++g_-k_-,\\
g_+=\frac{\gamma}{2t_*},\qquad
g_-=\frac{\gamma}{2t_*}-\frac{\log\alpha}{t_*h}>g_+.
\tag{NKR.9}
$$

Each degree index here refers to the entire six-dimensional
position/velocity block of that mode. The transfer has no zero
eigenvector; zero belongs to its infinite-dimensional spectrum as an
accumulation point. The generator
$\widehat H=-\log\widehat P_4/(4t_*h)$ has a unique vacuum.

The actual six-dimensional B2 phase-velocity algebra is a reducing
subspace. On it the native transition is the positive Mehler operator
with the same $r_+,r_-$ and the actual variances (NKR.6).
Let $\mathcal H_{B2}$ be the smallest closed subspace containing the
vacuum and invariant under that operator and multiplication by all
bounded functions of the actual B2 colors, their conjugates and
availability marks. The entire stationary **matched B2 color
history**, sampled every four updates, has site and link reflection
positivity, and its physical transfer and generator are exactly these
operators restricted to $\mathcal H_{B2}$. Its physical gap is

$$
\operatorname{gap}H_{B2}=g_+
\quad\text{for every }0<\kappa<\infty,
\quad0\le\delta<\infty.
\tag{NKR.10}
$$

This holds both for raw component color multipliers and for the
smaller component-frame projector dictionary with its full native
history. It does not assign the same gap to the common-frame scalar
orbit dictionary.
:::

:::{prf:proof}
Iterate (NKR.2) four times. Its deterministic matrix is (NKR.8),
and stationarity gives its accumulated innovation covariance. After
whitening each full Gaussian block, the transition is the usual
Gaussian kernel with independent innovation covariance
$(1-r_a^2)I_6$. Its density relative to its invariant Gaussian
product is symmetric. For the Hermite generating function, original
Gaussian integration gives

$$
E[\exp(t\cdot Y'-|t|^2/2)\mid Y]
 =\exp(r_a t\cdot Y-r_a^2|t|^2/2).
$$

Coefficient comparison gives (NKR.9), and completeness of the
Gaussian Hermite polynomials proves positivity, selfadjointness,
injectivity and the unique vacuum. The spectral logarithm therefore
has the displayed positive energies and domain.

Because the full deterministic stride map is a scalar on each mode,
the next phase velocity is $r_aZ_a$ plus an independent Gaussian
whose variance is $(1-r_a^2)T_a$. Thus functions of current phase
velocities map to functions of those phase velocities. Selfadjointness
makes this invariant subspace reducing. This is an actual native
Markov factor, rather than a proposed independent-color update.

Write $A=\mathbf1_{\nu|\Delta|>\delta}$ and
$n=\Delta/|\Delta|$ on the available locus. The actual colors are

$$
C_{1,a}=An_a e^{i\kappa(M_a-\Delta_a/2)},\qquad
C_{2,a}=-An_a e^{i\kappa(M_a+\Delta_a/2)},\qquad
P_i=C_iC_i^\dagger.
\tag{NKR.11}
$$

They are zero on their actual unavailable locus. These are bounded
measurable functions of the current native phase-velocity factor.
Their bounded multiplication operators and the native positive
transfer generate the stated closed history subspace. Since the
transfer is selfadjoint, its restriction is again selfadjoint and
positive.

For completeness, if $F$ is a bounded function of finitely many
future color samples, let $JF$ be its conditional expectation given
the current native factor state. Markov conditional independence of
past and future and native detailed balance give its site-reflected
form $\langle JF,JF\rangle$, and the link-reflected form is
$\langle JF,\widehat P_4JF\rangle$. Both are nonnegative.
Linear combinations and closure give the corresponding physical
Hilbert quotient. Successive bounded cylinder factors give exactly
the stated multiplication/transfer generated subspace and its
transfer, so the reconstruction applies to the complete color
history and its true correlations.

To identify its lowest nonzero mode, take distinct component indices
$a,b\in\{1,2,3\}$. Its existing bounded joint projector readout is

$$
D_{ab}=\operatorname{Im}[(P_1)_{ab}(P_2)_{ab}]
       =A n_a^2n_b^2\sin[2\kappa(M_a-M_b)].
$$

The independent original radial, angular and centroid Gaussian laws
give exactly

$$
E[M_aD_{ab}]
=\frac{2\kappa\sigma_M^2}{15}
      p_Ae^{-4\kappa^2\sigma_M^2}>0.
\tag{NKR.12}
$$

Here $E[n_a^2n_b^2]=1/15$ and
$E[M_a\sin(2\kappa(M_a-M_b))]
=2\kappa\sigma_M^2e^{-4\kappa^2\sigma_M^2}$ by the original
Gaussian characteristic function. Thus $D_{ab}$ has a nonzero
centroid degree-one projection. The isolated transfer eigenvalue
$r_+$ is reached by a spectral filter of $\widehat P_4$ applied to
$D_{ab}$; such filters preserve the closed history subspace.
Its whole-state degree-one energy $g_+$ is a lower bound on every
nonvacuum restricted energy. The nonzero projection attains it,
proving (NKR.10). All availability thresholds and all finite positive
phases have been retained in this exact calculation.
:::

(sec-nkr-scalar-clock)=
## 4. Actual scalar orbit sector and the failed recording strides

:::{prf:corollary} Scalar-orbit physical sector at the second kick
:label: cor-nkr-scalar-gap

For the actual common-frame $SU(3)$ scalar dictionary

$$
G=C_1^\dagger C_2
 =-A\sum_{a=1}^3n_a^2e^{i\kappa\Delta_a},\qquad
(A,G,\overline G),
\tag{NKR.13}
$$

the complete native stride-four history is positive and its exact
physical gap is $g_-$ for every finite positive phase and finite
threshold. Its smaller **primitive cyclic** space generated by these
single-time readouts retains the full classification of
{prf:ref}`thm-nso-full-history-gap` and
{prf:ref}`thm-nso-primitive-cyclic-gap`, with their original variance
$\sigma^2$ replaced by the explicitly computed
$\sigma_\Delta^2=2T_-$ in (NKR.6).
This replacement identifies a different consumed Gaussian marginal;
it changes no force, field or phase convention.

For the component projector history and every $\kappa>0$, finite
$\delta\ge0$, a native recording stride $m\ge1$ has both site and
link reflection positivity exactly when $m\equiv0\pmod4$.
If $m$ is odd, its site form already has a negative bounded color
cylinder; if $m\equiv2\pmod4$, its link form does.
:::

:::{prf:proof}
The scalar Gram (NKR.13) is independent of the centroid factor.
The native relative phase-velocity transition at stride four is
exactly the original three-dimensional positive Mehler operator
with multiplier $r_-$ and Gaussian difference variance
$\sigma_\Delta^2$. Thus every operator, original available radial
indicator and original bounded Gram multiplier in the complete
scalar proof is literally the same function of a Gaussian relative
variable, with that computed variance. The radial Laguerre bridges
at positive threshold and the squared-Gram first-mode certificate
at zero threshold prove a nonzero relative degree-one mode for
every positive phase. They identify the exact gap $g_-$.
The primitive cyclic conclusions follow from the same explicit
Hermite coefficients, including their dimensionless exceptional
phase and the distinction between a single-time cyclic space and
the full multiplication-generated history.

For the failed clocks use the existing $D_{ab}$ from (NKR.12).
It is odd under simultaneous sign reversal of the full centroid
block and even under reversal of the full relative block, even
though its actual phase velocity is correlated with its current
position inside a block. For odd $m$, the native transition at lag
$2m$ is a Gaussian Mehler operator with multipliers
$-c^m$ and $-c^m\alpha^{2m}$. Every nonzero Hermite term of
$D_{ab}$ has odd centroid degree and even relative degree.
Consequently

$$
E[D_{ab}(Y_0)D_{ab}(Y_{2m})]<0.
\tag{NKR.14}
$$

This is a negative site-reflected cylinder at strictly future time
$m$. If $m=2\ell$ with $\ell$ odd, its lag-$m$ transition has
negative multipliers $-c^\ell$ and
$-c^\ell\alpha^{2\ell}$; the same Hermite sign gives a negative
link form. Nonzero norm follows already from (NKR.12).
When $m$ is a multiple of four, both multipliers are positive and
the native Gaussian proof in the preceding theorem applies.
This proves the exact component-color clock classification without
testing a freely selected Hamiltonian.
:::

(sec-nkr-witness)=
## 5. Evaluated original parameter witness and source scope

:::{prf:proposition} Explicit positive interacting B2 witness
:label: prop-nkr-witness

Take the original values

$$
h=1,\quad c=\alpha=1/2,\quad\gamma=\log2,\quad
\lambda=2,\quad\nu=1/2,\quad q=1,\quad s=0,
\quad b_O^2=\frac{2\log2}{1-1/4},\quad\rho>0.
$$

For the actual B2 clock,

$$
\begin{gathered}
\widehat A_+=\begin{pmatrix}-1/2&3/4\\-1&1/2\end{pmatrix},\qquad
\widehat A_-=\begin{pmatrix}-1/8&3/16\\-3/4&1/8\end{pmatrix},\\
\widehat C_+=\frac23\begin{pmatrix}1&1\\1&2\end{pmatrix},\qquad
\widehat C_-=\frac1{63}\begin{pmatrix}17&30\\30&68\end{pmatrix},\\
r_+=1/4,\quad r_-=1/64,\quad
\sigma_M^2=2/3,\quad\sigma_\Delta^2=136/63,\\
g_+=\frac{\log2}{2t_*},\qquad
g_-=\frac{3\log2}{2t_*}.
\end{gathered}
\tag{NKR.15}
$$

With its original $\kappa=1$, $\delta=1/10$, the exact availability
and first-mode coefficient satisfy
$p_A=0.9993328885919585\ldots$ and
$E[M_aD_{ab}]=0.006172186490650729\ldots>0$.
The B1 and B2 phase-velocity variances differ, but their independently
identified physical stride-four energies coincide.
:::

:::{prf:proof}
Insert the displayed original parameters in (NKR.2)--(NKR.7).
Direct rational multiplication verifies
$\widehat C_a=\widehat A_a\widehat C_a\widehat A_a^T+
q^2bb^T$. Here $z_0=0.00926470588235294\ldots$ and
$p_A=\operatorname{erfc}(\sqrt{z_0})+
2\sqrt{z_0}e^{-z_0}/\sqrt\pi$; (NKR.12) gives the coefficient.
Its strict positivity already follows analytically from that exact
formula, so these decimal evaluations are not a numerical proof
assumption.
:::

This result binds the actual `MatchedKick { B2 }` arm. With no clone
or incarnation changes, `PrecedingKick` is its same color history
translated by one **contiguous actual update**, provided its original
previous frame is retained. It therefore inherits the same stationary
color law and physical reconstruction. A stored-frame gap of four
does not satisfy the code's `previous.step + 1 == frame.step` test;
without the required preceding record that arm is unavailable. The
matched B2 arm reads its own frame and has no such preceding-frame
requirement.

`PrecedingForce` and `ReferenceOffset` use the selected force with a
different phase velocity; they are separate actual readouts. A joint
B1/B2 record instrument is also a separate dictionary: positivity of
each single-stage color history does not prove reflection positivity
of their joint within-update history. The native stage covariance
and source chronology above are the objects against which such a
joint identification must be tested.

The fixed supplied phase above is an actual existing configuration.
The following section evaluates the distinct original frozen warm-up
configuration and its persistent calibration variable. The original color phase
continues to act component by component in its recorded basis;
neither the component projector sector nor its Gram restriction
has been assigned a local continuum $SU(3)$ connection by this
positive finite native reconstruction. These exact $N=2$ results do
not assert the target population-uniform continuum Yang--Mills gap.

(sec-nkr-frozen-calibration)=
## 6. Original frozen warm-up calibration and its zero-energy sector

:::{prf:definition} Literal frozen calibration register
:label: def-nkr-frozen-calibration

Retain (NKR.1), the matched B2 arm and its original fixed threshold.
Here consume the existing Rust spectroscopy
`LengthScale::WarmupCompanionMedian` or `WarmupEdgeMean` in place of
`LengthScale::Fixed`. The accumulator's actual `collect` method pools
the finite lengths of its pre-clone frames, with its existing
per-frame calibration-sample cap. Its actual `freeze` method resolves
that finite array **once**, stores the calibration in the measurement
and reuses it on every subsequent frame. The length affects this
readout and does not feed the configured stochastic dynamics.

Write $\chi=\kappa\in(0,\infty)$ on its successful positive-length
branch, and $\chi=\bot$ on its unavailable-length branch.
The latter retains the reported reason and decolored channels.
`phase_length` returns its original capability error when there is
no positive resolved length; `freeze` retains a zero length and marks
the affected channels unavailable. It has no fallback-one rule.
The affected channel is then removed from the measurement plan.
A bounded color test below evaluates to zero on this distinct missing
record label and retains that label; this is not a stored zero-color
payload or an assertion that its internal temporary `kappa=0`
`FrameState` is a valid calibrated measurement.
Additional fixed or recorded calibration fields retain their values;
only this phase is consumed by the color dictionary here.

Let $\mu_\chi$ be the actual pushforward of the original finite
warm-up record through this map. It is an explicitly defined native
source law, rather than an assumed independent calibration law.
Let $Y_F$ be the actual full B2-input state at a specified recorded
post-warm-up stage. Conditional on the complete record through this
stage, the future source addresses in (NKR.2) are their unchanged
independent Gaussian addresses.
:::

:::{prf:theorem} Full native stationary limit of the frozen calibration
:label: thm-nkr-frozen-stationary-limit

Let $\widehat C=\widehat C_+\otimes I_3\oplus
\widehat C_-\otimes I_3$. At $n\ge1$ stride-four updates after
the specified stage, the actual joint law satisfies

$$
\begin{gathered}
\|\mathcal L(\chi,Y_{F+4n})-
                \mu_\chi\otimes\widehat\pi\|_{\rm TV}
       \le E\epsilon_n(Y_F)\longrightarrow0,\\
\epsilon_n(y)=\min\left\{1,
\frac{r_+^n|\widehat C^{-1/2}y|}
          {\sqrt{2\pi(1-r_+^{2n})}}
 +\sqrt{\frac32\sum_{a\in\{+, -\}}
       [-r_a^{2n}-\log(1-r_a^{2n})]}\right\}.
\end{gathered}
\tag{NKR.16}
$$

The same bound applies, independently of its horizon, to every future
matched B2 color path starting at this stage clock and to its frozen
calibration metadata. It does not compare the already consumed
earlier B1/O fields of the starting frame as if they were fresh
future sources.

No warm-up moment assumption is required for convergence. In this
included linear native family, deterministic finite entering data
also give explicit Gaussian mean/covariance formulas for $Y_F$;
inserting them in the right side gives an ordinary primitive moment
bound whenever that stronger bound is wanted.
The actual stationary augmented transition is

$$
\mathscr P_4=I_{\chi}\otimes\widehat P_4
\quad\text{on}\quad
L^2(\mu_\chi\otimes\widehat\pi).
\tag{NKR.17}
$$

It is positive and selfadjoint. Its fixed space is precisely
$L^2(\mu_\chi)$, embedded as functions constant in the native state.
Its generator has the same nonzero Hermite energies (NKR.9) in
each available calibration fiber. The calibration variable is
static and has not been removed from the actual ensemble.
:::

:::{prf:proof}
Conditional on the entire finite record through $Y_F$, the future
stride chain has Gaussian mean
$(r_+^nY_{F,+},r_-^nY_{F,-})$ and block covariance
$(1-r_a^{2n})\widehat C_a\otimes I_3$. The original calibration is
measurable in that record and remains frozen. For equal-covariance
Gaussians the total variation of a mean shift $m$ is
$2\Phi(|\Sigma^{-1/2}m|/2)-1$, bounded by
$|\Sigma^{-1/2}m|/\sqrt{2\pi}$. This gives the first term of
(NKR.16). To compare its zero-mean covariance with the stationary
one, original Gaussian integration gives relative entropy

$$
3\sum_{a\in\{+, -\}}
          [-r_a^{2n}-\log(1-r_a^{2n})].
$$

The elementary entropy-to-TV bound gives the second term.
Triangle inequality, truncation at one and averaging the actual
finite warm-up record prove the joint bound, without assuming that
its calibration and state were independent at freezing. Since the
displayed bounded function tends to zero for every finite $y$,
dominated convergence proves the limit without an additional tail
condition. For deterministic initial data the actual finite-stage
linear recursion computes its Gaussian mean and covariance from
the configured $q,s$ and original elapsed update count.

Starting from either law, subsequent matched B2 colors are the
same Markov instrument given $(\chi,Y)$, with their original
unavailable branch retained. Data processing therefore gives the
entire future path comparison with the identical bound, for every
finite horizon and then for cylinders of the complete path.
This establishes the native product stationary limit instead of
postulating an independent frozen scale.

Since the actual stochastic transition does not change $\chi$,
its limiting stationary kernel is exactly (NKR.17). Its Gaussian
factor is already the derived positive native operator. The Hermite
expansion shows that its only fixed vectors in each factor are
constants; hence its complete fixed space is $L^2(\mu_\chi)$.
This also identifies all its zero-energy modes and the fiber
energies stated above.
:::

:::{prf:theorem} The actual color history detects its frozen calibration
:label: thm-nkr-frozen-color-zero-modes

In the derived stationary limit, let $\mathcal H_{\rm frozen}$
be the native multiplication/transfer sector generated by the
matched B2 projector colors and their original availability.
It is positive, but its complete zero-energy space is

$$
\ker H_{\rm frozen}=L^2(\mu_\chi).
\tag{NKR.18}
$$

Here the color-only dictionary identifies the unavailable label
$\bot$ and every successful finite positive phase; distinct recorded
error reasons with identical unavailable colors need their own
metadata if they are to be distinguished.
If $\mu_\chi$ is not a point mass, the vacuum vector $1$ is not the
only zero-energy color state. If a successful positive calibration
has positive probability, the exact first energy above this
**entire** ground subspace is still $g_+$.
Conditioning the physically retained calibration to a realized
successful value gives the unique-vacuum positive fiber of
(NKR.10). The unconditioned ensemble retains (NKR.18).

For the existing bounded color readout
$R_{ab}=\operatorname{Re}[(P_1)_{ab}(P_2)_{ab}]$, its actual
conditional mean is

$$
b(\chi)=\begin{cases}
\displaystyle\frac{p_A}{15}
      e^{-4\sigma_M^2\kappa^2},&\chi=\kappa>0,\\
0,&\chi=\bot.
\end{cases}
\tag{NKR.19}
$$

It is injective on this calibration state space. Its native
stationary time covariance obeys

$$
0\le\operatorname{Cov}(R_{ab,0},R_{ab,4n})
           -\operatorname{Var}_{\mu_\chi}b
       \le r_+^n/16.
\tag{NKR.20}
$$

Thus a nonconstant frozen phase produces an explicit nonzero
long-time color covariance, while preserving positivity of this
same reconstructed native process.
:::

:::{prf:proof}
Apply (NKR.11) fiberwise with the actual frozen positive $\kappa$.
The original independent centroid Gaussian and relative angular
integrations give (NKR.19). On the unavailable branch the affected
colors remain unavailable and their readout is zero. The map in
(NKR.19) is strictly decreasing on $(0,\infty)$, with positive
values, so its zero value also distinguishes $\bot$.

The full augmented ground projection $E_0$ is conditional
expectation given $\chi$. The native operator powers converge
to $E_0$ in operator norm at rate $r_+^n$ on its orthogonal
complement. Hence their application to $R_{ab}$ converges in
$L^2$ to $b(\chi)$. Closedness and native-transfer invariance put
this zero-energy vector in $\mathcal H_{\rm frozen}$.

More completely, on ground vectors $f(\chi)$ the actual bounded
color multiplication followed by ground projection obeys
$E_0M_{R_{ab}}f=b(\chi)f$. Thus all polynomials in $b$ belong
to the ground part of this sector. Polynomials are dense in
$L^2$ of its bounded pushforward law on $[0,p_A/15]$.
Injectivity of $b$ identifies that space with $L^2(\mu_\chi)$.
The full augmented Gaussian kernel has no other ground vectors,
proving (NKR.18).

On the orthogonal complement of that entire ground space the
augmented operator norm is $r_+$, so no smaller positive energy
occurs. Its existing $D_{ab}$ has the nonzero fiber first-mode
coefficient (NKR.12) at every successful finite positive phase.
When such phases have positive probability, its degree-one
projection has nonzero joint norm and is inside the closed native
sector, proving attainment of $g_+$. An everywhere unavailable
color instrument has no such excitation and makes no positive
color-gap assertion.

Finally decompose $R_{ab}=b+(R_{ab}-b)$ in the stationary product
law. The two terms are orthogonal, the first is invariant, and
positivity plus the centered Gaussian norm bound give
$0\le\langle R-b,\mathscr P_4^n(R-b)\rangle
\le r_+^nE\operatorname{Var}(R\mid\chi)$.
Since $|R|\le1/4$, this variance is at most $1/16$.
This proves (NKR.20) with the actual persistent calibration
correlation included.
:::

:::{prf:proposition} Explicit native warm-up law and nonconstant ground mode
:label: prop-nkr-frozen-witness

In the family (NKR.1) begin at the original deterministic
$x_i=v_i=0$. Consume existing current, nonself, count-one distance
companions, retain their pre-clone records, ingest the first two
consecutive updates and configure the original spectroscopy
`warmup=2`, `WarmupCompanionMedian`. The calibration-sample cap
is at least two per frame. With the original positive
$b_\ell=m_{\rm phase}/\hbar_{\rm eff}$, its frozen phase has
the exact Maxwell law

$$
\kappa=\sigma_\chi|G_3|,\qquad
\sigma_\chi^2=\frac{b_\ell^2}{2}(t^2q^2+s^2)>0.
\tag{NKR.21}
$$

It is finite, positive and nonconstant almost surely. In its
actual stationary augmented limit the color ground variance is
explicitly

$$
\operatorname{Var}b(\kappa)=
\left(\frac{p_A}{15}\right)^2
\left[(1+16\sigma_M^2\sigma_\chi^2)^{-3/2}
 -(1+8\sigma_M^2\sigma_\chi^2)^{-3}\right]>0.
\tag{NKR.22}
$$

For (NKR.15), $b_\ell=1$ and $\delta=1/10$, this is
$0.0002865766327423706\ldots$. It is an actual nonconstant
zero-energy color mode, rather than a conjectured calibration
defect.
:::

:::{prf:proof}
The original first pre-clone frame is the zero initial state.
Its two eligible nonself companion lengths are both zero.
The first completed native update has terminal positions
$x_{i,1}=tq\xi_{i,0}+s\zeta_{i,0}$, because the first B kick
and every gate vanish at that origin. The next pre-clone frame
therefore has both companion lengths equal to
$R_1=|x_{2,1}-x_{1,1}|>0$ almost surely. The original
`collect` cap is
$\max\{1,\lfloor500000/\max(\mathrm{warmup},1)\rfloor\}$,
so it retains both rows in this configured case. The pooled
array is $[0,0,R_1,R_1]$. The literal mean of the middle
order statistics gives the positive resolved length $R_1/2$.
No fallback or altered sample source enters this computation.
Its independent original Gaussian difference has covariance
$2(t^2q^2+s^2)I_3$, giving (NKR.21).

For $\kappa=\sigma_\chi|G_3|$, original Gaussian integration
gives $E e^{-a\kappa^2}=(1+2a\sigma_\chi^2)^{-3/2}$.
Apply it once and twice to (NKR.19) to get (NKR.22).
Strict positivity follows because the positive Maxwell variable
is nonconstant and $b$ is strictly monotone. The stated numerical
value follows from $\sigma_\chi^2=1/8$,
$\sigma_M^2=2/3$ and the availability in (NKR.15).
The actual state/calibration correlation at freezing remains
present and is handled by (NKR.16); the product stationary
limit used here is derived from that same trajectory.
:::

(sec-nkr-other-alignment)=
## 7. The existing preceding-force arm at its actual current phase velocity

:::{prf:theorem} Positive native preceding-force reconstruction with no terminal position diffusion
:label: thm-nkr-preceding-force

Retain the complete family (NKR.1), set its existing
$\sigma_x=0$ so $s=0$, and consume the actual spectroscopy
`ColorAlignment::PrecedingForce` with fixed supplied positive phase
and original threshold $\delta\ge0$. Its previous B2 record must be
retained as the actual contiguous preceding update. The force is
from that preceding record, while its phase velocity is the current
frame's pre-clone velocity. Then this actual color arm is an
instantaneous function of the **current original full state**.
At stride four its complete color history has a positive native
reconstruction, with exact component/projector gap $g_+$ and
exact common-frame scalar-history gap $g_-$, for every finite
$\kappa>0$ and finite $\delta\ge0$.

At zero threshold, its primitive scalar cyclic sector already has
gap $g_-$ for every positive phase. Its amplitude and phase
variables differ; the exceptional phase of the matched B2
primitive scalar sector is not assigned to this arm.
:::

:::{prf:proof}
Let the current terminal/pre-clone pair differences be
$V=v_2-v_1$ and $X=x_2-x_1$. Since $s=0$, the actual preceding
second kick leaves position unchanged and gives
$V=\alpha\Delta Z-t\lambda X$. Its inverse is exactly

$$
W=\Delta Z=(V+t\lambda X)/\alpha.
\tag{NKR.23}
$$

Thus the actual preceding force is $(\nu W,-\nu W)$ and the
configured current phase uses $v_1,v_2$. No hidden preceding
velocity is freely assigned. The current native state is the
already proved stationary Gaussian state of Chapter NIR; its
actual stride-four transfer is positive on that full state.

Write $C_-=C_\alpha$, the original current terminal covariance in
{prf:ref}`thm-nir-full-stationarity`. Its exact scalar variances are

$$
\begin{gathered}
\sigma_W^2=\frac2{\alpha^2}
 [(C_-)_{22}+2t\lambda(C_-)_{12}
                        +t^2\lambda^2(C_-)_{11}],\qquad
\sigma_V^2=2(C_-)_{22},\\
\sigma_{WV}=\frac2\alpha[(C_-)_{22}+t\lambda(C_-)_{12}],\qquad
\beta=\sigma_{WV}/\sigma_W^2,\qquad
\sigma_e^2=\sigma_V^2-\sigma_{WV}^2/\sigma_W^2>0,\\
V=\beta W+\sigma_e\eta,\qquad
W\sim\mathcal N(0,\sigma_W^2I_3),\quad
\eta\sim\mathcal N(0,I_3),\quad W\perp\eta .
\end{gathered}
\tag{NKR.24}
$$

Positive definiteness follows because the native relative covariance
$C_-$ is positive definite and $t\lambda/\alpha\ne0$ makes the
linear map $(X,V)\mapsto(W,V)$ invertible. Consequently the
conditional residual variance $\sigma_e^2$ is strictly positive.
At stride four the deterministic relative map is $r_-I$ on both
coordinates; its Gaussian innovation covariance is scaled by
$1-r_-^2$. Hence $W/\sigma_W$ and $\eta$ are independent
three-dimensional Mehler coordinates of that same native relative
factor. This decomposition is a proof coordinate change of the
existing state, not an inserted source variable.

The current centroid mean $M=(v_1+v_2)/2$ is independent of this
entire relative block, with variance
$\sigma_{M,0}^2=(C_+)_{22}/2>0$. Set
$A_W=\mathbf1_{\nu|W|>\delta}$, $n=W/|W|$ on its available
locus. The actual readouts are

$$
\begin{gathered}
C_{1,a}=A_W n_a e^{i\kappa(M_a-V_a/2)},\qquad
C_{2,a}=-A_W n_a e^{i\kappa(M_a+V_a/2)},\\
G_W=C_1^\dagger C_2
 =-A_W\sum_a n_a^2e^{i\kappa V_a},\qquad
D_{ab}=\operatorname{Im}[(P_1)_{ab}(P_2)_{ab}]
 =A_W n_a^2n_b^2\sin[2\kappa(M_a-M_b)].
\end{gathered}
\tag{NKR.25}
$$

These are bounded functions of that actual current state. The
native Gaussian operator and their bounded multiplications give
the positive complete-history reconstruction as before. Its
centroid first-mode coefficient is

$$
E[M_aD_{ab}]=
 \frac{2\kappa\sigma_{M,0}^2}{15}
 \frac{\Gamma(3/2,\delta^2/(2\nu^2\sigma_W^2))}
      {\Gamma(3/2)}e^{-4\kappa^2\sigma_{M,0}^2}>0.
\tag{NKR.26}
$$

This proves the component/projector gap $g_+$ with all actual
thresholds retained. The scalar Gram does not depend on the
centroid state, so every nonvacuum scalar energy is at least $g_-$.
We next prove that its full native history attains that energy.

First take $\delta=0$. Put $\theta=\kappa\beta$ and
$a_\theta=\theta^2\sigma_W^2/2$. Original independent Gaussian
integration in (NKR.24) gives the scalar first-Hermite coefficients

$$
\begin{aligned}
E[W_b\operatorname{Im}G_W]
 &=-e^{-\kappa^2\sigma_e^2/2}I(\theta),\\
E[\eta_b\operatorname{Im}G_W]
 &=-\kappa\sigma_e e^{-\kappa^2\sigma_e^2/2}J(\theta),\\
I(\theta)&=\theta\sigma_W^2
 \left[e^{-a_\theta}
         -\int_0^1z^{3/2}e^{-a_\theta z}\,dz\right],\\
J(\theta)&=e^{-a_\theta}
         -\int_0^1z^{1/2}e^{-a_\theta z}\,dz.
\end{aligned}
\tag{NKR.27}
$$

The $I$ formula is the original Gaussian/radial coefficient in
{prf:ref}`lem-nso-first-coefficient`. To verify the $J$ formula
directly, use $|W|^{-2}=\int_0^\infty e^{-r|W|^2}\,dr$.
Gaussian integration and $z=(1+2\sigma_W^2r)^{-1}$ give
$J=\tfrac12\int_0^1z^{1/2}(1-2a_\theta z)
e^{-a_\theta z}\,dz$.
Integrating the derivative of $z^{3/2}e^{-a_\theta z}$ proves
the displayed expression. Cross component first moments vanish
by their original coordinate sign symmetries, leaving precisely
(NKR.27).

These two coefficients cannot both vanish. If $\theta=0$,
$J=1/3$. If $\theta\ne0$ and $I=0$, then
$e^{-a_\theta}=\int_0^1z^{3/2}e^{-a_\theta z}dz$.
The strictly larger integral with power $1/2$ makes $J<0$.
Since $\kappa\sigma_e>0$, at least one genuine native relative
degree-one coefficient is nonzero for every positive phase.
It lies in even the primitive transfer-cyclic scalar space,
giving its exact gap $g_-$ at zero threshold.

Now let $\delta>0$ and put
$z=|W|^2/(2\sigma_W^2)$,
$z_0=\delta^2/(2\nu^2\sigma_W^2)>0$.
The degree-one-in-$\eta_b$, angular-degree-zero-in-$W$
part of $\operatorname{Im}G_W$ is proportional to

$$
\eta_b\mathbf1_{z>z_0}
h_0(\kappa\beta\sigma_W\sqrt{2z}),\qquad
h_0(t)=\int_0^1x^2\cos(tx)\,dx .
\tag{NKR.28}
$$

Its proportionality constant is nonzero, since
$\kappa\sigma_e e^{-\kappa^2\sigma_e^2/2}>0$.
The original entire function $h_0$ has $h_0(0)=1/3$ and
cannot vanish identically on the available radial interval.
Thus this radial function has a nonzero coefficient in the
complete basis $\eta_bL_n^{1/2}(z)$ for some $n\ge0$.
These are native relative Hermite vectors of total degree
$2n+1$.

For clarity the availability bridges have the exact matrix

$$
\begin{gathered}
d_n=\Gamma(n+3/2)/n!,\qquad
B_{mn}=\frac{\int_{z_0}^{\infty}
 z^{1/2}e^{-z}L_m^{1/2}(z)L_n^{1/2}(z)\,dz}
 {\sqrt{d_md_n}},\qquad p_0=z_0^{3/2}e^{-z_0},\\
B_{0n}=\frac{p_0(L_n^{1/2})'(z_0)}{n\sqrt{d_0d_n}}
 \quad(n\ge1),\\
B_{0n}=0,\ n>1\ \Longrightarrow\
B_{1n}=\frac{p_0L_n^{1/2}(z_0)}{(n-1)\sqrt{d_1d_n}}
 \ne0,\qquad B_{01}\ne0.
\end{gathered}
\tag{NKR.29}
$$

These follow by integrating the Laguerre Sturm--Liouville
equation and its Wronskian with $L_1^{1/2}=3/2-z$.
At a stationary derivative of $L_n$ the polynomial itself is
nonzero by uniqueness of its second-order equation; $z_0>0$
makes its leading coefficient regular. Thus a nonzero coefficient
from (NKR.28) reaches degree one directly through
$E_1M_{A_W}E_{2n+1}$, or, if that bridge is zero, through
$E_1M_{A_W}E_3M_{A_W}E_{2n+1}$.

Here $E_k$ is the native relative Hermite-degree projection,
obtainable by spectral filters of the actual positive transfer.
Availability is radial in $W$, so its multiplication preserves
the $\eta$ Hermite degree and the $W$ angular degree.
Consequently every other $\eta$ or angular component of these
filtered readouts cannot cancel this final $\eta_b$ degree-one
coefficient. All these operations belong to the complete native
scalar-history sector. Its positive-threshold gap is therefore
exactly $g_-$, completing the proof.
:::

`ReferenceOffset` reads the current B1 force with current pre-clone
velocity. In this zero-accepted-clone, singleton-collision family
that velocity is exactly the B1 input, so this arm coincides with
`MatchedKick { B1 }` and retains Chapter NIR's positive result.
`PrecedingKick` retains the matched B2 history translated by one
actual contiguous update, as already specified. `PrecedingForce`
with $s>0$ retains the actual extra terminal position innovation;
the deterministic inverse (NKR.23) is then unavailable, so the
preceding theorem does not assign it that instantaneous transfer.

The separate Python current-mean channel in
`qft/analysis.py::_compute_particle_observables` recomputes its
phase length from its current `x_before_clone` companion distances
and uses its actual fallback one. Its force is
`RunHistory.force_viscous[t-1]` with current `v_before_clone[t]`.
Both Python kinetic implementations fill that stored viscous force
in `return_info` **before the first B kick**. Thus it is a
preceding-B1/current-velocity channel, not the Rust preceding-B2
arm proved here. The Python pooled midpoint median is another
distinct finite-history calibration source. Rust warm-up median
and edge mean are the once-frozen sources evaluated in Section 6.
No current median is installed into the Rust `PhaseScale` by these
calculations, and no verified Rust-to-`RunHistory` export is used
to identify those different forces or source alignments.


(sec-nkr-scalar-calibration)=
## 8. Retained frozen calibration in the actual scalar gauge orbit

:::{prf:corollary} Frozen native calibration also leaves a scalar-orbit ground mode
:label: cor-nkr-frozen-scalar-ground

Retain the original matched B2 scalar dictionary (NKR.13) and the
derived frozen-calibration stationary law. Its complete scalar
history is positive. Let $\mathcal K_{\rm scalar}$ be its entire
native zero-energy space, the intersection of its closed history
sector with $L^2(\mu_\chi)$. If the successful phase has positive
probability, its exact first energy above $\mathcal K_{\rm scalar}$
is $g_-$. This does not silently identify
$\mathcal K_{\rm scalar}$ with the component-projector ground
space (NKR.18).

For the explicit Maxwell warm-up law (NKR.21), a nonconstant
scalar zero-energy mode is the actual conditional mean
$b_s(\kappa)=E[\operatorname{Re}G\mid\kappa]$. Put

$$
\begin{gathered}
m_j=E[|\Delta|^j\mathbf1_{\nu|\Delta|>\delta}]
 =\sigma_\Delta^j2^{j/2}
       \frac{\Gamma((j+3)/2,z_0)}{\Gamma(3/2)},\qquad
\epsilon=\sqrt{m_2/(4m_4)}>0,\\
F_3(r)=P(|G_3|\le r),\qquad
p_1=F_3(\epsilon/\sigma_\chi),\qquad
p_2=F_3(3\epsilon/\sigma_\chi)
                     -F_3(2\epsilon/\sigma_\chi)>0.
\end{gathered}
$$

Then its original scalar ground variance and covariance obey

$$
\begin{gathered}
\operatorname{Var}_{\mu_\chi} b_s
             \ge\frac14p_1p_2\epsilon^4m_2^2>0,\\
0\le\operatorname{Cov}(\operatorname{Re}G_0,
                       \operatorname{Re}G_{4n})
       -\operatorname{Var}_{\mu_\chi}b_s
       \le r_-^n.
\end{gathered}
\tag{NKR.30}
$$

Thus the retained random warm-up phase produces an explicit
zero-energy scalar color mode and a nonzero long-time scalar
covariance in the original included algorithm, even if its
calibration metadata are omitted from the observed dictionary.
:::

:::{prf:proof}
The scalar Gram depends only on the native relative Gaussian
factor and its actual frozen phase. Its augmented positive
transfer is $I_\chi\otimes P_{-,4}$. Ground projection is
conditional expectation given $\chi$, and closedness of the
history sector puts $b_s=\lim_{n\to\infty}P_{-,4}^n
\operatorname{Re}G$ in its ground part.
Its orthogonal complement is therefore centered in each
calibration fiber, with transfer norm at most $r_-$.
The finite radial-availability bridges and original Gram/squared-Gram
operators proving {prf:ref}`thm-nso-full-history-gap` are native
operators in this same augmented scalar sector. At every successful
phase one of that countable set of operators gives a nonzero
relative first-Hermite coefficient. When such phases have positive
probability, at least one operator has nonzero joint first-mode
norm. This proves attainment of $g_-$ above the entire actual
ground space without identifying additional phase aliases.

For the explicit scalar ground witness, original radial/angular
Gaussian integration and the fourth-order cosine remainder give

$$
\left|b_s(\kappa)+p_A-\frac3{10}\kappa^2m_2\right|
                   \le\kappa^4m_4/56.
\tag{NKR.31}
$$

Indeed $E\sum_a n_a^4=3/5$ and
$E\sum_a n_a^6=3/7$ on the original uniform sphere.
For $0<\kappa\le\epsilon$ this bounds $b_s$ above by
$-p_A+3\epsilon^2m_2/10+\epsilon^4m_4/56$.
For $2\epsilon\le\kappa\le3\epsilon$ it bounds $b_s$ below by
$-p_A+6\epsilon^2m_2/5-81\epsilon^4m_4/56$.
Their separation is at least
$\epsilon^2m_2(9/10-41/112)>\epsilon^2m_2/2$.
Both intervals have the displayed positive probabilities under
the actual nondegenerate Maxwell calibration law.
Apply $\operatorname{Var}b_s
=\tfrac12E[(b_s(\kappa)-b_s(\kappa'))^2]$ with two
independent copies of that actual law. The two interval orders
give its lower bound in (NKR.30).

Decomposing $\operatorname{Re}G=b_s+
(\operatorname{Re}G-b_s)$, positivity and the fiber centered
norm $r_-^n$ give the stated covariance estimate.
Its conditional variance is at most one because $|G|\le1$.
No independent sampled calibration or extra ergodicity
assumption is used.
:::
