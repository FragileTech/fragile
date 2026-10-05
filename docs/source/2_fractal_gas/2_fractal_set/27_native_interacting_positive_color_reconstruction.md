# Positive native reconstruction of an interacting stationary viscous-color history

(sec-nir-register)=
## 1. Existing algorithm and complete parameter family

:::{prf:definition} Interacting row-viscosity reconstruction register
:label: def-nir-register

Retain every original address and primitive field of
{prf:ref}`def-native-complete-execution-record`.
Use the EXISTING real-coordinate dense Gaussian row-viscosity
algorithm with $N=2$, $d=3$, bandwidth $\rho>0$, harmonic
force/reward $-\lambda x$ with $\lambda>0$, original BAOAB
$h,\gamma,b_O>0$, optional original terminal position diffusion
$\sigma_x\ge0$, unbounded/all-alive boundary, and ORIGINAL
`velocity_cap=None`.
There is no graph, curl, adaptive, history or elite feedback.
Configure the existing fitness exponents $p_r=p_s=0$.
Every original companion, donor, logistic, standardizer, gate,
clone-period, unapplied jitter and Haar parameter retains its value.
Fitness is constant positive, every accepted gate is zero and
every collision component is a singleton on EVERY update.
Both positive viscous kicks remain present.

The precise binding is Rust `QftExecutionConfig.viscosity`,
`row_normalized=true`, dense Gaussian force:
`src/kinetic.rs` divides by the actual eligible row mass and
has only its zero-mass branch. In real arithmetic a finite
Gaussian mass is positive. The unique nonself companion
therefore makes the original force EXACTLY
$F_1=\nu(v_2-v_1)$, $F_2=\nu(v_1-v_2)$ at both input stages.
This is not the Python empty-CSR/floored-row variant or a
finite-precision underflow assertion.

Put

$$
t=h/2,\quad c=e^{-\gamma h},\quad
q^2=b_O^2(1-c^2)/(2\gamma),\quad
s^2=\sigma_x^2h,\quad u=t^2\lambda.
$$

The included positive-viscosity family is

$$
\frac13<c<1,\qquad u=\frac12,\qquad
\alpha=\frac{1-c}{2c}\in(0,1),\qquad
t\nu=\frac{1-\alpha}{2}=\frac{3c-1}{4c}>0 .
\tag{NIR.1}
$$

Equivalently $0<\gamma h<\log3$, $\lambda=2/h^2$ and
$\nu=(3c-1)/(2ch)$. These are derived finite configured
parameter values, rather than assumptions on an unknown law.

Consume the original matched-B1 spectroscopy
`ColorSource::ViscousForce` with original strict threshold
$0\le\delta<\infty$, original matched clone-deletion mask,
and fixed supplied `PhaseScale`/`LengthScale::Fixed`.
Its ORIGINAL $\kappa=m_{\rm phase}\ell_0/\hbar_{\rm eff}>0$
is finite. All positive finite original phase scales and
thresholds are included.
The literal lecture clamp uses its ACTUAL effective threshold.
No unavailable row is silently made available.

The physical dictionary is the actual component-frame
raw color or projector entries and availability marks.
The complete recorded history retains every intermediate
update; the tested physical subsequence uses the existing
recording stride four, with physical interval $4t_*h>0$.
The same original trajectory is used throughout.
:::

(sec-nir-native-gaussian)=
## 2. Exact interacting full-state law

:::{prf:theorem} Unique full stationary position and velocity law
:label: thm-nir-full-stationarity

Let the actual orthonormal centroid/relative coordinates be

$$
X_+=(x_1+x_2)/\sqrt2,\quad V_+=(v_1+v_2)/\sqrt2,\qquad
X_-=(x_1-x_2)/\sqrt2,\quad V_-=(v_1-v_2)/\sqrt2 .
$$

For $a=1$ on the centroid mode and $a=\alpha$ on the
relative mode, the original full one-update law is

$$
\binom{X_a^+}{V_a^+}
 =A_a\binom{X_a}{V_a}
       +q\binom t{a-u}\xi_a+s\binom10\zeta_a,
\qquad
A_a=
\begin{pmatrix}
p&ta(1+c)\\
-t\lambda(ca+p)&ca^2-ua(1+c)
\end{pmatrix},
\quad p=1-u(1+c).
\tag{NIR.2}
$$

The orthogonal transforms of the ORIGINAL two-row Gaussian
sources are independent standard three-Gaussians by mode
and update. Define the primitive $2\times2$ covariance

$$
Q_a=q^2\binom t{a-u}\begin{pmatrix}t&a-u\end{pmatrix}
                   +s^2\binom10\begin{pmatrix}1&0\end{pmatrix},
\qquad
C_a=\frac{Q_a+A_aQ_aA_a^T}{1-c^2a^4}>0 .
\tag{NIR.3}
$$

The actual complete chain has the unique invariant probability

$$
\mu=
N(0,C_1\otimes I_3)\otimes
N(0,C_\alpha\otimes I_3),
\tag{NIR.4}
$$

in centroid/relative coordinates. Its full positions ARE stationary.
Every original position/velocity Gaussian moment follows from
these explicitly evaluated matrices.
:::

:::{prf:proof}
The original dense row force is independent of position because
its positive Gaussian mass cancels for the unique companion.
The kick $v\mapsto v+tF$ has mode coefficient $a=1$ or
$a=1-2t\nu=\alpha$. The original potential remains
$-\lambda X$ in both modes. Accordingly B1 gives
$v_1=aV-t\lambda X$, A1 gives
$p_{\rm A1}=(1-u)X+taV$, O gives
$z=caV-ct\lambda X+q\xi$, and A2 gives
$y=pX+ta(1+c)V+tq\xi$.
The ORIGINAL B2 is $w=az-t\lambda y$.
Add its terminal $s\zeta$ only to position. These identities
give exactly (NIR.2), including both viscous and potential
force inputs and all original Gaussian coefficients.

At (NIR.1), both matrices have trace zero and
$\det A_a=ca^2\in(0,1)$: the centroid trace is
$(1+c)(1-2u)=0$, and the relative trace is

$$
1+c\alpha^2-\tfrac12(1+c)(1+\alpha)
 =\tfrac12(\alpha-1)[2c\alpha-(1-c)]=0 .
$$

Cayley--Hamilton gives the actual identities

$$
A_a^2=-ca^2I,\qquad A_a^4=c^2a^4I .
\tag{NIR.5}
$$

Thus their original Lyapunov series is the convergent,
evaluated expression (NIR.3); no invariant relative law
has been presumed.
For $B_a=(t,a-u)^T$, direct multiplication gives

$$
\det[B_a,A_aB_a]
  =-ta[1+a^2-u(1+a)]
  =-ta(a^2-a/2+1/2)\ne0 .
\tag{NIR.6}
$$

The last quadratic is strictly positive.
Consequently the original OU sources alone make
$Q_a+A_aQ_aA_a^T$ positive definite, even when $s=0$.
Equation (NIR.3) is a full positive covariance in both modes.

The twelve-dimensional Gaussian (NIR.4) is invariant under
the EXACT full one-update map, since
$C_a=A_aC_aA_a^T+Q_a$.
Iteration from any state converges weakly to it because
$A_a^n\to0$ and its accumulated ORIGINAL covariance
converges to $C_a$.
For an arbitrary invariant probability, its characteristic
function obeys the iterated Gaussian transition identity;
letting $n\to\infty$ sends the initial characteristic
argument to zero and forces precisely (NIR.4).
This proves uniqueness without any moment assumption on a
second candidate invariant law.
:::

:::{prf:remark} Original parameter regimes outside this resonance
:label: rem-nir-nonresonant-parameters

The same unchanged $N=2$ row law has the exact matrix
(NIR.2) for every original $u$ and $a=\alpha=1-2t\nu$.
For $0<\alpha<1$, primitive Schur stability of BOTH modes is
exactly

$$
0<u<\min\left\{1,
 \frac{2(1+c\alpha^2)}{(1+c)(1+\alpha)}\right\}.
\tag{NIR.7}
$$

This follows from $\det A_a=ca^2<1$ and
$-(1+ca^2)<\operatorname{tr}A_a<1+ca^2$.
When it passes the original Gaussian Lyapunov series
converges; its covariance is full precisely when its original
OU/position pair is controllable.
If $s=0$, the exact OU condition is
$a[1+a^2-u(1+a)]\ne0$.
If $s>0$, $Q_a>0$ whenever $a-u\ne0$; in the remaining
case controllability is checked by $(e_1,A_ae_1)$.
The resonant family automatically passes all these tests.
No positive transfer is assigned to a different stable
matrix merely from its Gaussian invariant law.
:::

(sec-nir-native-positive-transfer)=
## 3. Positive physical transfer from the actual stationary dynamics

:::{prf:theorem} Native interacting Mehler transfer and positive energy
:label: thm-nir-positive-transfer

The actual stride-four conditional kernel has independent
centroid/relative Mehler factors

$$
S_a^+=r_aS_a+\sqrt{1-r_a^2}\,G_a,\qquad
r_a=c^2a^4\in(0,1),\qquad
G_a\sim N(0,C_a\otimes I_3).
\tag{NIR.8}
$$

Its Markov operator $P_4$ on $L^2(\mu)$ is selfadjoint,
positive and injective. Its spectrum is the closure of
the actual Hermite point eigenvalues

$$
r_1^{k_+}r_\alpha^{k_-},\qquad k_+,k_-\in\mathbb N_0,
\tag{NIR.9}
$$

with the finite Gaussian multiplicities of the two six-dimensional
modes. In particular zero is in its spectrum but is not
an eigenvalue. Its actual Hamiltonian is

$$
H=-\frac1{4t_*h}\log P_4,\qquad
\operatorname{Spec}(H)=
\{g_+k_++g_-k_-:k_+,k_-\in\mathbb N_0\},
\tag{NIR.10}
$$
$$
g_+=\frac{\gamma}{2t_*},\qquad
g_-=\frac{\gamma}{2t_*}
                         -\frac{\log\alpha}{t_*h}>g_+.
$$

The constant vacuum is unique, the actual complete-state gap
is $g_+$, and the energy operator is $\hbar_{\rm eff}H$
with physical gap $\hbar_{\rm eff}g_+$.
This is the logarithm of the tested ORIGINAL transition.
:::

:::{prf:proof}
The original four-source accumulated covariance is

$$
\sum_{j=0}^3A_a^jQ_a(A_a^j)^T
 =(1+c^2a^4)(Q_a+A_aQ_aA_a^T)
 =(1-r_a^2)C_a .
$$

Together with $A_a^4=r_aI$ this proves (NIR.8).
The orthogonal original Gaussian mode transformation
retains independence; it adds no noise.
Whitening with $C_a^{-1/2}$ gives a standard six-dimensional
Mehler factor in each mode.
The normalized original Hermite polynomials form a complete
orthonormal Gaussian basis; conditional Gaussian integration
gives exactly (NIR.9).
All eigenvalues are strictly positive, at most one, and
converge to zero with increasing total degree.
Their constant eigenvalue one is simple.
Spectral functional calculus gives (NIR.10), with domain
the coefficients $f_{k_+,k_-}$ for which
$\sum(g_+k_++g_-k_-)^2|f_{k_+,k_-}|^2<\infty$.
The smallest nonzero value is $g_+$, since $\alpha<1$.
:::

(sec-nir-full-color-history)=
## 4. Complete actual color-history reconstruction and its exact gap

:::{prf:definition} Actual retained colors and physical history sector
:label: def-nir-physical-sector

Write $\Delta=v_2-v_1$, $M=(v_1+v_2)/2$ and
$A=\mathbf1_{\{\nu|\Delta|>\delta\}}$.
The ORIGINAL matched-B1 color records are

$$
C_{1a}=A\,\frac{\Delta_a}{|\Delta|}
                    e^{i\kappa(M_a-\Delta_a/2)},\qquad
C_{2a}=-A\,\frac{\Delta_a}{|\Delta|}
                    e^{i\kappa(M_a+\Delta_a/2)},\qquad
P_i=C_iC_i^\dagger .
\tag{NIR.11}
$$

Every availability mark and original clone mask is retained.
Let $\mathcal D$ be either the actual joint projector
dictionary or the actual raw-color dictionary, with bounded
measurable passive functions.
Define $\mathcal K_{\mathcal D}$ as the smallest closed subspace
of $L^2(\mu)$ containing $1$ and invariant under
$P_4$ and multiplication by every bounded $g\in\mathcal D$.
It is determined by these native operators.
:::

:::{prf:theorem} Positive full viscous-color histories and exact physical sector
:label: thm-nir-full-color-reconstruction

The stationary actual stride-four COLOR history is reflection
positive for both site and adjacent-link reflection.
Its site-reflection quotient completion is exactly
$\mathcal K_{\mathcal D}$ with its actual Gaussian $L^2$ Gram form.
Its physical transfer and Hamiltonian are
$P_4|_{\mathcal K_{\mathcal D}}$ and
$H|_{\mathcal K_{\mathcal D}}$.
Every original finite threshold and phase coefficient $\kappa>0$
gives a nontrivial physical color sector with unique vacuum and
EXACT gap

$$
\operatorname{gap}H|_{\mathcal K_{\mathcal D}}
                            =\frac{\gamma}{2t_*}.
\tag{NIR.12}
$$

The same exact gap holds for the smaller native $P_4$-cyclic
space generated by the complete joint projector dictionary.
:::

:::{prf:proof}
For a finite future dictionary cylinder $F$, let
$JF=E[F\mid S_0]$ under the ORIGINAL stationary dynamics.
Native Markov conditioning expresses this function by a
finite product of actual multipliers and $P_4$.
Their span is dense in the subspace just defined.
Since $P_4$ is selfadjoint, this closed invariant subspace is
reducing; its restricted transfer is positive and injective.

The original reversible Mehler law has conditional independent
past and future at the reflection site.
Its site and link forms, with time-index reflection and scalar
complex conjugation, are exactly

$$
E[\overline{F\circ\theta_0}G]=\langle JF,JG\rangle_{L^2(\mu)},
\qquad
E[\overline{F\circ\theta_{1/2}}G]
                         =\langle JF,P_4JG\rangle_{L^2(\mu)} .
\tag{NIR.13}
$$

The second expression uses cylinders beginning on the future
side of that actual link. Both forms are nonnegative.
The site null vectors are exactly $\ker J$; completion gives
$\mathcal K_{\mathcal D}$, with the native transfer and log
Hamiltonian already proved.

Under (NIR.4), $M$ and $\Delta$ are independent isotropic
three-Gaussians with component variances

$$
\sigma_M^2=(C_1)_{vv}/2>0,\qquad
\sigma_\Delta^2=2(C_\alpha)_{vv}>0 .
$$

For distinct components $a,b$, the ORIGINAL joint projector
product is exactly

$$
(P_1)_{ab}(P_2)_{ab}
 =A\frac{\Delta_a^2\Delta_b^2}{|\Delta|^4}
                         e^{2i\kappa(M_a-M_b)} .
\tag{NIR.14}
$$

Let $p_A=P(\nu|\Delta|>\delta)>0$.
Its angular moment is $p_A/15$, with every radial threshold
included. Original Gaussian integration gives

$$
E\!\left[M_a\,\Im((P_1)_{ab}(P_2)_{ab})\right]
 =\frac{2\kappa\sigma_M^2p_A}{15}
                        e^{-4\kappa^2\sigma_M^2}\ne0 .
\tag{NIR.15}
$$

This holds for EVERY finite original $\kappa>0$ and threshold.
The centroid first Hermite space has eigenvalue $r_1$.
It is the unique nonconstant Hermite eigenvalue of this size:
all relative first-degree eigenvalues satisfy $r_\alpha<r_1$,
and all higher centroid degrees satisfy $r_1^k<r_1$.
Its spectral projection of the bounded actual product lies
in the native cyclic space and is nonzero by (NIR.15).
For example it is the $L^2$ limit of
$r_1^{-n}P_4^n(f-Ef)$ for this product $f$.
Thus $g_+$ is an actual physical-color energy, while every
nonconstant full-state energy is at least $g_+$.
This proves the EXACT gap in both native spaces.
The raw dictionary contains its projector products by actual
multiplication and conjugation, so the same argument applies.
The constant full-state vacuum is unique and remains so on
these sectors.
:::

:::{prf:remark} Zero-admitting tag and smaller scalar gauge quotients
:label: rem-nir-dictionary-regimes

The positive-mass spectroscopy tag has $\kappa>0$ as stated.
For a separate ORIGINAL supplied-angle tag admitting zero,
$\kappa=0$ gives a different exactly characterized sector.
The projector dictionary then depends only on $\Delta$,
is globally even in the relative full state, and has a
nonzero second-Hermite angular moment. Its cyclic and
full-history sector gap is therefore exactly $2g_-$.
The raw-color dictionary at zero phase is globally odd in
relative velocity and has a nonzero first-Hermite moment;
its corresponding exact gap is $g_-$.
For example

$$
E[A(\Delta_a^2/|\Delta|^2-1/3)\Delta_a^2]
 =4E[|\Delta|^2A]/45>0,\qquad
E[A\Delta_a^2/|\Delta|]=E[|\Delta|A]/3>0 .
$$

These prove the second- and first-degree assertions.
No centroid mode is inferred from the zero-phase dictionaries.

The common-frame scalar $SU(3)$ orbit is a smaller dictionary.
Its Gram entry is
$C_1^\dagger C_2=-A\sum_a
\Delta_a^2|\Delta|^{-2}e^{i\kappa\Delta_a}$;
it loses the centroid phase in (NIR.14).
It still has a positive native reconstructed history on
stride four as an actual measurable subdictionary, but
(NIR.15) does not assign it the full component-color gap.
Its nonconstant energies are relative Hermite energies,
at least $g_-$. A scalar-orbit spectral claim must use
its actual retained Hermite coefficients.
:::

(sec-nir-exact-clock-regimes)=
## 5. Exact recording-clock classification in the same interacting family

:::{prf:theorem} Positive clocks and failed clocks of the actual color history
:label: thm-nir-clock-classification

Keep (NIR.1), ANY original positive $\kappa>0$, finite
threshold $\delta$, and recording stride $m\ge1$.
The actual full component-color history is reflection positive
for both site and link precisely when $m$ is divisible by four.
For odd $m$ its actual site-reflection diagonal below is negative.
For $m\equiv2\pmod4$ its actual link-reflection diagonal is negative.

The full-state transfer is reversible exactly for even $m$.
For $m\equiv2\pmod4$ it is selfadjoint with negative
first-Hermite eigenvalues, and for $m\equiv0\pmod4$ it is
positive with the same $g_+,g_-$ after time normalization.
:::

:::{prf:proof}
Equation (NIR.5) gives scalar contractions for every even
stride, positive exactly at multiples of four.
For odd stride the two first-degree Gaussian mode matrices
are nonreversible. To verify this, the actual Lyapunov equation
gives

$$
(A_aC_a-C_aA_a^T)_{12}
 =\frac{ta(a^2-a/2+1/2)q^2
                  +t\lambda(ca+p)s^2}{1-ca^2}>0 .
\tag{NIR.16}
$$

Indeed $S-A_aSA_a^T=A_aQ_a-Q_aA_a^T$ and
$A_aSA_a^T=(\det A_a)S$ for a two-dimensional
antisymmetric matrix $S$.
The original OU contribution is
$-\det[B_a,A_aB_a]q^2$, and the terminal position
contribution is $t\lambda(ca+p)s^2$.
Odd powers are a nonzero scalar times $A_a$, so their
Gaussian reversibility defect persists.
This proves the full-state classification; it is not
substituted for a color test.

For the COLOR test use the actual bounded real cylinder

$$
D=\Im((P_1)_{ab}(P_2)_{ab})
 =A\frac{\Delta_a^2\Delta_b^2}{|\Delta|^4}
                           \sin(2\kappa(M_a-M_b)).
\tag{NIR.17}
$$

It is nonzero by (NIR.15), odd under inversion of the
centroid mode and even under inversion of the relative mode.
Its Hermite expansion therefore has only odd $k_+$ and
even $k_-$.
For any actual lag $2\ell$ with odd $\ell$, the native
mode transfer is exactly the reversible NEGATIVE Mehler
factor with coefficients $-c^\ell$ and
$-c^\ell\alpha^{2\ell}$. Consequently

$$
E[D(S_0)D(S_{2\ell})]
 =\sum_{\substack{k_+\ {\rm odd}\\k_-\ {\rm even}}}
 (-c^\ell)^{k_+}
 (-c^\ell\alpha^{2\ell})^{k_-}
                      \|D_{k_+,k_-}\|_2^2<0 .
\tag{NIR.18}
$$

Every nonzero term has the same STRICT negative sign;
the series converges absolutely and its first-degree
centroid term is nonzero. Original noises and masks are retained.
If $m$ is odd, the actual sampled site-reflection test of
$D$ at future sampled time one has lag $2m$, so $\ell=m$.
If $m\equiv2\pmod4$, the adjacent sampled-link test has
lag $m=2\ell$ with odd $\ell$.
These are ACTUAL color-only reflected diagonals.
Positive reconstruction for multiples of four follows
from (NIR.13) for their positive scalar contractions.
The $\kappa>0$ restriction is essential; the separate
zero-angle dictionaries keep their own even/odd sectors.
:::

(sec-nir-evaluated-witness)=
## 6. Evaluated interacting witness and arithmetic scope

:::{prf:example} Full stationary interacting original viscous-color reconstruction
:label: ex-nir-positive-witness

Choose ORIGINAL $h=1$, $\gamma=\log2$, $\lambda=2$,
$\nu=1/2$, any finite $\rho>0$, $q=1$, $\sigma_x=0$,
and $b_O=\sqrt{2\log2/(1-1/4)}$.
Then $c=\alpha=1/2$ and

$$
A_1=\begin{pmatrix}1/4&3/4\\-3/4&-1/4\end{pmatrix},
\qquad
A_\alpha=\begin{pmatrix}1/4&3/8\\-1/2&-1/4\end{pmatrix},
$$
$$
C_1=\frac23I,\qquad
C_\alpha=\frac1{63}
          \begin{pmatrix}17&-2\\-2&4\end{pmatrix},\qquad
r_1=\frac14,\quad r_\alpha=\frac1{64}.
\tag{NIR.19}
$$

The full covariance is positive:
$\det C_\alpha=64/63^2>0$.
Original fixed phase scales mass $=h_{\rm eff}=$ length $=1$
give $\kappa=1>0$.
For its original strict threshold $\delta=0.1$,
$\sigma_M^2=1/3$, $\sigma_\Delta^2=8/63$ and

$$
p_A=\operatorname{erfc}\sqrt z+
                     2\sqrt{z/\pi}\,e^{-z}>0,\qquad z=.1575,
$$
$$
E[M_aD]=\frac{2p_A}{45}e^{-4/3}>0 .
\tag{NIR.20}
$$

Thus the ACTUAL full component-color history has positive
reconstruction at stride four, unique vacuum, and exact gap
$\log2/(2t_*)$.
Its unbounded positions have the evaluated stationary law;
both original viscous forces are nonzero almost surely
and original threshold availability is paid by $p_A$.
At strides one and two its actual color-only reflection
diagonals are negative. These conclusions concern different
original recording clocks on the SAME interacting trajectory.
:::

:::{prf:remark} Complete execution, numerical branches and endpoint scope
:label: rem-nir-execution-scope

The exact Gaussian transfer is a theorem for the already
defined real-coordinate continuous-Gaussian dense row tag.
The original finite-arithmetic implementation can encounter
Gaussian mass underflow and take its configured zero-mass
branch. That variant is not silently given the cancellation
or Gaussian law above. Original finite streams, rounding,
numerical masks and version/address records remain distinct.
The Python mass-floor and geometric-CSR force are different
original algorithms and are not assigned this law.
No force normalization, cap, thermostat draw, threshold or
history calibration has been changed to obtain a transfer.

Every passive intermediate record is retained.
The positive theorem consumes the specified stride-four
color subsequence; adjoining all intermediate colors to
its physical dictionary invokes the failed-clock tests
already evaluated in Section 5.
A monotonically increasing clock/address history is not
treated as a stationary mixing coordinate.

This is a positive interacting ORIGINAL gauge-color history
with a unique full position/velocity law and identified
native physical sector. It is an $N=2$ parameter family
with the specified included row force and physical clock.
Its Gaussian color gap does not assert a population limit,
a local continuum Yang--Mills Hamiltonian, fermions, or
identification of the target physical gauge sector.
:::
