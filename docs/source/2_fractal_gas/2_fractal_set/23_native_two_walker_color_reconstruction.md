# Positive reconstruction of the existing two-walker dense-row color history

(sec-ntc-register)=
## 1. Exact algorithm, arithmetic and recorded color instrument

:::{prf:definition} Native two-walker color register
:label: def-ntc-register

Retain the complete execution record of
{prf:ref}`def-native-complete-execution-record`.
The algorithm is its EXISTING real-coordinate canonical dense gas,
with $N=2,d=3$, current nonself count-one measurement and cloning
companions, quadratic force/reward at $\lambda=0$, unbounded alive
domain, original BAOAB step $h>0$, damping $\gamma>0$ and positive
constant isotropic Gaussian diffusion amplitude $b_O$.
Put

$$
t=h/2,\quad \nu=1/t,\quad \rho>0,\quad
c=e^{-\gamma h}\in(0,1),\quad
q^2=b_O^2(1-c^2)/(2\gamma)>0,\quad
\sigma^2=q^2/(1-c^2)=b_O^2/(2\gamma).
\tag{NTC.1}
$$

Both original viscous kicks use the configured off-diagonal
ROW normalization. There is no viscosity degree cap, volume
weighting, neighbor penalty, graph feedback, elite or history
donor, adaptive force/diffusion, curl or innovation shift.
The configured final velocity cap is `None` and the configured
terminal position diffusion is zero. These are existing parameter
branches, not deletions made inside the proof.
The actual boundary is unbounded/all-alive. The original donor
widths, fitness powers, positive standardizers and rescale,
clone schedule, acceptance scale, jitter amplitude and collision
coefficient remain arbitrary configured finite values consistent
with the canonical pipeline. The two nonself companions are
the unique other walker at every update.
Retain the original positive time calibration $t_*>0$, energy
scale $\hbar_{\rm eff}>0$ and integer recording stride $m\ge1$.

This dense real-coordinate force is directly bound to
`QftExecutionConfig.viscosity` in Rust `kinetic.rs`:
the off-diagonal Gaussian mass is its actual denominator,
without a $10^{-12}$ floor. Dense and graph viscosity are
mutually exclusive. Rust records the B1 viscous force and
the matching `force_input_velocity` with their stage/version.
Its recorded color channel consumes those inputs at B1.
The finite-float zero-mass branch and addressed finite random
stream are distinct arithmetic laws; exact Gaussian Mehler
claims below use the continuous independent-Gaussian real tag.

The Python `EuclideanGas` graph branch is different:
at $N=2,d=3$ its ordinary Delaunay/Voronoi constructions return
empty edges, so that kinetic viscous force is zero. Its Gaussian
edge row normalization also has a $10^{-12}$ mass floor.
Neither is silently identified with this dense native transition.

Use a fixed ORIGINAL supplied phase coefficient $\kappa$ and
the original strict force threshold $\delta\ge0$. For the Rust
recorded color request these are the consumed values after its
literal clamps: $\kappa\in[-\pi,\pi]$, $\delta\in[10^{-15},1]$.
Other original color instruments retain their actual threshold
and phase ranges. In particular the actual Rust angle zero is
an available fixed-phase branch; no inferred position calibration
is frozen to manufacture it.

Let $\Delta=v_2-v_1$, $R=|\Delta|$, and
$A=\mathbf1_{\{\nu R>\delta\}}$. The available B1 colors are

$$
c_{1a}=\frac{\Delta_a}{R}e^{i\kappa v_{1a}},\qquad
c_{2a}=-\frac{\Delta_a}{R}e^{i\kappa v_{2a}}.
\tag{NTC.2}
$$

At unavailable slots the original raw normalization/mask remains
in the record. The executed ray-projector descriptor used below
is $P_i=A c_i c_i^\dagger$, zero on unavailable slots, together
with $A$. This is the existing projector derived from the
recorded colors, not an added sampled coordinate.
Raw color representatives and the common-$SU(3)$ scalar orbit
dictionary are distinguished below.
:::

(sec-ntc-native-transition)=
## 2. The unchanged native velocity factor

:::{prf:theorem} Exact two-walker dense-row BAOAB factor
:label: thm-ntc-native-factor

In the register above every actual cloning gate is zero.
There is no accepted clone, applied recipient jitter or
nontrivial collision component. Both B kicks are exactly
the walker-swap matrix
$S(v_1,v_2)=(v_2,v_1)$.
The complete native update satisfies

$$
v_{n+1}=cv_n+qS G_n,\qquad
x_{n+1}=x_n+t[(1+c)S v_n+qG_n],
\quad G_n\sim N(0,I_6)\ \hbox{independently}.
\tag{NTC.3}
$$

The actual B1 projector/color readout is a deterministic
function of $v_n$ through (NTC.2). Its velocity factor has
the unique invariant probability $\mu=N(0,\sigma^2I_6)$.
With that velocity marginal the entire recorded color history
is stationary for any compatible original position initial law.
The full unbounded position/velocity chain has NO invariant
probability.
:::

:::{prf:proof}
The two configured reward values are identically zero.
The unique current nonself companions give the SAME symmetric
algorithmic distance to both measurement slots. Hence every
actual standardized reward and diversity value, positive rescale
and fitness is equal across the two walkers, including every tie.
Their actual acceptance scores
$(V_j-V_i)/[s_c(V_i+\epsilon_c)]$ are zero.
The alive marks are both one, so mandatory revival never applies.
The original simultaneous-copy, jitter and collision algorithms
therefore act as identities for every realized draw. Reserved
unused random addresses remain passive records.

At either kick the one off-diagonal Gaussian weight is strictly
positive at every finite position. Dividing it by its SAME
row mass gives weight one, regardless of $\rho$ and positions.
The viscous force is $\nu(v_j-v_i)$.
Since $t\nu=1$, $v_i+t\nu(v_j-v_i)=v_j$.
After B1, A1, O, A2 and B2 respectively the velocities/positions
are $Sv$, $x+tSv$, $cSv+qG$,
$x+t[(1+c)Sv+qG]$ and $cv+qSG$.
This proves the literal stage identities.

The swap is orthogonal, so $SG$ is again an original standard
Gaussian vector, independently at each update.
The stationary covariance equation is
$\sigma^2=c^2\sigma^2+q^2$.
For any invariant velocity law with characteristic function
$\phi$, iteration gives
$\phi(u)=\phi(c^nu)\exp[-q^2(1-c^{2n})|u|^2/
 [2(1-c^2)]]$. Continuity at zero identifies uniquely $\mu$.
The same formula gives convergence from any initial velocity law.

For the complete state, write
$C_n=(x_{1n}+x_{2n})/2$, $M_n=(v_{1n}+v_{2n})/2$.
Exactly,
$M_{n+1}=cM_n+q\overline G_n$,
$C_{n+1}=C_n+t(M_n+M_{n+1})$,
where $\overline G_n\sim N(0,I_3/2)$ independently.
The existing state function

$$
B_n=C_n+\frac{t(1+c)}{1-c}M_n
$$

therefore satisfies
$B_{n+1}=B_n+[2tq/(1-c)]\overline G_n$.
If a full-state invariant probability existed, its $B$ marginal
would be invariant under this nondegenerate independent Gaussian
translation. Its characteristic function would satisfy
$\phi_B(u)=\phi_B(u)e^{-t^2q^2|u|^2/(1-c)^2}$.
It would vanish at every nonzero $u$, contradicting continuity
at zero. The stationary color factor consequently does not
assert nonexistent full-position stationarity.
:::

(sec-ntc-mehler)=
## 3. Actual positive transfer and spectral calculus

:::{prf:theorem} Native six-dimensional positive Mehler transfer
:label: thm-ntc-mehler

The velocity observable transfer is exactly

$$
\mathcal P f(v)=\int f(cv+qg)\,\gamma_6(dg)
\quad\hbox{on }L^2(\mu).
\tag{NTC.4}
$$

It is a self-adjoint positive injective contraction, with
orthogonal Hermite eigenspaces $\mathcal H_k$ of total degree $k$:

$$
\mathcal P|_{\mathcal H_k}=c^kI,\qquad
\dim\mathcal H_k=\binom{k+5}{5},\qquad
\operatorname{spec}\mathcal P=\{c^k:k\ge0\}\cup\{0\}.
\tag{NTC.5}
$$

Zero is not an eigenvalue. The constant vacuum is the unique
unit-eigenvalue sector. The native frequency Hamiltonian and
physical-energy convention are

$$
\mathsf H=-\frac1{t_*h}\log\mathcal P,\qquad
H_{\rm energy}=\hbar_{\rm eff}\mathsf H,\qquad
\mathsf H|_{\mathcal H_k}=\frac{\gamma k}{t_*}I.
\tag{NTC.6}
$$

The centered complete-velocity gap is $\gamma/t_*$, or
$\hbar_{\rm eff}\gamma/t_*$ in energy units.
For the existing recording stride $m\ge1$, its original
sampled transfer is $\mathcal P^m$ and clock $mt_*h$,
giving the SAME calibrated frequency Hamiltonian.
:::

:::{prf:proof}
The stationary pair $(v_n,v_{n+1})$ is centered jointly Gaussian
with covariance blocks $\sigma^2I_6$ and $c\sigma^2I_6$.
Its law is unchanged by exchanging its two entries, which proves
self-adjointness from their actual two-time correlation.
For a standard normal vector $z=v/\sigma$, the original kernel is
$z'=cz+\sqrt{1-c^2}\,g$.
Apply it to the generating function
$\exp(u\cdot z-|u|^2/2)$:
its conditional expectation is
$\exp(cu\cdot z-c^2|u|^2/2)$.
Comparing its polynomial coefficients proves the Hermite
eigenvalues $c^k$. Gaussian integration of two generating
functions proves their orthogonality and factorial norms.

For completeness, if an $L^2$ function is orthogonal to all
these polynomials, its Gaussian-weighted moment-generating
function has all derivatives zero at zero and is entire,
by Cauchy--Schwarz with Gaussian exponential moments.
It is therefore zero on imaginary arguments. Fourier uniqueness,
or Gaussian convolution and Fourier inversion followed by
removal of that convolution, makes the function zero.
Thus the normalized Hermites form a complete orthonormal basis.
Every coefficient is multiplied by a strictly positive $c^k$,
proving positivity and injectivity. Their finite multiplicities
and $c^k\to0$ give compactness and (NTC.5), including its
non-eigenvalue zero. Spectral calculus defines the self-adjoint
log on its natural squared-Hermite-weight domain, with the
energies stated in (NTC.6). No finite edge matrix or freely
assigned exponential transfer appears in this correspondence.
:::

(sec-ntc-history)=
## 4. The actual projector-color history reconstruction

:::{prf:definition} Native color dictionary and history sector
:label: def-ntc-color-sector

Let $D_{\kappa,\delta}(v)=(A,P_1,P_2)$ be the actual available
ray-color dictionary above. Let $\mathcal A_D$ consist of its
bounded complex measurable functions, including constants.
Define its one-time cyclic native sector

$$
\mathcal K_{\rm cyc}
=\overline{\operatorname{span}}\{
 \mathcal P^na(D): n\ge0,\ a\in\mathcal A_D\}\subset L^2(\mu).
\tag{NTC.7}
$$

For the COMPLETE color history, the physical sector is

$$
\mathcal K_{\rm hist}=
\overline{\operatorname{span}}\{
 M_{a_0}\mathcal P^{n_1}M_{a_1}\cdots
       \mathcal P^{n_j}M_{a_j}1:
 j\ge0,\ n_l\ge1,\ a_l\in\mathcal A_D\}.
\tag{NTC.8}
$$

It is the minimal closed sector containing constants and invariant
under the actual $\mathcal P$ and multiplication by original bounded
color observables. The closure of conditional predictions of ALL
finite future color cylinders is exactly $\mathcal K_{\rm hist}$.
Ordinary $\mathcal P$-cyclic closure (NTC.7) need not equal that
complete-history closure; both sectors are retained.
There is no additional Gauss projection or assumed local CAR mode.
:::

:::{prf:theorem} Positive reconstruction of the complete native ray-color history
:label: thm-ntc-color-reconstruction

The stationary native two-sided color history is reflection positive
for integer-site reflection and for the original adjacent-time
reflection. Its site-reflection quotient completion is exactly
$\mathcal K_{\rm hist}$, with inner product inherited from $L^2(\mu)$.
The reconstructed one-update transfer is the RESTRICTION of
the actual native $\mathcal P$ to this sector.
It is positive, self-adjoint and injective. Its logarithm is the
restriction of (NTC.6); the vacuum is unique and every nonvacuum
frequency is an integer multiple of $\gamma/t_*$.
:::

:::{prf:proof}
Construct the stationary two-sided velocity law from its derived
Gaussian stationary marginal and actual reversible transition.
For a bounded future color cylinder $F$ at nonnegative times define
$J F(v_0)=E[F\mid v_0]$. Conditional on the actual $v_0$,
the past and future velocity paths are independent; the reflected
past has the SAME conditional law as the future by the proved
reversibility. Thus for cylinders $F,G$,

$$
E[\overline{F\circ\theta_0}\,G]
=\langle JF,JG\rangle_{L^2(\mu)}.
\tag{NTC.9}
$$

The null space is exactly $\{F:JF=0\}$. For cylinder products,
the original Markov conditional-expectation formula gives exactly
the words in (NTC.8); linear combinations of these products
generate bounded cylinder tests and their $L^2$ completion.
This identifies the quotient Hilbert space without assuming
observed color is itself a Markov process.

Moving every cylinder time forward by one actual update gives
$J(\tau F)=\mathcal P(JF)$.
Multiplying at time zero by $a(D)$ gives $J(aF)=M_aJF$.
Hence its image closure is invariant under both operations,
and the reconstructed transfer is the claimed native restriction.
Since $\mathcal P$ is self-adjoint, this invariant closed
subspace reduces it: its orthogonal complement is invariant
by the inner-product identity. Positivity and injectivity
therefore restrict as well.

For adjacent-time reflection $n\mapsto1-n$ and future cylinders
starting at time one, the same actual Markov disintegration gives
the form $\langle JF,\mathcal P JG\rangle$.
Its diagonal is nonnegative by the derived positivity of
$\mathcal P$. This supplies positive transfer as well as the
site-reflection Gram positivity.
Spectral calculus on the reducing sector gives its log and
energies; the constant eigenspace of the full transfer is already
one dimensional. Every statement concerns the ORIGINAL color
history of (NTC.2), including the original availability mask.
:::

(sec-ntc-sector-gap)=
## 5. Exact ray-color gaps at every phase

:::{prf:definition} Original Gaussian threshold moment
:label: def-ntc-threshold-moment

Under the DERIVED stationary velocity law,
$\Delta\sim N(0,2\sigma^2I_3)$.
Its primitive active radial second moment is

$$
\mathfrak m_2=
E[R^2\mathbf1_{\{\nu R>\delta\}}]
=4\sigma^2
\frac{\Gamma(5/2,\delta^2/(4\nu^2\sigma^2))}
     {\Gamma(3/2)}>0
\quad\hbox{for every finite }\delta\ge0.
\tag{NTC.10}
$$

At zero threshold it is $6\sigma^2$.
The original available-color probability is likewise
$\Gamma(3/2,\delta^2/(4\nu^2\sigma^2))/\Gamma(3/2)>0$.
These are explicit integrals of original Gaussian draws.
:::

:::{prf:theorem} Exact zero-phase projector/history gap
:label: thm-ntc-zero-phase-gap

At the original fixed phase $\kappa=0$, both
$\mathcal K_{\rm cyc}$ and $\mathcal K_{\rm hist}$
have exact nonvacuum frequency gap $2\gamma/t_*$.
In energy units the gap is $2\hbar_{\rm eff}\gamma/t_*$.
It is nontrivial for every finite threshold $\delta$.
:::

:::{prf:proof}
At $\kappa=0$,
$P_1=P_2=A\,\Delta\Delta^\top/R^2$.
The full dictionary is invariant under central inversion
$(v_1,v_2)\mapsto(-v_1,-v_2)$.
Multiplication by its bounded functions and the native transfer
preserve the globally EVEN subspace of $L^2(\mu)$.
Both sectors are therefore even: every odd-total-degree
Hermite coefficient, in particular degree one, vanishes.

For $a\ne b$, the ORIGINAL bounded off-diagonal readout is
$(P_1)_{ab}=A\Delta_a\Delta_b/R^2$.
Its inner product with the degree-two Hermite polynomial
$\Delta_a\Delta_b$ is exactly

$$
E[\Delta_a\Delta_b(P_1)_{ab}]
=E[A\Delta_a^2\Delta_b^2/R^2]
=\mathfrak m_2/15>0 .
\tag{NTC.11}
$$

The last equality is angular Gaussian integration:
for the uniform sphere direction in three dimensions,
$E[n_a^2n_b^2]=1/15$. It also follows by expanding
$E|G|^4=15$ and the independent $E G_a^2G_b^2=1$.
This readout therefore has a nonzero degree-two projection.
That projection belongs already to $\mathcal K_{\rm cyc}$:
the isolated spectral projector for $c^2$ is a continuous
function of the native compact $\mathcal P$ on its spectrum,
and its polynomial approximations preserve cyclic closure.
Consequently both sectors contain an actual eigenvector
at frequency $2\gamma/t_*$, and their even parity excludes
any lower nonvacuum frequency. This proves the exact gap.
:::

:::{prf:theorem} Primitive positive phase interval with exact first-mode gap
:label: thm-ntc-small-phase-gap

Put

$$
C_3=\sqrt{30}\,\sigma^4/6,\qquad
\kappa_*=
\left[\frac{\mathfrak m_2}{20\sqrt{30}\,\sigma^4}\right]^{1/2}>0.
\tag{NTC.12}
$$

For every original fixed phase $0<|\kappa|\le\kappa_*$,
intersected with its instrument's actual allowed parameter range,
both native ray-color sectors have exact gap $\gamma/t_*$
and energy gap $\hbar_{\rm eff}\gamma/t_*$.
For each pair $a\ne b$ the original projector readout has
the quantitative first-Hermite coefficient

$$
\left|E[v_{1a}\,\Im(P_1)_{ab}]\right|
\ge\frac{|\kappa|\mathfrak m_2}{120}>0.
\tag{NTC.13}
$$

Hence a nonzero native first mode is proved from the actual
Gaussian law, even at a positive hard threshold.
:::

:::{prf:proof}
Let $M=(v_1+v_2)/2$, independent of $\Delta$, with
$M\sim N(0,\sigma^2I_3/2)$ and $v_1=M-\Delta/2$.
For $L=v_{1a}-v_{1b}$ define
$j(\kappa)=E[v_{1a}A\Delta_a\Delta_b R^{-2}\sin(\kappa L)]$.
Its derivative at zero is finite by original Gaussian moments.
Conditional on $\Delta$,

$$
E[v_{1a}L\mid\Delta]
=\sigma^2/2+\Delta_a(\Delta_a-\Delta_b)/4.
$$

Angular parity kills its $\sigma^2$ and cubic-cross terms
after multiplying by $A\Delta_a\Delta_b/R^2$.
The remaining term gives exactly
$j'(0)=-\mathfrak m_2/60$.

The original inequality
$|\sin u-u|\le|u|^3/6$ and
$|\Delta_a\Delta_b|/R^2\le1/2$ give

$$
|j(\kappa)+\kappa\mathfrak m_2/60|
\le\frac{|\kappa|^3}{12}E[|v_{1a}||L|^3]
\le C_3|\kappa|^3.
\tag{NTC.14}
$$

Indeed $E v_{1a}^2=\sigma^2$ and
$E L^6=15(2\sigma^2)^3=120\sigma^6$;
Cauchy--Schwarz gives the displayed constant.
The threshold indicator only decreases this remainder bound.
For $|\kappa|\le\kappa_*$ its cubic remainder is at most
$|\kappa|\mathfrak m_2/120$, proving (NTC.13).
Since $v_{1a}$ is a native degree-one Hermite function, the
actual bounded observable has a nonzero projection to
$\mathcal H_1$. The same isolated spectral-projection argument
puts it in both native sectors. The full transfer has no
frequency between zero and $\gamma/t_*$, proving the exact gap.
:::

:::{prf:theorem} Every nonzero original phase has the exact joint color gap
:label: thm-ntc-all-phase-gap

For EVERY finite original fixed phase $\kappa\ne0$ and finite
threshold $\delta$, both two-row native ray-color sectors
$\mathcal K_{\rm cyc}$ and $\mathcal K_{\rm hist}$ have exact
frequency gap $\gamma/t_*$.
For $a\ne b$, the original bounded joint color readout obeys

$$
E\!\left[v_{1a}\,
 \Im\big((P_1)_{ab}(P_2)_{ab}\big)\right]
=\frac{\kappa\sigma^2}{15}
 \frac{\Gamma(3/2,\delta^2/(4\nu^2\sigma^2))}
      {\Gamma(3/2)}
 e^{-2\kappa^2\sigma^2}\ne0 .
\tag{NTC.15}
$$

This retains the two original recorded projectors and their
common force-availability event; no new trajectory variable or
phase calibration is consumed.
:::

:::{prf:proof}
The two actual off-diagonal projector entries multiply to

$$
(P_1)_{ab}(P_2)_{ab}
=A\frac{\Delta_a^2\Delta_b^2}{R^4}
       e^{2i\kappa(M_a-M_b)}.
$$

The derived stationary center $M$ and difference $\Delta$
are independent. The expectation of $\Delta_a$ times
$A\Delta_a^2\Delta_b^2/R^4$ is zero by sign symmetry.
The remaining $M_a$ contribution satisfies, by differentiation
of its original Gaussian characteristic function,

$$
E[M_a\sin(2\kappa(M_a-M_b))]
=\kappa\sigma^2e^{-2\kappa^2\sigma^2}.
$$

Its independent angular/radial multiplier is
$E[A\Delta_a^2\Delta_b^2/R^4]=P(A=1)/15$.
This proves (NTC.15). The readout is a bounded function of
the actual joint dictionary, and $v_{1a}$ is a degree-one
Hermite polynomial. The nonzero degree-one spectral projection
therefore belongs to $\mathcal K_{\rm cyc}$ and hence to
$\mathcal K_{\rm hist}$, by the same native spectral calculus
already proved. The complete transfer's first positive
frequency is $\gamma/t_*$, proving equality.
:::

:::{prf:remark} Original dictionary distinctions and higher color spectrum
:label: rem-ntc-dictionary-gaps

Reflection positivity and the derived transfer/log correspondence
hold for EVERY fixed finite phase and finite threshold in the
original range. The actual joint projector dictionary has gap
$2\gamma/t_*$ at zero phase and $\gamma/t_*$ at every nonzero
phase. The small-phase test additionally proves a first mode
in an INDIVIDUAL row's bounded off-diagonal readout.
The complete higher spectrum is characterized by the same
original Gaussian integrals: the reducing history sector is
(NTC.8), and its degree-$k$ spectral multiplicity is the rank
of the degree-$k$ Hermite projections of those actual words.
This is a computable known-Gaussian characterization with every
$\kappa,\delta,\sigma$ retained, not an unknown-law hypothesis.

The RAW color-vector history is a larger/different descriptor.
At $\kappa=0$ its available vector $A\Delta/R$ is odd,
and $E[\Delta_a\,A\Delta_a/R]=E[AR]/3>0$.
Its sector already contains degree one, so its gap is
$\gamma/t_*$. The $2\gamma/t_*$ assertion concerns the executed
ray PROJECTORS and mask, not an invented parity of raw vectors.

A common-$SU(3)$ scalar orbit dictionary is different again.
At zero phase the two available rays coincide, so their
scalar overlap invariants are constants apart from the
original availability $A$. At positive finite threshold
that radial mask has a nonzero degree-two coefficient:

$$
E[(R^2-6\sigma^2)A]
=\frac{4\sigma^2
 [\delta^2/(4\nu^2\sigma^2)]^{3/2}
 e^{-\delta^2/(4\nu^2\sigma^2)}}{\Gamma(3/2)}>0 .
\tag{NTC.16}
$$

Its actual mask/orbit history therefore also has exact
zero-phase gap $2\gamma/t_*$.
At $\delta=0$ that pure scalar orbit is vacuum only, while
the matrix-valued ray projector remains angularly nontrivial.
No ambient projector entry is silently declared a scalar
Gauss-invariant observable. A local non-Abelian continuum
gauge sector and its spatial physical algebra remain separate.
:::

(sec-ntc-complete-parameter-scope)=
## 6. Parameter regimes, actual time and arithmetic limits

:::{prf:corollary} Evaluated existing positive color family
:label: cor-ntc-positive-family

An explicit included real-coordinate family is
$h=\gamma=b_O=\rho=1$, $N=2,d=3$,
$\lambda=0$, $\nu=2$, cap `None`, position diffusion zero,
unbounded boundary, unique current nonself companions,
and ANY original positive fitness/standardization parameters.
It has $c=e^{-1}$, $\sigma^2=1/2$.
For every original fixed finite threshold the primitive
(NTC.10) and (NTC.15) give EVERY allowed nonzero phase a
nontrivial positive native
ray-color reconstruction and gap $1/t_*$.
The actual zero-angle branch has gap $2/t_*$ for its
ray-projector history. Original finite recording strides
give these same calibrated gaps.
For example, the original Rust threshold $\delta=0.1$ gives
$u=\delta^2/(4\nu^2\sigma^2)=0.00125$,
availability $0.9999667797327$,
$\mathfrak m_2=2.9999999501767$ and
$\kappa_*=0.3309750892163$.
The consumed phase $\kappa=0.1$ therefore lies in the proved
positive interval; its first-mode coefficient is at least
$0.0024999999584$ by (NTC.13).
:::

:::{prf:proof}
All dense-row weights are positive in real coordinates,
all statuses alive, and the two computed fitnesses equal.
The chosen viscosity satisfies $t\nu=1$ exactly.
Every required original scalar parameter is finite and
positive where its configured branch consumes positivity.
The incomplete gamma integral is positive at every finite
threshold, so the sufficient phase interval is nonempty
and intersects the original Rust interval $[-\pi,\pi]$.
Apply the already derived correspondence and gaps.
:::

:::{prf:remark} Failed tests and finite-arithmetic execution
:label: rem-ntc-arithmetic-scope

The Swap identity consumes exactly TWO current eligible walkers,
unique nonself current companions, row normalization, $t\nu=1$,
zero configured force, all-alive boundary, and absence of the
named additional force/noise/cap branches. If a primitive test
fails, the corresponding original kernel remains its actual
kernel and is not assigned this Mehler or color gap.
The statements include arbitrary original $\rho>0$ and all
unapplied clone/collision parameters; their irrelevance is
proved from zero original gates, not by deleting them.

Original finite floating-point row mass can be zero by underflow.
Its recorded zero-mass force branch then invalidates exact Swap.
A multiplication/division rounding path can also differ before
zero mass. The finite random stream need not be an independent
continuous Gaussian sequence. Their complete arithmetic and
randomness comparison obligations remain explicit.
In particular no infinite-time floating-point color reconstruction
is inferred from the continuous real-coordinate theorem.
The Python two-walker empty-graph branch records zero viscous
force, hence unavailable color at every positive threshold;
it is not this nontrivial dense color family.

For a finite real Gaussian horizon and deterministic initial
relative position $x_{20}-x_{10}=r$, independent of the derived
stationary velocities, the exact relative position is

$$
x_{2n}-x_{1n}
=r-t\left[\Delta_0+2\sum_{j=1}^{n-1}\Delta_j+\Delta_n\right],
\quad E[\Delta_i\Delta_j^\top]=2\sigma^2c^{|i-j|}I_3.
\tag{NTC.17}
$$

Its Gaussian covariance is bounded above by
$8\sigma^2t^2n^2I_3$. Thus, for $L\ge1$ and $R_0>|r|$,

$$
P\!\left(\max_{0\le n\le L}|x_{2n}-x_{1n}|>R_0\right)
\le(L+1)2^{3/2}
 e^{-(R_0-|r|)^2/(32\sigma^2t^2L^2)}.
\tag{NTC.18}
$$

This is a primitive finite-horizon tail for the ORIGINAL
unbounded real innovations. Both B1 and B2 separations are
among these states. An arithmetic mass/rounding error budget
may use its actual certified switch radius together with
this tail and its independently retained arithmetic comparison
error. The tail does not certify a floating-point kernel
or random stream by itself. No innovation is clipped.
:::

:::{prf:proof}
The relative native velocity is the stationary Gaussian AR(1)
with covariance displayed in (NTC.17); the relative positions
advance by $-t(\Delta_n+\Delta_{n+1})$.
Sum these original increments to get that identity.
Their nonnegative covariance sum is at most
$2\sigma^2t^2(2n)^2I_3$.
For a standard three-dimensional Gaussian,
$P(|G|>u)\le E e^{|G|^2/4}e^{-u^2/4}
=2^{3/2}e^{-u^2/4}$.
Apply it at each original time and take the finite union.
The arithmetic remarks follow directly from the original
zero-mass and graph-force branches; they do not replace
those branches by real arithmetic.
:::

:::{prf:remark} Physical identification achieved and remaining endpoint
:label: rem-ntc-physical-scope

This reconstructs a NONTRIVIAL actual native ray-color history
with positive transfer, a unique vacuum and an explicitly
proved calibrated gap. The transfer, physical sector,
Hamiltonian and covariance all come from the same existing
stationary velocity factor and its executed color instrument.
It extends the earlier positive centroid sector to a genuine
native color descriptor.

The population is exactly two, the landscape is the existing
zero-curvature quadratic branch, and positions have no invariant
probability. It is not the fixed positive-confinement/finite-cap
reference, a population-uniform continuum non-Abelian
Yang--Mills field, or a proof of its spacelike local algebra.
Those native identifications retain their actual remaining
obligations. In particular the same-record projector sector
is explicit, rather than a freely chosen fermionic mode or
an assigned Gauss projection.
:::
