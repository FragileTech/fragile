# Full stationary native color reconstruction at arbitrary population

(sec-nhr-register)=
## 1. Existing full algorithm and actual recorded-field color

:::{prf:definition} Harmonic recorded-field population register
:label: def-nhr-register

Retain the complete execution and arithmetic record of
{prf:ref}`def-native-complete-execution-record`.
Use any finite $N\ge2$, $d=3$, the existing quadratic provider
$F(x)=-\lambda x$, unbounded/all-alive boundary, original BAOAB
step $h>0$, friction $\gamma>0$, original independent Gaussian
constant isotropic diffusion amplitude $b_O>0$, and the original
terminal position diffusion $\sigma_x\ge0$.
The configured cap is `None`, dense viscosity coefficient is
$\nu=0$, and no graph viscosity, curl, adaptive force/diffusion,
elite, history donor or consumed geometry feedback is configured.
All those choices are existing parameter branches.
Use the original fitness exponents $p_r=p_s=0$.
Every declared donor width, positive map, standardizer, separation
regularizer, acceptance scale/floor, applied-jitter amplitude,
collision coefficient and clone schedule retains its original value.

This theorem consumes the EXISTING explicit
`ColorSource::RecordedField` configuration

$$
\mathrm{stage}=\mathrm{B1},\qquad
\mathrm{amplitude}=\mathrm{potential\_force},\qquad
\mathrm{phase}=\mathrm{force\_input\_velocity},\qquad
\mathrm{alignment}=\mathrm{Matched}.
\tag{NHR.1}
$$

Rust `recorded_force` emits these aligned fields even when the
viscous coefficient is zero; `spectroscopy::frame::stage_kick`
reads their common stage and version. The original matched
clone-deletion rule remains active. The default
`ViscousForce` color source is a DIFFERENT configured instrument:
at $\nu=0$ it is unavailable and is not assigned the colors below.
No force field is replaced by the recorded-field option.
The kinetic provider still uses its actual harmonic force.

Use the original fixed phase calibration:
`PhaseScale` has finite positive mass $m_c$, finite positive
action scale $\hbar_c$, and `LengthScale::Fixed` with $\ell_0>0$.
Its consumed coefficient is $\kappa=m_c\ell_0/\hbar_c>0$.
The original threshold is finite $\delta\ge0$.
Warm-up/history or current-output calibrations are not silently
assigned this fixed coefficient. The literal zero-phase endpoint
is outside this spectroscopy tag's positive-mass validation.
An instrument that actually permits zero phase retains its own
zero-phase result below.

Write

$$
t=h/2,\quad c=e^{-\gamma h}\in(0,1),\quad
q^2=\frac{b_O^2(1-c^2)}{2\gamma}>0,\quad
s^2=\sigma_x^2h,\quad E=1-t^2\lambda .
\tag{NHR.2}
$$

Choose ORIGINAL integer recording stride $m\ge3$ and an integer
$j$ with $1\le j<m/2$, and consume the derived parameter family

$$
\theta=\frac{2\pi j}{m}\in(0,\pi),\qquad
t^2\lambda=\frac12\left[
 1-\frac{2\sqrt c}{1+c}\cos\theta\right].
\tag{NHR.3}
$$

The actual intermediate updates are retained.
The reconstructed color history consists of the complete
configured B1 color arrays at the declared stride, with their
original masks and alignment labels.
It does not treat all intermediate-stage archive fields as
additional reflection-positive observables.
Keep the original time calibration $t_*>0$ and energy calibration
$\hbar_{\rm eff}>0$; its recorded physical interval is $mt_*h$.
The exact identities use the existing continuous-independent-Gaussian
real-coordinate analytic tag. Finite arithmetic, addressed finite
streams, numerical failure branches and any enabled float comparison
errors retain their separately declared laws.
:::

(sec-nhr-native-gaussian-law)=
## 2. Actual full stationary law and primitive positive transfer

:::{prf:theorem} Native full harmonic population and resonant recorded transfer
:label: thm-nhr-full-positive-transfer

Every actual clone gate is zero in this register.
The unchanged full state is a product of $3N$ two-coordinate
linear Gaussian chains. On each original $(x_a,v_a)$ pair put

$$
\begin{gathered}
p=1-t^2\lambda(1+c),\quad b=t(1+c),\\
D=c-t^2\lambda(1+c),\quad k=-t\lambda(c+p),\\
A=\begin{pmatrix}p&b\\k&D\end{pmatrix},\qquad
Q=q^2\binom{t}{E}(t,E)+s^2\binom{1}{0}(1,0).
\end{gathered}
\tag{NHR.4}
$$

Its exact update is $Z_{n+1}=AZ_n+\eta_n$ with independent
centered Gaussian innovations of covariance $Q$.
The full physical state has the unique invariant probability
$\mu_N=N(0,C\otimes I_{3N})$, with all original position
components followed by all velocity components. The covariance is the FINITE
primitive formula

$$
r=c^{m/2}\in(0,1),\qquad
A^m=rI,\qquad
C=\frac1{1-r^2}
       \sum_{l=0}^{m-1}A^lQ(A^l)^\top\succ0 .
\tag{NHR.5}
$$

The actual recorded-state transfer is exactly

$$
\mathcal P_m f(z)=
 E f\!\left(rz+\sqrt{1-r^2}\,C_N^{1/2}G\right),
\qquad C_N=C\otimes I_{3N},\quad G\sim N(0,I_{6N}).
\tag{NHR.6}
$$

It is positive, injective and self-adjoint on $L^2(\mu_N)$.
Its Hermite spectrum and native frequency Hamiltonian are

$$
\mathcal P_m|_{\mathcal H_k}=r^kI,\qquad
\mathsf H_m=-\frac1{mt_*h}\log\mathcal P_m,\qquad
\mathsf H_m|_{\mathcal H_k}=\frac{\gamma k}{2t_*}I.
\tag{NHR.7}
$$

The vacuum is unique. The exact complete-state gap
$\gamma/(2t_*)$, or $\hbar_{\rm eff}\gamma/(2t_*)$ in energy
units, is uniform in every admitted $N$.
:::

:::{prf:proof}
The original nonnegative fitness exponents permit zero:
each positive channel raised to exponent zero is one.
Thus every consumed positive fitness is equal, regardless of
the actual rewards, distances or sampled donors. Every living
acceptance score is zero, including ties and any clone schedule.
The all-alive boundary excludes revival. No accepted copy, applied
jitter or nontrivial collision component occurs for ANY original
source realization. Reserved donor, gate, jitter and Haar records
remain passive.

At zero viscosity the two actual B kicks use $-\lambda x$.
B1 gives $v_1=v-t\lambda x$; A1 gives
$p_1=Ex+tv$; O gives $z=cv_1+q\xi$; A2 gives
$y=p_1+tz$; B2 gives $u=z-t\lambda y$.
The terminal position map adds the original $s\zeta$ to $y$,
after B2; its configured diffusion does not enter that earlier
force evaluation. Hence $(y+s\zeta,u)=A(x,v)+q(t,E)\xi+(s,0)\zeta$.
This proves (NHR.4), including the stage placement of BOTH
original Gaussian sources.

Direct calculation gives $\det A=c$ and
$\operatorname{tr}A=(1+c)(1-2t^2\lambda)$.
Equation (NHR.3) puts its distinct eigenvalues at
$\sqrt c\,e^{\pm i\theta}$; it also gives $0<t^2\lambda<1$.
Its characteristic polynomial therefore proves $A^m=rI$
without changing any intermediate update.
The stable affine recursion has covariance
$\sum_{l\ge0}A^lQ(A^l)^\top$.
Grouping this convergent series into original $m$-step blocks
gives exactly (NHR.5).

Positivity of $C$ follows already from the original OU source.
For $B=q(t,E)^\top$,

$$
\det[B,AB]=-2tE q^2\ne0 .
$$

Thus $BB^\top+ABB^\top A^\top$ is positive definite;
the finite sum in (NHR.5) contains both terms.
The additional original $s^2$ contribution is nonnegative,
including its existing zero-amplitude branch.
Iteration of an invariant characteristic function, using
$A^n\to0$, gives the displayed unique Gaussian law.
It also gives convergence from any original initial state law.
Different pairs use their original independent source coordinates,
so their full invariant law is the product asserted, rather than
an assumed invariant relative law.

The actual $m$-step Gaussian innovation has covariance
$C-A^mC(A^m)^\top=(1-r^2)C$.
This proves the literal transition (NHR.6).
Its stationary pair law is Gaussian with covariance blocks
$C_N$ and $rC_N$, invariant under swapping the two times.
Whitening by the DERIVED positive covariance gives the same
Mehler generating-function calculation as
{prf:ref}`thm-ntc-mehler` in dimension $6N$.
Consequently its complete orthogonal Hermite basis has
eigenvalues $r^k>0$, finite multiplicities and no zero eigenvector;
zero belongs to the compact transfer's spectrum.
Taking the log on its actual reducing Hermite domains gives
(NHR.7), since $-\log r/(mh)=\gamma/2$.
There is no freely assigned transfer or Hamiltonian.
:::

(sec-nhr-full-color-history)=
## 3. The configured complete color history and its physical sector

:::{prf:theorem} Full native recorded-field color reflection and reconstruction
:label: thm-nhr-color-reconstruction

At every recorded B1 input the original color is

$$
a_i=\mathbf1_{\{\lambda|x_i|>\delta\}},\qquad
(c_i)_a=-\frac{x_{ia}}{|x_i|}e^{i\kappa v_{ia}}
       \quad\hbox{when }a_i=1 .
\tag{NHR.8}
$$

The literal zero extension and matched clone-deletion mask
remain its original values. The latter deletes no row because
every original gate was proved zero.
Both the COMPLETE raw-color dictionary
$D_{\rm raw}=(a_i,a_ic_i)_{i=1}^N$ and its existing derived
ray-projector dictionary
$D_{\rm ray}=(a_i,P_i=a_ic_ic_i^\dagger)_{i=1}^N$
have positive stationary color histories at the declared stride.

For either dictionary their exact physical Hilbert space is

$$
\mathcal K_D=\overline{\operatorname{span}}\{
 M_{f_0}\mathcal P_m^{n_1}M_{f_1}\cdots
             \mathcal P_m^{n_l}M_{f_l}1:
 l\ge0,\ n_b\ge1,\ f_b\in L^\infty(D)\}.
\tag{NHR.9}
$$

Their site-reflection quotient is $\mathcal K_D$ with its
actual $L^2(\mu_N)$ Gram form. The adjacent-time reflection form
is also positive. The reconstructed transfer is the actual
restriction $\mathcal P_m|_{\mathcal K_D}$ and the Hamiltonian
is the restriction of (NHR.7). Each nontrivial color sector has
a unique vacuum and gap at least $\gamma/(2t_*)$.
This physical correspondence holds for every finite consumed
threshold and phase calibration of the configured instrument.
:::

:::{prf:proof}
The emitted B1 fields are exactly $-\lambda x_i$ and $v_i$
at their common actual version. Strict availability and original
normalization give (NHR.8). Their source masks cover every
finite alive state; the actual force-threshold mask remains.
The derived stationary covariance has full support, so its
available probability is positive at every finite threshold.

The complete recorded-state chain is the actual positive reversible
Gaussian chain (NHR.6). For each future color cylinder $F$ define
$JF=E[F\mid Z_0]$. Conditional past/future independence and the
proved time reversibility give

$$
E[\overline{F\circ\theta_0}G]=\langle JF,JG\rangle,\qquad
E[\overline{F\circ\theta_{1/2}}G]
                         =\langle JF,\mathcal P_mJG\rangle .
\tag{NHR.10}
$$

In the second form future cylinders start at the next recorded
time and homogeneity identifies their conditional predictors.
The first form's null space is exactly $\ker J$.
The actual Markov conditional-expectation formula produces the
words in (NHR.9), and finite cylinder products generate its
completion. Multiplication at the zero recorded time and shifting
by the original stride correspond to $M_f$ and $\mathcal P_m$.
Thus this is the minimal closed complete-history sector under
the ACTUAL observation and transfer, not merely a one-time
cyclic span or an assumed autonomous Markov law for color.
It is invariant under the self-adjoint transfer, hence reducing.
Positivity, injectivity, the log, vacuum uniqueness and the full
transfer's spectral lower bound restrict to it. Its observables
are the original supplied color source; no Gauss projection,
fermionic field or hypothetical local algebra is assigned.
:::

(sec-nhr-color-gap)=
## 4. Primitive exact color-sector gaps

:::{prf:definition} Derived original threshold moments and velocity residual
:label: def-nhr-color-moments

Put $X=C_{11}>0$, $V=C_{22}>0$, $B=C_{12}$,
$\beta=B/X$ and $\upsilon=V-B^2/X>0$.
These are the finite primitive matrix entries in (NHR.5).
For one original row, $x\sim N(0,XI_3)$ and
$v=\beta x+\sqrt\upsilon\,G$, with $G$ independent standard
three-Gaussian under the DERIVED stationary law.
Writing $R=|x|$, $u=\delta^2/(2\lambda^2X)$ gives

$$
p_A=P(\lambda R>\delta)
      =\frac{\Gamma(3/2,u)}{\Gamma(3/2)}>0,\qquad
m_b=E[R^b\mathbf1_{\{\lambda R>\delta\}}]
      =(2X)^{b/2}
        \frac{\Gamma((3+b)/2,u)}{\Gamma(3/2)}
        \quad(b\ge0).
\tag{NHR.11}
$$

All moments retain the ORIGINAL unbounded Gaussian law.
:::

:::{prf:theorem} Exact projector gap on a primitive positive phase interval
:label: thm-nhr-projector-gap

For arbitrary original $\sigma_x\ge0$ define

$$
C_0=\frac{\upsilon p_A}{15}+\frac{\beta^2m_2}{35}>0,\qquad
C_3=\frac{2\sqrt{30}}3V^2,\qquad
\kappa_{\rm ray}=\sqrt{C_0/C_3}>0 .
\tag{NHR.12}
$$

For every original fixed $0<\kappa\le\kappa_{\rm ray}$ the
complete native ray-color sector has exact frequency gap
$\gamma/(2t_*)$ and energy gap
$\hbar_{\rm eff}\gamma/(2t_*)$, uniformly in $N$.
For $a\ne b$ its original bounded squared-entry readout satisfies

$$
\left|E[v_a\Im(P_{ab}^2)]\right|\ge\kappa C_0>0 .
\tag{NHR.13}
$$

In the existing branch $\sigma_x=0$, this conclusion holds for
EVERY original finite positive phase: its exact coefficient is

$$
C=\begin{pmatrix}
q^2/[\lambda(1-c^2)]&0\\
0&E q^2/(1-c^2)
\end{pmatrix},\qquad
E[v_a\Im(P_{ab}^2)]
 =\frac{2\kappa Vp_A}{15}e^{-4\kappa^2V}\ne0 .
\tag{NHR.14}
$$

If a different original instrument allows zero phase, its
ray-projector history at that endpoint has exact gap $\gamma/t_*$.
The present positive-mass spectroscopy tag does not consume that
endpoint.
:::

:::{prf:proof}
The original squared entry is
$P_{ab}^2=a\,x_a^2x_b^2R^{-4}e^{2i\kappa(v_a-v_b)}$.
Conditional on $x$,
$E[v_a(v_a-v_b)\mid x]=\upsilon+\beta^2x_a(x_a-x_b)$.
The angular identities
$E[n_a^2n_b^2]=1/15$ and
$E[n_a^4n_b^2]=1/35$ for $a\ne b$, and odd sign symmetry,
give derivative at phase zero exactly $2C_0$.
This derivative is computed from the known original Gaussian
formula; it does not require a zero-phase algorithm parameter.

Since $x_a^2x_b^2/R^4\le1/4$,
$|\sin z-z|\le|z|^3/6$ gives

$$
|E[v_a\Im(P_{ab}^2)]-2\kappa C_0|
\le\frac{\kappa^3}{3}E[|v_a||v_a-v_b|^3]
\le C_3\kappa^3 .
$$

The marginal velocities are independent across ambient components
with variance $V$, so Cauchy--Schwarz uses
$E[v_a^2]=V$ and $E[(v_a-v_b)^6]=120V^3$.
For the stated interval the error is at most $\kappa C_0$.
The bounded actual projector function has a nonzero projection
to the full transfer's degree-one Hermite space. Its isolated
spectral projection belongs to the complete color-history
sector, by polynomial approximation to that actual transfer's
spectral projection. The complete-state gap proves equality.

When $s=0$, substitution of the displayed diagonal covariance
into $C=ACA^\top+q^2(t,E)^\top(t,E)$ proves it exactly.
The stationary position and velocity are consequently independent.
Their angular/radial factor is $p_A/15$ and the original Gaussian
identity
$E[v_a\sin(2\kappa(v_a-v_b))]
 =2\kappa V e^{-4\kappa^2V}$ proves (NHR.14) for all phases.
At a permitted zero phase the dictionary is globally even,
so every odd-total-degree Hermite coefficient vanishes.
Its actual off-diagonal projector has
$E[x_ax_bP_{ab}]=m_2/15>0$.
It therefore contains degree two and its exact gap is twice
the full Mehler gap, as asserted.
:::

:::{prf:theorem} Exact raw-color gap without changing its representative
:label: thm-nhr-raw-gap

For every original $\sigma_x\ge0$ put

$$
\kappa_{\rm raw}=
       \left[\frac{m_1}{9\sqrt X\,V}\right]^{1/2}>0 .
\tag{NHR.15}
$$

At every $0<\kappa\le\kappa_{\rm raw}$ the COMPLETE raw-color
history has exact gap $\gamma/(2t_*)$.
For the existing $\sigma_x=0$ branch this holds at EVERY finite
original positive phase, by the exact first-mode coefficient

$$
E[x_a\Re(a c_a)]
 =-\frac{m_1}{3}e^{-\kappa^2V/2}\ne0 .
\tag{NHR.16}
$$

An actually permitted zero-phase raw vector also has this gap;
it is odd, whereas the ray projector is even at that endpoint.
:::

:::{prf:proof}
At phase zero the coefficient is $-m_1/3$ by angular integration.
Its change at positive phase is bounded, using the original
$|1-\cos z|\le z^2/2$, by
$\kappa^2 E[R v_a^2]/2\le
3\kappa^2\sqrt X\,V/2$.
Cauchy--Schwarz uses $ER^2=3X$ and $Ev_a^4=3V^2$.
The stated interval leaves a coefficient of magnitude at least
$m_1/6$. Independence at $s=0$ gives (NHR.16) exactly.
The degree-one Hermite projection and actual reducing-sector
argument prove the claimed gap. No raw vector's parity is
replaced by projector parity.
:::

(sec-nhr-physical-scope)=
## 5. Exact primitive full-state reflection tests

:::{prf:theorem} Complete harmonic recording-stride positivity classification
:label: thm-nhr-stride-classification

In the SAME existing harmonic/$\nu=0$/constant-fitness branch,
now retain any original $0<t^2\lambda<1$ and integer recording
stride $m\ge1$, without imposing (NHR.3).
Its full invariant covariance is the derived positive series
$C=\sum_{l\ge0}A^lQ(A^l)^\top$.
Put $\tau=(1+c)(1-2t^2\lambda)$.
The complete recorded-state transfer is self-adjoint if and only
if its $A^m$ is scalar. The scalar occurs exactly in the
complex-root regime $|\tau|<2\sqrt c$, where

$$
\theta=\arccos(\tau/(2\sqrt c))\in(0,\pi),\qquad
m\theta\in\pi\mathbb Z,\qquad
A^m=(-1)^{m\theta/\pi}c^{m/2}I .
\tag{NHR.17}
$$

It is a POSITIVE self-adjoint transfer if and only if
$m\theta\in2\pi\mathbb Z$.
Distinct real roots or a repeated root give no self-adjoint
complete transfer at any finite stride.
At an odd scalar resonance the full transfer has actual negative
first-Hermite eigenvalues and fails the positive-transfer test.
These are exact native full-state tests, including every original
$q>0,s\ge0$; failure of one is not automatically assigned to a
smaller color-only observable sector.
:::

:::{prf:proof}
The strict harmonic stability $0<t^2\lambda<1$, determinant
$c\in(0,1)$ and endpoint characteristic-polynomial tests give
spectral radius less than one, as already derived in
{prf:ref}`def-nct-lyapunov-budget`.
The same source controllability determinant above makes $C$
positive for every $s\ge0$.
The stationary Gaussian pair has cross covariance $A^mC$.
It is exchange invariant exactly when
$A^mC=C(A^m)^\top$; exchange invariance is equivalent to
self-adjointness of its actual observable kernel.
For necessity one may already test its original linear coordinates.

Let $S=AC-CA^\top$.
The stationary covariance equation gives
$S=ASA^\top+AQ-QA^\top$.
Every two-dimensional skew matrix satisfies $ASA^\top=cS$,
and the actual source vectors in (NHR.4) give

$$
S_{12}=\frac{2tE q^2-ks^2}{1-c}>0,\qquad
k=-t\lambda(1+c)E<0 .
\tag{NHR.18}
$$

Thus the original one-update kernel is never complete-state
reversible on this strict branch; extra terminal position noise
does not cancel its antisymmetric covariance.
By Cayley--Hamilton write $A^m=u_mA+v_mI$.
Then $A^mC-C(A^m)^\top=u_mS$, so the full recorded transfer is
reversible exactly when $u_m=0$.
For distinct roots $\lambda_\pm$,
$u_m=(\lambda_+^m-\lambda_-^m)/(\lambda_+-\lambda_-)$.
If they are real their positive product is $c$ and their signs
agree; distinct same-sign magnitudes cannot have equal $m$th
powers. For a repeated root $\lambda_0=\pm\sqrt c$,
$u_m=m\lambda_0^{m-1}\ne0$; $A$ is not scalar since $b>0$.
For complex roots,
$u_m=c^{(m-1)/2}\sin(m\theta)/\sin\theta$.
Its zero condition is exactly (NHR.17), and the scalar is
the displayed signed value.

At an even scalar resonance the actual Gaussian transfer is
the positive Mehler operator already derived.
At an odd resonance its first Hermite eigenvalue is the
negative scalar $-c^{m/2}$, although the stochastic kernel is
reversible. Its adjacent reflection quadratic form on that
native linear coordinate is negative.
Bounded truncations of that square-integrable coordinate
retain a negative form by $L^2$ continuity.
No other reversible case exists, proving the classification.
The argument tests the actual full state; it does not claim
that an untested color quotient contains this linear coordinate.
:::

(sec-nhr-positive-family)=
## 6. Evaluated full-population family and exact scope

:::{prf:corollary} Existing nontrivial stationary color family at every population
:label: cor-nhr-positive-family

Choose any $N\ge3$, $h=b_O=1$, $m=3,j=1$ and

$$
c=(3-\sqrt5)/2,\quad \gamma=-\log c,\quad
\lambda=4/(1+c),\quad \sigma_x=0,\quad \nu=0 .
\tag{NHR.19}
$$

Keep all other original parameters of the register, especially
the actual recorded-field source (NHR.1), positive fixed phase
scales and any finite force threshold. The FULL stationary
physical state and every configured complete B1 color array at
stride three have positive reconstruction. Both complete raw
and ray-color sectors have exact frequency gap $\gamma/(2t_*)$
for every allowed fixed phase; vacuum uniqueness and the gap
are uniform over every finite $N$.
Positions now have their DERIVED confining stationary Gaussian
law; no unbounded source is cut off.
:::

:::{prf:proof}
Here $t=1/2$, $p=0$, $\operatorname{tr}A=c-1=-\sqrt c$ and
$A^3=c^{3/2}I$, so (NHR.3) holds at $\theta=2\pi/3$.
The covariance (NHR.14) is positive and finite. Every finite
threshold has $p_A,m_1>0$, and (NHR.14)/(NHR.16) put degree one
in the two complete color sectors at every positive phase.
All intermediate updates and complete masks are those already
derived above. Apply their exact same-law reconstruction.
:::

:::{prf:remark} Primitive tests, original calibrations and remaining field endpoint
:label: rem-nhr-scope

The positive full-state transfer consumes the actual resonance
$A^m=rI$ with $r>0$, the original independent Gaussian law,
constant original fitness, confining harmonic force, cap `None`,
all-alive unbounded boundary and $\nu=0$.
When any test fails, its actual kernel is not assigned (NHR.6).
In particular positive viscosity with finite Gaussian bandwidth
has position-dependent kicks, and the existing validation requires
FINITE $\rho>0$. An infinite-bandwidth complete-graph force is
not an included substitute. Active accepted-clone or cap branches
likewise retain their actual transitions.

For $s>0$, the complete transfer and reflection reconstruction
remain proved with the exact covariance (NHR.5); the displayed
primitive small-phase intervals give exact raw/ray gaps.
Outside those intervals the native color gap remains at least
$\gamma/(2t_*)$ and its exact spectral multiplicities are the
ranks of Hermite projections of the ORIGINAL words (NHR.9),
with every covariance/calibration/threshold parameter retained.
This known-Gaussian characterization imposes no unknown-law
hypothesis. The $s=0$ all-phase formulas give exact equality
throughout the original configured phase range.

Fixed phase scales are consumed exactly as configured.
A history/warm-up scale keeps its actual availability and
capability errors; a current-output random scale keeps its
same-record correlations. Neither is made fixed inside this proof.
The original finite arithmetic law retains rounding, overflow,
source-stream and error-policy comparisons; exact Gaussian
Mehler reflection is not asserted for a finite-seed execution.

This establishes an arbitrary-population, fully stationary,
nontrivial ORIGINAL configured color-history physical transfer,
Hamiltonian, vacuum and uniform gap. Its recorded-field color
is the explicit included potential-force instrument, in its
executed component frame. The default viscous source at
$\nu=0$ has no nontrivial color and is not relabeled.
Local non-Abelian gauge invariance, continuum Yang--Mills
identification and spacelike commutators are not inferred from
this statistical/physical reconstruction. The actual interacting
positive-viscosity color-history question retains its separate
native tests.
:::
