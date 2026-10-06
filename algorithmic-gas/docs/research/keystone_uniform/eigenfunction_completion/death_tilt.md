# Exact native singleton death tilts and the failure of a local contraction candidate

(sec-kudt-retained)=
## 1. Retained state and the concrete next move

The source revision is 6107b67b9e85259581c1932565c1871e8a7e253a.
This note retains the unchanged native Rastrigin terminal-box gas,
with the entire original measurement, companion, fitness, revival,
accepted-component Haar, clone-jitter, two-viscous-kick, OU,
final-position Gaussian and cap record. It treats both configured
count and Gaussian row viscosity. Its task is the proposed local
step from draft 16's raw matched-source velocity comparison to a
singleton-event conditioned comparison. The global endpoint remains
the uniform bound on the actual full-state eigenfunction ratio.

The concrete input is an actual singleton: row 1 is the unique living
donor at the origin, every retained velocity is \(\theta e_1\),
and the other rows are dead. Take \(|\theta|<.1\). Each stored
dead position may, for example, be \(3e_2\), so its terminal flag
is consistent and the coordinate-1 reflection fixes the entering
dead-position features. Every revived row must copy the unique donor
and gets its original independent
own-row clone jitter. The donor itself does not copy. The accepted
component is the revival star; its shared Haar action has zero
centered velocity to rotate. Thus every collision row is exactly
\(\theta e_1\). This consequence uses the actual mandatory
revival, not a probability lower bound or a fitness-gap assumption.
Dead-position measurement features have not been discarded from
the general kernel; on this particular singleton input they cannot
change the unique donor or these mandatory gates.

Mandatory position revival does not donor-copy velocities. The
star collision uses all members' original slot velocities in its
conserved mean and centered restitution/Haar map. In particular
there is no authorized reduction of a singleton's complete kernel
to its living donor position and velocity alone. The special
constant-velocity family below is chosen precisely so that this
original full-star velocity operation can be evaluated exactly.

Use the primitive reference constants
\[
 t=.02,\quad c=e^{-.04},\quad b=t(1+c),\quad
 \eta=bt,\quad q^2=(1-e^{-.08})/2,\quad
 s=.02,\quad \sigma_J=.1,\quad V=2,\quad a=t\nu=.006,
\]
and
\[
 \sigma_h^2=t^2q^2+s^2,\qquad
 F_r(x)=-2x_r-20\pi\sin(2\pi x_r),\qquad
 \psi(x)=x+\eta F(x).
\]
The terminal box is \(D=[-2,2]^3\) and the central core is
\(C=[-.01,.01]^3\). No new absorbing or reflecting phase boundary
is introduced.

(sec-kudt-exact-star)=
## 2. Exact complete-law factorization on this original star

:::{prf:lemma} Position factorization despite the original viscosity
:label: lem-kudt-star-position-factorization

Let \(X_1=0\) and \(X_i=\sigma_J Z_i\) for \(i\ge2\),
with the original independent three-dimensional clone Gaussians.
Both original first-kick matrices preserve constant velocities.
Consequently, exactly,
\[
 u_i=\theta e_1+tF(X_i),\quad
 w_i=c\theta e_1+ctF(X_i)+q\xi_i,
\]
\[
 y_i=\psi(X_i)+b\theta e_1+tq\xi_i,\qquad
 x_i^+=\psi(X_i)+b\theta e_1+tq\xi_i+s\chi_i.
\tag{KUDT.1}
\]
The final positions are independent across rows and coordinates.
The original B2 graph and force are evaluated at \(y\) and set
\[
 z_i=W_yw_i+tF(y_i),\qquad v_i^+=\frac{Vz_i}{V+|z_i|}.
\tag{KUDT.2}
\]
They affect retained velocities but have no feedback into (KUDT.1).

Define the genuine terminal event
\[
 E_\theta=\{x_1^+\in C,\ x_i^+\notin D\text{ for all }i\ge2\}.
\tag{KUDT.3}
\]
Conditional on \(E_0\), the pre-B2 input tuples of different rows
are still independent, with the donor core tilt and copied-row
death tilt. The retained post-B2 velocities are not independent.
:::

:::{prf:proof}
The constant collision array is preserved by every row-stochastic
first viscosity matrix, including both configured normalizations.
Substituting the actual B1, OU and two drift formulas gives
(KUDT.1). The source jitters, OU innovations and final innovations
are independent before B2. Their terminal event factors by rows;
conditioning therefore preserves that row factorization at the
pre-B2 stage. B2's full graph dependence remains in (KUDT.2).
\(\square\)
:::

:::{prf:lemma} The exact conditional likelihood score
:label: lem-kudt-conditional-score

For a bounded observable \(f\) of the retained output, put
\[
 G=\frac b s\sum_{i=1}^N\chi_{i1},\qquad
 C_N=N(b/s)^2.
\]
Represent its output under parameter \(\theta\) using the same
\(X,\xi\), the shifted final innovation
\(\chi_i- (b/s)\theta e_1\), and the original velocity update.
This keeps all final positions and terminal flags fixed. Exactly,
\[
 \mathbb E[f(V_\theta)\mid E_\theta]
 =\frac{\mathbb E[
       f(V_\theta)e^{\theta G-\theta^2C_N/2}\mid E_0]}
       {\mathbb E[e^{\theta G-\theta^2C_N/2}\mid E_0]}.
\tag{KUDT.4}
\]
For differentiable velocity observables with bounded derivative,
\[
 \left.\frac d{d\theta}\mathbb E[f(V_\theta)\mid E_\theta]
 \right|_{\theta=0}
 =\mathbb E[\partial_\theta f(V_0)\mid E_0]
       +\operatorname{Cov}_{E_0}(f(V_0),G).
\tag{KUDT.5}
\]
The covariance term is absent from a raw common-noise tangent
estimate. It has not been removed by mandatory revival.
:::

:::{prf:proof}
The final innovation density ratio is
\(\phi(\chi_i-(b/s)\theta e_1)/\phi(\chi_i)
 =\exp[(b/s)\theta\chi_{i1}-(b/s)^2\theta^2/2]\).
Multiplying these original Gaussian density ratios proves
(KUDT.4). The final innovation has no velocity feedback.
Differentiation is legitimate for the stated observables since the
cap bounds velocities and the Gaussian exponential has every finite
moment under the positive-probability event. It gives (KUDT.5).
\(\square\)
:::

(sec-kudt-counter-candidate)=
## 3. An actual two-particle conditioned velocity map expands

:::{prf:theorem} Failure of a singleton-conditioned kinetic contraction
:label: thm-kudt-conditioned-velocity-expansion

For the exact input above with \(N=2\), let \(\mathcal P_\theta\)
be the law of the actual retained output conditional on \(E_\theta\).
For either original viscosity normalization,
\[
 \left.\frac d{d\theta}\mathbb E_{\mathcal P_\theta}v_{21}^+
 \right|_{\theta=0}>1.42>\sqrt2.
\tag{KUDT.6}
\]
Thus this conditional one-update map cannot have a contracting
Wasserstein coefficient in the normalized full-array Euclidean
velocity metric. Indeed, for sufficiently small \(\theta\ne0\),
\[
 W_2(\mathcal P_\theta,\mathcal P_0;
                    \|\cdot\|_{2,2})
 \ge\frac{|\mathbb E_{\mathcal P_\theta}v_{21}^+
                   -\mathbb E_{\mathcal P_0}v_{21}^+|}{\sqrt2}
 >(1.42/\sqrt2)|\theta|.
\tag{KUDT.7}
\]
The entering velocity distance is \(|\theta|\).
This refutes this local conditional contraction candidate. It does
not refute a multistep stopped-return comparison, a symmetry-quotient
comparison, phase communication, or a uniform eigenfunction bound.
:::

:::{prf:proof}
Both next \(y\) arrays are translated by \(b\theta e_1\), so
the original B2 pair distances and weights are identical under the
parameter shift. Its velocity difference is the constant
\(c\theta e_1\). Hence
\[
 \partial_\theta z_{i1}
 =c-2\eta-40\pi^2\eta\cos(2\pi y_{i1})> .649.
\tag{KUDT.8}
\]
The cap has
\[
 J_{11}(z)=\frac V{V+|z|}
       -\frac{Vz_1^2}{|z|(V+|z|)^2}\ge0,
\]
with the continuous value at zero. Thus the pathwise derivative
term of (KUDT.5) is nonnegative and can be omitted in a lower bound.
All reflection symmetries give zero mean to the score and to
\(v_{21}^+\) at \(\theta=0\).

First compute an auxiliary one-row cap value \(V^{(0)}\) using
\(z^{(0)}=w_2+tF(y_2)\). This is an analytic comparison integrand;
the configured viscosity remains present in the actual output.
Define
\[
 L_F=2+40\pi^2,\qquad \beta=t^2q^2/\sigma_h^2,
 \qquad u(x)=(2-\psi(x))/\sigma_h.
\]
Let \(X\sim N(0,\sigma_J^2)\) and write \(\varphi_J\) for
its density. The exact one-coordinate right-death probability is
\[
 T=\int_{\mathbb R}\varphi_J(x)\Phi(-u(x))\,dx.
\tag{KUDT.9}
\]
Its two-sided death probability is \(q_0=2T\), and its alive
probability is \(p_0=1-q_0\). Elementary Gaussian trigonometric
moments give
\[
 \begin{split}
 M_{w,0}&=c^2t^2\mathbb EF(X)^2+q^2<.442,\\
 M_y&=\mathbb E\psi(X)^2+t^2q^2<.005572,\\
 M_z&=(\sqrt{M_{w,0}}+tL_F\sqrt{M_y})^2<1.579.
 \end{split}
\tag{KUDT.10}
\]
Explicitly, with \(v=\sigma_J^2\),
\(A=1-2\eta\), \(K=20\pi\eta\),
\[
 S_1=2\pi v e^{-2\pi^2v},\qquad
 S_2=(1-e^{-8\pi^2v})/2,
\]
\[
 \mathbb EF(X)^2=4v+80\pi S_1+400\pi^2S_2,
 \qquad M_y=A^2v-2AKS_1+K^2S_2+t^2q^2.
\]
Since \(p_0>.99\), an alive-tilted auxiliary other-coordinate
second moment is below 2. For a fixed positive first uncapped
coordinate \(z\), Jensen in the other two squares therefore gives
an average cap coordinate at least
\(2z/(2+\sqrt{z^2+4})\). For negative \(z\), the cap coordinate
is at least \(\max\{z,-2\}\). Define the global 1-Lipschitz
lower function
\[
 L(z)=\begin{cases}
 2z/(2+\sqrt{z^2+4}),&z\ge0,\\
 \max\{z,-2\},&z<0.
 \end{cases}
\tag{KUDT.11}
\]

Gaussian integration over the final first-coordinate innovation
gives
\[
 \mathbb E[\chi_1 1_{\{y+s\chi_1\notin[-2,2]\}}\mid y]
 =\phi((2-y)/s)-\phi((-2-y)/s).
\]
After the other two coordinates are required alive, coordinate
reflection turns the two boundary terms into twice the right term.
Under that boundary term, the actual noisy B2 position conditional
on \(X=x\) is Gaussian with mean and variance
\[
 \bar y(x)=\psi(x)+\beta(2-\psi(x)),\qquad
 v_b=t^2q^2s^2/\sigma_h^2.
\]
Its auxiliary uncapped velocity is
\(z=w+tF(y)\), and
\(|\partial_yz|\le t^{-1}+tL_F\). Consequently its averaged
cap lower bound is
\[
 L(\bar z(x))-\varepsilon_y,\quad
 \bar z(x)=ctF(x)+[\bar y(x)-\psi(x)]/t+tF(\bar y(x)),\quad
 \varepsilon_y=(t^{-1}+tL_F)\sqrt{2v_b/\pi}<.178.
\]
Set the complete scalar integral
\[
 I=\int_{\mathbb R}\varphi_J(x)\frac{s}{\sigma_h}
       \phi(u(x))[L(\bar z(x))-\varepsilon_y]\,dx.
\tag{KUDT.12}
\]
The true box-death probability is at most \(3q_0=6T\).
Thus the exact score covariance for the auxiliary cap has the
rigorous lower bound
\[
 \frac b s\mathbb E[V_1^{(0)}\chi_{21}\mid E_0]
 \ge\frac b s\frac{p_0^2 I}{3T}>1.493.
\tag{KUDT.13}
\]
The integral is positive; the quantitative certificate below
establishes this strict estimate, including its negative regions
and unbounded Gaussian tails.

Now restore the original B2 viscosity. At \(N=2\), in row mode
its uncapped difference from the auxiliary row is
\(a(w_1-w_2)\). In count mode it is
\((a/2)K(y_1,y_2)(w_1-w_2)\). Hence in both cases
\[
 |v_2^+-V^{(0)}|\le a(|w_1|+|w_2|)
\tag{KUDT.14}
\]
by nonexpansiveness of the exact original radial cap. The
computed complete conditional moments are
\[
 \begin{split}
 \mathbb E[|w_2|^2\mid E_0]&<1.9,
 &\mathbb E[\chi_{21}^2\mid E_0]&<8.64,\\
 \mathbb E[|w_1|^2\mid E_0]&\le3q^2,
 &\mathbb E[\chi_{11}^2\mid E_0]&<.27.
 \end{split}
\tag{KUDT.15}
\]
The auxiliary row is independent of the donor tuple and its
mean donor score is zero. Cauchy--Schwarz therefore bounds the
entire original graph correction to the score covariance by
\[
 \frac b s a(\sqrt{1.9}+\sqrt{3q^2})
                  (\sqrt{8.64}+\sqrt{.27})<.071.
\tag{KUDT.16}
\]
Equations (KUDT.5), (KUDT.8), (KUDT.13)--(KUDT.16) give a
derivative larger than \(1.493-.071=1.422>1.42\).
The copied-slot projection has Lipschitz constant \(\sqrt2\)
in the normalized two-row metric. Its mean difference yields
(KUDT.7). \(\square\)
:::

(sec-kudt-numeric-certificate)=
## 4. Explicit finite certificate for the death-tilted scalar integrals

:::{prf:lemma} Integral and moment enclosures retaining every Gaussian tail
:label: lem-kudt-scalar-integral-certificate

The primitive integrals above satisfy
\[
 .8523<e^{190}T<.8743,\qquad
 2.0380<e^{190}I<2.2248.
\tag{KUDT.17}
\]
Using their unrounded outward interval values gives
\((b/s)(.99)^2I/(3T)>1.4934196848\).
The associated moment integrals give the bounds in (KUDT.15).
:::

:::{prf:proof}
All integrations are one dimensional. For the copied-coordinate
right-death tilt, define
\[
 J=\int\varphi_J(x)u(x)\phi(u(x))\,dx,
\]
\[
 W=\int\varphi_J(x)
 \left[(ctF(x))^2\Phi(-u)
 +2ctF(x)q\frac{tq}{\sigma_h}\phi(u)
 +q^2\{\Phi(-u)+\beta u\phi(u)\}\right]dx.
\tag{KUDT.18}
\]
The exact coordinate death moments are
\[
 M_{\chi,d}=1+(1-\beta)J/T,\qquad M_{w,d}=W/T.
\tag{KUDT.19}
\]
The first identity is Gaussian conditioning of \(\chi\) on the
combined terminal Gaussian; the second retains the first native
force and its covariance with the original OU innovation. The
exact box-death coordinate mixture gives
\[
 M_{\chi,\rm box}
 =\frac{(1-q_0)^2M_{\chi,d}+(2-q_0)}{3-3q_0+q_0^2}
 \le(M_{\chi,d}+2)/3,
\]
because \(M_{\chi,d}>1\). For the whole copied \(w\) vector,
\[
 M_{w,\rm box}
 =\frac{3[(1-q_0)^2M_{w,d}+(2-q_0)M_{w,0}]}
             {3-3q_0+q_0^2}<M_{w,d}+.883.
\tag{KUDT.20}
\]
Here \(2M_{w,0}=.8823455\ldots\) and
\(q_0<e^{-180}\), with more than enough slack for the displayed
finite replacement. The donor has no jitter and its combined
terminal Gaussian is conditioned to the core. Its exact Gaussian
conditional variances give
\[
 M_{\chi,\rm donor}
 \le\beta+(1-\beta)\frac{.01^2}{\sigma_h^2}<.27,
 \qquad M_{w,\rm donor}\le3q^2.
\tag{KUDT.21}
\]

Here is a finite outward enclosure procedure for all remaining
scalar integrals. Partition \([-3,3]\) into 12000 intervals of
width \(.0005\). For an interval \(A\) with midpoint \(m_A\),
combine the two Gaussian exponents before enclosing:
\[
 B(x)=\frac{e^{190}}{2\pi\sigma_J}
       \exp[-x^2/(2\sigma_J^2)-u(x)^2/2].
\]
If \(D_A\) encloses
\(-x/\sigma_J^2+(2-\psi(x))\psi'(x)/\sigma_h^2\), then
\[
 \log B(A)\subseteq\log B(m_A)
       +[-.00025\sup|D_A|,\ .00025\sup|D_A|].
\tag{KUDT.22}
\]
This mean-value enclosure keeps the actual cancellation between
the two Gaussian exponents; it does not separately maximize them.

Put \(R(u)=\Phi(-u)/\phi(u)\), which is decreasing on the
whole line by its exact integral
\(R(u)=\int_0^\infty e^{-uz-z^2/2}\,dz\). For \(u\ge3\),
integration by parts gives the explicit endpoint bounds
\[
 U(u)-945u^{-11}\le R(u)\le U(u),\qquad
 U(u)=u^{-1}-u^{-3}+3u^{-5}-15u^{-7}+105u^{-9}.
\tag{KUDT.23}
\]
For \(u\le-3\), use \(.5/\phi(u)\le R(u)\le1/\phi(u)\).
For \(|u|\le3\), use the integrated exponential Taylor sum
through \(n=60\) in the standard Gaussian CDF. Its omitted
alternating tail is less than \(10^{-40}\). Evaluate the two
endpoint bounds for \(R(A)\) using monotonicity.

For every cell, integrate the interval enclosures of
\[
 B R,\qquad (s/\sigma_h)B[L(\bar z)-\varepsilon_y],
\]
\[
 B[R+(1-\beta)u],\qquad
 B[(ctF)^2R+2ctFq(tq/\sigma_h)+q^2(R+\beta u)].
\tag{KUDT.24}
\]
These are, respectively, \(e^{190}T\), \(e^{190}I\),
the scaled chi-square numerator, and \(e^{190}W\).
Outward arithmetic at 35 decimal places gives
\[
 \begin{array}{c|c}
 \text{quantity}&\text{outward enclosure}\\\hline
 e^{190}T &[.8523076286960573,.8742092929408664]\\
 e^{190}I &[2.0380610852847438,2.2247697064696336]\\
 M_{\chi,d}&[22.75991089177154,23.91785536974670]\\
 M_{w,d}&[.96146888351502,1.01331405610032]
 \end{array}
\tag{KUDT.25}
\]
Their quotients imply
\(M_{\chi,\rm box}<8.639286\),
\(M_{w,\rm box}<1.896315\), and the claimed score bound.
The force, sine and exponential in these enclosures are elementary
functions of the stated primitive parameters; there is no fitted
noise variance or assumed moment.

The omitted \(|X|>3\) Gaussian tails are retained with absolute
scaled errors \(10^{-100}\), \(10^{-99}\), \(10^{-97}\),
\(10^{-96}\), respectively, for the four lines of (KUDT.24).
Indeed \(\Pr(|X|>3)\) has exponent at least 450,
\(|L-\varepsilon_y|<2.178\), \(|u\phi(u)|<1/4\), and
\(|F(x)|\le2|x|+20\pi\); Gaussian second-moment integration
of the last bound still has scaled exponent below \(-250\).
All four error allowances exceed these full-tail bounds. This
completes a finite certificate for the original unbounded law.
\(\square\)
:::

(sec-kudt-common-marked-mass)=
## 5. The original shared-innovation common marked mass also vanishes

:::{prf:theorem} Explicit population dependence of synchronous marked overlap
:label: thm-kudt-common-marked-overlap

Use the same original Gaussian tuples for the input velocities
\(\theta=+.1\) and \(\theta=-.1\). Let \(\delta=.1b<.01\)
and let \(B=\psi(\sigma_JZ)+\sigma_h\zeta\) be the zero-input
copied coordinate. Its density is positive everywhere. Define
\[
 q=\Pr((B_1+\delta,B_2,B_3)\notin D),\qquad
 q_\cap=\Pr((B_1\pm\delta,B_2,B_3)\notin D
                       \text{ for both signs}).
\]
Then
\[
 0<\rho=q_\cap/q<1,\qquad
 \frac{\Pr(E_{+.1}\cap E_{-.1})}{\Pr(E_{+.1})}
 \le\rho^{N-1}\le\exp[-(N-1)e^{-202}].
\tag{KUDT.26}
\]
This statement concerns the original synchronous innovation
coupling. It does not bound the overlap of optimally reparameterized
conditional velocity laws.
:::

:::{prf:proof}
The donor common core event is contained in the positive-input
donor core event. The copied-row pairs are independent by
(KUDT.1), so their overlap ratio multiplies exactly. Moreover
\[
 q-q_\cap
 =\Pr(2-\delta<B_1<2+\delta,\ |B_2|,|B_3|<2)>0.
\]
A completely explicit lower event is
\[
 X_1\in[1.869,1.871],\quad \zeta_1\in[4.80,4.84],
 \quad |X_r|\le.1,\ |\zeta_r|\le1\quad(r=2,3).
\]
On this event \(1.902<\psi(X_1)<1.904\),
\(.0203<\sigma_h<.0204\), and the output first coordinate
lies strictly in the required strip. The other coordinates lie
in the terminal box. Minimum Gaussian densities on these fixed
intervals, with \((2\pi)^{-1/2}>1/3\), give probability
larger than \(e^{-202}\). Since \(q\le1\),
\(1-\rho>e^{-202}\). This proves (KUDT.26).
\(\square\)
:::

The exact one-dimensional native integral also gives the diagnostic
values \(\log q\simeq-188.2128956\) and
\(\rho\simeq.6915439492\) for this fixed comparison. These
decimal diagnostics are not needed for its strict analytic bound.
The mechanism is Gaussian mark translation and branch reweighting;
it does not use equal or near-equal fitness as an obstruction.

(sec-kudt-symmetry-hazard-route)=
## 6. What the local failure does and does not say about the eigenfunction

The copied-velocity observable in (KUDT.6) is odd under the global
reflection of coordinate 1. The native kernel and central event
commute with that reflection. The unique normalized positive
eigenfunction has the proved global reflection symmetry. Therefore
its conditional readout satisfies the exact identity
\[
 \mathbb E[h_N(S_\theta^+)\mid E_\theta]
 =\mathbb E[h_N(S_{-\theta}^+)\mid E_{-\theta}].
\tag{KUDT.27}
\]
Thus the amplified odd mean velocity is not an eigenfunction
counterexample. For a fixed reflection-even retained readout \(f\),
the pure score tilt in (KUDT.4) has the exact quotient
\[
 \frac{\mathbb E_{E_0}[f\cosh(\theta G)]}
      {\mathbb E_{E_0}[\cosh(\theta G)]}.
\tag{KUDT.28}
\]
Its first-order score covariance is zero. The score moment itself
is primitive: row independence before B2 gives
\[
 \mathbb E_{E_0}e^{r\sum_i\chi_{i1}}
 =M_{\rm donor}(r)M_{\rm copy}(r)^{N-1},
\]
\[
 M_{\rm donor}(r)
 =e^{r^2/2}\frac{p_C(sr)}{p_C(0)},\qquad
 M_{\rm copy}(r)
 =e^{r^2/2}\frac{q_D(sr)}{q_D(0)}.
\tag{KUDT.29}
\]
Here \(p_C\) and \(q_D\) are the actual donor core and copied
box-death integrals from (KUDT.1). These formulas compute rather
than assume the conditioned score cost.

On the same input family, the actual killing readout is exactly
\[
 \kappa_N(\theta)=q_{\rm donor}(b\theta)
                      q_D(b\theta)^{N-1}.
\tag{KUDT.30}
\]
For \(|\theta|\le.1\), the full Gaussian convolution and native
bound \(|\psi(x)-(1-2\eta)x|\le20\pi\eta\) give, explicitly,
\[
 q_D(b\theta)<e^{-180},\qquad
 q_{\rm donor}(b\theta)<e^{-4700},\qquad
 \kappa_N(\theta)<e^{-4700-180(N-1)}.
\tag{KUDT.31}
\]
For example apply the one-coordinate Gaussian Chernoff upper tail
with copied variance \((1-2\eta)^2\sigma_J^2+\sigma_h^2\)
and eroded threshold \(2-20\pi\eta-.1b\), then sum six faces.
The donor has no clone jitter and uses the full \(\sigma_h\)
variance. This preserves its much smaller killing probability.

The exact next productive route is therefore a symmetry- or
phase-resolved hazard comparison, or a multistep posterior transport
and physical return-time estimate. It cannot require the failed
one-update conditioned velocity contraction as a necessary premise.
The rare-event score formulas and all conditional moment bounds
above supply actual coefficients for that next analysis. They do
not bound the complete discounted return kernel or close the
global eigenfunction ratio. No equilibrium basin has been closed,
no noise has been clipped, and no alternate force has been used.

(sec-kudt-uniform-hazard-sensitivity)=
## 7. A positive uniform killing-readout estimate on every singleton

:::{prf:theorem} Complete singleton hazard sensitivity for both normalizations
:label: thm-kudt-uniform-singleton-hazard-sensitivity

Keep the native Rastrigin record and the original star, including
every retained dead-slot velocity. Let its unique living donor
position be any \(\mu\in D\). Consider a line of actual inputs
\(\mu_\theta=\mu+\theta z\) and
\(v_\theta=v+\theta e\) on an interval where all donor positions
remain in \(D\) and every input row satisfies the original cap
envelope. Here \(e\) is an arbitrary full-slot velocity array,
not a constant mean mode. Define
\[
 L_\psi=1-2\eta+40\pi^2\eta,
 \qquad L_{\psi,2}=80\pi^3\eta,
\]
\[
 H_{2,\rm count}=1,\qquad
 H_{2,\rm row}=1-a+a\sqrt{7188},\qquad
 B=L_\psi|z|+bH_2\|e\|_{2,N}.
\tag{KUDT.32}
\]
Import the fully proved original-law extinction envelopes from
draft 17,
\(a_{E,\rm count}=2\,10^{-6}\) and
\(a_{E,\rm row}=7\,10^{-11}\).
At every point of this line the actual killing probability has
\[
 |\kappa_N'(\theta)|
 \le\frac{\sqrt N}{s}e^{-Na_E/2}B,
\tag{KUDT.33}
\]
\[
 |\kappa_N''(\theta)|
 \le e^{-Na_E/2}
 \left[\frac{\sqrt2 N}{s^2}B^2
       +\frac{\sqrt N}{s}L_{\psi,2}|z|^2\right].
\tag{KUDT.34}
\]
Consequently, uniformly over all population sizes, donor wells and
all retained velocity arrays in this singleton source class,
\[
 |\kappa_N'|\le\frac{B}{s\sqrt{a_E e}},\qquad
 |\kappa_N''|\le\frac{2\sqrt2}{a_E e s^2}B^2
             +\frac{L_{\psi,2}}{s\sqrt{a_E e}}|z|^2.
\tag{KUDT.35}
\]
These are estimates of the original physical killing readout.
They do not require a contracting conditioned velocity kernel,
a right-eigenfunction bound, or attraction between different wells.
:::

:::{prf:proof}
At a singleton every companion used for revival is the unique
living donor. The original component is the full revival star.
Freeze all original measurement inputs and its original Haar
rotation. Its linear collision map gives
\[
 e^{\rm col}_i=\bar e+
      \alpha_{\rm col}O(e_i-\bar e),\qquad
 \|e^{\rm col}\|_{2,N}^2
 =|\bar e|^2+\alpha_{\rm col}^2
                   \|e-\bar e\|_{2,N}^2
 \le\|e\|_{2,N}^2.
\]
Thus every retained velocity, including a dead row's velocity,
is present in the computed collision discrepancy.

Couple the original own clone jitters. All prepared positions
translate by the same \(\theta z\), including the persistent
unjittered donor. Their pair distances, and hence both original
first-kick matrices, remain exactly identical. With shared OU
innovations the actual terminal pre-final-noise means are
\[
 m_i(\theta)=\psi(X_i+\theta z)
           +b(W_Xv^{\rm col}_\theta)_i+tq\xi_i.
\]
The native global derivative bounds are
\(\|D\psi\|\le L_\psi\) and
\(|D^2\psi[z,z]|\le L_{\psi,2}|z|^2\).
The proved count contraction and global Gaussian incoming-column
envelope give
\(\|W_Xe^{\rm col}\|_{2,N}\le H_2\|e\|_{2,N}\), even
though \(X\) is an unbounded jittered array. Therefore
\[
 \|m'(\theta)\|_{2,N}\le B,
 \qquad \|m''(\theta)\|_{2,N}\le L_{\psi,2}|z|^2.
\]

Condition on the complete preparation and OU tuple. Translating
the independent final Gaussians to keep all terminal positions
fixed gives the exact Gaussian score and second score
\[
 G_\theta=\sum_i\chi_i\cdot m_i'/s,
\quad C_\theta=\sum_i|m_i'|^2/s^2,
\quad A_\theta=\sum_i\chi_i\cdot m_i''/s,
\]
\[
 \kappa_N'=\mathbb E[1_\dagger G_\theta],\qquad
 \kappa_N''=\mathbb E[1_\dagger
             (G_\theta^2-C_\theta+A_\theta)].
\]
Before restricting to extinction, the independent final Gaussian
identities are
\(\mathbb EG_\theta^2=C_\theta\),
\(\mathbb E(G_\theta^2-C_\theta)^2=2C_\theta^2\), and
\(\mathbb EA_\theta^2=\sum_i|m_i''|^2/s^2\).
Conditional and then unconditional Cauchy--Schwarz, with the
proved actual \(\kappa_N\le e^{-Na_E}\), yields
(KUDT.33)--(KUDT.34). Every source/noise correlation is retained
by the conditioning and the deterministic averaged norm bounds.

Finally
\(\sup_{x>0}\sqrt x e^{-a_Ex/2}=1/\sqrt{a_E e}\) and
\(\sup_{x>0}x e^{-a_Ex/2}=2/(a_E e)\).
They prove (KUDT.35) without replacing the population by a
fitted effective sample size. The second kick and cap still
determine the complete velocity output; by the actual update
order they have no feedback into the terminal killing event.
\(\square\)
:::

The computed uniform first-derivative coefficients in front of
\((|z|,\|e\|_{2,N})\) are below
\((28051,841)\) in count mode and
\((4741369,213602)\) in row mode. The pure-velocity second
coefficient is below \(2000242\) and \(129048985180\),
respectively. These large numbers come from the proved conservative
global extinction floors. The stronger population decay in
(KUDT.33)--(KUDT.34) remains available before maximizing over \(N\).
No single-well or near-equal-fitness regime is removed.

:::{prf:corollary} Sharper even killing readout on the central mean family
:label: cor-kudt-central-even-killing-sensitivity

For the explicit origin-centered family of Section 2 and
\(|\theta|\le.1\),
\[
 |\kappa_N'|\le2N e^{-2000-90(N-1)},\qquad
 |\kappa_N''|\le8N^2e^{-2000-90(N-1)}.
\tag{KUDT.36}
\]
Reflection gives \(\kappa_N'(0)=0\), so
\[
 |\kappa_N(\theta)-\kappa_N(0)|
 \le4N^2e^{-2000-90(N-1)}\theta^2
 \le4e^{-2000}\theta^2.
\tag{KUDT.37}
\]
:::

:::{prf:proof}
Write \(d=q_{\rm donor}(b\theta)\),
\(q_c=q_D(b\theta)\), and \(n=N-1\). The exact killing
probability is \(dq_c^n\). For either one-row probability
\(q(s)\), translation of the full combined terminal Gaussian
gives
\[
 |q'(s)|\le\sigma_h^{-1}\sqrt{q(s)},\qquad
 |q''(s)|\le\sqrt2\,\sigma_h^{-2}\sqrt{q(s)},
\]
by its first two Hermite moments. Combine them with the actual
\(d<e^{-4700}\), \(q_c<e^{-180}\) from (KUDT.31), and
\(b/\sigma_h<2\). The product's second derivative has the four
terms
\[
 b^2[d''q_c^n+2n d'q_c^{n-1}q_c'
       +n dq_c^{n-1}q_c''
       +n(n-1)dq_c^{n-2}(q_c')^2].
\]
Terms with a zero coefficient are absent, including the apparent
negative powers at \(N=1,2\). Each surviving exponential is
bounded by \(e^{-2000-90n}\); the remaining coefficient sum is
at most \(2(n+1)^2\). This proves the second estimate in
(KUDT.36), and the first is the corresponding two-term product
derivative. Since \(N^2e^{-90(N-1)}\le1\) for all \(N\ge1\),
Taylor's integral remainder proves (KUDT.37). \(\square\)
:::

The positive hazard estimates are a completed step toward an
eigenfunction comparison that reads the actual killing potential
along coupled or phase-resolved paths. A global proof still needs
the corresponding physical path or stopped-return transport, with
its actual information and waiting-time costs. The derivative of
a killed readout and the spectrum of a full retained-state kernel
are distinct objects; no global eigenfunction closure is asserted
from (KUDT.35) or (KUDT.37).
