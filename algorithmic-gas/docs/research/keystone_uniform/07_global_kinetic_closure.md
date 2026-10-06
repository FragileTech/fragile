# A global cap metric and the exact coupled closure terms

(sec-kug-cap-metric)=
## 1. An exact common metric for the harmonic BAOAB and radial cap

The force in this note is the configured reference acceleration $F(x)=-x$.
Write $t=h/2$, $c=e^{-\gamma h}$, $b=t(1+c)$,
$a=1-tb$, $C=t(c+a)$, $d_v=c-tb$, and $\beta=b/a$.
The letters $C,d_v$ are kinetic coefficients, distinct from the Gaussian
column bound and spatial dimension. All position and OU innovations retain
their declared Gaussian laws. Every comparison uses equal corresponding
innovations; the original final position noise cancels in a physical
coordinate difference but remains in the terminal status calculation.

:::{prf:definition} Harmonic cap comparison metric
:label: def-kug-cap-metric

For $0<a$ and $0<\beta<1$, define

$$
P=\begin{pmatrix}1&\beta\\\beta&1\end{pmatrix},\qquad
\mathcal Q_N(r,z)=\|r\|_{2,N}^2+2\beta\langle r,z\rangle_N+
\|z\|_{2,N}^2.
$$

Its eigenvalues are $1-\beta,1+\beta$. The zero-viscosity affine
part of the **actual viscous comparison** has the matrix

$$
M=\begin{pmatrix}a&b\\-C&d_v\end{pmatrix}.
                                                               \tag{KUG.1}
$$

This decomposition will retain every actual viscous contribution in
Section 3; it does not change the configured positive viscosity.
:::

:::{prf:theorem} Global harmonic affine-part contraction with the actual cap
:label: thm-kug-global-harmonic-cap

Compute the two primitive residual matrices

$$
\begin{aligned}
D_0&=\begin{pmatrix}0&0\\0&a^2-b^2\end{pmatrix},\\
D_1&=\begin{pmatrix}
C(2b-C)&d_v(C-b)+\beta bC\\
d_v(C-b)+\beta bC&a^2-b^2-d_v^2-2\beta bd_v
\end{pmatrix}.
\end{aligned}                                                   \tag{KUG.2}
$$

If $0<a<1$, $0<\beta<1$, $D_0\succeq0$ and $D_1\succeq0$,
then for every population size, every pair of unbounded prepared positions,
every prepared velocity array, and every realization of shared noise,

$$
\mathcal Q_N(R,DZ)\le a^2\mathcal Q_N(r,z),
\qquad (R,Z)=M(r,z),                                          \tag{KUG.3}
$$

where $D$ is **any** symmetric operator with $0\preceq D\preceq I$.
In particular, it holds for the actual cap secant operator
$C_V(u)-C_V(\widetilde u)=D(u-\widetilde u)$, for every $V>0$.
The equality of all corresponding innovations gives exactly that
deterministic affine difference before the actual cap.

For the unchanged $h=.04$, $\gamma=1$ reference, all tests pass:

$$
\begin{gathered}
\beta=0.039246570487420335,\quad
a^2=0.998431983599914,\quad
a^2-b^2=0.9968941055100375,\\
D_{1,11}=0.0015378778438160184,\quad
D_{1,12}=0.00004527335501946116,\quad
D_{1,22}=0.07232920921007878,\\
\det D_1=0.00011123143862823892,
\quad\lambda_{\min}(D_1)=0.0015378488900472942.
\end{gathered}                                                   \tag{KUG.4}
$$

These decimal evaluations are diagnostics of the analytic matrices in
(KUG.2). Their signs follow, for example, by inserting the alternating
Taylor bounds $\sum_{k=0}^{9}(-.04)^k/k!<e^{-.04}<
\sum_{k=0}^{8}(-.04)^k/k!$ into those rational formulas.

*Proof.* The radial cap has a symmetric Jacobian with radial eigenvalue
$V^2/(V+|u|)^2$ and tangential eigenvalue $V/(V+|u|)$, both in
$[0,1]$. Its integral along the segment between the two pre-cap velocities
therefore gives a symmetric secant operator $0\preceq D\preceq I$.
It can depend on the complete noise realization; the argument is pointwise.

Diagonalize $D$ orthogonally in the full $Nd$-dimensional velocity space.
All blocks of $M$ and $P$ are scalar multiples of the spatial identity,
so the same orthogonal change of basis applies to positions and velocities.
For each eigenvalue $q\in[0,1]$, put
$M_q=\operatorname{diag}(1,q)M$. Its quadratic matrix is convex in $q$:

$$
M_q^\top PM_q\preceq(1-q)M_0^\top PM_0+qM_1^\top PM_1,
$$

because the difference of the right and left sides is
$q(1-q)(-C,d_v)^\top(-C,d_v)\succeq0$.
Direct multiplication gives
$a^2P-M_0^\top PM_0=D_0$ and
$a^2P-M_1^\top PM_1=D_1$, using $a^2\beta=ab$.
The two endpoint tests imply the result for every eigenvalue, hence
for the original full operator without a dimension-dependent constant.
$\square$
:::

(sec-kug-viscous-background)=
## 2. Positive viscosity, both graph stages, and noncommuting cap operators

:::{prf:theorem} A contracting count-viscous background at the reference coupling
:label: thm-kug-count-background

Retain $h=.04$, $\gamma=1$, $\nu=.3$ and $\rho=1$.
Let $W_0,W_2$ be any two actual count-normalized kick matrices
$I-t\nu L_{x}$ and $I-t\nu L_y$, on any finite position arrays.
Their Gaussian graphs need not commute with one another or with the cap
secant operator. Set $\kappa=t\nu=.006$,
$S=I-W_0$, $T=I-W_2$, so $\|S\|,\|T\|\le\kappa$.
Define the retained two-graph background difference by

$$
\begin{aligned}
R_B&=a r+bW_0z,\\
Z_B&=-t(cW_2+aI)r+(cW_2-tbI)W_0z.
\end{aligned}                                                   \tag{KUG.5}
$$

For every symmetric $0\preceq D\preceq I$, every $N$, and all
physical arrays,

$$
\mathcal Q_N(R_B,DZ_B)\le q_B^2\mathcal Q_N(r,z),
\quad q_B^2<0.998492<1.                                       \tag{KUG.6}
$$

This is a positive-viscosity component of the actual comparison.
Section 3 evaluates the changed-position graph defects; (KUG.6) does
not assume that two different swarms have the same graphs.

More generally this background result holds for **either** normalization
whenever its two actual matrices satisfy
$\|I-W_j\|\le\kappa$ and the explicit tests below pass.
The general proof does not assume either matrix is symmetric.
The primitive count bound is $\kappa=t\nu$. The dimension-only
primitive row bound is
$\kappa=t\nu(1+\sqrt{\overline C_d})$; at the reference row
configuration this coarse bound does not pass the following test.
It must not be presented as a failed test of the actual row dynamics.

*Proof.* First retain a directional dissipation margin in (KUG.3).
The computed endpoint matrices obey

$$
D_0\succeq\operatorname{diag}(0,\mu),\qquad
D_1\succeq\operatorname{diag}(0,\mu),\qquad\mu=7/100.
$$

For $D_1$, this is the explicit two-by-two test
$D_{1,11}>0$ and
$D_{1,11}(D_{1,22}-.07)>D_{1,12}^2$.
The rational Taylor interval for $c$ used after (KUG.4) gives
$3.57998956\,10^{-6}<D_{1,11}(D_{1,22}-.07)-D_{1,12}^2
<3.57998957\,10^{-6}$.
The same endpoint convexity proof now gives, for arbitrary $D$,

$$
\mathcal Q_N(R_0,DZ_0)\le
a^2\mathcal Q_N(r,z)-\mu\|z\|_{2,N}^2,
\quad R_0=ar+bz,\quad Z_0=-Cr+d_vz.                            \tag{KUG.7}
$$

Subtract the backgrounds in (KUG.5):

$$
e_R=R_B-R_0=-bSz,\qquad
e_Z=Z_B-Z_0=ctTr-(d_vS+cT-cTS)z.
$$

Set

$$
A_R=b\kappa,\quad B_Z=ct\kappa,\quad
H_Z=(|d_v|+c)\kappa+c\kappa^2.
$$

Then $\|e_R\|\le A_R\|z\|$,
$\|e_Z\|\le B_Z\|r\|+H_Z\|z\|$. Exact quadratic expansion,
followed only by $\|D\|\le1$, bounds the background perturbation by
$E_r\|r\|^2+E_{rz}\|r\|\|z\|+E_v\|z\|^2$, with the
following fully computed coefficients:

$$
\begin{aligned}
E_r={}&2\beta aB_Z+2CB_Z+B_Z^2,\\
E_{rz}={}&2aA_R+2\beta(aH_Z+bB_Z)+2\beta A_RC
+2\beta A_RB_Z+2(CH_Z+d_vB_Z)+2B_ZH_Z,\\
E_v={}&2bA_R+A_R^2+2\beta bH_Z+2\beta A_Rd_v
+2\beta A_RH_Z+2d_vH_Z+H_Z^2.
\end{aligned}                                                   \tag{KUG.8}
$$

At the reference all displayed $a,b,C,d_v,\beta$ are positive.
For a different parameter record their absolute values should be used
where a norm bound appears. No graph/cap commutation was used in this
expansion. For any $\theta>0$ satisfying $E_v+\theta\le\mu$,
Young's inequality and (KUG.7) give

$$
\mathcal Q_N(R_B,DZ_B)\le
\left[a^2+
\frac{E_r+E_{rz}^2/(4\theta)}{1-\beta}\right]
\mathcal Q_N(r,z).                                            \tag{KUG.9}
$$

The reference constants are

$$
\begin{gathered}
A_R=0.00023529473269827877,
\quad B_Z=0.0001152947326982788,
\quad H_Z=0.011559355794983397,\\
E_r=0.00001809517131745374,
\quad E_{rz}=0.002508208296195968,
\quad E_v=0.022399735692950486.
\end{gathered}
$$

Choose $\theta=.04$. Its remaining velocity margin is
$\mu-E_v-\theta=0.00760026430704952>0$; (KUG.9) gives
$q_B^2=0.998491743575690<1$. The same rational Taylor interval
certifies $q_B^2<.998491743576<.998492$ and $E_v+.04<.07$.
Gaussian positions may be arbitrarily
large: the count Laplacian bound $0\preceq L\preceq I$ supplies
the same $\kappa$ for every array. For row normalization,
$\|P_x\|\le\sqrt{\overline C_d}$ gives the stated primitive
$\kappa$, and the identical algebra proves the conditional test.
$\square$
:::

(sec-kug-actual-defects)=
## 3. Exact changed-position graph defects and the complete preparation

:::{prf:proposition} Actual two-graph closure identity
:label: prop-kug-actual-closure

Use the actual paired Keystone preparations
$(x,v),(\widetilde x,\widetilde v)$, their two measured fitness vectors,
their source/copying tokens, the shared row jitters and the matched
component-Haar coupling. Put $r=x-\widetilde x$, $z=v-\widetilde v$.
Let $W_0=W_x$, $W_2=W_y$ be the first swarm's actual first and
second matrices and define the **actual** two feedback defects

$$
e_0=(W_x-W_{\widetilde x})\widetilde v,\qquad
e_1=(W_y-W_{\widetilde y})\widetilde w,
                                                               \tag{KUG.10}
$$

where $w,\widetilde w$ are the unbounded post-OU velocities and
$y,\widetilde y$ are the actual A2 positions. Then

$$
R=R_B+b e_0,\qquad
Z=Z_B+(cW_2-tbI)e_0+e_1.                                     \tag{KUG.11}
$$

The full original cap gives $\Delta C=DZ$ with its actual random
secant $0\preceq D\preceq I$. Put
$E_R=be_0$, $E_Z=(cW_2-tbI)e_0+e_1$.
The exact signed correction to the contracting background is

$$
\begin{aligned}
\mathcal I_E={}&2\langle R_B,E_R\rangle_N+\|E_R\|_{2,N}^2\\
&+2\beta[\langle R_B,DE_Z\rangle_N+
\langle E_R,DZ_B\rangle_N+\langle E_R,DE_Z\rangle_N]\\
&+2\langle DZ_B,DE_Z\rangle_N+\|DE_Z\|_{2,N}^2.
\end{aligned}                                                   \tag{KUG.12}
$$

Every term is evaluated against the same actual preparation and full
innovation law. In count mode the primitive finite estimates are

$$
\|e_0\|_{2,N}\le2\sqrt2\,t\nu V_c\ell_\rho\|r\|_{2,N},
\quad
(\mathbb E[\|e_1\|_{2,N}^2\mid\mathrm{prep}])^{1/2}
\le4t\nu\ell_\rho\|R\|_{4,N}\widetilde{\mathcal W}_4,
                                                               \tag{KUG.13}
$$

with the exact conditional noncentral Gaussian fourth moment
$\widetilde{\mathcal W}_4$ of (KUK.23). In row mode, the polynomial
normalization derivative gives

$$
\begin{aligned}
\|e_0\|_{2,N}
&\le\frac{t\nu D_dV_c}{2\rho^2}
(\|x\|_{6,N}+\|\widetilde x\|_{6,N})\|r\|_{6,N},\\
(\mathbb E[\|e_1\|_{2,N}^2\mid\mathrm{prep}])^{1/2}
&\le\frac{t\nu D_d}{2\rho^2}\|R\|_{6,N}
(\mathcal Y_6+\widetilde{\mathcal Y}_6)
\widetilde{\mathcal W}_6.
\end{aligned}                                                   \tag{KUG.14}
$$

These are computed $N$-independent moment coefficients; they do not
give either signed defect an artificial restoring sign.

For the count reference, let $\mathcal P_N$ be the exact physical
preparation increment in (KU.28a), with $\alpha=\gamma_P=1$ and
the $\beta$ of (KUG.1), setting its mask coefficient to zero.
Let $m_0,m_+$ be the averaged input and
terminal-mask mismatch probabilities under the same complete coupling.
Then the full marked comparison satisfies

$$
\mathbb E\Delta\mathcal Q_{\rm marked}
\le-(1-q_B^2)\mathcal Q_N(S,\widetilde S)
+q_B^2\mathbb E\mathcal P_N+
\mathbb E\mathcal I_E+\lambda_a(m_+-m_0).                     \tag{KUG.15}
$$

The physical input form is denoted $\mathcal Q_N(S,\widetilde S)$.
Mandatory revival removes the input mask discrepancy before kinetics;
the terminal contribution restores its actual output mismatch. The
cloning term retains incoming donor load, signed barycenter terms,
source mismatch, sampled global normalization, copying indicators and
component-Haar moments. A passive geometry readout contributes only its
measurable pushforward; it cannot alter the configured kernel.

*Proof.* Subtract B1 in the form
$\Delta u=W_0z-tr+e_0$.
Shared OU makes its velocity difference $c\Delta u$, and the two
drifts give $R=r+b\Delta u=R_B+be_0$.
Subtract B2 using its actual $W_y$ and $W_{\widetilde y}$:
$Z=cW_2\Delta u-tR+e_1$, which is (KUG.11).
Apply the original cap secant and expand the quadratic around
$(R_B,DZ_B)$ to obtain (KUG.12).

For count mode, interpolate the positions and use the derivative
estimate $\|DL_x[r]\widetilde v\|_{2,N}
\le2\sqrt2 V_c\ell_\rho\|r\|_{2,N}$, proved by pairing the
Gaussian derivatives and using
$N^{-2}\sum_{i,j}|r_i-r_j|^2\le2\|r\|_{2,N}^2$.
This gives the first bound in (KUG.13); its second is the complete
mixed-fourth-moment calculation of (KUK.24).
The normalized polynomial derivative in (KUP.1), followed by
conditional sixth-moment Hölder, gives (KUG.14).
No bound on a maximum Gaussian over the $N$ rows is used.
Finally the exact preparation law gives
$\mathbb E\mathcal Q_N(r,z)=\mathcal Q_N(S,\widetilde S)+
\mathbb E\mathcal P_N$. Insert (KUG.6), add (KUG.12), and add
the actual input-to-terminal mask increment to obtain (KUG.15).
$\square$
:::

(sec-kug-marked-metric)=
## 4. The conditional terminal channel has a first-order position modulus

:::{prf:proposition} A quadratic physical cost cannot alone absorb terminal masks
:label: prop-kug-marked-first-order

The Gaussian terminal classification is not globally quadratically
Lipschitz in its conditional mean. For a scalar boundary interval
$[-L,L]$ and noise standard deviation $s>0$, write

$$
\theta(m)=\Phi((L-m)/s)-\Phi((-L-m)/s).
$$

At a mean with $\theta'(m)\ne0$, any coupling of the two completed
outputs has

$$
\Pr(a^+\ne\widetilde a^+)
\ge|\theta(m+\varepsilon)-\theta(m)|
=|\theta'(m)|\,|\varepsilon|+o(|\varepsilon|).
                                                               \tag{KUG.16}
$$

Thus the conditional terminal classification channel, measured with a
positive constant mask penalty, cannot have a bound proportional only
to the squared difference of its means. A proof that separates this
channel from the preceding preparation must retain its first-order
modulus. This conclusion concerns the conditional channel; it does not
assert noncontraction of the preparation-averaged complete kernel,
failure of global mixing, or failure of a delayed survivor block.

*Proof.* The two terminal masks have Bernoulli parameters
$\theta(m+\varepsilon),\theta(m)$. The probability of disagreement
in any coupling is at least the difference of their success
probabilities. Differentiate the displayed Gaussian integral and use
its first-order Taylor expansion. A quadratic input cost is of order
$\varepsilon^2$, while the mandatory output mask penalty is of order
$|\varepsilon|$. The box case follows by multiplying the other
coordinate interval probabilities. $\square$
:::

(sec-kug-closure-scope)=
## 5. What the new calculation closes

:::{prf:remark} Global background versus global complete-kernel contraction
:label: rem-kug-global-closure

The common cap metric closes the global affine harmonic comparison
without an additive variance floor or a bounded-noise event. The count
background theorem also retains the configured $\nu=.3$, both actual
graph matrices, and arbitrary noncommuting cap operators with a strict
$N$-independent coefficient. These are improvements over the unsigned
scalar stability test in Chapter 21, whose $q_\nu>1$ is a failed
upper-bound certificate and is not a dynamical noncontraction theorem.

For the full actual algorithm, the graph-change terms are precisely
(KUG.10)--(KUG.14), and the complete source/mark terms are precisely
(KUG.15). The background contraction must not be reported as contraction
of the complete kernel. A global delayed-block proof must bound the
signed preparation and graph feedback over the actual preparation law,
retain the conditional terminal channel's first-order modulus, and carry survival conditioning through
the original killed kernel. None of these terms can be dropped from a
claim about a global eigenfunction ratio or modified LSI.

The new estimate does not yet prove those remaining aggregate signs or a
globally decaying full-array coupling at the configured reference. A
worst conditioned preparation array, a failed unsigned norm test, and a
near-equal-fitness example are none of them a proof that a delayed,
preparation-averaged closure is impossible. Both the completed global
component and the remaining exact quantities are explicit above.
:::
