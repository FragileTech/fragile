# Conditional Gaussian tensor for the actual signed count block

This note continues the exact marked-law response account of research55.
It keeps that independently audited source unchanged and isolates the
conditional signed calculation needed to improve its absolute feedback
bound. The conclusion below is a complete physical differential identity;
its signed absorption into a default delayed nonlinear gap is a separate
problem.

:::{div} feynman-prose

The same Gaussian kick changes both a particle's position and its velocity.
An alignment force therefore sees a correlated pair of random quantities.
We can integrate that pair exactly: multiplying its Gaussian density by the
Gaussian interaction kernel produces another Gaussian, whose mean and
covariance we know. This exposes the signed response without replacing the
noisy interaction graph by an independent graph. A favorable term in this
formula becomes a contraction estimate only after the remaining terms are
controlled in the same account.

:::

:::{prf:definition} Actual harmonic count-stage register
:label: def-dsg-register

Use the current-frame harmonic arm of {prf:ref}`def-dmc-register`,
with $d=3$, $h=.04$, $F(x)=-x$, count-normalized viscosity $\nu=.3$,
interaction radius $\rho=1$, native stored-velocity cap $V=2$,
and source-preparation speed bound $V_c=4$. Put
$$
t=h/2=.02,\qquad c=e^{-.04},\qquad
b=t(1+c),\qquad a_x=1-tb,\qquad
q^2=(1-c^2)/2,\qquad a=t\nu=.006.
$$
Here $a_x+tb=1$ and $a_x>0$. The physical quadratic form is
$$
Q_\beta(x,p)=|x|^2+|p|^2+2\beta\langle x,p\rangle,
\qquad\beta=.04.
$$
For a finite prepared array, all inner products and norms are normalized
by $N$ and every count-field pair sum has the actual denominator $N$.
For a prepared population law they are probability integrals. The actual
first count output is $U=P-aL_1P$. Since $0\le a\le1$ and
$0\le K\le1$, each row is a convex combination of prepared velocities,
so $|U|\le V_c$. The uncapped OU stage and next landing are
$$
w=c(U-tX)+q\xi,\qquad y=a_xX+bU+tq\xi,
\qquad\xi\sim N(0,I_d).
$$
The second graph is built from this same $y$ and acts on this same $w$;
no independent velocity or graph replaces them. The differential result
below concerns the physical kinetic output before terminal marking. It
allows the complete actual source, copying, revival and component law
through its prepared input; it asserts no contraction of that preparation.
:::

(sec-dsg-signed-gaussian)=
## 1. Exact Gaussian integration of the signed second-kick account

:::{prf:lemma} Conditional Gaussian alignment-response tensor
:label: lem-dsg-gaussian-response-tensor

Keep the default $\rho=1$ count kernel $K(y)=e^{-|y|^2/2}$.
For any fixed prepared pair put
$$
D=X-X',\qquad H=U-U',\qquad
Y=a_xD+bH+tq\Xi,\qquad W=c(H-tD)+q\Xi,
\quad \Xi\sim N(0,2I_d).
$$
Let
$$
\sigma_G^2=2t^2q^2,\quad \Gamma=1+\sigma_G^2,\quad
\mu_G=a_xD+bH,\quad \bar w=c(H-tD),
$$
$$
\kappa_G=\Gamma^{-d/2}
       e^{-|\mu_G|^2/(2\Gamma)},\qquad
\mathcal M_G=\mathbb E[K(Y)WY^{\mathsf T}].
$$
The complete Gaussian pair has the exact conditional moments
$$
\mathbb E K(Y)=\kappa_G,
$$
$$
\mathcal M_G=\kappa_G\left[
 \frac{\bar w\mu_G^{\mathsf T}}\Gamma
 +\frac{\sigma_G^2}t
       \left(\frac I\Gamma
                    -\frac{\mu_G\mu_G^{\mathsf T}}{\Gamma^2}\right)
                         \right]
\tag{DSG.1}
$$
$$
=\kappa_G\left[
 \frac{c}{a_x\Gamma}H\mu_G^{\mathsf T}
 +\frac{\sigma_G^2}{t\Gamma}I
 -\left(\frac{ct}{a_x\Gamma}
           +\frac{\sigma_G^2}{t\Gamma^2}\right)
                         \mu_G\mu_G^{\mathsf T}\right].
\tag{DSG.2}
$$
Neither position and velocity nor source and first-kick velocity
are assumed independent. The formulas integrate every fresh OU
outcome and retain the entire prepared pair.
:::

:::{prf:proof}
$Y$ is Gaussian with mean $\mu_G$ and covariance $\sigma_G^2I$.
Completing the square in its Gaussian density times $K$ gives
the total mass $\kappa_G$ and the tilted Gaussian with mean
$\mu_G/\Gamma$ and covariance $\sigma_G^2I/\Gamma$.
Also $W=\bar w+(Y-\mu_G)/t$ exactly, so
$$
\begin{aligned}
\mathbb E[K(Y)Y]&=\kappa_G\mu_G/\Gamma,\\
\mathbb E[K(Y)(Y-\mu_G)Y^{\mathsf T}]
 &=\kappa_G\sigma_G^2
       [I/\Gamma-\mu_G\mu_G^{\mathsf T}/\Gamma^2].
\end{aligned}
$$
Substitution proves (DSG.1). Since $a_x+tb=1$,
$\bar w=c(H-t\mu_G)/a_x$; this gives (DSG.2).
Each identity concerns the joint Gaussian pair, not a product of
its marginals.
:::

:::{prf:theorem} Signed OU-integrated second-kick and native-cap identity
:label: thm-dsg-signed-second-response

Couple two actual prepared population laws with square-integrable
positions and speed at most $V_c$, and interpolate their physical
coordinates before sharing fresh original OU and final Gaussians.
At each interpolation point let $U$ be the actual own first count
output, $E=\dot U$, $r=\dot X$, and
$$
R=\dot y=a_xr+bE,\qquad
\mathcal W=\dot w=c(E-tr),\qquad
T_0=\mathcal W+(\beta-t)R,\qquad \beta=.04.
$$
The complete first-kick spatial force is retained in $E$.
Let $L_2$ be the actual noisy second count Laplacian and
$B_2=-L_{\dot k_2}w$ its actual spatial derivative force.
The actual complete velocity derivative and its cap loss are
$$
Z=\mathcal W-tR+a(B_2-L_2\mathcal W),\quad
C=D_CZ,\quad D_C=DC_V(z),\quad a=t\nu=.006,
$$
$$
\mathcal L_C=
\|(I-D_C)Z+\beta R\|^2
 +2\langle(I-D_C)Z,D_CZ\rangle\ge0.
$$
On an independent coupled pair, fixed before its fresh OU difference,
write
$$
\eta=R-R',\qquad \psi=\mathcal W-\mathcal W',\qquad
\zeta=T_0-T'_0=\psi+(\beta-t)\eta.
$$
Evaluate $\kappa_G,\mathcal M_G$ of
{prf:ref}`lem-dsg-gaussian-response-tensor` at the actual prepared
$D=X-X'$ and $H=U-U'$. Then the exact physical differential account is
$$
\begin{split}
\mathbb E Q_\beta(R,C)={}&
\mathbb E[|R|^2+|T_0|^2]\\
&+a\mathbb E_{\rm prep,pair}
           [\zeta\cdot(\mathcal M_G\eta-\kappa_G\psi)]\\
&+a^2\mathbb E\|B_2-L_2\mathcal W\|^2
                                  -\mathbb E\mathcal L_C.
\end{split}
\tag{DSG.3}
$$
The identity also holds for each finite array, with normalized
inner products and the actual $N^{-2}$ pair sum before outer
expectation. It is an exact second-kick account, not a claimed
sign for its full pair expression or a marked-law contraction.
:::

:::{prf:proof}
The actual derivative of the noisy count field is
$-L_2\mathcal W+B_2$; its position derivative uses the complete
$R$ and its own conductances. This gives the stated $Z$.
The radial native-cap Jacobian is self-adjoint with
$0\le D_C\le I$. Expanding $Z=D_CZ+(I-D_C)Z$ proves the exact
sector identity
$$
Q_\beta(R,C)=\|R\|^2+\|Z+\beta R\|^2-\mathcal L_C.
$$
The two factors $I-D_C$ and $D_C$ commute with each other and
have a nonnegative product, so their inner product in the loss
is nonnegative. No commutation with either count graph is used.

Since $Z+\beta R=T_0+a(B_2-L_2\mathcal W)$, expand this square.
Pair symmetrization of the two actual second-kick terms gives
$$
\langle T_0,L_2\mathcal W\rangle
       =\tfrac12\mathbb E[K(Y)\zeta\cdot\psi],
$$
$$
\langle T_0,B_2\rangle
       =-\tfrac12\mathbb E[
           (\nabla K(Y)\cdot\eta)(\zeta\cdot W)]
       =\tfrac12\mathbb E[K(Y)(Y\cdot\eta)(\zeta\cdot W)].
$$
Here $T_0,R,\mathcal W$ and their pair differences are fixed
before the fresh OU draw, even though the second graph is not.
Conditional on the entire prepared pair, $Y,W$ have exactly the
joint law in (DSG.1). Therefore the complete linear-in-$a$ term
is $a\mathbb E_{\rm prep,pair}
[\zeta\cdot(\mathcal M_G\eta-\kappa_G\psi)]$.
The quadratic term and the native-cap loss remain expectations
under their actual correlated OU law. This proves (DSG.3).

All expectations are well defined without a fourth-moment
displacement assumption. Indeed $|H|\le2V_c$ and
$$
W=\frac c{a_x}H-\frac{ct}{a_x}Y
                    +q(1+ct^2/a_x)\Xi.
$$
The pointwise Gaussian maxima and $\mathbb E|\Xi|^2=2d$ give
the uniform conditional bound
$$
\mathbb E[|Y|^2e^{-|Y|^2}|W|^2]
\le C_G,
$$
$$
C_G=\frac{3(c/a_x)^2(2V_c)^2}{e}
 +\frac{12(ct/a_x)^2}{e^2}
 +\frac{6dq^2(1+ct^2/a_x)^2}{e}<\infty.
\tag{DSG.4}
$$
It follows by Jensen that
$\|B_2\|_2^2\le2C_G\|R\|_2^2$.
The first count derivative has finite $L^2$ norm because its
kernel gradient and prepared velocities are bounded, so $R$ and
$\mathcal W$ are in $L^2$. Also $\|L_2\mathcal W\|_2
\le\|\mathcal W\|_2$. These bounds justify the products,
conditioning and interpolation differentiation; bounded positional
truncation followed by $L^2$ domination gives the same identities
for unbounded square-integrable source laws.

For finite arrays the pair identities are the exact normalized
sums. On every $i\ne j$, the independent row noises give
$\Xi=\xi_i-\xi_j\sim N(0,2I_d)$ conditional on the complete
prepared array. Diagonal pair differences $\eta,\psi,\zeta$
vanish and their contributions are zero. Their zero values can
therefore be retained in the $N^{-2}$ sum without assigning an
independent noise to a self pair. No empirical moment is replaced
by a population moment.
:::

:::{prf:remark} Signed consumer still required
:label: rem-dsg-signed-consumer

(DSG.2)--(DSG.3) expose the actual source/first-velocity tensor,
the negative rank-one Gaussian term, count alignment and the
complete native-cap loss in the same account. Their conditional
integration precedes every comparison of coefficients. The quadratic
count/force square and cap loss still contain the full noisy graph;
neither is assigned an independent Gaussian marginal.

The mixed $H\mu_G^{\mathsf T}$ term has no uniform favorable sign
for general actual preparations. A delayed physical or marked-law
gap requires its signed consumption together with the first spatial
force, component/source preparation and terminal alive normalization.
No such additional consumption or invariant comparison class is
presumed here. This second exact intermediate supplies a signed
consumer for that remaining calculation, while
{prf:ref}`thm-dlb-delayed-response` supplies
the full-law history identity. Their composition into a default
nonlinear mixing rate remains open.
:::

(sec-dsg-inward-family)=
## 2. Signed absorption on a varying inward-velocity family

:::{div} feynman-prose

Here is a case where the sign can be used. Place equal masses on opposite
sides of the origin and give each an inward velocity equal to half its
distance from the origin. Changing the separation changes both the shape
and the velocities. The first graph's response is then an exact scalar,
and the Gaussian tensor shows that the second graph's linear contribution
reduces the differential energy. The proof still charges the complete
quadratic noisy remainder and retains the native cap loss. This family
provides a one-step test of the signed calculation; its preservation under
later noisy updates would require another argument.

:::

:::{prf:theorem} Optimal physical contraction of the symmetric inward family
:label: thm-dsg-inward-family

For a unit vector $e_1\in\mathbb R^3$ and $z\in[1/5,3/10]$, define the
prepared population law
$$
\lambda_z=\tfrac12\delta_{(ze_1,-ze_1/2)}
                +\tfrac12\delta_{(-ze_1,ze_1/2)}.
\tag{DSG.5}
$$
Use the actual harmonic kinetic update in
{prf:ref}`def-dsg-register`, including both own count graphs, uncapped
joint OU variables, the native cap and the independent final position
Gaussian. Let $\mathcal K\lambda_z$ be its retained physical output law,
before any conditioning on terminal marks. With
$G_\beta=\begin{psmallmatrix}I&\beta I\\\beta I&I\end{psmallmatrix}$,
$$
W_{2,G_\beta}(\mathcal K\lambda_{z_0},\mathcal K\lambda_{z_1})^2
 \le\frac{99}{100}
       W_{2,G_\beta}(\lambda_{z_0},\lambda_{z_1})^2
 =\frac{99}{100}\frac{121}{100}|z_1-z_0|^2.
\tag{DSG.6}
$$
For every even $N\ge2$, the same estimate holds for the full finite-array
physical output laws from prepared arrays having $N/2$ copies of each
atom and the same fixed sign assignment to row labels in both inputs,
using the normalized array $G_\beta$ cost. It also holds for the laws
of their random empirical physical measures, with outer squared transport
cost induced by $W_{2,G_\beta}^2$. The empirical-law assertion permits
arbitrary orderings of the balanced input arrays.

Both velocities vary with $z$, and each law has nonconstant velocities
with spread $z\in[.2,.3]$. No component-preparation contraction, terminal
alive normalization or invariant comparison family is asserted.
:::

:::{prf:proof}
Interpolate $z$ within the stated interval, pair equal signs in the two
inputs, and share the complete fresh row noises along this interpolation.
Write $k(z)=e^{-2z^2}$. The actual first count graph has
$L_1P=k(z)P$: exactly half the mass is at the opposite atom, while every
same-atom velocity difference is zero. This is also the exact finite-array
identity with denominator $N$ and zero self contribution. Therefore
$$
U_\pm(z)=\mp\tfrac z2[1-ak(z)]e_1,
\quad
\dot U_\pm(z)=\mp\tfrac12g(z)e_1,
\quad
 g(z)=1-a(1-4z^2)k(z).
\tag{DSG.7}
$$
Here $.994\le g(z)\le1$, since $0\le1-4z^2\le1$ and $a=.006$.
The complete first spatial force is included in this derivative.
Define the two positive scalars
$$
A(z)=a_x-bg(z)/2,\qquad B(z)=c[g(z)/2+t].
$$
The actual pre-OU derivatives are
$$
R_\pm=\pm A e_1,\quad
\mathcal W_\pm=\mp B e_1,\quad
T_{0,\pm}=\pm[(\beta-t)A-B]e_1.
\tag{DSG.8}
$$
The elementary alternating exponential bounds give
$.960<c<.961$, hence $.0392<b<.03922$,
$.99921<a_x<.999216$ and $q^2<.04$. Consequently
$$
.9796<A<.9797336<.98,\qquad .49632<B<.49972<.5,
$$
$$
\mathbb E(|R|^2+|T_0|^2)
 \le(.9797336)^2+(.480128)^2
 =1.19040082335296<1.1905.
\tag{DSG.9}
$$
For the second line, $B-(\beta-t)A>0$ and its upper bound is
$.49972-.02(.9796)=.480128$. All displayed decimals in these coefficient
bounds are finite rational numbers; no rounded diagnostic is used.

We next consume the full OU-integrated linear term. For a same-sign pair,
$\eta=\psi=\zeta=0$. For an opposite-sign pair, $H$, $\mu_G$ and $\eta$
are collinear with $e_1$, with
$$
|H|=z[1-ak(z)]\le.3,\qquad
|\mu_G|=2z\{a_x-b[1-ak(z)]/2\}<.6.
$$
Put $\ell=B/A>.49632$. Then $\psi=-\ell\eta$ and
$\zeta=(.02-\ell)\eta$. Let $M_\parallel=e_1^{\mathsf T}\mathcal M_Ge_1$.
Formula (DSG.2), including its complete Gaussian covariance, yields
$$
\frac{M_\parallel}{\kappa_G}
\ge-|H|\,|\mu_G|
      -(.02+.0016)|\mu_G|^2
\ge-.18-.0216(.36)>-.188.
\tag{DSG.10}
$$
Indeed $c/(a_x\Gamma)<1$,
$ct/(a_x\Gamma)<.02$ and
$\sigma_G^2/(t\Gamma^2)=2tq^2/\Gamma^2<.0016$;
we may drop the positive isotropic covariance term for this lower bound.
It follows that
$$
\zeta\cdot(\mathcal M_G\eta-\kappa_G\psi)
 =\kappa_G(.02-\ell)
       (M_\parallel/\kappa_G+\ell)|\eta|^2\le0.
\tag{DSG.11}
$$
Thus the signed linear contribution is nonpositive before any estimate of
its magnitude. It has not been replaced by an absolute force budget.

For the actual correlated quadratic remainder, use (DSG.4) with
$|H|\le.3$ in place of $2V_c$. The same proof gives
$$
C_G\le\frac{3(c/a_x)^2(.3)^2}{e}
 +\frac{12(ct/a_x)^2}{e^2}
 +\frac{18q^2(1+ct^2/a_x)^2}{e}<\frac12.
\tag{DSG.12}
$$
To check the last bound without numerical integration, use $e>2$,
$c/a_x<1$, $ct/a_x<.02$, $q^2<.04$ and
$1+ct^2/a_x<1.001$. The resulting rational upper bound is
$.135+.0012+.36(1.001)^2=.49692036<.5$.
Conditional pair integration and Jensen therefore give
$\|B_2\|_2^2\le2C_G\|R\|_2^2<\|R\|_2^2$.
Also the actual count Laplacian has $0\le L_2\le I$ in the normalized
Hilbert space, so
$$
a^2\mathbb E\|B_2-L_2\mathcal W\|^2
 <(.006)^2(.98+.5)^2<.000081.
\tag{DSG.13}
$$
This includes every fresh OU outcome, including arbitrarily large
uncapped velocities; the Gaussian bound was conditional on the actual
prepared pair before multiplying its differential displacement.

Apply (DSG.3), use (DSG.11), and retain the fact that the actual cap loss
is nonnegative. Equations (DSG.9) and (DSG.13) give
$$
\mathbb E Q_\beta(R,C)<1.190581
 <1.1979=\frac{99}{100}\frac{121}{100},
\qquad
Q_\beta(\dot X,\dot P)=1+\tfrac14-\beta=\frac{121}{100}.
\tag{DSG.14}
$$
The final position Gaussian is common under the coupling and contributes
zero to this differential. Minkowski's integral inequality in the
$L^2(G_\beta)$ output space converts (DSG.14) to the squared displacement
bound in (DSG.6). Equal-sign pairing is optimal for the two input laws:
the two possible cross-atom costs are
$(121/100)|z_1-z_0|^2$ and $(121/100)|z_1+z_0|^2$, and both $z_j$ are
positive. Taking the infimum over output couplings proves the population
claim.

For finite arrays, condition on the complete prepared array. Each opposite
pair has exactly the joint fresh Gaussian law used above; diagonal and
same-sign differential pairs contribute zero to the linear term. The
conditional quadratic estimate uses the actual normalized pair sum and
Jensen, without independence of the second graph from its velocities.
Minkowski then proves the same finite-array bound. Finally the chosen row
matching is an admissible transport plan between each pair of empirical
outputs. Its normalized cost bounds their optimal empirical cost, and the
constructed noise coupling is admissible for the outer empirical-law
transport. The input empirical measures are deterministically the two
laws in (DSG.5). The actual kinetic update is permutation equivariant,
with independent identically distributed row noises; hence reordering an
input array leaves its empirical output law unchanged. This proves the
arbitrary-ordering assertion and completes the last claim.
:::

:::{prf:remark} Scope of the signed family result
:label: rem-dsg-family-scope

The actual noisy kinetic map has a strict optimal physical transport gap
on (DSG.5), even though its velocities vary appreciably within each law.
The signed Gaussian tensor is sufficient to consume its second-kick linear
term on this family. After the update the positions and uncapped stages
have Gaussian tails, and the retained velocity need not remain proportional
to position. Neither a second-step membership statement nor a delayed
marked-law contraction follows from (DSG.6). Applying this kinetic estimate
to a complete active gas requires a separately verified prepared input in
(DSG.5), or a larger proved comparison class, together with its actual
component and alive-normalization account.
:::
