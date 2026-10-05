# Independent audit of the signed Gaussian count response

(sec-dsga65-source)=
## 1. Frozen source and quantified endpoint

:::{prf:definition} Reviewed signed-response source
:label: def-dsga65-source

This independent review concerns
`61_default_signed_gaussian_tensor.md` at SHA-256
`bb2a271f9905c42a3edf9ac3881a7990bf77c8e264fdb3f37d01785670776425`.
The actual register is {prf:ref}`def-dsg-register`: harmonic force
$F(x)=-x$, $d=3$, $h=.04$, count viscosity $\nu=.3$,
radius $\rho=1$, stored cap $V=2$ and prepared speed bound $V_c=4$.
Both own count fields, their original denominator $N$ at finite size,
the complete OU innovation, the native radial cap and the independent
final position Gaussian are retained.

The reviewed conclusions are the conditional Gaussian tensor
{prf:ref}`lem-dsg-gaussian-response-tensor`, the exact differential
identity {prf:ref}`thm-dsg-signed-second-response`, and the physical
kinetic-family contraction {prf:ref}`thm-dsg-inward-family`.
The last result concerns retained physical outputs before conditioning
on terminal marks. It does not assert that active preparation maps an
arbitrary entering swarm into its balanced family.

All finite-array phase costs are normalized by $N$. The finite labeled
array statement requires the same fixed assignment of signs to row labels
in both inputs. The outer empirical-measure-law statement permits arbitrary
orderings, using the actual permutation equivariance of the kinetic update.
:::

(sec-dsga65-gaussian)=
## 2. The actual joint Gaussian tensor

:::{prf:lemma} Gaussian tensor verification
:label: lem-dsga65-tensor

The formulas (DSG.1)--(DSG.2) hold with the actual correlated
position/velocity pair, without an independent graph or velocity draw.
:::

:::{prf:proof}
Fix the complete prepared pair before the fresh innovation and use the
source notation
$$
D=X-X',\qquad H=U-U',\qquad
Y=a_xD+bH+tq\Xi,\qquad W=c(H-tD)+q\Xi,
\qquad \Xi\sim N(0,2I_3).
$$
Thus $Y$ has mean $\mu_G=a_xD+bH$ and covariance
$\sigma_G^2I$, where $\sigma_G^2=2t^2q^2$.
Multiplication by $K(Y)=e^{-|Y|^2/2}$ changes its total mass to
$$
\kappa_G=(1+\sigma_G^2)^{-3/2}
           e^{-|\mu_G|^2/[2(1+\sigma_G^2)]}
$$
and its normalized mean and covariance to
$\mu_G/\Gamma$ and $\sigma_G^2I/\Gamma$,
where $\Gamma=1+\sigma_G^2$.

The identity $W=\bar w+(Y-\mu_G)/t$ is pointwise, with
$\bar w=c(H-tD)$. Therefore
$$
\mathbb E[K(Y)WY^{\mathsf T}]
=\kappa_G\left[
 \frac{\bar w\mu_G^{\mathsf T}}\Gamma
 +\frac{\sigma_G^2}t
    \left(\frac I\Gamma-
                \frac{\mu_G\mu_G^{\mathsf T}}{\Gamma^2}\right)
 \right].
$$
The relation $a_x+tb=1$ gives
$\bar w=c(H-t\mu_G)/a_x$, proving the second displayed tensor.
In particular the isotropic Gaussian covariance term and the negative
rank-one term both have their stated coefficients. This calculation
conditions on the prepared pair; it does not split the position and
velocity marginals of the noisy stage.
:::

(sec-dsga65-ledger)=
## 3. Signed force account and the complete cap

:::{prf:lemma} Exact differential account verification
:label: lem-dsga65-ledger

The sign, factor and cap loss in (DSG.3) are correct. Its products are
integrable for the declared square-integrable prepared laws, and its
finite-array identity uses the actual self terms correctly.
:::

:::{prf:proof}
Let $E=\dot U$ retain the entire own first-field derivative. Then
$R=a_xr+bE$ and $\mathcal W=c(E-tr)$ are fixed before the fresh OU
draw. The own second-field derivative is exactly
$B_2-L_2\mathcal W$, so its full pre-cap velocity derivative is
$$
Z=\mathcal W-tR+a(B_2-L_2\mathcal W).
$$
For $D_C=DC_V(z)$, the native radial cap satisfies
$0\le D_C\le I$ and is self-adjoint. Writing
$C=D_CZ$ gives the exact square identity
$$
Q_\beta(R,C)=|R|^2+|Z+\beta R|^2
 -|(I-D_C)Z+\beta R|^2
 -2\langle(I-D_C)Z,D_CZ\rangle.
$$
The last inner product is nonnegative because $D_C$ and $I-D_C$
have nonnegative product. Neither has been commuted through a count
Laplacian. This verifies the full nonnegative loss $\mathcal L_C$.

Put $T_0=\mathcal W+(\beta-t)R$ and pair differences
$\eta=R-R'$, $\psi=\mathcal W-\mathcal W'$ and
$\zeta=T_0-T_0'$. Symmetry of the actual count conductances gives
$$
\langle T_0,L_2\mathcal W\rangle
=\tfrac12\mathbb E[K(Y)\zeta\cdot\psi],
\qquad
\langle T_0,B_2\rangle
=\tfrac12\mathbb E[K(Y)(Y\cdot\eta)(\zeta\cdot W)].
$$
The plus sign in the second identity follows from
$\nabla K(Y)=-YK(Y)$ and $B_2=-L_{\dot k_2}w$.
The square contributes twice each inner product, leaving the single
factor $a$ in
$a\mathbb E_{\rm prep,pair}\zeta\cdot
(\mathcal M_G\eta-\kappa_G\psi)$.
The quadratic noisy square and the cap loss retain their original joint
Gaussian law. Thus (DSG.3) follows without an averaged-Jacobian
factorization.

For integrability, the exact affine identity
$$
W=\frac c{a_x}H-\frac{ct}{a_x}Y
                 +q(1+ct^2/a_x)\Xi
$$
and $|H|\le2V_c$ give (DSG.4), using
$\sup r^2e^{-r^2}=e^{-1}$,
$\sup r^4e^{-r^2}=4e^{-2}$ and
$\mathbb E|\Xi|^2=6$.
It is a conditional bound independent of $D$, so it can be multiplied
by the actual, possibly correlated, squared displacement after conditioning.
Jensen and the pair identity then imply
$\|B_2\|_2^2\le2C_G\|R\|_2^2$.
The first-field derivative has finite $L^2$ norm since its velocity and
kernel-gradient factors are bounded. Also $0\le L_2\le I$ for the
Gaussian count kernel: its normalized kernel operator is positive
semidefinite, and its degree multiplication operator is at most $I$.
These bounds justify every product and the stated $L^2$ truncation argument.

For a finite array, each off-diagonal pair has
$\Xi=\xi_i-\xi_j\sim N(0,2I_3)$ conditional on the whole prepared
array. Diagonal differences $\eta,\psi,\zeta$ vanish, so their force
and differential pair contributions are zero. Retaining them in the
$N^{-2}$ sum does not assign independent noise to a self pair.
:::

(sec-dsga65-family)=
## 4. Exact signed absorption on the inward family

:::{prf:lemma} Rational coefficient and sign verification
:label: lem-dsga65-family

Every strict coefficient comparison in (DSG.9)--(DSG.14) follows from
the stated analytic bounds and exact rational arithmetic. In particular
the actual second-field linear term is nonpositive on the full interval
$z\in[1/5,3/10]$.
:::

:::{prf:proof}
For the balanced law (DSG.5), the own first graph obeys
$L_1P=e^{-2z^2}P$. Hence
$$
\dot U_\pm=\mp\tfrac12g(z)e_1,\qquad
g(z)=1-a(1-4z^2)e^{-2z^2}\in[.994,1].
$$
This derivative includes the changing first graph. With
$A=a_x-bg/2$ and $B=c(g/2+t)$, the coarse exponential bounds give
$$
.9796<A<.9797336,
\qquad .49632<B<.49972.
$$
The baseline square is therefore at most
$$
(.9797336)^2+(.480128)^2
=1.19040082335296<1.1905.
$$
All decimals in this audit denote their exact terminating rational values.

Same-sign pairs have zero differential pair differences. Opposite-sign
pairs have $|H|\le.3$, $|\mu_G|<.6$ and
$$
\psi=-\ell\eta,\qquad \zeta=(.02-\ell)\eta,
\qquad \ell=B/A>.49632.
$$
The exact tensor gives
$$
M_\parallel/\kappa_G
\ge-(.3)(.6)-(.02+.0016)(.6)^2
=-.187776>-.188.
$$
Thus $.02-\ell<0$ and
$M_\parallel/\kappa_G+\ell>0$, which prove the claimed nonpositive
linear term. This is a conditional signed calculation rather than an
absolute force estimate or a presumed favorable provider response.

For the actual quadratic remainder, the conditional Gaussian bound with
$|H|\le.3$ is at most
$$
.135+.0012+.36(1.001)^2=.49692036<\tfrac12.
$$
Consequently $\|B_2\|_2<\|R\|_2<.98$ and
$\|L_2\mathcal W\|_2<.5$. Its full charged square is less than
$$
(.006)^2(1.48)^2=.0000788544<.000081.
$$
Keeping the nonnegative native-cap loss yields
$$
\mathbb E Q_\beta(R,C)<1.190581
 <1.1979=(99/100)(121/100).
$$
The input differential has exactly
$Q_\beta(\dot X,\dot P)=1+1/4-\beta=121/100$.
The estimates integrate all Gaussian outcomes and do not bound an
uncapped velocity pointwise.
:::

(sec-dsga65-transport)=
## 5. Optimal transport and finite-label quantifiers

:::{prf:remark} The upper transport bound has the correct direction
:label: rem-dsga65-transport

For each interpolation parameter, the constructed shared-noise outputs
have the actual own-field kinetic marginals. Minkowski in the
$L^2(G_\beta)$ output space therefore integrates the differential bound
to an admissible output coupling. The final position Gaussian is shared
and contributes no differential displacement. Its cost is an upper bound
on the infimum defining optimal output transport.

The input laws have two possible atom-pair costs,
$(121/100)|z_1-z_0|^2$ and $(121/100)|z_1+z_0|^2$.
Because $z_0,z_1>0$, equal-sign matching is optimal. This proves the
population factor $99/100$ relative to the actual optimal input cost,
rather than relative to an arbitrary larger input-plan cost.

For labeled finite arrays, equal-sign interpolation requires the same
fixed sign assignment in both arrays. This hypothesis is explicitly
included in the frozen theorem. At every interpolation point the exact
normalized finite graph and its pair sums give the same bound, without a
finite-size residual. For each coupled pair of output arrays, row matching
is an admissible transport plan between their empirical measures. Taking
the outer infimum can only decrease its cost. Finally permutation
equivariance and iid row noises leave the empirical output law unchanged
under an input reordering. This proves the separate arbitrary-ordering
empirical-law assertion; it does not assert arbitrary-ordering contraction
in a fixed labeled array metric.
:::

(sec-dsga65-scope)=
## 6. Accepted result and unresolved composition

:::{prf:remark} Scope of this accepted audit
:label: rem-dsga65-scope

The frozen source's exact tensor, signed differential identity and
$99/100$ optimal physical kinetic contraction pass this independent
review. The balanced family has genuinely varying within-law velocities;
its proof consumes the signed Gaussian tensor instead of assuming a
constant-velocity input or a narrow pointwise velocity band.

The source does not prove that a complete active preparation produces
this family, that a noisy output remains in it, or that terminal marks and
separate survivor normalizations preserve its physical estimate. Those
are additional interfaces. In particular the accepted one-update kinetic
bound alone does not establish a delayed default nonlinear law rate,
a finite-particle quasi-stationary mixing rate or an iterated invariant
comparison class. This audit neither assumes those conclusions nor
refutes a later proof of them.
:::
