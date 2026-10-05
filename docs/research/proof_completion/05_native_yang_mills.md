# Native force scores on the recorded geometry and color fibers

(sec-native-ym-execution)=
## 1. Complete execution and source coordinates

:::{prf:definition} Execution and observation ledger for the native fiber calculation
:label: def-native-ym-execution-ledger

Every result is indexed by the complete execution record
{prf:ref}`def-native-complete-execution-record`. In particular it retains the
initial law, all landscape and reward providers, donor and clone parameters,
collision convention, domain, arithmetic and innovation convention, every
nested configuration field, recording and alignment choices, color threshold,
weights, tests and physical calibrations. The positive calculation below uses
the existing real-coordinate recorded viscous Euclidean instance of
{prf:ref}`def-native-jg-ledger`: terminal boundary schedule, Gaussian isotropic
OU noise, independent final Gaussian position diffusion, actual dense Gaussian
viscosity with either count or nonself row normalization, and no graph force,
curl or geometry feedback. The population has been revived before this
executed kinetic step. All preceding cloning, selection and collision decisions
are retained in its actual preparation law. A deterministic fixed-seed run
uses its own execution convention and does not have the Gaussian conditional
law proved here.

Let $\mathcal H$ denote the actual complete preparation through A1 of one
executed step, including its preceding record. Put

$$
a=h/2,\qquad c=e^{-\gamma h},\qquad
q^2=b_O^2
\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}
\qquad s^2=\sigma_x^2h,\qquad
\alpha=aq,\qquad \tau^2=\alpha^2+s^2.
$$

Here $h>0$, $q>0$, and $s>0$ for the principal result. The recorded A1
positions and pre-O velocities are $x_1,v_1$, and
$m=x_1+acv_1$. Arrays are concatenated over all $N$ rows and $d$ coordinates;
$k=Nd$ is their real dimension. Conditional on $\mathcal H$, the actual
standardized O and final-position draws $\xi,\zeta$ are independent
$N(0,I_k)$ vectors. The unclassified terminal position array and the B2
force inputs are exactly

$$
Y=m+\alpha\xi+s\zeta,\qquad
x^{\mathrm{B2}}=m+\alpha\xi,\qquad
v^{\mathrm{B2}}=cv_1+q\xi.
$$

The final B kick and velocity cap do not change $Y$. The terminal alive/dead
array is the actual function $\mathbf1_D(Y_i)$. Terminal geometry denotes an
existing passive readout measurable in $(\mathcal H,Y)$, including its actual
projection, metric, tessellation, numerical policy and retained-site masks.
In particular, an alive-only terminal position tessellation is included. A
readout which additionally consumes B2 positions, B2 velocities or later
random inputs must retain those extra coordinates; it is not covered by
terminal position conditioning alone.

The source coordinate $u\in\mathbb R^k$ is a finite deterministic direction
in the existing `QftExecutionConfig.innovation_shifts`, applied at the chosen
step, stream `Kinetic`, substep $2$. Its parameter $\theta$ adds $\theta u$
to those standardized O draws and changes no other configuration entry.
Thus the source is an existing addressed innovation intervention. The baseline retains the reference zero innovation shifts; the source
activates that existing configuration field. All parameters of its map remain
in $\mathfrak P$, even when a displayed Gaussian coefficient is independent
of them. Before this one addressed intervention, the preparation law is
unchanged. The same one-stage calculation applies to the already defined
predictable Gaussian force probe after conditioning on its actual preparation
and substituting its known standardized direction.
:::

(sec-native-ym-conditional-source)=
## 2. The actual force source has a geometry-preserving conditional component

:::{prf:theorem} Exact conditional source law at the terminal positions
:label: thm-native-ym-terminal-conditional-source

Use {prf:ref}`def-native-ym-execution-ledger` and set

$$
\chi=\frac{s^2}{\alpha^2+s^2},\qquad
\ell(\mathcal H,y)=\frac{\alpha}{\tau^2}(y-m),\qquad
m_G(\mathcal H,y)=\frac{\alpha}{\tau^2}u\cdot(y-m).
$$

The actual conditional law of the O draw is

$$
\xi\mid(\mathcal H,Y=y)
\ \stackrel{\rm law}{=}\ \ell(\mathcal H,y)+\sqrt\chi\,\eta,
\qquad \eta\sim N(0,I_k).
$$

Under the existing O intervention, the terminal-position density is changed
by

$$
Q_\theta(\mathcal H,y)
=\exp\left[\theta m_G(\mathcal H,y)
 -\frac{\theta^2\alpha^2|u|^2}{2\tau^2}\right],
$$

and its conditional O law at the same position array is

$$
\xi\mid(\mathcal H,Y=y;\theta)
\ \stackrel{\rm law}{=}\
\ell(\mathcal H,y)+\theta\chi u+\sqrt\chi\,\eta.
$$

Consequently the complete one-stage likelihood
$L_\theta=\exp(\theta u\cdot\xi-\theta^2|u|^2/2)$ factors exactly into the
terminal geometry density and the conditional O density:

$$
L_\theta=Q_\theta(\mathcal H,Y)\,R_\theta(\mathcal H,Y,\xi),\qquad
R_\theta=\exp\left[\theta u\cdot(\xi-\ell)
                         -\frac{\theta^2\chi|u|^2}{2}\right].
$$

The score decomposition and information split are

$$
M=u\cdot\xi=m_G+M_\perp,qquad
M_\perp=u\cdot(\xi-\ell),
$$

$$
\mathbb E[M_\perp\mid\mathcal H,Y]=0,\qquad
\mathbb E[M_\perp^2\mid\mathcal H,Y]=\chi|u|^2,
\qquad
\mathbb E[m_G^2\mid\mathcal H]=(1-\chi)|u|^2.
$$

These formulas retain the actual preparation and all terminal survival
marks. Conditioning on any survival event measurable in $(\mathcal H,Y)$
does not change the conditional O law at fixed $(\mathcal H,Y)$.
:::

:::{prf:proof}
Fix a preparation. The joint covariance of $(\xi,Y-m)$ has blocks
$I_k$, $\alpha I_k$ and $\tau^2I_k$. Put
$Z=\xi-\alpha(Y-m)/\tau^2$. Direct multiplication gives
$\operatorname{Cov}(Z,Y-m)=0$ and
$\operatorname{Cov}(Z)=I_k-\alpha^2I_k/\tau^2=\chi I_k$.
Both are linear combinations of the actual independent Gaussian draws.
Their joint density, or its characteristic function, therefore factors,
proving independence and the displayed conditional law without an
unconditional independence claim about the prepared swarm.

With the source, $\mathbb E_\theta\xi=\theta u$ and
$\mathbb E_\theta Y=m+\theta\alpha u$, with the same covariances. Thus the
conditional mean becomes
$\theta u+\alpha(y-m-\theta\alpha u)/\tau^2
=\ell+\theta\chi u$. The Gaussian density ratio for $Y$ is $Q_\theta$,
and the ratio for the conditional covariance $\chi I_k$ and mean change
$\theta\chi u$ is $R_\theta$. Their linear terms add to $\theta u\cdot\xi$
and their quadratic terms add because
$\alpha^2/\tau^2+\chi=1$. This proves the factorization. The centered
conditional Gaussian gives the first two score assertions. Since
$Y-m\sim N(0,\tau^2I_k)$, the last assertion follows by direct variance
calculation. A selection event already determined by $(\mathcal H,Y)$ is
constant in its conditional O integral, so its normalized selection changes
no such conditional law.
:::

:::{prf:corollary} The reference source retains most of its conditional noise
:label: cor-native-ym-reference-conditional-noise

For the unchanged viscous reference parameters
$h=0.04$, $\gamma=1$, $b_O=1$, and $\sigma_x=0.1$, one has

$$
q^2=\frac{1-e^{-0.08}}2,
\qquad \alpha^2=0.0004\frac{1-e^{-0.08}}2,
\qquad s^2=0.0004,
\qquad \chi=0.9629812418815075\ldots.
$$

This value is the conditional variance ratio of the actual O draw after
terminal positions and preparation have been fixed. It is independent of
$\nu$, $\rho$, the count/row choice, force and reward parameters because those
entries change the preparation or the B2 color map, rather than this last
two-noise linear covariance. The reference values of those entries remain
$\nu=0.3$, $\rho=1$, count normalization, and $U(x)=|x|^2/2$.

More generally, for $\sigma_x>0$ and fixed $b_O,\gamma$,
$\chi=[1+h q^2/(4\sigma_x^2)]^{-1}\to1$ as $h\downarrow0$.
If $s=0$ and $q>0$, conditioning on the full terminal positions determines
$\xi=(Y-m)/\alpha$ exactly. If instead the geometry retains the full B2
position array, it determines $\xi$ even when $s>0$. Its remaining conditional
O variance is zero. These conclusions concern the specified actual record
stages; they do not identify B2 and terminal geometry fibers.
:::

:::{prf:proof}
Substitute the preset values in the preceding theorem. As $h\downarrow0$,
$q^2=b_O^2h+O(h^2)$, so $h q^2/(4\sigma_x^2)\to0$. For $s=0$ the displayed
terminal position map is invertible in $\xi$; the same is true of the B2 map
for $\alpha>0$. Thus both zero-variance conclusions follow from the actual
recorded maps, rather than a changed innovation law.
:::

(sec-native-ym-weak-variation)=
## 3. A proved weak variation identity for the native conditional readout

:::{prf:theorem} Native conditional transport and its action score
:label: thm-native-ym-conditional-weak-variation

Use {prf:ref}`thm-native-ym-terminal-conditional-source`. At fixed
$(\mathcal H,y)$ write any bounded actual gauge-channel readout as
$O=\mathcal O(\mathcal H,y,\xi)$, using its complete implemented masks,
alignment, graph and normalization. Let $\nu^{\mathcal H,y}$ be the baseline
conditional O law. The finite-source conditional gauge expectation is
exactly

$$
\int\mathcal O(\mathcal H,y,z)\,\nu_\theta^{\mathcal H,y}(dz)
=\int\mathcal O(\mathcal H,y,z+\theta\chi u)
                                      \,\nu^{\mathcal H,y}(dz).
$$

For every bounded measurable $\mathcal O$, its conditional expectation is
differentiable at zero and obeys

$$
\left.\frac{d}{d\theta}\int O\,d\nu_\theta^{\mathcal H,y}\right|_0
=\int O M_\perp\,d\nu^{\mathcal H,y},
\qquad
\left|\left.\frac{d}{d\theta}\int O\,d\nu_\theta^{\mathcal H,y}\right|_0\right|
\le\sqrt\chi\,|u|\,\|O\|_{L^2(\nu^{\mathcal H,y})}.
$$

For a $C^1$ test with bounded first derivative in $z$, define the actual
conditional coordinate tangent $X_uO=\chi u\cdot\nabla_z\mathcal O$.
Then the previously required weak response identity is proved on this test
class:

$$
\int X_uO\,d\nu^{\mathcal H,y}
=\int O M_\perp\,d\nu^{\mathcal H,y}.
$$

In the actual conditional Gaussian innovation chart, the baseline action
relative to Lebesgue volume is

$$
S_0^{\mathcal H,y}(z)=\frac{|z-\ell|^2}{2\chi}
                         +\frac{k}{2}\log(2\pi\chi),
\qquad X_uS_0^{\mathcal H,y}=M_\perp,
\qquad \operatorname{div}_{dz}X_u=0.
$$

Thus the required action-score formula, including its reference-volume
divergence, holds in this native innovation chart. The complete conditional
source action increment is

$$
\Delta S_\theta^{\mathcal H,y}
=-\theta M_\perp+\frac{\theta^2\chi|u|^2}{2}.
$$

If only a descriptor $D=\mathcal D(\mathcal H,y,\xi)$ is retained, its
conditional score is
$\mathbb E[M_\perp\mid\mathcal H,y,D]$, and the same expectation identity
holds for its pullback tests. In particular the equality is proved using the
existing Gaussian update and actual descriptor map, rather than by prescribing
a connection whose divergence has the desired score. A discontinuous mask has
the bounded-measurable response above. A branch derivative alone omits its
translation across the mask boundary and is not asserted to equal that
response.
:::

:::{prf:proof}
The finite-source identity is the conditional translation already calculated.
Its density ratio is $R_\theta$. For $|\theta|\le r$, its derivative is bounded
by a constant times
$(1+|M_\perp|)\exp(r|M_\perp|)$, which is integrable for the centered Gaussian
with variance $\chi|u|^2$. Dominated convergence gives the derivative for
bounded measurable tests. Cauchy--Schwarz and the calculated variance prove
the response bound. For the specified smooth tests the translated integrand
has bounded derivative $\chi u\cdot\nabla\mathcal O$; a second application of
dominated convergence gives its expectation, proving the weak identity.
The conditional Gaussian density gives the displayed baseline action;
differentiating it in the constant direction $\chi u$ gives $M_\perp$,
and a constant vector has zero Lebesgue divergence. Taking
$-\log R_\theta$ gives its source increment. Conditioning its derivative on the
retained descriptor gives the descriptor score by the tower property. No
pointwise derivative of a discontinuous indicator is used in any of these
calculations.
:::

:::{prf:theorem} Exact preparation-posterior term on the actual coarser geometry fiber
:label: thm-native-ym-preparation-posterior-response

Let $G$ be an actual geometry descriptor measurable in $(\mathcal H,Y)$.
Let $D$ retain the chosen actual gauge readouts, and put

$$
j=\mathbb E[M\mid G,D],\qquad j^G=\mathbb E[M\mid G],\qquad s_f=j-j^G.
$$

Then the actual native fiber score decomposes as

$$
s_f
=\mathbb E[M_\perp\mid G,D]
 +\mathbb E[m_G\mid G,D]-\mathbb E[m_G\mid G],
\qquad
\mathbb E[M_\perp\mid G]=0.
$$

For a bounded gauge test $O(G,D)$ whose pullback is $C^1$ with bounded
$z$ derivative at fixed $(\mathcal H,Y)$, its full fixed-geometry response is

$$
\mathbb E[O s_f\mid G]
=\mathbb E[X_uO\mid G]
  +\mathbb E\left[O\left(m_G-\mathbb E[m_G\mid G]\right)\middle|G\right].
$$

Both terms are determined by the actual complete execution record. The second
is the change of the preparation posterior within the recorded geometry fiber;
it has not been discarded by fixing geometry. Its total size obeys the
parameter-derived bound

$$
\mathbb E\left|\mathbb E\left[
 O\left(m_G-\mathbb E[m_G\mid G]\right)\middle|G\right]\right|
\le\sqrt{1-\chi}\,|u|\,\|O\|_{L^2}.
$$

For the refined fiber $(\mathcal H,Y)$ this term is zero and the preceding
weak transport identity closes completely. For a coarser geometry the displayed
term remains even if $G$ fixes all terminal positions. At the reference
parameters its coefficient is
$\sqrt{1-\chi}=0.192402593845\ldots$; it is not zero. For $h\downarrow0$
with fixed $\sigma_x>0$, it is at most $h b_O/(2\sigma_x)$ because
$q^2\le b_O^2h$. This estimate concerns one addressed source and does not
claim that an accumulated multistep remainder vanishes.

For survival selection on an event $E$ measurable in $G$, with baseline
probability $p_E>0$, the same fiber formula holds: the normalized selected
complete score is $M-\mathbb E[M\mid E]$, and its constant cancels in
$j-j^G$. The posterior-term bound under the selected law has coefficient
$\sqrt{(1-\chi)/p_E}\,|u|$. Its survival factor is retained. The conditional
variance of $M_\perp$ at $(\mathcal H,Y)$ remains $\chi|u|^2$. A future
survival event which is not measurable in $(\mathcal H,Y)$ requires its
additional future-kernel weight and is not covered by this one-step selection
claim.
:::

:::{prf:proof}
The tower property and measurability of $G$ in $(\mathcal H,Y)$ give
$\mathbb E[M_\perp\mid G]=0$. Condition the identity
$M=M_\perp+m_G$ on $(G,D)$ and on $G$, and subtract. This proves the score
formula. Since $O$ is $(G,D)$ measurable,
$\mathbb E[O s_f\mid G]=\mathbb E[O M\mid G]
-\mathbb E[O\mid G]\mathbb E[M\mid G]$. The preceding conditional weak
identity, first at $(\mathcal H,Y)$ and then at $G$, replaces
$\mathbb E[O M_\perp\mid G]$ by $\mathbb E[X_uO\mid G]$. This yields the
response formula.

Conditional expectation is an orthogonal projection in $L^2$, hence
$\|m_G-\mathbb E[m_G\mid G]\|_2^2\le\mathbb E m_G^2
=(1-\chi)|u|^2$. Cauchy--Schwarz proves the bound. On the refined fiber
$m_G$ is measurable, so its centered posterior part is zero. Finally
$1-\chi=\alpha^2/\tau^2\le\alpha^2/s^2=h q^2/(4\sigma_x^2)$, and
$q^2\le b_O^2h$ follows directly from $1-e^{-x}\le x$. Substitution gives
the declared bound and preset coefficient. Under selection on $E\in\sigma(G)$,
the conditional law on each selected geometry fiber is unchanged, and the
normalizing derivative subtracts the constant $\mathbb E[M\mid E]$ from
both conditional scores. For the selected bound use
$\mathbb E[m_G^2\mid E]\le\mathbb E[m_G^2]/p_E$ before the same projection
and Cauchy--Schwarz argument. The conditional residual Gaussian variance is
unchanged because $E$ is already fixed at $(\mathcal H,Y)$.
:::

(sec-native-ym-color-direction)=
## 4. The induced traceless matrix direction is determined by the actual B2 color

:::{prf:lemma} Explicit conditional tangent of both native B2 viscosity normalizations
:label: lem-native-ym-b2-color-tangent

At fixed $(\mathcal H,Y)$ differentiate the preceding conditional translation.
Write a dot for its direction and use the actual matched B2 color alignment.
Then

$$
\dot x_i^{\mathrm{B2}}=\alpha\chi u_i,
\qquad \dot v_i^{\mathrm{B2}}=q\chi u_i.
$$

For the Gaussian kernel $K_{ij}=\exp(-|x_i-x_j|^2/(2\rho^2))$ put

$$
L_{ij}=-\frac{\alpha\chi}{\rho^2}(x_i-x_j)\cdot(u_i-u_j),
\qquad \dot K_{ij}=K_{ij}L_{ij}.
$$

With count normalization the full, reevaluated actual viscous force has tangent

$$
\dot F_i^{\mathrm{visc}}
=\frac\nu N\sum_{j\ne i}K_{ij}
 \left[q\chi(u_j-u_i)+L_{ij}(v_j-v_i)\right].
$$

With nonself row normalization, for $N\ge2$, put
$w_{ij}=K_{ij}/\sum_{l\ne i}K_{il}$ and
$\overline L_i=\sum_{l\ne i}w_{il}L_{il}$. Its tangent is

$$
\dot F_i^{\mathrm{visc}}
=\nu\sum_{j\ne i}w_{ij}
 \left[q\chi(u_j-u_i)+(L_{ij}-\overline L_i)(v_j-v_i)\right].
$$

For $N=1$ the actual viscosity is zero. The row denominator above is strictly
positive in the real-coordinate Gaussian implementation; no count denominator
is substituted for it.

On the actual valid branch $r_i=|F_i^{\mathrm{visc}}|>\delta_c$, define
$n_i=F_i^{\mathrm{visc}}/r_i$ and
$D_i=\operatorname{diag}(e^{i\kappa v_i^1},e^{i\kappa v_i^2},e^{i\kappa v_i^3})$,
where $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$. The native color tangent is

$$
\dot c_i=D_i\left[
 \frac{(I-n_in_i^{\mathsf T})\dot F_i^{\mathrm{visc}}}{r_i}
 +i\kappa\,n_i\odot\dot v_i^{\mathrm{B2}}\right],
$$

$$
\|\dot c_i\|\le
 \frac{|\dot F_i^{\mathrm{visc}}|}{\delta_c}
 +|\kappa|q\chi|u_i|.
$$

These formulas keep the actual B2 spatial change induced by conditional O
transport, even though terminal geometry is fixed. The preceding force source,
terminal force, capped velocity and two-velocity alignment have their different
actual chain-rule formulas and are not identified with this B2 pair.
:::

:::{prf:proof}
The conditional translation sends $\xi$ to $\xi+\theta\chi u$ in the unchanged
B2 position and velocity maps. Differentiate those maps to obtain the first
two derivatives. Differentiating the Gaussian exponent gives $L_{ij}$. In the
count case its denominator is the fixed eligible population $N$, so the
product rule gives the displayed sum. In the row case the quotient rule gives
$\dot w_{ij}=w_{ij}(L_{ij}-\overline L_i)$, which gives its displayed sum.
For $N=1$ there are no nonself summands. All these sums are the actual full
B2 reevaluation after A2.

The native phase factor differentiates as
$\dot D_i=i\kappa\operatorname{diag}(\dot v_i)D_i$, and the derivative of
normalization is
$\dot n_i=(I-n_in_i^{\mathsf T})\dot F_i/r_i$. The product rule proves the
color formula. Orthogonal projection and multiplication by a unitary diagonal
matrix are contractions, and $|n_i\odot\dot v_i|\le|\dot v_i|$. The branch
threshold therefore gives the stated norm bound.
:::

:::{prf:proposition} The actual color differential determines a minimal local $\mathfrak{su}(3)$ direction
:label: prop-native-ym-color-minimal-su3-direction

Let $c_i$ and $\dot c_i$ be the valid native colors and their proved direction
above. Set

$$
P_i=c_ic_i^\dagger,\qquad
w_i=(I-P_i)\dot c_i,\qquad
b_i=-i c_i^\dagger\dot c_i\in\mathbb R,
$$

$$
T_i=w_ic_i^\dagger-c_iw_i^\dagger
                       +\frac{i b_i}{2}(3P_i-I_3).
$$

Then $T_i^\dagger=-T_i$, $\operatorname{Tr}T_i=0$, and
$T_ic_i=\dot c_i$. Every traceless anti-Hermitian matrix with that same
image of $c_i$ is uniquely $T_i+B_i$, where

$$
B_i c_i=0,\qquad B_i^\dagger=-B_i,\qquad \operatorname{Tr}B_i=0.
$$

Such $B_i$ acts only on the two-dimensional orthogonal complement and forms
its $\mathfrak{su}(2)$ stabilizer. The displayed $T_i$ is the unique solution
of minimum Hilbert--Schmidt norm, with

$$
\|T_i\|_{\mathrm{HS}}^2=2\|w_i\|^2+\frac32 b_i^2,
\qquad \|T_i\|_{\mathrm{HS}}\le\sqrt2\,\|\dot c_i\|.
$$

For the actual contraction readouts, their same-source directional derivatives
are therefore exactly

$$
\dot q_{ij}=c_i^\dagger(T_j-T_i)c_j,
$$

$$
\dot b_{ijk}=
\det[T_ic_i,c_j,c_k]+\det[c_i,T_jc_j,c_k]
                                      +\det[c_i,c_j,T_kc_k],
$$

with $\dot\Pi_{ijk}$ obtained by the ordinary product rule. This constructs
no independent link field: $T_i$ is an algebraic expression for the already
executed readout tangent. It proves a local traceless color direction in
source-coordinate space. Spacetime parallel transport and identification of
its native action with a Yang--Mills action require the additional comparison
specified below; they are not inferred from the matrix direction.
:::

:::{prf:proof}
Differentiating $c_i^\dagger c_i=1$ gives
$2\operatorname{Re}(c_i^\dagger\dot c_i)=0$, so $b_i$ is real and
$\dot c_i=w_i+i b_i c_i$. Orthogonality gives $c_i^\dagger w_i=0$.
Thus the first two terms of $T_i$ are anti-Hermitian and traceless, and the
last is anti-Hermitian with trace $i b_i(3-3)/2=0$.
Multiplication by $c_i$ gives $w_i+i b_i c_i$, proving its required image.
The difference of any other solution and $T_i$ annihilates $c_i$. Its
anti-Hermiticity also makes its row from the complement to $c_i$ zero, so
it is precisely a traceless anti-Hermitian complement block. This proves the
stabilizer characterization and uniqueness of that difference.

In a unitary basis whose first vector is $c_i$, $T_i$ has diagonal blocks
$i b_i$ and $-i b_i I_2/2$, and off-diagonal blocks $w_i,-w_i^\dagger$.
The stabilizer block is Hilbert--Schmidt orthogonal both to those off-diagonal
blocks and to the scalar complement block. Hence
$\|T_i+B_i\|_{\mathrm{HS}}^2=\|T_i\|_{\mathrm{HS}}^2+\|B_i\|_{\mathrm{HS}}^2$.
Its computed squared norm is $2\|w_i\|^2+3b_i^2/2$, bounded by
$2(\|w_i\|^2+b_i^2)=2\|\dot c_i\|^2$. This proves minimality and the bound.
Finally differentiate $c_i^\dagger c_j$ and use $T_i^\dagger=-T_i$.
Multilinearity of the actual determinant gives the three displayed terms.
The triangle product derivative follows by its product rule. Every formula
uses the proved native tangent and introduces no new dynamics.
:::

(sec-native-ym-residual)=
## 5. Remaining local action identification

:::{prf:remark} What these native calculations discharge and what they retain
:label: rem-native-ym-local-action-residual

The fixed-preparation terminal geometry calculation now proves a complete
finite-source conditional transport law, its conditional action and the weak
first-variation identity. On the actual coarser geometry fiber it proves the
additional preparation-posterior term and bounds that term from the configured
noise coefficients. The actual B2 source tangent gives its complete count and
row force derivatives, native color derivatives and local traceless matrix
directions. These results apply to the unchanged viscous reference; they
require neither stationary chaos nor an assumed local gauge law.

The native geometry-fiber action in
{prf:ref}`thm-ym-native-geometry-fiber-action` remains the negative logarithm
of the actual conditional pushforward likelihood. The calculation above does
not establish that likelihood as a quadratic local curvature functional.
A continuum local Yang--Mills identification still requires all of the
following for this same record: a spacetime transport construction whose
actual color and link differentials agree with the source directions just
calculated; the retained preparation and mask contributions in its weak
first variations; and convergence of the native conditional action, including
its reference-volume term, to the claimed local curvature action. The
$\mathfrak{su}(2)$ complement freedom proved above is an unrecorded direction
of a single color vector and cannot be supplied as an extra fluctuating field
without further native data.

For a native descriptor that includes full B2 positions at fixed preparation,
the fresh O conditional direction is zero, as already proved; its remaining
source response is the actual preparation-mixture response. For a terminal
position descriptor, the conditional component is the positive $\chi$ above.
This distinction characterizes the existing recording regimes. It does not
classify the algorithm itself as lacking gauge fluctuations, and it does not
replace the source action by a Wilson action.
:::
