# Native projected non-Abelian transport and its action correspondence

(sec-npc-native-record)=
## 1. Complete record, ambient comparisons, and actual native projectors

:::{prf:definition} Native projected-connection parameter record
:label: def-npc-complete-record

Retain the complete execution record $\mathfrak P$ in
{prf:ref}`def-native-complete-execution-record`, including every nested
algorithm, provider, initial-law, random-stream, arithmetic, boundary,
recording, masking, geometry and calibration entry. The positive native
calculation uses the existing real-coordinate viscous reference and its
matched B2 color instrument in
{prf:ref}`def-ngf-parameter-ledger`. The actual full preparation and its
terminal-position fiber are those in
{prf:ref}`def-native-ym-execution-ledger`; both count and nonself row
normalizations retain their original denominators. The source is the
existing addressed O innovation shift, with conditional coordinate tangent
$\delta_f z_i=q\chi f_i$ and
$\delta_f X_i=aq\chi f_i$. All Gaussian innovations and all earlier
cloning/collision decisions remain in their original law.

On an available color row,

$$
c_i=\frac{F_i^{\rm visc}}{|F_i^{\rm visc}|}
       \odot e^{i\kappa z_i},\qquad
|F_i^{\rm visc}|>\delta_c,\qquad
P_i=c_ic_i^\dagger,\quad Q_i=I-P_i,\qquad
\kappa=m\ell_0/\hbar_{\rm eff}.
$$

The force is the actual recorded **viscous** force, not the complete B2
force substituted into that formula. Retain the force threshold, stage,
same-stage alignment and original finite-value tests. The phase scale is
an existing fixed-length readout, or an actual preceding-record calibration
fixed on this source fiber. If its calibration itself changes under this
source, its derivative is an additional consumed term and the displayed
fixed-$\kappa$ calculation does not discard it. A mask deleting every row
that accepted a clone is a different channel and is not used by the revived
tag construction below.

The native full projector algebra includes $Q_i=I-P_i$ and products of
these matrices, since they are polynomials in the already retained $P_i$.
Their common-frame contractions are computed in the algorithm's actual
ambient $\mathbb C^3$. An ambient comparison is initially the identity
between these stored component frames. Under a **coordinate** frame change
$\Omega_i\in SU(3)$ it becomes $\Omega_j\Omega_i^{-1}$ from $i$ to $j$.
This does not presume that physical changes of the force and velocity
components are symmetries of the gas law.

The deterministic composite transport derived below is a function of these
native projectors. It is distinguished from the literal $x+iv$ ray edge
transport of {prf:ref}`def-nyc-complete-record`, from the assigned attribution
$SU(2)$ transports, and from a separately sampled lattice gauge field.
It adds no random input or force. Its availability requires two rank-one
projectors with $|c_i^\dagger c_j|>0$, using their actual retained masks.
:::

(sec-npc-projection-transport)=
## 2. The non-Abelian transport determined by native projection products

:::{prf:theorem} Canonical native projection-product transport
:label: thm-npc-native-direct-rotation

For two available native projectors $P_i,P_j$ let

$$
S_{j\leftarrow i}=P_jP_i+Q_jQ_i,
\qquad \Delta=P_j-P_i,\qquad
U_{j\leftarrow i}=S_{j\leftarrow i}(I-\Delta^2)^{-1/2}.
$$

If $t=|c_i^\dagger c_j|>0$, this matrix is a uniquely determined
$SU(3)$ element, and

$$
S^\dagger S=I-\Delta^2,\quad
U_{j\leftarrow i}P_iU_{j\leftarrow i}^\dagger=P_j,\quad
U_{i\leftarrow j}=U_{j\leftarrow i}^{-1},\quad
U_{j\leftarrow i}c_i
 =c_j\frac{c_j^\dagger c_i}{|c_j^\dagger c_i|}.
$$

Its eigenvalues are $1,e^{i\theta},e^{-i\theta}$, where
$\theta=\arccos t\in[0,\pi/2)$. Thus
$\|P_j-P_i\|=\sin\theta$ and
$\|U_{j\leftarrow i}-I\|=2\sin(\theta/2)$.
For $t\ge\eta>0$ the actual first-order comparison has the explicit error

$$
\|U_{j\leftarrow i}-I+[P_i,\Delta]\|
\le\left[1+\frac1{\eta(1+\eta)}\right]\|\Delta\|^2.
$$

With nontrivial ambient comparison $T_{j\leftarrow i}$, the same construction
applies to $P_j$ and $T_{j\leftarrow i}P_iT_{j\leftarrow i}^\dagger$,
followed by $T_{j\leftarrow i}$. Under coordinate frames it therefore obeys
the true two-endpoint law

$$
U_{j\leftarrow i}'=\Omega_jU_{j\leftarrow i}\Omega_i^{-1}.
$$

Leaving the ambient comparison equal to $I$ after unrelated local frame
changes instead computes a different matrix. The common-frame projector
algebra is not silently declared locally invariant in that manner.
:::

:::{prf:proof}
The projections $P_i,Q_i$ are complementary, as are $P_j,Q_j$.
Multiplying $S^\dagger S$ and using these identities gives
$I-(P_j-P_i)^2$. Also $SP_i=P_jS$, and
$\Delta^2$ commutes with $P_i$, so the same intertwining holds for $U$.
Choose the phase of $c_j$ so that $c_i^\dagger c_j=t>0$.
On the plane spanned by $c_i,c_j$, choose the orthonormal coordinates
$c_i,e$ with $c_j=t c_i+\sqrt{1-t^2}\,e$. In these coordinates

$$
S=\begin{pmatrix}t^2&-t\sqrt{1-t^2}\\
                   t\sqrt{1-t^2}&t^2\end{pmatrix},\qquad
U=\begin{pmatrix}t&-\sqrt{1-t^2}\\
                   \sqrt{1-t^2}&t\end{pmatrix}.
$$

On the remaining one-dimensional common complement both matrices are the
identity. This proves unitarity, determinant one, the eigenvalues, the
line-transport formula and the norms, including the coincident-projector
case by continuity. It also proves inverse reversal; alternatively use
the inverse polar factors of the adjoint.

The exact identity $S-I=\Delta(2P_i-I)$ gives $\|S-I\|=\|\Delta\|$.
Writing $r=\|\Delta\|$ and $t=\sqrt{1-r^2}$,

$$
\|U-S\|\le(1-r^2)^{-1/2}-1
=\frac{r^2}{t(1+t)}.
$$

The projector identity $P_i\Delta+\Delta P_i+\Delta^2=\Delta$ gives
$S-I=-[P_i,\Delta]-\Delta^2$. Combining these proves the displayed
error. With an ambient comparison, form both projections in its target
fiber. Their projection products and positive square roots conjugate by
$\Omega_j$, and the final comparison transforms at its two endpoints.
Multiplication proves the stated frame law without changing the original
ambient comparison.
:::

:::{prf:theorem} Projected connection and non-Abelian curvature on native charts
:label: thm-npc-projected-connection-curvature

Let $P(s)$ be an available native rank-one projector on any smooth
execution-coordinate chart of the actual color map. In its retained ambient
frame define the derivative by the existing projection comparisons:

$$
\nabla v=P\,d(Pv)+Q\,d(Qv)=dv+\omega v,
\qquad \omega=[P,dP].
$$

Then $\omega$ is anti-Hermitian and traceless, and its curvature is

$$
\mathcal F=d\omega+\omega\wedge\omega=dP\wedge dP,
\qquad \nabla P=dP+[\omega,P]=0.
$$

In a changed ambient frame with ambient connection $\Gamma$ these same
native comparisons give

$$
\nabla=d+\Gamma+[P,D^\Gamma P],\qquad
\omega'=\Omega\omega\Omega^{-1}-d\Omega\,\Omega^{-1},\qquad
\mathcal F'=\Omega\mathcal F\Omega^{-1}.
$$

Thus this is a local non-Abelian connection of the actual line/complement
splitting, including its reference ambient comparison. It is reducible:
its parallel transport preserves the native line and its rank-two
complement. At a point with unit color $c$, write horizontal color tangents
$w_X=Q\partial_Xc$. Its curvature is explicitly

$$
\mathcal F(X,Y)
=w_Xw_Y^\dagger-w_Yw_X^\dagger
 +c(w_X^\dagger w_Y-w_Y^\dagger w_X)c^\dagger.
$$

In particular the rank-two component may be noncommuting; it is not merely
the scalar color-line phase curvature. This statement concerns a native
execution chart. It asserts no unproved smooth spacetime interpolation.
:::

:::{prf:proof}
Differentiate $P^2=P$ to get $P(dP)P=Q(dP)Q=0$. Hence $dP$ is
off-diagonal relative to $P\oplus Q$, and
$[P,dP]=(2P-I)dP$. Expanding the displayed projected derivative gives
$dv+[P,dP]v$. Since $P,dP$ are Hermitian, their commutator is
anti-Hermitian and has trace zero. Direct differentiation gives
$d[P,dP]=2dP\wedge dP$; multiplication of its off-diagonal blocks gives
$[P,dP]\wedge[P,dP]=-dP\wedge dP$. This proves the curvature formula.
The same projector identities give $[[P,dP],P]=-dP$, proving parallel
preservation of $P$.

Replacing $d$ by the actual ambient derivative $D^\Gamma$ gives the
covariant projected formula. A frame change transforms the entire
ambient derivative as well as the projections. Applying the ordinary
product rule to a transformed section yields the displayed connection
law, and squaring that derivative yields the curvature law. For the
curvature coefficient, differentiate $P=cc^\dagger$. The phase component
of $dc$ cancels, so
$\partial_XP=w_Xc^\dagger+cw_X^\dagger$. Multiply these two matrices
in both orders and subtract, using $c^\dagger w_X=0$. This gives exactly
the stated curvature.
:::

:::{prf:theorem} Native projection products converge to their determined transport
:label: thm-npc-projection-product-limit

Along a $C^2$ available native chart path $P(t)$, $0\le t\le T$,
let $U_\pi$ be the ordered product of the actual direct rotations in
{prf:ref}`thm-npc-native-direct-rotation` over a partition $\pi$.
Set $M_1=\sup|P'(t)|$, $M_2=\sup|P''(t)|$ on this finite chart path.
For sufficiently small mesh $|\pi|$, every edge is available and

$$
\|U_\pi-\mathcal U(T)\|
\le T|\pi|\left[\tfrac12 M_2+4M_1^2
 +\tfrac12\sup_t|\omega'(t)|
 +\tfrac12\sup_t|\omega(t)|^2\right],
$$

where $\mathcal U$ solves the uniquely determined unitary transport
equation

$$
\mathcal U'(t)=-[P(t),P'(t)]\mathcal U(t),\qquad \mathcal U(0)=I.
$$

Here $\sup|\omega|\le2M_1$ and
$\sup|\omega'|\le2M_2$, so the right-hand side is an explicit finite
profile of that actual path. The line restriction of this product is the
inverse-oriented normalized native color-overlap product. It therefore
retains the existing color triangle phase. The rank-two restriction
contains the additional non-Abelian comparisons determined by the full
native projector algebra.
:::

:::{prf:proof}
For an interval of length $\Delta t$,
$\Delta P=P'\Delta t+R$ with $|R|\le M_2\Delta t^2/2$.
For small mesh ensure $|\Delta P|\le1/2$, so its overlap is at least
$\sqrt3/2$. The preceding direct-rotation error is at most
$2|\Delta P|^2$. The Taylor remainder $R$ is Hermitian, so
$\|[P,R]\|=\|PRQ\|\le\|R\|$. Thus

$$
U_{t+\Delta t\leftarrow t}
=I-[P(t),P'(t)]\Delta t+R_U,
\qquad |R_U|\le(\tfrac12 M_2+2M_1^2)\Delta t^2.
$$

The solution of the displayed transport equation exists by successive
integrals on this bounded continuous chart coefficient; their factorial
majorant converges uniformly. Differentiating $\mathcal U^\dagger\mathcal U$
and its determinant gives unitarity and determinant one. Its exact one-step
propagator differs from $I-\omega(t)\Delta t$ by at most
$[\sup|\omega'|+\sup|\omega|^2]\Delta t^2/2$.
Telescope the two products; each earlier and later factor has norm one,
so the errors add and $\sum\Delta t^2\le T|\pi|$.
The resulting bound is no larger than the displayed one after using
$\sup|\omega'|\le2M_2$ and $\sup|\omega|\le2M_1$.
Finally multiply the exact line restrictions from the first theorem.
Their phases are the conjugates of the forward overlaps, giving the
stated ordered line product. The complement is preserved at each edge,
so its product is its determined rank-two transport.
:::

(sec-npc-native-curvature-support)=
## 3. Non-Abelian curvature generated by the actual B2 innovations

:::{prf:theorem} A full native projector tangent and noncommuting curvature regime
:label: thm-npc-native-curvature-nondegeneracy

Keep the complete three-dimensional reference dynamical parameters, either
count or its declared row branch, and the existing matched B2 color
instrument. Require the primitive values

$$
N\ge4,\quad \nu>0,\quad q\chi>0,\quad \sigma_J>0,
\quad\kappa\ne0,\quad 1-a^2\lambda>0,\quad R>\delta_c.
$$

Take the actual marked preparation in which only donor slot $4$ is alive
near $0$, all entering velocities are near $0$, and the other slots
undergo their mandatory revival. This is a positive-probability stratum
under the proved incoming QSD at each finite $N$. Write
$n=(1,1,1)/\sqrt3$ and $n_*=N$ for count or $N-1$ for row.
There is a positive-probability native B2 neighborhood of

$$
z_1=0,\quad z_2=\frac{n_*R}{\nu}n,\quad z_i=0\ (i\ne2),
\qquad X_i^{\rm B2}=0\ (1\le i\le N),
$$

in which the tag-$1$ color is valid and its projector map has real
differential rank four, the full dimension of $\mathbb{CP}^2$.
On this neighborhood there are fixed actual O-source directions
$f_1,f_2,f_3$ such that

$$
[\mathcal F(f_1,f_2),\mathcal F(f_1,f_3)]\ne0.
$$

The norm of this commutator is a coordinate-frame invariant native
curvature observable and is bounded away from zero on a smaller
positive-probability neighborhood. It follows from the actual force
map and independent residual innovations, rather than from an assumed
distribution of $SU(3)$ links.

For $N=200$, $R=.01$, the unchanged reference count branch uses
$|z_2|=6.6666667\ldots$ and the required revived pre-B1 position
$|X_2^J|=.13338669\ldots$; all other displayed pre-B1 positions are zero.
These are allowed, uncapped O inputs and actual Gaussian clone jitters.
The threshold $\delta_c=10^{-12}$ is strictly passed. Constants are
finite-population and may be very small in probability; no uniform
physical curvature lower bound is inferred.
:::

:::{prf:proof}
At the displayed B2 positions the Gaussian kernel is one and its first
spatial derivative is zero. The actual count/row force of tag $1$ is
$F_1^{\rm visc}=\nu z_2/n_*=Rn$. Its differential is exactly

$$
\delta F_1^{\rm visc}
=\frac\nu{n_*}\left[\sum_{j\ne1}\delta z_j-(N-1)\delta z_1\right].
$$

The quotient derivative of the row normalizer is zero here because
every $\delta K_{1j}=0$. Consequently any prescribed pair
$(\delta F_1,\delta z_1)$ is produced by changing only O rows $1,2$:

$$
\delta z_2=\frac{n_*}{\nu}\delta F_1+(N-1)\delta z_1,
\qquad \delta z_j=0\ (j\ne1,2).
$$

These are actual source directions $f_i=\delta z_i/(q\chi)$.
At this point $c_1=n$. Choose a real orthonormal pair $e,f$ in
$n^\perp$. A force direction $\delta F_1=Re$, with $\delta z_1=0$,
gives horizontal color tangent $w=e$; likewise obtain $w=f$.
To obtain $w=if$, set $\delta F_1=0$ and
$(\delta z_1)_a=f_a/(\kappa n_a)$, and use the compensating row-$2$
direction above. The phase derivative is $i f$, already orthogonal
to $n$. Similarly obtain $ie$. Thus the differential spans four real
dimensions, and its rank remains four on a sufficiently small valid
neighborhood.

By the exact curvature formula the complement blocks for the tangent
pairs $(e,f)$ and $(e,if)$ are

$$
K_1=ef^\dagger-fe^\dagger,\qquad
K_2=-i(ef^\dagger+fe^\dagger).
$$

Their commutator is a nonzero diagonal traceless matrix on this two-plane
with eigenvalues $2i,-2i$. The line blocks vanish for these chosen pairs,
so the full curvature commutator is also nonzero. Fix these three
computed innovation directions; continuity bounds its norm away from
zero on a smaller neighborhood.

To verify actual support, start from the nonempty marked state with
only slot $4$ alive, donor position and every velocity zero. All dead rows
are revived from this actual eligible source. Their existing independent
Gaussian jitters give positive density for every finite collection of
post-copy positions. Copying and all collision rotations preserve the zero
velocities. The first potential half-kick and A1 give
$p_i=(1-a^2\lambda)X_i^J$. Thus choose the legitimate jitters
$X_i^J=-az_i/(1-a^2\lambda)$; donor $4$ remains at zero.
All displayed B2 positions are then zero. The original O draws have
positive density at the finite required $z_i$. At a fixed finite terminal
position array, the independent final position diffusion leaves the
conditional O variance $\chi q^2>0$, hence the same support.
Slight changes of entering state, jitter, rotation and O draws preserve
the claimed strict rank and curvature conditions. These are open events
of the actual native kernels, so their joint probability is positive.

For completeness this marked entering neighborhood has positive incoming
QSD probability, without assuming a new QSD support property. The
preparation-independent target inverse/density in
{prf:ref}`thm-nyc-uniform-target-inverse` gives a positive full terminal
density at all finite positions and at velocities near zero, from every
nonempty entering state, for the unchanged count reference. Choose its
position boxes so only slot $4$ is in $D$ and all other rows are outside.
The killed kernel assigns this nonempty marked neighborhood positive mass.
The already proved QSD identity $\nu_NQ_N=\alpha_N\nu_N$ then gives it
positive $\nu_N$ mass. For the declared row reference use its already
proved positive finite target density in
{prf:ref}`thm-cgd-finite-n-qsd`; the same support argument applies.
The count arithmetic values follow from
$200(.01)/.3=20/3$ and
$.02(20/3)/(1-.0004)$. No Gaussian draw is clipped and no normalized
source-direction lower bound independent of $N$ is claimed.
:::

:::{prf:corollary} Explicit native rank-four tangent covariance
:label: cor-npc-native-projector-bracket

At the actual prototype of
{prf:ref}`thm-npc-native-curvature-nondegeneracy`, put
$t=\nu/(n_*R)$ and $b=\kappa/\sqrt3$. Express the horizontal color
tangent in a real orthonormal pair of $n^\perp$, recording real and
imaginary parts separately. Applying its native differential to independent
isotropic velocity tangents of amplitude $a_{\rm noise}$ gives two
identical covariance blocks

$$
\Sigma_2=a_{\rm noise}^2
\begin{pmatrix}
t^2N(N-1)&-t(N-1)b\\
-t(N-1)b&b^2
\end{pmatrix}.
$$

Their minimum eigenvalue has the explicit lower bound

$$
\lambda_{\min}(\Sigma_2)
\ge a_{\rm noise}^2
 \frac{t^2b^2(N-1)}{t^2N(N-1)+b^2}>0.
$$

For the fixed terminal-position residual innovation use
$a_{\rm noise}=q\sqrt\chi$, not the mean-source coefficient $q\chi$.
For an actual continuous-coordinate velocity diffusion of amplitude $b_O$
in a separately proved native scaling regime use $a_{\rm noise}=b_O$;
the result is its local projector bracket on this valid chart. In
Frobenius-orthonormal matrix tangent coordinates $\delta P$, the same
covariance has an additional factor two.

At $N=200$, $R=.01$, $\nu=.3$, count normalization, $\kappa=1$ and
$a_{\rm noise}=1$, the bound is $.0016660465116279\ldots$.
This is an explicit **Jacobian/tangent covariance**; it is not the
covariance of the nonlinear color at a finite-noise step and is not a
population-uniform limiting gauge-field bracket.
:::

:::{prf:proof}
The actual force derivative and phase derivative proved above give, in
each of those perpendicular real coordinates,

$$
\Re w=t\left[\sum_{j\ne1}\delta z_j-(N-1)\delta z_1\right],
\qquad \Im w=b\delta z_1.
$$

Independent isotropic tangent rows give the displayed variances and
cross covariance. The two chosen perpendicular coordinates are independent.
The unscaled block has trace $t^2N(N-1)+b^2$ and determinant
$t^2b^2(N-1)>0$. For its two positive eigenvalues,
$\lambda_{\min}=\det/\lambda_{\max}\ge\det/\operatorname{tr}$,
giving the bound. The exact conditional O law has covariance $\chi I$
and $z=cv_1+q\xi$, so its residual standard deviation is $q\sqrt\chi$.
The identity $\|wc^\dagger+cw^\dagger\|_{\rm HS}^2=2\|w\|^2$
proves the matrix-coordinate factor. Direct substitution gives the number.
:::

:::{prf:remark} Derived degenerate native regimes
:label: rem-npc-degenerate-native-regimes

For $\nu=0$ the native viscous color is invalid, so this projector branch
is absent. For $q\chi=0$ the fixed-terminal-position innovation has no
residual direction. At $\kappa=0$ the color is real; the tag's rank-four
construction reduces to the two-dimensional real projective tangent and
does not give the two noncommuting complement curvature matrices above.
An extra clone-deletion mask removes the deliberately revived tag.
The default B1 reader is a distinct stage and receives no B2 support result.
A threshold exceeding the constructed $R$ rejects this neighborhood;
because the O inputs and revival jitters are unbounded, any finite larger
threshold may instead be tested with its actual larger $R$, with its
different evaluated probability and source profiles. None of these tests
changes the configured force or erases a parameter.
:::

(sec-npc-native-source-action)=
## 4. Native action response on the composite connection

:::{prf:theorem} Native composite link derivatives and their weak action score
:label: thm-npc-native-connection-action-response

Use the actual geometry fiber $(\mathcal H,Y=y)$ and its conditional O law
in {prf:ref}`thm-native-ym-terminal-conditional-source`. Retain a finite
existing record graph and its available native color projectors. Derive
the composite transports above on its actual edges. On a smooth branch
with $|c_i^\dagger c_j|\ge\eta>0$, the actual native O source has

$$
\delta_fP_i=(\delta_fc_i)c_i^\dagger+c_i(\delta_fc_i)^\dagger,
\quad
\delta_f\omega=[\delta_fP,dP]+[P,d(\delta_fP)],
\quad
\delta_f\mathcal F=d(\delta_fP)\wedge dP+dP\wedge d(\delta_fP).
$$

The color derivative is the fully reevaluated count/row formula in
{prf:ref}`lem-native-ym-b2-color-tangent`. The finite-link derivative is
determined by

$$
\delta_f S_{j\leftarrow i}
=(\delta_fP_j)(2P_i-I)+(2P_j-I)\delta_fP_i.
$$

Its polar derivative satisfies

$$
\|\delta_fU_{j\leftarrow i}\|
\le\frac{\|\delta_fP_i\|+\|\delta_fP_j\|}{\eta}.
$$

For every bounded smooth composite-link cylinder $O$ whose native
innovation pullback has bounded derivative and is supported away from
the actual availability boundaries, its source response is the proved
connection variation

$$
\int\delta_fO\,d\mu^{\mathcal H,y}
=\int O\,f\cdot(\xi-\ell)\,d\mu^{\mathcal H,y}.
$$

For a retained composite connection descriptor $D$, let $V_f(D)$ be the
conditional mean of its actual tangent, and
$j_f(D)=E[f\cdot(\xi-\ell)\mid\mathcal H,y,D]$. Then

$$
\int DO[V_f],d\mu_D^{\mathcal H,y}
=\int O j_f\,d\mu_D^{\mathcal H,y},\qquad
\|j_f\|_2\le\sqrt\chi|f|.
$$

Thus the native weak connection/action identity is discharged for this
specific composite connection, including the fiber averaging of its actual
tangent. It is not an arbitrary independent variation of every link.
For bounded masked cylinders the exact density-ratio response remains
valid; crossing a force/overlap threshold retains its boundary response.
The coarser geometry fiber retains the already calculated preparation
posterior correction.
:::

:::{prf:proof}
Differentiate the actual projector and its proved connection formulas.
Differentiate the projection products to obtain $\delta S$.
To bound the polar derivative write $S=UH$, $H=(S^\dagger S)^{1/2}\ge\eta I$,
and let $K=U^\dagger\delta U$ be anti-Hermitian. Differentiation and
subtraction of adjoints gives

$$
KH+HK=U^\dagger\delta S-(\delta S)^\dagger U.
$$

Its unique solution is the convergent integral of
$e^{-tH}[U^\dagger\delta S-(\delta S)^\dagger U]e^{-tH}$ over
$t\ge0$. Its norm is at most $\|\delta S\|/\eta$.
Since $\delta U=UK$ and $\|2P-I\|=1$, this gives the bound.
The smooth test is a pullback through this actual deterministic native
map. Apply the exact conditional Gaussian translation identity in
{prf:ref}`thm-native-ym-conditional-weak-variation` to that pullback.
Disintegrate its chain-rule derivative and its Gaussian score against
$D$ to obtain the conditional-mean-tangent identity. Conditional
expectation contracts the $L^2$ norm, and the residual Gaussian has
$E[(f\cdot(\xi-\ell))^2]=\chi|f|^2$, giving the bound. A hard
availability mask cannot be differentiated only inside its strata;
use the exact finite likelihood ratio for its bounded cylinder instead.
:::

:::{prf:corollary} Explicit composite Wilson variation and native difference
:label: cor-npc-native-wilson-variation

For the determined composite loop $H_P$ and the actual finite coefficients
$\beta_P$, let

$$
S_W=\sum_P\beta_P(1-\tfrac13\Re\operatorname{Tr}H_P).
$$

Its native derivative is the ordered product-rule derivative and obeys

$$
|\delta_fS_W|
\le\sum_P|\beta_P|\sum_{e=(i,j)\in\partial P}
 \frac{\|\delta_fP_i\|+\|\delta_fP_j\|}{\eta_e}.
$$

For count normalization on a complete valid branch with $\delta_c>0$,
the actual prepared-position constant $C_p$ in
{prf:ref}`thm-nyc-global-b2-inverse` gives

$$
\|\delta_fP_i\|
\le2q\chi\left[\frac{2\nu C_p}{\delta_c}+|\kappa|\right]|f|.
$$

This bound concerns the Gaussian force derivative; it does not require
the inverse-map test $\beta_->0$. For zero threshold retain the actual
$|F_i^{\rm visc}|>0$ in the denominator. Row normalization retains its
actual quotient derivative instead of this count bound.

On the native innovation chart the actual action is the conditional
Gaussian action $S_G=|\xi-\ell|^2/(2\chi)+\mathrm{const}$, so its exact
source-coordinate discrepancy from this particular Wilson readout is

$$
\mathcal E_f(\xi)
=f\cdot(\xi-\ell)-\delta_fS_W(\xi).
$$

No equality of $S_W$ and the native coarse connection action follows from
the Wilson derivative alone. Its required coarse discrepancy includes the
descriptor reference-volume divergence and the conditional mean tangent
in {prf:ref}`thm-npc-native-connection-action-response`. Both the native
score and the ordered Wilson derivative are now explicitly obtained from
the same original record, rather than from independent fitted links.
:::

:::{prf:proof}
Differentiate each loop without exchanging noncommuting factors. Since all
other factors are unitary, its derivative norm is at most the sum of its
edge derivative norms. The normalized real trace is bounded by the matrix
norm; use the preceding polar bound. The count force is
$-\nu\mathcal G(z)$ and its global Jacobian norm is at most $2\nu C_p$,
as proved from the real Gaussian pairs. Multiply by $q\chi|f|$ and use
the actual normalized-color derivative and
$\|\delta P_i\|\le2\|\delta c_i\|$. This proves the bound. A constant
conditional coordinate source has zero innovation-volume divergence and
$\delta_fS_G=f\cdot(\xi-\ell)$, giving the displayed exact difference.
For the coarser descriptor, use its disintegrated tangent and score from
the preceding theorem; their volume divergence is an additional change
of coordinates, not removed by the fine-chart computation.
:::

(sec-npc-physical-correspondence)=
## 5. Physical transport requirements and an actual finite-action boundary

:::{prf:theorem} The native centroid innovation does not supply same-slice composite curvature
:label: thm-npc-centroid-fiber-cancellation

Use a single actual matched B2 population, the fixed terminal-position
geometry fiber, fixed phase calibration, and any count/row Gaussian
viscosity in {prf:ref}`def-npc-complete-record`. Build a same-slice
composite graph whose support and geometric weights are fixed on that
fiber. Let $D$ retain only its projector conjugation invariants, including
all closed composite-loop traces and their actual available-face masks.
It does not retain a noninvariant component frame or the full complex
color determinant.

For a common O-source direction $f_i=g\in\mathbb R^3$ at every row,
the complete finite conditional source law of $D$ is unchanged:

$$
\mu_{D,\theta}^{\mathcal H,y}=\mu_{D,0}^{\mathcal H,y},\qquad
j_f(D)=0.
$$

In particular the strictly nonzero native determinant centroid innovation
proved in the native determinant chapter is not a nonzero curvature
innovation of this same-slice composite algebra. The native noncommuting
curvature proved above uses other actual innovation directions. Loops
joining different recorded stages or times do not receive this
same-slice cancellation without retaining their different source maps.
:::

:::{prf:proof}
The conditional source translates every B2 velocity by the same
$\theta q\chi g$ and every B2 position by $\theta aq\chi g$.
All spatial and velocity differences are unchanged, so both actual
viscosity normalizations and force-threshold masks are unchanged.
Every color is multiplied by the same diagonal unitary

$$
T_\theta=\operatorname{diag}
 (e^{i\kappa\theta q\chi g_1},e^{i\kappa\theta q\chi g_2},
  e^{i\kappa\theta q\chi g_3}).
$$

Thus $P_i\mapsto T_\theta P_iT_\theta^\dagger$ and each determined
same-slice composite link and loop transforms by common conjugation.
Its trace, overlap modulus and all these retained masks remain unchanged.
The terminal geometry was already fixed. Hence the descriptor is
pointwise unchanged under the exact conditional innovation translation,
which proves equality of its finite laws and zero score.
Equivalently, conditional on $(\mathcal H,y)$ the Gaussian centroid of
$\xi$ is independent of its centered relative array. The retained
descriptor depends only on that relative array, whereas the common-source
score is its centered centroid coordinate. Their conditional covariance
is zero. A color determinant instead acquires
$\det T_\theta=e^{i\kappa\theta q\chi\sum_ag_a}$, so its previously
proved component is a distinct native channel.
:::

:::{prf:theorem} Native edge increments required by a spacetime connection
:label: thm-npc-native-spacetime-increment

Let $\ell_e>0$ be the actual calibrated length of an available record edge,
and $\delta_e=\|P_j-P_i\|$. The determined composite transport obeys

$$
\delta_e\le\|U_e-I\|\le\sqrt2\,\delta_e,
\qquad
\|\log U_e\|=\arcsin\delta_e\quad(\delta_e<1).
$$

Therefore a claimed finite first-order physical connection coefficient on
shrinking actual edges requires $\delta_e=O(\ell_e)$ on that same
record. The coefficient error, before any unproved limiting identification,
is explicitly

$$
\left\|\frac{U_e-I}{\ell_e}
           +\frac{[P_i,P_j-P_i]}{\ell_e}\right\|
\le\left[1+\frac1{\eta_e(1+\eta_e)}\right]
       \frac{\delta_e^2}{\ell_e}.
$$

All terms are computable from the actual native colors, masks, edges and
physical calibration. The configuration-chart connection and its positive
native curvature do not themselves establish the required spatial
increment rate. In particular recording time and an independently declared
fourth position coordinate remain distinct unless their reconstruction is
proved for the same native descriptors.
:::

:::{prf:proof}
The first theorem gives $\delta=\sin\theta$ and
$\|U-I\|=2\sin(\theta/2)$, so their ratio is
$1/\cos(\theta/2)\in[1,\sqrt2]$. Its principal logarithm has
eigenvalues $0,i\theta,-i\theta$, giving its norm. Divide the already
proved first-order comparison error by the actual length. If
$U-I=O(\ell)$, the lower norm inequality gives the necessary native
projector increment rate; no assumption on the unknown continuum law is
substituted for it.
:::

:::{prf:theorem} Reducible holonomy and the finite Haar-Wilson boundary
:label: thm-npc-finite-action-boundary

For the canonical composite connection on any actual closed record path
based at $i$, its complete holonomy satisfies

$$
H_PP_iH_P^\dagger=P_i.
$$

Thus all its based holonomies lie in the native stabilizer
$S(U(1)\times U(2))\subset SU(3)$. Their rank-two part can be non-Abelian,
as already proved, but this route carries a covariantly constant line.
Each single direct-rotation edge also lies in the proper set

$$
\mathcal Z=\{U\in SU(3):\det(U-I)=0\}.
$$

This set has $SU(3)$ Haar measure zero. Consequently the law of any
nonempty finite list of these native composite links is singular with
respect to product Haar measure. It cannot equal an unrestricted finite
$SU(3)$ Wilson density $Z^{-1}e^{-S_W}\prod_e d\mathrm{Haar}(U_e)$
with finite face coefficients: that latter density is positive everywhere.

This is a characterization of the **determined composite route** and its
actual probability law, not a theorem that no non-Abelian connection can
emerge from the full gas. It also does not exclude a continuum limit with
a separately proved descriptor/reference-volume correspondence. Such a
correspondence must account for the native line-preservation constraints
and the action on their actual support.
:::

:::{prf:proof}
Every direct-rotation edge maps its source line onto its target line.
Compose around the ordered closed boundary to obtain line preservation
at the base. The first theorem gives the eigenvalue one of every such
edge, placing it in $\mathcal Z$.

Here is a direct Haar null-set argument. If $U$ is Haar distributed,
its first column has the rotation-invariant distribution on the complex
unit sphere in $\mathbb C^3$, equivalently $Z/|Z|$ for three independent
nondegenerate complex normal variables. This follows from the transitive
$SU(3)$ action and invariance: averaging any fixed unit vector over Haar
gives the same invariant sphere law as averaging that Gaussian direction.
In particular $P(U_{11}=0)=0$.
For almost every fixed $U$, left multiply by the diagonal torus

$$
D(\theta,\varphi)=\operatorname{diag}
(e^{i\theta},e^{i\varphi},e^{-i(\theta+\varphi)}).
$$

The determinant $\det(DU-I)$ is the nonzero Laurent polynomial

$$
U_{11}e^{i\theta}+U_{22}e^{i\varphi}
 +U_{33}e^{-i(\theta+\varphi)}
-\overline U_{11}e^{-i\theta}-\overline U_{22}e^{-i\varphi}
-\overline U_{33}e^{i(\theta+\varphi)}.
$$

The six monomials have different exponents, and $U_{11}\ne0$ makes
this polynomial nonzero. A nonzero Laurent polynomial in two torus
variables has a zero set of area zero: collect its finitely many powers
of the first variable; except at the finitely many zeros of a nonzero
coefficient polynomial in the second, the first-variable polynomial has
only finitely many zeros. Fubini proves the assertion. Haar left invariance
and a second application of Fubini therefore give
$P(\det(U-I)=0)=0$. The actual composite marginal is concentrated on
that set; a product-Haar law gives it zero mass in any specified edge.
Finally finite Wilson coefficients give a bounded real action on a finite
compact link set, so its exponential is strictly positive with a finite
positive normalizer. It is equivalent to product Haar and hence singular
to the native composite link law.
:::

:::{prf:remark} Remaining identification after these native deductions
:label: rem-npc-remaining-identification

The native projector algebra now supplies a determined non-Abelian
line/complement transport, its local frame law, curvature, positive native
innovation regime and weak likelihood first variations. Its finite action
support is also characterized. These results advance beyond the common-frame
orbit algebra and beyond a Wilson Taylor expansion.

They do not identify an irreducible physical $SU(3)$ Yang--Mills field. The
remaining step must derive the actual spacetime increment and temporal
scaling of an appropriate native transport, characterize its full
conditional descriptor density/reference volume and show the target action
and first variations on that support. If the canonical composite route is
used, its covariantly constant line and finite singular support must survive
or be accounted for in that proof. If another already prescribed IG/IA
transport is used, its actual map must be derived rather than replaced by
these composites. The original force, cloning law, masks, unbounded noise,
physical-time reconstruction and parameter regimes remain unchanged.
:::
