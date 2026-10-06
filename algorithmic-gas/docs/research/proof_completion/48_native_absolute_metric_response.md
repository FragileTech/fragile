# Uncut coupled action response for the existing absolute covariance metric

## 1. The distinct absolute-ridge configuration

:::{prf:definition} Complete absolute-metric Einstein–Hilbert register
:label: def-namr-register

Retain every parameter, stage, source, initial law, landscape, boundary,
donor, cloning, arithmetic, cache and recording field of
{prf:ref}`def-ncma-register`. This chapter consumes the EXISTING
`RidgeScale::Absolute` covariance-metric choice, with its default
`MetricPolicy::Clipped`. It is a distinct configured component tuple; it
does not silently change the relative-trace Einstein–Hilbert preset.
All reward, curvature, volume, graph-force, curl and thermostat formulas
remain the same native ones. The only altered component tag relative to
that preset is explicitly the absolute metric scale.

The absolute covariance regularization is

$$
 M_i=C_i+\rho I,\qquad \rho>0,\qquad
 \varepsilon=d\epsilon_{\mathcal E}.
$$

If $\lambda_{i,j}$ are the eigenvalues of $M_i$, the executed metric
eigenvalues are

$$
 g_{i,j}
 =\left[
 {\mathbf1_{\{\lambda_{i,j}>\varepsilon\lambda_{i,\max}\}}
       \over\lambda_{i,j}}
 \right]_{b_-}^{b_+},                                  \tag{NAMR.1}
$$

where an absent lower bound means $b_-=0$ and an absent upper bound
means $b_+=\infty$. The validated finite bounds retain $0\le b_-\le b_+$.
Default values are $\rho=10^{-5},b_-=10^{-6},b_+=\infty$.
The comparison epsilon is the actual run-precision epsilon; it is not set
to zero. Site ranks retain their separate positive f64 cutoff and their
original projection/duplicate/error policies. In particular every actual
rank-one path in Chapter NCMA remains present.

The original $u_i,v_i,D_{ij},w^R_{ij},R_i,r_i,H$ are precisely (NCMA.2),
including $f_R,f_V>0$, the inverse-distance and row-sum floors, and the
unit-coordinate density volume. All the graph-viscosity weights and both
separately evaluated Boris kicks consume the actual absolute-metric
post-cloning graph. The original independent isotropic O innovation has
$q^2=T(1-e^{-2\gamma h})$, $t=h/2$, $s=tq>0$.

The default unbounded/all-alive, zero-potential, zero-jitter, no-cap,
no-extra-position-noise restrictions, clone regularizer $\varepsilon_c=0$,
saturation $s_c=1$, period twenty and complete mutual matching law remain
as in Chapter NCMA. General fixed configured $\varepsilon_c\ge0,s_c>0$
retain the explicit native bounds from that chapter. Original finite
arithmetic, resource errors and other variant tags keep their actual
outcomes. Real-coordinate formulas retain every positive comparison
constant; finite floating draws or rounded payloads are not assigned a
continuous-Gaussian derivative without their comparison.
:::

## 2. Global original action bounds

:::{prf:theorem} Absolute native action bounds over every graph and rank branch
:label: thm-namr-global-action-bound

For every successfully defined configuration in
{prf:ref}`def-namr-register` put

$$
 G_+(\rho)=\min\{b_+,\max(\rho^{-1},b_-)\},\quad
 U_\rho=1+{|\log f_R|+d\log^+G_+\over2d},\quad
 V_\rho=\sqrt{\max(f_V,G_+^d)} .
$$

Then

$$
 0\preceq g_i\preceq G_+I,\qquad
 |u_i|\le U_\rho,\quad 0<v_i\le V_\rho,\quad
 |H|\le H_*:=4|\lambda_R|(d-1)N U_\rho V_\rho .
                                                               \tag{NAMR.2}
$$

For $d=1$ the literal action is zero. The bounds hold across all actual
pseudoinverse deletions, metric clamps, determinant floors, ranks, CSR
degrees and graph branches. They do not require geometric coercivity,
stationarity, compact coordinates, Gaussian clipping or a regularity
assumption on the native law. Every absolute moment of an existing
defined action is finite, with bound $H_*^p$.

If the native execution reports an error instead of an action, the
unnormalized successful-action measure obeys the same bound; the error
record is retained separately. No success normalization or boundary-death
reinterpretation is introduced.
:::

:::{prf:proof}
All eigenvalues of $M_i$ are at least $\rho$. Its retained inverse
eigenvalues therefore lie in $(0,\rho^{-1}]$; its deleted values are
exactly zero. Applying the original bounds gives (NAMR.2)'s matrix
inequality even when the unbounded covariance makes a mode cross the
relative-eigenvalue cutoff. Consequently $0\le\det g_i\le G_+^d$.
The two actual determinant floors bound the displayed logarithm and
volume. Nonnegative row-normalized inverse-distance weights have row
mass at most one, also on the original row-sum-floor branch. Thus
$|R_i|\le4(d-1)U_\rho$, and summation proves the action bound.
The bound depends only on the original parameter tuple and population,
not on the positions or graph. The moment and error-measure statements
are direct consequences of this pointwise bound.
:::

:::{prf:theorem} Full uncut Gaussian, thermal and native reward-scale response
:label: thm-namr-full-uncut-response

Fix an all-alive entering state and the actual full matching/accepted-plan
preparation of {prf:ref}`thm-ncma-coupled-position-response`. At fixed
absolute metric parameters the original terminal action is bounded.
For any existing parameter path on which the computed preparation means
$m_C$, scale $s>0$ and probabilities have derivatives, its uncut
successful-action integral has

$$
\begin{split}
 \partial_\theta E_{\rm succ}H
 =E_D\sum_C\bigg[&
 \pi'_C E_G H(m_C+sG)\\
 &+\pi_C E_G H(m_C+sG)
 \left({m'_C\cdot G\over s}
       +{s'\over s}(|G|^2-Nd)\right)\bigg] .
\end{split}                                             \tag{NAMR.3}
$$

Here the integrand means the original action on successful terminal
evaluations and zero contribution from errors to that integral, without
assigning errors an action in the transition. The terminal geometry map,
its rank tests, pseudoinverse cutoff and masks remain fixed functions of
physical positions under this parameter variation. In particular (NAMR.3)
holds for temperature variation and original Gaussian source/mean
variation through the complete native preparation. If an existing
parameter also changes the terminal metric/readout, its direct response
must be included; the ridge response is derived in the next section.

For the original origin-start first step, $Y=sG,m=0$ for every absolute
ridge, all gates vanish and B1 is zero. B2 is still executed. The full
uncut thermal response is

$$
 \partial_T E_{\rm succ}H
 ={1\over2T}E_{\rm succ}
           \left[H\sum_i(|G_i|^2-d)\right].             \tag{NAMR.4}
$$

For any all-alive initial law and finite native horizon, the original
reward-scale variation $\lambda_R>0$ has the full finite-history response
of {prf:ref}`thm-ncma-finite-history-feedback`. Because the absolute
terminal action is now bounded, this applies to the action itself,
adding its exact direct term $H/\lambda_R$. If $\Psi$ is a fixed bounded
function of the phase/mark history and $H_n$ is its actual final action,

$$
 \partial_{\lambda_R}^+E_{\rm succ}[\Psi H_n]
 = E_{\rm succ}[\Psi H_n]/\lambda_R
   +\text{the chronological pattern terms of (NCMA.24)},
$$

with absolute bound

$$
 \|\Psi\|_\infty H_*
       \left(\lambda_R^{-1}
       +|\mathcal I_n|\,2N B_N(\lambda_-)\right)         \tag{NAMR.5}
$$

on a positive compact scale interval. The $H_*$ in this estimate may
be its supremum over that interval. This is the actual coupled action
and selection feedback, rather than an independent action sample.
:::

:::{prf:proof}
At fixed terminal metric parameters, the original action/error function
is a bounded Borel function of the final physical positions, even where
the tessellation or pseudoinverse jumps. The signed Gaussian position
kernel derivative is integrable in total variation by (NCMA.5)--(NCMA.6).
The boundedness in (NAMR.2) permits its uncut integration with no
truncation-removal assumption. This proves (NAMR.3) and retains every
terminal retessellation through the actual law.

At the coincident origin, the complete first B stage and gate preparation
are zero, so the original A2 positions are $sG$ with $s=tq$ and
$\partial_Ts/s=1/(2T)$. The second kick occurs after A2 and does not move
these positions. Equation (NAMR.3) gives (NAMR.4).
For reward scale, the conditional physical trajectory given all patterns,
Gaussian sources and matching contexts is independent of $\lambda_R$.
Its original action is exactly linear in that scale. Apply the full
chronological source/pattern derivative already proved in Chapter NCMA
and add this direct linear derivative. Both are bounded by (NAMR.2)
and the native, regularizer/saturation-dependent gate bound. This proves
(NAMR.5). Parameter-dependent stored bookkeeping observations retain
their own direct terms; acceptance uniforms are integrated in their
original Bernoulli probabilities in this total-variation statement.
:::

## 3. A full ridge response with the actual spectral interface retained

:::{prf:lemma} Original dilation freezes terminal cutoff tests
:label: lem-namr-ridge-pullback

Use the original open-domain real-coordinate geometry and absolute
clipped metric. Set $Y=\sqrt\rho Z$. Its actual graph, duplicates,
principal-rank tests and mesh branches are those of $Z$, dilated by
$\sqrt\rho$. Covariances satisfy $C_i(Y)=\rho C_i(Z)$.
Define the actual retained/deleted matrix at unit ridge by

$$
 B_i(Z)=Q_i\operatorname{diag}\left(
 {\mathbf1_{\{1+\lambda_j(C_i(Z))>
       \varepsilon(1+\lambda_{\max}(C_i(Z)))\}}
       \over1+\lambda_j(C_i(Z))}\right)Q_i^T .
$$

Then the original metric is EXACTLY

$$
 g_{\rho,i}(\sqrt\rho Z)
 =Q_i\operatorname{diag}
        ([\beta_{i,j}(Z)/\rho]_{b_-}^{b_+})Q_i^T .      \tag{NAMR.6}
$$

The hard terminal deletion tests are independent of $\rho$ in $Z$.
The remaining eigenvalue-clamp crossings are continuous. At their
one-sided branches,

$$
 g'_{i,j}=
 \begin{cases}-g_{i,j}/\rho,&
       \beta_{i,j}>0,\ b_-<\beta_{i,j}/\rho<b_+,\\
                0,&\text{strict fixed-clamp or deleted branch}.
 \end{cases}                                           \tag{NAMR.7}
$$

Writing $\widehat H_\rho(Z)=H_\rho(\sqrt\rho Z)$, every original defined
branch obeys

$$
 |\partial_\rho u_i|\le{1\over2\rho},\quad
 |\partial_\rho v_i|\le{d\,v_i\over2\rho},\quad
 0\le\partial_\rho D_{ij}\le D_{ij}/\rho,\quad
 \sum_j|\partial_\rho w_{ij}^R|\le1/\rho .              \tag{NAMR.8}
$$

Consequently on every positive compact ridge interval
$[\rho_-,\rho_+]$,

$$
 |\partial_\rho\widehat H_\rho|
 \le {2|\lambda_R|(d-1)(d+3)\over\rho_-}
                       (U_*+1)N V_*=:J_*<\infty,     \tag{NAMR.9}
$$

where $U_*,V_*$ are the explicit suprema from (NAMR.2).
All actual rank-one branches and spectral deletions are included.
:::

:::{prf:proof}
Open Delaunay geometry and the principal-rank comparison scale
homogeneously. The code compares both singular values and its positive
tolerance by the same dilation factor. The displacement covariance
is quadratic, including a zero covariance, so
$C_i(\sqrt\rho Z)+\rho I=\rho(C_i(Z)+I)$.
Its eigenvectors and relative-eigenvalue cutoff tests are unchanged
in the variable $Z$. Applying the original ABSOLUTE bounds gives
(NAMR.6), without omitting a deleted mode.

Each retained inverse eigenvalue inside its clamps has derivative
$-g_{i,j}/\rho$; fixed clamp or deleted values have derivative zero.
These scalar functions are locally Lipschitz, with their original
one-sided derivatives at equality. Their logarithmic derivative has
magnitude at most $1/\rho$ whenever positive. If the determinant is
zero, its original positive floor is locally constant; otherwise
sum these eigenvalue derivatives. This gives the first two bounds
in (NAMR.8), retaining both determinant floors.

More precisely the eigenvalues of $\rho g_i$ are either a fixed
retained $\beta_{i,j}$ or $\rho b_-$ or $\rho b_+$.
Their derivatives lie between zero and their value divided by
$\rho$, also for a mode deleted to zero before its lower clamp.
Thus
$0\preceq\partial_\rho(\rho g_i)\preceq(\rho g_i)/\rho$.
In $Z$ coordinates the original intrinsic edge square is
$D_{ij}=z_{ij}^T\rho(g_i+g_j)z_{ij}/2$. The matrix inequality proves
the bound for $D_{ij}$, including zero distance.
The original inverse-distance floor and its row-sum floor give
$|(\log k_{ij}^R)'|\le1/(2\rho)$ and the displayed row bound.
Differentiate the original Laplacian and volume allocation as in
the proof of (NCMA.10). Their pointwise bound is
$2|\lambda_R|(d-1)(d+3)(U_\rho+1)\sum_iv_i/\rho$,
which gives (NAMR.9).

Only proof coordinates were changed. The physical metric, action,
source and comparison constants remain the executed ones. In particular
the argument does not claim that its fixed-physical-position ridge
derivative has no spectral interface term.
:::

:::{prf:theorem} Complete uncut ridge likelihood/action correspondence
:label: thm-namr-uncut-ridge-response

Fix the actual all-alive entering state and donor/plan preparation
in Chapter NCMA. At a ridge value where the finite computed preparation
probabilities and means have derivatives, define

$$
 \mathcal B_\rho(Y)
 =\left.\partial_\rho\widehat H_\rho(Z)\right|_{Z=Y/\sqrt\rho}.
$$

For the absolute clipped metric, the uncut successful-action integral has

$$
\begin{split}
 \partial_\rho E_{\rm succ}H_\rho
 =E_D\sum_C\bigg[
 &\pi'_C E_GH_\rho(Y)
 +\pi_C E_G\mathcal B_\rho(Y)\\
 &+\pi_C E_GH_\rho(Y)\left\{
 { (m'_C-m_C/(2\rho))\cdot G\over s}
 +\left({s'\over s}-{1\over2\rho}\right)(|G|^2-Nd)
                         \right\}\bigg],
 \qquad Y=m_C+sG .                                    \tag{NAMR.10}
\end{split}
$$

For a ridge-only path $s'=0$; all native preparation and gate derivatives
still remain. This formula retains the ENTIRE actual terminal spectral
boundary response. It is not obtained by integrating the derivative
inside a frozen pseudoinverse stratum.

At the original origin start, every $m_C$ and gate derivative vanishes
for the positive ridge interval, giving the positive complete regime

$$
 \partial_\rho E_{\rm succ}H_\rho(sG)
 =E_G\mathcal B_\rho(sG)
   -{1\over2\rho}E_{\rm succ}
             [H_\rho(sG)(|G|^2-Nd)] .                 \tag{NAMR.11}
$$

No actual terminal graph/rank branch or unbounded Gaussian innovation
is excluded. Default determinant floors and clamps are retained.
The same one-sided formulation applies at continuous clamp/floor
crossings. Strict metric-policy rejection or a geometry/error rule
which does not share the declared real-coordinate dilation retains
its actual changed success mask and requires that additional boundary
term; it is not silently assigned (NAMR.10).
:::

:::{prf:proof}
For each native preparation component, make the exact Gaussian
change of variables $Y=\sqrt\rho Z$ in its full position integral.
The $Z$ density is Gaussian with mean $m_C/\sqrt\rho$ and scale
$s/\sqrt\rho$. Its logarithmic derivative evaluated at
$Z=(m_C+sG)/\sqrt\rho$ is

$$
 {(m'_C-m_C/(2\rho))\cdot G\over s}
 +\left({s'\over s}-{1\over2\rho}\right)(|G|^2-Nd).
$$

The direct observable derivative is $\mathcal B_\rho$.
In these coordinates the original terminal graph and spectral
deletion tests are fixed. Clipped policy does not turn a repaired
metric into an error, and the declared open real-coordinate
mesh/rank/resource rules have the same success event under dilation;
every subsequent geometry field is finite by the metric bound and
the configured positive floors. Reported mesh/graph errors remain
their original zero contribution to the successful-action integral
with the same success mask in these proof coordinates.

Equations (NAMR.2) and (NAMR.9) give bounded observable and derivative
envelopes uniformly on a positive compact ridge interval. Gaussian
linear/quadratic scores have all moments. The mean-value formula and
dominated differentiation therefore justify the full uncut integral,
including the finite native pattern and matching sums. Differentiate
their products by (NCMA.6). This proves (NAMR.10).
At the origin $Y=sG$ with ridge-independent $s$, zero first-kick
preparation and equal-fitness zero gates. B2 remains executed after
the position drift. Substitution proves (NAMR.11).

The Gaussian-density term in the changed variables accounts for
movement across every original fixed-position spectral threshold.
The terminal hard deletion is consequently not ignored, even
though it no longer moves in $Z$. The last scope statement retains
an error/Strict-policy mask whose change is not covered by this
dilation computation.
:::

## 4. Literal preparation jumps and a nonzero included action

:::{prf:proposition} Exact conditional kernel jumps at preparation thresholds
:label: prop-namr-preparation-jump

At a ridge value where the actual finite entering or post-cloning
geometry changes a pseudoinverse branch before O, retain its exact
left and right preparation limits. If they exist, the conditional
position-kernel jump is the signed density

$$
 K_+(y)-K_-(y)
 =E_D\sum_C[
 \pi_{C,+}\varphi_{s_+}^{\,Nd}(y-m_{C,+})
 -\pi_{C,-}\varphi_{s_-}^{\,Nd}(y-m_{C,-})].           \tag{NAMR.12}
$$

It can be nonzero; a classical derivative is not assigned at a
nonzero jump. Strict-policy rejection and recorded error outcomes
have their corresponding retained atomic jump terms.
This fully separates earlier physical preparation discontinuity
from the terminal interface already included in (NAMR.10).
:::

:::{prf:proof}
Each one-sided conditional native pattern prepares its own exact
mean and scale and then executes the unchanged Gaussian O/A2
position map. Taking its one-sided Gaussian density limits
in $L^1$ and summing their actual finite mixture proves
(NAMR.12). An error outcome is a distinct measure component.
A nonzero measure jump precludes a finite classical derivative.
The origin start avoids these preparation jumps because all
copied/initial velocities and B1 forces are zero for every
positive ridge; no terminal-jump omission is inferred from that fact.
:::

:::{prf:proposition} A nonzero native absolute-ridge action on original Gaussian records
:label: prop-namr-nonzero-action

Choose the existing absolute ridge $\rho=1$, default $b_-=10^{-6}$,
no upper clamp, $\lambda_R=1,d=2,N=3$, and the actual triangle
$Y_0=(0,0),Y_1=(1,0),Y_2=(0,1)$. All cutoff, floor and rank
branches have strict margins. Put

$$
 k_a=(\sqrt{20/33}+10^{-8})^{-1},\qquad
 k_b=(\sqrt{10/11}+10^{-8})^{-1}.
$$

The original allocated action is exactly

$$
 H=\left({1\over3}
       -{2k_a\over\sqrt{11}(k_a+k_b)}\right)\log(11/9)>0 .
                                                               \tag{NAMR.13}
$$

It is nonzero on an open full-rank event of positive probability
under every original nondegenerate Gaussian terminal preparation.
Together with the collapsed-cloud limit $H\to0$, this proves a
finite, strictly positive action variance at the actual origin-start
Gaussian transition for this included absolute-metric configuration.
Thus its bounded-action regime is not limited to a constant or zero action.
It does not identify the graph conformal Laplacian with continuum
Einstein curvature.
:::

:::{prf:proof}
The literal covariances are

$$
 C_0=\tfrac12I,\quad
 C_1=\begin{pmatrix}1&-1/2\\-1/2&1/2\end{pmatrix},\quad
 C_2=\begin{pmatrix}1/2&-1/2\\-1/2&1\end{pmatrix}.
$$

Adding the configured absolute ridge gives $v_0=2/3$,
$v_1=v_2=2/\sqrt{11}$ and
$u_0-u_1=u_0-u_2=\log(11/9)/4$. The original intrinsic edge
squares are $D_{01}=D_{02}=20/33,D_{12}=10/11$.
Every row sum is above its floor. The native paired
action identity (NCMA.22) then gives (NAMR.13).
For positivity, $k_b/k_a>\sqrt{2/3}>6/\sqrt{11}-1$.
The first inequality follows because the positive
$10^{-8}$ moves that ratio toward one. The second follows
by squaring positive sides:
$8\cdot1089>3\cdot2809$. This is an exact rational
certificate of the required strict sign.

Continuity with strict branches gives the open positive event.
At a small homothetic full-rank cloud, the original absolute
metric tends to $\rho^{-1}I$, all $u_i$ tend to the same
constant and $H\to0$. These small noncoincident configurations
also have open Gaussian neighborhoods. A constant almost-sure
action is therefore impossible. Global boundedness proves
finite variance, and the two positive-probability separated
action events prove it is strictly positive.
:::

## 5. Parameter and endpoint scope

The relative-trace preset's actual divergence in Chapter NCMA is
not removed by this proof. The positive uncut action regime here
is its existing **absolute** covariance-metric configuration,
with all native graph, force, cloning, thresholds and Gaussian
sources retained. The global bound applies to every original
rank and spectral branch. At fixed metric parameters the full
uncut Gaussian/temperature and finite-history reward-scale
responses close without an action-tail assumption. The ridge
pullback also closes the full terminal spectral interface at
the actual origin start and on every computed smooth native
preparation branch.

Other configurations retain the explicit changes they execute:
relative trace is not absolute; Strict metric rejection is not
clipping; a cap, different noise factor, position boundary,
landscape, geometry schedule or jitter changes the corresponding
preparation law; failed execution remains an error record.
The derivative of a fixed finite arithmetic payload is not
the real-coordinate derivative. All these tags and parameters
remain in the register, and the first-variation formulas apply
only after their consumed position/preparation law is evaluated.

This identifies an actual bounded nontrivial gravitational-action
regime and its complete original weak source/metric response.
It does not insert a metric-dependent thermal precision into the
isotropic preset, equate that raw Gaussian log density with the
graph action, replace the scalar curvature estimator by Regge or
Einstein curvature, or assert a continuum physical gravitational
field equation or stationary population law.
