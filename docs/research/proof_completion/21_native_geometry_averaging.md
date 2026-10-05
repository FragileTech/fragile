# Same-record empirical geometry and the native conformal curvature law

(sec-nga-register)=
## 1. The actual transition and geometric observations

:::{prf:definition} Empirical native geometry register
:label: def-nga-register

Retain every field of the complete execution record in
{prf:ref}`def-native-complete-execution-record`, the spatial register in
{prf:ref}`def-nsg-complete-register` and the geometric record in
{prf:ref}`def-ng-complete-geometric-record`. This chapter uses the existing
quadratic capped count or row gas, with its original donor, fitness,
acceptance, collision, both kicks and unbounded innovations. The geometry
is a passive configured observation at the terminal positions. It uses
full spatial projection, open Euclidean tessellation and the recorded
eligible-site convention. The consumed Einstein--Hilbert, graph-feedback,
metric-noise and Boris variants have different laws and are not replaced
by this passive instrument.

Conditional on the complete post-jitter/collision preparation
$\mathcal F$, the actual terminal rows are independent:

$$
Y_i=m_i+\tau Z_i,\qquad
\rho_N(x)=N^{-1}\sum_i\varphi_\tau(x-m_i),\qquad
r_N=N^{-1/d},\qquad \tau^2=t^2q^2+s^2>0.
$$

The original primitive moments $H_p$ in (NSG.1), Gaussian derivative
bounds $L_{\tau,k}$, bounded donor domain, finite cap and terminal boundary
remain parameters. No independence of the centers is assumed.
The positive results concern the real-coordinate estimators on a
successful full-dimensional Euclidean tessellation. Returned arithmetic,
graph, validity and allocation discrepancies retain their actual separate
comparison errors, as in {prf:ref}`thm-nsg-qsd-transfer`.

A one-neighborhood observation reads the root's Delaunay star and the
stars and cells of its neighbors. This includes the neighbor-covariance
metric, endpoint-metric edge lengths, named weights, cell volumes and
the configured conformal-Laplacian scalar. It does not assert that an
arbitrary globally fitted, cached or historical readout is local.
:::

(sec-nga-two-root)=
## 2. Two-root control with the original conditional environment

:::{prf:lemma} Two-root Poisson comparison and neighborhood protection
:label: lem-nga-two-root

Fix a compact spatial set $K\subset B(0,R_0)$, a center core
$\mathcal C_M=\{N^{-1}\sum_j|m_j|^p\le M^p/2\}$ and $N\ge8$.
Let $\beta$ be the positive Gaussian core bound (NSG.5), enlarged to
cover the unscaled one-unit neighborhood of $K$.
For $r_NR\le1$, a root and all its neighbor cells and stars are determined
by the sites in $B(Y_i,r_NR)$ except on an event of probability at most

$$
\eta_M(R)=C_d(1+L_{\tau,0}R^d)
              \exp[-\beta v_d(R/64)].
\tag{NGA.1}
$$

The dimension constant can be taken as
$C_d=2\,9^d(1+v_d(1))$ after increasing the harmless ball factors.
The same bound holds for a homogeneous Poisson process of intensity
$\rho_N(Y_i)$.
For two different roots conditioned at $Y_i=x,Y_k=y\in K$ with
$|x-y|>2r_NR$, their determining windows couple to two independent
homogeneous Poisson configurations, of their actual intensities
$\rho_N(x),\rho_N(y)$, with error at most

$$
\epsilon_N(R)+2\eta_M(R),\qquad
\epsilon_N(R)=\frac{L_{\tau,0}^2(2v_d(R))^2+
                         4L_{\tau,0}v_d(R)}{N}
           +2r_NL_{\tau,1}\int_{B(0,R)}|u|\,du.
\tag{NGA.2}
$$

The close-root probability, conditional on $\mathcal F$, is at most
$L_{\tau,0}v_d(2R)/N$. For alive-only geometry also require the
determining windows to lie inside the actual alive domain.
:::

:::{prf:proof}
Use the guard construction of {prf:ref}`lem-nsg-protected-star` with
root protection radius $R/8$. Its neighbors lie in $B(0,R/4)$.
Protect each candidate point in that ball with radius $R/8$ as well;
its determining window is then inside $B(0,R/2)$.
Failure for the root is bounded by $9^d e^{-\beta v_d(R/64)}$.
For a possible neighbor, condition also on its addressed position.
After deleting two labels at least $N/4$ bounded centers remain on
$\mathcal C_M$. The same empty-guard bound therefore applies.
The sum of probabilities of candidate positions in $B(0,R/4)$ is at
most $L_{\tau,0}v_d(R/4)$. A union bound over their labels proves
(NGA.1), with the displayed larger constant. For the Poisson process,
expand its finite-window Poisson count and integrate the position of
one distinguished candidate; the same intensity bound and guard
probability give the same estimate. This is a counting identity, not
an assumption that neighboring cells are independent.

For separated roots, each remaining original row contributes at most
one point to the union of the two disjoint windows. Couple that
Bernoulli point to a Poisson point process on the union, just as in
{prf:ref}`lem-nsg-local-poisson-tv`. Its probability is at most
$2L_{\tau,0}v_d(R)/N$, so summing squared probabilities gives the
first numerator of (NGA.2). Deleting two labels and using the
Gaussian derivative bound gives the other terms. A Poisson process
on disjoint windows has independent restrictions, by its finite-count
series or the product of their probability generating functions.
The independent homogeneous comparison configurations therefore have
the stated intensities. Protect their root and neighboring cells in
the two Poisson windows. On a successful window coupling those same
guard points also protect the actual configurations, so the Poisson
failure bound $2\eta_M(R)$ removes the windows. This uses no guard
estimate after conditioning on a third original row.
Finally, conditional on one root, the other root has
density at most $L_{\tau,0}$; integrating over its $2r_NR$ ball proves
the close-root bound. No realized jitter or OU innovation was bounded.
:::

(sec-nga-empirical)=
## 3. Empirical geometric observations concentrate around the same environment

:::{prf:theorem} Conditional empirical geometry law
:label: thm-nga-empirical-law

Let $F_N(x,\mathcal S)$ be a measurable one-neighborhood observation
with $|F_N|\le B$, zero for $x\notin K$. Its numerical scale and floors
may depend on $r_N$ through the exact existing estimator formulas.
Put

$$
A_N=N^{-1}\sum_i F_N(Y_i,\mathcal S_{N,i}),\qquad
\Phi_N(\mathcal F)=\int\rho_N(x)
 \mathbb E F_N(x,\mathcal S(\Pi_{\rho_N(x)}))\,dx.
\tag{NGA.3}
$$

Here all neighboring metrics and cells in $\mathcal S$ come from the
same Poisson configuration. Uniformly on $\mathcal C_M$, for the
radii and interior margins of the preceding lemma,

$$
\mathbb E[|A_N-\Phi_N|^2\mid\mathcal F]
\le C B^2\left[N^{-1}+\frac{L_{\tau,0}v_d(2R)}N
                   +\epsilon_N(R)+\eta_M(R)\right],
\tag{NGA.4}
$$

where $C=64$ is sufficient. Unconditionally add
$8B^2H_p/M^p$ to this bound.
Thus $A_N-\Phi_N\to0$ in $L^2$ by taking $N$, then $R$, then $M$
to infinity. The same assertion holds under the original QSD output
law, with its proved $O((1-a_*)^N)$ total-variation error.

If the actual center empirical law converges along a subsequence to
$\mathsf M$, bounded continuous tests of the jointly limiting local
readouts have empirical limit

$$
\int \rho_{\mathsf M}(x)
       \mathbb E F(x,\mathcal S(\Pi_{\rho_{\mathsf M}(x)}))\,dx,
\qquad \rho_{\mathsf M}=\varphi_\tau*\mathsf M.
\tag{NGA.5}
$$

The limit may be random. No unique stationary phase is silently added.
:::

:::{prf:proof}
The one-root comparison gives the conditional bias in (NGA.3) at
most $2B[\epsilon_N(R)+2\eta_M(R)]$, since averaging the tag
densities gives exactly $\rho_N$. For the second moment, separate
the $N$ diagonal pairs, costing at most $B^2/N$.
Condition each off-diagonal pair on its actual positions. On separated
positions the preceding coupling replaces their observations by
independent Poisson observations at error at most
$2B^2[\epsilon_N(R)+2\eta_M(R)]$ in expectation.
The close pairs cost at most $2B^2L_{\tau,0}v_d(2R)/N$.

The sum of the resulting independent products, including its diagonal
products, is exactly $\Phi_N^2$: the finite-array tag densities
average to $\rho_N$ in each factor. Adding those missing diagonal
products costs at most $B^2/N$. Combining with the bias bounds proves
(NGA.4) with the stated larger constant; probabilities may always be
capped at one. Outside the core $|A_N-\Phi_N|\le2B$ and
$P(\mathcal C_M^c)\le2H_p/M^p$, proving the unconditional assertion.
The selected-law claim uses the same bounded measurable pushforward
and {prf:ref}`thm-nsg-qsd-transfer`.

Gaussian convolution and its uniform derivative bounds make
$\rho_N\to\rho_{\mathsf M}$ uniformly on compact sets along the
center-law subsequence. Homogeneous Poisson windows couple continuously
as their positive intensities vary. Uniform protection on every compact
positive intensity interval removes the windows. The metric and weight
limits in {prf:ref}`thm-nsg-covariance-ridge-limits` and
{prf:ref}`cor-nsg-relative-metric-weights` hold on each determining
finite configuration. Bounded convergence proves (NGA.5) for the
bounded continuous readout tests. This argument keeps shared neighbors
and their metric correlations throughout.
:::

(sec-nga-curvature)=
## 4. The configured conformal curvature and its empirical limiting law

:::{prf:theorem} Native conformal-Laplacian and allocated-density limits
:label: thm-nga-native-curvature

Use the relative-trace neighbor covariance metric with its positive
ridge and the primitive pseudo-inverse test (NSG.11), and a positive
successful metric. `Clipped` retains its configured relative bounds;
`Strict` uses the sufficient no-repair bounds in that theorem.
Use the existing normalized `InverseRiemannianDistance` weights,
$\texttt{ConformalLaplacian}$ curvature with its own determinant floor
$f_R>0$, and $\texttt{SqrtDetMetric}$ volume with $f_V>0$.
All these values remain their original parameters.

On the same neighboring Poisson stars put

$$
u_z^*=\frac{\log\det G_z}{2d},\quad
Q_z=z^T\frac{G_0+G_z}{2}z,\quad
k_z^*=\frac1{\sqrt{\max(Q_z,10^{-8})}+10^{-8}},\quad
w_z^*=\frac{k_z^*}{\max(\sum_{v\sim0}k_v^*,10^{-12})}.
$$

The actual root curvature and its existing allocated density have the
joint distributional limits

$$
R_{N,i}\Rightarrow R_*=-2(d-1)\sum_{z\sim0}w_z^*(u_z^*-u_0^*),
$$
$$
r_N^d\lambda_R R_{N,i}
               \sqrt{\max(\det g_i,f_V)}
\Rightarrow \lambda_R R_*\sqrt{\det G_0}.
\tag{NGA.6}
$$

The scale $r_N^d$ in the second display describes the actual recorded
density's growth; it does not change the reward fed to any update.
Replacing `SqrtDetMetric` by the already configured `RiemannianCell`
gives instead an unscaled limit
$\lambda_R R_*\operatorname{vol}(\mathcal V_0)\sqrt{\det G_0}$.
These assertions concern the passive readout branch identified above.

The law of $R_*$ is independent of the positive Poisson intensity.
For full-slot geometry, consequently, every bounded continuous $\psi$ has

$$
\frac1N\sum_i\psi(R_{N,i})
\longrightarrow\mathbb E_{\Pi_1}\psi(R_*)
\quad\hbox{in probability}.
\tag{NGA.7}
$$

For alive-only geometry the same conclusion holds for the actual
alive average, whose root limits use interior eligible tags.
Its ineligible identity/zero rows keep their actual different output.
All statements transfer to the native QSD in the
proved positive-survival regime. No stationary chaos is needed for
(NGA.7); the phase environment cancels from this particular readout law.
:::

:::{prf:proof}
Every root and neighboring covariance on a determining configuration
has its same-record relative metric limit $r_N^2g_z\to G_z$.
Each $G_z$ is positive definite. The finite set of determinant floors
is therefore eventually inactive. Exactly

$$
u_{N,z}=-\log r_N+u_z^*+o(1).
$$

The common $-\log r_N$ cancels from each difference in the executed
curvature formula. Endpoint-metric lengths converge to $Q_z$, and the
literal distance and row-sum floors are continuous positive functions.
This proves the first display of (NGA.6). Its volume and reward factors
are the exact factors in `volume.rs` and
{prf:ref}`thm-ng-native-reward-weight-variation`; their same-record
limits prove the remaining displays. There is no additional continuum
scalar-curvature prefactor or silently removed graph normalization.

A Poisson configuration of intensity $c$ is obtained from intensity
one by $z\mapsto c^{-1/d}z$. Relative covariance metrics then scale
as $G_z\mapsto c^{2/d}G_z$, including all relative clamps.
Thus $Q_z$ and every $w_z^*$ are unchanged, while each
$u_z^*$ acquires the same $(\log c)/d$. The differences and $R_*$
are unchanged, proving intensity independence.

Apply {prf:ref}`thm-nga-empirical-law` to the bounded readout test
$\psi(R_{N,i})$ on compact tag sets. Its Poisson limit expectation
is the same number at every positive intensity. Uniform native tag
moments from (NSG.1) remove the compact restriction. Equation (NGA.4)
then proves (NGA.7). No unbounded curvature moment is required.

For alive-only geometry first restrict tags to a compact subset of
$D$ with positive interior margin. The determining windows then
contain only alive points. The terminal tag densities are bounded by
$L_{\tau,0}$, so the expected proportion in a shrinking boundary
layer is bounded by its volume times that constant; for the actual
box this tends to zero. The same empirical theorem gives numerator
$\mu_N(D)\mathbb E\psi(R_*)+o_P(1)$.
Conditional independence of the alive marks gives their fraction
$\mu_N(D)+o_P(1)$, with conditional variance at most $1/(4N)$.
Write $a_*>0$ for the existing quadratic landing floor and
$A_N^{\rm alive}=N^{-1}\sum_i\mathbf1_D(Y_i)$.
The original high-alive binomial estimate and conditional variance give

$$
P(\mu_N(D)<a_*/4)
\le P(A_N^{\rm alive}<a_*/2)
       +P(|A_N^{\rm alive}-\mu_N(D)|\ge a_*/4)
\le o(1)+\frac4{Na_*^2}.
$$

Thus this actual denominator is bounded below in probability. Dividing
proves the alive-average claim.
The QSD total-variation transfer completes the proof.
:::

(sec-nga-scope)=
## 5. Exact reach of the assembled estimates

:::{prf:remark} Assembled geometry versus the remaining physical action
:label: rem-nga-scope

The conditional empirical theorem supplies a law of large numbers for
bounded same-record metric, volume, curvature and allocated-density tests.
It retains the actual random population environment whenever it does not
cancel. In particular local metric randomness does not prevent the
proved empirical curvature law.

Unbounded averages of the original action require uniform integrability
of their actual volume/curvature products. Neither bounded weak tests nor
a one-cell inverse formula establishes that integrability. A feedback
Einstein--Hilbert run needs its own sampling and stage controls, since
its graph, metric, curl and reward are consumed by the transition.
The present Poisson law and allocated density do not identify an
Einstein field equation or a Yang--Mills likelihood action.
Numerical comparison errors and finite allocation limits stay in their
complete execution record. Fixed allocation limits do not admit an
infinite-population numerical sequence.
:::
