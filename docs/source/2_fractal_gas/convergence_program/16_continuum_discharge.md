# Continuum Limits: Hypotheses and Consistency

(sec-continuum-scope)=
## 1. What a Continuum Limit Requires

:::{div} feynman-prose
A cloud of points can approximate a probability distribution without telling us
how to measure a distance between two points. Those are separate questions.
There is a third question when we estimate a differential operator: as its
neighborhood shrinks, do we still have enough samples to control the noise?

This chapter separates these questions. We can prove regularity of specified
smoothed fields and compute the normalization and variance of an estimator.
Identifying the graph with a smooth spacetime requires additional geometric
information. Identifying an episode law with a particular Gibbs density
requires an equation for that law. The conditions below say exactly where
those pieces enter.
:::

:::{prf:definition} Scope of the continuum hypotheses
:label: def-continuum-hypothesis-status

The hypotheses A1–A6 in {prf:ref}`assm-fractal-gas-nonlocal` concern the
reconstruction in {doc}`../2_fractal_set/02_causal_set_theory`. Their application
requires the following data.

1. **A1: geometry.** A specified smooth spatial manifold, a positive definite
   field $g_R$, and a product metric $g=-c^2dt^2+g_R$ with $c>0$. Global
   hyperbolicity and agreement of reconstructed distances and causal
   neighborhoods with this geometry require separate proofs or hypotheses.
2. **A2: regularity.** Bounds on derivatives of the fields used by the
   reconstruction, positivity of every denominator, and integrable bounds
   for derivatives of spatial marginals. Bounds needed for a local estimator
   are imposed on a neighborhood of its evaluation region. On an unbounded
   space, a confining potential may itself be unbounded.
3. **A3: sampling.** A normalized episode law with respect to a specified
   reference measure. An exact Gibbs formula, when used, is an additional
   identification of that law.
4. **A4: dependence.** A variance or covariance estimate for the actual
   bandwidth-dependent summands. A law of large numbers for each fixed test
   function does not supply such an estimate.
5. **A5: kernel.** Verified moments of the kernel on its actual cutoff domain,
   together with a local consistency estimate for that domain and geometry.
6. **A6: scaling.** A chosen sequence of sample sizes and reconstruction
   bandwidths for which the preceding errors vanish, with the constants
   controlled along the sequence.

The lemmas below give sufficient conditions and direct estimates. Verifying
all six items for episodes of the interacting algorithm remains a separate
application of these results. In particular, none of the geometric or exact
sampling identifications is inferred from propagation of chaos.
:::

(sec-continuum-regularity)=
## 2. Regularity of Smoothed Fields

:::{div} feynman-prose
Smoothing is an operation we can estimate. If we convolve a probability measure
with a Gaussian of fixed width, each spatial derivative can be moved onto the
Gaussian. The size of that derivative tells us the price of smoothing at that
width. Time derivatives are different: smoothing in space cannot repair a jump
in time.

There is also a derivative to count. A metric formed from a Hessian uses two
spatial derivatives before we begin differentiating the metric. Four bounded
derivatives of that metric therefore require up to six derivatives of its
potential. Keeping this count explicit prevents a regularity assumption from
being lost in a composition.
:::

:::{prf:lemma} Sufficient regularity and spectral bounds for A2
:label: lem-continuum-a2-smooth-fields

Let $I$ be a time interval. In Euclidean spatial coordinates, fix a smoothing
width $\sigma>0$ and write

$$
G_\sigma(x)=(2\pi\sigma^2)^{-d/2}e^{-|x|^2/(2\sigma^2)},
\qquad F(x,t)=\int G_\sigma(x-y)\,\mu_t(dy).
$$

Assume $t\mapsto\mu_t$ is $C^4$ in the total variation norm of finite signed
measures, with $\|\partial_t^j\mu_t\|_{\mathrm{TV}}\le M_j$ for $0\le j\le4$.
Then, for every spatial multi-index $\alpha$ and $j\le4$,

$$
\|\partial_x^\alpha\partial_t^j F\|_\infty
\le M_j\,C_\alpha\sigma^{-d-|\alpha|},
\qquad C_\alpha=\|\partial^\alpha G_1\|_\infty.
$$

More generally, the same conclusion holds for a smoothed field whose
space-time derivatives admit the corresponding dominated integral bounds.
The time regularity of the underlying measure is part of the hypothesis.

Suppose a potential $V$ has bounded derivatives needed to form
$A=\nabla_x^2V+\epsilon_\Sigma I$ and its space-time derivatives of total order
at most four. Assume

$$
-\Lambda_-I\preceq\nabla_x^2V\preceq\Lambda_+I,
\qquad a:=\epsilon_\Sigma-\Lambda_->0,
\qquad b:=\epsilon_\Sigma+\Lambda_+<\infty.
$$

Then the metric and diffusion convention of
{prf:ref}`def-adaptive-diffusion-tensor-latent` gives

$$
g_R=A,\qquad D_{\mathrm{reg}}=A^{-1},\qquad
 aI\preceq g_R\preceq bI,\qquad
 b^{-1}I\preceq D_{\mathrm{reg}}\preceq a^{-1}I.
$$

Both fields are $C^4$. If $A_k$ bounds all derivatives of $A$ of total order
$k$, bounds for derivatives of $A^{-1}$ are given recursively by

$$
B_0=a^{-1},\qquad
B_k=a^{-1}\sum_{j=1}^k {k\choose j}A_jB_{k-j},\quad 1\le k\le4.
$$

For an effective potential defined from a positive field $s$ by
$U=-T\log s$, assume $s\ge s_*>0$ on the region in question and let $S_k$
bound its derivatives of order $k$. Derivatives of $\log s$ of order $k\ge1$
are bounded by

$$
L_1=S_1/s_*,\qquad
L_k=s_*^{-1}\left(S_k+
\sum_{j=1}^{k-1}{k-1\choose j}S_jL_{k-j}\right),\quad 2\le k\le4.
$$

Thus derivatives of $U$ are bounded by $TL_k$. This conclusion concerns the
specified field $s$; identifying it with the episode density is additional.

Finally, if $b(t,x)$ is $C^4$ in time and
$|\partial_t^j b(t,x)|\le B_j(x)$ for integrable $B_j$, then

$$
Z(t):=\int b(t,x)\,dx\in C^4(I),\qquad
|Z^{(j)}(t)|\le\|B_j\|_{L^1}.
$$

The same assertion applies to a rate $r(t)$ represented by such an integral.
Positive lower bounds on $Z$ and $r$ must be checked separately wherever their
reciprocals occur. In particular, the lemma applies to
$b=\sqrt{\det g_R}\,e^{-U/T}$ when its time derivatives have these integrable
bounds.
:::

:::{prf:proof}
Differentiation under the convolution is justified by the bounded derivatives
of $G_\sigma$ and differentiability of $\mu_t$ in total variation. Scaling the
Gaussian gives
$\partial^\alpha G_\sigma(x)=\sigma^{-d-|\alpha|}
(\partial^\alpha G_1)(x/\sigma)$, proving the first estimate.

The spectral inequalities follow by diagonalizing $A$. For a multi-index
$\beta$ with $|\beta|=k$, differentiate $AA^{-1}=I$:

$$
\partial^\beta A^{-1}
=-A^{-1}\sum_{0<\gamma\le\beta}{\beta\choose\gamma}
(\partial^\gamma A)(\partial^{\beta-\gamma}A^{-1}).
$$

Take operator norms and use
$\sum_{|\gamma|=j,\,\gamma\le\beta}{\beta\choose\gamma}={k\choose j}$
to obtain the recursion for $B_k$. The matrix field
$\Sigma_{\mathrm{reg}}=A^{-1/2}$ is also $C^4$: use

$$
A^{-1/2}=\frac1\pi\int_0^\infty u^{-1/2}(A+uI)^{-1}\,du.
$$

The preceding recursion, with $a$ replaced by $a+u$, bounds derivatives of
the resolvent. For positive derivative order these are finite sums of products
with at least two resolvent factors, so the differentiated integrands are
integrable at both endpoints. Differentiation under this integral is therefore
valid through order four.

For the logarithm, differentiate
$s\,\partial_i\log s=\partial_i s$ a further $k-1$ times and isolate the
highest derivative of $\log s$. The same multi-index summation gives the
stated recursion. The positive lower bound justifies division by $s$.

The marginal formula and its derivative bounds follow from dominated
convergence applied successively to $\partial_t^j b$. Continuity follows by the
same domination. Each assertion is thus a consequence of its stated
regularity, spectral, or domination hypotheses. $\square$
:::

:::{prf:remark} Spatial, temporal, and global bounds
:label: rem-continuum-regularity-scope

For a probability measure, $M_0=1$ in the convolution estimate. The constants
increase as $\sigma\downarrow0$; uniformity along a changing smoothing scale
requires a separate estimate. The Gaussian argument supplies spatial
regularity even for atomic measures, but the total variation hypothesis on
time derivatives is stronger and must be verified or replaced by a suitable
differentiation-under-the-integral argument.

For $g_R=\nabla_x^2V+\epsilon_\Sigma I$, $C^4$ regularity of $V$ alone gives
only two spatial derivatives of $g_R$. A positive lower bound on $g_R$ gives
an upper bound on its inverse; the lower diffusion bound also requires the
upper bound $g_R\preceq bI$. A spectral shift must dominate the negative part
of the Hessian to ensure positivity at all.

On an unbounded space, local positivity of $s$, global integrability of a
confining density, and uniform bounds on derivatives are distinct conditions.
A finite time interval supplies none of these spatial estimates by itself.
:::

(sec-continuum-geometry)=
## 3. Product Geometry and Graph Identification

:::{div} feynman-prose
A speed bound becomes useful once we know what measures speed. Suppose our
spatial metric is everywhere at least a fixed multiple of an ordinary
complete metric. Then a causal curve has a bounded speed in that ordinary
metric too. During a finite time interval it can travel only a bounded
distance. This is the geometric fact that can keep causal diamonds compact,
even when space itself is unbounded.
:::

:::{prf:lemma} A sufficient condition for global hyperbolicity
:label: lem-continuum-a1-geometry

Let $(\mathcal X,h)$ be a connected complete Riemannian manifold whose closed
bounded sets are compact. Let $I$ be an open interval and let $g_R(t)$ be a
$C^4$ family of Riemannian metrics satisfying

$$
a h\preceq g_R(t)\preceq b h\qquad(t\in I)
$$

for $0<a\le b<\infty$. Then
$M=I\times\mathcal X$, with $g=-c^2dt^2+g_R(t)$ and $c>0$, is globally
hyperbolic, and every slice $\{t\}\times\mathcal X$ is a Cauchy hypersurface.
A closed observation window inside $I$ may be used for sampling.

These hypotheses allow unbounded spatial domains, including $\mathbb R^d$
with its Euclidean background metric. They concern a specified product
manifold. Identifying it with a limit of the IG graph or its distances is an
additional hypothesis.
:::

:::{prf:proof}
For a future-directed causal curve parametrized by time,

$$
g_R(t)(\dot x,\dot x)\le c^2,
\qquad |\dot x|_h\le c/\sqrt a.
$$

Consequently its spatial part is $c/\sqrt a$-Lipschitz. If the time parameter
of an inextendible causal curve had a finite endpoint in $I$, completeness
would give a spatial limit there. Appending a vertical timelike segment would
extend the curve. Its time range is therefore all of $I$, and it intersects
each time slice exactly once.

For $p=(s,x)$ and $q=(t,y)$ with $s\le t$, every point of the causal diamond
lies in

$$
[s,t]\times\overline B_h\bigl(x,c(t-s)/\sqrt a\bigr),
$$

which is compact. The diamond is closed: for a convergent sequence of its
points, the causal curves from $p$ to these points and from them to $q$ have
the same Lipschitz bound and lie in a common compact set. Extend their spatial
parts constantly to common time intervals and apply Arzelà–Ascoli. A uniformly
convergent subsequence has a Lipschitz limit. In each coordinate neighborhood,
continuity of $g_R$ and weak lower semicontinuity of the quadratic energy give
$g_R(t)(\dot x,\dot x)\le c^2$ almost everywhere on each limiting segment.
Thus the limiting point remains between $p$ and $q$ causally.

Strong causality also follows from the speed estimate. Inside any product
neighborhood of a point, choose a smaller product neighborhood with time
width so short that a curve with speed at most $c/\sqrt a$ cannot leave the
larger spatial neighborhood between two points of the smaller one. Time
monotonicity keeps the intervening segment inside the larger time interval.
This is the local strong-causality criterion. Together with compact causal
diamonds it gives global hyperbolicity. The slice conclusion follows from
the inextendibility argument above. $\square$
:::

:::{prf:remark} Required geometric identifications
:label: rem-continuum-graph-identification

Propagation of chaos is convergence of specified particle marginals. It does
not construct a manifold, a metric, or a shortest-path approximation theorem.
A confining probability envelope controls tails of a measure; compactness of
causal diamonds is a property of the metric and causal curves.

The graph-Laplacian result of
[Belkin and Niyogi](https://misha.belkin-wang.org/papers/TT_JCSS_08.pdf)
starts with samples on a specified compact embedded manifold; its main
convergence theorem concerns the Laplace–Beltrami operator. It supplies no
shortest-path convergence or Lorentzian reconstruction theorem. Its compact
sampling model is not imposed here on an unbounded Fractal Gas.

For an application using graph distances, one must specify an embedding of
episodes, a distance approximation bound, and control of its effect on the
shrinking kernel and causal cutoff. A bound on discrete endpoint displacement
also requires an interpolation argument before it becomes a pointwise causal
speed bound for a time-dependent metric. Equality at the speed bound is null;
strict inequality gives a timelike curve.
:::

(sec-continuum-sampling)=
## 4. Sampling Laws, QSD, and Normalization

:::{div} feynman-prose
The weight in an importance sampler is the reciprocal of the density relative
to the measure we want to integrate. This rule fixes every normalization
constant. It also tells us what to do if the actual density contains a factor
that a proposed Gibbs formula omitted: that factor belongs in the weight.

Quasi-stationarity concerns particles conditional on having survived. Sampling
a long trajectory of a process conditioned to survive forever is another
experiment. Those experiments can have different limiting laws, so we must
name the experiment before choosing the weight.
:::

:::{prf:lemma} Normalized sampling and the distinction between QSD and invariance
:label: lem-continuum-a3-qsd-sampling

Let $P_t$ be a killed sub-Markov semigroup. A probability measure $\nu$
satisfying $\nu P_t=e^{-\lambda t}\nu$ is a QSD: conditioning on survival
preserves its one-time law. If $\lambda>0$, it is not an invariant probability
measure for $P_t$.

If a positive function $h$ satisfies $P_th=e^{-\lambda t}h$ and
$0<\nu(h)<\infty$, define

$$
Q_t\phi=e^{\lambda t}h^{-1}P_t(h\phi),\qquad
\beta(dx)=\frac{h(x)\nu(dx)}{\nu(h)}.
$$

Then $Q_t$ is conservative and $\beta$ is invariant for it. Identification with
a process conditioned on indefinite survival requires the corresponding
conditioning-limit theorem.

Independently, let episodes $Y_i$ have a specified common probability law
$\pi(dy)=q(y)\,d\mathrm{vol}_g(y)$. On the support of an integrand $F$, assume
$q>0$ and $F$ integrable against $d\mathrm{vol}_g$. Then

$$
\mathbb E\!\left[\frac1N\sum_{i=1}^N\frac{F(Y_i)}{q(Y_i)}\right]
=\int F\,d\mathrm{vol}_g.
$$

Independence is unnecessary for this expectation identity. It is relevant to
the variance and convergence of the average.

For a finite window and the exact coordinate sampling law

$$
p(t,x)=\frac{r(t)}{RZ(t)}\sqrt{\det g_R(t,x)}e^{-U(t,x)/T},
\quad Z(t)=\int\sqrt{\det g_R(t,x)}e^{-U(t,x)/T}\,dx,
\quad R=\int r(t)\,dt,
$$

assume $0<R<\infty$, $r,Z>0$, and $Z<\infty$. Since
$d\mathrm{vol}_g=c\sqrt{\det g_R}\,dt\,dx$, the density and weight are

$$
q(t,x)=\frac{r(t)}{cRZ(t)}e^{-U(t,x)/T},
\qquad q(t,x)^{-1}=\frac{cRZ(t)}{r(t)}e^{U(t,x)/T}.
$$

For the product measure $\sqrt{\det g_R}\,dt\,dx$ the factor $c$ is omitted.
Rescaling time to $ct$ also absorbs this constant, provided the time density
is transformed consistently.
:::

:::{prf:proof}
Integrating $\nu P_t=e^{-\lambda t}\nu$ against $1$ gives the survival
probability $e^{-\lambda t}$. Dividing by it gives the conditional law $\nu$.
When $\lambda>0$, the unconditioned surviving mass decreases, which excludes
invariance. For the transformed semigroup,

$$
Q_t1=1,\qquad
\int Q_t\phi\,d\beta
=\frac{e^{\lambda t}}{\nu(h)}\nu P_t(h\phi)
=\frac{\nu(h\phi)}{\nu(h)}.
$$

The importance-sampling identity follows by integrating $F/q$ against
$q\,d\mathrm{vol}_g$ and using linearity of expectation. The displayed
coordinate density integrates to one, first in space and then in time.
Dividing it by $c\sqrt{\det g_R}$ gives $q$ and its reciprocal. $\square$
:::

:::{prf:remark} What an exact density formula requires
:label: rem-continuum-exact-density

An upper or lower Gibbs envelope for a QSD does not establish the equality
$p\propto\sqrt{\det g_R}e^{-U/T}$. A multiplicative decoration changes the
importance weight unless it is included in the specified potential and its
normalizer. An exact identification requires verification of the density
against the relevant generator or killed eigenmeasure equation and its
boundary conditions. A stationary law for a drift-augmented diffusion is a
statement about that diffusion.

Exchangeability concerns permutation of particle labels. It supplies neither
time stationarity nor independence. A time-dependent slice law also requires
an argument connecting the episode sampling protocol to those slices.
[Champagnat and Villemonais](https://arxiv.org/abs/1712.08092) give conditions
for convergence to a QSD and for existence and ergodicity of the associated
conditioned process. Those conditions must be checked for the particular
process; they do not identify its QSD with a prescribed Gibbs formula.
:::

(sec-continuum-kernels)=
## 5. Kernel Moments and Deterministic Consistency

:::{div} feynman-prose
For a second derivative, the quadratic moment of the kernel does the work.
A wave operator needs opposite signs in the temporal and spatial moments,
which a nonnegative kernel cannot provide. Signed weights can provide them,
but the moments must be computed on the actual integration domain.

A finite cutoff selects a frame. Spatial rotations still make the spatial
moments equal, and reflection can kill odd moments. Neither symmetry fixes
the sign or magnitude of the temporal moment. That is an additional equation.
:::

:::{prf:lemma} Verified moments and an auxiliary tangent-space construction
:label: lem-continuum-a5-kernel

Use length coordinates $(ct,x)$ in a Lorentz-orthonormal frame and set
$\eta=\operatorname{diag}(-1,1,\ldots,1)$ in dimension $D=d+1$. Let
$J=-J\subset\mathbb R^D$ be a bounded measurable set. A bounded measurable even kernel $k$ supported in $J$ is admissible for
the local difference construction below when

$$
\int_J k(\zeta)\zeta^\mu\zeta^\nu\,d\zeta
=2m_2\eta^{\mu\nu},\qquad m_2>0.
$$

The condition $\int_Jk=0$ may additionally be imposed. It is unnecessary for
annihilation of constants in an operator containing $f(y)-f(p)$.

There is an explicit auxiliary construction when the interior of $J$ contains
an open set and its reflection. Choose an even
$\phi\in C_c^\infty(\operatorname{int}J)$ with $\int\phi=1$ and define

$$
k(\zeta)=m_2\eta^{\mu\nu}\partial_\mu\partial_\nu\phi(\zeta).
$$

Then $k$ is smooth, even, compactly supported, has zero integral, and has the
specified second moments. In particular, its support can be chosen inside a
bounded timelike double cone. This is a mathematical example of a tangent
kernel; it neither changes the particle algorithm nor identifies the existing
proper-time reconstruction kernel with this example.

For a proper-time-only kernel, take $J$ inside the timelike double cone and
write $\tau(\zeta)=\sqrt{(\zeta^0)^2-|\zeta^{\mathrm{space}}|^2}$ and
$k(\zeta)=K(\tau(\zeta))$. Suppose $J$ is invariant under time reversal and
spatial rotations. A finite ansatz
$K=\sum_{j=1}^m a_j\phi_j$, with $\phi_j\in C_c^2([0,1])$, is valid precisely
when its actual moment matrix
$C$ admits a solution to

$$
Ca=(0,-2m_2,2m_2)^\top,
\quad C_{0j}=\int_J\phi_j(\tau)\,d\zeta,
\quad C_{tj}=\int_J\phi_j(\tau)(\zeta^0)^2\,d\zeta,
\quad C_{sj}=\int_J\phi_j(\tau)(\zeta^1)^2\,d\zeta.
$$

This criterion applies for $d\ge1$ when all the moments are finite. Full row
rank is a sufficient condition. The rank or solvability must be checked for
the chosen domain and functions.
:::

:::{prf:proof}
Evenness makes every odd monomial integrate to zero. For the auxiliary
construction, compact support removes boundary terms in integration by parts.
The integral of each second derivative of $\phi$ is zero, while

$$
\int \partial_\mu\partial_\nu\phi\,
\zeta^\alpha\zeta^\beta\,d\zeta
=\delta_\mu^\alpha\delta_\nu^\beta+
\delta_\mu^\beta\delta_\nu^\alpha.
$$

Contraction with $m_2\eta^{\mu\nu}$ gives the required second moment. A smooth
nonnegative bump and its reflection, divided by their combined integral,
provide the stated $\phi$.

For the proper-time ansatz, time reversal removes temporal-spatial moments;
spatial rotational symmetry removes off-diagonal spatial moments and makes
the diagonal spatial moments equal. The remaining constraints are exactly
the three displayed linear equations. A cutoff bounded in the chosen time
coordinate is not invariant under arbitrary Lorentz boosts, so it supplies
no identity relating $C_{tj}$ to $C_{sj}$. Two coefficients and two nonzero
scalar moments therefore do not establish solvability of these three
constraints. $\square$
:::

:::{prf:lemma} Direct local bias estimate in specified normal coordinates
:label: lem-continuum-local-bias

Fix an interior point $p$ of a specified Lorentzian manifold and a normal
coordinate map $\Psi_p$ with $\Psi_p(0)=p$ and metric $\eta$ at the origin.
Suppose this map is defined on a Euclidean ball containing $\varepsilon J$
for all sufficiently small $\varepsilon$. Write

$$
F_p(\xi)=f(\Psi_p(\xi)),\qquad
\Psi_p^*(d\mathrm{vol}_g)=j_p(\xi)\,d\xi.
$$

Assume $F_p\in C^4$, $j_p\in C^3$, $j_p(0)=1$, $Dj_p(0)=0$, and bounds
$\|D^kF_p\|\le L_k$ for $1\le k\le4$,
$\|D^2j_p\|\le J_2$, and $\|D^3j_p\|\le J_3$ on that ball. The derivative
norms are Euclidean multilinear operator norms. With the verified kernel of
{prf:ref}`lem-continuum-a5-kernel`, define

$$
L_\varepsilon f(p)=\frac1{m_2\varepsilon^{D+2}}
\int_{\Psi_p(\varepsilon J)}
k\!\left(\frac{\Psi_p^{-1}(y)}{\varepsilon}\right)
\bigl(f(y)-f(p)\bigr)\,d\mathrm{vol}_g(y).
$$

Then

$$
|L_\varepsilon f(p)-\Box_g f(p)|\le C_{\mathrm b}\varepsilon^2,
\qquad
C_{\mathrm b}=\frac1{m_2}
\left(\frac{L_4}{24}+\frac{L_1J_3}{6}+\frac{L_2J_2}{4}\right)
\int_J|k(\zeta)||\zeta|^4\,d\zeta.
$$

This statement concerns the displayed coordinate construction. Applying it
to the existing reconstruction requires a comparison of that reconstruction's
proper-time proxy, domain, and metric with this construction, with an error
after the factor $\varepsilon^{-D-2}$ that tends to zero.
:::

:::{prf:proof}
Taylor expansion gives
$F_p(\xi)-F_p(0)=P_1(\xi)+P_2(\xi)+P_3(\xi)+R_4(\xi)$, where $P_k$ is
homogeneous of degree $k$, $P_2=\tfrac12D^2F_p(0)[\xi,\xi]$, and
$|R_4|\le L_4|\xi|^4/24$. Similarly,

$$
j_p(\xi)-1=Q_2(\xi)+R_3^j(\xi),\quad
|R_3^j|\le J_3|\xi|^3/6,\quad
|j_p(\xi)-1|\le J_2|\xi|^2/2.
$$

Decompose the integrand as

$$
(F_p-F_p(0))j_p
=(F_p-F_p(0))+P_1(j_p-1)
 +(F_p-F_p(0)-P_1)(j_p-1).
$$

The integrals of $P_1$, $P_3$, and $P_1Q_2$ vanish by evenness. The remaining
error is bounded in absolute value by

$$
\left(\frac{L_4}{24}+\frac{L_1J_3}{6}+\frac{L_2J_2}{4}\right)|\xi|^4,
$$

using $|F_p-F_p(0)-P_1|\le L_2|\xi|^2/2$. After substituting
$\xi=\varepsilon\zeta$, the quadratic term is
$\eta^{\mu\nu}\partial_\mu\partial_\nu F_p(0)=\Box_g f(p)$, since the
Christoffel symbols vanish at the normal-coordinate origin. The error yields
the stated constant and power of $\varepsilon$. $\square$
:::

(sec-continuum-variance)=
## 6. Variance and an Explicit Scaling Regime

:::{div} feynman-prose
The shrinking neighborhood gives us a smaller bias and a noisier estimate.
Its volume is of order $\varepsilon^D$. The difference $f(y)-f(p)$ contributes
one power of $\varepsilon$, but a second-derivative estimator divides by
$\varepsilon^{D+2}$. Squaring and integrating these factors gives the variance
scale below. Correlation enters through a separate multiplier, which must
stay controlled as the neighborhood shrinks.
:::

:::{prf:lemma} A covariance condition sufficient for A4
:label: lem-continuum-a4-mixing

Fix the deterministic evaluation point $p$ and the coordinate construction of
{prf:ref}`lem-continuum-local-bias`. Let $Y_1,\ldots,Y_N$ have common law
$\pi=q\,d\mathrm{vol}_g$, with $q\ge q_*>0$ on the kernel neighborhood and
$j_p\le J_0$ there. Define

$$
H_{\varepsilon,p}(y)=\frac{
\mathbf1_{\Psi_p(\varepsilon J)}(y)
 k(\Psi_p^{-1}(y)/\varepsilon)(f(y)-f(p))
}{m_2\varepsilon^{D+2}q(y)},
\qquad
\widehat L_{N,\varepsilon}f(p)=\frac1N\sum_{i=1}^N H_{\varepsilon,p}(Y_i).
$$

Assume, for these summands and all $N,\varepsilon$ under consideration,

$$
\sum_{i,j=1}^N
\left|\operatorname{Cov}(H_{\varepsilon,p}(Y_i),
                         H_{\varepsilon,p}(Y_j))\right|
\le C_{\mathrm{mix}}N\,\mathbb E[H_{\varepsilon,p}(Y_1)^2]
$$

with a constant independent of $N$ and $\varepsilon$. Then

$$
\mathbb E\widehat L_{N,\varepsilon}f(p)=L_\varepsilon f(p),
\qquad
\operatorname{Var}(\widehat L_{N,\varepsilon}f(p))
\le\frac{C_{\mathrm{mix}}C_{\mathrm v}}{N\varepsilon^{D+2}},
\quad
C_{\mathrm v}=\frac{J_0L_1^2}{m_2^2q_*}
\int_J k(\zeta)^2|\zeta|^2\,d\zeta.
$$

The kernel moment is finite by boundedness of $k$ and $J$.
Independent samples satisfy the covariance condition with $C_{\mathrm{mix}}=1$.
A stationary sequence satisfying
$|\operatorname{Cov}(H(Y_0),H(Y_\ell))|
\le C_0e^{-\lambda\ell}\mathbb E H(Y_0)^2$ uniformly over this kernel family
satisfies it with
$C_{\mathrm{mix}}=1+2C_0/(e^\lambda-1)$.
:::

:::{prf:proof}
The expectation is the importance-sampling identity in
{prf:ref}`lem-continuum-a3-qsd-sampling`. For the second moment, cancel one
factor of $q$ against the sampling law and use normal coordinates:

$$
\begin{aligned}
\mathbb E H_{\varepsilon,p}(Y_1)^2
&=\frac1{m_2^2\varepsilon^{2D+4}}
\int_{\Psi_p(\varepsilon J)}
\frac{k(\Psi_p^{-1}(y)/\varepsilon)^2(f(y)-f(p))^2}{q(y)}
\,d\mathrm{vol}_g(y)\\
&\le\frac{J_0L_1^2}{m_2^2q_*\varepsilon^{D+2}}
\int_J k(\zeta)^2|\zeta|^2\,d\zeta.
\end{aligned}
$$

Expanding the variance of the average and applying the covariance hypothesis
gives the claim. For a stationary sequence the diagonal sum is at most
$N\mathbb EH^2$; each lag $\ell\ge1$ contributes at most
$2NC_0e^{-\lambda\ell}\mathbb EH^2$. Sum the geometric series. $\square$
:::

:::{prf:remark} Dependence and conditioning in an application
:label: rem-continuum-shrinking-observables

The covariance condition is imposed on the bandwidth-dependent observables,
whose sizes grow as $\varepsilon$ decreases. A fixed-observable law of large
numbers, exchangeability, or a propagation-of-chaos statement without
bandwidth-dependent constants is insufficient to verify it. A logarithmic
Sobolev inequality for a measure yields dynamical information only after
specifying a compatible generator and proving the required dissipation or
correlation bound. It does not supply correlations for an arbitrary episode
sampling protocol.

If $p$ is itself one of the sampled episodes, the sampling and covariance
hypotheses must hold for the remaining samples conditional on $p$, with the
appropriate sample count. Similarly, estimated weights and reconstructed
distances require their own error estimates. The lemma uses exact $q$ and a
specified geometry. On an unbounded space its positive density bound is
needed only on the local integration region.
:::

:::{prf:lemma} A sufficient bandwidth schedule for A6
:label: lem-continuum-a6-scaling

Under {prf:ref}`lem-continuum-local-bias` and
{prf:ref}`lem-continuum-a4-mixing`, with constants independent of $N$, choose

$$
\varepsilon_N=\ell_0N^{-\alpha},\qquad
\ell_0>0,\qquad 0<\alpha<\frac1{D+4}.
$$

Here $\ell_0$ is a fixed reference length. Then
$\varepsilon_N\to0$, $N\varepsilon_N^{D+4}\to\infty$, and

$$
\mathbb E\left|\widehat L_{N,\varepsilon_N}f(p)-\Box_gf(p)\right|^2
\le C_{\mathrm b}^2\ell_0^4N^{-4\alpha}
 +\frac{C_{\mathrm{mix}}C_{\mathrm v}}{\ell_0^{D+2}}
 N^{-1+\alpha(D+2)}\longrightarrow0.
$$

Thus the estimator converges in probability. The choice
$\alpha=1/(D+6)$ balances these two bounds, giving a mean square error of
order $N^{-4/(D+6)}$ under these hypotheses.
:::

:::{prf:proof}
The squared error decomposes into variance plus squared bias. Substitute the
preceding bounds and the stated bandwidth. Both exponents are negative,
since $\alpha>0$ and $\alpha(D+2)<1$. Also
$N\varepsilon_N^{D+4}=\ell_0^{D+4}N^{1-\alpha(D+4)}\to\infty$.
Chebyshev's inequality gives convergence in probability. Equating
$4\alpha=1-\alpha(D+2)$ yields $\alpha=1/(D+6)$. $\square$
:::

(sec-continuum-conditional-conclusion)=
## 7. Conditional Consistency and the Remaining Identifications

:::{div} feynman-prose
We can now see exactly what the limit theorem costs. With the geometry and
sampling law specified, a Taylor estimate controls the bias and a covariance
estimate controls the fluctuations. To apply that calculation to recorded
episodes, we still have to show that the reconstructed operator approximates
the operator just analyzed. Making the bandwidth a function of the sample
count is a choice in this calculation; it does not force an existing run to
produce the required geometry or sampling law.
:::

:::{prf:corollary} Conditional continuum consistency
:label: cor-continuum-consistency-conditional

Assume the geometric and normal-coordinate hypotheses of
{prf:ref}`lem-continuum-local-bias`, the exact normalized sampling and covariance
hypotheses of {prf:ref}`lem-continuum-a4-mixing`, and the scaling of
{prf:ref}`lem-continuum-a6-scaling`. Then the specified local estimator
$\widehat L_{N,\varepsilon_N}f(p)$ converges in mean square to $\Box_gf(p)$
at each fixed interior evaluation point.

A reconstructed episode operator $\widetilde L_N$ has the same limit in
probability provided its density, metric, neighborhood, and kernel
approximations additionally satisfy

$$
\widetilde L_Nf(p)-\widehat L_{N,\varepsilon_N}f(p)
\xrightarrow{\mathbb P}0.
$$

This comparison is an explicit hypothesis for the reconstruction in
{prf:ref}`def-cst-fractal-dalembertian`. Its verification must control errors
after the normalization by $\varepsilon_N^{-D-2}$.

For an action limit, suppose further that these mean square bounds hold
uniformly on the compact support of $f$, and let $Z_1,\ldots,Z_M$ be independent
samples with law $q\,d\mathrm{vol}_g$, independent of the samples defining
$\widehat L$. Assume $q$ is bounded below there and
$f\Box_gf\in L^2(q^{-1}d\mathrm{vol}_g)$. Then, as $M,N\to\infty$,

$$
\widehat S_{M,N}[f]:=\frac1{2M}\sum_{j=1}^M
\frac{f(Z_j)\widehat L_{N,\varepsilon_N}f(Z_j)}{q(Z_j)}
\xrightarrow{\mathbb P}\frac12\int f\Box_gf\,d\mathrm{vol}_g.
$$

This independent evaluation construction is a sufficient mathematical
example. For the action evaluated on the same interacting episodes as the
operator, the corresponding joint weighted error and quadrature estimates
remain additional conditions.
:::

:::{prf:proof}
The pointwise mean square conclusion is
{prf:ref}`lem-continuum-a6-scaling`. Adding an error that tends to zero in
probability preserves convergence in probability.

For the action, let $a_N\to0$ bound the mean square operator error uniformly
on $\operatorname{supp}f$. Independence of the evaluation sample and
Cauchy–Schwarz give

$$
\mathbb E\left|\widehat S_{M,N}[f]
-\frac1{2M}\sum_{j=1}^M\frac{f(Z_j)\Box_gf(Z_j)}{q(Z_j)}\right|
\le\frac{\sqrt{a_N}}2\int|f|\,d\mathrm{vol}_g.
$$

This error tends to zero by Markov's inequality. The remaining independent
sample average has the desired expectation and variance at most
$(4M)^{-1}\int(f\Box_gf)^2/q\,d\mathrm{vol}_g$, which tends to zero.
Combining the two estimates proves the action limit. $\square$
:::

:::{prf:remark} Conditions to retain in downstream applications
:label: rem-continuum-downstream-conditions

An application of the corollary retains the specified continuum geometry,
exact sampling law or justified importance weights, verified cutoff moments,
local coordinate bounds, and covariance estimates for shrinking neighborhoods.
Graph reconstruction errors must vanish at the estimator's normalization,
and an action limit requires a joint integration estimate.

The bandwidth schedule is a sequence of analytic reconstructions. Identifying
it with the algorithmic radius
$R_{\mathrm{loc}}=\min(\rho,\varepsilon_c)$ requires an explicitly specified
family of runs and uniform estimates as these parameters change. Neither a
finite sample size nor an empirical $N^{-1/2}$ error for fixed observables
forces that identification. The sample count here is the number of observations
used by the estimator; time and particle correlations determine its effective
information content through the covariance estimate.

The conditional results therefore support continuum calculations once these
conditions are verified. Their verification for the full interacting episode
process, its graph geometry, and its existing same-sample action remains open
within this chapter.
:::
