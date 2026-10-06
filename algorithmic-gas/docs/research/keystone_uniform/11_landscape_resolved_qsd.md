# Landscape-resolved quasi-stationary phase balance

(sec-klq-rastrigin-cells)=
## 1. Actual landscape cells and complete-kernel fluxes

:::{prf:definition} Rastrigin cells and population phases
:label: def-klq-landscape-cells

Use the unchanged standard Rastrigin force of
{prf:ref}`ex-slc-rastrigin-residence`,

$$
U(x)=|x|^2+10\sum_{r=1}^d[1-\cos(2\pi x_r)],\qquad
F_r(x)=-2x_r-20\pi\sin(2\pi x_r).
\tag{KLQ.1}
$$

For the declared terminal box $D=[-2,2]^d$, let $\zeta_k$ be
the unique stable one-coordinate force zero in
$[k-1/8,k+1/8]$, $k=-2,-1,0,1,2$. Let $\beta_{k+1/2}$ be
the unique unstable zero in $[k+3/8,k+5/8]$, $k=-2,-1,0,1$.
The one-coordinate well intervals are the intervals between
successive $\beta$ values, truncated at $-2,2$, with a fixed
half-open endpoint convention. Cartesian products are denoted
$B_\ell$, $\ell\in\{-2,-1,0,1,2\}^d$.

An output row has label $\ell$ when its terminal position lies
in $B_\ell$, and label $\dagger$ when it lies outside $D$.
This is an exact coarse observation of the original terminal
mark. It introduces neither a new death event nor a reflecting
well boundary. Population phases $G_{i,N}$ are any declared
measurable full-state classes, for example sets of alive
well-count vectors near the stationary population phases
already considered in Chapter 06a Sections 12--13. Include
an exterior class $G_{0,N}$ so they partition all nonextinct
states. A well label alone is not asserted to be a stationary
population law or an equilibrium orbit.
:::

:::{prf:proof}
On $[k-1/8,k+1/8]$,
$U''(x)=2+40\pi^2\cos(2\pi x)\ge2+20\sqrt2\pi^2>0$.
Its derivative at the two endpoints is
$2k\pm1/4\pm10\sqrt2\pi$ with matching signs. For $|k|\le2$
these have opposite signs, so strict monotonicity and the
intermediate value theorem give the unique stable zero.
On $[k+3/8,k+5/8]$ the second derivative is at most
$2-20\sqrt2\pi^2<0$; its endpoint derivatives have opposite
signs in the reverse order. This gives the unique unstable
zero. The latter separate the former. All intervals and
Cartesian products therefore have the asserted ordering.
Endpoint conventions do not affect probabilities because the
actual positive Gaussian position noise gives zero landing
mass on each such hypersurface. $\square$
:::

:::{prf:lemma} Primitive force and reward profiles for every well
:label: lem-klq-rastrigin-profiles

For the actual force and reward $R=-U$,

$$
\begin{aligned}
|F(x)|&\le2|x|+20\pi\sqrt d,\\
\sup_x\|DF(x)\|&\le2+40\pi^2,\\
|x|^2&\le U(x)\le |x|^2+20d,\\
\operatorname{osc}_{[-L_D,L_D]^d}R&\le dL_D^2+20d,\\
\operatorname{Lip}_{[-L_D,L_D]^d}R&\le2\sqrt dL_D+20\pi\sqrt d.
\end{aligned}
\tag{KLQ.2}
$$

On the enlarged cores of {prf:ref}`ex-slc-rastrigin-residence`,
the local Hessian interval is
$[2+20\sqrt2\pi^2,2+40\pi^2]$; the barrier intervals instead
have a negative upper Hessian bound. Thus neither the moment
nor the phase-convergence calculation replaces these distinct
landscape regions by one globally attractive bowl.
:::

:::{prf:proof}
Use $|\sin|\le1$, $|\cos|\le1$ and the exact diagonal derivative
of (KLQ.1). Each $1-\cos$ lies in $[0,2]$, giving the potential
and reward-oscillation bounds. The gradient norm on the convex
box is bounded by the force-growth expression, proving the
reward Lipschitz bound. The local Hessian calculation is the
one just proved for the stable and unstable intervals.
$\square$
:::

:::{prf:lemma} Gaussian generating function for actual population phase flux
:label: lem-klq-exact-phase-integral

Retain the complete canonical marked kernel, either configured
Gaussian viscosity normalization, all sampled fitness and
component-Haar inputs, unbounded clone jitter and both kinetic
Gaussian innovations. Freeze the entire actual preparation.
The conditional final means and variance are

$$
M_i=(1-2bt)X_i-20\pi bt\sin(2\pi X_i)+b(W_Xv^{\rm col})_i,
\qquad \sigma_h^2=t^2q^2+s^2,
\tag{KLQ.3}
$$

with the sine taken coordinatewise. For a well box
$B_\ell=\prod_r[l_{\ell,r},u_{\ell,r}]$, put

$$
p_{i\ell}=
\prod_{r=1}^d\left[
\Phi((u_{\ell,r}-M_{i,r})/\sigma_h)
-\Phi((l_{\ell,r}-M_{i,r})/\sigma_h)\right],\qquad
p_{i\dagger}=1-\sum_\ell p_{i\ell}.
$$

The exact conditional generating polynomial of row-label counts is

$$
\mathcal P_S(z)=
\prod_{i=1}^N\left[p_{i\dagger}z_\dagger+
\sum_\ell p_{i\ell}z_\ell\right].
\tag{KLQ.4}
$$

If a phase $G_{j,N}$ is defined by a specified set $\mathcal H_j$
of well-count vectors, its actual transition coefficient is

$$
Q_N(S,G_{j,N})=
\mathbb E_S^{\rm prep}\sum_{n\in\mathcal H_j}
[z^n]\mathcal P_S(z).
\tag{KLQ.5}
$$

The all-dead coefficient is exactly
$\mathbb E_S^{\rm prep}\prod_i p_{i\dagger}$ and is the
original extinction hazard. For phase classes that also
restrict velocities, empirical laws, a passive graph readout,
or other recorded coordinates, replace the coefficient sum
by their actual indicator in the full source/Haar/two-Gaussian
integral. B2 and the cap remain in that integral.
:::

:::{prf:proof}
Substitute (KLQ.1) in the exact position identity (KU.S2).
Conditional on the complete preparation, the final position
Gaussians are independent across rows. Integrating each
coordinate over its true well interval gives $p_{i\ell}$.
The product of their row-label generating polynomials gives
(KLQ.4), and its coefficient extracts the specified count
event. Only afterward average over the actual measured
features, two companion roles, gate/revival decisions,
component rotations and clone jitters. This proves (KLQ.5)
without independence of prepared populations or lumpability
of their well labels. Velocities and readouts do not alter
the already completed positions, but they must be retained
when they enter the phase event or eigenfunction statistic.
$\square$
:::

:::{prf:theorem} Primitive core-to-core transfer intervals with all clone jitters integrated
:label: thm-klq-primitive-core-flux

Let $I_{i,r}\subset[-L_D,L_D]$ be declared coordinate core intervals
and let $C_i=\prod_r I_{i,r}$. Let $G_i$ consist of swarms whose
every row is alive and has position in $C_i$, with arbitrary admissible
capped velocities. There is no constraint on which companions or clone
components are sampled. The output event $G_j$ uses the same velocity
cap, so its only additional restrictions are its actual position and
alive-mark restrictions. Set

$$
g(x)=x-\eta[2x+20\pi\sin(2\pi x)],\qquad
R_v=b\kappa_\nu V_c,\qquad \eta=bt.
$$

For a target interval $I_{j,r}=[l,u]$, define its true Gaussian
landing probability and the two explicit shift envelopes

$$
\begin{aligned}
\Psi_{j,r}(m)&=\Phi((u-m)/\sigma_h)-\Phi((l-m)/\sigma_h),\\
L_{j,r}(x)&=\min\{\Psi_{j,r}(g(x)-R_v),
                         \Psi_{j,r}(g(x)+R_v)\},\\
U_{j,r}(x)&=\Psi_{j,r}\!\left(
 \operatorname{proj}_{[g(x)-R_v,g(x)+R_v]}((l+u)/2)\right).
\end{aligned}
\tag{KLQ.5a}
$$

Write $\phi$ for the standard Gaussian density. The four coordinate
coefficients are

$$
\begin{aligned}
\ell^0_{ijr}&=\inf_{x\in I_{i,r}}L_{j,r}(x),&
u^0_{ijr}&=\sup_{x\in I_{i,r}}U_{j,r}(x),\\
\ell^J_{ijr}&=\inf_{x\in I_{i,r}}
  \int_{\mathbb R}L_{j,r}(x+\sigma_Jz)\phi(z)\,dz,&
u^J_{ijr}&=\sup_{x\in I_{i,r}}
  \int_{\mathbb R}U_{j,r}(x+\sigma_Jz)\phi(z)\,dz.
\end{aligned}
$$

Then, with

$$
a^-_{ij}=\min\left\{\prod_r\ell^0_{ijr},
                            \prod_r\ell^J_{ijr}\right\},\qquad
a^+_{ij}=\max\left\{\prod_ru^0_{ijr},
                            \prod_ru^J_{ijr}\right\},
$$

the original complete killed kernel satisfies the primitive intervals

$$
\boxed{\qquad
(a^-_{ij})^N\le Q_N(S,G_j)\le(a^+_{ij})^N
\quad(S\in G_i).
\qquad}
\tag{KLQ.5b}
$$

These are finite-dimensional Gaussian integrals of the actual nonlinear
force. They retain the possible $N$ dependence of the full-population
crossing event. They are not asserted to be asymptotically sharp enough
to identify the common relative crossing scale of every phase pair.
For phase classes specified by count neighborhoods rather than
homogeneous cores, (KLQ.4) supplies the actual count polynomial;
interval integration of that polynomial must be used for those classes.
:::

:::{prf:proof}
The absolute row-sum bound for either configured viscosity convention
gives $|(bW_Xv^{\rm col})_{i,r}|\le R_v$ for every realization,
including arbitrary clone jitters and Haar components. The function
$\Psi_{j,r}$ is symmetric about the interval midpoint, increases up
to it, and decreases afterward. Its minimum over a closed shift interval
is consequently at an endpoint, and its maximum is at the projection
of the midpoint. Thus the actual row landing probability is between
$\prod_rL_{j,r}(X_{i,r})$ and $\prod_rU_{j,r}(X_{i,r})$.

Freeze the complete measured-feature, companion, gate and component
pattern before its independent clone jitters. Every source position is
in $C_i$, even when several outputs share the same donor. An uncopied
row therefore has its landing probability between the two products
using superscript $0$. A copied row has its own independent Gaussian
jitter in every coordinate. Averaging its two products over that jitter
gives the products using superscript $J$, with the displayed infimum
and supremum accommodating the actual donor. The shift envelopes
have removed the dependence of $W_X$ on every row's jitter only by a
proved pointwise bound, so they do not assume independent prepared
velocities or positions. All output position innovations are independent
conditional on the full preparation. Consequently the all-row event
has lower bound $(a^-_{ij})^N$ and upper bound $(a^+_{ij})^N$ after
averaging the independent row jitters. Mixing the frozen patterns
preserves both bounds. Every target position lies in $D$; B2 and the
radial cap supply the declared capped output velocities without adding
a new event. This proves (KLQ.5b). $\square$
:::

:::{prf:lemma} Explicit integration and spatial optimization errors
:label: lem-klq-gaussian-coefficient-errors

The coefficients in (KLQ.5a)--(KLQ.5b) can be enclosed without clipping
the algorithm's clone jitter. Set

$$
L_g=|1-2\eta|+40\pi^2\eta,\qquad
L_*=L_g\phi(0)/\sigma_h.
$$

Both $L_{j,r}$ and $U_{j,r}$ are $L_*$-Lipschitz. Their Gaussian
averages are also $L_*$-Lipschitz in the source coordinate $x$.
For either function $f$, approximate
$\int_{-Z}^Z f(x+\sigma_Jz)\phi(z)\,dz$ by the midpoint rule
on $m$ equal intervals. Its error, including the omitted Gaussian
tail, is at most

$$
\varepsilon_{Z,m}=
2\Phi(-Z)+\frac{Z^2}{m}
 [\sigma_JL_*\phi(0)+\phi(1)].
\tag{KLQ.5c}
$$

For a source grid containing both endpoints with maximum gap
$\Delta_x$, subtract or add $L_*\Delta_x/2$ to the grid minimum
or maximum. For the jittered coefficients also subtract or add
$\varepsilon_{Z,m}$. Intersect every resulting interval with $[0,1]$.
Monotone products, minima, maxima and the $N$th power then give
rigorous intervals for (KLQ.5b). The parameters $Z,m,\Delta_x$ must
be chosen so these certified errors meet the declared *relative*
crossing error budget in Section 13.4; mere absolute convergence of
the quadrature is not that budget.
:::

:::{prf:proof}
The derivative of the interval Gaussian landing probability satisfies
$|\Psi'|\le\phi(0)/\sigma_h$, and $|g'|\le L_g$.
An infimum or supremum of translates with a common Lipschitz constant
retains that constant, giving the claims for $L,U$ and their averages.
As a function of $z$, the truncated integrand has Lipschitz constant
at most $\sigma_JL_*\phi(0)+\sup_z|z|\phi(z)$; the last supremum
is $\phi(1)$. On an interval of length $\Delta$, midpoint error for
an $L$-Lipschitz function is at most $L\Delta^2/4$.
Summing over $m$ intervals with $\Delta=2Z/m$ gives $LZ^2/m$.
Because $0\le f\le1$, the omitted integral is at most
$2\Phi(-Z)$. Every point of the source interval is at distance
at most $\Delta_x/2$ from the endpoint-inclusive grid, proving the
optimization errors. This truncates a *numerical integral*, never the
noise law of the configured algorithm. $\square$
:::

:::{prf:corollary} Exact label-count moments under the original survival conditioning
:label: cor-klq-surviving-label-moments

Use the well boxes that partition the actual terminal box $D$, with
$\dagger$ denoting exactly its complement. Conditional on the complete
preparation put

$$
\mu=\frac1N\sum_i p_i,\qquad
C=\frac1{N^2}\sum_i[\operatorname{diag}(p_i)-p_ip_i^\top],
\qquad d=\prod_i p_{i\dagger},
$$

and let $e_\dagger$ be the unit vector of the death label. The normalized
count vector $Y$ has these mean and covariance under the raw transition.
Its exact mean and covariance conditional on original nonextinction are

$$
\begin{aligned}
\mu^{\rm surv}&=\frac{\mu-de_\dagger}{1-d},\\
C^{\rm surv}&=
\frac{C+\mu\mu^\top-de_\dagger e_\dagger^\top}{1-d}
-\mu^{\rm surv}(\mu^{\rm surv})^\top.
\end{aligned}
\tag{KLQ.5d}
$$

For an original random preparation law, first compute
$\overline\mu=\mathbb E\mu$,
$\overline C=\mathbb EC+\operatorname{Cov}(\mu)$ and
$\overline d=\mathbb Ed$. The same formula with these barred quantities
is the full surviving transition mean and covariance. Equivalently,
the conditional-survival preparation law is tilted by
$(1-d)/(1-\overline d)$ before its conditional moments are averaged.
At a QSD, $1-\overline d=\alpha_N$. The displayed raw $1/(4N)$
conditional coordinate-variance bound is not silently asserted after
survival conditioning.

If boxes cover only declared cores, their spatial exterior also contains
alive terminal positions. In that case the original extinction indicator
must be integrated separately; being outside every core is not death.
:::

:::{prf:proof}
Raw final labels are conditionally independent, so summing their
categorical means and covariances gives $\mu,C$. On original extinction
every row label is $\dagger$, hence $Y=e_\dagger$ exactly. Subtract
the extinction contributions $de_\dagger$ and
$de_\dagger e_\dagger^\top$ from the raw first and second moments,
divide by $1-d$, and subtract the square of the conditional mean.
This proves (KLQ.5d). For random preparation the total covariance
identity gives $\overline C$ before the same subtraction. Bayes' rule
gives the stated preparation tilt. The QSD identity identifies its
unconditional one-step survival mass with $\alpha_N$. $\square$
:::

(sec-klq-killed-phase-weights)=
## 2. The killed extension of the existing stationary phase-weight calculation

:::{prf:theorem} Exact QSD phase flux and the centered killing correction
:label: thm-klq-killed-phase-balance

Let $\nu_NQ_N=\alpha_N\nu_N$ be the actual finite-$N$ QSD.
For the complete phase partition, define

$$
w_i=\nu_N(G_{i,N}),\quad
f_{ij}=\int_{G_{i,N}}Q_N(S,G_{j,N})\,\nu_N(dS),\quad
\kappa_i=\nu_N(1-Q_N1\mid G_{i,N}),\quad
\bar\kappa=1-\alpha_N.
$$

Conditional coefficients are used only for $w_i>0$; zero-mass
classes have zero flux. Then

$$
\boxed{\qquad
\sum_{i\ne j}f_{ij}-\sum_{i\ne j}f_{ji}
=(\kappa_j-\bar\kappa)w_j.
\qquad}
\tag{KLQ.6}
$$

Unlike a conservative invariant law, a QSD has the displayed
phase-dependent killing correction. Identical killing rates
cancel exactly.

Use the same actual-kernel phase-transfer hypotheses, scale
$c_N$, generator $A$, matrix $B$, vector $\theta$ and inverse
constant $K_A$ as {prf:ref}`thm-slcj-phase-weights`, replacing
$P_N(S,G_j)$ there by $Q_N(S,G_j)$ here. In particular,

$$
\left|Q_N(S,G_{j,N})/c_N-a_{ij}\right|\le\epsilon_N
\quad(i\ne j,\ i,j\ge1),\qquad
Q_N(S,G_{0,N})\le c_N\eta_N,
$$

and $w_0\le\delta_N$. Define the actual centered defect

$$
\mathcal D_{\kappa,N}=\sum_{j=1}^m
w_j|\kappa_j-\bar\kappa|.
$$

Then the existing finite-dimensional calculation yields

$$
\boxed{\quad
\|w-\theta\|_1\le K_A\left[
2(m-1)\epsilon_N+\eta_N+
\frac{\delta_N+\mathcal D_{\kappa,N}}{c_N}+\delta_N\right].
\quad}
\tag{KLQ.7}
$$

Every entry is a full-kernel phase flux such as (KLQ.5), with
integration and truncation errors charged at the relative
crossing scale $c_N$. The theorem assumes no globally common
phase, no independent phase-label chain, and no attraction
between different initialized populations.
:::

:::{prf:proof}
The QSD identity applied to $\mathbf1_{G_j}$ gives
$\sum_i f_{ij}=\alpha_N w_j$.
Summing the outgoing row gives
$\sum_i f_{ji}=(1-\kappa_j)w_j$.
Subtract and remove the common diagonal flux, obtaining
(KLQ.6).

For $i,j\ge1$, the uniform conditional transfer bound gives
$|f_{ij}/c_N-w_i a_{ij}|\le w_i\epsilon_N$.
As in the original proof, each directed error enters the
vector balance twice, so its total $\ell^1$ contribution is
at most $2(m-1)\epsilon_N$. Exterior incoming and outgoing
fluxes total at most $\delta_N+c_N\eta_N$.
Equation (KLQ.6) contributes the additional vector of
coordinates $w_j(\kappa_j-\bar\kappa)/c_N$, whose $\ell^1$
norm is $\mathcal D_{\kappa,N}/c_N$. Consequently

$$
\|wA\|_1\le2(m-1)\epsilon_N+\eta_N+
(\delta_N+\mathcal D_{\kappa,N})/c_N.
$$

The first $m-1$ coordinates of $(w-\theta)B$ are those
of $wA$, and its last is $-w_0$. The same adjugate inverse
bound from {prf:ref}`thm-slcj-phase-weights` proves (KLQ.7).
All operations concern the original killed fluxes.
$\square$
:::

:::{prf:corollary} Computable killing error and the actual crossing scale
:label: cor-klq-killing-crossing-comparison

The elementary bound
$\mathcal D_{\kappa,N}\le2\bar\kappa$
is valid. A sharper bound is obtained from any explicit
reference number $\kappa_*$ and primitive phase intervals
$1-Q_N1(S)\in[d_i^-,d_i^+]$ for $S\in G_i$:

$$
\mathcal D_{\kappa,N}
\le2\sum_{i=0}^m w_i
\max\{|d_i^--\kappa_*|,|d_i^+-\kappa_*|\}.
\tag{KLQ.8}
$$

For an exterior interval $[0,1]$ its contribution is at most
$2\delta_N\max\{|\kappa_*|,|1-\kappa_*|\}$.
The intervals are computed from the actual all-dead integral
in (KLQ.4)--(KLQ.5), or the already proved safe-center/noise
and averaged-energy extinction estimates.

If $\bar\kappa\le C_\dagger e^{-NI_\dagger}$ and
$c_N\ge c_*e^{-NI_{\rm cross}}$ are actually certified, then

$$
\mathcal D_{\kappa,N}/c_N
\le(2C_\dagger/c_*)e^{-N(I_\dagger-I_{\rm cross})}.
\tag{KLQ.9}
$$

Thus the original conservative phase weights remain the
identified QSD weights when the verified killing correction
vanishes relative to communication. If it does not, retain
the killed phase generator; replacing it by conservative
phase weights is unjustified. Inter-phase communication may
depend exponentially on $N$ exactly as allowed by Section
14.5, while the local row and normalized moment coefficients
remain independent of $N$.
:::

:::{prf:proof}
Since $\kappa_i,\bar\kappa\ge0$ and
$\sum_i w_i\kappa_i=\bar\kappa$, extending the sum to
all phases gives $\sum_iw_i|\kappa_i-\bar\kappa|\le2\bar\kappa$.
Also
$\sum_iw_i|\kappa_i-\bar\kappa|
\le\sum_iw_i|\kappa_i-\kappa_*|+|\bar\kappa-\kappa_*|
\le2\sum_iw_i|\kappa_i-\kappa_*|$.
The uniform phase intervals prove (KLQ.8). Insert the two
certified exponential bounds to get (KLQ.9).
$\square$
:::

:::{prf:definition} Exact finite killed phase matrix
:label: def-klq-finite-killed-matrix

Include every nonextinct class, including the exterior, and
set $T_{ij}=f_{ij}/w_i$ when $w_i>0$.
Discard zero-mass classes from this finite matrix.
This is the QSD-conditional average of the actual transition,
not a lumped Markov model asserted for arbitrary trajectories.
Its row sums are $1-\kappa_i$ and

$$
wT=\alpha_Nw.
$$

For any declared $c_N>0$, write
$T=I+c_N A_N-\operatorname{diag}(\kappa_i)$, where
$(A_N)_{ij}=T_{ij}/c_N$ for $i\ne j$ and its row sums
are zero. Then exactly

$$
w\left[A_N-\operatorname{diag}(\kappa_i/c_N)\right]
=-\frac{1-\alpha_N}{c_N}w.
\tag{KLQ.10}
$$

The killing-biased phase weights are the positive left
eigenvector of this finite matrix. Uniform primitive bounds
on (KLQ.5) enclose its entries even though the averaging
law is the actual QSD. The phase rates are retained instead
of being set equal or replaced by one global contraction.
:::

(sec-klq-eigenfunction-phases)=
## 3. Exact eigenfunction weights and landscape orbits

:::{prf:theorem} Finite-dimensional weighted eigenfunction balance
:label: thm-klq-phase-eigenfunction-balance

Let $h_N>0$ be the positive eigenfunction of the actual
killed kernel associated with the QSD eigenvalue,
$Q_Nh_N=\alpha_Nh_N$, and suppose $0<\nu_Nh_N<\infty$.
For each positive-mass class let

$$
H_i=\nu_N(h_N\mid G_i),\qquad
\mathsf B_{ij}=
\frac{\nu_N(Q_N(h_N\mathbf1_{G_j})\mid G_i)}{H_j}.
\tag{KLQ.11}
$$

Then the exact finite-dimensional identities are

$$
\boxed{\qquad
\mathsf BH=\alpha_NH,\qquad
w\mathsf B=\alpha_Nw.
\qquad}
\tag{KLQ.12}
$$

If an actual within-phase certificate bounds
$R_i=\sup_{G_i}h_N/\inf_{G_i}h_N<\infty$, then
the primitive transition intervals
$q_{ij}^-\le Q_N(S,G_j)\le q_{ij}^+$ for $S\in G_i$
give

$$
q_{ij}^-/R_j\le\mathsf B_{ij}\le R_jq_{ij}^+,
\qquad
\frac{h_N(S)}{h_N(T)}\le R_iR_j\frac{H_i}{H_j}
\quad(S\in G_i,\ T\in G_j).
\tag{KLQ.13}
$$

The phase factors $H_i/H_j$ are retained. They are not
silently replaced by one merely because local relaxation
coefficients are independent of $N$.

For irreducible $\mathsf B$, let
$\mathsf M=\alpha_N I-\mathsf B$. Its adjugate satisfies,
for every fixed column $r$ and every $i,j$,

$$
\frac{H_i}{H_j}
=\frac{\operatorname{adj}(\mathsf M)_{ir}}
{\operatorname{adj}(\mathsf M)_{jr}}.
\tag{KLQ.14}
$$

The ratios are independent of the choice of nonzero column.
Entries of (KLQ.11) are full actual source/Haar/kinetic
integrals with the displayed target weight; (KLQ.13) supplies
primitive enclosures when the within-phase certificate has
been discharged. No lumpability hypothesis is used.
The weighted matrix itself depends on the unknown eigenfunction:
(KLQ.14) is an exact identity, not a primitive estimate of its ratio.
It cannot discharge the eigenfunction certificate by itself.
:::

:::{prf:proof}
Integrate $Q_Nh_N=\alpha_Nh_N$ under the conditional law
on $G_i$, and split its output over the complete phase
partition. This proves the right identity in (KLQ.12).
For the left identity,

$$
\sum_i w_i\mathsf B_{ij}
=\frac{\nu_NQ_N(h_N\mathbf1_{G_j})}{H_j}
=\alpha_N\frac{\nu_N(h_N\mathbf1_{G_j})}{H_j}
=\alpha_Nw_j.
$$

Since $\inf_{G_j}h_N\le H_j\le\sup_{G_j}h_N$,
the ratio $h_N/H_j$ lies in $[1/R_j,R_j]$ on $G_j$.
Insert these bounds in the defining positive integral for
$\mathsf B_{ij}$ and average the uniform $q$ bounds to
get (KLQ.13). The same inequalities give its statewise
comparison through $H_i,H_j$.

The finite matrix
$\mathsf P_{ij}=\mathsf B_{ij}H_j/(\alpha_NH_i)$
is stochastic by the right identity and irreducible when
$\mathsf B$ is. The finite-state maximum principle implies
that its right harmonic functions are constant: at a maximum,
every positively connected successor must have the same value,
and irreducibility propagates equality. Its left invariant
space is one-dimensional by the same finite irreducible
stationary argument used in the existing phase-weight theorem.
Thus $\mathsf M$ has rank one less than its dimension;
its adjugate is a nonzero constant multiple of $Hw$.
Every $w_r$ is positive, so the ratios in its $r$th column
are exactly (KLQ.14). $\square$
:::

:::{prf:lemma} Primitive phase cone and path comparison
:label: lem-klq-unweighted-phase-cone

For the complete phase partition, suppose the actual eigenfunction
has finite extrema $0<m_i=\inf_{G_i}h_N\le M_i=\sup_{G_i}h_N<\infty$.
Let $K^-_{ij}\le Q_N(S,G_j)\le K^+_{ij}$ hold for every $S\in G_i$.
These bounds may be computed from (KLQ.4)--(KLQ.5c), the Section 8
residence and discovery calculations, or the same full-state Gaussian
integral with additional velocity restrictions. Then the unweighted
primitive matrices give the rigorous cone inequalities

$$
\boxed{\qquad
K^-m\le\alpha_Nm,\qquad
\alpha_NM\le K^+M.
\qquad}
\tag{KLQ.14a}
$$

If an independently proved within-phase estimate supplies
$M_i/m_i\le R_i$, every directed path
$j=i_0\longrightarrow i_1\longrightarrow\cdots\longrightarrow i_L=i$
with $K^-_{i_{r-1},i_r}>0$ gives

$$
\frac{H_i}{H_j}
\le\alpha_N^L
\prod_{r=1}^L\frac{R_{i_r}}{K^-_{i_{r-1},i_r}}
\le\prod_{r=1}^L\frac{R_{i_r}}{K^-_{i_{r-1},i_r}}.
\tag{KLQ.14b}
$$

Taking the minimum of these bounds over positive paths is a completely
explicit phase-factor estimate once the *independent* $R_i$ and actual
unweighted intervals have been computed. In particular a lower edge
bound $K^-_{ij}\ge c_N(a_{ij}-\epsilon_N)>0$ retains the factor
$c_N^{-L}$ in this comparison. It does not discard an exponentially
small phase-crossing rate. Neither (KLQ.14a) nor the existence of
Gaussian full support alone supplies the missing within-phase $R_i$.
:::

:::{prf:proof}
For each $S\in G_i$, split its positive eigenfunction integral over
the complete partition and insert the target extrema. Taking the source
infimum or supremum gives (KLQ.14a). Under the independently verified
$R_i$ hypothesis, (KLQ.13) gives
$\mathsf B_{ab}\ge K^-_{ab}/R_b$. For a directed edge $a\to b$,
the eigenvector equation therefore gives
$H_a\ge K^-_{ab}H_b/(\alpha_NR_b)$, equivalently
$H_b/H_a\le\alpha_NR_b/K^-_{ab}$.
Multiply this upper bound along the displayed path from $j$ to $i$.
Since a killed
kernel has $0<\alpha_N\le1$, this proves both bounds in (KLQ.14b).
Its coefficients are the unweighted primitive intervals and the stated
independently proved local constants; the unknown entries of
$\mathsf B$ do not have to be evaluated. $\square$
:::

:::{prf:corollary} QSD and Doob phase weights determine the retained ratio
:label: cor-klq-qsd-doob-phase-ratio

Let $\pi_N=h_N\nu_N/\nu_Nh_N$ be the actual Doob invariant
law, and let $\widehat w_i=\pi_N(G_i)$. Then exactly

$$
\widehat w_i=\frac{w_iH_i}{\nu_Nh_N},\qquad
\frac{H_i}{H_j}=
\frac{\widehat w_i/w_i}{\widehat w_j/w_j}.
\tag{KLQ.15}
$$

This distinguishes the conditional physical law from its
conservative Doob law. An identified phase mixture for one
does not supply equal phase weights for the other.
:::

:::{prf:proof}
Integrate the defining reweighting over each phase, then
divide the two identities. $\square$
:::

:::{prf:theorem} Exact symmetry-orbit equality without inter-phase contraction
:label: thm-klq-rastrigin-symmetry-orbits

Suppose the actual configured kernel is equivariant under
a group $\mathcal G$ of signed coordinate permutations,
acting simultaneously on positions and velocities, and its
finite-$N$ QSD and positive eigenfunction are unique up to
normalization. If $G_j=gG_i$ for $g\in\mathcal G$, then

$$
\nu_N(G_j)=\nu_N(G_i),\qquad
\pi_N(G_j)=\pi_N(G_i),\qquad H_j=H_i,
\qquad h_N(gS)=h_N(S).
\tag{KLQ.16}
$$

For standard Rastrigin with symmetric terminal box and the
canonical isotropic feature-distance, Gaussian companion,
Gaussian innovation, component-Haar and radial-cap conventions,
the equivariance is verified directly below. Integer
translations are not symmetries of $U$ because its quadratic
term changes. Therefore different well orbits can carry
different weights. The statement applies to symmetry-related
population phases, rather than declaring that every force
zero is a stationary population phase.
:::

:::{prf:proof}
For a signed coordinate permutation $g$,
$U(gx)=U(x)$ and $F(gx)=gF(x)$ follow from (KLQ.1).
The symmetric box and terminal marking are preserved.
The configured feature-distance and physical Gaussian
neighbor kernels are unchanged under the same orthogonal
action. Thus both companion probability arrays are unchanged;
raw reward, measured diversity, their alive-only means and
standard deviations, fitness and acceptance decisions have
the same laws. Source copying commutes with $g$.
Gaussian jitter, thermostat and final-position innovations
are orthogonally invariant. A component rotation is mapped
to $gOg^{-1}$, whose Haar law is the same, so the actual
component collision commutes in law. Both Gaussian-viscous
normalizations preserve their scalar weights and transform
their vector outputs by $g$. The radial cap commutes with
orthogonal transformations. These checks prove full physical
kernel equivariance, including all retained marks and
velocities. A passive readout is transformed by its declared
pushforward and has no feedback; if a phase uses it, its
class must also be transformed by that pushforward.

The pushforward $g_\#\nu_N$ is another QSD with the same
eigenvalue, so uniqueness gives $g_\#\nu_N=\nu_N$.
Likewise $h_N\circ g=c_gh_N$ for some $c_g>0$ by uniqueness up
to normalization. A signed coordinate permutation has a finite order
$r$, so iteration gives $h_N=c_g^rh_N$ and hence $c_g=1$.
The Doob invariant law is then invariant as well.
Their phase identities and the equality of conditional
eigenfunction scales prove (KLQ.16). No mixing or attraction
between the two phases was used. $\square$
:::

(sec-klq-existing-machinery)=
## 4. Connection to the existing within-phase and crossing certificates

:::{prf:remark} Quantifiers and coefficients retained from Chapter 06a
:label: rem-klq-existing-phase-machinery

The Rastrigin core residence certificate in Section 8 applies
separately to its declared low-noise configurations and
cores. Its local Hessian, jitter and Gaussian-tail constants
are not transplanted into a different configured timestep,
force normalization or noise scale. Formula (KLQ.3) permits
re-evaluating the same safe-center calculation with the
actual coupled first kick: its rowwise bounded collision
term is $b\kappa_\nu V_c$, while (KLQ.2) and each well's
local force profile retain the sinusoidal landscape.

Section 12's full-law increment and residual-dissipation
certificates follow one nonlinear population trajectory and
may select different stationary limits in different phases.
They are used with that same quantifier. They do not require
arbitrary initial swarms to contract toward one common law.
They also do not alone prove a full finite-array eigenfunction
oscillation bound $R_i$ in (KLQ.13); any such claimed bound
must use the appropriate actual finite-particle phase
certificate and its proved integration and exit budgets.

Section 13.4's actual-flux matrix and adjugate inverse are
retained in (KLQ.6)--(KLQ.10). The only change for the QSD
is the explicitly computed centered killing correction.
When that correction competes with crossings, the killed
finite matrix retains it rather than suppressing it.
Section 14.5's population-dependent phase-crossing scale
is consistent with this balance: uniform local coefficients
and different phase weights do not imply uniform global
inter-phase mixing or a globally equal eigenfunction.

Equations (KLQ.13)--(KLQ.16) provide the orbit-resolved
comparison. Uniform within-phase constants, once discharged,
multiply the actual finite-dimensional phase ratios. Symmetry
orbits have exact ratio one; other orbit ratios retain the
landscape-dependent transition and extinction information.
No phase boundary has been made absorbing and no force or
unbounded innovation has been altered.
:::
