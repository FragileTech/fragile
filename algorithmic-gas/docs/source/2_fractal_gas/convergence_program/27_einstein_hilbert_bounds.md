(sec-eh-rust-bounds)=
# Einstein–Hilbert Gas: Native Bounds and Population Limits

:::{div} feynman-prose
Follow the cloud relative to its own center, and calculate what each implemented stage does to its spread. The signed cloning and kinetic formulas combine into an exact twenty-phase drift decomposition, with quantitative bounds on the accumulated random residual. Concentration, Wasserstein estimates, and conditional population limits connect these stage calculations to the evolving law. From the reference start, the first transition also has an explicit Gaussian limit. We keep three conclusions separate: convergence at fixed $N$, moment bounds uniform over $N$, and mixing rates independent of $N$. The operator estimates identify quantities needed to investigate each; they do not equate these different claims.
:::

(sec-eh-native-scope)=
## 1. The transition and its domain

:::{div} feynman-prose
First fix the transition whose moments we will calculate. Cloning uses mutual pairs and keeps each walker's velocity; the subsequent graph remains frozen through both kinetic B stages, while their curl fits use the current positions and velocities. That order determines the centered drift. The mathematical formulas retain the implementation's regularization and geometry choices, with successful geometry evaluations as an explicit requirement. Finite precision and resource budgets enter when we compare those formulas with executable runs; they are not substitutes for probabilistic moment estimates.
:::

:::{prf:definition} Real-coordinate interpretation of the native Einstein–Hilbert update
:label: def-eh-native-interpretation

Fix the component tuple of {prf:ref}`def-variant-einstein-hilbert`, including
the mutual matchings, the period $q=20$, zero cloning jitter, identity elastic
collision, relative-trace covariance metric, and both graph Boris B stages.
Use the post-cloning graph and its weights in both B stages; recompute the
curl using each stage's current positions and velocities. Put

$$
a=\frac{h\nu}{4},\qquad c=e^{-\gamma h},\qquad
\sigma^2=T(1-c^2),\qquad \gamma=1,\quad \nu=3.
$$

The **real-coordinate interpretation** evaluates these specified formulas
with real arithmetic and independent ideal Gaussian and uniform innovations.
The positive numerical ridge parameters and the geometry rules remain part
of the formulas. Statements below about an executed step require that every
geometry evaluation returns its specified finite output. A statement over a
time interval requires this at every refresh in that interval, almost surely.
This requirement has not been established for all states of the native
implementation. It is not a conditioning operation on successful runs.

The Rust engine instead uses addressed pseudorandom streams, the selected
floating-point precision, finite index ranges, and explicit resource budgets.
An execution error is not an absorbing-boundary death and does not define a
quasi-stationary transition. No limit $N\to\infty$ is taken with a fixed memory
budget or fixed graph index range. Such a limit concerns the mathematical
family of these formulas; executables approximate individual members.
:::

:::{prf:remark} Actual covariance regularization and finite execution
:label: rem-eh-relative-metric

Let $C_i$ be the neighbor displacement covariance in projected dimension $d'$.
The default `RidgeScale::RelativeToTrace` uses

$$
\tau_i=\begin{cases}\operatorname{tr}(C_i)/d',&\operatorname{tr}(C_i)>0,\\
1,&\operatorname{tr}(C_i)=0.\end{cases}
$$

It pseudoinverts $C_i+10^{-5}\tau_i I$, with cutoff
$d'\epsilon_{\mathrm{mach}}\lambda_{\max}$, then clamps the metric eigenvalues
below by $10^{-6}/\tau_i$. There is no configured upper eigenvalue clamp.
Consequently the constants $10^{-5}$ and $10^{-6}$ are not absolute coercivity
or boundedness constants for the metric as the population changes.
For a noncollapsed rescaling $x\mapsto\lambda x$, the metric scales as
$\lambda^{-2}$ as long as the same covariance branch is used.

Finite coordinates do not ensure successful execution. For example, the
default duplicate policy lifts $N$ coincident walkers to a clique with
$N(N-1)/2$ undirected edges. With $N=11$ and `max_batch_elements=99`, the
geometry receives an undirected edge budget of $49$ and rejects the $55$
required edges. The regression in `tests/einstein_hilbert_gas.rs` executes
this case. `EmptyGraph` handles a mesh failure, not this subsequent edge-budget
failure. Fitness underflow, nonfinite derived fields, local solver failures,
and memory exhaustion are also errors. A finite resource budget cannot be
used to prove a population-uniform probabilistic bound.
:::

:::{prf:lemma} Native curvature balance, action bounds, and spatial scaling
:label: lem-eh-native-curvature-balance

For a successful geometric evaluation in projected dimension $d'\ge2$, put
$b_i=\sqrt{\max(\det g_i,10^{-12})}$ and $u_i=(\log b_i)/d'$.
Let $\ell_{ij}$ be its symmetric geodesic edge length. The actual curvature
weights and reversible masses are

$$
k_{ij}=\frac1{\sqrt{\max(\ell_{ij}^2,10^{-8})}+10^{-8}},\quad
D_i=\max\left\{\sum_{j\sim i}k_{ij},10^{-12}\right\},\quad
C_{ij}=k_{ij}/D_i,\quad \mu_i=\frac{D_i}{\sum_l D_l}.
$$

Write $\mathcal E_u=\sum_{i,j}\mu_i C_{ij}(u_j-u_i)^2$ and
$R_i=2(d'-1)\sum_j C_{ij}(u_i-u_j)$. Then, including normalization floors
and isolated vertices,

$$
\begin{aligned}
\sum_i\mu_iR_i&=0, &\sum_i\mu_i u_iR_i&=(d'-1)\mathcal E_u,\\
\sum_i\mu_i b_iR_i
 &=(d'-1)\sum_{i,j}\mu_i C_{ij}(b_i-b_j)(u_i-u_j)\ge0,\\
d'(d'-1)b_{\min}\mathcal E_u
 &\le\sum_i\mu_i b_iR_i\le d'(d'-1)b_{\max}\mathcal E_u,\\
\sum_i\mu_iR_i^2&\le4(d'-1)^2\mathcal E_u,
&\max_i|R_i|&\le2(d'-1)\operatorname{osc}(u).
\end{aligned}
\tag{EH.G1}
$$

The numerical constants do not depend on $N$. Here $b_{\min},b_{\max}$ and
$\operatorname{osc}(u)$ are quantities of the supplied geometry; no uniform
bound on them is asserted. The balanced average uses the curvature masses
$\mu$, which differ from the viscosity masses in
{prf:ref}`lem-eh-frozen-reversible-energy`.
The native total action is $\sum_i r_i$; (EH.G1) controls the explicitly
weighted action $\sum_i\mu_i r_i$.

Under a dilation $x\mapsto sx$, $s>0$, assume the projected neighbor graph
is unchanged, all local covariance traces are positive, and the determinant
floor is inactive at both scales. The native relative-trace metric and
curvature then satisfy

$$
g_i(sx)=s^{-2}g_i(x),\quad b_i(sx)=s^{-d'}b_i(x),\quad
\ell_{ij}(sx)=\ell_{ij}(x),\quad R_i(sx)=R_i(x),\quad
r_i(sx)=s^{-d'}r_i(x).
\tag{EH.G2}
$$

Thus a curvature histogram alone does not determine the spatial scale of
the centered cloud. Equation (EH.G2) is a geometric transformation law;
the fixed-temperature stochastic transition is not asserted to commute
with dilation.
:::

:::{prf:proof}
Symmetry gives $\mu_iC_{ij}=k_{ij}/\sum_lD_l=\mu_jC_{ji}$.
Pairing the two orientations of every edge proves the first three identities
in (EH.G1). Since $b_i=e^{d'u_i}$, the mean-value theorem gives
$d'b_{\min}(u_i-u_j)^2\le(b_i-b_j)(u_i-u_j)
\le d'b_{\max}(u_i-u_j)^2$, proving the action bounds. Each row sum of
$C$ is at most one. Weighted Cauchy--Schwarz gives
$R_i^2\le4(d'-1)^2\sum_j C_{ij}(u_i-u_j)^2$; summing proves the
squared-curvature estimate. The oscillation bound follows directly from
the row-sum bound.

Under the stated dilation, the displacement covariance and its relative
ridge multiply by $s^2$. The pseudoinverse cutoff is relative to its largest
eigenvalue, and the metric floor is $10^{-6}/\tau_i$; hence both transform
with the covariance scale and the metric multiplies by $s^{-2}$.
The edge quadratic forms are unchanged. The inactive determinant floor
therefore gives $b_i\mapsto s^{-d'}b_i$ and $u_i\mapsto u_i-\log s$.
The curvature weights are unchanged and their Laplacian annihilates
constants. This proves (EH.G2), including the reward identity $r_i=b_iR_i$.
:::

(sec-eh-graph-estimates)=
## 2. Bounds for the graph kicks and thermostat

:::{div} feynman-prose
The kinetic calculation starts with three concrete operations: a convex graph average, a norm-preserving rotation, and an OU thermostat. Their bounds control the velocity terms that enter the centered position drift. Row averages control the largest speed; column sums control amplification of empirical moments, so population-independent estimates need the stated column bound. A frozen graph also has a reversible weighted energy. To carry that energy across a geometry refresh, one must compare the old and new weights explicitly. Keeping these constants visible makes the drift estimate quantitative.
:::

:::{prf:lemma} Convex graph kicks and the correct empirical moment constant
:label: lem-eh-convex-kick

For a frozen graph let $w_{ij}\ge0$, $w_{ii}=0$, and
$s_i=\sum_jw_{ij}\le1$. Suppose $a\max_i s_i\le1$, which follows from
$h\nu\le4$. Define

$$
P_{ij}=a w_{ij}\ (j\ne i),\qquad P_{ii}=1-a s_i,\qquad
\kappa=\max_j\left(1-a s_j+a\sum_iw_{ij}\right).
$$

Then $P$ is stochastic, $\kappa\ge1$, and for every $p\ge1$,

$$
\max_i |(Pv)_i|\le\max_i|v_i|,\qquad
\frac1N\sum_i|(Pv)_i|^p\le
\kappa\frac1N\sum_i|v_i|^p.
$$

The native free B stage is $v\mapsto P Q(v,x) P v$, where $Q$ is block
diagonal with one orthogonal Cayley matrix per row. Consequently its maximum
speed does not increase, and its empirical $p$-moment increases by at most
$\kappa^2$. These constants do not depend on the curl magnitude. The map
$v\mapsto Q(v,x)v$ need not be a contraction between two different inputs.
:::

:::{prf:proof}
Nonnegativity follows from $a s_i\le1$, and every row sums to one. Jensen's
inequality gives $|(Pv)_i|^p\le\sum_jP_{ij}|v_j|^p$. Taking a maximum proves
the first bound; summing over rows and then taking the maximum column sum
proves the second. The average column sum is one, so $\kappa\ge1$.
Each Cayley matrix is orthogonal by {prf:ref}`prop-variant-eh-identities` and
therefore preserves its row norm for each realized input, even though the
matrix depends on that input. Apply the Jensen bound before and after these
rotations. No difference estimate for input-dependent rotations was used.
$\square$
:::

:::{prf:lemma} Reversible energy for one frozen native geometry
:label: lem-eh-frozen-reversible-energy

Let $b_i=\sqrt{\max\{\det g_i,10^{-12}\}}>0$ and let
$k_{ij}=\exp[-d_g(i,j)^2/(2\ell^2)]$ on the undirected tessellation edges.
Put

$$
Z_i=\sum_{j\sim i}k_{ij}b_j,\quad D_i=\max\{Z_i,10^{-12}\},\quad
\pi_i=\frac{b_iD_i}{\sum_l b_lD_l}.
$$

For the actual weights $w_{ij}=k_{ij}b_j/D_i$, one has
$\pi_iw_{ij}=\pi_jw_{ji}$. Under the convexity condition of
{prf:ref}`lem-eh-convex-kick`, the same $\pi$ is invariant for $P$.
Both B stages are contractions of $\sum_i\pi_i|v_i|^p$, $p\ge1$, for this
frozen geometry. In particular, conditional on the post-cloning state,

$$
\mathbb E\left[\sum_i\pi_i|v_i^+|^2\mid\widetilde S\right]
\le c^2\sum_i\pi_i|v_i|^2+d\sigma^2.
$$

The weights on the two sides are those of the same post-cloning graph,
including when positions and curl change during BAOAB.
:::

:::{prf:proof}
Symmetry of the graph and of $d_g(i,j)^2$ gives
$b_iD_iw_{ij}=b_ib_jk_{ij}=b_jD_jw_{ji}$. The diagonal completion in $P$
therefore gives detailed balance and $\pi P=\pi$, also for an isolated row.
Multiply Jensen's inequality by $\pi_i$ and sum. Rowwise rotations preserve
the weighted sum exactly. At the O stage, conditional on its input $u$,
$\mathbb E|cu_i+\sigma\xi_i|^2=c^2|u_i|^2+d\sigma^2$.
The post-cloning graph, and hence $\pi$, is fixed before this noise is drawn.
Apply the deterministic weighted contraction to B2, then the conditional
OU identity, then the B1 contraction. $\square$
:::

:::{prf:remark} Why the frozen estimate cannot be iterated with new weights
:label: rem-eh-weight-transfer

At the next refresh the values $b_i,D_i,\pi_i$ change. The preceding lemma
does not compare the old and new weighted energies. Neither a bound
$\sup_{N,i}N\pi_i<\infty$, a positive lower bound on $N\pi_i$, nor a
uniform transfer ratio between successive weights follows from row
normalization. Relative-trace regularization is not such a bound.
The normalized unweighted energy need not decrease even in one quarter-kick:
on a five-node star with center-to-leaf weights $1/4$, leaf-to-center weights
$1$, $a=1$, and only the center moving, its sum of squared speeds grows from
$1$ to $4$. This example tests what row normalization alone implies; it is
not asserted to be a particular native geometry realization.
:::

:::{prf:theorem} Velocity and position bounds with their population dependence
:label: thm-eh-native-moments

Under {prf:ref}`def-eh-native-interpretation`, suppose $h\nu\le4$ and all
rows initially live. Let $R_n=\max_i|v_{n,i}|$, $X_n=\max_i|x_{n,i}|$, and
$Z_n=\max_i|\xi_{n,i}|$. For every successfully defined real-coordinate
trajectory,

$$
R_{n+1}\le cR_n+\sigma Z_n,\qquad
X_{n+1}\le X_n+\frac h2[(1+c)R_n+\sigma Z_n].
$$

In particular, with $L_{N,d}=2d\log(2dN)+2d$,

$$
\|R_n\|_{L^2}\le c^n\|R_0\|_{L^2}
+\sigma\frac{1-c^n}{1-c}\sqrt{L_{N,d}}.
$$

This is a bound uniform in time for each fixed $N$, with logarithmic
population dependence, not a uniform-in-$N$ empirical moment estimate.

For an additional, explicit hypothesis, suppose every post-cloning graph on
the interval satisfies $\kappa\le K$ for the constant of
{prf:ref}`lem-eh-convex-kick`, with $K$ independent of $N$. Set
$E_n=\mathbb E[N^{-1}\sum_i|v_{n,i}|^2]$ and $\lambda=c^2K^4$. Then

$$
E_n\le\lambda^n E_0+K^2d\sigma^2\sum_{j=0}^{n-1}\lambda^j.
$$

This estimate is uniform in $N$ on finite horizons if $\sup_NE_0<\infty$.
It is uniform also in time if $\lambda<1$. No such $K$ and strict margin
have been established for every geometry generated by the default preset.
:::

:::{prf:proof}
Cloning keeps each frozen velocity and copies positions within disjoint
pairs, so it increases neither maximum. B1 and B2 do not increase the
maximum velocity norm by {prf:ref}`lem-eh-convex-kick`. At O the triangle
inequality gives the first recursion. The two drifts use the B1 and O
velocities, respectively, giving the second. Iterating the first and using
Minkowski gives the claimed $L^2$ bound once $\mathbb E Z_n^2\le L_{N,d}$.
For $M=\max_{i,b}|\xi_{i,b}|$, the Gaussian exponential moment and a union
bound give $\mathbb P(M^2>t)\le\min\{1,2dN e^{-t/2}\}$. Integrating this
bound, split at $2\log(2dN)$, yields
$\mathbb EM^2\le2\log(2dN)+2$. Now $Z_n^2\le dM^2$.

For the empirical estimate, both B stages multiply the sum of squared norms
by at most $K^2$. The graph is fixed before O, and the Gaussian innovations
are independent of its input. Thus
$E_{n+1}\le K^2(c^2K^2E_n+d\sigma^2)$. Iteration proves the assertion.
Positions also have uniform finite-horizon second moments under these
additional hypotheses: if $S_n=\mathbb E N^{-1}\sum_i|x_{n,i}|^2$, disjoint
pair cloning gives $\widetilde S_n\le2S_n$, and Minkowski gives

$$
\sqrt{S_{n+1}}\le\sqrt{2S_n}
+\frac h2\left[K\sqrt{E_n}+\sqrt{c^2K^2E_n+d\sigma^2}\right].
$$

This positional estimate grows with the horizon and proves no confinement.
$\square$
:::

(sec-eh-matched-selection)=
## 3. Uniform population estimates for the actual selection stage

:::{div} feynman-prose
The native sample normalizer supplies useful control before we estimate any raw reward moments: each standardized channel has mean zero and empirical second moment at most one. Consequently only a uniformly controlled fraction of walkers can have extremely small fitness. That lower-tail estimate gives a population-independent continuity modulus for cloning, while uniform mutual pairing gives exact conditional averages and fluctuations of order $1/N$ in variance for bounded observables. Together these results advance the selection law when its fitness-marked input converges. They require neither a common positive minimum fitness nor a moment bound on the raw curvature reward.
:::

:::{prf:theorem} Matched cloning: conditional mean, variance, and population map
:label: thm-eh-matched-cloning-limit

Fix $N\ge2$ live rows $z_i=(x_i,v_i,f_i)$ with $f_i>0$, after the distance
matching and fitness computation. At an open gate let

$$
A(f,g)=\min\{1,(g-f)_+/f\},\qquad \eta_N=\frac1N\sum_i\delta_{z_i}.
$$

Draw the independent uniform mutual cloning matching and independent
acceptance uniforms prescribed by the native algorithm. Let $\widetilde\mu_N$
be the empirical position-velocity law after the identity elastic collision.
For a bounded measurable $\varphi(x,v)$ with $|\varphi|\le B$, define

$$
\mathcal C(\eta)\varphi=\int\varphi(x,v)\,\eta(dz)
+\iint A(f,g)[\varphi(y,v)-\varphi(x,v)]\,\eta(dz)\eta(dz'),
\quad z'=(y,w,g).
$$

Uniformly over these fixed input rows,

$$
\left|\mathbb E[\widetilde\mu_N\varphi\mid z]-\mathcal C(\eta_N)\varphi\right|
\le\begin{cases}2B/(N-1),&N\text{ even},\\0,&N\text{ odd},\end{cases}
\qquad
\operatorname{Var}(\widetilde\mu_N\varphi\mid z)\le\frac{65B^2}{N}.
$$

If $\eta_N$ converges weakly in probability to a deterministic probability
$\eta$ on $\mathbb R^{2d}\times(0,\infty)$, then $\widetilde\mu_N$ converges
weakly in probability to $\mathcal C(\eta)$. At a closed gate the map is the
position-velocity projection of $\eta$. The theorem permits dependence among
the fitness marks and arbitrary state-dependent geometric rewards.
:::

:::{prf:proof}
Write $H_{ij}=A(f_i,f_j)[\varphi(x_j,v_i)-\varphi(x_i,v_i)]$, so
$H_{ii}=0$ and $|H_{ij}|\le2B$. Conditional on its partner $j$, row $i$'s
expected contribution is $\varphi(x_i,v_i)+H_{ij}$. An even matching has
partner probabilities $1/(N-1)$ for $j\ne i$. An odd matching has probability
$1/N$ for every $j$, including its uniformly chosen singleton. Therefore
the difference from $N^{-2}\sum_{ij}H_{ij}$ is zero for odd $N$ and at most
$2B/(N-1)$ for even $N$.

Conditional on the matching, the acceptances are independent. Each row's
conditional variance is at most $(2B)^2/4=B^2$, so the variance of the
empirical average is at most $B^2/N$. To bound the remaining matching
variance, realize the matching by a uniform permutation paired in consecutive
positions, with the last position a singleton when necessary. Transposing
two permutation positions changes at most four rows' partners. Each expected
row value lies in $[-B,B]$, so it changes the conditional empirical mean by
at most $8B/N$. Expose the permutation successively. Two possible choices
of the next entry admit a bijection of their remaining completions by one
transposition. Their conditional expectations therefore differ by at most
$8B/N$. The Doob martingale has at most $N$ increments, each of absolute
value at most $8B/N$. Orthogonality of martingale increments bounds its
variance by $64B^2/N$. Total variance proves the stated bound.

For bounded continuous $\varphi$, the integrand defining $\mathcal C$ is
bounded and continuous on positive fitness marks, including at $f=g$ and
at the saturation threshold. Weak convergence of product measures gives
$\mathcal C(\eta_N)\varphi\to\mathcal C(\eta)\varphi$. The uniform variance
and bias bounds give convergence in probability of the empirical tests.
A countable convergence-determining family on $\mathbb R^{2d}$ yields weak
convergence in probability. This argument conditions on the marks; it never
declares the previously sampled diversity companions independent rows.
$\square$
:::

:::{prf:corollary} Native fitness tails and a reward-moment-free selection modulus
:label: cor-eh-native-fitness-tightness

For any successful all-live native fitness evaluation with $N\ge4$, let
$z_i^R,z_i^D$ be the two actual sample-standardized channels. Regardless of
the magnitudes of the finite raw rewards and distances,

$$
\frac1N\sum_i z_i^R=\frac1N\sum_i z_i^D=0,\qquad
\frac1N\sum_i(z_i^R)^2,\ \frac1N\sum_i(z_i^D)^2\le1.
$$

Set $a_L=4/(1+e^L)^2$ for $L>0$. The actual fitness-marked empirical law
therefore satisfies the deterministic, population-uniform bounds

$$
0<f_i<4,\qquad
\eta_N\{f<a_L\}\le\frac{2}{1+L^2}.
\tag{EH.M0a}
$$

Every weak subsequential limit of these marked laws on
$\mathbb R^{2d}\times[0,4]$ assigns zero mass to $f=0$ and obeys the same
tail inequality. Thus tightness of the state marginals implies tightness of
the native positive-fitness marked laws, without a raw reward moment bound.

For any two such marked probability laws $\eta,\eta'$, let

$$
\Delta=\inf_\pi\int\left[
 \min\{1,|x-x'|+|v-v'|\}+|f-f'|\right],d\pi(z,z'),
$$

where the infimum is over their couplings. For tests with $|\varphi|\le1$
and Lipschitz constant at most one in $|\Delta x|+|\Delta v|$,

$$
|\mathcal C(\eta)\varphi-\mathcal C(\eta')\varphi|
\le\left(4+\frac6{a_L}\right)\Delta+\frac8{1+L^2}
\quad(L>0).
\tag{EH.M0b}
$$

Together with {prf:ref}`thm-eh-matched-cloning-limit`, a deterministic
marked limit $\eta$ satisfying these bounds gives the quantitative estimate

$$
\begin{split}
\mathbb E|\widetilde\mu_N\varphi-\mathcal C(\eta)\varphi|
\le{}&\sqrt{65/N}+\frac{2\mathbf1_{\{N\text{ even}\}}}{N-1}\\
&+\left(4+\frac6{a_L}\right)\mathbb E\Delta(\eta_N,\eta)
 +\frac8{1+L^2}.
\end{split}
\tag{EH.M0c}
$$

All constants are independent of $N$. Convergence of the fitness-marked
input law remains an input condition; (EH.M0a) supplies its fitness-tail
control directly from the actual normalizer. No finite second moment of
the raw EH action is substituted for this control.
:::

:::{prf:proof}
For either channel $y$, sample standardization uses
$s^2=\sum_i(y_i-\bar y)^2/(N-1)$ and $z_i=(y_i-\bar y)/(s+10^{-30})$.
Its empirical mean is zero and its empirical second moment is
$(N-1)s^2/[N(s+10^{-30})^2]\le1$, also when $s=0$.
For a uniformly sampled row $Z$ of either standardized channel and any
$b>0$, Markov's inequality gives
$\mathbb P(Z<-L)\le(1+b^2)/(L+b)^2$. Choosing $b=1/L$ yields
$\mathbb P(Z<-L)\le1/(1+L^2)$.
If both channels are at least $-L$, their logistic product is at least
$a_L$. A union bound proves (EH.M0a). Portmanteau on the open sets
$\{f<a_L\}$ preserves this bound in every weak limit. Sending $L$ to
infinity gives zero mass at zero. The same bound, together with a compact
set controlling the state marginal, proves the positive-mark tightness.

For the modulus replace $A(f,g)$ temporarily by
$A_a(f,g)=[g/\max\{f,a\}-1]_0^1$. This agrees with $A$ when $f\ge a$,
so the corresponding selection operator differs by at most
$2\eta\{f<a\}$ on a test of magnitude one. At $a=a_L$, the two operator
replacement errors total at most $8/(1+L^2)$.
The clipped ratio satisfies
$|A_a(f,g)-A_a(f',g')|\le2|f-f'|/a+|g-g'|/a$; this follows by its
piecewise derivatives, which are bounded throughout the unsaturated strip
$1<g/\max\{f,a\}<2$.
Couple recipient marks by $\pi$, donor marks by an independent copy of
$\pi$, and acceptance decisions by the same uniform variable. The unchanged
and copied-state test differences contribute at most four times the
expected truncated state cost. Different decisions contribute at most
twice the acceptance-probability difference, hence at most $6/a$ times the
expected fitness cost. Taking the infimum over $\pi$ proves (EH.M0b).
Finally condition on $\eta_N$, use Cauchy--Schwarz with the variance and
bias bounds of {prf:ref}`thm-eh-matched-cloning-limit`, then average and
apply (EH.M0b). This proves (EH.M0c).
:::

:::{prf:theorem} Both native matchings and sample standardization
:label: thm-eh-selection-mean-field

Suppose the empirical geometric marked law
$\rho_N=N^{-1}\sum_i\delta_{(x_i,v_i,r_i)}$ converges in probability in
$W_2$ to a deterministic $\rho$ on $\mathbb R^{2d+1}$. This is an input
hypothesis on the actual geometric rewards, not an established property of
the Einstein–Hilbert trajectories. Let $Z=(X,V,R)$ and
$Z'=(Y,W,R')$ be independent with law $\rho$, and set

$$
D=\sqrt{|X-Y|^2+\delta_D^2},\quad
\bar R=\mathbb ER,\quad s_R=\sqrt{\operatorname{Var}R},\quad
\bar D=\mathbb ED,\quad s_D=\sqrt{\operatorname{Var}D},
$$

$$
F(Z,Z')=\frac{2}{1+\exp[-(R-\bar R)/(s_R+\varepsilon_{\mathrm{std}})]}
\frac{2}{1+\exp[-(D-\bar D)/(s_D+\varepsilon_{\mathrm{std}})]},
\qquad \eta=\operatorname{Law}(X,V,F(Z,Z')).
$$

The empirical position-velocity law after the two matchings, native sample
standardization, open-gate cloning, and identity collision converges weakly
in probability to $\mathcal C(\eta)$. The same conclusion with no cloning
holds at a closed gate. Zero-variance channels cause no singularity because
the implemented $\varepsilon_{\mathrm{std}}=10^{-30}$ remains positive and
fixed in this limit.
:::

:::{prf:proof}
Conditional on the marked input rows, a bounded continuous test of the
ordered pair $(z_i,z_{c^D(i)})$ has the product-empirical expectation up to
$O(B/N)$. This follows from the same partner probabilities as in
{prf:ref}`thm-eh-matched-cloning-limit`, with a diagonal contribution of at
most $2B/N$ when $N$ is odd. A transposition changes at most four tested
rows, so the same martingale argument bounds the variance by $64B^2/N$.
Consequently the empirical ordered-pair law converges to $\rho\otimes\rho$.

Both marginals of this pair law are exactly $\rho_N$, since a matching is
a permutation, also with a singleton. The assumed $W_2$ convergence makes
their squared norms uniformly integrable. It also makes the squared norm
on the product uniformly integrable: for nonnegative $u,v$,
$(u+v)\mathbf1_{u+v>L}\le2u\mathbf1_{u>L/2}+2v\mathbf1_{v>L/2}$.
Since $D^2\le2|X|^2+2|Y|^2+\delta_D^2$, truncation now proves convergence
of the empirical first and second moments of $D$. The reward moments
converge directly from $W_2$ convergence. The sample factor $N/(N-1)$ tends
to one, so both implemented standard deviations converge to $s_R,s_D$.

Standardization and the logistic functions are continuous with the positive
fixed regularizer, also when $s_R$ or $s_D$ is zero. On each compact subset
of the marked pair space, the maps with these empirical parameters converge
uniformly to $F$; tightness controls the complement. Thus the empirical
fitness-marked law converges weakly to $\eta$. The latter assigns probability
one to strictly positive finite fitness, without requiring a positive
uniform lower bound. The cloning matching is independent of the distance
matching given its resulting marks. Apply
{prf:ref}`thm-eh-matched-cloning-limit`. $\square$
:::

(sec-eh-centered-operators)=
## 4. Quantitative centered cloning and transition estimates

:::{div} feynman-prose
Measure the cloned cloud first around its old center. Each accepted pair contributes a signed change in squared distance; recentering subtracts the squared center displacement exactly. Retain both kinetic kicks and their noise correlations, then add the stage increments over all twenty phases. This gives the exact observed change as a sum of conditional drifts and centered random residuals, without demanding contraction at every phase. The cloning spread variance is controlled by the fourth moment divided by $N$. With the stated moment bounds, the accumulated residual averaged over cycles concentrates, making the signed cycle drift a quantitative object to compare with native trajectories.
:::

:::{prf:definition} Centered observables and the retained conditioning
:label: def-eh-centered-observables

For $N\ge4$ live rows write
$\langle x,v\rangle_c=N^{-1}\sum_i(x_i-\bar x)\cdot(v_i-\bar v)$,
$W_x=\langle x,x\rangle_c$, $V_v=\langle v,v\rangle_c$, and
$E_v=N^{-1}\sum_i|v_i|^2$. These are the centered observables of
{prf:ref}`def-variance-conversions` and
{prf:ref}`thm-kuk-signed-keystone-variance`, with the native EH preparation
in place of the independent-companion preparation.

Condition first on the positions, velocities and the **realized complete
fitness vector** following the distance matching. Put
$a_{ij}=A(f_i,f_j)$ at an open gate and $a_{ij}=0$ at a closed gate.
The next, independent matching is uniform and mutual. In each unordered
edge $e=\{i,j\}$, $i<j$, at most one direction has positive acceptance.
Define

$$
\begin{aligned}
h_e&=(a_{ij}-a_{ji})(x_j-x_i),&
k_e&=(a_{ij}+a_{ji})|x_j-x_i|^2,\\
s_e&=(a_{ij}-a_{ji})(|x_j-\bar x|^2-|x_i-\bar x|^2),&
H&=\sum_e h_e,\quad H_i=\sum_{e\ni i}h_e.
\end{aligned}
\tag{EH.M1}
$$

Every incident sum uses the same vector $h_e$, without changing its sign
at the second endpoint: $h_e$ is the displacement of the **total position
sum**, not an oriented graph flux. Let

$$
(q_N,r_N)=
\begin{cases}
((N-1)^{-1},[(N-1)(N-3)]^{-1}),&N\text{ even},\\
(N^{-1},[N(N-2)]^{-1}),&N\text{ odd}.
\end{cases}
$$

These are the inclusion probabilities of one specified edge and of two
specified disjoint edges. The odd law includes its uniformly chosen
singleton. All expectations below include the matching and its Bernoulli
gates; an outer expectation over the distance matching gives conditioning
on the entering swarm alone.
:::

:::{prf:theorem} Exact signed cloning drift and population-uniform center fluctuations
:label: thm-eh-centered-cloning

In {prf:ref}`def-eh-centered-observables`, let $X$ be the cloned positions,
$\Delta\bar x=\bar X-\bar x$, and put

$$
B_N=\frac{q_N\sum_e k_e+
 r_N(|H|^2+\sum_e|h_e|^2-\sum_i|H_i|^2)}{N^2}.
\tag{EH.M2}
$$

The exact moment identities are

$$
\mathbb E\Delta\bar x=\frac{q_N H}{N},\qquad
\mathbb E|\Delta\bar x|^2=B_N,\qquad
\mathbb EW_X=W_x+\frac{q_N}{N}\sum_e s_e-B_N.
\tag{EH.M3}
$$

Furthermore, uniformly over every entering cloud and positive fitness vector,

$$
\operatorname{tr}\operatorname{Cov}(\Delta\bar x)
=B_N-\frac{q_N^2|H|^2}{N^2}
\le(q_N+r_N)W_x\le\frac{3W_x}{N}.
\tag{EH.M4}
$$

Let $M_{4,x}=N^{-1}\sum_i|x_i-\bar x|^4$. The spread itself satisfies

$$
\operatorname{Var}(W_X)\le
(q_N+r_N)\bigl(\sqrt{M_{4,x}-W_x^2}+2W_x\bigr)^2
\le\frac{15M_{4,x}}{N}.
\tag{EH.M4a}
$$

At a closed gate this variance is zero. All these estimates condition on
the entering state and the already sampled diversity fitness.

Cloning preserves every frozen velocity exactly. Consequently
$V_v$ and $E_v$ are unchanged, and

$$
\begin{split}
\mathbb E\langle X,v\rangle_c=\langle x,v\rangle_c+
\frac{q_N}{N}\sum_{i<j}(x_j-x_i)\cdot
 [a_{ij}(v_i-\bar v)-a_{ji}(v_j-\bar v)].
\end{split}
\tag{EH.M5}
$$

Pathwise, $W_X\le2W_x$, and for any fixed center $z$ and $p\ge1$,
$N^{-1}\sum_i|X_i-z|^p\le2N^{-1}\sum_i|x_i-z|^p$.
Neither the signed term in (EH.M3) nor its negative is discarded.
:::

:::{prf:proof}
A specified edge is included with probability $q_N$. Conditional on its
inclusion, removing its two vertices leaves the same uniform matching law
on $N-2$ vertices, so two disjoint edges have probability $r_N$. Intersecting
distinct edges cannot both occur. Let $D_e$ denote the actual sum displacement
on an included edge. Its conditional mean is $h_e$ and second moment is
$k_e$. Gates on disjoint edges are independent conditional on the matching.
Thus for $S=N\Delta\bar x$,

$$
\mathbb ES=q_NH,\quad
\mathbb E|S|^2=q_N\sum_e k_e+
 2r_N\sum_{\{e,f\}:e\cap f=\varnothing}h_e\cdot h_f.
$$

Expanding $|H|^2$ and the incident sums gives
$2\sum_{e\cap f=\varnothing}h_e\cdot h_f
=|H|^2+\sum_e|h_e|^2-\sum_i|H_i|^2$, proving (EH.M2).
The expected change of $N^{-1}\sum_i|X_i-\bar x|^2$ is
$q_N\sum_e s_e/N$. Recentering subtracts
$|\bar X-\bar x|^2$ exactly. This is the variance-decomposition strategy of
{prf:ref}`lem-variance-change-decomposition`, with the mutual-matching
covariances retained rather than replaced by independent donor rows.

Since $\sum_iH_i=2H$, Cauchy--Schwarz gives
$\sum_i|H_i|^2\ge4|H|^2/N$. In both parity cases
$r_N-q_N^2-4r_N/N\le0$. Also $|h_e|^2\le k_e\le|x_i-x_j|^2$,
because only one direction accepts and its probability is at most one.
Subtract $|\mathbb ES|^2$ from its second moment and use
$\sum_{i<j}|x_i-x_j|^2=N^2W_x$ to obtain (EH.M4).
For even $N\ge4$,
$N(q_N+r_N)=N(N-2)/[(N-1)(N-3)]\le8/3<3$;
for odd $N\ge5$ it is $(N-1)/(N-2)\le4/3$.

The identity collision preserves $v_i$; summing each recipient's expected
displacement against $v_i-\bar v$ proves (EH.M5). Each disjoint pair retains
its two positions or duplicates one of them. Thus the sum of any nonnegative
one-row observable increases by at most a factor two. Apply this to
$|x_i-z|^p$ and choose $z=\bar x,p=2$; recentering can only decrease the
sum. This proves the pathwise bounds.

For (EH.M4a), set $g_i=|x_i-\bar x|^2$ and
$U=N^{-1}\sum_i|X_i-\bar x|^2$. Apply the preceding scalar matching
calculation to $g_i$ in place of $x_i$. Its mean displacement is $U-W_x$,
so $\operatorname{Var}(U)\le(q_N+r_N)(M_{4,x}-W_x^2)$.
For any realized matching, at most one recipient in each pair moves.
Cauchy--Schwarz and disjointness give

$$
|\Delta\bar x|^2\le\frac{\lfloor N/2\rfloor}{N^2}
 \sum_{\{i,j\}\text{ matched}}|x_j-x_i|^2\le W_x.
$$

The function $z\mapsto|z|^2$ is $2\sqrt{W_x}$-Lipschitz on this ball.
Using an independent copy of $\Delta\bar x$ and (EH.M4),
$\operatorname{Var}(|\Delta\bar x|^2)\le4(q_N+r_N)W_x^2$.
Now $W_X=U-|\Delta\bar x|^2$; the triangle inequality for centered
$L^2$ norms proves the first bound in (EH.M4a). Finally
$(\sqrt{M_{4,x}-W_x^2}+2W_x)^2\le5M_{4,x}$ by Cauchy--Schwarz.
:::

:::{prf:lemma} Concentration and stability of the actual selection operator
:label: lem-eh-selection-quantitative

For the bounded test in {prf:ref}`thm-eh-matched-cloning-limit`, every $u>0$
satisfies the conditional, population-uniform bound

$$
\Pr\{|\widetilde\mu_N\varphi-
 \mathbb E[\widetilde\mu_N\varphi\mid z]|\ge u\mid z\}
 \le2\exp[-Nu^2/(136B^2)].
\tag{EH.M6}
$$

For two input arrays, couple the distance permutations, cloning permutations
and gate uniforms identically. Write $\|a\|_{2,N}^2=N^{-1}\sum_i|a_i|^2$.
If $R,R'$ are their geometric reward arrays, the expected fraction $D_N$ of
different cloning decisions is bounded by

$$
\mathbb ED_N\le\min\{1,4[L_R\|R-R'\|_{2,N}
                         +2L_D\|x-x'\|_{2,N}]\},
\tag{EH.M7}
$$

where always $L_R=L_D=\varepsilon_{\rm std}^{-1}=10^{30}$ is valid.
If both realized channel sample deviations are at least $s_*>0$, the
corresponding constant can be replaced by
$\min\{\varepsilon_{\rm std}^{-1},2/(s_*+\varepsilon_{\rm std})\}$.
The latter replacement is a bound for those realized inputs; when it is
used inside an expectation over distance matchings it must hold for every
matching in its scope, or the complementary event must retain the global
constant. No positive population-uniform minimum of the fitness is required.

For $d_0((x,v),(y,w))=(|x-y|\wedge1)+(|v-w|\wedge1)$ the same coupling gives

$$
\mathbb E\frac1N\sum_i d_0((X_i,v_i),(X_i',v_i'))
\le\frac2N\sum_i(|x_i-x_i'|\wedge1)
 +\frac1N\sum_i(|v_i-v_i'|\wedge1)+\mathbb ED_N.
\tag{EH.M8}
$$

The constants are independent of $N$. Equation (EH.M7) retains the reward
array error: it is not a geometric Lipschitz theorem for $R(x)$.
:::

:::{prf:proof}
Expose the cloning permutation and then the independent acceptance uniforms.
A transposition changes at most four partners, hence changes the conditional
empirical mean by at most $8B/N$. Each exposed gate changes the empirical
test by at most $2B/N$. The resulting Doob martingale has squared increment
bounds summing to at most $68B^2/N$. The conditional exponential-moment
bound for a mean-zero bounded increment, followed by exponential Markov
and optimization, gives (EH.M6).

Put $\ell_i=\log f_i$. The native acceptance is
$G(\ell_j-\ell_i)$, where $G(t)=\min\{1,(e^t-1)_+\}$ is globally
$2$-Lipschitz. A shared uniform produces different decisions with probability
equal to the absolute difference of their acceptance probabilities. Since
a matching is a permutation, summing over recipients gives
$\mathbb ED_N\le4N^{-1}\sum_i|\ell_i-\ell_i'|$.
The log of the amplitude-two logistic has derivative $1/(1+e^z)\le1$.
Therefore the mean log-fitness error is at most the sum of the two standardized
channel errors in $\|\cdot\|_{2,N}$.

On the centered subspace the sample-standardization map is
$u\mapsto u/(\sqrt{N/(N-1)}\|u\|_{2,N}+\varepsilon_{\rm std})$.
Its radial and tangential derivatives have norms at most
$1/\varepsilon_{\rm std}$; centering is an orthogonal projection. This proves
the global constant, including a constant channel. If both deviations exceed
$s_*$, subtract the two quotients and use the reverse triangle inequality
for the deviations to obtain $2/(s_*+\varepsilon_{\rm std})$ instead.
For the common distance permutation $\pi$, the regularized Euclidean norm
satisfies $|D_i-D_i'|\le|x_i-x_i'|+|x_{\pi(i)}-x_{\pi(i)}'|$, so its
$\|\cdot\|_{2,N}$ error is at most $2\|x-x'\|_{2,N}$. This proves (EH.M7).
When decisions agree, each copied source error is charged at most twice;
when they differ, the additional truncated position cost is at most one.
Velocities are unchanged. Summing proves (EH.M8).
:::

:::{prf:theorem} Both Boris kicks and the exact centered BAOAB budget
:label: thm-eh-centered-transition

Condition on the complete post-cloning state $(x,v)$ and the graph consumed
in this step. Let $u=B_1(x,v)$ be the actual first $PQP$ kick, put
$t=h/2$, $b=t(1+c)$, and draw independent standard Gaussian rows $\xi_i$.
The native stages are exactly

$$
w=cu+\sigma\xi,\qquad y=x+bu+t\sigma\xi,\qquad
v^+=B_2(y,w),\qquad x^+=y.
\tag{EH.K1}
$$

The second kick uses the same weights and its newly evaluated curl.
Write $e=B_2(y,w)-w$, $V_u=\langle u,u\rangle_c$,
$C_{xu}=\langle x,u\rangle_c$, and $d_N=d(1-1/N)$. Then

$$
\begin{aligned}
\mathbb EW_{x^+}&=W_x+2bC_{xu}+b^2V_u+d_Nt^2\sigma^2,\\
\mathbb E\langle x^+,v^+\rangle_c
 &=cC_{xu}+bcV_u+d_Nt\sigma^2+\mathbb E\langle y,e\rangle_c,\\
\mathbb EV_{v^+}
 &=c^2V_u+d_N\sigma^2+
        \mathbb E[2\langle w,e\rangle_c+\langle e,e\rangle_c].
\end{aligned}
\tag{EH.K2}
$$

With $\tau^2=t^2\sigma^2$ and $m=x+bu$, the first identity has the exact
conditional fluctuation formula

$$
\operatorname{Var}(W_{x^+})=
 \frac{4\tau^2 W_m}{N}+\frac{2d(N-1)\tau^4}{N^2}.
\tag{EH.K3}
$$

For $Q=\alpha W_x+2\beta\langle x,v\rangle_c+\gamma_P V_v$,
where $\alpha,\gamma_P>0$ and $\alpha\gamma_P>\beta^2$, substitution gives

$$
\begin{split}
\mathbb EQ(x^+,v^+)
={}&\alpha W_x+2(\alpha b+\beta c)C_{xu}
 +(\alpha b^2+2\beta bc+\gamma_Pc^2)V_u\\
 &+d_N\sigma^2(\alpha t^2+2\beta t+\gamma_P)
 +\mathbb E[2\beta\langle y,e\rangle_c
       +\gamma_P(2\langle w,e\rangle_c+\langle e,e\rangle_c)].
\end{split}
\tag{EH.K4}
$$

This is the native EH specialization of the signed quadratic calculation
in {prf:ref}`lem-kuk-signed-increment`; the second curl/noise correlation
is retained in the last line. No Gaussian cancellation is assigned to $e$.
:::

:::{prf:proof}
The first A stage gives $x+tu$; the O stage gives $w$; the second A stage
gives $x+tu+tw=y$. B2 does not change positions, proving (EH.K1).
For centered arrays, bilinearity gives the three expansions (EH.K2).
The mixed terms with a deterministic array and $\xi$ have zero expectation,
and $\mathbb E\langle\xi,\xi\rangle_c=d_N$.
The variable $e$ depends on those same Gaussian rows and stays inside its
expectation. Multiplying the three identities proves (EH.K4).

For (EH.K3), let $H_N=I-\mathbf1\mathbf1^T/N$ be the centering projector.
Then $NW_{x^+}=|H_Nm+\tau H_N\xi|^2$ on $\mathbb R^{Nd}$.
The centered Gaussian has $d(N-1)$ independent unit coordinates in an
orthonormal basis of this subspace. Expanding the square gives variance
$4\tau^2|H_Nm|^2+2d(N-1)\tau^4$; the linear/quadratic covariance vanishes
by Gaussian symmetry. Divide by $N^2$.
:::

:::{prf:corollary} Computed drift coefficients for a centered Lyapunov estimate
:label: cor-eh-centered-drift

Assume $h\nu\le4$. Let $\mathcal J_N=q_N\sum_e s_e/N$ and $B_N$ be (EH.M2), computed at
the entering state after distance matching. For any $\delta>0$, each actual
post-cloning graph with column factor $\kappa$ satisfies

$$
\begin{aligned}
\mathbb E[W_{x^+}\mid S]
&\le(1+\delta)\mathbb E[W_x+\mathcal J_N-B_N\mid S]
 +(1+\delta^{-1})b^2\mathbb E[\kappa^2 E_v\mid S]+dt^2\sigma^2,\\
\mathbb E[E_{v^+}\mid S]
&\le\mathbb E[c^2\kappa^4E_v+\kappa^2d\sigma^2\mid S].
\end{aligned}
\tag{EH.K5}
$$

Here $E_v$ is the entering energy, unchanged by cloning. Suppose, on a
specified family of transitions, one **proves** $\kappa\le K$ and
$\mathbb E[W_x+\mathcal J_N-B_N\mid S]\le\rho W_x+A E_v+C$
with $K,\rho,A,C$ independent of $N$, $0\le\rho<1$, $A,C\ge0$.
Set

$$
a_x=(1+\delta)\rho,\quad
D=(1+\delta)A+(1+\delta^{-1})b^2K^2,\quad
\lambda=c^2K^4.
$$

If $a_x<1$, $\lambda<1$, and
$L>D/(1-\lambda)$, then for $V=W_x+LE_v$,

$$
\mathbb E[V(S^+)\mid S]\le rV(S)+M,\quad
r=\max\{a_x,\lambda+D/L\}<1,\quad
M=(1+\delta)C+dt^2\sigma^2+LK^2d\sigma^2.
\tag{EH.K6}
$$

Consequently $\mathbb EV(S_n)\le r^n\mathbb EV(S_0)+M/(1-r)$, with
all constants independent of $N$. This corollary computes the required
coefficient test; its signed cloning and graph hypotheses are not automatic
consequences of the preset. In particular the closed cloning gates have
$\mathcal J_N=B_N=0$. A contraction proof for the period-$20$ algorithm
must compose all $20$ phase-dependent budgets or establish a stronger
cross-term estimate using (EH.K4); it cannot apply an open-gate estimate
at every step. Moment drift alone is also not an ergodicity theorem.
:::

:::{prf:proof}
In (EH.K2) use
$2bC_{xu}\le\delta W_x+b^2V_u/\delta$ and
$V_u\le E_u\le\kappa^2E_v$ from {prf:ref}`lem-eh-convex-kick`.
Average (EH.M3) over the actual preparation to obtain the first inequality.
B2 gives $E_{v^+}\le\kappa^2 E_w$ pathwise. The graph is already fixed
before O, so $\mathbb E[E_w\mid x,v]=c^2E_u+d\sigma^2$, proving the
second. Multiply by $L$, add, and compare the coefficients of $W_x,E_v$
to obtain (EH.K6). Iteration of the scalar recursion gives its moment bound.
:::

:::{prf:lemma} Quantitative continuity of a frozen graph/Boris stage
:label: lem-eh-curl-stability

Fix the graph, weights and positions at a B query and assume $h\nu\le4$. Set
$s_* =\max_i\sum_j w_{ij}\le1$, and define the actually used least-squares
matrices

$$
M_i=\sum_jw_{ij}r_{ij}r_{ij}^T+\lambda_i I,\quad
\lambda_i=\max\{\epsilon_T^{1/2}\sum_jw_{ij}|r_{ij}|^2/d,
                       \operatorname{MINPOS}_T\},\quad r_{ij}=x_j-x_i.
$$

Let $\ell_i=\sum_jw_{ij}|r_{ij}|$, $\mu_i=\lambda_{\min}(M_i)>0$,
$L_{\rm curl}=4\nu s_*\max_i\ell_i/\mu_i$, and $k=h\beta_{\rm curl}/4$.
For the maximum row norm, on velocities bounded by $R$,

$$
\|B_x(v)-B_x(v')\|_\infty
\le(1+2kRL_{\rm curl})\|v-v'\|_\infty.
\tag{EH.K7}
$$

There is also a coefficient perturbation bound without differentiating a
tessellation. Given two successfully evaluated B inputs, use their actual
$P,P'$, curls $\Omega_i,\Omega_i'$, and put
$\varepsilon_P=\max_i\sum_j|P_{ij}-P'_{ij}|$,
$\varepsilon_\Omega=\max_i\|\Omega_i-\Omega_i'\|_{\rm op}$,
$R=\|v'\|_\infty$. Then

$$
\|P Q Pv-P'Q'P'v'\|_\infty
\le\|v-v'\|_\infty+2R\varepsilon_P+2kR\varepsilon_\Omega.
\tag{EH.K8}
$$

The curls in (EH.K8) are the actual pre-quarter-kick viscous-force fits.
Their errors cannot be set to zero merely because both Cayley maps preserve
norms. The formulas are independent of $N$ when their displayed local
geometry and velocity bounds are independent of $N$.
:::

:::{prf:proof}
The viscous-force difference has maximum row norm at most
$2\nu s_*\|v-v'\|_\infty$. Each neighbor difference of these forces is
at most twice this. The least-squares numerator error is therefore at most
$4\nu s_*\ell_i\|v-v'\|_\infty$ in operator norm. Right multiplication
by $M_i^{-1}$ and taking the skew part proves the curl bound.
For a real skew matrix $A$, $\|(I-A)^{-1}\|_{\rm op}\le1$ and
$Q(A)=2(I-A)^{-1}-I$. The resolvent identity gives
$\|Q(A)-Q(A')\|_{\rm op}\le2\|A-A'\|_{\rm op}$.
Both convex $P$ maps contract maximum row norm. Insert and subtract the
intermediate expressions with $P'$, $Q'$ and $v'$ to obtain (EH.K8).
For identical positions and weights, only the velocity-dependent curl
changes; its preceding bound gives (EH.K7).
:::

:::{prf:lemma} Dissipation identity for the native frozen geometry
:label: lem-eh-exact-dissipation

Retain the reversible $\pi$ and $P$ of
{prf:ref}`lem-eh-frozen-reversible-energy`. Define

$$
\mathcal D_P(v)=\frac12\sum_i\pi_i\sum_{j,k}P_{ij}P_{ik}|v_j-v_k|^2.
$$

For a B stage with its realized rotation $Q$, set
$\mathcal D_B(v)=\mathcal D_P(v)+\mathcal D_P(QPv)\ge0$.
Then, exactly,

$$
E_\pi(Bv)=E_\pi(v)-\mathcal D_B(v),\qquad
E_\pi(v)=\sum_i\pi_i|v_i|^2.
\tag{EH.D1}
$$

Conditional on the actual post-cloning graph and state, both kicks give

$$
\mathbb E E_\pi(v^+)-E_\pi(v)
=-(1-c^2)E_\pi(v)-c^2\mathcal D_{B_1}(v)
 -\mathbb E\mathcal D_{B_2}(w)+d\sigma^2.
\tag{EH.D2}
$$

For the fourth weighted moment $M_{4,\pi}(v)=\sum_i\pi_i|v_i|^4$,

$$
\mathbb EM_{4,\pi}(v^+)
\le c^4M_{4,\pi}(v)+2(d+2)c^2\sigma^2E_\pi(v)
                         +d(d+2)\sigma^4.
\tag{EH.D3}
$$

Every coefficient is independent of $N$; these are statements about the
same graph weights on both sides. If $\pi^+$ denotes the next refreshed
weights, the additional energy change is exactly
$\sum_i(\pi_i^+-\pi_i)|v_i^+|^2$. If, in a stated regime,
$\pi_i^+\le R\pi_i$ pathwise, (EH.D2) implies the iterated energy budget
$\mathbb E E_{\pi^+}(v^+)\le Rc^2E_\pi(v)+Rd\sigma^2$.
Thus $Rc^2<1$ is an explicit sufficient transfer margin; the definition of
$\pi$ alone does not prove that margin.
:::

:::{prf:proof}
The variance identity for a probability row is
$\sum_jP_{ij}|v_j|^2-|\sum_jP_{ij}v_j|^2
=\frac12\sum_{j,k}P_{ij}P_{ik}|v_j-v_k|^2$.
Multiply by $\pi_i$ and use $\pi P=\pi$. The intervening rowwise rotation
preserves $E_\pi$ exactly, so the two quarter-kick losses add. This proves
(EH.D1). Apply it to B2, integrate O using
$\mathbb E|cu_i+\sigma\xi_i|^2=c^2|u_i|^2+d\sigma^2$, and apply it to
B1. The result is (EH.D2), including the nonnegative B2 loss which depends
on the noise. The Gaussian fourth-moment identity is
$\mathbb E|cz+\sigma\xi|^4=c^4|z|^4+2(d+2)c^2\sigma^2|z|^2
+d(d+2)\sigma^4$. Jensen and rowwise norm preservation contract both
weighted moments through each B stage. This proves (EH.D3). The refresh
statements follow by subtraction and termwise comparison of positive weights.
:::

:::{prf:lemma} Native position smoothing and quantitative concentration
:label: lem-eh-position-smoothing

Condition as in {prf:ref}`thm-eh-centered-transition`, and let
$\tau=t\sigma>0$, $m=x+bu$, $k_N=d(N-1)$. The final position rows are
independent Gaussians $\mathcal N(m_i,\tau^2I_d)$ under this conditioning,
although their completed position-velocity pairs need not be independent.
Each positional density is bounded by $(2\pi\tau^2)^{-d/2}$ and its gradient
by $(2\pi\tau^2)^{-d/2}/(\tau\sqrt e)$.
The conditional center has covariance $\tau^2 I_d/N$. Moreover,

$$
\mathbb E e^{\theta NW_{x^+}}
=(1-2\theta\tau^2)^{-k_N/2}
 \exp\left(\frac{\theta NW_m}{1-2\theta\tau^2}\right),
\quad \theta<(2\tau^2)^{-1}.
\tag{EH.D4}
$$

Put $v_N=2\tau^2W_m/N+k_N\tau^4/N^2$. For $z>0$,

$$
\Pr\left\{|W_{x^+}-\mathbb EW_{x^+}|>
       2\sqrt{v_Nz}+2\tau^2z/N\right\}\le2e^{-z}.
\tag{EH.D5}
$$

This is a uniform-in-$N$ fluctuation estimate whenever $W_m$ has the stated
uniform bound. It imposes no compact support on the Gaussian innovations.
:::

:::{prf:proof}
Equation (EH.K1) proves the position laws; B2 changes no position.
Differentiating their densities gives the displayed bounds. In an orthonormal
basis of the centered subspace, complete the square in each one-dimensional
Gaussian integral to obtain (EH.D4). For the centered random variable
$Z=W_{x^+}-\mathbb EW_{x^+}$, its logarithmic moment generating function
satisfies
$\log\mathbb E e^{uZ}\le v_Nu^2/(1-2\tau^2u/N)$ for
$0<u<N/(2\tau^2)$, by
$-\log(1-a)-a\le a^2/[2(1-a)]$ and the noncentral term in (EH.D4).
For $-Z$ it is at most $v_Nu^2$ for $u>0$, using
$a-\log(1+a)\le a^2/2$ and its noncentral term.
Exponential Markov with these bounds gives upper deviation
$2\sqrt{v_Nz}+2\tau^2z/N$ and lower deviation $2\sqrt{v_Nz}$,
each with probability at most $e^{-z}$. Their union proves (EH.D5).
:::

:::{prf:corollary} Exact composition of the twenty phase budgets
:label: cor-eh-period-budget

For phase $j=0,\ldots,q-1$, assume $h\nu\le4$, that every actual
post-cloning graph has $\kappa\le K_j$, and that its conditional cloning
budget is $\mathbb E[W_X\mid S]\le\rho_jW_x+A_jE_v+C_j$.
Here $K_j\ge1$ and $\rho_j,A_j,C_j\ge0$ are independent of $N$.
No contraction is demanded at an individual phase. Choose $\delta_j>0$ and write

$$
T_j=\begin{pmatrix}a_j&D_j\\0&\lambda_j\end{pmatrix},\quad
c_j=\binom{(1+\delta_j)C_j+dt^2\sigma^2}{K_j^2d\sigma^2},
$$

where $a_j=(1+\delta_j)\rho_j$,
$D_j=(1+\delta_j)A_j+(1+\delta_j^{-1})b^2K_j^2$, and
$\lambda_j=c^2K_j^4$. Then the $q=20$ skeleton has the componentwise moment
budget

$$
\mathbb E\binom{W_{n+q}}{E_{n+q}}\le
T\binom{W_n}{E_n}+C_*,\qquad
T=T_{q-1}\cdots T_0,\quad
C_*=\sum_{j=0}^{q-1}T_{q-1}\cdots T_{j+1}c_j.
\tag{EH.D6}
$$

An empty product is the identity. The entries are explicitly

$$
T_{11}=\prod_j a_j,\quad T_{22}=\prod_j\lambda_j,\quad
T_{12}=\sum_{j=0}^{q-1}\left(\prod_{k>j}a_k\right)D_j
                                  \left(\prod_{k<j}\lambda_k\right).
$$

If $T_{11},T_{22}<1$, choose $L>T_{12}/(1-T_{22})$ and obtain
$r=\max\{T_{11},T_{22}+T_{12}/L\}<1$ and
$M=(C_*)_1+L(C_*)_2$ for $W+LE$ on the skeleton. The intermediate phase
bounds follow from the corresponding partial products, all uniformly in $N$.
At every closed gate one must use $\rho_j=1,A_j=C_j=0$.
With constant $K,\delta$ and one open gate having coefficient $\rho$,
these sufficient diagonal tests are exactly
$\rho(1+\delta)^{20}<1$ and $(c^2K^4)^{20}<1$.
:::

:::{prf:proof}
(EH.K5) gives the one-phase componentwise inequality. Its coefficients are
nonnegative, so conditional expectation and substitution preserve the
inequality at the next phase. Induction gives (EH.D6). Multiplication of
upper triangular matrices gives the displayed entries. Apply the coefficient
comparison of (EH.K6) to $T,C_*$ and iterate. Closed gates are the identity
cloning operator, which fixes their stated coefficients.
:::

:::{prf:corollary} Signed cycle drift and its sampling error
:label: cor-eh-signed-cycle

At step $k$, expose the entering cloud and diversity matching first, then
the cloning matching and decisions, and finally the OU innovation. Let
$X_k$ be the post-cloning positions, $u_k=B_1(X_k,v_k)$, and define

$$
\begin{aligned}
d_k^{\mathrm C}&=\mathcal J_{N,k}-B_{N,k},\\
d_k^{\mathrm K}&=2b\langle X_k,u_k\rangle_c+b^2V_{u_k}
                          +d_Nt^2\sigma^2,\\
\epsilon_k^{\mathrm C}&=W_{X_k}-W_{x_k}-d_k^{\mathrm C},\\
\epsilon_k^{\mathrm K}&=W_{x_{k+1}}-W_{X_k}-d_k^{\mathrm K}.
\end{aligned}
$$

For any $q=20$ consecutive native steps, starting at any phase,

$$
W_{x_{k+q}}-W_{x_k}
=\sum_{j=k}^{k+q-1}(d_j^{\mathrm C}+d_j^{\mathrm K})
 +\sum_{j=k}^{k+q-1}(\epsilon_j^{\mathrm C}+\epsilon_j^{\mathrm K}).
\tag{EH.D7}
$$

Each residual has zero conditional mean at its own exposure boundary.
The kinetic drift is measured after cloning; it is not claimed to be
measurable before cloning. Put $m_k=X_k+bu_k$ and $\tau=t\sigma$. For
any deterministic horizon $T$ at which the displayed moments are finite,
the accumulated residual $M_T=\sum_{k<T}(\epsilon_k^{\mathrm C}
+\epsilon_k^{\mathrm K})$ obeys

$$
\mathbb E M_T^2\le\frac1N\sum_{k<T}
\left[15\mathbf1_{\{\text{gate }k\text{ open}\}}\mathbb E M_{4,x_k}
+4\tau^2\mathbb EW_{m_k}+2d\tau^4\right].
\tag{EH.D8}
$$

In particular, if $T=20L$ and these moments are bounded respectively by
$A_4,A_2$ throughout the specified trajectory laws, then

$$
\mathbb P\left(\left|\frac{M_{20L}}L\right|>u\right)
\le\frac{15A_4+20(4\tau^2A_2+2d\tau^4)}{NLu^2}.
\tag{EH.D9}
$$

The constants are independent of $N$; the moment bounds are hypotheses of
(EH.D9), not estimates inferred from the largest observation in a run.
Equation (EH.D7) retains the signed covariance at all nineteen closed
gates and at the open gate. It therefore evaluates the regional centered
drift used in {prf:ref}`thm-kul-centered-regional-attraction` without
requiring every phase to decrease spread.

A finite-horizon comparison also has a bound using the realized predictable
variance budget. Let $v_k^{\mathrm C}$ be the first bound in (EH.M4a), zero
at closed gates, and $v_k^{\mathrm K}$ the exact expression in (EH.K3).
Set $\mathcal V_T=\sum_{k<T}(v_k^{\mathrm C}+v_k^{\mathrm K})$.
For any fixed $R>0$, $c_R=\max\{2R/3,2\tau^2/N\}$,
$0<\lambda<c_R^{-1}$, and $0<\delta<1$,

$$
\mathbb P\left\{
|M_T|\ge\frac{\log(2/\delta)}{\lambda}
 +\frac{\lambda\mathcal V_T}{2(1-c_R\lambda)},\quad
\max_{\substack{k<T\\\text{gate open}}}W_{x_k}\le R
\right\}\le\delta.
\tag{EH.D10}
$$

The constants $R,\lambda$ are fixed before observing the run. Selection
from any predeclared finite collection of $J$ such pairs is valid after
replacing $\log(2/\delta)$ by $\log(2J/\delta)$. This comparison needs no
unproved uniform fourth-moment expectation: its budget contains the actual
conditional fourth moments of the entering finite clouds.
:::

:::{prf:proof}
Subtract the conditional means (EH.M3) and (EH.K2) at their respective
exposure boundaries. Adding the resulting two equalities and telescoping
proves (EH.D7). The chronological list of the two residuals at each step
is a martingale-difference sequence. Distinct residuals are orthogonal in
$L^2$, including the cloning and kinetic residuals of the same step, since
the latter is centered conditional on the entire realized cloning outcome.
Use (EH.M4a) for the cloning variance, zero at closed gates, and (EH.K3)
for the kinetic variance, bounding $(N-1)/N\le1$. This proves (EH.D8).
Exactly $L$ gates open in $20L$ steps. Insert $A_4,A_2$, divide by $L^2$,
and apply Chebyshev's inequality to prove (EH.D9).

For (EH.D10), the cloning spread belongs to $[0,2W_{x_k}]$, so its centered
residual has magnitude at most $2W_{x_k}$. For $m\ge2$,
$\mathbb E|\epsilon_k^{\mathrm C}|^m\le
(2W_{x_k})^{m-2}v_k^{\mathrm C}$.
Expand its exponential, use $m!\ge2\,3^{m-2}$, and then
$1+z\le e^z$ to obtain the conditional logarithmic moment bound

$$
\log\mathbb E[e^{\lambda\epsilon_k^{\mathrm C}}\mid\text{before cloning}]
\le\frac{\lambda^2v_k^{\mathrm C}}
          {2(1-2W_{x_k}|\lambda|/3)}.
$$

Expanding the noncentral Gaussian quadratic moment-generating function in
{prf:ref}`lem-eh-position-smoothing` gives the same inequality for
$\epsilon_k^{\mathrm K}$ with $v_k^{\mathrm K}$ and denominator
$2(1-2\tau^2|\lambda|/N)$. Stop the residual sequence before the first
open gate with $W_{x_k}>R$. At each successive exposure, conditional
expectation of its exponential factor after subtracting
$\lambda^2v/[2(1-c_R\lambda)]$ is at most one. Iteration gives expectation
at most one for the product, for either sign of $\lambda$. Markov's
inequality and a union bound over the two signs prove (EH.D10) on the
event where stopping has not occurred. A further union bound proves the
finite-collection assertion. No independence between steps is used.
:::

(sec-eh-marked-kinetic-limit)=
### 4.1. Population limit of the transition with native neighborhood marks

:::{prf:theorem} Quantitative native graph transition law
:label: thm-eh-marked-kinetic-limit

Condition on a successfully defined post-cloning array and its actual
undirected graph $G_N$, weights, and component parameters. For a vertex $i$
let $B_r(i)$ be its graph ball of radius $r$. Let $\Gamma_N$ be the empirical
law of rooted radius-six neighborhoods, marked with the post-cloning
positions and velocities and every directed weight consumed in the two B
stages. Boundary marks include the incidences and weights required to
perform the root calculation; the root lies six edges from the cut boundary.
Define $F_\varphi(\mathcal G)$ by executing the root's native two-kick
transition on this marked neighborhood, with independent standard Gaussian
marks, and averaging the bounded continuous output test $\varphi$ over
these marks. Let $|\varphi|\le B$, and put

$$
L_N=\frac1N\sum_i |B_6(i)|.
$$

For the full empirical output $\mu_N^+$,

$$
\mathbb E[\mu_N^+\varphi\mid G_N,x,v]=\Gamma_NF_\varphi,\qquad
\operatorname{Var}(\mu_N^+\varphi\mid G_N,x,v)\le\frac{B^2L_N}{N}.
\tag{EH.G1}
$$

Suppose $\Gamma_N$ converges weakly in probability to a deterministic
probability $\Gamma$ on finite rooted marked neighborhoods, with discrete
incidence topology and convergent Euclidean marks, and
$\mathbb EL_N/N\to0$. The limiting neighborhoods must have the positive
least-squares ridge and the convex, finite weights specified above. Then
$\mu_N^+$ converges weakly in probability to the probability law
$\mathcal K\Gamma$ characterized by
$(\mathcal K\Gamma)\varphi=\Gamma F_\varphi$. Quantitatively,

$$
\mathbb E|\mu_N^+\varphi-(\mathcal K\Gamma)\varphi|
\le\mathbb E|\Gamma_NF_\varphi-\Gamma F_\varphi|
                          +B\sqrt{\mathbb EL_N/N}.
\tag{EH.G2}
$$

Thus a proved $\mathbb EL_N\le L$ gives an explicit $B\sqrt{L/N}$
transition fluctuation term. Neither a dense replacement of the native graph
nor independent output walkers is assumed. This theorem proves the
transition step of a neighborhood-marked population argument. To iterate it,
one must prove the next refreshed marked-law convergence and neighborhood
bound from the preceding output; that geometry assertion is not assumed to
follow from one-particle weak convergence alone.
:::

:::{prf:proof}
A viscous force uses radius one; its least-squares curl uses forces at the
root and its neighbors, hence radius two. The incoming quarter-kick uses
radius one. The outgoing quarter-kick averages its neighbors' rotated values,
so one B stage uses radius three. Both B stages use the same graph.
Composing them, the intervening rowwise OU stage and the two A stages
therefore needs radius six of the initial deterministic marks. This proves
the conditional mean formula.

After conditioning on the complete post-cloning array, B1 is deterministic.
The only innovations in the kinetic step are the independent OU rows.
The root output uses these innovations only within radius three, because
only B2 follows O. Two output tests are consequently independent under this
conditioning if their radius-three balls are disjoint. If they intersect,
their root distance is at most six. Each covariance has magnitude at most
$B^2$ by Cauchy--Schwarz. Summing the covariances and dividing by $N^2$
gives (EH.G1); this argument retains all dependence among nearby outputs.

On any fixed finite incidence pattern, the additions, force fits with
positive ridge, and skew Cayley solves are continuous in the supplied marks.
The latter inverses exist for every real skew matrix. With Gaussian marks
coupled identically, dominated convergence makes $F_\varphi$ bounded and
continuous on this disjoint union of finite marked neighborhoods. Thus
$\Gamma_NF_\varphi\to\Gamma F_\varphi$ in probability and, by boundedness,
in $L^1$. Conditional Cauchy--Schwarz and Jensen give (EH.G2). A countable
convergence-determining set of tests gives the asserted empirical-law limit.
:::

:::{prf:corollary} An explicit Wasserstein error with unbounded positional tails
:label: cor-eh-quantitative-wasserstein

Let $D=2d$ and $a_D=1/(4D+4)$. Conditional on the frozen pre-cloning
arrays of {prf:ref}`thm-eh-centered-cloning`, put
$M_2=N^{-1}\sum_i(|x_i|^2+|v_i|^2)$. For the actual cloned empirical law,

$$
\mathbb EW_1(\widetilde\mu_N,\mathcal C(\eta_N))
\le\left[4M_2+2+\sqrt{65}(1+2\sqrt D)^D\right]N^{-a_D}
 +\mathbf1_{\{N\text{ even}\}}\frac{2\sqrt{M_2}}{N-1}.
\tag{EH.W1}
$$

The norm in $W_1$ is the Euclidean norm on $\mathbb R^{2d}$ in the
fixed units of the component tuple. This explicit rate is conservative;
its role is to control an empirical probability metric, rather than a
finite list of tests, uniformly in $N$ under an input second-moment bound.
It requires no positive lower bound on the fitness.

For the conditional native kinetic law of
{prf:ref}`thm-eh-marked-kinetic-limit`, let
$M_+=\mathbb E[N^{-1}\sum_i(|x_i^+|^2+|v_i^+|^2)\mid G_N,x,v]$.
With $\nu_N=\mathcal K\Gamma_N$, one likewise has

$$
\mathbb EW_1(\mu_N^+,\nu_N)
\le[2M_++2+(1+2\sqrt D)^D](L_N/N)^{a_D}.
\tag{EH.W2}
$$

For a supplied graph with $\kappa\le K$, an explicit bound is

$$
M_+\le2S_x+(2b^2K^2+c^2K^4)E_v
                         +d\sigma^2(t^2+K^2),\quad
S_x=N^{-1}\sum_i|x_i|^2.
\tag{EH.W3}
$$

Thus geometric input-law errors can be combined by the triangle inequality
with (EH.W1)--(EH.W2). Convergence of the deterministic target law in $W_1$
still requires its own marked-law convergence and moment control.
:::

:::{prf:proof}
We use a finite partition only inside the estimate. No dynamics is
truncated. For a random probability $\mu$ on $\mathbb R^D$ with
$\nu=\mathbb E\mu$, assume every indicator test has variance at most $v$
and $\mathbb E\mu|z|^2\le M$. Project radially to the closed ball of radius
$R$. The sum of the two expected $W_1$ projection errors is at most $2M/R$.
Partition this ball into at most $(1+2R\sqrt D/\varepsilon)^D$ cubes of
side $\varepsilon/\sqrt D$, choosing one representative from every
nonempty intersection. Each point moves by at most $\varepsilon$.
For these finitely supported measures a coupling that retains common mass
and transports the remaining mass a distance at most $2R$ has cost at most
$R\sum_C|\mu(C)-\nu(C)|$. The projection preimages of the cells are
measurable, so conditional Cauchy--Schwarz bounds each expected absolute
mass error by $\sqrt v$. Consequently

$$
\mathbb EW_1(\mu,\nu)\le
2M/R+2\varepsilon+
R(1+2R\sqrt D/\varepsilon)^D\sqrt v.
\tag{EH.W4}
$$

For cloning use $M=2M_2$ by the pathwise bound, $v=65/N$ by
{prf:ref}`thm-eh-matched-cloning-limit`, $R=N^{a_D}$ and
$\varepsilon=N^{-a_D}$. Since
$(2D+1)a_D-1/2=-a_D$, these terms give the first term of (EH.W1).
For odd $N$ the mean law equals $\mathcal C(\eta_N)$. For even $N$,
$\mathbb E\widetilde\mu_N-\mathcal C(\eta_N)
=(\mathcal C(\eta_N)-\mu_N)/(N-1)$ as signed measures, where $\mu_N$
is the entering position-velocity law. A cloning transport coupling gives
$W_1(\mathcal C(\eta_N),\mu_N)\le N^{-2}\sum_{ij}|x_j-x_i|
\le2\sqrt{M_2}$; the dual characterization of $W_1$ proves the correction.

For kinetics apply (EH.G1) to indicator tests, whose proof does not require
continuity, so $v=L_N/N\le1$. In (EH.W4) take
$R=v^{-a_D}$, $\varepsilon=v^{a_D}$, $M=M_+$ to prove (EH.W2).
Finally $\mathbb E|x_i+bu_i+t\sigma\xi_i|^2
=|x_i+bu_i|^2+dt^2\sigma^2$, and the bound
$|x_i+bu_i|^2\le2|x_i|^2+2b^2|u_i|^2$, followed by the two graph moment
bounds, proves (EH.W3).
:::

:::{prf:proposition} Native balanced two-cluster pressure
:label: prop-eh-two-cluster-pressure

Take $N=2m\ge4$ walkers at $x_i=-Re_1$ for $i\le m$ and $x_i=Re_1$
for $i>m$, with zero velocities, in the supported three-dimensional
preset. Use the default clique lifting of duplicate sites and its rank-one
site edge, on the successfully evaluated branch. Every neighbor covariance
is the same, so every determinant mark is the same and the native conformal
Laplacian reward is zero. A distance matching with $k$ cross-cluster pairs
has $k$ high-diversity rows and $m-k$ low-diversity rows in each cluster.
Here $0\le k\le m$ and $m-k$ is even. Let $p$ be the native acceptance
from the low to the high fitness; when there is only one class put $p=0$.
Then the exact open-gate drift is

$$
\frac{\mathbb EW_X}{R^2}
=1-\frac{8k(m-k)}{N^2}
 \left[\frac{p}{N-1}-\frac{(m-1)p^2}{(N-1)(N-3)}\right].
\tag{EH.C1}
$$

In particular $1-[2(N-1)]^{-1}\le\mathbb EW_X/R^2\le1$.
For $N$ divisible by eight, $k=m/2$ and $R\ge1$, the reference
standardization/logistic gate has $p=1$, and

$$
\mathbb EW_X/R^2=1-\frac{N-4}{4(N-1)(N-3)}.
\tag{EH.C2}
$$

This computes the cloning pressure on a realizable native geometric family:
it is of order $1/N$ in (EH.C2). A population-independent positive cloning
margin cannot be inferred from the pairing cancellation or from zero
geometric reward. The full kinetic contribution remains (EH.K2).
:::

:::{prf:proof}
Clique lifting gives the complete graph: the two site groups contribute
within-group cliques, and the site edge lifts to all cross-group edges.
The neighbor covariance at each row is
$4mR^2e_1e_1^T/(N-1)$ in projected coordinates. Its identical regularized
metric gives a constant log determinant, whose graph Laplacian is zero.
Distances are $\delta_D$ within a group and
$\sqrt{4R^2+\delta_D^2}$ between groups. Hence the two groups have identical
fitness multisets even after sample standardization.

All radii about the center are $R$, so $\sum_es_e=0$. The two cross-cluster
low-to-high directions contribute equal opposite total displacements, so
$H=0$. Directly from (EH.M1),

$$
\sum_e k_e=8pk(m-k)R^2,\quad
\sum_e|h_e|^2=8p^2k(m-k)R^2,\quad
\sum_i|H_i|^2=8mp^2k(m-k)R^2.
$$

Substitution into (EH.M2)--(EH.M3) proves (EH.C1).
Nonnegativity of $B_N$ gives the upper bound. Its negative quadratic term
and $k(m-k)\le m^2/4$ give $B_N/R^2\le1/[2(N-1)]$.
When $k=m/2$, the standardized distances are $\pm z$, where
$z=\sqrt{(N-1)/N}/[1+\varepsilon_{\rm std}/s_D]$.
For $N\ge8,R\ge1$, this exceeds $\log2$, so the ratio of the two logistic
fitnesses is $e^z>2$ and the acceptance saturates at one. Simplifying
(EH.C1) proves (EH.C2). The native geometry, standardizer, logistic and
acceptance kernels are checked at $N=8,16,32,64$ and $R=1,16$ in the
Rust regression suite.
:::

:::{prf:theorem} Exact first transition and population limit from the reference start
:label: thm-eh-reference-first-step

At the reference coincident, resting start $x_i=v_i=0$, $N\ge4$, the first
cloning gate is closed. The default duplicate geometry is a complete graph
with $w_{ij}=1/(N-1)$. Put
$\Pi_N=\mathbf1\mathbf1^T/N$, $H_N=I-\Pi_N$ and
$r_N=1-aN/(N-1)$, with $0\le a=h\nu/4\le1$. The actual first transition is

$$
x_1=t\sigma\xi,\qquad
v_1=\sigma(\Pi_N+r_N^2H_N)\xi.
\tag{EH.C3}
$$

In particular,

$$
\mathbb EW_{x_1}=d(1-1/N)t^2\sigma^2,\quad
\mathbb EV_{v_1}=d(1-1/N)r_N^4\sigma^2,\quad
\mathbb EE_{v_1}=d\sigma^2[N^{-1}+(1-N^{-1})r_N^4]\le d\sigma^2.
\tag{EH.C4}
$$

Let $r=1-a$ and couple the independent limiting rows
$(\widehat x_i,\widehat v_i)=(t\sigma\xi_i,r^2\sigma\xi_i)$ with these same
Gaussians. Each row satisfies the exact coupling error

$$
\mathbb E(|x_{1,i}-\widehat x_i|^2+|v_{1,i}-\widehat v_i|^2)
=d\sigma^2[(1-N^{-1})(r_N^2-r^2)^2+N^{-1}(1-r^2)^2]
\le d\sigma^2\left[\frac{4a^2}{(N-1)^2}
                       +\frac{(1-r^2)^2}{N}\right].
\tag{EH.C5}
$$

Thus every fixed number of first-step rows converges to independent rows
of the stated Gaussian position-velocity law, with squared transport error
at most that number times (EH.C5). This explicit calculation covers the
complete graph at the reference start, where the sufficient sparse-overlap
bound (EH.G1) would only give $L_N=N$.
:::

:::{prf:proof}
The first B and A stages are identities at zero velocity; O gives
$w=\sigma\xi$, and the second A gives $y=tw$.
On the complete graph, the force is
$F_i=-\nu N(w_i-\bar w)/(N-1)$, hence every force difference is a scalar
multiple of its position difference. The fitted Jacobian at each row has
the form $-\nu N C_i(C_i+\lambda_iI)^{-1}/[t(N-1)]$, with $C_i$
symmetric nonnegative. It is symmetric, so the actually recomputed B2 curl
vanishes in real arithmetic, including rank-deficient $C_i$ because of the
positive ridge. Therefore B2 is precisely $P^2$ and
$P=\Pi_N+r_NH_N$, proving (EH.C3). The Gaussian center and centered array
are orthogonal and independent, giving (EH.C4). Decompose the coupling
error into its center and centered parts to obtain (EH.C5).
Since $|r_N|,|r|\le1$ for $N\ge4,a\le1$,
$|r_N^2-r^2|\le2a/(N-1)$. Sum row errors for a fixed marginal to conclude.
These are first-step laws; they do not assert that the later refreshed
geometry remains complete or that later curl fields vanish.
:::

:::{prf:corollary} Quantitative limit of a global quadratic drift argument
:label: cor-eh-global-quadratic-margin

Consider the fixed-phase $q=20$ kernel whose first step has an open gate
and whose other $q-1$ steps have closed gates. Retain the balanced inputs
of {prf:ref}`prop-eh-two-cluster-pressure`. Suppose these steps are defined
almost surely as required in {prf:ref}`def-eh-native-interpretation`.
Set

$$
\begin{aligned}
\Lambda_{N,d}&=2d\log(2dN)+2d,\\
K_N&=\sigma\sqrt{\Lambda_{N,d}}\frac{1-c^q}{1-c},\\
J_N&=t\sigma\sqrt{\Lambda_{N,d}}
\left[q+\frac{1+c}{1-c}
\left(q-\frac{1-c^q}{1-c}\right)\right].
\end{aligned}
$$

Then the actual skeleton satisfies

$$
\mathbb EW_{x_q}\ge
\left(1-\frac1{2(N-1)}\right)R^2-2RJ_N,
\qquad
\mathbb EE_{v_q}\le K_N^2.
\tag{EH.C6}
$$

For the centered positive quadratic $Q$ of (EH.K4),

$$
\mathbb EQ(x_q,v_q)\ge
\alpha\left(1-\frac1{2(N-1)}\right)R^2
 -2\alpha RJ_N-2|\beta|(R+J_N)K_N.
\tag{EH.C7}
$$

Consequently a drift $P_N^qQ\le\rho Q+C$ with $\rho<1$ and $C$ independent
of $N$ cannot hold on **all** such states for an unbounded increasing-population
family with these native formulas and fixed quadratic coefficients.
The same conclusion holds for $W_x+LE_v$ with fixed $L>0$.
This identifies a limitation of a global quadratic Foster argument.
It does not exclude a different functional, a distributional drift argument,
or convergence from the reference start, and it is not an inference from a
simulation or from a singleton.
:::

:::{prf:proof}
Only the first step clones, leaving positions $X_i\in\{-Re_1,Re_1\}$ and
velocities zero. By (EH.C1), after averaging also over the distance matching,
$\mathbb EW_X\ge[1-1/(2(N-1))]R^2$; pathwise $W_X\le R^2$.
Let $D_i=x_{q,i}-X_i$ be the displacement accumulated by the subsequent
kinetic stages, including the kinetic part of the first step. There are
no additional position copies on these $q$ steps. The maximum-speed
recursion in {prf:ref}`thm-eh-native-moments`, started at zero velocity,
gives $\|\max_i|v_{q,i}|\|_{L^2}\le K_N$ and

$$
\|\max_i|D_i|\|_{L^2}
\le t\sum_{l=0}^{q-1}
 \big[(1+c)\|\max_i|v_{l,i}|\|_{L^2}
                      +\sigma\|\max_i|\xi_{l,i}|\|_{L^2}\big]
\le J_N.
$$

These bounds use only convex quarter-kicks and orthogonal rotations, so they
are independent of the entering spatial radius $R$ and retain changing
geometries. Expanding $W_{X+D}$ and applying Cauchy--Schwarz gives
$\mathbb EW_{x_q}\ge\mathbb EW_X-2RJ_N$ and
$\sqrt{\mathbb EW_{x_q}}\le R+J_N$. This proves (EH.C6).
The cross covariance has expected absolute value at most
$(R+J_N)K_N$; its coefficient in $Q$ is $2\beta$, and its velocity-square
term is nonnegative. This proves (EH.C7).
Initially $Q=\alpha R^2$. For any proposed $\rho<1$, choose a fixed even
$N$ with $1-1/[2(N-1)]>\rho$. Divide the proposed drift by $R^2$ and let
$R\to\infty$. The explicitly bounded linear terms vanish and yield a
contradiction. For $W+LE$ the positive velocity term can simply be dropped
in the lower bound. The assertion concerns a mathematical population family;
fixed machine index and allocation limits do not define an $N\to\infty$
stationary theorem.
:::

(sec-eh-geometry-limit)=
## 5. What remains for a full population limit

:::{div} feynman-prose
The exact cycle formula tracks centered spread while the neighborhood-marked transition theorem advances the local kinetic law under its stated input assumptions. The Wasserstein estimates control sampling error with unbounded tails retained. To connect successive transitions, we must also control the refreshed geometry and reward marks along the laws being studied. This is where estimates for the centered shape and the local geometry meet. A finite-$N$ convergence proof may have population-dependent constants; uniform moment control requires bounds valid across populations, and an $N$-independent mixing rate requires a separate quantitative argument.
:::

:::{prf:proposition} Native path force has a population-independent sampling variance
:label: prop-eh-sparse-force-obstruction

In the real-coordinate interpretation, take the supported $d=1$
Einstein–Hilbert preset, $4\le N<1/\epsilon_{64}$,
where $\epsilon_{64}$ is the f64 epsilon used in the geometry rank test, and positions
$x_i=i/(N-1)$, $i=0,\ldots,N-1$, and independent initial velocities
$v_i\sim\mathcal N(0,T)$. At the first (closed-gate) B stage, every interior
row has weights $1/2$ on its two neighbors. Conditional on its own velocity,

$$
\mathbb E[F_i\mid v_i]=-\nu v_i,\qquad
\mathbb E\left[|F_i+\nu v_i|^2\mid v_i\right]=\frac{\nu^2T}{2}.
$$

Thus the native neighbor fluctuation is not the $\nu^2T/(N-1)$ variance of
an average of all other walkers. This finite-$N$ identity holds throughout
the executable graph index range. In a separate real-geometry extension
whose nonconstant one-dimensional point sets remain paths for arbitrarily
large $N$, the displayed error persists as $N\to\infty$.
:::

:::{prf:proof}
The native affine-rank test compares the single nonzero singular value
$s$ to $N\epsilon_{64}s$. The stated bound on $N$ makes the rank one, so
the one-dimensional tessellation connects consecutive sites. With spacing
$\Delta=1/(N-1)$, each row, including the endpoints, has displacement
covariance $\Delta^2$. The native relative-trace ridge gives the common
metric $g=[(1+10^{-5})\Delta^2]^{-1}$; its lower eigenvalue bound does not
bind and its nonzero eigenvalue exceeds the pseudoinverse cutoff. Every
edge has the same metric length and both endpoint volumes are equal.
The raw weights are
$\exp[-1/(2(1+10^{-5}))]/(\sqrt{1+10^{-5}}\Delta)$, so the row-sum floor
does not bind. Each interior row therefore has two weights equal to $1/2$.
Curvature in dimension one is zero; the first cloning gate is closed.
Thus $F_i=\nu[(v_{i-1}+v_{i+1})/2-v_i]$. Independence gives the two
displayed identities. Averaging over the interior rows gives the lower
bound $(N-2)\nu^2T/(2N)$ for the full population-averaged force error.
For the separately stated path-preserving extension this tends to
$\nu^2T/2$. A literal $N\to\infty$ with the retained positive rank tolerance
does not meet $N\epsilon_{64}<1$; at that scale the rank test itself changes
branch. The extension is therefore an additional idealization, not a
fixed-precision Rust limit. Neither calculation excludes a population limit
that retains random neighborhood marks.
$\square$
:::

:::{prf:conjecture} Full graph-dependent population limit
:label: conj-eh-full-population-limit

A full finite-horizon population limit for the native Einstein–Hilbert
update requires an identified joint limit of the position-velocity law,
the geometric reward marks, and the neighborhoods consumed by both B
stages. Its existence and uniqueness are unresolved here. A one-particle
closure must either prove concentration of its graph coefficients or retain
the limiting neighborhood randomness. The preceding proposition computes
the neighbor fluctuation that a dense-force substitution would discard,
and records the separate role of the finite-precision rank cutoff.

The missing estimates are: moment and tail control of the relative-trace
metric and geometric reward at intermediate clone coincidences; consistency
of the evolving tessellation and its two force/curl evaluations; and
stability sufficient to iterate those limits. For long-time statements one
additionally needs a translation-reduced or explicitly confined state
space, tightness there, and a verified ergodicity argument for the relevant
fixed-phase kernel. A uniform-in-$N$ stationary conclusion requires
population-independent constants for that argument and control of the
interchange of the time and population limits.
:::

:::{prf:remark} Scope of the available general theory
:label: rem-eh-literature-scope

The Harris theorem of [Hairer and Mattingly](https://arxiv.org/abs/0810.2777)
requires a suitable Lyapunov drift and minorization; neither is supplied by
the frozen weighted energy inequality for this position-velocity process.
It is not invoked as a theorem about the preset. Work on
[sparse-network diffusions by Oliveira, Reis, and Stolerman](https://arxiv.org/abs/1812.11924)
illustrates why neighborhood laws may enter a population limit. Its graph
and dynamical hypotheses have not been checked for the changing native
tessellation and discrete cloning/Boris update, so it is not used to assert
a limit here. The elementary results above have their complete proofs in
this chapter.
:::

(sec-eh-native-validation)=
## 6. Correspondence with executable checks

:::{div} feynman-prose
The Rust checks connect the measured changes to the signed operator budgets stage by stage. Independent checkpoint replicas test the conditional predictions, while complete trajectories track the sum of cloning and kinetic drifts over successive cycles. The finite-horizon martingale bound compares that accumulated prediction with the observed change using its variance budget; it does not treat successive times as independent samples. Geometry and velocity diagnostics measure local behavior alongside the cloud's centered spread. These are separate measurements of the same executed dynamics.
:::

:::{prf:remark} Implementation and regression coverage
:label: rem-eh-native-validation

`variants::einstein_hilbert::graph_kick_bounds` computes $s_{\max}$, the
maximum column sum of $W$, and $\kappa$ from the supplied native graph and
weights. It rejects nonfinite or negative weights and nonconvex quarter-kicks;
it does not alter the executed dynamics. A bound computed for one graph is
not a global bound for future random geometries. Its floating-point values
are diagnostics, not interval-certified upper bounds.

`cloning::apply_component_rotations` copies each frozen velocity exactly for
unit restitution and an exact identity rotation. Reconstructing the same
mathematical identity as $\bar v+(v-\bar v)$ could previously erase a small
velocity through cancellation. This repair preserves the specified
real-coordinate algorithm and retains the collision diagnostics.

The native regression suites exercise this cancellation in both precisions,
the actual graph force/curl/Cayley kernels, isolated and substochastic rows,
the unweighted energy counterexample, the native path-neighbor variance,
detailed balance with and without a binding row-sum floor, exhaustive even
and odd matching expectations, and the finite coincident-input resource failure. They
also retain the preset serialization and archive replay checks. These are
implementation checks of the proved identities and obstructions; sampled
trajectories do not establish the conjectured population limit.
:::

:::{prf:remark} Computed reference estimates and native simulation evidence
:label: rem-eh-computed-validation

The `gas-eh-proof` runner retains $d=3,h=0.002,T=0.33,\nu=3$, unit friction,
period-$20$ cloning, both native matchings and graph refreshes, and f64 arithmetic.
Its reference initial state is coincident and at rest. The two retained JSON
configurations in `algorithmic-gas/proof-validation/` use the following
evaluated primitive constants:

$$
\begin{aligned}
a&=0.0015,\qquad c=0.9980019986673331,\\
b&=0.0019980019986673334,\qquad
\sigma^2=0.0013173635164828103,\\
\tau^2&=1.3173635164828103\times10^{-9}.
\end{aligned}
$$

Thus the positional Gaussian contribution in (EH.K2) is
$(1-1/N)\,3.952090549448431\times10^{-9}$ per step. The configurations specify:

- $N=32,64,128,256$, seeds $7,1729,991$, $400$ steps each, with $128$
  independent future-seed replicas at gate $40$ for $N=32,64$;
- $N=32,128,500$, the same three seeds, $4000$ steps each, with $256$
  independent future-seed replicas at gate $4000$ for $N=128,500$.

Across $40{,}800$ trajectory steps and $768$ replicas, there were zero failed
algebraic stage checks. The maximum residual, normalized by
$1+|\text{left}|+|\text{right}|$, was $1.335\times10^{-15}$.
The cloned variance predictions use (EH.M2)--(EH.M3), conditional on each
replica's actually sampled fitness. The kinetic predictions use
(EH.K2)--(EH.K3), conditional on each realized post-cloning state and graph.
Each reported standard error is computed over independent completed replicas.

| $N$ | Gate | Replicas | Cloning mean residual / SE | Kinetic mean residual / SE |
|---:|---:|---:|---:|---:|
| 32 | 40 | 128 | 1.028 | -0.794 |
| 64 | 40 | 128 | 1.029 | 1.395 |
| 128 | 4000 | 256 | -0.763 | 0.132 |
| 500 | 4000 | 256 | -0.392 | -0.189 |

The longer runs give the following means over the three complete trajectories.
The cloning ratio is $\mathbb E[W_X\mid x,v,f]/W_x$, averaged at gates
$2020,2040,\ldots,4000$; negative drift counts pool those $100$ gates from
each of the three runs.

| $N$ | $W_x$ at time $8$ | $E_v$ at time $8$ | Mean late cloning ratio | Negative late drifts | Maximum $\kappa$ |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.838764 | 0.265035 | 1.001767 | 161/300 | 1.008112 |
| 128 | 2.063589 | 0.313732 | 1.007416 | 91/300 | 1.007567 |
| 500 | 3.768393 | 0.292508 | 1.007620 | 37/300 | 1.009319 |

The sufficient unweighted velocity test $c^2K^4<1$ requires
$K<e^{h/2}=1.0010005\ldots$. These recorded maxima exceed it, while the
actual mean squared speeds stay much smaller than this worst-column bound
would predict. Likewise the signed cloning term has both signs. These data
support the exact operator identities; they do not discharge the global
coefficient hypotheses of (EH.D6), or prove nonconvergence. In particular,
bounded observed velocities and continued growth of centered positions over
this finite horizon cannot establish a uniform stationary moment bound.

The exact finite-law regressions separately enumerate even and odd matchings,
check (EH.C2) through the actual native geometry and fitness pipeline, and
verify the native first-step linear map (EH.C3). The sampled estimates never
replace those algebraic proofs. Configuration files, the source and executable
hashes, compact numerical tables, and commands for regenerating the reports
are retained with the proof-validation tools.
:::

```{figure} figures/eh_operator_validation.png
:name: fig-eh-operator-validation
:alt: Centered spread, squared speeds, exact conditional cloning increments, and independent-replica residuals of the native Einstein-Hilbert gas.

Reference Rust operator validation. Top panels show three-seed means; shading
is the minimum-to-maximum range across those seeds. Bottom left shows the
three-seed mean of $\mathbb E[W_X-W_x\mid x,v,f]/W_x$ at open gates, using
each run's realized fitness vector. Bottom right shows mean observed-minus-predicted
replica residuals divided by their estimated standard errors; error bars are
$\pm1$ on this standardized scale. These finite-horizon comparisons test the
operator formulas. The table in {prf:ref}`rem-eh-computed-validation` specifies
the populations, gates and replica counts.
```

:::{prf:remark} Longer native trajectories, centered shape, and geometric observables
:label: rem-eh-extended-validation

`gas-eh-stationarity` executes the same reference transition and records the
signed cycle identity (EH.D7). It runs $N=32,128,500$, seeds $7,1729,991$,
for $50{,}000$ steps each (time $100$), then extends the three $N=32$
trajectories to $500{,}000$ steps (time $1000$). The repeated $N=32$ prefixes
have exactly identical cycle records. This accounts for $1{,}950{,}000$
executed steps and $1{,}800{,}000$ distinct seeded steps in these experiments.
There are zero failed stage or cycle identities; the maximum normalized
residual is $1.059\times10^{-12}$. The fluctuation regression additionally
enumerates every matching and acceptance pattern for $N=4,5,6,7$ and
verifies (EH.M4a). Native geometry tests verify (EH.G1)--(EH.G2), and actual
fitness evaluations verify (EH.M0a), including raw channel magnitudes
$10^{120}$.

The following entries average time-window means over the three seeds.
Curvature quantiles are computed within each run's late window and then
averaged across seeds. Positions are centered by their empirical mean;
no length rescaling is applied to $W$.

| $N$ | Mean $W$, time $50$–$75$ | Mean $W$, time $75$–$100$ | Late mean $E_v$ | Late curvature $10/50/90\%$ |
|---:|---:|---:|---:|:---|
| 32 | 5.08790 | 7.23565 | 0.29205 | $-3.082\;/\;0.108\;/\;3.027$ |
| 128 | 21.57966 | 24.76780 | 0.28991 | $-2.682\;/\;0.153\;/\;2.892$ |
| 500 | 66.28417 | 100.46081 | 0.28580 | $-2.118\;/\;0.087\;/\;2.489$ |

For the $N=32$ extension, the mean spread is $25.73337$ on time
$500$–$750$ and $45.58740$ on $750$–$1000$. The corresponding mean squared
speeds are $0.29280$ and $0.29588$. The late curvature quantiles are
$(-3.232,0.094,3.336)$. These finite data show substantially steadier local
geometric and velocity statistics than unscaled centered spread. They do
not establish either eventual convergence or nonconvergence of that spread.

The late signed budgets per period-$20$ cycle are fully evaluated:

| Population and time window | Cloning $d^{\mathrm C}$ | Signed transport $2bC_{Xu}$ | Complete drift budget |
|:---|---:|---:|---:|
| $N=32$, $75$–$100$ | 0.00369563 | 0.00511361 | 0.00883148 |
| $N=128$, $75$–$100$ | 0.01016289 | 0.00563869 | 0.01582442 |
| $N=500$, $75$–$100$ | 0.04169737 | 0.00527614 | 0.04699621 |
| $N=32$, $500$–$750$ | -0.00960400 | 0.00422470 | -0.00535702 |
| $N=32$, $750$–$1000$ | 0.00777747 | 0.00506765 | 0.01286763 |

The final column includes ballistic and thermal terms, with every phase
included. It is the sampled mean of the successive conditional drift
budgets in (EH.D7); subtracting it from the observed spread increment leaves
the two recorded martingale residuals. It is not an independent-sample
estimate based on the individual walkers. The signs in the last two rows
also show why one observed window cannot certify a global restoring drift.

The report evaluates (EH.D10) with $R\in\{2^j:0\le j\le20\}$ and
$\lambda=2^{-l}/c_R$, $1\le l\le20$, using the union factor $J=420$ and
$\delta=0.01$ per run. Each recorded cycle has one open gate and its
variance budget is at least $4(q_N+r_N)W_{\rm gate}^2$ by (EH.M4a);
this checks the radius condition from the saved cycle records. All
accumulated residuals lie within the resulting
bounds. The bounds are conservative: on the time-$1000$ runs their radii
are approximately $7009$–$12173$ while the observed absolute residuals are
$3.59$–$203.71$. They control the consistency check without asserting a
statistically resolved sign of the stationary drift. The ideal-innovation
interpretation of {prf:ref}`def-eh-native-interpretation` applies.

Snapshots are recorded every $500$ steps in the first ensemble and every
$5000$ in the extension, giving $100$ snapshots per run. The distribution
comparison uses $51$ fixed projection directions: the coordinate axes and
$48$ normalized Gaussian directions from analysis seed $104729$.
For two equal-size projected samples, each squared one-dimensional
$W_2$ distance is the mean squared difference of their sorted coordinates;
the reported sliced distance is the square root of their average over the
directions. Consecutive quarter-horizon windows are compared within each
seed. The separate RMS-normalized shape comparison divides every snapshot
by its own centered root-mean-square radius. It does not replace the
unscaled centered distribution in the convergence question.

These simulations support the explicit operator and fluctuation estimates.
The distinction between geometric and spatial observables is consistent
with the exact dilation law (EH.G2). A theorem for long-time convergence
of either law still requires the corresponding drift and refreshed-geometry
estimates. The compact reports, source hashes and reproducible commands are
in `algorithmic-gas/proof-validation/einstein-hilbert-results/` and its
`extended/` subdirectory.
:::

:::{div} feynman-prose
The operator budgets agree with the executed stages, while the longer runs separate three observables: local curvature, the full centered cloud, and shape after rescaling. Centering removes translation; RMS normalization also removes size. Under its stated hypotheses, the curvature dilation law shows how steadier local geometry can coexist with changing spatial scale. The finite window comparisons describe these trajectories and leave eventual convergence or nonconvergence unresolved.
:::

```{figure} figures/eh_stationarity_validation.png
:name: fig-eh-stationarity-validation
:alt: Native Einstein-Hilbert spread, energy, curvature and volume distributions, centered shape distances, and signed cycle drift through time 100.

Three seeds for each native population through time $100$. Spread and
energy lines average nonoverlapping time blocks; their bands are the
minimum-to-maximum range across seeds. Dashed geometric curves are the
$10\%$ and $90\%$ quantiles, and solid curves the medians. They summarize
dependent snapshots descriptively. The bottom panels retain unscaled
centered positions and the complete signed cycle budget.
```

```{figure} figures/eh_extended_validation.png
:name: fig-eh-extended-validation
:alt: Three native N=32 Einstein-Hilbert trajectories through time 1000 with geometry distributions and the signed cycle budget.

The same three $N=32$ seeds extended through time $1000$, with the same
observable definitions. Centered spread and local curvature are measured
separately; no spatial normalization is hidden in the displayed spread.
```
