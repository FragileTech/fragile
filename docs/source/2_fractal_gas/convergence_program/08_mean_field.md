# The Mean-Field Law of the Euclidean Gas

(sec-mean-field-foundations)=
## 1. A population law for one complete update

:::{div} feynman-prose
Sit on one walker just before an update. It measures a companion, receives a fitness, and may copy a donor. Other walkers may copy it. These accepted edges join walkers into collision groups, and a group shares one rotation about its center-of-mass velocity. A description of our walker must therefore include the group that can change its velocity.

Increasing the population does not make cloning rare. It makes the random neighborhood of a tagged walker approach a definite probability law. We construct that law, apply the actual kinetic stages, and obtain a discrete nonlinear evolution. The time step stays fixed throughout this construction.

The retained coordinates of a dead slot also matter. They determine its revival donor probabilities, and its retained velocity participates in its collision group. A single number called “dead mass” cannot carry this information. Our probability law includes alive and dead marked states.
:::

:::{prf:definition} Complete one-slot state for the canonical gas
:label: def-mean-field-marked-state

Fix a time step $h>0$, dimension $d\geq1$, velocity radius $V>0$, and restitution $\alpha\in[0,1]$. A slot has state

$$
z=(x,v,a)\in E:=\mathbb R^d\times\overline B_V\times\{0,1\},
$$

where $a$ records eligibility at the completed update. A dead slot retains $x$ and $v$. The full empirical probability and its alive restriction are

$$
L_N=\frac1N\sum_{i=1}^N\delta_{z_i},\qquad
\mu^a(dz)=a\mu(dz),\qquad m(\mu)=\mu(a=1),\qquad
\rho_\mu=\frac{\mu^a}{m(\mu)}.
$$

The canonical configuration uses current-frame, independent, single-companion sampling; global regularized statistics; positive logistic fitness maps; simultaneous component collisions; Gaussian BAOAB noise; independent final Gaussian position noise of amplitude $\sigma_x\sqrt h$, with $\sigma_x>0$; the final radial velocity cap; and terminal boundary classification. There is no absorbing check between these stages.

In the absorbing configuration, $a=\mathbf1_D(x)$ for the closed box $D=\prod_{k=1}^d[\ell_k,u_k]$ at admission and after each complete update. In the unbounded configuration, every finite state is eligible. The objective has continuous reward $r$ with at most quadratic growth, and its potential satisfies

$$
|\nabla U(x)-\nabla U(y)|\leq L_U|x-y|,\qquad
|\nabla U(x)|\leq L_U|x|+B_U.
$$

The results below concern the real-arithmetic transition specified by these operations. Numerical overflow is an execution failure, not an additional physical killing rule.
:::

:::{prf:remark} Retained memory and other configurations
:label: remark-mean-field-regularity

The canonical current-frame configuration is Markov on $E^N$. If donor history, mutable observation providers, or time-dependent objectives are enabled, their state must be included in the population state before defining its transition. A projection onto $(x,v,a)$ then generally has memory. The finite-component proof below uses one current donor and strict increase of one frozen fitness along every accepted live edge; it does not assert that arbitrary historical or multiple-donor mechanisms have that property. The complete-state construction is developed in {doc}`../3_fitness_manifold/04_field_equations`.
:::

:::{prf:definition} Physical state space and densities
:label: def-mean-field-phase-space

The alive phase space is $\Omega=D\times\overline B_V$, with $D=\mathbb R^d$ allowed. Measures, rather than densities, are primary: finite populations are atomic, and a density need not be available for every initial law. The smooth radial cap sends finite velocities into the open ball; the closed ball is used for weak compactness. When the alive restriction has a density, write $f_n$ for that sub-probability density and $\rho_n=f_n/m_n$. This notation does not discard the marked dead law.
:::

:::{prf:definition} Alive and dead population balance
:label: def-phase-space-density

At update $n$, the alive and dead masses are $m_n=\mu_n(a=1)$ and $1-m_n$. With $m_n>0$, every dead slot draws an eligible current donor and is revived during cloning. Subsequent terminal classification determines the new dead mass. With $m_n=0$, companion-based evolution stops at extinction; no restart distribution is implied.
:::

:::{prf:remark} Alive normalization
:label: remark-mean-field-sum-to-integral

An average over alive slots converges to an integral against $\rho_\mu$, whereas an average over all slots uses $\mu$. A donor-selected pair has law $\rho_\mu(dz)P_D(\mu;z,dy)$, not generally $\rho_\mu(dz)\rho_\mu(dy)$. Both distinctions enter the fitness moments.
:::

(sec-mean-field-measurements)=
## 2. Sampled measurements and accepted edges

:::{div} feynman-prose
A walker keeps the measurement it actually drew. Suppose a rare companion gives a large diversity score. Standardization and acceptance act on that score before we average over companions. Replacing it by the average diversity would change which edges are accepted.

We can retain this randomness without retaining a whole finite population. Give each representative walker a measurement mark, compute the population moments of those marked walkers, and carry the resulting fitness into the graph construction. The bounds below come from the actual bounded comparison features and the positive normalization floors.
:::

:::{prf:definition} Squashed comparison and weighted companion laws
:label: def-mean-field-measurement-law

Let $S_R(u)=Ru/(R+|u|)$, and define

$$
\Phi(z)=\bigl(S_{R_x}(x),\sqrt\lambda S_{R_v}(v)\bigr),\qquad
D(z,y)=|\Phi(z)-\Phi(y)|.
$$

For measurement and cloning widths $\epsilon_D,\epsilon_C>0$, put

$$
w_b(z,y)=\exp\!\left[-\frac{D(z,y)^2}{2\epsilon_b^2}\right],\qquad
Z_b(\mu;z)=\int a_yw_b(z,y)\mu(dy),\qquad
P_b(\mu;z,dy)=\frac{a_yw_b(z,y)\mu(dy)}{Z_b(\mu;z)},
\quad b\in\{D,C\}.
$$

Since $D^2\leq D_*^2:=4R_x^2+4\lambda R_v^2$,

$$
0<\kappa_b:=e^{-D_*^2/(2\epsilon_b^2)}\leq w_b\leq1,
\qquad Z_b(\mu;z)\geq\kappa_bm(\mu).
$$

These bounds hold for arbitrarily large physical positions, including retained dead positions. They bound the comparison features, not the physical domain. Finite alive recipients exclude themselves from companion sampling; dead recipients are not in the eligible pool. A singleton alive pool has no eligible distinct measurement companion and uses the specified zero-distance measurement. This finite exception disappears along sequences with a positive limiting alive fraction.
:::

:::{prf:definition} The marked measurement law
:label: def-mean-field-moments

Given $\mu$ with $m>0$, attach $Y_D\sim P_D(\mu;z,\cdot)$ independently to each alive type. Dead types have a dummy mark $\dagger$. Denote the resulting probability on $(z,Y_D)$ by $\widehat\eta_\mu$. The measured separation is

$$
s(z,y)=\sqrt{D(z,y)^2+\delta_D^2},\qquad\delta_D>0.
$$

The four required moments are

$$
\bar r=\int r(z)\rho_\mu(dz),\quad
s_r^2=\int(r(z)-\bar r)^2\rho_\mu(dz),
$$

$$
\bar s=\int\rho_\mu(dz)\int P_D(\mu;z,dy)s(z,y),\quad
s_s^2=\int\rho_\mu(dz)\int P_D(\mu;z,dy)(s(z,y)-\bar s)^2.
$$

Thus the diversity variance includes the companion sampling randomness. It is not the variance of the conditional mean separation.
:::

:::{prf:definition} Regularized standardization and sampled fitness
:label: def-mean-field-fitness-potential

With $\sigma_r,\sigma_s>0$, positive amplitudes $A_r,A_s$, positive floors $\eta_r,\eta_s$, and exponents $p_r,p_s\geq0$, define

$$
\widehat s_r=\sqrt{s_r^2+\sigma_r^2},\qquad
\widehat s_s=\sqrt{s_s^2+\sigma_s^2},\qquad
g_b(q)=\frac{A_b}{1+e^{-q}}+\eta_b,
$$

$$
F_\mu(z,y)=
g_r\!\left(\frac{r(z)-\bar r}{\widehat s_r}\right)^{p_r}
g_s\!\left(\frac{s(z,y)-\bar s}{\widehat s_s}\right)^{p_s}.
$$

Let $\eta_\mu$ be the law of $t=(z,Y_D,F_\mu(z,Y_D))$, with a dummy fitness for dead types. There are configuration constants $0<F_*\leq F_\mu\leq F^*<\infty$. Fitness is sampled once and frozen through the cloning decision.
:::

:::{prf:lemma} Measurement normalization and the self-exclusion error
:label: lem-mean-field-measurement-consistency

Suppose deterministic input arrays satisfy $L_N\Rightarrow\mu$, $m(\mu)>0$, and their first two alive reward moments converge. Then the empirical marked fitness law converges in probability to $\eta_\mu$. In a bounded alive domain, the random errors in the two empirical diversity moments have mean squares $O(N^{-1})$. Their contribution to the normalized fitness error has the same mean-square order.

*Proof.* Conditional on the input array, the measurement draws of different recipients are independent. For a bounded marked test $\psi$, the variance of its empirical average is at most $\|\psi\|_\infty^2/N$. The conditional average is the empirical integral of its donor kernel. The denominator is bounded below by $\kappa_Dm$ in the limit. Removing the mass of one self atom changes the normalized donor law in total variation by at most $1/(\kappa_DM)$, where $M$ is the finite alive count, whenever the distinct-donor pool is nonempty. Consequently its averaged error vanishes.

The separation and its square are bounded. Applying the variance calculation to each gives their concentration. The variance is a continuous polynomial of these two moments. The square-root regularizers have denominators bounded away from zero; on bounded measurement ranges, the standardizers, logistic maps, and powers on $[F_*,F^*]$ are Lipschitz. Their composition gives the asserted fitness estimate. For unbounded reward, first truncate the reward and then use the stated reward-moment convergence. This establishes the marked-law convergence without averaging fitness before acceptance. $\square$
:::

:::{prf:definition} Accepted graph of a frozen population
:label: def-mean-field-accepted-graph

Conditional on the input and measurement marks, each alive recipient independently draws a cloning donor and a uniform gate. Write

$$
p(F,G)=\min\!\left(1,\frac{(G-F)_+}{s_c(F+\epsilon_c)}\right),
\qquad s_c,\epsilon_c>0.
$$

For distinct alive $i,j$, the probability of an accepted edge $i\to j$ is

$$
b^N_{ij}=\frac{w_C(z_i,z_j)}{\sum_{k:a_k=1,\ k\ne i}w_C(z_i,z_k)}p(F_i,F_j).
$$

A dead recipient draws from the same weighted current-donor module using its retained $(x_i,v_i)$ and accepts with probability one:

$$
b^N_{ij}=\frac{a_jw_C(z_i,z_j)}{\sum_{k:a_k=1}w_C(z_i,z_k)}.
$$

Each row has at most one outgoing accepted edge. The undirected connected components of these edges are the collision groups. A rejected proposal creates no edge.
:::

:::{prf:lemma} Finite collision components without weak selection
:label: lem-mean-field-component-bound

If $M\geq m_*N$ and $N\geq2/m_*$, set $C=2/(\kappa_Cm_*)$. Conditional on all input states and measurement marks, the accepted graph is a forest and

$$
\mathbb E|\mathcal C_N(i)|\leq e^{2C},\qquad
\mathbb P\!\left(\operatorname{rad}(\mathcal C_N(i),i)\geq r\right)
\leq\frac{(2C)^r}{r!},\qquad
\mathbb P(|\mathcal C_N(i)|>K)\leq\frac{e^{2C}}K.
$$

*Proof.* A live accepted edge strictly increases frozen fitness. A dead vertex cannot be a target and has one outgoing edge. Order vertices by live fitness, putting all dead vertices first and breaking ties arbitrarily. Every edge increases this order, and every vertex has outdegree at most one. A finite undirected cycle would require every vertex on the cycle to use its outgoing edge within the cycle, creating a directed cycle. Strict increase rules this out.

Every edge probability is at most $C/N$. A simple path of length $\ell$ from $i$ has an increasing leg of length $a$ followed by a decreasing leg of length $\ell-a$: an internal vertex cannot send two outgoing path edges. The possible labels on the increasing leg number at most $N^a/a!$ because their order is fixed once selected. The decreasing leg contributes at most $N^{\ell-a}/(\ell-a)!$. Ignoring intersections only enlarges these bounds. Each edge uses a different recipient draw, so row independence bounds the probability of that path by $(C/N)^\ell$. Therefore the expected number of length-$\ell$ simple paths is at most

$$
\sum_{a=0}^{\ell}\frac{C^\ell}{a!(\ell-a)!}=\frac{(2C)^\ell}{\ell!}.
$$

A vertex at distance $r$ provides such a path; summing over $\ell\geq0$ bounds the expected component size. Markov's inequality gives the final bound. No small acceptance probability was used. $\square$
:::

(sec-mean-field-reactions)=
## 3. The limiting collision neighborhood

:::{div} feynman-prose
The graph estimate explains why a one-walker population description is possible even though a collision changes several velocities. A typical walker has a finite random collision neighborhood, uniformly as the swarm grows. Its neighborhood does not have to be a pair.

There are two ways to meet a neighbor. Our walker can select a donor, or another walker can select it. The first gives at most one outgoing edge. The second is a collection of rare incoming selections, which becomes a Poisson point process of neighbors. This Poisson law concerns the number of neighbors within a single simultaneous update. It introduces no continuous-time attempt clock.
:::

:::{prf:definition} Rooted-component population collision map
:label: def-mean-field-rooted-collision

For $\eta=\eta_\mu$, define the accepted-edge density relative to $\eta(du)$ by

$$
\beta_\mu(t,u)=\frac{a_uw_C(z_t,z_u)}{Z_C(\mu;z_t)}
\begin{cases}p(F_t,F_u),&a_t=1,\\1,&a_t=0.\end{cases}
$$

Its outgoing mass $q_\mu(t)=\int\beta_\mu(t,u)\eta(du)$ is at most one and equals one for a dead type. Construct a rooted marked tree as follows.

1. Draw root type $t_0\sim\eta$. Its outgoing edge is absent with probability $1-q_\mu(t_0)$; conditional on an edge being present, its target has probability law $\beta_\mu(t_0,u)\eta(du)/q_\mu(t_0)$. Continue the freely exposed outgoing chain by the same rule.
2. At each exposed vertex of type $t$, the additional incoming children form a Poisson point process with intensity $\eta(du)\beta_\mu(u,t)$. Recursively expose their incoming children.
3. A child reached through its outgoing edge to its parent has already used its outgoing choice. Do not draw another. A known incoming child is already present; the additional Poisson process does not duplicate that vertex identity. Its intensity remains $\eta(du)\beta_\mu(u,t)$, including any atoms: new vertices may have the same type as the known child. Dead children have no incoming children.

The finite-component bound transfers to this construction by finite exploration and monotone convergence, so its component is finite almost surely. On a component $\mathcal C$ with at least one edge, draw one independent Haar matrix $R_\mathcal C\in O(d)$ and set

$$
\bar v_\mathcal C=\frac1{|\mathcal C|}\sum_{j\in\mathcal C}v_j,
\qquad v_j^c=\bar v_\mathcal C+\alpha R_\mathcal C(v_j-\bar v_\mathcal C).
$$

Every accepted recipient, including a revived slot, receives

$$
x_j^c=x_{\operatorname{donor}(j)}+\sigma_J\xi_j^J,
\qquad\xi_j^J\sim N(0,I_d),
$$

using frozen donor positions and independent jitter. A row without an outgoing accepted edge retains its position. Its velocity still changes if it belongs to a nontrivial collision component. An isolated row retains both coordinates. All rows are alive at this stage. The law of the root output is $\mathcal J(\mu)$.
:::

:::{prf:theorem} Component momentum, energy, and shared covariance
:label: thm-mean-field-component-identities

For every realized component,

$$
\sum_{j\in\mathcal C}v_j^c=\sum_{j\in\mathcal C}v_j,\qquad
\sum_{j\in\mathcal C}|v_j^c-\bar v_\mathcal C|^2
=\alpha^2\sum_{j\in\mathcal C}|v_j-\bar v_\mathcal C|^2.
$$

Conditional on the component and its pre-collision types,

$$
\mathbb E v_i^c=\bar v_\mathcal C,\qquad
\operatorname{Cov}(v_i^c,v_j^c)
=\frac{\alpha^2}{d}
\bigl[(v_i-\bar v_\mathcal C)\cdot(v_j-\bar v_\mathcal C)\bigr]I_d.
$$

These are full-slot identities, including retained dead velocities. Alive-only momentum before revival need not be conserved across revival.

*Proof.* The centered velocities sum to zero, and an orthogonal matrix preserves their norms. Haar invariance under $R\mapsto-R$ gives $\mathbb E R=0$. Rotational invariance makes $\mathbb E[(Ru)(Rw)^T]$ a scalar multiple of $I_d$; taking its trace gives $(u\cdot w)/d$. Substitution proves the covariance formula. In dimension one, Haar $O(1)$ is a uniform sign, so the same calculation applies. Summing the covariance over a whole component gives zero, in agreement with exact momentum conservation. $\square$
:::

:::{prf:theorem} One-step collision consistency
:label: thm-mean-field-one-step-consistency

For input arrays satisfying the measurement-consistency hypotheses and $m(\mu)>0$, the empirical post-collision law converges in probability to $\mathcal J(\mu)$. For every bounded continuous $\phi$,

$$
\mathbb E[L_N^c\phi]\longrightarrow\mathcal J(\mu)\phi,
\qquad \operatorname{Var}(L_N^c\phi)\longrightarrow0.
$$

The same assertion holds for random input arrays converging in probability to the deterministic law $\mu$, with the required moment convergence in probability and uniform integrability when expectations of unbounded quantities are used.

*Proof.* First condition on the complete measured array. Restrict exploration to at most $K$ vertices. For any unexplored row, the probability of hitting one of $K$ exposed targets is at most $CK/N$. These row choices are independent. In the incoming point processes, the sum of squared hit probabilities is at most $C^2K^2/N$. Expanding the product of their probability generating functions therefore gives independent Poisson limits with the displayed intensity measures. Removing exposed labels and conditioning a row not to have hit an earlier target changes its remaining probabilities by $O(CK/N)$; over the bounded exploration the resulting error vanishes. The accumulated finite exploration and repeated conditioning errors are bounded by $A(C,K)/N$ for fixed $K$, with a conservative bound $A(C,K)=O((1+C)^2K^3)$.

The empirical type law converges by {prf:ref}`lem-mean-field-measurement-consistency`. The weights are bounded continuous, their denominators stay positive, and acceptance is continuous at fitness ties as well as elsewhere. Thus the finite exploration converges to the rooted construction. The shared Haar mark and independent jitters can be attached to this finite tree in both constructions. Each finite-tree output is a continuous function of these types and marks.

Remove the exploration cutoff using $\mathbb P(|\mathcal C|>K)\leq e^{2C}/K$, first taking $N\to\infty$ and then $K\to\infty$. This proves convergence for one uniformly tagged root. Explore two uniformly distinct roots together. Their finite neighborhoods have independent limits; the probability of an exploration collision is bounded by $A(C,K)/N$ before removing the cutoff. Convergent global normalizers are deterministic and introduce no residual common mark. Expanding the empirical variance into the diagonal term, bounded by $\|\phi\|_\infty^2/N$, and the two-root covariance proves the second assertion. Conditioning and the subsequence characterization of convergence in probability extend the argument to random inputs. $\square$
:::

:::{prf:remark} Rates and what the graph estimate controls
:label: remark-important-nonlocal-nonlinear

The proof supplies a concrete truncation error $2\|\phi\|_\infty e^{2C}/K$ and finite-exploration errors vanishing with $N$ at fixed $K$. Relative to a prescribed limiting input $\mu$, the empirical kernel-integral error also depends on how the initial empirical law and its moments approach $\mu$. The within-component covariance remains present at every population size; chaos concerns finitely many uniformly chosen distinct slots, whose components separate in the limit.

The conditional fluctuation around the actual finite-population expectation
has a quantitative bound: {prf:ref}`thm-chaos-canonical-conditional-variance`
proves $\operatorname{Var}(L_N'\phi\mid S)\leq A_\phi/N$ for the full
update. Its proof controls all component moments and recomputes the global
statistics when a measurement innovation is changed. This conditional
variance and the bias relative to $\mathcal F_h(L_N(S))$ are distinct terms
in the one-step mean-square error.

For that empirical-input comparison,
{prf:ref}`thm-chaos-canonical-quantitative-bias` also bounds the bias by
$2\|\phi\|_\infty B_*/\sqrt N$, giving a full conditional mean-square
error of order $N^{-1}$. It derives the normalization error and the finite
marked-exploration error separately; convergence of $L_N(S)$ to a different
prescribed law remains a separate input approximation.
:::

:::{div} feynman-prose
Imagine saving one input swarm and running its next update many times with fresh
random draws. The output observable scatters around its finite-population mean.
Measurement draws contribute to that scatter, together with donor choices,
acceptance, shared rotations, jitter, and kinetic noise. The conditional variance
measures this scatter after the complete update, including the boundary decision.

Now compare that mean with the population map applied to the saved swarm's
empirical law. Their difference is the finite-population bias. Repeating the
experiment estimates it more accurately; increasing the population controls its
size. The two estimates above keep these effects separate while accounting for
the correlations inside each collision component and the fluctuations of the
global fitness statistics. Neither calculation replaces a sampled fitness by its
average before acceptance. Both follow the same algorithm that generated the
output.
:::

(sec-mean-field-kinetic)=
## 4. Composing the actual kinetic stages

:::{div} feynman-prose
After collision, the law is still only halfway through an update. A force kick changes velocity, a drift changes position, and the thermostat adds a fresh velocity innovation. Their order determines where the next force is evaluated. We keep that order in the population equation.

The final position noise is also part of the algorithm. In an absorbing box it gives every row a positive chance of landing inside the box, even when its deterministic drift points outward. This lets us derive an alive-mass bound from the actual update rather than insert a revival rate into a differential equation.
:::

:::{prf:definition} Exact BAOAB population stages
:label: def-baoab-update-rule

For the canonical isotropic thermostat with constant factor $B=bI_d$, put

$$
c_h=e^{-\gamma h},\qquad
s_h^2=\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0.\end{cases}
$$

Starting from $(X_0,V_0)\sim\mathcal J(\mu)$, apply

$$
V_1=V_0+\tfrac h2 f_{\lambda_0}(X_0,V_0),\qquad
X_1=X_0+\tfrac h2V_1,
$$

$$
V_2=c_hV_1+s_hB\xi^O,\qquad
X_2=X_1+\tfrac h2V_2,\qquad
V_3=V_2+\tfrac h2 f_{\lambda_2}(X_2,V_2),
$$

$$
X_3=X_2+\sigma_x\sqrt h\,\xi^x,\qquad
V_4=\Pi_V(V_3),\qquad A_4=\mathbf1_D(X_3),
\qquad \Pi_V(v)=\frac{Vv}{V+|v|}.
$$

Intermediate velocity laws live on $\mathbb R^d$; only the completed state is capped. The two Gaussian innovations are independent of each other and of the collision graph, rotations, and jitters. The force is $-\nabla U$ in the canonical configuration; $\lambda_0$ and $\lambda_2$ denote the actual laws at the two force inputs if an explicitly enabled population force is present. The output law of $(X_3,V_4,A_4)$ is $\mathcal F_h(\mu)$. For the unbounded configuration, $A_4=1$.
:::

:::{prf:remark} Explicit viscosity extension
:label: remark-separation-kinetic-death

For a Gaussian locality weight $w_\nu$, either specified normalization gives

$$
f_\lambda(x,v)=-\nabla U(x)+\nu\frac{\int w_\nu(x,y)(w-v)\lambda(dy,dw)}{Z_\lambda(x)},
$$

with $Z_\lambda(x)=\int w_\nu(x,y)\lambda(dy,dw)$ for row normalization and $Z_\lambda=1$ for eligible-count normalization, since revival makes all rows alive before kinetics. The finite empirical convention excludes self where specified; its numerator self-term is zero. The force uses the current intermediate population, including changes to donor velocities. A theorem for the zero-viscosity canonical configuration is not automatically a theorem for a different force normalization.
:::

:::{prf:theorem} The fixed-step mean-field equation
:label: thm-mean-field-equation

For every admissible initial probability $\mu_0$ with positive alive mass, the canonical population evolution is uniquely defined by

$$
\boxed{\quad\mu_{n+1}=\mathcal F_h(\mu_n),\qquad n=0,1,\ldots.\quad}
$$

The map is the rooted collision law followed by precisely the stages in {prf:ref}`def-baoab-update-rule`. For every bounded measurable test $\phi$,

$$
\mu_{n+1}\phi=\mathbb E_{\eta_{\mu_n},\,\mathcal C,\,R,\,\xi^J,\,\xi^O,\,\xi^x}
\phi(X_3,V_4,A_4).
$$

It preserves positivity and total probability. At every fixed finite horizon it is the population limit of the canonical particle update. On the absorbing configuration, use convergence of the full marked initial laws; atoms on the position boundary are allowed. On the unbounded configuration, assume a uniformly bounded initial position moment of order $4+\delta$ for some $\delta>0$. This supplies uniform integrability of the fourth moments needed for the quadratic reward variance, and the same property propagates at each fixed finite horizon.
:::

:::{prf:proof}
:label: proof-mean-field-equation

The graph construction defines a probability because its component is finite almost surely and its outgoing law has total mass at most one. Independent Haar and Gaussian kernels and the deterministic stage maps preserve probability. Existence of each successive law follows once its moments and positive alive mass are established below; uniqueness here means the uniquely specified iterates from a given initial law.

Collision consistency is {prf:ref}`thm-mean-field-one-step-consistency`. For $f=-\nabla U$, the kicks and drifts are continuous and have linear growth. Independent row Gaussian innovations give conditional empirical concentration for bounded tests. The radial cap is continuous. Final position noise gives an absolutely continuous position law, so the boundary of a box has zero output probability; terminal marking is therefore continuous almost surely. Truncation using the finite-horizon moment bounds below handles unbounded functions. These facts prove one-step consistency for the composed map. The positive alive-mass estimate makes the next donor denominators nonzero. Induction yields the assertion at any fixed number of updates. The detailed exchangeability and marginal-chaos consequences are proved in {doc}`09_propagation_chaos`.
:::

:::{prf:theorem} Exact mass and weak field balances
:label: thm-mass-conservation

Let $(Z,Z')$ be the coupled root input and output from the complete construction. Then

$$
\mathcal F_h(\mu)(1)=1,\qquad
m(\mathcal F_h(\mu))=\mathbb P(X_3\in D),
$$

$$
m(\mathcal F_h(\mu))-m(\mu)
=(1-m(\mu))-\mathbb P(X_3\notin D).
$$

For any integrable $\phi$, the exact weak increment is

$$
\frac{\mathcal F_h(\mu)\phi-\mu\phi}{h}
=\frac1h\mathbb E[\phi(Z')-\phi(Z)].
$$

The numerator can be telescoped over collision, B1, A1, O, A2, B2, position diffusion, cap, and terminal marking, using the actual intermediate states. This is an equality of per-step balances, including all source terms.

*Proof.* Every input slot produces one output slot. Revival changes all entering dead marks to alive before the terminal test; terminally dead rows are exactly the event $X_3\notin D$. Subtract the input alive mass for the second identity. The weak identity is the definition of the pushforward law; inserting each intermediate value gives its telescoping form. $\square$
:::

(sec-mean-field-boundary)=
## 5. Alive mass, moments, and stability

:::{div} feynman-prose
A denominator bound is useful only if the dynamics keeps enough donors available. Here we can calculate such a bound. Copying places each slot at an eligible donor position, collision velocities remain bounded, and a controlled set of Gaussian innovations keeps the subsequent drift within a finite radius. The final independent position noise then puts a definite fraction back inside the domain.

The resulting lower bound can be extremely small. It proves that the population map remains defined; it is not a prediction that a simulation should sit near that lower bound. The measured alive fraction should instead be compared with the actual conditional terminal probabilities.
:::

:::{prf:corollary} Positive alive mass from the terminal update
:label: cor-mean-field-positive-alive-mass

In a bounded box, choose a core ball $B(0,r_0)\Subset D$, and let $R_D=\sup_{x\in D}|x|$. All input velocities, including dead ones, have norm at most $V$. Set

$$
W=(1+2\alpha)V,\qquad F_J=L_U(R_D+J)+B_U,
$$

$$
L=R_D+J+\tfrac h2(1+c_h)(W+\tfrac h2F_J)
+\tfrac h2s_h\|B\|G,
$$

where $J,G>0$. For Gaussian jitter amplitude $\sigma_J$, define

$$
p_J=\mathbb P(|\sigma_J\xi^J|\leq J),\quad
p_G=\mathbb P(|\xi^O|\leq G),\quad p=p_Jp_G,
$$

$$
p_0=|B(0,r_0)|(2\pi\sigma_x^2h)^{-d/2}
\exp\!\left[-\frac{(L+r_0)^2}{2\sigma_x^2h}\right]>0.
$$

For every admitted finite population with at least one alive donor,

$$
\mathbb P\!\left(\frac{M_{n+1}}N<\frac{p_0p}{4}\,\middle|\,S_n\right)
\leq e^{-pN/8}+e^{-p_0pN/16}.
$$

Moreover $m(\mathcal F_h(\mu))\geq p_0p>0$ for every $m(\mu)>0$. In the unbounded configuration $m(\mathcal F_h(\mu))=1$.

*Proof.* Literal copying and mandatory revival place each slot inside $D$ before jitter. The component formula gives $|v_i^c|\leq|\bar v|+\alpha(|v_i|+|\bar v|)\leq W$. On $|\sigma_J\xi_i^J|\leq J$ and $|\xi_i^O|\leq G$, the B1 force is at most $F_J$, so the center $X_2$ before final position noise has norm at most $L$. B2 and the cap do not change that center.

Assign independent latent jitter innovations even to rows that do not clone; their good events imply the same bound. Conditional on the entire graph and rotations, the jitter/O good events are independent with probability $p$. A multiplicative Chernoff estimate gives at least $pN/2$ good rows except with probability $e^{-pN/8}$. Conditional on all these preceding innovations, the final position noises remain independent. On every good row, integrating their Gaussian density over the core gives probability at least $p_0$. A second Chernoff bound gives at least $p_0pN/4$ surviving rows with the asserted exception probability. For one limiting root, the same good-event argument gives the expectation bound $p_0p$. $\square$
:::

:::{prf:lemma} Finite-horizon moments on the unbounded domain
:label: lem-mean-field-finite-moments

For the canonical force and any $q\geq1$, there are finite configuration constants $A_q,B_q$ such that, when all finite rows are eligible,

$$
\mathbb E\!\left[L_N'|x|^q\mid S\right]
\leq A_qL_N|x|^q+B_q,\qquad |v_i'|\leq V.
$$

The same estimate holds for $\mathcal F_h$. Thus finite initial moments propagate uniformly in $N$ at every fixed finite horizon. If an initial moment of order $q+\delta$ is uniformly bounded, moments of order $q$ are uniformly integrable at those horizons.

*Proof.* For $N\geq2$, each eligible donor probability is at most $2/(\kappa_CN)$; for a singleton the position is retained. Therefore frozen position copying obeys

$$
\mathbb E\!\left[\frac1N\sum_i|x_i^{\rm copy}|^q\mid S\right]
\leq\left(1+\frac2{\kappa_C}\right)L_N|x|^q.
$$

Gaussian jitter has every finite moment. Collision velocities are bounded by $W$. Linear growth of $\nabla U$ and the finite BAOAB coefficients then bound $|X_2|^q$ by a constant times $1+|X_0|^q+|\xi^O|^q$. Add the finite final position-noise moment. The cap proves the velocity bound, and induction gives the assertion. Applying the estimate at $q+\delta$ proves uniform integrability at order $q$. No compact physical support or stationary moment bound was used. $\square$
:::

:::{prf:lemma} Continuity of the actual population map
:label: lem-mean-field-map-continuity

On the absorbing-box state space with $m\geq m_*>0$, $\mathcal F_h$ is continuous for weak convergence of marked laws. On the unbounded state space it is continuous when weak convergence is accompanied by convergence of the first two reward moments and the moment bounds needed for the kinetic stages.

*Proof.* The bounded positive donor kernels have denominators uniformly separated from zero. Eligibility is the retained discrete mark, so initial atoms on the position boundary cause no discontinuity in these input integrals. Eligible reward is bounded on the box, so its moments and the marked measurement law converge. In the unbounded case use the stated reward-moment convergence. For fixed exploration cutoff $K$, all outgoing integrals, incoming intensity measures, and finite-tree readouts converge by their bounded continuous kernels. Remove the cutoff with the uniform component bound. This proves continuity of $\mathcal J$ for bounded continuous tests. The kinetic composition is continuous by the same Gaussian, cap, and terminal-boundary argument used in {prf:ref}`thm-mean-field-equation`. $\square$
:::

:::{prf:remark} Stability without an assumed contraction
:label: rem-mean-field-analytic-results

Continuity gives stability at every fixed number of iterations. On any compact invariant set $K$, it supplies a uniform modulus

$$
\omega_K(\delta)=\sup\{d_{\rm BL}(\mathcal F_h\mu,\mathcal F_h\nu):
\mu,\nu\in K,\ d_{\rm BL}(\mu,\nu)\leq\delta\}\longrightarrow0.
$$

This follows by compactness and the preceding lemma. Iterating that modulus transfers small input errors through any fixed number of steps. It does not make $\omega_K(\delta)<\delta$, prove uniqueness of a stationary law, or control arbitrarily long times.
:::

(sec-mean-field-analysis)=
## 6. Stationarity, physical time, and the experiments

:::{div} feynman-prose
A stationary population reproduces its law after one complete update. That is a fixed point of the map we have just constructed. Existence of such a law, attraction toward it, and convergence of finite-population conditioned laws are distinct calculations. We can prove the first in the bounded-domain canonical setting. The others require control of the same nonlinear map over long times.

There is also a useful way to write a rate: divide an exact one-step increment by the duration of the step. This is a finite difference. It does not authorize replacing the update by a differential equation that continuously re-evaluates fitness and collision neighborhoods between updates.
:::

:::{prf:theorem} Stationary existence for the canonical absorbing-box map
:label: thm-mean-field-stationary-existence

The canonical absorbing-box population map has at least one stationary marked probability $\mu_*$ with $m(\mu_*)>0$:

$$
\mu_*=\mathcal F_h(\mu_*).
$$

*Proof.* Every entering row copies an eligible donor or retains its own eligible position, so its pre-jitter position lies in $D$. Velocities before collision are capped, and the collision velocities are bounded by $W$. The linear-growth force and Gaussian innovations give a uniform output second position moment $M_2$, independent of the entering dead coordinates. Final position convolution bounds the output position density by $H=(2\pi\sigma_x^2h)^{-d/2}$. The output alive mass is at least $p_0p$, and its velocity lies in $\overline B_V$.

Let $K$ be the set of marked laws with second position moment at most $M_2$, position marginal dominated by $H$ times Lebesgue measure, alive mass at least $p_0p$, capped velocity, and terminally consistent mark $a=\mathbf1_D(x)$. It is nonempty because it contains the output of any admissible law. It is convex and tight. The moment constraint is weakly closed by lower semicontinuity; the density constraint is weakly closed by testing nonnegative continuous compactly supported functions. The latter also excludes position mass on $\partial D$, so terminal mark consistency is preserved by weak limits. Thus $K$ is compact. The output bounds give $\mathcal F_h(K)\subset K$, and {prf:ref}`lem-mean-field-map-continuity` gives continuity on $K$. The compact-convex fixed-point theorem applied in the locally convex space of finite signed measures with the weak topology yields a fixed point. $\square$
:::

:::{prf:remark} What stationary identification still requires
:label: remark-cemetery-state

A finite-$N$ killed-chain QSD satisfies a conditioned eigenmeasure equation, whereas $\mu_*$ satisfies the nonlinear fixed-step equation above. Identifying limits of finite-$N$ QSDs requires the actual survival probabilities and concentration or attraction estimates. The finite-component bound proves finite-time consistency but is not a long-time contraction. Positive noise alone does not prove that the nonlinear map has a unique attractor. For an unbounded confining domain, the finite-horizon moment estimate likewise does not yet supply a uniform-in-time Lyapunov bound. These unresolved stationary steps and their dependent claims are treated explicitly in {doc}`09_propagation_chaos`.
:::

:::{prf:proposition} Fixed-step balance and the small-step obstruction
:label: rem-mean-field-attempt-scaling

Define the exact nonlinear increment functional

$$
\mathcal A_h(\mu)\phi=\frac{\mathcal F_h(\mu)\phi-\mu\phi}{h}.
$$

Then $\mu_{n+1}\phi-\mu_n\phi=h\mathcal A_h(\mu_n)\phi$. Suppose along $h\downarrow0$ the complete one-step law converges to $\mathcal F_0(\mu)$ and there is a bounded continuous test with $\mathcal F_0(\mu)\phi\ne\mu\phi$. Then $\mathcal A_h(\mu)\phi$ has no finite limit.

*Proof.* Its numerator converges to a nonzero constant. Division by $h\downarrow0$ diverges. $\square$

For an all-alive initial law and fixed nonzero jitter/collision parameters, the vanishing-step kinetic stages tend to the identity before the terminal cap and classification. Thus $\mathcal F_0$ retains the order-one collision, copying, jitter, cap, and terminal-mark operations. The smooth cap is itself an order-one operation: for any nonzero finite velocity, $|\Pi_V(v)|<|v|$, even strictly inside the radius $V$. Repeating this same cap alone gives $|v_n|^{-1}=|v_0|^{-1}+n/V$ when $v_0\ne0$. Thus at $n\approx t/h$ its effect is singular as $h\downarrow0$; it cannot be replaced by a reflecting velocity boundary. The full $\mathcal F_0$ need not be the identity. A continuous physical-time limit can therefore have an initial fast relaxation or require a different state/time description. A finite-rate differential equation is identified only after proving the relevant limit of these same iterates; changing acceptance probabilities to enforce that limit changes the configured algorithm.
:::

:::{prf:remark} Boundary losses at the actual noise scale
:label: remark-numerical-validation-killing-rate

The terminal position increment includes $\sigma_x\sqrt h\,\xi^x$. Its boundary layer therefore differs from a purely ballistic position update with only $O(h^{3/2})$ integrated thermostat noise. At fixed $h$, the exact quantity is $\mathbb P(X_3\notin D)$ under the full root construction. Neither a ballistic flux formula nor a smooth interior killing rate can replace this probability without a matching limiting argument and boundary regularity estimates.
:::

:::{prf:theorem} Transfer of independent-reference coupling estimates
:label: thm-mean-field-limit-informal

Let $X_1,\ldots,X_N$ be interacting states and let $Y_1,\ldots,Y_N$ be independent with common law $\mu$. For a metric $d_E$ and a coupling with
$\varepsilon_N=N^{-1}\sum_i\mathbb E d_E(X_i,Y_i)^2<\infty$,

$$
\mathbb E W_2(L_N^X,\mu)
\leq\sqrt{\varepsilon_N}+\mathbb E W_2(L_N^Y,\mu).
$$

For an $L$-Lipschitz test $\phi$ of finite $\mu$-variance,

$$
\mathbb E|L_N^X\phi-\mu\phi|
\leq L\sqrt{\varepsilon_N}+\sqrt{\operatorname{Var}_\mu(\phi)/N}.
$$

*Proof.* Pair $X_i$ with $Y_i$ to couple their empirical measures. The triangle inequality and Jensen's inequality give the Wasserstein bound. For the test, split through $L_N^Y\phi$, use the Lipschitz bound for the paired term, and use independence to compute the reference empirical variance. $\square$

This transfer inequality is available when a coupling estimate has been established. The rooted-component proof supplies empirical consistency directly and does not assume that estimate as a premise.
:::

:::{prf:remark} Experimental quantities tied to the population map
:label: remark-mean-field-experiments

The Part III Rust experiments test the actual stage predictions:

- accepted live edges increase frozen sampled fitness, and the accepted graph is a forest;
- full-component momentum and relative energy obey {prf:ref}`thm-mean-field-component-identities`, including revived slots;
- conditional cross covariance uses the shared rotation, with all donor velocity changes included;
- one-step root observables are compared with the rooted-component law, keeping finite-population error separate from independent integration error;
- increasing-$N$ comparisons hold $h$ and the number of updates fixed, and estimate uncertainty across independently seeded complete runs;
- killing and revival diagnostics count actual events and compare terminal alive mass with the conditional final-noise probabilities;
- stationary concentration and dependence on initialization are measured separately from fixed-horizon chaos.

The bounds on component tails and alive mass are inequality predictions. Their conservative constants are not fitted rates or expected equalities. Empirical agreement with a one-step identity does not supply the unresolved stationary attraction estimate.
:::
