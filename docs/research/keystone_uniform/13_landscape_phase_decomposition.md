# Landscape observations inside the original Keystone calculation

:::{prf:definition} Phase-resolved centered structural error
:label: def-kulp-centered-phase-error

Use the exact two entering alive centers and common alive index set
$I_{11}$ of {prf:ref}`thm-keystone-discharged-averaged-pressure`.
Set $d_i=(x_i-\bar x_A)-(\widetilde x_i-\bar x_{\widetilde A})$.
Assign each alive row a declared landscape-phase label. Partition
$I_{11}$ by the paired labels into groups $I_g$, and put

$$
w_g=|I_g|/N,\quad \mu_g=|I_g|^{-1}\sum_{i\in I_g}d_i,\quad
W_g=|I_g|^{-1}\sum_{i\in I_g}|d_i-\mu_g|^2.
\tag{KULP.1}
$$

Empty groups contribute zero. The labels are an observation of the
unchanged state, not a new proposal, death rule or dynamics. They may
come from the Rastrigin basin cells or from the declared population
phase classes of {prf:ref}`def-slcpd-increment`.
:::

:::{prf:proposition} Exact within-phase and between-phase Keystone accounting
:label: prop-kulp-keystone-phase-split

The original structural coordinate and its actual pressure split as

$$
W=\sum_gw_g[W_g+|\mu_g|^2],
\tag{KULP.2}
$$

$$
\mathcal A_{11}=\frac1N\mathbb E_m\sum_g\sum_{i\in I_g}
(p_i+\widetilde p_i)
\big[|d_i-\mu_g|^2+2\mu_g\cdot(d_i-\mu_g)+|\mu_g|^2\big].
\tag{KULP.3}
$$

Here $p_i,\widetilde p_i$ are the actual recipient probabilities conditional on
the retained measurements. Their complete normalizers and every
measurement outcome are retained. In particular the acceptance-weighted
cross term in (KULP.3) does not vanish merely because
$\sum_{I_g}(d_i-\mu_g)=0$.

With exactly the original primitive $\chi_*,W_0,B_*$, the same
Keystone theorem remains

$$
\mathcal A_{11}\ge
\chi_*\left(\sum_gw_g[W_g+|\mu_g|^2]-W_0\right)-B_*/N^2.
\tag{KULP.4}
$$

No raw fixed-slot displacement has been substituted for $W$.
The signed incoming-donor, barycenter, collision and kinetic terms
in (KU.F1) remain necessary, including their between-phase parts.
Thus positive pressure does not require stationary populations in
different phases to have vanishing mutual distance.
:::

:::{prf:proof}
Expand $|d_i|^2=|d_i-\mu_g|^2+2\mu_g\cdot(d_i-\mu_g)+|\mu_g|^2$.
The unweighted cross term sums to zero within each group, proving
(KULP.2). Expand the same square before applying the actual varying
acceptance weights to obtain (KULP.3); this weighted cross term is
retained. Substitute (KULP.2), without any other change, in the
already proved Keystone bound to obtain (KULP.4).
Both the common-index normalization $N$ and each entering alive
centering are exactly those of that theorem. A common translation
of one whole alive array leaves its centered positions unchanged;
its centroid displacement belongs to its own exact update ledger,
not to this pressure coordinate. $\square$
:::

:::{prf:lemma} Alive empirical variance with phase occupations retained
:label: lem-kulp-alive-variance-split

For one realized nonextinct array, let $M_a$ alive rows have phase
label $a$, $\omega_a=M_a/M$, and let $\bar x_a,W_a$ be their
actual mean and centered variance. Then exactly

$$
W_A=\sum_a\omega_aW_a+
\sum_a\omega_a|\bar x_a-\bar x_A|^2.
\tag{KULP.5}
$$

The second summand is between-phase geometry. Both summands
are retained for an equilibrium population occupying several wells.
For a stochastic preparation this identity is applied to each actual
realization before averaging. A ratio of expected phase counts and
expected phase moments is not the expected within-phase variance.
:::

:::{prf:proof}
On phase $a$, write $x_i-\bar x_A=(x_i-\bar x_a)+
(\bar x_a-\bar x_A)$. Its unweighted cross term sums to zero.
Divide each resulting sum by the actual $M$ and sum the phases.
This gives (KULP.5) with the actual random denominators if the
array was produced by a random update. $\square$
:::

:::{prf:lemma} Complete Gaussian phase moments without deleting the exterior
:label: lem-kulp-gaussian-phase-moments

For $X\sim\mathcal N(m,\sigma^2I_d)$, $\sigma>0$, a phase box
$B_a=\prod_r[l_{ar},u_{ar}]$ and declared center $o_a$, put
$A_r=(l_{ar}-m_r)/\sigma$, $B_r=(u_{ar}-m_r)/\sigma$,
$P_r=\Phi(B_r)-\Phi(A_r)$ and $P_{-r}=\prod_{s\ne r}P_s$.
The exact unnormalized probability and moments are

$$
\begin{aligned}
P_a&=\prod_rP_r,\\
\mathbb E[X_r\mathbf1_{B_a}]&=
[m_rP_r+\sigma(\phi(A_r)-\phi(B_r))]P_{-r},\\
\mathbb E[|X-o_a|^2\mathbf1_{B_a}]&=
\sum_rP_{-r}\big\{[(m_r-o_{ar})^2+\sigma^2]P_r\\
&\qquad+2\sigma(m_r-o_{ar})[\phi(A_r)-\phi(B_r)]
+\sigma^2[A_r\phi(A_r)-B_r\phi(B_r)]\big\}.
\end{aligned}
\tag{KULP.6}
$$

For disjoint phase boxes retain the exterior probability
$1-\sum_aP_a$, first moment $m-\sum_a\mathbb E[X\mathbf1_{B_a}]$,
and raw second moment $|m|^2+d\sigma^2-
\sum_a\mathbb E[|X|^2\mathbf1_{B_a}]$. Thus the law and all
its noise tails remain the original Gaussian law.
:::

:::{prf:proof}
Integrate $z\phi(z)=-\phi'(z)$ and
$z^2\phi(z)=\phi(z)-(z\phi(z))'$ over $[A_r,B_r]$.
Expand $X_r=m_r+\sigma z$ or $X_r-o_{ar}$ and multiply the
independent other-coordinate masses. This gives (KULP.6).
The exterior is the complement of the disjoint boxes, so subtraction
from the complete Gaussian probability and moments gives the stated
identities. This is integration, not a modification of the noise.
$\square$
:::

:::{prf:theorem} Actual source, jitter and kinetic phase flux in the same ledger
:label: thm-kulp-complete-phase-flux

Freeze the entire actual measured-fitness vector before using its
conditional source tokens $\rho_i(e,j)$ from (KU.15). Let
$\ell(x)$ denote the observed phase, with a separate exterior label.
For persistence use $G_{a,0}(x)=\mathbf1_{\{\ell(x)=a\}}$;
for an accepted copy use
$G_{a,1}(x)=\Pr\{x+\sigma_J Z\in B_a\}$ from (KULP.6).
Then the exact normalized recipient-to-source and source-to-prepared
flux matrices are

$$
C_{ab}=\frac1N\sum_{i,e,j}\rho_i(e,j)
\mathbf1_{\{\ell(x_i)=a,\ell(x_j)=b\}},\qquad
J_{ab}=\frac1N\sum_{i,e,j}\rho_i(e,j)
\mathbf1_{\{\ell(x_j)=a\}}G_{b,e}(x_j).
\tag{KULP.7}
$$

The complete prepared expected occupation is
$\mathbb E\omega_b^C=\sum_aJ_{ab}$. The same token sum with
the second moment in (KULP.6) gives its exact expected fixed-center
error. Mandatory dead-row revival uses its actual $e=1$ source law;
its original recipient and its alive donor remain different indices.
No donor fitness, incoming load or random phase count is discarded.

After freezing the entire realized source, jitter and component-Haar
preparation $(X,V^C)$, let

$$
m_i=X_i+b\{V_i^C+t[F(X_i)+F_{\rm visc}(X,V^C)_i]\},\qquad
\sigma_h^2=t^2q^2+s^2,\qquad
p_{ib}=P_{B_b,\sigma_h}(m_i).
\tag{KULP.8}
$$

Here the label is a spatial-box observation; set
$p_{i\dagger}=1-\sum_{b\ne\dagger}p_{ib}$ for the complete exterior.
The final positions have the exact row phase probabilities $p_{ib}$.
If $Y_b=N^{-1}\sum_i\mathbf1_{\{\ell(x_i^+)=b\}}$, their
conditional normalized-count covariance is

$$
\mathbb E[Y\mid\mathrm{prep}]=\frac1N\sum_ip_i,\qquad
\operatorname{Cov}(Y\mid\mathrm{prep})=
\frac1{N^2}\sum_i[\operatorname{diag}(p_i)-p_ip_i^\top],
\qquad
\operatorname{Var}(Y_b\mid\mathrm{prep})\le\frac1{4N}.
\tag{KULP.9}
$$

This covariance concerns the raw transition before survival conditioning.
The complete raw covariance also retains the covariance of its conditional
mean under the full preparation law. Actual survivor conditioning uses
{prf:ref}`cor-klq-surviving-label-moments` when the cells partition $D$;
a generic core exterior is not terminal death. The latter carries all shared normalizer and Haar
information. The second force, its dense viscosity and the cap
are retained in the full joint integral whenever the observed
phase includes velocities or other full-state variables; they do
not change the already determined position in (KULP.8).
:::

:::{prf:proof}
Condition in the original algorithmic order. Source tokens include
actual persistence, acceptance and mandatory revival; their
conditional rows are independent only after the full measured
normalizers are frozen. Sum their exact probabilities to obtain
$C$. Each accepted recipient subsequently receives its own
independent Gaussian jitter, giving $J$ and its moment identity
by (KULP.6). Component rotations do not alter the prepared positions.
Their retained own-slot velocities enter (KULP.8), which is exactly
the existing two-drift identity (KUK.1) with both original noise
stages integrated. Conditional on this full preparation, those final
positions have independent Gaussian innovations, hence independent
categorical labels. Sum their exact categorical covariances to get
(KULP.9). The law of total covariance restores the preparation
dependence before taking any complete-kernel expectation.
The B2 map acts on the full uncapped post-OU velocity and the
original noisy A2 positions, and the cap acts afterward on that
velocity. Therefore position marginals alone omit neither a
position-stage force nor a noise draw, but cannot replace a joint
full-state integral. $\square$
:::

:::{prf:remark} Relation to the existing landscape examples and execution record
:label: rem-kulp-original-strategy

These identities refine the original Chapter 3 Keystone estimate and
Chapter 06a regional donor, residence and phase-weight calculations.
They do not replace them by a requirement that two arbitrary swarms
contract toward each other. The target for the existing trajectory
theorems remains the actual moving population law from its specified
initial law. Stationary phases, periodic laws or other attainable
limits retain their own centroid, occupancy and transition information.
An identified force zero is not by itself an identified stationary
swarm law.

`fragile.fractalai.theory.landscape_phase` evaluates these Gaussian
moments, complete conditional source fluxes, both actual first-viscous
landing maps, alive empirical within/between variance and conditional
count covariance. `rastrigin_uniform_kinetic_budget` computes all
uncapped stages under the native linear-plus-periodic force and both
viscosity normalizations, using dimension-only Gaussian row columns.
No reward normalization, companion law, force, Gaussian innovation,
velocity cap or terminal boundary is changed. Its floating root and
moment evaluations are diagnostics of the analytic formulas.
:::
