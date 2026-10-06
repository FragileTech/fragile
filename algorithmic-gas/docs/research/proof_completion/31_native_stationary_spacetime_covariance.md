# Quantitative native stationary spatial and recorded-time covariance

(sec-nsc-register)=
## 1. Complete parameters, actual records and observation budgets

:::{prf:definition} Native stationary covariance register
:label: def-nsc-register

Retain every algorithm, landscape, arithmetic, mask, stage and calibration
field of {prf:ref}`def-nje-register` and
{prf:ref}`def-nqg-register`. Spatial results use the existing reset
relation $a_x=0$ of (NQG.14), with original $q,s,t,\nu,\rho>0$.
The full active count gas, sampled fitness and global normalization,
both donor roles, mandatory revival, frozen sources, component Haar
collisions, unbounded jitters and Gaussians, configured cap and every
terminal status remain unchanged. No independent incoming swarm is
substituted. The analytic kernel is the proved real-coordinate kernel;
finite arithmetic and graph/payload disagreement retain their actual
comparison errors. The larger-cap reference and other force or
normalization choices retain their previously proved separate scopes.

Use $\eta_N,\eta_*,d_A,\mathcal R_{A,N},V_1,Z_2$ from Chapter NQG.
For a terminal query core $K\subset B(0,R_x)$ put

$$
h_\tau=(2\pi\tau^2)^{-d/2},\quad M_m=bV_c,\quad
d_x=h_\tau e^{-(R_x+M_m)^2/(2\tau^2)},\quad
\Xi=(h_\tau/d_x)^2,\quad r_N=N^{-1/d}.
\tag{NSC.1}
$$

Every prepared center satisfies $|m_i|\le M_m$ deterministically.
In particular $d_x\le\rho_{\eta_N}(x),\rho_{\eta_*}(x)\le h_\tau$
on $K$, rather than merely with high probability.
The preparation quadratic certificate is
$E W_{2,A}(\eta_N,\eta_*)^2\le\mathcal R_{A,N}$.

Let $H_N,G_N$ be bounded actual rooted one-neighborhood observations,
with bounds $B_H,B_G$. Their two determining windows $W_H,W_G$
are contained in $B(0,R)$, $r_NR\le1$, and they consume the smooth
budgets of {prf:ref}`def-nqg-observation-budget`.
Retain their exact finite-$N$ metric ridges, clamps, floors, volumes,
edge/face weights and calibration. Their geometric dependence may
be measurable; their mark dependence has its displayed consumed
Lipschitz budgets. The canonical color channel has $d=3$; position,
posterior, force and time estimates otherwise retain general $d$.

The spatial decorrelation theorems below use a FIXED configured
finite color calibration and fixed global observation parameters,
including the existing explicitly supplied reference-length branch.
A calibration computed from the complete current output or a
different history frame remains its actual random shared readout;
it is not part of $\eta_N$ merely because it is retained in the
complete execution ledger. The explicit comparison below keeps
its extra error. Independent Poisson spatial configurations do
not make a random shared calibration independent.

Write $C_H=C_{H,W_H}^{\rm env}$, $C_G=C_{G,W_G}^{\rm env}$
for (NQG.28), and $L_{F,H},L_{F,G}$ for their original force budgets.
The reference conditional means $\Phi_{H,N}(\eta,x)$ and
$\Phi_{G,N}(\eta,y)$ use the complete field $F_\eta$, root posterior
at $x,y$, and the SAME Poisson geometry and metric marks within
each window. A neighbor at $u$ keeps its actual physical argument
$x+r_Nu$ in (NQG.16). The reference means are derived comparison
instruments, not modified executed algorithms.

For observations defined directly on fixed windows there is no
determination error. A readout of actual full Delaunay stars uses
the protection term below. An alive-only geometry additionally
requires its determining windows inside the original alive box.
:::

(sec-nsc-pair-law)=
## 2. Exact two-root posterior and the unchanged joint geometry

:::{prf:lemma} Native two-root preparation weight and source correction
:label: lem-nsc-two-root-posterior

Select two distinct slots $I,J$ uniformly after the actual preparation.
For fixed finite queries $x,y\in K$, let
$\psi_x(A)=\varphi_\tau(x-m(A))$. Their conditional joint position
density on the complete preparation is exactly

$$
w_N(\eta_N;x,y)=
\frac{N\rho_{\eta_N}(x)\rho_{\eta_N}(y)
             -\eta_N(\psi_x\psi_y)}{N-1}.
\tag{NSC.2}
$$

It lies in $[d_x^2,h_\tau^2]$. Thus conditioning the ENTIRE raw
law on these queries tilts its original preparation law by
$w_N/Ew_N\le\Xi$.

Within a fixed preparation, the actual root-source law is the
product of its two original posterior slot laws conditioned on
the slots being distinct. Its distance from the unconditioned
product posterior is at most

$$
p_N=\min\{1,\Xi/N\}.
\tag{NSC.3}
$$

Every original posterior residual remains an independent standard
Gaussian on its own distinct root. No same-slot residual is
assigned to two distinct native roots.
:::

:::{prf:proof}
Conditional on the prepared array, the original terminal position
rows are independent Gaussians with their respective centers $m_i$.
Average $\psi_x(A_i)\psi_y(A_j)$ over $i\ne j$ to obtain (NSC.2).
Each product is between $d_x^2$ and $h_\tau^2$, proving both
the density bounds and the global Bayes factor.

Independently tilt uniform slots by $\psi_x$ and $\psi_y$.
Their same-slot probability is
$\sum_i\psi_x(A_i)\psi_y(A_i)/
 [N^2\rho_{\eta_N}(x)\rho_{\eta_N}(y)]\le\Xi/N$.
Conditioning this product on distinct slots gives precisely the
actual pair weights proportional to $\psi_x(A_i)\psi_y(A_j)$.
The TV of a law and its conditioning on an event is the complement
probability. Append the native residuals using (NMG.2)--(NMG.3).
They are independent on the two retained distinct original rows.
:::

:::{prf:definition} Primitive separated-window geometry error
:label: def-nsc-geometry-error

Suppose $|x-y|>2r_NR$, and put

$$
\begin{gathered}
V_W=|W_H|+|W_G|,\qquad
M_W=\int_{W_H}|u|\,du+\int_{W_G}|u|\,du,\\
\epsilon_{\rm win}(N,R)=
p_N+\frac{h_\tau^2V_W^2+2h_\tau V_W}{N}
                         +r_NL_{\tau,1}M_W.
\end{gathered}
\tag{NSC.4}
$$

For full one-neighborhood stars let

$$
\beta=\frac14h_\tau
 e^{-(R_x+1+M_m)^2/(2\tau^2)},\quad
C_d=2\,9^d(1+v_d(1)),\quad
\eta_{\rm star}(R)=C_d[1+h_\tau v_d(R)]
                              e^{-\beta v_d(R/64)} ,
\tag{NSC.5}
$$

where $v_d(r)=v_d(1)r^d$ is ball volume. Define
$\epsilon_{\rm geom}=\epsilon_{\rm win}+2\eta_{\rm star}$.
For fixed-window observations put instead
$\epsilon_{\rm geom}=\epsilon_{\rm win}$.
Each probability error can be capped at one.
:::

:::{prf:lemma} Two native marked windows retain their full geometry
:label: lem-nsc-joint-marked-windows

For $N\ge8$ and the separated queries above, the population-force
versions of the two actual observations couple, on the same
prepared array, to conditionally independent marked Poisson
observations with means
$\Phi_{H,N}(\eta_N,x),\Phi_{G,N}(\eta_N,y)$,
at error at most $\epsilon_{\rm geom}$.
They use the same complete population force and actual calibration.
Independence holds between the two comparison configurations
conditional on the common preparation, and need not hold between
their unconditional outputs.
:::

:::{prf:proof}
First retain the actual distinct root marks. Every remaining
original row contributes at most one point to the union of the
two disjoint windows. Its hit probability is at most $h_\tau V_W/N$.
The original Bernoulli-to-Poisson coupling on that marked union
costs at most the sum of squared probabilities
$h_\tau^2V_W^2/N$. The residual marks are conditionally independent
of terminal positions by (NMG.2); appending them and their original
prepared source marks does not change this coupling.
Freeze each intensity at the corresponding root query. Gaussian
gradient control costs $r_NL_{\tau,1}M_W$. Restore both removed
root atoms in both complete intensities, at mass at most
$2h_\tau V_W/N$. Restrictions of the resulting Poisson measure to
the disjoint windows are independent by its count generating
function. Finally couple the two root marks to independent
posterior roots by (NSC.3).

For full stars the original guard argument of
{prf:ref}`lem-nga-two-root` applies uniformly, without its moment
core exception: all current centers are bounded by $M_m$.
After deleting the two roots and one candidate neighbor, $N\ge8$
leaves at least $N/2$ rows. Their original Gaussian densities on
the one-unit query enlargement are bounded below by $4\beta$.
Each guard has its original rescaled volume, so the same
empty-guard and candidate-count union bound is bounded by
(NSC.5). Each of the two comparison stars costs at most
$\eta_{\rm star}$. Successful marked coupling and protection
make all coordinates, residuals and adjacent stars agree.
Retessellation, metric construction, weights, volumes and masks
are then evaluated on these identical configurations.
No continuity of a tessellation on failed configurations is used.
:::

(sec-nsc-original-force)=
## 3. Original two-root-conditioned dense forces

:::{prf:lemma} Two-query native count-force budget
:label: lem-nsc-two-query-force

Put $S_H=1+h_\tau|W_H|$, $S_G=1+h_\tau|W_G|$ and

$$
C_{F,2}=512(1+d)\nu^2
\left[dq^2+(1+\Xi)c^2V_1^2+
 a_Y^2(R_x+1+M_m)^2+q^2\chi d\right].
\tag{NSC.6}
$$

Before the global preparation Bayes tilt, the expected sum
of original finite-B2 force discrepancies over the root
and its selected marks is at most
$S_H\sqrt{C_{F,2}/N}$, and similarly for $G$.
After globally conditioning the two queries the bound is
at most $\Xi$ times that value. Hence replacing original
forces by their own complete common-stage field changes
the two observations in expected absolute value by at most

$$
D_{H,N}=\Xi L_{F,H}S_H\sqrt{C_{F,2}/N},\qquad
D_{G,N}=\Xi L_{F,G}S_G\sqrt{C_{F,2}/N}.
\tag{NSC.7}
$$

No final B2 force is recomputed from a local window.
:::

:::{prf:proof}
Freeze the full preparation and each observed row's own
posterior stage inputs. All O rows except the two conditioned
roots and that own row are still original independent Gaussians.
The sum of their centered count-kernel variances, divided by
$N^2$, is bounded using
$K_\rho\le1$ and
$|z_l-z_j|^2\le2|z_l|^2+2|z_j|^2$.
The independently averaged own row is missing, and each
other conditioned root replaces one averaged summand.
Their squared corrections cost $N^{-2}$ times their original
and posterior second moments. This is exactly the calculation
of (NMG.8), retaining at most two extra conditioned summands.
More explicitly, with $C$ the set of at most two other
conditioned roots and $M_2=\eta_N|v_1|^2$, the squared error
is bounded, conditionally on all these own inputs, by

$$
\frac{4\nu^2}{N}[c^2M_2+dq^2+|z_j|^2]
 +\frac{\nu^2}{N^2}
\left[24\sum_{l\in C}(|z_l|^2+c^2|v_{1l}|^2+dq^2+2|z_j|^2)
              +12(c^2|v_{1j}|^2+dq^2+|z_j|^2)\right].
$$

The first term is twice the independent variance sum.
For each conditioned summand use
$|T_{jl}-ET_{jl}|^2
\le4|z_l|^2+4c^2|v_{1l}|^2+4dq^2+8|z_j|^2$.
The missing self average costs at most
$2(c^2|v_{1j}|^2+dq^2+|z_j|^2)$.
The squared sum of these at most three biases costs three
times their squared sum, followed by the factor two for
adding variance and bias. These give the displayed coefficients.

In the actual distinct-root posterior, the weight of any
one fixed root slot is at most $\Xi/N$, by the numerator
$h_\tau^2$ and denominator $d_x^2$ in (NSC.2). Thus each
root's $v_1$ second moment is at most $\Xi\eta_N|v_1|^2$.
Every selected neighbor has density at most $h_\tau$ and,
at its actual query $x+r_Nu$, posterior bound

$$
E(|z_j|^2\mid\mathcal F,Y_j=x+r_Nu)
\le3c^2|v_{1j}|^2+
3a_Y^2(R_x+1+M_m)^2+3q^2\chi d .
$$

Its expected count is at most $h_\tau|W_H|$.
Consequently summing the variance and the three missing/conditioned
row corrections over the root and these neighbors is bounded
by $S_H C_{F,2}/N$, after original preparation averaging:
$E\eta_N|v_1|^2\le V_1^2$.
The displayed factor $512(1+d)$ exceeds the squared-sum
factors and the at most three row corrections for $N\ge8$.
An additional other root lying in a determining window is
excluded by the stipulated separation.

Cauchy--Schwarz on the random root/neighbor count gives
$E\sum|\Delta F|\le
\sqrt{E\#}\sqrt{E\sum|\Delta F|^2}
\le S_H\sqrt{C_{F,2}/N}$.
Apply its consumed force coefficient and the bounded global
preparation tilt. Correlations of selected positions and
their force-input velocities remain in the posterior bound.
:::

(sec-nsc-spatial-covariance)=
## 4. Quantitative native spatial covariance

:::{prf:theorem} Separated-root covariance for the actual QSD and Doob record
:label: thm-nsc-spatial-covariance

For the two globally conditioned distinct native root queries
$x,y$ above, the raw one-update law from $\nu_N$ satisfies

$$
\left|\operatorname{Cov}(H_N,G_N\mid Y_I=x,Y_J=y)\right|
\le
2\Xi C_HC_G\mathcal R_{A,N}
 +2B_GD_{H,N}+2B_HD_{G,N}
 +6B_HB_G\epsilon_{\rm geom}(N,R).
\tag{NSC.8}
$$

For its actual QSD surviving output add
$6B_HB_G\epsilon_{{\rm kill},2}$, where

$$
\epsilon_{{\rm kill},2}=
\begin{cases}
0,&x\in D\ \hbox{or}\ y\in D,\\
(1-a_{\rm reset})^{N-2},&\hbox{otherwise},
\end{cases}
\quad
a_{\rm reset}=
[\Phi((L_D-M_m)/\tau)-\Phi((-L_D-M_m)/\tau)]^d.
\tag{NSC.9}
$$

For $N\ge\widehat N_*$ define
$m_N=1-\widehat b_N$, $d_N=\widehat b_N/m_N$
using the sharper eigenfunction constants in Chapter NJE.
The actual stationary Doob record, with both root queries
conditioned in its complete original instrument, adds only
$6B_HB_Gd_N$. These corrections also apply to all the
native metric and force/calibration marks consumed by $H_N,G_N$.
:::

:::{prf:proof}
Replace the original force observations by their common-field
versions on the SAME native record, using (NSC.7).
For any paired bounded random variables, changing them with
expected absolute discrepancies $a,b$ changes covariance by
at most $2B_Ga+2B_Hb$: split the product expectation and
the product of its two expectations.

Under the exact globally query-tilted preparation law,
couple the common-field pair to the conditionally independent
Poisson pair of the preceding lemma. A bounded covariance
changes by at most $6B_HB_G$ times its TV error.
The comparison covariance is exactly

$$
\operatorname{Cov}_{w_N/Ew_N}
 \bigl(\Phi_{H,N}(\eta_N,x),\Phi_{G,N}(\eta_N,y)\bigr).
$$

Subtract the deterministic phase means
$\Phi_{H,N}(\eta_*,x),\Phi_{G,N}(\eta_*,y)$.
Both deviations are bounded by $C_H\epsilon,C_G\epsilon$,
where $\epsilon=W_{2,A}(\eta_N,\eta_*)$,
by (NQG.28) and its same-configuration marked coupling.
The product expectation and product of expectations are each
bounded by $C_HC_GE\epsilon^2$. The exact preparation tilt
is at most $\Xi$, giving the first term in (NSC.8).
This uses the QUADRATIC preparation certificate; replacing it
by its mean square root would lose the stated covariance rate.

Given the prepared array and the two original terminal root
queries, each of the $N-2$ other positions still uses its
independent original Gaussian with center bounded by $M_m$.
Each hits the actual box with probability at least $a_{\rm reset}$.
If neither root is alive, their joint all-dead probability is
bounded by (NSC.9); if a root is alive it is zero.
Conditioning survival changes TV by at most this probability.
The QSD equation identifies that surviving output with $\nu_N$,
including its actual transition record conditioned accordingly.

The complete stationary Doob history and native $\nu_N$-started
survivor history have exact density $e_N(S_1)/\nu_N(e_N)$.
Conditioning their joint root queries changes its normalization,
but still tilts by the SAME terminal function in $[m_N,1]$.
Thus their conditional TV is at most $d_N$, without dividing
an unconditional TV error by a zero-probability query event.
The covariance perturbation gives its stated correction.
:::

:::{prf:corollary} Explicit rate at a fixed positive physical separation
:label: cor-nsc-spatial-rate

For full stars use

$$
R_N=64\left[\frac{2\log N}{\beta v_d(1)}\right]^{1/d}.
\tag{NSC.10}
$$

At sufficiently large permitted $N$, require $r_NR_N\le1$,
$|x-y|>2r_NR_N$, and the actual alive-only interior margins,
when that is the consumed channel. These are explicit tests;
they eventually hold at fixed strictly separated interior queries.
If the observation bounds and summed mark/force budgets are
uniform, $C_H,C_G=O(1+R_N^d)$, so (NSC.8)--(NSC.9) give

$$
|\operatorname{Cov}(H_N,G_N)|
=O\!\left[
(\log N)^2N^{-a_p}
 +N^{-1/2}\log N+
N^{-1/d}(\log N)^{1+1/d}
 +\frac{(\log N)^2}{N}\right]
+O(d_N+\epsilon_{{\rm kill},2}).
\tag{NSC.11}
$$

All implied constants are the displayed primitive budgets.
For $d=3,p=4$, the phase term is
$(\log N)^2N^{-4/35}$. The physical queries are
$\ell_*x,\ell_*y$, their determining radii are
$\ell_*r_NR_N$, and their separation test retains $\ell_*$.
For fixed windows the corresponding bound has no logarithmic
growth or star-protection term.
:::

:::{prf:proof}
Equation (NSC.10) makes
$e^{-\beta v_d(R_N/64)}=N^{-2}$, so the star error is
$O((1+\log N)N^{-2})$.
The Gaussian/Poisson union terms are respectively
$O(R_N^{2d}/N)$ and $O(r_NR_N^{d+1})$.
The force expected count is $O(1+R_N^d)$.
Each common-environment coefficient in (NQG.28) has
at most linear determining-volume growth under the stated
uniform consumed budgets. Insert the already proved
$\mathcal R_{A,N}=O(N^{-a_p})$ with its retained low-alive
and survival terms into (NSC.8).
The original Gaussian innovations remain unbounded.
:::

:::{prf:corollary} Literal hard-color covariance with a derived modulus
:label: cor-nsc-hard-color

Retain the actual strict threshold $\delta_c\ge0$, available
projectors, finite $\kappa_c$, velocity cutoffs and budgets of
{prf:ref}`cor-nqg-hard-observation` for each window.
For $0<\epsilon\le1/4$ put

$$
\begin{aligned}
E_H^{\rm hard}(\epsilon)
={}&2B_H[L_\psi+|W_H|L_{\tau,1}]
                         \sqrt{\mathcal R_{A,N}}\\
&+C_H^{\rm hard}\left[
 S_H\mathfrak B(2\epsilon)+
 \frac{C_{W_H}\sqrt{\mathcal R_{A,N}}
                    +S_H\sqrt{C_{F,2}/N}}{\epsilon}\right],
\end{aligned}
\tag{NSC.12}
$$

and the corresponding $E_G^{\rm hard}$.
Here $C_H^{\rm hard},C_{W_H},\mathfrak B$ are exactly
the primitive mask, coordinate and band budgets of
(NQG.27), (NQG.31)--(NQG.32), including their enlarged
neighbor query radius. Then the actual separated-root hard
covariance is at most

$$
2\Xi[B_GE_H^{\rm hard}(\epsilon)+
                         B_HE_G^{\rm hard}(\epsilon)]
+6B_HB_G[\epsilon_{\rm geom}
                    +\epsilon_{{\rm kill},2}+d_N].
\tag{NSC.13}
$$

Minimize over the displayed $\epsilon$ and permitted $R$.
The resulting primitive bound tends to zero at fixed strictly
separated interior queries: first hold $R,\epsilon$ fixed,
pass $N$, decrease $\epsilon$, then increase $R$.
On fixed compact own/OU supports its band has the explicit
power of (NQG.24); no unknown-law anti-concentration profile
or global power is added.
:::

:::{prf:proof}
Use the same successful two-window marked coupling. Its
root and neighbors retain their actual stage arguments.
The primitive hard-band proof controls changed availability
and normalization by (NSC.12); the two-root force budget
replaces its one-root finite-force budget.
On an unmatched source/geometry event pay the bounded
observation. Compare the actual pair to the independent
deterministic-phase pair in expected absolute observation
errors, then use the bounded covariance perturbation as above.
The exact global preparation tilt costs at most $\Xi$.
All expected band counts concern the actual common-field
comparison with source posterior at the root; the threshold,
force normalization, original phase and cutoff derivatives
are unchanged. The stated order of limits removes every
displayed primitive error, and the finite infimum gives the
same conclusion without imposing a rate on an unknown law.
:::

(sec-nsc-calibration)=
:::{prf:corollary} Original random calibration keeps its explicit covariance error
:label: cor-nsc-calibration-error

Retain an actually consumed random calibration vector $\chi_N$
and compare its readout on the SAME original record to a fixed
configured value $\chi_0$. Suppose its consumed pointwise mark
budget gives
$|H_N^{\chi_N}-H_N^{\chi_0}|
\le \min\{2B_H,L_{\chi,H}|\chi_N-\chi_0|\}$,
and similarly for $G$.
Then in any of the declared raw, survivor or Doob ensembles
the actual covariance satisfies

$$
|\operatorname{Cov}(H_N^{\chi_N},G_N^{\chi_N})|
\le|\operatorname{Cov}(H_N^{\chi_0},G_N^{\chi_0})|
 +2B_GE\min\{2B_H,L_{\chi,H}|\chi_N-\chi_0|\}
 +2B_HE\min\{2B_G,L_{\chi,G}|\chi_N-\chi_0|\}.
\tag{NSC.14}
$$

This is an exact same-record comparison; it makes no independence
assumption about $\chi_N$. For the actual projector color phase,
on the consumed $|z|\le R_F$ cutoff one may take
$L_{\kappa,H}=2R_FL_{P,H}$ plus its other explicit calibration
and cutoff derivatives, since
$\|\partial_\kappa P\|_F\le2|z|$.
The phase and available-force masks are evaluated from the
same original inputs.

Specifically, the Python aggregation `estimate_ell0` midpoint
SPATIAL companion-distance median and analysis current alive
SPATIAL companion-distance mean have, in the canonical current
alive-donor box branch with their actual source alignment,
deterministic length bound
$0\le\ell_0\le\max\{1,2R_D\}$, including their original fallback one.
Thus their actual
$\kappa=m_{\rm col}\ell_0/h_{\rm den}$ has the primitive range
$|\kappa|\le |m_{\rm col}|\max\{1,2R_D\}/|h_{\rm den}|$.
Here $h_{\rm den}$ is that readout's LITERAL positive consumed
denominator, including its floor exactly when that channel uses one.
The range gives a finite deterministic upper bound on each
expectation in (NSC.14). It does not give a vanishing calibration
error or a deterministic calibration limit. Without a derived
stronger budget, those existing random-calibration spatial channels
retain precisely this extra term, rather than the fixed-calibration
decorrelation conclusion.
The Rust spectroscopy `phase_length` warm-up/history channel
has its own declared sampler and a capability error when no
positive sampled length exists; it has NO fallback one.
Its original availability/error rule remains in the comparison.
The displayed $2R_D$ bound applies only to actual spatial
alive-box pairs, not to an anisotropic phase-space distance,
an uncapped historical/post-jitter pair or an unbounded box.

For squared assembled or alive-normalized errors the same comparison
gives

$$
E|A^{\chi_N}-a^{*,\chi_0}|^2
\le2E|A^{\chi_0}-a^{*,\chi_0}|^2+
2E\min\{2B,L_\chi|\chi_N-\chi_0|\}^2.
\tag{NSC.15}
$$

The actual random calibration may still be consumed by the
complete-update native-time theorem when it is a measurable
function of that original finite future instrument. A calibration
using earlier shared history retains that earlier dependency.
:::

:::{prf:proof}
Compare the bounded observations pointwise on their shared actual
record. The product expectation and product of means each cost
$B_G E|\Delta H|+B_HE|\Delta G|$, proving (NSC.14).
The squared two-term triangle proves (NSC.15).
For $c_a=(F_a/|F|)e^{i\kappa z_a}$,
$|\partial_\kappa c|\le|z|$ and
$\partial_\kappa P=(\partial_\kappa c)c^\dagger+
c(\partial_\kappa c)^\dagger$, giving the stated projector budget.
In the actual canonical current-donor branch, each consumed alive
query and its eligible companion lie in $D$, so their distance
is at most $2R_D$. The original midpoint median or alive mean
has that same bound when nonempty; its original empty or zero
fallback is one in those named Python channels. Apply the literal phase formula with its
recorded denominator. No law-concentration hypothesis is added.
:::

(sec-nsc-assembled)=
## 5. Actual assembled regional observations and alive normalization

:::{prf:definition} Exact assembled regional means
:label: def-nsc-assembled-budget

Let a bounded spatial function $f_H$ have compact support $K_H\subset K$.
Absorb its bound into $B_H,L_{F,H},C_H$ and assume these are uniform
on its support. Write $|K_H|$ for its Lebesgue volume and

$$
A_{H,N}=\frac1N\sum_i f_H(Y_i)H_N(i),\qquad
a_{H,N}^*=\int f_H(x)\rho_{\eta_*}(x)
                              \Phi_{H,N}(\eta_*,x)\,dx .
$$

Its exact phase functional still keeps every finite-$N$ geometric
scale. A primitive Lipschitz coefficient is

$$
C_H^{\rm asm}=|K_H|
  [h_\tau C_H+B_HL_{\tau,1}],\qquad
d_H^{\rm asm}=|K_H|h_\tau L_{F,H}S_H\sqrt{C_{F,2}/N}.
\tag{NSC.16}
$$

Here the notation permits $f_H$ to have already been absorbed
into the observation; no spatial differentiability of $f_H$ is
needed. For an alive-only numerator require the actual
$\mathbf1_D(Y_i)$ in $f_HH_N$, hence
$|a_{H,N}^*|\le B_Ha_*$
with $a_*=\int_D\rho_{\eta_*}(x)\,dx\ge a_{\rm reset}$.
:::

:::{prf:theorem} Quadratic assembled concentration with actual same-record force
:label: thm-nsc-assembled-l2

Set

$$
\begin{gathered}
\Gamma_{N,R}=N^{-1}+h_\tau v_d(2R)/N+
                                      \epsilon_{\rm geom}(N,R),\\
\mathcal E_{H,N}=
6B_Hd_H^{\rm asm}
 +192B_H^2\Gamma_{N,R}
 +3(C_H^{\rm asm})^2\mathcal R_{A,N}
 +4B_H^2\varepsilon_N .
\end{gathered}
\tag{NSC.17}
$$

Then the actual QSD surviving record satisfies

$$
E|A_{H,N}-a_{H,N}^*|^2\le\mathcal E_{H,N}.
\tag{NSC.18}
$$

The stationary Doob record satisfies the same inequality with
$m_N^{-1}\mathcal E_{H,N}$.
In particular for two such observations,

$$
|\operatorname{Cov}(A_{H,N},A_{G,N})|
\le\sqrt{\mathcal E_{H,N}\mathcal E_{G,N}}
\tag{NSC.19}
$$

under the QSD record, and the same bound times $m_N^{-1}$
under the stationary Doob record.
All centers are deterministically reset bounded, so no
unknown-law moment core or mixed-phase covariance is dropped.
The bound applies whether or not the spatial supports overlap;
separated supports exclude its close-pair cost but are not needed
for the displayed larger certificate.
:::

:::{prf:proof}
First replace original dense forces by their own common field on
the actual record. Integrate the local force bound over its root
query with the original density, at most $h_\tau$, to obtain
$E|A_{H,N}-A_{H,N}^{\rm pop}|\le d_H^{\rm asm}$.
Both averages have absolute bound $B_H$, so their squared
discrepancy is at most $2B_Hd_H^{\rm asm}$ in expectation.

For the common-field average freeze the complete original
preparation. The marked two-root comparison applies just as
the conditional geometric variance proof (NGA.4), retaining
the same source residuals, bulk field and geometric scales.
Split diagonal pairs, close root positions and separated
determining windows. Their costs are respectively
$O(B_H^2/N)$, $O(B_H^2h_\tau v_d(2R)/N)$ and
$O(B_H^2\epsilon_{\rm geom})$.
The same original one-root posterior averaging identifies the
conditional phase mean
$\int f_H(x)\rho_{\eta_N}(x)\Phi_{H,N}(\eta_N,x)\,dx$.
The explicit larger constant $64$ of (NGA.4) therefore bounds
the conditional squared discrepancy by
$64B_H^2\Gamma_{N,R}$.
Its diagonal and removed-root terms are contained in the
already larger (NSC.4); no independent finite output array
is inserted.

The phase functional differs from $a_{H,N}^*$ by at most
$C_H^{\rm asm}W_{2,A}(\eta_N,\eta_*)$.
Indeed the conditional marked mean costs $C_H$ and its
density at most $h_\tau$; changing the density costs
$L_{\tau,1}W_{2,A}$ times $B_H$. Integrate over $K_H$.
The squared three-term triangle now gives the first three
terms of (NSC.17). Passing the complete raw record to its
actual survivor condition costs at most
$4B_H^2\varepsilon_N$ for this bounded squared discrepancy.
The terminal Doob density is at most $m_N^{-1}$.
Finally each variance is bounded by its displayed squared
distance from the deterministic phase mean, and
Cauchy--Schwarz proves (NSC.19).
:::

:::{prf:theorem} Exact alive-normalized regional covariance
:label: thm-nsc-alive-normalization

Let $P_N=M_N/N$ be the ACTUAL terminal alive fraction.
On the QSD record $M_N\ge1$; retain the original ratios
$B_{H,N}=A_{H,N}/P_N$ of alive-only numerators, and their
phase ratios $b_{H,N}^*=a_{H,N}^*/a_*$.
No denominator is clipped. Define

$$
L_{\rm alive}=\frac{2\sqrt d}{\tau\sqrt{2\pi}},\qquad
\mathcal E_{P,N}=\frac1{2N}
                    +2L_{\rm alive}^2\mathcal R_{A,N}
                    +\varepsilon_N,\qquad
\delta_{{\rm alive},N}=e^{-a_{\rm reset}N/8}.
\tag{NSC.20}
$$

Then

$$
E|B_{H,N}-b_{H,N}^*|^2
\le\mathcal E_{H,N}^{\rm alive}
:=\frac8{a_{\rm reset}^2}
 [\mathcal E_{H,N}+B_H^2\mathcal E_{P,N}]
 +\frac{4B_H^2\delta_{{\rm alive},N}}{1-\varepsilon_N}.
\tag{NSC.21}
$$

Thus the actual alive-normalized QSD covariance is at most
$\sqrt{\mathcal E_{H,N}^{\rm alive}\mathcal E_{G,N}^{\rm alive}}$;
the stationary Doob covariance is at most $m_N^{-1}$
times that number. Under the uniform budgets and (NSC.10),
these covariance budgets have the same algebraic orders
as (NSC.11), with primitive $a_{\rm reset}^{-1}$ constants.
:::

:::{prf:proof}
Conditional on the full preparation, terminal positions are
the actual independent Gaussians and their box success
probabilities are at least $a_{\rm reset}$. The Bernoulli
Chernoff calculation therefore gives
$P(P_N<a_{\rm reset}/2)\le e^{-a_{\rm reset}N/8}$.
The QSD survivor conditioning costs the factor
$(1-\varepsilon_N)^{-1}$, since its actual one-step
survival probability is at least $1-\varepsilon_N$.
This retains every status and every dead coordinate.

The conditional Bernoulli variance of $P_N$ is at most $1/(4N)$.
Its conditional mean is the empirical original Gaussian box
landing probability. That function of $m$ has gradient norm
at most $L_{\rm alive}$: differentiate the original product
of Gaussian interval probabilities and bound each derivative
by $2/(\tau\sqrt{2\pi})$.
Its phase value is $a_*$. A squared two-term triangle and
(NQG.10) give the first two terms in $\mathcal E_{P,N}$.
The bounded survivor selection costs at most $\varepsilon_N$.

On $P_N\ge a_{\rm reset}/2$, the literal ratio algebra and
$|a_{H,N}^*|\le B_Ha_*$ give

$$
|A_{H,N}/P_N-a_{H,N}^*/a_*|
\le\frac2{a_{\rm reset}}
       [|A_{H,N}-a_{H,N}^*|+B_H|P_N-a_*|].
$$

Its square is bounded by the first term of (NSC.21).
On the remaining event both actual and phase ratios have
absolute bound $B_H$, so their squared discrepancy is at
most $4B_H^2$. This proves the rare-event term and retains
the actual random denominator. The Doob squared-moment
comparison uses its bounded terminal density; variance and
covariance follow by Cauchy--Schwarz.
:::

(sec-nsc-native-time-operator)=
## 6. Full-state $L^2$ blocks and actual recorded-time covariance

:::{prf:theorem} Native stationary $L^2$ block contraction
:label: thm-nsc-native-l2-block

For $N\ge\widehat N_*$ put $P=P_N^e$ and
$T_N=\widehat T_N$, $\Delta_N=2\widehat b_N/(1-\widehat b_N)$.
On the ACTUAL full physical-state Hilbert space $L^2(\pi_N)$,
including dead coordinates and statuses,

$$
\|P^{T_N}\|_{L^2_0(\pi_N)\to L^2_0(\pi_N)}
\le\sqrt{\Delta_N},\qquad
\|P^g\|_{L^2_0\to L^2_0}
\le\Delta_N^{\lfloor g/T_N\rfloor/2}.
\tag{NSC.22}
$$

Consequently

$$
r_{\rm spec}(P|L^2_0(\pi_N))
\le e^{-\kappa_N^{\rm ent}/2}.
\tag{NSC.23}
$$

The self-adjoint block symmetrization has the verified coercivity

$$
\left\langle f,
\left[I-\frac{P^{T_N}+(P^*)^{T_N}}2\right]f\right\rangle_{\pi_N}
\ge(1-\sqrt{\Delta_N})\|f\|_2^2,
\qquad f\in L^2_0(\pi_N).
\tag{NSC.24}
$$

For the finite permitted $N<N_{\rm rate}$ the primitive common-part
factor $1-a_{F,N}$ of Chapter NJE gives the corresponding one-update
$L^2_0$ norm at most $\sqrt{1-a_{F,N}}$.
Throughout the admitted population family the native asymptotic
$L^2$ spectral-radius exponent is therefore at least
$\kappa_{\rm all}/2>0$ per update, or
$\kappa_{\rm all}/(2t_*h)$ in its native physical-time units.
:::

:::{prf:proof}
Start with a bounded real $\pi_N$-centered $f$. For sufficiently
small real $\epsilon$, $(1+\epsilon f)\pi_N$ is a probability.
Its image by $P^{T_N}$ has density
$1+\epsilon(P^*)^{T_N}f$ relative to $\pi_N$.
The adjoint is the original stationary reverse conditional
expectation, hence is Markov and preserves boundedness.
Apply the PROVED block KL inequality (NJE.16) to these two laws.
Taylor's formula for $(1+u)\log(1+u)$, with its bounded remainder,
gives after division by $\epsilon^2/2$ and passage to zero

$$
\|(P^*)^{T_N}f\|_2^2\le\Delta_N\|f\|_2^2.
$$

Bounded centered functions are dense in $L^2_0$ by truncation
and subtraction of their mean. Both stationary Markov operators
are $L^2$ contractions by conditional Jensen.
The displayed inequality extends by density. Hilbert adjoint
norm equality on the invariant centered subspace proves the
first assertion for $P$, rather than silently exchanging
a density operator and an observable operator.
Complex functions follow by their real and imaginary parts.

Take complete blocks and contract each remaining update by
Jensen to obtain (NSC.22). The spectral-radius formula gives
(NSC.23). Cauchy--Schwarz in its real part gives (NSC.24).
For each finite population apply the same density-perturbation
argument to its primitive one-update KL factor $1-a_{F,N}$.
The finite primitive minimum defining $\kappa_{\rm all}$
then supplies the stated family exponent.
This is a stationary stochastic block estimate; it does
not identify a self-adjoint physical gauge Hamiltonian.
Writing (NSC.22) as a pure exponential permits the explicit
prefactor $\Delta_N^{-1/2}$:
$\|P^g\|_{L^2_0}\le
\Delta_N^{-1/2}e^{-\kappa_N^{\rm ent}g/2}$.
That prefactor is not population-uniform; the displayed
complete-block bound has prefactor one.
:::

:::{prf:theorem} Native complete-record time covariance
:label: thm-nsc-recorded-time-covariance

Under the stationary Doob law, let $F$ consume a finite block of
original records ending at physical state $S_a$, and let $G$
consume $S_b$ and a finite block of subsequent original transition
records, with $g=b-a\ge0$. The future readout uses its actual
conditional transition instrument from $S_b$ and the fixed
configured parameters. For square-integrable $F,G$,

$$
|\operatorname{Cov}_{\pi_N}^e(F,G)|
\le\sqrt{\operatorname{Var}F\,\operatorname{Var}G}\,
                  \Delta_N^{\lfloor g/T_N\rfloor/2}.
\tag{NSC.25}
$$

For bounded $|F|\le B_F$, $|G|\le B_G$ the original
full-state Dobrushin block yields the stronger bounded-test bound

$$
|\operatorname{Cov}_{\pi_N}^e(F,G)|
\le B_FB_G\Delta_N^{\lfloor g/T_N\rfloor}.
\tag{NSC.26}
$$

These statements include all local native color, metric, loop and
action readouts whose square integrability or displayed bounded
observation budget is actually available. An unbounded action is
not assigned a finite variance without its separate moment proof.
No full output rows are assumed independent.

For two single-update records ending at updates $n,n+k$,
use $g=k-1$. With existing recording stride $m$, records
$j$ strides apart have $k=mj$ and the same $g=mj-1$.
The physical clock is $t_*h$ per actual update; spatial calibration
and the configured comparison cone do not change this gap.
:::

:::{prf:proof}
Let $f(S_a)=E[F\mid S_a]$ and $g_0(S_b)=E[G\mid S_b]$.
Conditional Markov evolution of the actual complete future
instrument gives $E[G\mid\mathcal F_a]=P^{b-a}g_0(S_a)$.
Subtract both means. Conditional Jensen bounds their centered
$L^2$ norms by the square roots of the original record variances.
Cauchy--Schwarz and (NSC.22) prove (NSC.25).

For the bounded version the common-mass row coupling gives
$\operatorname{osc}(P^{b-a}g_0)
\le2B_G\Delta_N^{\lfloor g/T_N\rfloor}$.
If real random variables have ranges of lengths $A,B$,
Cauchy--Schwarz and their midpoint variance bounds give
$|\operatorname{Cov}|\le AB/4$. Apply this to $F$ and
its future conditional expectation, proving (NSC.26).
The recorded-time gap follows from the start state of the
later original transition; its own sampled record is not
declared a function of the terminal state alone.
:::

:::{prf:corollary} Complete survivor-history covariance and all-horizon comparison
:label: cor-nsc-survivor-time-covariance

For a native horizon $L$ containing both bounded record blocks,
start the original killed algorithm at $\nu_N$ and condition
ONCE on survival through $L$. Then

$$
|\operatorname{Cov}_{\mathbb S_{N,L}}(F,G)|
\le B_FB_G\Delta_N^{\lfloor g/T_N\rfloor}
                         +6B_FB_Gd_N .
\tag{NSC.27}
$$

For the unconditioned killed path with any bounded cemetery
extensions, add $6B_FB_G(1-\alpha_N^L)
\le6B_FB_GL\varepsilon_N$.
The complete survivor/Doob density remains a bounded terminal
tilt even after conditioning any common recorded query event:
the conditional TV is at most $d_N$ for events of positive
probability, and for regular conditional root queries almost
everywhere in their common support.
This transfers a covariance bound already proved IN THAT
conditional ensemble; it does not turn (NSC.26) into a
time-mixing assertion after arbitrary future postselection.
:::

:::{prf:proof}
The exact complete-history density is
$e_N(S_L)/\nu_N(e_N)$ by (NUE.25).
It lies between the two normalized bounds from $m_N\le e_N\le1$,
so its TV is at most $d_N$, independently of $L$.
The bounded covariance perturbation costs $6B_FB_Gd_N$.
The unconditioned killed history is the survivor mixture with
complement probability $1-\alpha_N^L$, giving its correction.
On a common conditioning event the same terminal function is
renormalized by its conditional mean, still between $m_N$ and one.
Disintegration gives the conditional statement without dividing
an unconditional TV estimate by an event probability.
:::

:::{prf:corollary} Square-integrable survivor-record covariance
:label: cor-nsc-survivor-l2-covariance

For the same complete native horizon and future-instrument scope,
assume only $F,G\in L^2(\mathbb S_{N,L})$. Then

$$
|\operatorname{Cov}_{\mathbb S_{N,L}}(F,G)|
\le
\sqrt{\operatorname{Var}_{\mathbb S}F\,
                       \operatorname{Var}_{\mathbb S}G}
\left[m_N^{-1}\Delta_N^{\lfloor g/T_N\rfloor/2}
                                      +d_N+d_N^2\right].
\tag{NSC.28}
$$

The bound is horizon independent apart from its actual gap.
No unbounded-observation transfer to an unconditioned killed
path is inferred from total variation alone.
:::

:::{prf:proof}
Let $R=e_N(S_L)/\nu_N(e_N)$ be the complete Doob density
relative to $\mathbb S=\mathbb S_{N,L}$.
It satisfies $R\le m_N^{-1}$ and $|R-1|\le d_N$.
Thus $\operatorname{Var}_{\mathbb P^e}F
\le m_N^{-1}\operatorname{Var}_{\mathbb S}F$ by evaluating
the squared deviation at the survivor mean, and similarly
for $G$. Both observations are $L^2$ under the Doob law.
Set $f=F-E_{\mathbb S}F$, $g_0=G-E_{\mathbb S}G$.
The exact covariance difference is

$$
\operatorname{Cov}_{\mathbb P^e}(F,G)
 -\operatorname{Cov}_{\mathbb S}(F,G)
=E_{\mathbb S}[(R-1)f g_0]
 -E_{\mathbb S}[(R-1)f]\,E_{\mathbb S}[(R-1)g_0].
$$

Cauchy--Schwarz bounds it by
$(d_N+d_N^2)
\sqrt{\operatorname{Var}_{\mathbb S}F\operatorname{Var}_{\mathbb S}G}$.
Apply (NSC.25) and the two variance comparisons.
:::

(sec-nsc-scope)=
## 7. Native regimes and remaining physical identification

:::{prf:remark} Exact statistical and physical scope
:label: rem-nsc-scope

The existing positive primitive witness of Chapters PC, NQC and
NUE has reset $a_x=0$, positive $q,s,\sigma_J,\rho,\nu$,
positive $\kappa_A$ and $r<1$, and therefore evaluates every
constant above to a finite number. Its $d=3,p=4$ preparation
rate supplies the explicitly stated spatial exponent $4/35$.
The unchanged configured cap is fixed with $N$ and all original
Gaussian tails remain. Failed inherited parameter tests give
no assigned deterministic phase or uniform time/spatial rate;
they retain their finite-QSD or qualitative geometry results.
No landscape or normalization branch is silently relabeled as
this quadratic count phase.

At one record, separated spatial-root covariance is controlled
by the native quadratic phase concentration, original marked
window/retessellation coupling and original dense-force variance.
At separated times, the complete physical-state transition
controls finite future record instruments. A calibration that
consumes arbitrary earlier history, a growing cached observable,
or an absolute genealogy archive is not a fresh instrument from
$S_b$ alone. Its history dependence must be retained, and the
time bound does not erase it. Fixed configured calibrations,
current-stage readouts and declared finite future windows meet
the stated instrument scope.

The full-state $L^2$ radius and block symmetrization are native
stochastic operators with a proved invariant law. They do not
supply one-update reversible coercivity, a population-uniform
per-update prefactor, spacelike quantum commutators, reflection
positivity, or the physical non-Abelian Hamiltonian identification.
The latter remains the same native gauge-history reconstruction
obligation identified in the physical-transfer chapters.
:::
