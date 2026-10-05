# Central discovery, protected phase production and the discounted clock

(sec-kupc-record)=
## 1. Retained objects and the exact endpoint

:::{prf:definition} Actual central count and the original marked kernel
:label: def-kupc-record

Use the complete native Rastrigin record of
{prf:ref}`def-kurc-record` and {prf:ref}`def-kuns-native-record`.
Both configured viscous normalizations, sampled reward/diversity
measurements, their empirical normalizers, both donor roles,
acceptance, mandatory revival, component-Haar rotations, unbounded
clone and thermostat innovations, final position diffusion, the
radial cap and terminal marks remain unchanged.

Write $P_N$ for its complete physical transition, $Q_N$ for the
transition killed only when every output row is dead, and
$\nu_NQ_N=\alpha_N\nu_N$ for the compatible finite-$N$ QSD.
Set

$$
C=[-R_C,R_C]^3,\quad R_C=1/16,\qquad
K_C(S)=\sum_i\mathbf1_{\{x_i\in C\}}.
\tag{KUPC.1}
$$

Since $C\Subset D$, every counted output is actually alive.
Let $\tau_C$ be first discovery of $K_C\ge1$ and $\tau_\dagger$
the actual all-dead time. A discovery set is distinguished from
a population phase and from the singleton set in (KUER.12).
No claim that one discovered row establishes a population phase
is part of this definition.

The actual prepared position formula is

$$
X_i=\mu_i+A_i\sigma_JZ_i^J,\quad \mu_i\in\overline D,
\qquad x_i^+=g(X_i)+bU_i+\tau Z_i,
\quad |U_i|\le V_c=4,
\tag{KUPC.2}
$$

where $g(x)=(1-2\eta)x-20\pi\eta\sin(2\pi x)$ coordinatewise,
$b=.02(1+e^{-.04})$, $\eta=.02b$, and
$\tau^2=.0004+.0004(1-e^{-.08})/2$.
The Gaussian $Z_i$ are independent after complete preparation;
$U$ depends on every realized jitter and the actual common Haar
law. Incoming dead positions need not be bounded: their actual
mandatory revival provides the bounded frozen sources in (KUPC.2).
:::

(sec-kupc-global-discovery)=
## 2. A closed central-discovery clock at large population

:::{prf:theorem} Primitive global one-row discovery and its complete count
:label: thm-kupc-global-central-discovery

For any $J>0$, define

$$
p_J=2\Phi(J/\sigma_J)-1,\qquad
B_J=g(2+J)+bV_c,
$$
$$
p_C(J)=p_J^3(2R_C)^3(2\pi\tau^2)^{-3/2}
 \exp\!\left[-\frac{3(B_J+R_C)^2}{2\tau^2}\right].
\tag{KUPC.3}
$$

For every nonextinct entering state, every $N\ge1$, and every
measurement/gate/Haar law prescribed by the original algorithm,

$$
\Pr_S(K_C^+\ge1)\ge\lambda_N(J):=1-[1-p_C(J)]^N.
\tag{KUPC.4}
$$

One may use the fully certified, conservative coefficient

$$
J=1/4000,\qquad p_C(J)>p_*:=e^{-18000}.
\tag{KUPC.5}
$$

This is a single-row tagged-noise construction. No event requiring
all $N$ rows to have bounded noise is imposed on the actual chain.
:::

:::{prf:proof}
The native $g$ is globally odd and increasing by (KURC.4).
Freeze the entire source/gate/component-Haar information before
the independent own-row clone jitters. On the tagged event
$|\sigma_J Z_{i,k}^J|\le J$ for $k=1,2,3$, row $i$ satisfies
$|X_{i,k}|\le2+J$, even when $A_i=0$ and the latent jitter is
unused. Therefore every coordinate of its actual position mean
has absolute value at most $B_J$, regardless of the other rows'
unbounded jitters. The Gaussian density throughout $C$ is at least
$(2\pi\tau^2)^{-3/2}
\exp[-3(B_J+R_C)^2/(2\tau^2)]$.

Conditional on all jitters, the actual membership indicators are
independent Bernoulli trials. Couple them with independent uniforms.
Each dominates its own tagged-jitter indicator times a uniform
indicator with the displayed position probability. These lower
indicators are independent conditional on the pre-jitter information,
and each has probability $p_C(J)$. Their sum is binomial; mixing
over the original source and Haar law preserves (KUPC.4). This
is the existing regional discovery method, applied to every
actual source phase through the primitive global endpoint profile.

For the conservative numerical certificate use
$e^{-.04}\in[.96,.9608]$ and
$1-e^{-.08}\in[.0768,.08]$, obtained from the first two
exponential Taylor inequalities. Thus

$$
.0392\le b\le.039216,\quad \eta\ge.000784,
\quad .00041536\le\tau^2\le.000416.
$$

At $J=1/4000$, $\sin(2\pi J)>0$, so
$g(2+J)\le(1-.001568)(2.00025)=1.997113608$.
Hence $B_J+R_C<2.217$ and

$$
\frac{3(B_J+R_C)^2}{2\tau^2}
<\frac{3(2.217)^2}{2(.00041536)}<17800.
$$

Also $p_J\ge2(J/\sigma_J)\phi(J/\sigma_J)>.00195>e^{-10}$.
The first inequality integrates the minimum Gaussian density over
the tagged interval; $\phi(.0025)>.39$ follows already from
$\pi<22/7$ and $e^{-x}\ge1-x$.
For the last comparison the fifth positive Taylor term gives
$e^{10}>10^5/5!>1/.00195$. Finally
$2R_C/\sqrt{2\pi\tau^2}>1$, so the three-dimensional volume
prefactor is greater than one. Together these inequalities give
$p_C(J)>e^{-17830}>e^{-18000}$. All calculations use the
original force and stage amplitudes. $\square$
:::

:::{prf:theorem} Actual discounted discovery before physical killing
:label: thm-kupc-discounted-discovery

Use any one of the certified global coefficients from (KURC.16),
$a=2\cdot10^{-6}$ in count mode or $a=7\cdot10^{-11}$ in
row mode. Set

$$
\delta_N=e^{-aN},\qquad
\lambda_N=1-(1-p_*)^N.
\tag{KUPC.6}
$$

For every entering survivor state outside the discovery set,

$$
\Pr_S(\tau_C<\tau_\dagger)
\ge\frac{\lambda_N}{\lambda_N+\delta_N}.
\tag{KUPC.7}
$$

The compatible actual QSD satisfies $\alpha_N\ge1-\delta_N$.
Whenever $\delta_N<\lambda_N$,

$$
\mathbb E_S[\alpha_N^{-\tau_C};\tau_C<\tau_\dagger]
\le\frac{\lambda_N}{\lambda_N-\delta_N}.
\tag{KUPC.8}
$$

In particular, for

$$
N\ge N_0(a):=\left\lceil\frac{18001}{a}\right\rceil,
$$
$$
\Pr_S(\tau_C<\tau_\dagger)\ge2/3,\qquad
\mathbb E_S[\alpha_N^{-\tau_C};\tau_C<\tau_\dagger]\le2.
\tag{KUPC.9}
$$

The explicit thresholds are
$N_0=9000500000$ in count mode and
$N_0=257157142857143$ in row mode. The row singleton may use
the stronger count survival coefficient, but this does not alter
the stated conservative row threshold.
:::

:::{prf:proof}
At every history before discovery or killing, the next discovery
probability is at least $\lambda_N$ by (KUPC.4)--(KUPC.5), and
the next actual extinction probability is at most $\delta_N$ by
(KURC.11). The events are disjoint. Their union time
$\tau=\tau_C\wedge\tau_\dagger$ is stochastically bounded
above by a geometric variable of parameter $\lambda_N$; in
particular $\Pr(\tau>n)\le(1-\lambda_N)^n$ and one of the
two events occurs almost surely.

Sum the competing one-step probabilities over histories. At each
one the death probability is at most $(\delta_N/\lambda_N)$
times its discovery probability. Thus
$\Pr(\tau_\dagger<\tau_C)
\le(\delta_N/\lambda_N)\Pr(\tau_C<\tau_\dagger)$,
which proves (KUPC.7).

The spectral lower bound is obtained by integrating the proved
*uniform maximum* extinction bound against the actual QSD:
$\alpha_N=\nu_NQ_N1\ge1-\delta_N$.
It is not an inference from the minimum killing rate of a central
configuration or from a closed-phase assumption.
For $\alpha_N>1-\lambda_N$, the increasing function
$\alpha_N^{-n}$ and geometric domination give

$$
\mathbb E[\alpha_N^{-\tau_C};\tau_C<\tau_\dagger]
\le\mathbb E\alpha_N^{-\tau}
\le\frac{\lambda_N}{\alpha_N-1+\lambda_N}
\le\frac{\lambda_N}{\lambda_N-\delta_N}.
$$

For $N\ge N_0(a)$, $\delta_N\le e^{-18001}<p_*/2$,
because $e>2$. Also $\lambda_N\ge p_*$. Substitute these
two inequalities in (KUPC.7)--(KUPC.8). $\square$
:::

:::{prf:remark} The strength and domain of the closed clock
:label: rem-kupc-clock-domain

The density formula (KUPC.3) at $J=1/4000$ has diagnostic
$-\log p_C\simeq17755.6003994$. Its conservative coefficient
and thresholds are deliberately rounded outward. The expected
discovery-or-death time is at most $1/\lambda_N$ updates;
one may use $\lambda_N\ge Np_* /(1+Np_*)$ to display the
population improvement. This is not a practical mixing-time
estimate: the fixed Gaussian coefficient is extremely small.

The theorem closes a discounted *discovery* clock for all sufficiently
large populations with a numerical bound independent of $N$.
It does not identify $\tau_C$ with singleton return, population-phase
establishment or full-state renewal. Those clocks include further
conditional information. Nor is (KUPC.8) asserted from these
coefficients for $N<N_0$: its sufficient discount comparison must
then be improved or supplied by the actual finite-$N$ certificate.
All actual between-phase paths remain permitted.
:::

(sec-kupc-seed-producer)=
## 3. A native seed producer stronger than an expected-count statement

:::{prf:proposition} Complete native lower offspring law for the exact seed
:label: prop-kupc-seed-pgf

Take the actual all-alive two-site input of
{prf:ref}`prop-kuns-exact-single-seed`: $N-1$ positions $e_1$,
one position zero, and every velocity zero. Let
$w=e^{-1/18}$, $p_D=p_C=w/(N-2+w)$, and let the unrestricted
central landing coefficients be the actual $s_0,s_J$ of (KUNS.4).
For $N\ge3$ the complete output central count stochastically
dominates independent variables

$$
Y_C^+\ \succeq\ {
m Bern}(s_0)
 +{
m Bin}\bigl(N-1,p_C(1-p_D)s_J\bigr).
\tag{KUPC.10}
$$

For $N=2$ replace the binomial term by ${\rm Bern}(s_J)$.
Consequently for every $N\ge2$ and every $0\le q\le1$,

$$
\mathbb E q^{Y_C^+}
\le[1-.993(1-q)]e^{-.09992(1-q)}.
\tag{KUPC.11}
$$

At $q=1/2$ this is less than $.479<q$, and
$\Pr(Y_C^+=0)<.006335$.
:::

:::{prf:proof}
For each resident use the favorable event that its original
measurement is near, its separate cloning proposal is the central
seed, and its own copied-central kinetic position lands in $C$.
The existing finite gate certificate proves acceptance one on
this favorable measurement/proposal event for *every* assignment
of all other measured marks. Its probability is
$p_C(1-p_D)s_J$. The original measurement, proposal and own-row
innovations are independent across recipients. Common empirical
normalizers do not enter the lower indicator after the gate-one
certificate has been applied.

Every original velocity is zero, so every actual component-Haar
collision velocity is zero and $W_Xv^{\rm col}=0$ in both modes,
even though $W_X$ depends on every jitter. Each successful central
copy therefore lands at $g(.1Z_i^J)+\tau Z_i$ independently of
the other recipients. The central seed always has the higher
reward and the largest realized diversity mark, so it never
accepts a resident donor. It persists without jitter and has its
independent ${\rm Bern}(s_0)$ core landing. These lower indicators
establish (KUPC.10); additional arrivals only increase $Y_C^+$.
For $N=2$ the actual far-resident gate is also one by the existing
finite certificate, giving the stated replacement.

Use $s_0>.993$, $s_J>.2$, and
$p_C(N-1)(1-p_D)\ge2w/(1+w)^2>.4996$ for $N\ge3$.
The binomial PGF is at most
$\exp[-.09992(1-q)]$; the $N=2$ incoming probability exceeds
$.2>.09992$, so the same bound holds. The seed PGF is at most
$1-.993(1-q)$. At $q=1/2$ the resulting number is
$.5035e^{-.04996}=.47896317338\ldots$; at zero it is
$.007e^{-.09992}=.00633436866\ldots$.
The first few alternating exponential terms certify the quoted
strict rational upper bounds. B2, cap and actual terminal marking
are retained; they do not change the counted positions, and
$C\Subset D$ makes every counted row alive. $\square$
:::

:::{prf:remark} Why this is a producer and not an establishment theorem
:label: rem-kupc-seed-scope

(KUPC.11) supplies an actual negative PGF increment for the native
exact seed, in addition to the positive expected count in (KUNS.5).
It uses both reward and diversity and every measured outcome.
Its input has zero velocities and exactly two source positions.
The next source/velocity law is dispersed by the original jitter,
thermostat and position diffusion. Its PGF inequality has not
been established on that whole reached class, so it cannot be
iterated as an autonomous offspring law.
:::

(sec-kupc-protected-principal-rate)=
## 4. A protected-phase principal-rate certificate with outward flux retained

:::{prf:definition} Exact count transform and phase leakage
:label: def-kupc-protected-producer

Let $G_N$ be a declared measurable class of actual full survivor
states. It may constrain alive phase counts, centroid, normalized
velocity energy and averaged tails; it must include the reached
mixture/exterior outcomes charged by its definition. For $0<q<1$,
define

$$
f_N(S)=\mathbf1_{G_N}(S)[1-q^{K_C(S)}],
$$
$$
\mathcal B_N(S;q)=\mathbb E_S^{\rm prep}
 \prod_i[1-(1-q)p_i^C],\qquad
p_i^C=\prod_{r=1}^3\ell_{R_C,\tau}
 \bigl(g(X_{ir})+bU_{ir}\bigr),
\tag{KUPC.12}
$$
$$
\mathcal E_{G,N}(S;q)=
\mathbb E_S[(1-q^{K_C(S^+)})\mathbf1_{\{S^+\notin G_N\}}].
\tag{KUPC.13}
$$

The expectation uses the actual complete source/acceptance/Haar,
jitter, OU, both-kick, cap and terminal law. For a class involving
retained velocities, (KUPC.13) uses the full joint OU/final-position
integral; it is not computed from independent position marginals.
Mixture and exterior phase flux, centroid motion, source transfer
and velocity constraints enter this term with their exact signs
and conditional information.
:::

:::{prf:theorem} A verified phase producer bounds the actual principal killing rate
:label: thm-kupc-protected-principal-rate

Suppose an explicit $0\le\varepsilon_N<1$ has been proved to satisfy,
for every $S\in G_N$,

$$
\mathcal B_N(S;q)+\mathcal E_{G,N}(S;q)
\le q^{K_C(S)}+\varepsilon_N[1-q^{K_C(S)}].
\tag{KUPC.14}
$$

If the compatible actual QSD has $\nu_Nf_N>0$, then

$$
Q_Nf_N\ge(1-\varepsilon_N)f_N,
\qquad 1-\alpha_N\le\varepsilon_N.
\tag{KUPC.15}
$$

Thus a computed producer $\varepsilon_N\le C_\kappa e^{-NI_{\rm prot}}$
gives that same exponential upper bound on the actual principal
killing rate. A one-step minimum killing probability at a central
state is not the hypothesis of this theorem.
:::

:::{prf:proof}
After complete preparation, the central memberships are independent
Bernoulli trials, so (KUPC.12) is exactly $\mathbb E q^{K_C(S^+)}$.
An all-dead output has $K_C=0$ and contributes zero to $f_N$.
Therefore, for an entering state in $G_N$,

$$
Q_Nf_N(S)=1-\mathcal B_N(S;q)-\mathcal E_{G,N}(S;q).
$$

This is an identity for the original physical kernel; no extra
phase-boundary killing has been introduced. Equation (KUPC.14)
proves (KUPC.15) on $G_N$. Outside it $f_N=0$ and
$Q_Nf_N\ge0$, so the inequality is global. Integrate against
$\nu_N$: $\alpha_N\nu_Nf_N\ge(1-\varepsilon_N)\nu_Nf_N$.
Division by the assumed positive mass proves the spectral bound.
Nonempty open classes in the established full-support domain may
use that finite-$N$ support theorem to verify this positive-mass
condition. $\square$
:::

:::{prf:proposition} The narrow entering band is not a repeatedly protected phase
:label: prop-kupc-narrow-band-exit

For the source/velocity band of {prf:ref}`def-kuns-band-class`,
with $\varepsilon=.001$, its actual one-step probability of
returning to the same whole-population band satisfies

$$
Q_N(S,G_N)\le b_{\rm band}^N,
\qquad
b_{\rm band}=\frac{2|B(0,\varepsilon)|}
 {(2\pi\tau^2)^{3/2}}<6.3\cdot10^{-5}.
\tag{KUPC.16}
$$

The actual subkernel killed additionally on leaving this band
has spectral radius at most $b_{\rm band}^N$.
This artificial band-exit bound is not a bound on the physical
$\alpha_N$, which retains all those outgoing states.
:::

:::{prf:proof}
Every position in that entering band lies in one of two radius
$\varepsilon$ balls. Conditional on complete preparation, each
actual final position has Gaussian covariance $\tau^2I_3$ and
density at most $(2\pi\tau^2)^{-3/2}$. Its probability of landing
in their union is at most $b_{\rm band}$. These positions are
conditionally independent, so requiring all $N$ to land in the
union costs $b_{\rm band}^N$. Additional velocity, count and
alive requirements only decrease the probability. The supremum
row mass bounds the norm and spectral radius of the artificially
restricted subkernel. Using $|B(0,\varepsilon)|=4\pi\varepsilon^3/3$
and the actual $\tau^2$ verifies the numerical bound. $\square$
:::

:::{prf:remark} Exact next producer
:label: rem-kupc-next-producer

The existing signed Keystone pressure, regional curvature,
donor incoming excess, centering terms and kinetic covariance
ledger remain the permitted production method for (KUPC.14).
The count PGF in (KUPC.11) is a verified local input. Completing
the principal-rate route requires its extension to a quantitatively
declared reached full-state class, with the outward charge
(KUPC.13) and the actual velocity/centroid law. A joint weight
$0\le w_N(S)\le1$ may replace the indicator in $f_N$ when
its signed whole-map increment is computed; the same QSD
integration applies to a proved $Q_Nf_N\ge(1-\varepsilon_N)f_N$.
No entropy gap, phase absorption, constant fitness difference,
or desired-law contraction is supplied as an unproved premise.
:::

(sec-kupc-establishment-clock)=
## 5. The precise establishment comparison needed after discovery

:::{prf:theorem} Establishment blocks and the actual spectral discount
:label: thm-kupc-establishment-clock

Let $H_N$ be a declared established population phase. Suppose a
computed block length $b_N$ and probability $r_N>0$ satisfy,
after every allowed history outside $H_N$ and before physical
killing, probability at least $r_N$ of entering $H_N$ by the
end of the next block, before killing. This must be a bound
on the whole reached class, not only on the original exact seed.
Suppose a proved principal-rate producer gives
$\alpha_N\ge1-\varepsilon_N$. If $b_N\varepsilon_N<r_N$, then

$$
\mathbb E_S[\alpha_N^{-\tau_{H_N}};
 \tau_{H_N}<\tau_\dagger]
\le\frac{r_N}{r_N-b_N\varepsilon_N}.
\tag{KUPC.17}
$$

For a produced discovery/establishment chain one may set
$r_N=\lambda_N p_{e,N}$ when its conditional establishment
probability $p_{e,N}$ holds for every discovery history within
the charged block and every uncontrolled exit is retained.
If $p_{e,N}\ge c_e e^{-JN}$,
$b_N\le c_b e^{J_bN}$, and
$\varepsilon_N\le C_\kappa e^{-I_{\rm prot}N}$, then
$I_{\rm prot}>J+J_b$ suffices for (KUPC.17) to be at most two
for all sufficiently large populations. Polynomial establishment
budgets require only a positive $I_{\rm prot}$ in this comparison.
These are conditional consequences of the displayed producers.
:::

:::{prf:proof}
The number of blocks until establishment or killing is bounded
above by a geometric variable of parameter $r_N$. If $T$ is
that number, the actual first event occurs no later than $b_NT$.
The successful discounted expectation is therefore at most

$$
\mathbb E\alpha_N^{-b_NT}
=\frac{r_N}{\alpha_N^{b_N}-1+r_N}
\le\frac{r_N}{r_N-b_N\varepsilon_N},
$$

using $(1-\varepsilon_N)^{b_N}\ge1-b_N\varepsilon_N$.
The strict premise ensures convergence of the geometric series.
For the exponent comparison use $\lambda_N\ge p_*$ and
substitute the three producer bounds. Their ratio is at most
$(c_bC_\kappa/(c_ep_*))
e^{-(I_{\rm prot}-J-J_b)N}$, which tends to zero under the
stated strict exponent inequality. No all-row phase replacement
or arbitrary-swarm distance contraction appears in this argument.
$\square$
:::

:::{prf:remark} Evidence status of this phase-clock route
:label: rem-kupc-status

The global one-row discovery bound, the large-population
discounted discovery clock, the exact native seed PGF and the
narrow-band exit calculation are proved actual-kernel results.
The protected principal-rate inequality (KUPC.14) and the
history-uniform establishment block in (KUPC.17) are explicit
new producers still required for the established-phase route.
Existing one-step central killing and source-growth coefficients
do not imply either producer. In particular the all-row direct
transfer exponent from record17 is not substituted for the
one-row discovery-and-growth path.

Global eigenfunction oscillation additionally requires the
retained full-state comparison on the reached or returned set,
including surviving-label and velocity information. This note
does not claim that the closed discovery clock settles that
separate comparison or the remaining small populations.
:::

(sec-kupc-moving-joint)=
## 6. Moving orbital readouts and a computed joint-energy producer

The regional force profiles, signed source sums and orbital ledgers in
Sections2,8,14 and17 of the Keystone chapter remain the method used here.
A moving target changes the observable used to read the actual update;
it does not translate its force, reward, empirical normalizers or noise.
The following results supply new one-update inputs to the protected
producer in Section4. Its repeated-update hypothesis is kept explicit.

:::{prf:theorem} Native source growth with an arbitrary common orbital velocity
:label: thm-kupc-moving-native-growth

In the exact single-seed configuration of
{prf:ref}`prop-kuns-exact-single-seed`, replace every zero input velocity
by the same $v_*\in\mathbb R^3$, $|v_*|\le V=2$. Define the actual
one-update orbital target

$$
C_B^+=bv_*+[-1/16,1/16]^3.
\tag{KUPC.18}
$$

The exact source coefficient (KUNS.2), the source lower bound (KUNS.3)
and the seed PGF (KUPC.10)--(KUPC.11) are unchanged, with $Y_C^+$
replaced by $Y_{C_B^+}^+$. In particular

$$
\mathbb E Y_{C_B^+}^+>1.0929\qquad(N\ge2).
\tag{KUPC.19}
$$

For the narrow two-region class in
{prf:ref}`def-kuns-band-class`, replace its velocity restriction by

$$
|v_i-v_*|\le\varepsilon=.001\quad\hbox{for every row},
\qquad |v_*|\le V-\varepsilon.
\tag{KUPC.20}
$$

Then the same signed coefficients $\beta_+,\beta_-,\beta_{BB}$ in
(KUNS.8) apply and, for both actual viscous normalization tags,

$$
\mathbb E Y_{C_B^+}^+\ge1.0387K.
\tag{KUPC.21}
$$

Every row counted in (KUPC.18) is actually alive and lies in the
actual central Rastrigin barrier cell. No bound on a maximum of
Gaussian innovations is used.
:::

:::{prf:proof}
The squashed velocity feature is identical for all rows in the exact
configuration. Thus its pairwise feature differences vanish, exactly
as for zero common velocity. Rewards, both independently sampled
companion roles, every realized diversity mark, all empirical
normalizers and every gate in the proof of (KUNS.2) remain the same.
An actual component rotation acts on velocities relative to its
component mean. Those relative velocities are zero, so
$v_i^{\rm col}=v_*$ for every graph and every Haar draw. Both actual
first-kick matrices preserve constants, including after every own
jitter has been revealed. Consequently $U_i=v_*$ and

$$
x_i^+-bv_*=g(\mu_i+A_i\sigma_JZ_i^J)+\tau Z_i.
$$

The central seed is not copied and its relative landing law is
$\tau Z_i$. Each accepted central-source offspring has relative
landing law $g(.1Z_i^J)+\tau Z_i$. Their probabilities are exactly
$s_0,s_J$ in (KUNS.4). The same independent favorable near-measurement,
central-donor and own-innovation indicators give the PGF bound in
Section3. A resident-source arrival can only add a counted row. This
proves (KUPC.19) and the asserted PGF.

For (KUPC.20), compare each velocity feature to the common
$\operatorname{squash}(v_*)$. The squash is 1-Lipschitz, so every
within/cross feature interval in (KUNS.7) is unchanged: its proof
uses deviations of size $\varepsilon$, not the absolute bulk speed.
The reward factors are unchanged as well. Thus the entire signed
source/gate proof of {prf:ref}`thm-kuns-band-selection` applies.
In each actual collision component, write $v_i=v_*+e_i$. Its
mean deviation has norm at most $\varepsilon$ and
$|e_i-\bar e|\le2\varepsilon$. With restitution $.5$ and one
orthogonal matrix per component,

$$
|v_i^{\rm col}-v_*|
\le|\bar e|+.5|e_i-\bar e|\le2\varepsilon.
$$

Both first-kick matrices are stochastic at $t\nu=.006$, hence
$|U_i-v_*|\le2\varepsilon$. The target shift $bv_*$ therefore
removes the common motion and leaves precisely the eroded landing
profiles in (KUNS.13), with erosion $2b\varepsilon$. Their proved
lower bounds $.993$ and $.2$ imply the same signed sum (KUNS.14),
which proves (KUPC.21). This argument allows $U_i-v_*$ to depend
on the complete jitter array; only its proved pointwise bound is
used before integrating the own-jitter lower landing function.

Finally $|bv_{*,r}|\le2b<.078432$, so the target satisfies
$|x_r|<.140932$. The actual adjacent central barrier roots lie
outside $[-3/8,3/8]$, by the retained root brackets of
{ref}`sec-kul-regional-record`. The target is therefore inside their
cell and inside $D$. Both the second kick and the original cap are
executed, and neither changes this final-position observable.
$\square$
:::

:::{prf:lemma} Both force evaluations in the moving two-phase orbit ledger
:label: lem-kupc-moving-baoab-ledger

Freeze an actual prepared source-label array with $n_B$ zero sources
and $n_A$ sources at $e_1$, $n_A+n_B=N$, and take the common
input velocity $v_*$. The deterministic zero-innovation reference
is a declared orbit anchor, not the expectation of the noisy update.
Put $\ell=1-2\eta$ and define

$$
\begin{array}{c|cc}
&B&A\\ \hline
\bar X&0&e_1\\
\bar u&v_*&v_*-2te_1\\
\bar y&bv_*&\ell e_1+bv_*\\
\bar w&cv_*&c(v_*-2te_1).
\end{array}
\tag{KUPC.22}
$$

With $K_*=\exp[-\ell^2/(2\rho^2)]$, the complete second-kick
cross coefficients are

$$
\begin{aligned}
\kappa_B^{\rm cnt}&=(n_A/N)K_*,&
\kappa_A^{\rm cnt}&=(n_B/N)K_*,\\
\kappa_B^{\rm row}&=\frac{n_AK_*}{n_B-1+n_AK_*},&
\kappa_A^{\rm row}&=\frac{n_BK_*}{n_A-1+n_BK_*}.
\end{aligned}
\tag{KUPC.23}
$$

Only present labels are evaluated; a single present phase has
zero cross coefficient, and the configured singleton viscous term
is zero. For either normalization, with $a=t\nu$, the actual
reference second kick and cap are

$$
\begin{aligned}
\bar z_B&=cv_*-2ct a\kappa_Be_1+tF(bv_*),\\
\bar z_A&=c(v_*-2te_1)+2ct a\kappa_Ae_1
              +tF(\ell e_1+bv_*),\\
\bar v_a^+&=\mathcal C_V(\bar z_a),
\qquad \mathcal C_V(z)=\frac{Vz}{V+|z|}.
\end{aligned}
\tag{KUPC.24}
$$

For the actual noisy prepared array, let
$\delta X=X-\bar X$, $\delta v^{\rm col}=v^{\rm col}-v_*$,
and use the actual $W_X,W_y$ and the reference $W_{\bar y}$.
Its complete deviations satisfy the identities

$$
\begin{aligned}
\delta u&=W_X\delta v^{\rm col}+t[F(X)-F(\bar X)],\\
\delta w&=c\delta u+q\xi,\\
\delta y&=\delta X+b\delta u+tq\xi,\\
\delta x^+&=\delta y+s\chi,\\
\delta z&=W_y\delta w+(W_y-W_{\bar y})\bar w
                   +t[F(y)-F(\bar y)],\\
\delta v^+&=\mathcal C_V(\bar z+\delta z)-\mathcal C_V(\bar z).
\end{aligned}
\tag{KUPC.25}
$$

These identities retain the common OU innovation at its two
occurrences and the independent final position innovation.
:::

:::{prf:proof}
Both first matrices preserve $v_*$. The native force is zero at
zero and equals $-2e_1$ at $e_1$, which gives (KUPC.22).
The two middle-position anchors are separated by $\ell e_1$;
therefore every cross Gaussian weight is exactly $K_*$, while
within-phase weights are one. Summing the actual count or
nonself row weights gives (KUPC.23). Their velocity difference
is $-2ct e_1$, which yields (KUPC.24) after adding the original
force evaluated at each middle-position anchor. Subtracting
these reference stages from each actual BAOAB stage gives
(KUPC.25). In particular its second-kick graph is the graph at
$y$, before the independent final position noise, as prescribed
by the original integrator. No independent replacement of the
OU noise in that force/graph stage has been introduced.
$\square$
:::

:::{prf:remark} The phase velocities and forces are not identified
:label: rem-kupc-translation-force

The position map satisfies $g(x+e_1)=g(x)+\ell e_1$, whereas
$F(x+e_1)=F(x)-2e_1$. The complete second-force difference is

$$
F_1(\ell e_1+bv_*)-F_1(bv_*)
=-2\ell-20\pi\{
 \sin[2\pi(\ell+bv_{*,1})]-\sin(2\pi bv_{*,1})\}.
$$

It is retained in (KUPC.24), together with its count/row
cross-viscous work and both capped velocity anchors. Thus a
common bulk velocity permits a moving positional readout,
but does not identify different phase orbits.
:::

:::{prf:theorem} Actual central-source joint-energy output at all populations
:label: thm-kupc-central-joint-energy

Consider any nonextinct entering state for which every actual frozen
donor source is $\mu_i=0$. The actual clone indicators, terminal
input labels, empirical fitness normalizers, component graph and
Haar draws are unrestricted. Every stored incoming velocity retains
the original cap $|v_i|\le V=2$. Conditional on this complete
pre-jitter pattern, and therefore also after averaging it, define

$$
\mathcal H_N^+
=\frac1N\sum_i\left[
 A_H|x_i^+|^2+\frac12|v_i^+|^2\right],
\qquad A_H=1+20\pi^2.
\tag{KUPC.26}
$$

This is a majorant for the actual normalized physical energy
$N^{-1}\sum_i[U(x_i^+)+|v_i^+|^2/2]$.
Let $\sigma=\sigma_J=.1$, $\omega=2\pi$,
$A_g=20\pi\eta$, $a_J=e^{-\omega^2\sigma^2/2}$,
$b_J=e^{-2\omega^2\sigma^2}$ and

$$
k_g=\ell^2\sigma^2-2\ell A_g\omega\sigma^2a_J
                     +\frac{A_g^2}{2}(1-b_J).
\tag{KUPC.27}
$$

Use the proved Gaussian-row column bound $C_3^{(40)}<7188$ from
{prf:ref}`lem-kuk-gaussian-column` and its finite evaluation, and put

$$
E_{
m cnt}=V=2,
\qquad E_{
m row}=V(1-a+a\sqrt{7188})
=2(.994+.006\sqrt{7188}).
\tag{KUPC.28}
$$

For either normalization $m$ and every $N\ge1$,

$$
\mathbb E\mathcal H_N^+
\le A_H\{[\sqrt{3k_g}+bE_m]^2+3\tau^2\}+V^2/2.
\tag{KUPC.29}
$$

The primitive evaluations give

$$
\begin{aligned}
.00555616116647902&<k_g<.00555616116647903,\\
\mathbb E\mathcal H_{N,\rm cnt}^+&<10.792376,\\
\mathbb E\mathcal H_{N,\rm row}^+&<14.347498.
\end{aligned}
\tag{KUPC.30}
$$

There is also the complete-population tail bound

$$
\begin{aligned}
\Pr(\mathcal H_{N,\rm cnt}^+\ge20)
&\le e^{-.042N}+e^{-.4649N},\\
\Pr(\mathcal H_{N,\rm row}^+\ge20)
&\le e^{-.02725N}+e^{-.0068N}.
\end{aligned}
\tag{KUPC.31}
$$

In particular either probability is at most $2e^{-.006N}$.
On the complementary event at least $\lceil N/81\rceil$ rows
are in the actual central joint core

$$
B_H=\{(x,v): A_H|x|^2+|v|^2/2<20.25\}.
\tag{KUPC.32}
$$

This core lies strictly inside the central barrier cell and $D$;
its counted rows are actual survivors. These conclusions use the
whole Gaussian laws, including both tails beyond every integration
partition used to certify the coefficients.
Consequently the actual one-step all-dead probability from this
declared source class is bounded by (KUPC.31). This local maximum
hazard is not a principal-rate bound for an unproved recurrent class.
:::

:::{prf:proof}
An orthogonal component rotation and restitution $.5$ preserve each
component mean and contract its centered velocity square. Consequently
the normalized full velocity square is at most $V^2$, even if some
participating input labels are dead. Count $W_X$ is a positive
semidefinite contraction at $a=.006$, so $\|U\|_{2,N}\le E_{
m cnt}$.
For the actual row normalization, its Gaussian nonself stochastic
matrix $P_X$ has column mass at most $C_3^{(40)}<7188$. Jensen's
inequality yields $\|P_Xv\|_{2,N}\le\sqrt{7188}\|v\|_{2,N}$.
Thus $(1-a)I+aP_X$ gives $\|U\|_{2,N}\le E_{
m row}$.
These bounds hold pointwise for every realized unbounded jitter
array. At $N=1$ the configured viscous term is zero, which is
bounded by the same constants.

With all frozen sources zero, $X_i=A_i\sigma Z_i^J$ and $g(0)=0$.
Oddness makes its Gaussian mean zero, while

$$
\mathbb E[Z\sin(\omega\sigma Z)]
=\omega\sigma e^{-\omega^2\sigma^2/2},\qquad
\mathbb E\sin^2(\omega\sigma Z)
=\tfrac12(1-e^{-2\omega^2\sigma^2})
$$

give (KUPC.27) exactly. Therefore
$\mathbb E\|g(X)\|_{2,N}^2\le3k_g$, with its actual accepted
fraction retained and then bounded above by one. Minkowski and the
pointwise energy envelope (KUPC.28), followed by conditional
centering of the complete final position innovation, give

$$
\mathbb E\|x^+\|_{2,N}^2
\le[\sqrt{3k_g}+bE_m]^2+3\tau^2.
$$

The original radial cap, after the actual second force and graph
kick, gives $|v_i^+|\le V$. Finally
$1-\cos z\le z^2/2$ gives $U(x)\le A_H|x|^2$ globally.
This proves (KUPC.29)--(KUPC.30). No sign for the second graph's
force work is being assumed: the actual physical cap supplies the
stated kinetic-energy term after that full graph/force operation.

For the population tail put

$$
M_g(\delta)=\mathbb E e^{\delta g(\sigma Z)^2}.
$$

The finite certificate below proves $\log M_g(10)<.061$.
Conditional on the complete pre-jitter pattern, each accepted own
coordinate innovation is independent and each unaccepted
coordinate contributes one. Since $M_g(10)\ge1$, Chernoff gives
for every $R_g>0$

$$
\Pr\{\|g(X)\|_{2,N}>R_g\}
\le\exp\{-N[10R_g^2-3(.061)]\}.
\tag{KUPC.33}
$$

This product is used only for the independent own-jitter array.
The possibly correlated $U(X)$ has already been removed by the
pointwise normalized energy bound, rather than factored from it.
Conditionally on the whole preparation, the actual final position
innovation $Z$ is independent rowwise standard Gaussian. For
$R_n^2>3\tau^2$, its chi-square Chernoff bound is

$$
\Pr\{\tau\|Z\|_{2,N}>R_n\}
\le e^{-NI_n(R_n)},\qquad
I_n(R_n)=\frac32\left[
 \frac{R_n^2}{3\tau^2}-1-\log\frac{R_n^2}{3\tau^2}\right].
\tag{KUPC.34}
$$

On the two complementary events, pathwise
$\|x^+\|_{2,N}\le R_g+bE_m+R_n$. The choices

$$
\begin{array}{c|ccccc}
m&R_g&R_n&A_H(R_g+bE_m+R_n)^2+2&
 10R_g^2-.183&I_n(R_n)\\ \hline
\rm cnt&.15&.05&<17.380177&.042&>.4649\\
\rm row&.145&.037&<19.838454&.02725&>.0068
\end{array}
\tag{KUPC.35}
$$

give (KUPC.31) by a union bound, without any independence premise
between these two complementary events. The bounds are conditional
uniformly over the complete source/gate/Haar pattern, so averaging
the original pattern preserves them.

If $\mathcal H_N^+<20$, the number of rows outside (KUPC.32) is
strictly less than $(20/20.25)N=(80/81)N$. Thus its number of core
rows exceeds $N/81$ and is at least $\lceil N/81\rceil$.
Every core row satisfies $|x|<\sqrt{20.25/A_H}<.320<3/8$,
which places it inside the actual central barrier cell. This
also verifies its actual survival mark. $\square$
:::

:::{prf:lemma} Finite Gaussian certificate for the joint-energy tail
:label: lem-kupc-joint-energy-mgf-certificate

With the original $\sigma=.1$, take $\delta=10$, $M=8000$,
$\Delta=8/M=.001$, $z_j=j\Delta$, and

$$
\begin{aligned}
a_T&=\tfrac12-\delta\ell^2\sigma^2>0,\qquad
B_T=2\delta\ell\sigma A_g,\\
T_8&=\frac{2}{\sqrt{2\pi}}
\frac{\exp(\delta A_g^2-64a_T+8B_T)}{16a_T-B_T},\\
\overline M_g&=
\frac{2\Delta}{\sqrt{2\pi}}
\sum_{j=1}^{8000}
\exp\{\delta g(\sigma z_j)^2-z_{j-1}^2/2\}+T_8.
\end{aligned}
\tag{KUPC.36}
$$

Then $M_g(10)\le\overline M_g$ and elementary outward evaluation
gives

$$
.06070467495893<\log\overline M_g<.06070467495894<.061.
\tag{KUPC.37}
$$
:::

:::{prf:proof}
The native scalar map is odd and increasing globally. On each
$[z_{j-1},z_j]\subset[0,8]$ the function
$e^{\delta g(\sigma z)^2}$ is increasing, while the Gaussian
density is decreasing. Their separate endpoint maxima give each
upper rectangle in (KUPC.36). This proves the interior bound
without a quadrature-error assumption. For $z\ge8$,
$|g(\sigma z)|\le\ell\sigma z+A_g$, so the two Gaussian tails
are at most

$$
\frac2{\sqrt{2\pi}}e^{\delta A_g^2}
\int_8^\infty e^{-a_Tz^2+B_Tz}\,dz.
$$

For $z=8+s$, its exponent is
$-64a_T+8B_T-(16a_T-B_T)s-a_Ts^2$.
Since $16a_T-B_T>0$, dropping only $-a_Ts^2$ and integrating
the remaining exponential gives exactly $T_8$. Both omitted
Gaussian tails are therefore included in the displayed coefficient.
An 80-decimal-place outward interval evaluation of the 8000
elementary summands and the tail gives (KUPC.37), as well as
(KUPC.30) and (KUPC.35). All arguments are rational intervals
combined with interval $\pi$, exponential, sine, square root and
logarithm; no floating quadrature value is used as a proof premise.
The finite expression (KUPC.36) is itself the reproducible receipt.
$\square$
:::

:::{prf:remark} What the moving and energy producers close
:label: rem-kupc-moving-joint-status

The moving target closes the common-velocity omission in the native
one-update central growth calculation. The complete velocity
anchors (KUPC.24), together with (KUPC.25), retain separate phase
orbits, both force kicks, graph work, the OU/position correlation
and the actual cap. The joint-energy result is a positive,
exponentially likely population-class output from its declared
central donor-source input, for both normalizations and every $N$.
It applies, for example, to the source array immediately after
mandatory revival from a single surviving donor at zero, with
arbitrary capped retained input velocities.

The reached joint-energy class permits a dispersed central
population and a charged exterior fraction. It is broader than
the all-zero donor-source input. Neither its mean energy nor its
exponentially likely central fraction proves that the next
selection/source preparation returns to the input class or has
the protected PGF inequality (KUPC.14). Closing that next producer
requires the original reward/diversity/normalizer source sum for
this reached mixture and the retained within/between phase energy
and flux ledgers. Its principal killing rate must then be compared
to the already proved discovery clock. No repeated protected rate
or eigenfunction oscillation bound is asserted from this single
transition.
:::

:::{prf:proposition} Complete source-profile producer with moving center and phase mixture
:label: prop-kupc-moving-source-profile

Freeze any actual complete pre-jitter source/gate/Haar pattern.
Let $c_*$ and $v_*$ be declared position and bulk-velocity anchors,
measurable before the own-jitter/OU/final-position innovations,
and put

$$
c_*^+=g(c_*)+bv_*,\qquad
R_v=\|v-v_*\|_{2,N},\quad
R_\infty=\max_i|v_i-v_*|.
$$

The maximum in $R_\infty$ is only over capped input velocities;
it is never taken over a Gaussian innovation. Set

$$
E_{v,\rm cnt}=R_v,\qquad
E_{v,\rm row}=\min\{(1-a+a\sqrt{7188})R_v,\ 2R_\infty\}.
\tag{KUPC.38}
$$

For $0<\delta<(2\ell^2\sigma_J^2)^{-1}$ define the exact scalar
source coefficient

$$
\begin{aligned}
M_\delta(\mu,c,0)&=e^{\delta[g(\mu)-g(c)]^2},\\
M_\delta(\mu,c,1)&=\int_{\mathbb R}
  e^{\delta[g(\mu+\sigma_Jz)-g(c)]^2}\phi(z)\,dz,\\
L_{\delta,N}(c_*)&=\frac1N\sum_{i=1}^N\sum_{r=1}^3
  \log M_\delta(\mu_{ir},c_{*,r},A_i).
\end{aligned}
\tag{KUPC.39}
$$

These finite coefficients retain each original source, its actual
acceptance/revival indicator and the original force. For any
$R_g,R_n>0$ with $R_n^2>3\tau^2$, the actual full update obeys

$$
\begin{aligned}
\Pr\{\|x^+-c_*^+\|_{2,N}>R_g+bE_{v,m}+R_n
       \mid\mathrm{pattern}\}
\le{}&\min\{1,e^{-N[\delta R_g^2-L_{\delta,N}(c_*)]}\}
       +e^{-NI_n(R_n)}.
\end{aligned}
\tag{KUPC.40}
$$

Its coefficient is exactly phase-resolved: partition the actual
frozen sources by their actual barrier cells $\Omega_j$ and let
$n_j$ be their counts. Set
$\bar L_{\delta,j}=n_j^{-1}\sum_{i:\mu_i\in\Omega_j}
 \sum_{r=1}^3\log M_\delta(\mu_{ir},c_{*,r},A_i)$,
including every own $A_i$. Then

$$
L_{\delta,N}=\sum_{j:n_j>0}\frac{n_j}{N}\bar L_{\delta,j}.
\tag{KUPC.41}
$$

Thus the between-phase mixture is present rather than replaced
by a single within-phase curvature. All unbounded jitter
excursions are included in the coefficients in (KUPC.39).

For a declared output phase, choose $r>0$ with
$B(c_*^+,r)$ inside its actual barrier cell and $D$. Define

$$
H_i^{\rm orb}=A_H|x_i^+-c_*^+|^2+|v_i^+|^2/2,
\qquad h=A_H(R_g+bE_{v,m}+R_n)^2+2.
\tag{KUPC.42}
$$

Whenever $h<A_Hr^2$, outside the event bounded in (KUPC.40),
more than $N(1-h/(A_Hr^2))$ actual output rows lie in the moving
joint core $H_i^{\rm orb}<A_Hr^2$. Every such row is an actual
survivor in that declared phase. The possible complementary
rows and their between-phase/exterior flux remain in the physical
transition; this assertion does not impose an absorbing phase.

The actual physical energy is related to this orbital readout by

$$
\frac1N\sum_i[U(x_i^+)+|v_i^+|^2/2]
\le U(c_*^+)-F(c_*^+)\cdot(\bar x^+-c_*^+)
                  +\frac1N\sum_iH_i^{\rm orb}.
\tag{KUPC.43}
$$

The linear force residual is retained when the moving center is
not a force root.
:::

:::{prf:proof}
Each collision preserves its mean and contracts its relative
velocity square. Consequently it contracts the full square
relative to any common $v_*$, not only relative to its own mean.
Count $W_X$ preserves constants and contracts $L^2_N$, giving
$\|U-v_*\|_{2,N}\le R_v$. The row column bound gives its first
bound in (KUPC.38). If $|v_i-v_*|\le R_\infty$, each component
mean deviation has norm at most $R_\infty$ and each centered
row deviation has norm at most $2R_\infty$. Restitution $.5$
therefore gives $|v_i^{\rm col}-v_*|\le2R_\infty$.
Stochasticity of the actual first row matrix preserves this
bound. Both conclusions hold after every jitter has been revealed.

Given the pattern, all own coordinates in $g(X)$ are independent.
Their exact exponential product is

$$
\mathbb E\exp\{\delta\sum_i|g(X_i)-g(c_*)|^2\}
=\prod_{i,r}M_\delta(\mu_{ir},c_{*,r},A_i)
=e^{NL_{\delta,N}(c_*)}.
$$

Chernoff gives the first term of (KUPC.40). The actual position
identity is

$$
x_i^+-c_*^+=g(X_i)-g(c_*)+b(U_i-v_*)+\tau Z_i.
$$

Its normalized triangle inequality and (KUPC.34) give (KUPC.40)
by a union bound. The $U(X)$ term is removed by its pointwise
bound before the product above is used. Thus no independent
factorization of the jitter-dependent graph term is required.
Equation (KUPC.41) only regroups this exact source product.

For a joint core, the actual second kick followed by the original
cap gives $|v_i^+|^2/2<2$ for every finite original noise outcome.
Hence the good event bounds $N^{-1}\sum_iH_i^{\rm orb}<h$.
Rows outside its core each
contribute at least $A_Hr^2$, which proves the asserted count.
Core membership implies $|x_i^+-c_*^+|<r$, establishing its
actual phase and survival marks. Finally the global native
Hessian has upper eigenvalue $2+40\pi^2=2A_H$.
Taylor's integral remainder for the actual potential, without
an assumption that the segment stays in a phase, gives

$$
U(x)\le U(c_*^+)-F(c_*^+)\cdot(x-c_*^+)
                                  +A_H|x-c_*^+|^2.
$$

Averaging and adding the actual capped kinetic energy proves
(KUPC.43). $\square$
:::

:::{prf:lemma} Primitive finite upper evaluation of every source-profile coefficient
:label: lem-kupc-full-source-mgf-finite

For an accepted source in (KUPC.39), put
$C=\ell|\mu-c|+2A_g$, $a_T=1/2-\delta\ell^2\sigma_J^2>0$,
$B_T=2\delta\ell\sigma_JC$ and choose any
$R>B_T/(2a_T)$. Let $-R=z_0<\cdots<z_M=R$ be a finite
partition and

$$
B_j=\max\{|g(\mu+\sigma_Jz_{j-1})-g(c)|,
               |g(\mu+\sigma_Jz_j)-g(c)|\}.
$$

Then the complete, unbounded coefficient has the primitive bound

$$
M_\delta(\mu,c,1)
\le\sum_{j=1}^Me^{\delta B_j^2}
                  [\Phi(z_j)-\Phi(z_{j-1})]
 +\frac2{\sqrt{2\pi}}
   \frac{e^{\delta C^2-a_TR^2+B_TR}}{2a_TR-B_T}.
\tag{KUPC.44}
$$

It can therefore be certified by finite elementary interval
arithmetic and an integrated Gaussian exponential series with its
remainder selected for the largest partition endpoint. Neither the
input source mixture nor the original noise law is changed.
:::

:::{prf:proof}
Since $g$ is increasing, the maximum of
$|g(\mu+\sigma_Jz)-g(c)|$ on each partition interval is attained
at an endpoint. This gives the finite interior sum. Globally,

$$
|g(\mu+\sigma_Jz)-g(c)|
\le\ell\sigma_J|z|+\ell|\mu-c|+2A_g
=\ell\sigma_J|z|+C.
$$

Each Gaussian tail is bounded by completing the same negative
quadratic as in the proof of (KUPC.36), now at $R$, and dropping
only its nonpositive $-a_Ts^2$ remainder. Integrating the retained
exponential gives the second term in (KUPC.44). This also proves
finiteness under the stated primitive restriction on $\delta$.
$\square$
:::

:::{prf:corollary} A macroscopic central orbital output with full common speed
:label: cor-kupc-central-common-speed-core

In {prf:ref}`thm-kupc-central-joint-energy`, suppose also that
every input velocity is the same $v_*$, $|v_*|\le2$.
For both viscous normalizations, put $c_*^+=bv_*$ and $r=.2965$.
Except with probability at most
$e^{-.042N}+e^{-.4649N}$, more than $.43N$ rows lie in the moving
central joint core

$$
A_H|x_i^+-bv_*|^2+|v_i^+|^2/2<A_H(.2965)^2.
\tag{KUPC.45}
$$

These rows are actual central-phase survivors. This is a
complete-population output assertion at the original $V=2$,
not a restriction to small common velocity.
:::

:::{prf:proof}
Here $R_v=R_\infty=0$, so the actual first graph has $U=v_*$
and both relative velocity constants in (KUPC.38) are zero.
Use $R_g=.15,R_n=.05$ and the certified central-source coefficient
$L_{10,N}(0)\le3(.061)$. Equations(KUPC.33)--(KUPC.34) give the
stated exceptional probability. On the good event,

$$
\frac1N\sum_iH_i^{\rm orb}
\le A_H(.15+.05)^2+2<9.935684,
\qquad A_H(.2965)^2>17.441094.
$$

Consequently its core fraction is greater than
$1-9.935684/17.441094>.43$. The actual center satisfies
$|c_{*,r}^+|\le2b<.078432$, and
$2b+.2965<.375$. Its core ball is therefore inside the central
barrier cell and $D$, proving the actual marks. This uses the
full second-kick/cap output only through its original kinetic
energy bound; all OU and final-position innovations still occur.
$\square$
:::

:::{prf:corollary} The joint-energy producer holds on a positive-volume donor band
:label: cor-kupc-positive-volume-donor-band

Use the original record and complete preparation of
{prf:ref}`thm-kupc-central-joint-energy`, and suppose

$$
\mu_i\in[-\varepsilon_s,\varepsilon_s]^3,
\qquad \varepsilon_s=.005,
\quad\hbox{for every actual frozen source}.
\tag{KUPC.46}
$$

All stored input velocities retain $|v_i|\le2$, with no small
relative-velocity premise. For every actual complete pre-jitter
pattern satisfying (KUPC.46), both normalization tags obey

$$
\begin{aligned}
\Pr(\mathcal H_{N,\rm cnt}^+\ge20)
  &\le e^{-.027N}+e^{-.4649N},\\
\Pr(\mathcal H_{N,\rm row}^+\ge20)
  &\le e^{-.01225N}+e^{-.0068N}.
\end{aligned}
\tag{KUPC.47}
$$

Thus the actual central joint-core output and local all-dead
bound $2e^{-.006N}$ still hold. When the input velocities are
all the same arbitrary $v_*$, $|v_*|\le2$, the moving core in
(KUPC.45) contains more than $.43N$ rows except with probability
at most $e^{-.027N}+e^{-.4649N}$, for either normalization.
In particular these estimates apply after actual mandatory
revival from a singleton donor anywhere in the positive-volume
band (KUPC.46), with the original retained dead-slot velocities.
:::

:::{prf:proof}
Odd monotonicity of the original scalar landing map gives, for
every $|\mu|\le\varepsilon_s$ and $z\in\mathbb R$,

$$
|g(\mu+\sigma_Jz)|\le g(\varepsilon_s+\sigma_J|z|).
$$

For an unaccepted coordinate, $|g(\mu)|\le g(\varepsilon_s)$
is bounded by the same increasing envelope. Therefore each
actual coefficient in (KUPC.39), with $c=0$, is at most

$$
M_s=\mathbb E e^{10g(\varepsilon_s+.1|Z|)^2}.
$$

Its finite upper certificate replaces $g(\sigma z_j)$ in
(KUPC.36) by $g(\varepsilon_s+\sigma z_j)$ and replaces the
tail constant $A_g$ by $C_s=\ell\varepsilon_s+A_g$.
Explicitly, use $a_T=1/2-10\ell^2\sigma^2$ and
$B_s=20\ell\sigma C_s$ to add the full tail

$$
\frac2{\sqrt{2\pi}}
\frac{e^{10C_s^2-64a_T+8B_s}}{16a_T-B_s}
$$

to the 8000 endpoint rectangles. The same 80-digit outward
evaluation gives

$$
.06599230532498<\log\overline M_s
 <.06599230532499<.066.
\tag{KUPC.48}
$$

The product bound is consequently $L_{10,N}(0)<3(.066)$.
For the same $(R_g,R_n)$ choices as (KUPC.35), the own-jitter
Chernoff exponents are $.225-.198=.027$ in count mode and
$.21025-.198=.01225$ in row mode. Their output norm/energy
budgets and Gaussian thermostat exponents are unchanged,
which proves (KUPC.47) and its core count. For common velocity
$U=v_*$ still holds after every jitter, even when the central
sources have different positions within the band. The proof of
(KUPC.45) therefore applies with the new $.027$ exponent.
Singleton revival has all actual sources equal to its donor
position, which verifies (KUPC.46) whenever that donor lies
in the displayed band. Its velocities are retained and are
already covered by the arbitrary capped-velocity proof.
$\square$
:::
