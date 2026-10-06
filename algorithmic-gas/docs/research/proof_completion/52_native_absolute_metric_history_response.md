# Full-history response of the native absolute-metric action

## 1. Complete executed histories and the consumed parameter paths

:::{prf:definition} Native absolute-metric history register
:label: def-namhr-register

Consume the complete registers of {prf:ref}`def-ncma-register` and
{prf:ref}`def-namr-register`, including the algorithm/variant and arithmetic
tags, initial state, all landscape and boundary fields, observation and
recording data, geometric projection, every comparison epsilon, rank
projection and duplicate policy, graph weights, inverse-distance and
row-sum floors, covariance ridge and clamps, curvature and volume floors,
reward scale, fitness regularizers and maps, matching laws, cloning period,
gate regularizer and saturation, simultaneous copying and component
collision, both separately evaluated Boris stages, OU noise, geometry
refresh schedule, cache and native error records.

The primary family retains the existing absolute clipped covariance metric
and the Einstein--Hilbert preset's zero potential, unbounded/all-alive
domain, zero jitter, identity elastic component collision, no velocity cap
and no additional position noise. Its fixed fields include the original
graph coefficient $\nu$, curl coefficient $\beta_{\rm curl}$, ambient
kinetic dimension $a$, projected dimension $d$, population $N$, time step
$h>0$ and friction $\gamma>0$. The reference values are
$a=3,d=2,N=500,h=0.002,\gamma=1,\nu=3,\beta_{\rm curl}=1$.
The literal $20$-step gate schedule, both uniform Fisher--Yates mutual
matchings and odd self companion remain unchanged. The fixed admissible
gate fields $\varepsilon_c\ge0,s_c>0$ remain arguments; their reference
values are $0,1$.

Write

$$
 c=e^{-\gamma h},\qquad b=1-c^2,\qquad
 q(T)=\sqrt{bT},\qquad t=h/2,\qquad T>0.                \tag{NAMHR.1}
$$

The temperature path changes the consumed isotropic thermostat amplitude
$\sqrt{2\gamma T}$ and hence this exact OU variance; no metric diffusion
factor is substituted. Every other configured parameter is fixed. The
initial law is independent of $T$. In particular the origin-start
initial state is permitted.

Fix a finite horizon $n\ge1$. Supply the original independent standard
Gaussian arrays $G_\ell\in\mathbb R^{Na}$, $0\le\ell<n$, and the original
mutual matching marks. A stopped execution retains its original error
record and draws no further executed innovations. For integration alone
one may adjoin independent unused Gaussian arrays after that stopping
time. Their integral is one. This extension does not change any executed
history or assign an action to an error.

The real-coordinate tag retains the actual positive numerical comparison
constants. Rounded floating histories, finite pseudorandom draw laws,
nonhomogeneous geometric domains and other variant tags retain their own
payloads and errors; no continuous-noise derivative is assigned to those
different laws.
:::

## 2. Complete thermal and original source response

:::{prf:theorem} Full chronological thermal response without a smooth branch hypothesis
:label: thm-namhr-full-temperature

In {prf:ref}`def-namhr-register`, let $\mathscr R_n$ denote the original
physical phase history together with any fixed recorded marks, accepted
plans, applied OU increments, geometric caches and native stopping/error
records. Its observation convention is fixed in physical coordinates and
has no explicit temperature-dependent bookkeeping field.
Then its probability law is differentiable in total variation for every
$T>0$. For every bounded Borel function $\Psi$ of this complete record,

$$
 \partial_T E_T\Psi(\mathscr R_n)
 =E_T\!\left[\Psi(\mathscr R_n)\,\mathcal S_{T,n}\right],
 \qquad
 \mathcal S_{T,n}
 ={1\over2T}\sum_{\ell=0}^{n-1}
       \bigl(|G_\ell|^2-Na\bigr).                     \tag{NAMHR.2}
$$

The original score has

$$
 E\mathcal S_{T,n}=0,\qquad
 E\mathcal S_{T,n}^2={nNa\over2T^2},\qquad
 \left|\partial_TE_T\Psi\right|
 \le{\|\Psi\|_\infty\sqrt{nNa}\over\sqrt2\,T}.          \tag{NAMHR.3}
$$

This formula includes every earlier rank projection, retessellation,
pseudoinverse deletion, metric clamp, positive-part cloning gate, accepted
plan and both force evaluations, at the original interaction strength.
It does not require any of these functions of the entering state to be
differentiable. For a stopped record the unused terms in (NAMHR.2) have
zero conditional contribution.

An observation containing the standard $G_\ell$ as an explicitly stored
coordinate requires its direct bookkeeping response under the proof's
applied-increment coordinates. The theorem does not assert a
total-variation derivative of the joint graph
$(G_\ell,q(T)G_\ell)$ while both of its coordinates are observed.
Acceptance uniforms are integrated in the original Bernoulli plan law;
the theorem likewise does not require observing a uniform and its moving
threshold indicator jointly.
:::

:::{prf:proof}
Put $z_\ell=q(T)G_\ell$. Conditional on the complete applied increment
array $z=(z_0,\ldots,z_{n-1})$, the initial state and the original matching
marks, apply the native chronology. At each enabled step evaluate the
actual current geometric rewards, the original fitness maps and clipped
gate probabilities, integrate the native independent acceptance uniforms
in their Bernoulli law, and execute the simultaneous plan. Proceed through
both Boris kicks, A/O/A and every prescribed refresh. This produces a
probability kernel $K(z,\cdot)$ of the full physical record.

All temperature dependence in this parameter path was the OU amplitude.
It has already been replaced by its actually applied increment $z_\ell$.
Thus $K$ is independent of $T$, even if it is discontinuous in $z$.
The graph weights, curl, geometry, reward feedback and gates continue to
depend on the entire executed history inside $K$. Native errors are
absorbing recorded outcomes of that same kernel. No success
normalization is introduced.

With $L=nNa$ the input density is the single $L$-dimensional Gaussian

$$
 \varphi_{T}(z)
 =(2\pi bT)^{-L/2}
    \exp\!\left(-{|z|^2\over2bT}\right),\qquad
 \partial_T\varphi_T
 =\varphi_T\,{|z|^2/(bT)-L\over2T}.                   \tag{NAMHR.4}
$$

On every positive compact temperature interval its derivative and the
difference-quotient remainder have integrable Gaussian-polynomial
envelopes. For completeness, differentiating once more gives
$\partial_T^2\varphi_T=\varphi_T(S_T^2+
[-|z|^2/(bT)+L/2]/T^2)$, where $S_T$ is the displayed first score.
The same envelope proves $L^1$ convergence of the first difference
quotient. Pushing a signed measure through the probability kernel $K$
contracts its total variation. Therefore $\varphi_TK$ has the stated
total-variation derivative. Substituting $z=qG$ proves (NAMHR.2).

The independent original Gaussian coordinates have centered square sum
of variance $2L$. This proves (NAMHR.3). Conditional on a stopping history,
unused independent arrays have centered score and integrate to zero.
All distinctions about directly observed $G$ or acceptance uniforms
follow because those would introduce a parameter-dependent deterministic
observation graph, whereas the physical record kernel just constructed
uses the actual increments and the integrated native plan law.
:::

:::{prf:corollary} Uncut full-history Einstein--Hilbert action and selected normalization
:label: cor-namhr-uncut-temperature

Let $H_n$ be the actual final absolute-metric action when defined, and put

$$
 \widetilde H_n
 =\mathbf1_{\{\text{successful final action}\}}H_n,\qquad
 H_*=4|\lambda_R|(d-1)N U_\rho V_\rho .               \tag{NAMHR.5}
$$

Here $\widetilde H_n$ is notation for its unnormalized successful integral,
not a replacement for the native error record. For every fixed bounded
physical-history test $\Psi$,

$$
 \partial_T E_T[\Psi\widetilde H_n]
 =E_T[\Psi\widetilde H_n\,\mathcal S_{T,n}],\qquad
 \left|\partial_T E_T[\Psi\widetilde H_n]\right|
 \le {\|\Psi\|_\infty H_*\sqrt{nNa}\over\sqrt2\,T}.     \tag{NAMHR.6}
$$

The same formula holds for every fixed finite action power, replacing
$H_*$ by $H_*^p$. No additional action-moment hypothesis is imposed.

For any fixed Borel recorded event $A$ of positive probability
$p_T=P_T(A)>0$, its existing selected normalization has

$$
 \partial_T E_T[\Psi\mid A]
 =E_T[\Psi\mathcal S_{T,n}\mid A]
  -E_T[\Psi\mid A]E_T[\mathcal S_{T,n}\mid A].         \tag{NAMHR.7}
$$

This is the derivative of that explicitly selected finite-history law.
It does not identify it with a QSD or assume an invariant law.
For the origin first step and a position-only final action, the unused
dropped ambient component integrates out and (NAMHR.6) reduces exactly
to (NAMR.4). At later steps the full ambient $a$ remains in (NAMHR.2),
because native graph-force/curl feedback can use the dropped kinetic
coordinate.
:::

:::{prf:proof}
The native bound of {prf:ref}`thm-namr-global-action-bound` holds pointwise
on every successful graph/rank branch. Its constant is independent of
temperature and the physical history. Apply
{prf:ref}`thm-namhr-full-temperature` to the bounded successful integrand;
errors remain recorded outcomes with zero contribution to that integral.
The power statement is identical. Apply (NAMHR.2) to
$\mathbf1_A\Psi$ and $\mathbf1_A$ and differentiate their ratio to obtain
(NAMHR.7). At the coincident origin the first kick is zero and
the final positions are $tqG$; the last ambient coordinate does not enter
the projected action. Its independent centered square integrates to zero.
No such independence is asserted after subsequent native coupling.
:::

:::{prf:proposition} Original Gaussian weak source variation through every branch
:label: prop-namhr-full-source

Fix $T$ and a finite original deterministic source direction
$A=(A_0,\ldots,A_{n-1})$, with $A_\ell\in\mathbb R^{Na}$.
Use the original source operation $G_\ell\mapsto G_\ell+\theta A_\ell$
through the complete executed update, with no alteration of its metric,
force, gate or action. Its weak first variation is

$$
 {d\over d\theta}\bigg|_0E\Psi(\mathscr R_n(G+\theta A))
 =E\!\left[\Psi(\mathscr R_n(G))
        \sum_{\ell} A_\ell\cdot G_\ell\right].         \tag{NAMHR.8}
$$

The formula holds in total variation for the physical record and all
bounded Borel tests; it applies uncut to $\widetilde H_n$ with bound
$H_*\sqrt{\sum_\ell|A_\ell|^2}$.
This is the weak source derivative of the executed pushforward. It does
not introduce a new configured force or assert a pointwise derivative of
the native tessellation.
:::

:::{prf:proof}
The applied increments have translated Gaussian input law
$z_\ell=q(G_\ell+\theta A_\ell)$. The same
temperature-independent execution kernel is now fixed also in $\theta$.
The translated input density has original score
$\sum A_\ell\cdot G_\ell$, whose variance is $\sum|A_\ell|^2$.
Its $L^1$ density differentiation, kernel contraction and the native
action bound prove all assertions.
:::

:::{prf:corollary} Actual record likelihood score and finite thermal comparison
:label: cor-namhr-record-likelihood

For any fixed physical subrecord $\mathscr Q_n$ of the record in
{prf:ref}`thm-namhr-full-temperature`, its actual likelihood derivative is
the conditional native score

$$
 \mathcal S^{\mathscr Q}_{T,n}
   =E[\mathcal S_{T,n}\mid\mathscr Q_n],\qquad
 E(\mathcal S^{\mathscr Q}_{T,n})^2
      \le {nNa\over2T^2}.                             \tag{NAMHR.21}
$$

In particular the uncut action response equals
$E[\widetilde H_n\mathcal S^{\mathscr Q}_{T,n}]$ whenever the successful
action is measurable in that subrecord. This identifies its original
likelihood score, without equating the action to a log density.

If the record includes all actually applied OU increments, write $\tau_n$
for their executed count before horizon/error stopping. Then the score
and its information reduce exactly to

$$
 \mathcal S^{\rm inc}_{T,n}
 ={1\over2T}\sum_{\ell<\tau_n}(|G_\ell|^2-Na),\qquad
 E(\mathcal S^{\rm inc}_{T,n})^2
 ={Na\,E\tau_n\over2T^2}.                            \tag{NAMHR.22}
$$

For two positive temperatures and any of these physical subrecords,

$$
 D(\mathcal L_T(\mathscr Q_n)\Vert
       \mathcal L_{T_0}(\mathscr Q_n))
 \le {nNa\over2}
       \left({T\over T_0}-1-\log{T\over T_0}\right).   \tag{NAMHR.23}
$$

The bound holds for the complete original error/success law. It requires
no prior mixing estimate and controls any finite population/horizon with
the displayed parameters.
:::

:::{prf:proof}
The Gaussian input measures at all positive temperatures are equivalent.
Their common execution kernel gives equivalent output laws. The
conditional expectation of their input derivative, proved in (NAMHR.2),
is precisely the output Radon--Nikodym derivative divided by its law.
Conditional $L^2$ contraction gives (NAMHR.21).

An unused Gaussian array has zero conditional score. For the increment
record every used $G_\ell=z_\ell/q$ is known, proving (NAMHR.22)'s first
identity. The event that an O innovation will be used is measurable
before that fresh innovation. Thus its centered square score is a
martingale difference with predictable execution indicator. Orthogonality
and conditional variance $2Na$ give the second identity, even when an
error caused by that innovation stops later stages of its update.

Finally the Gaussian density ratio is
$\Lambda(z)=\varphi_T(z)/\varphi_{T_0}(z)$.
The output ratio is $E_{T_0}[\Lambda\mid\mathscr Q_n]$.
Conditional Jensen for the convex function $x\log x$ proves that the
output entropy is at most the input entropy. Direct original Gaussian
integration gives (NAMHR.23).
:::

## 3. An existing full-history pure-ridge regime

:::{prf:definition} Origin-start absolute metric with zero graph coefficient
:label: def-namhr-zero-viscosity

This section uses the EXISTING graph coefficient $\nu=0$ in the otherwise
unchanged complete register {prf:ref}`def-namhr-register`. It is an
explicit different parameter regime from the reference $\nu=3$.
The graph module and original curl coefficient remain configured and
executed. With the preset zero potential, the original graph viscous
force is exactly zero at both evaluations; its fitted curl is zero and
its Cayley map is identity. No graph weight is replaced.

The initial positions and velocities are exactly zero. Temperature,
$h,\gamma,\lambda_R,\varepsilon_c,s_c$ and all other parameters are fixed,
while only the existing absolute ridge $\rho$ varies through
$I=[\rho_-,\rho_+]\subset(0,\infty)$. The actual $b_-,b_+,f_R,f_V$,
precision/rank comparisons, volume weights and row floors are fixed.
The reward and diversity standardizers are the preset's
`LegacySample` with their original positive
$\epsilon_r=\epsilon_s=10^{-30}$; the actual distance regularizer is
$\delta_D=10^{-30}$. Both positive maps are amplitude-two zero-floor
logistics and both powers are one. The uniform donor distance is the
original unsquared Euclidean position distance.

Nonconstant geometric reward, sampled diversity and the scheduled
accepted cloning plans remain coupled. They are not set to zero or
replaced by independent particles. Every geometric refresh and both
original B stages are still executed.
:::

:::{prf:lemma} Chronology-wise ridge pullback and primitive full gate budgets
:label: lem-namhr-chronological-pullback

In {prf:ref}`def-namhr-zero-viscosity`, put

$$
 z_\ell={qG_\ell\over\sqrt\rho},\qquad
 M_n(z)=hn\sum_{\ell<n}|z_\ell|,\qquad L=nNa.          \tag{NAMHR.9}
$$

Conditional on all actual matching marks and accepted patterns, every
physical phase coordinate is $\sqrt\rho$ times a trajectory
$(\bar X_k,\bar V_k)$ independent of $\rho$. For every such pattern path,

$$
 \max_{k\le n,i}|\bar V_{k,i}|\le\sum_{\ell<n}|z_\ell|,
 \qquad \max_{k\le n,i}|\bar X_{k,i}|\le M_n(z).       \tag{NAMHR.10}
$$

All actual geometric graphs, ranks, spectral deletions and their
success/error flags at every chronological refresh are independent of
$\rho$ in these proof coordinates. Kept metric clamps and determinant
floors may change; their pulled-back values are continuous and locally
absolutely continuous. Thus every pulled-back reward, fitness and native
gate is locally absolutely continuous, with its actual one-sided values
at clamp and positive-part boundaries.

The following uniform constants derive solely from the original register:

$$
\begin{split}
 U_*&=\sup_{\rho\in I}U_\rho,\qquad
 V_*=\sup_{\rho\in I}V_\rho,\\
 R_*&=4|\lambda_R|(d-1)U_*V_*,\\
 J_*&={2|\lambda_R|(d-1)(d+3)(U_*+1)V_*\over\rho_-},\\
 A_N&=2+\sqrt{2(N-1)},\qquad
 \ell_N={2\over1+\exp\sqrt{N-1}},\\
 C_F(z)&=A_N\left({J_*\over\epsilon_r}
                  +{M_n(z)\over\epsilon_s\sqrt{\rho_-}}\right),\\
 B(z)&={16N\over s_c\ell_N^2}\,C_F(z).
\end{split}                                             \tag{NAMHR.11}
$$

For $N\ge2$, each reward and fitness obeys, on both one-sided branches,

$$
 |r_i|\le R_*,\qquad |\partial_\rho\widehat r_i|\le J_*,
 \qquad
 \ell_N^2\le f_i\le4,\qquad
 |\partial_\rho\log\widehat f_i|\le C_F(z).            \tag{NAMHR.12}
$$

If $\pi_{k,C}$ is one enabled step's full original accepted-pattern
probability, without dividing by any zero gate, then

$$
 \sum_C|\partial_\rho\widehat\pi_{k,C}|\le B(z),\qquad
 \sum_{\mathbf C}
       |\partial_\rho\widehat W_{\mathbf C}|
       \le|\mathcal I_n|B(z),                         \tag{NAMHR.13}
$$

where $\widehat W_{\mathbf C}$ is the product of the chronological
conditional probabilities and $\mathcal I_n$ is the actual set of enabled
steps. Its primitive integrability bound is

$$
 E_\rho B(z)\le
 \overline B_n:={16NA_N\over s_c\ell_N^2}
 \left({J_*\over\epsilon_r}
       +{hn^2q\sqrt{Na}\over\epsilon_s\rho_-}\right).
                                                               \tag{NAMHR.14}
$$

No independence between a cloning decision and its metric is used.
:::

:::{prf:proof}
At zero graph coefficient both actual viscous forces are zero. The curl
fit of the zero force is zero; the original positive fit regularizer
keeps its inverse defined, and its Cayley map is identity. Hence the
only velocity evolution is
$\bar V_{k+1}=c\bar V_k+z_k$. Identity elastic collisions return precisely
the frozen pre-cloning velocity field even on accepted components.
The simultaneous position-copy map is a row selector. It commutes with
a common dilation and does not increase the maximum row norm.
The two A displacements total $t(1+c)\bar V_k+tz_k$.
Since $0<c<1$ and $t=h/2$, induction gives (NAMHR.10).
Intermediate stage coordinates obey the same bounds.

At every refresh, apply
{prf:ref}`lem-namr-ridge-pullback` to the actual scaled positions.
The literal positive affine-rank comparison, projected tessellation,
duplicates and relative-eigenvalue deletion test are fixed in $\bar X$,
including the rank-one branch. The pulled-back metric eigenvalues are
exactly $[\beta_j(\bar X)/\rho]_{b_-}^{b_+}$ on that fixed retained set.
Their continuous clamp derivatives and the original determinant floors
give the row-wise form of (NAMR.8)--(NAMR.9):
$|\widehat r_i'|\le J_*$.
Finite real arithmetic expressions and positive regularizers retain
their native successful/error branches. Allocation sizes depend on the
same fixed graphs and dimensions. A strict metric/error policy or
nonhomogeneous domain is not included in this clipped open-domain lemma.

For a sample vector $y$, let
$\sigma=\|y-\bar y\mathbf1\|_2/\sqrt{N-1}$ and
$z_i=(y_i-\bar y)/(\sigma+\epsilon)$.
For any absolutely continuous vector path with
$\max_i|y_i'|\le J$, one-sided norm differentiation gives
$|\sigma'|\le\sqrt2J$. The centered numerator has derivative at most $2J$,
and the original standardized values obey
$|z_i|\le\sqrt{N-1}$. Therefore

$$
 |z_i'|\le {A_NJ\over\epsilon}.                       \tag{NAMHR.15}
$$

This includes a constant sample and its original positive epsilon;
there is no division by a vanishing sample variance.
The reward bound uses $J=J_*$. For the distance donor of row $i$,
write $D_i=|\bar X_i-\bar X_{j(i)}|$. Its actual separation is
$\sqrt{\rho D_i^2+\delta_D^2}$, with derivative at most
$D_i/(2\sqrt\rho)\le M_n(z)/\sqrt{\rho_-}$.
Apply (NAMHR.15) with $\epsilon_s$. The log derivative of the
amplitude-two logistic has absolute value at most one in its standardized
argument. Multiplying the two original channels proves (NAMHR.12).

For the native gate let
$u=(f_j-f_i)/[(f_i+\varepsilon_c)s_c]$ and $p=[u]_0^1$.
Its clip is one-Lipschitz. Since
$f_j/f_i\le4/\ell_N^2$ and $\varepsilon_c\ge0$,

$$
 |p'|\le {8\over s_c\ell_N^2}C_F(z).                 \tag{NAMHR.16}
$$

Differentiate the original product probability
$\pi_C=\prod_i p_i^{C_i}(1-p_i)^{1-C_i}$ as a product of its factors,
including their values at $p_i=0,1$. Summing absolute derivatives over
all patterns gives at most $2\sum_i|p_i'|$, which is (NAMHR.13)'s first
bound. Disabled steps have one pattern of probability one.
For the chronological product, differentiate one step at a time.
Sum all later conditional patterns to one, then use the same uniform
bound at that earlier node and sum its past probabilities to one.
This proves the second bound even though every later fitness depends
on the earlier accepted plans.

Under the actual Gaussian input law in (NAMHR.9),
$E_\rho|z_\ell|\le q\sqrt{Na}/\sqrt{\rho_-}$.
Thus $E_\rho M_n\le hn^2q\sqrt{Na}/\sqrt{\rho_-}$,
which proves (NAMHR.14). These are finite explicit bounds even with the
very small original $10^{-30}$ standardizers.
The same proof covers one-sided derivatives at the original continuous
clamp, floor, sample-norm and gate boundaries: at fixed $\bar X$ the
eigenvalues and raw distances have the displayed analytic/rational
dependence on $\rho$, and their finitely many continuous max/min and norm
operations have those one-sided derivatives. An $N=1$ population has
zero native curvature action and constant standardized channels instead.
:::

## 4. Uncut chronological ridge variation

:::{prf:theorem} Full-history native ridge/action response with earlier interfaces paid
:label: thm-namhr-full-ridge

In {prf:ref}`def-namhr-zero-viscosity`, let
$\widehat H_{n,\mathbf C}(\rho,z)$ be the actual final action on its
successful branch for the pattern history $\mathbf C$, evaluated at its
pulled-back physical positions. Its success indicator is the fixed
one of {prf:ref}`lem-namhr-chronological-pullback`.
Retain the full chronological matching context $D$ and its original law.
Then the uncut successful-action integral has both one-sided derivatives
at every interior $\rho\in I$:

$$
\begin{split}
 \partial_\rho^\pm E_{\rm succ}H_n
 =E_D E_{\rho,z}\sum_{\mathbf C}\bigg[
   &(\partial_\rho^\pm\widehat W_{\mathbf C})
                           \widehat H_{n,\mathbf C}\\
   &+\widehat W_{\mathbf C}
                           \partial_\rho^\pm\widehat H_{n,\mathbf C}\\
   &-{\widehat W_{\mathbf C}\widehat H_{n,\mathbf C}\over2\rho}
                      \left(\sum_{\ell<n}|G_\ell|^2-nNa\right)
                              \bigg].               \tag{NAMHR.17}
\end{split}
$$

All action terms denote their original unnormalized successful integrals.
The direct action derivative is exactly the native derivative in the
ridge-dilated coordinates of (NAMR.7)--(NAMR.9), summed over the original
reward allocation; it is not a different action.
The complete bound is

$$
 |\partial_\rho^\pm E_{\rm succ}H_n|
 \le NJ_*+
 H_*\left(|\mathcal I_n|\overline B_n+
                 {\sqrt{2nNa}\over2\rho_-}\right),    \tag{NAMHR.18}
$$

where $H_*$ may be its supremum over $I$.
The third term in (NAMHR.17) pays all original chronological rank,
tessellation and spectral-deletion interfaces through the Gaussian
history density. The first term retains the actual geometric fitness and
all accepted-plan feedback.

For a fixed differentiable test of the physical phase coordinates
$\Psi(X_0,V_0,\ldots,X_n,V_n)$ with bounded derivatives, the same response
for $\Psi H_n$ has its additional original material term

$$
 {H_n\over2\rho}\mathcal D\Psi,\qquad
 \mathcal D\Psi
 =\sum_{k=0}^n\sum_i
       \bigl(X_{k,i}\cdot\nabla_{X_{k,i}}\Psi
             +V_{k,i}\cdot\nabla_{V_{k,i}}\Psi\bigr). \tag{NAMHR.19}
$$

This term is integrable by (NAMHR.10) and the original Gaussian law.
A fixed bounded dilation-invariant test needs no such term.
No full-history total-variation ridge derivative for arbitrary
discontinuous physical tests is inferred from (NAMHR.17).
:::

:::{prf:proof}
For each chronological pattern path the physical trajectory is the
original one evaluated at $\sqrt\rho(\bar X,\bar V)$.
The complete array $z$ has density

$$
 \psi_\rho(z)
 =\left({\rho\over2\pi q^2}\right)^{L/2}
       \exp\!\left(-{\rho|z|^2\over2q^2}\right),\qquad
 \partial_\rho\log\psi_\rho
 =-{\,|q^{-1}\sqrt\rho z|^2-L\over2\rho}.             \tag{NAMHR.20}
$$

The original integral is therefore exactly
$E_D\int\psi_\rho(z)\sum_{\mathbf C}
\widehat W_{\mathbf C}(\rho,z)
\widehat H_{n,\mathbf C}(\rho,z)\,dz$.
Every action is bounded by $H_*$, its pulled-back direct derivative by
$NJ_*$, and the sum of all weight derivatives by
$|\mathcal I_n|B(z)$. Gaussian differentiation adds its integrable
quadratic score. Equations (NAMHR.10)--(NAMHR.14) give a common
Gaussian-polynomial dominating function on $I$. One-sided dominated
differentiation proves (NAMHR.17) without removing or reinterpreting any
earlier discontinuous metric interface. The original source Gaussian
has $E||G|^2-L|\le\sqrt{2L}$, proving (NAMHR.18).

For a differentiable physical test,
$\partial_\rho(X,V)=(X,V)/(2\rho)$ in the fixed $z$ coordinates.
The chain rule gives exactly (NAMHR.19). Its bounded gradients and the
Gaussian linear trajectory envelope justify the same differentiation.
A dilation-invariant bounded Borel test is fixed under this coordinate
change and needs no material derivative. An arbitrary Borel test need
not be differentiable under that deterministic observation dilation;
the theorem makes no such assertion.
:::

:::{prf:proposition} Literal metric payload prevents a full-record TV ridge assertion
:label: prop-namhr-native-metric-tv

Use the existing allowed $N=3,d=2$ absolute-metric configuration at
$\rho_0=1$, the default lower bound $10^{-6}$ and absent upper bound,
with the original positive comparison constants and determinant floors.
From the coincident origin, any fixed $\nu\ge0$ and the original
$q>0$ give a first terminal position law independent of $\rho$.
If the consumed native record includes its original metric payload,
the joint law of $(Y,g)$ is not continuous in total variation as
$\rho\to\rho_0$.

Specifically there is a fixed strict open triangle event $O$ with
probability $p_O>0$, independent of all sufficiently nearby $\rho$,
such that

$$
 \|\mathcal L_{\rho}(Y,g)-\mathcal L_{\rho_0}(Y,g)\|_{\rm TV}
       \ge p_O,\qquad 0<|\rho-\rho_0|\text{ sufficiently small}.
                                                               \tag{NAMHR.24}
$$

This concerns the actual `TessellationGeometry.metric` payload, rather
than a hypothetical new observable. It does not invalidate the proved
uncut action variation, which uses its original material response.
:::

:::{prf:proof}
Take a sufficiently small neighborhood of the original strict triangle
$(0,0),(1,0),(0,1)$. Its literal rank and graph branches are strict.
The three displacement covariances displayed in
{prf:ref}`prop-namr-nonzero-action` have eigenvalues of order one.
At $\rho_0=1$ and all nearby ridges every pseudoinverse mode is retained
and every original clamp is inactive. The original metric is therefore
exactly $g_i(Y;\rho)=(C_i(Y)+\rho I)^{-1}$ on this open neighborhood.

At the origin, all gates and the first B kick vanish. A2 positions are
$Y=tqG$ and the executed second kick does not move them.
Their nondegenerate Gaussian law has positive probability $p_O$ of that
same triangle neighborhood, regardless of $\rho$. Define the fixed
measurable record event
$A=\{Y\in O,\ g_0=(C_0(Y)+\rho_0I)^{-1}\}$.
Its probability is $p_O$ at $\rho_0$.
For every different nearby $\rho$ its probability is zero: if
$(C+\rho I)^{-1}=(C+\rho_0I)^{-1}$ for positive definite matrices,
inverting gives $\rho=\rho_0$. Taking this event in the defining TV
supremum proves (NAMHR.24).
Every step, metric and graph used here is an existing native payload.
Only an already computed record coordinate was selected.
:::

## 5. Full interaction-strength ridge response through ambient positions

:::{prf:definition} Full-force chronological position coordinates
:label: def-namhr-full-force

Retain {prf:ref}`def-namhr-register` and its origin initial state, but
now allow every finite configured graph coefficient $\nu\ge0$ and
curl coefficient $\beta_{\rm curl}\ge0$. Keep the original normalized
`RiemannianKernelVolume` weight specification, whose configured positive
length is $\ell$ (reference $\ell=1$). Keep every other field fixed while
the original absolute ridge varies in $I=[\rho_-,\rho_+]$.
The reference $(h,\nu,\beta_{\rm curl})=(0.002,3,1)$ is included.

Let $Y_k\in\mathbb R^{Na}$ be the FULL ambient terminal position array;
$Y_0=0$. This coordinate includes the kinetic coordinate excluded by
`DropLast` from the metric. Put $Y_k=\sqrt\rho Z_k$.
Conditional on the original matching and accepted-plan history, all
cloned positions are row selections of these arrays. There are only the
native pre-fitness, post-cloning and post-kinetic geometric refreshes of
the consumed schedule. The B2 graph/weights remain the actual cached
post-cloning ones, even though its fitted curl uses the current A2
positions and current force.

Write $\tau=h/4$ and $b_{\rm C}=\beta_{\rm curl}h/4$.
The two literal graph B maps are denoted $\mathcal B_{1,k}$ and
$\mathcal B_{2,k}$; they each contain their original quarter kick,
current curl fit, Cayley rotation and reevaluated quarter force.
This is an observation-coordinate change, not a replacement update.
:::

:::{prf:lemma} Exact Gaussian position history and native velocity reconstruction
:label: lem-namhr-full-position-history

For each accepted pattern path the complete native velocities reconstruct
recursively from the chronological positions:

$$
\begin{split}
 V_0&=0,\quad X_{C,k}=\text{the actual simultaneous row selection of }Y_k,\\
 V_{{\rm postclone},k}&=V_k
       \quad\text{after the configured frozen elastic identity collision},\\
 U_k&=\mathcal B_{1,k}(X_{C,k},V_k),\\
 m_k&=X_{C,k}+t(1+c)U_k,\qquad s=tq,\\
 W_{k+1}&=(Y_{k+1}-X_{C,k})/t-U_k,\\
 V_{k+1}&=\mathcal B_{2,k}(Y_{k+1},W_{k+1}).
\end{split}                                             \tag{NAMHR.25}
$$

The conditional transition density in the full ambient position is
$\varphi_s^{Na}(Y_{k+1}-m_k)$. Thus the original successful history
integral is exactly a sum over original matching/pattern paths with
density

$$
 \widehat W_{\mathbf C}(\rho,Z)
 \prod_{k=0}^{n-1}
 \rho^{Na/2}\varphi_s^{Na}
       (\sqrt\rho Z_{k+1}-m_k(\rho,Z_{\le k})).        \tag{NAMHR.26}
$$

All chronological graph/rank/pseudoinverse-deletion tests are fixed in
$Z$. Both force-dependent velocities in (NAMHR.25) are locally
absolutely continuous ridge functions on those fixed geometries.
The actual native quarter kick and Cayley map satisfy

$$
 \|\mathcal B_j(V)\|_{\max}\le\Lambda\|V\|_{\max},
 \quad
 L_\nu=\begin{cases}1,&h\nu\le4,\\1+h\nu/2,&h\nu>4,\end{cases}
 \qquad \Lambda=L_\nu^2 .                            \tag{NAMHR.27}
$$

In particular $\Lambda=1$ at the reference.
All one-sided ridge derivatives in (NAMHR.26) have primitive polynomial
position envelopes established below; no smooth entering-law premise is
required.
:::

:::{prf:proof}
The literal copy temporarily selects donor velocities as well as positions.
The configured collision then restores the frozen pre-copy velocity
$V_k$ on every accepted component:
the native identity branch writes `old.row(i)` from the pre-cloning
population, not from its donor-copied destination.
Unaccepted isolated rows already equal $V_k$.
Thus the zero-potential first B map gives $U_k$ at the cloned positions
and this original restored velocity.
A1 moves them to $X_{C,k}+tU_k$. The original O step gives
$W_{k+1}=cU_k+qG_k$, and A2 moves them to
$X_{C,k}+t(1+c)U_k+tqG_k$. Inverting this affine A2 relation gives
exactly (NAMHR.25), before the actually executed B2 map.
Its coefficient $s>0$ is scalar on the FULL ambient Gaussian array.
Hence the position density is the stated Gaussian, even though the
terminal velocity is its correlated deterministic native pushforward.
Iterating gives the chronological product density. This does not
postulate a joint $(x,v)$ Lebesgue density.

Every graph refresh sees only $Y_k$, its cloned row selection, or
$Y_{k+1}$. Their common dilation freezes the original principal-rank
test, tessellation and spectral deletion, by Chapter NAMR. The actual
volume/kernel weights retain their floors and become continuous
piecewise differentiable functions of $\rho$. The force fit uses
positive native regularization and its Cayley denominator is invertible.
Consequently the recursively reconstructed velocities have local
absolute continuity; explicit global derivative budgets are supplied
next.

Each original normalized weight row has mass at most one. If
$\tau\nu\le1$, its quarter map is a convex combination and cannot
increase the maximum row norm. Otherwise its triangle inequality bound
is $1+2\tau\nu$. The skew Cayley map preserves each row norm. Apply these
two quarter bounds with the separately recomputed force to prove
(NAMHR.27). No condition on a changing graph's column sum is needed for
this maximum norm statement.
:::

:::{prf:lemma} Primitive polynomial derivative bounds for both actual Boris maps
:label: lem-namhr-native-boris-profile

Let $R=|Z|$ be the Euclidean norm of the entire ambient position path.
The following explicitly configured nonnegative polynomials bound all
native force and curl derivatives on every chronological graph branch:

$$
\begin{split}
 D(R)&=2\sqrt{\rho_+}(1+R),\\
 C_w(R)&={d\over2\rho_-}
              +{G_+^*D(R)^2\over2\ell^2\rho_-},
       \qquad G_+^*=\sup_{\rho\in I}G_+(\rho),\\
 J_B(R)&=(2C_w(R)+\rho_-^{-1})D(R)^2,\\
 P_1(R)&={4\nu D(R)\over\eta_{\mathcal E}},\\
 P_0(R)&={16\nu C_w(R)D(R)+2\nu D(R)/\rho_-
                   \over\eta_{\mathcal E}}
       +{4\nu D(R)(1+\sqrt{\epsilon_{\mathcal E}})J_B(R)
                   \over\eta_{\mathcal E}^2},
       \qquad \eta_{\mathcal E}=\mathrm{MIN\_POSITIVE}_{\mathcal E}>0,\\
 P_n(R)&={\Lambda D(R)\over t}
                  \sum_{j=0}^{n-1}\Lambda^{2j},
       \qquad H_n(R)=1+\Lambda P_n(R)+D(R)/t,\\
 K_1(R)&=\Lambda[1+2b_{\rm C}H_n(R)P_1(R)],\\
 K_0(R)&=8\tau\nu L_\nu C_w(R)H_n(R)
                +2b_{\rm C}\Lambda P_0(R)H_n(R)^2 .
\end{split}                                             \tag{NAMHR.28}
$$

For a B-stage input velocity with maximum norm at most $H_n(R)$
and derivative maximum $J$, its original output has derivative
maximum at most $K_1(R)J+K_0(R)$. Both reconstructed B inputs and
outputs have maximum norms bounded by the displayed polynomials.
In particular define

$$
\begin{split}
 J_0(R)&=0,\\
 J^U_k(R)&=K_1(R)J_k(R)+K_0(R),\\
 J_{k+1}(R)&=K_1(R)
             [D(R)/(2t\rho_-)+J^U_k(R)]+K_0(R).
\end{split}                                             \tag{NAMHR.29}
$$

Then $\max_i|\partial_\rho^\pm V_{k,i}|\le J_k(R)$ and
$\max_i|\partial_\rho^\pm U_{k,i}|\le J^U_k(R)$.
Every constant and polynomial coefficient is a function of the complete
native register, even though the inverse-MIN-POSITIVE bounds can be
extremely large.
:::

:::{prf:proof}
All ambient edge displacements at either fitted B stage have norm at
most $D(R)$ and derivative equal to their value divided by $2\rho$.
The cached intrinsic edge square obeys
$0\le D_{ij}'\le D_{ij}/\rho$ and
$D_{ij}\le G_+^*D(R)^2$.
The actual raw viscous weight is
$k_{ij}=\exp[-D_{ij}/(2\ell^2)]v_j$.
Its volume floor gives $|(\log v_j)'|\le d/(2\rho)$.
Thus $|(\log k_{ij})'|\le C_w(R)$.
Normalize by the ORIGINAL $\max(\sum k,10^{-12})$, including its floor:

$$
 \sum_j|w_{ij}'|\le2C_w(R),\qquad \sum_jw_{ij}\le1 .
                                                               \tag{NAMHR.30}
$$

These bounds include an empty row. For initial velocity maximum $V$
and derivative maximum $J$, the actual viscous field satisfies
$\|F\|_{\max}\le2\nu V$ and
$\|F'\|_{\max}\le2\nu J+4\nu C_wV$.

Write the actual least-squares matrices as
$\mathfrak A_i=\sum_jw_{ij}(F_j-F_i)\Delta x_{ij}^T$ and
$\mathfrak B_i=\sum_jw_{ij}\Delta x_{ij}\Delta x_{ij}^T$.
They obey

$$
\begin{split}
 \|\mathfrak A_i\|&\le4\nu VD,\quad
 \|\mathfrak B_i'\|\le J_B,\\
 \|\mathfrak A_i'\|
 &\le4\nu DJ+(16\nu C_wD+2\nu D/\rho_-)V .
\end{split}
$$

The literal fit is
$J_i^{\rm fit}=\mathfrak A_i
(\mathfrak B_i+\zeta_i I)^{-1}$, with

$$
 \zeta_i=\max\{(\operatorname{tr}\mathfrak B_i/a)
                  \sqrt{\epsilon_{\mathcal E}},
                  \eta_{\mathcal E}\},\qquad
 \Omega_i=(J_i^{\rm fit}-(J_i^{\rm fit})^T)/2 .        \tag{NAMHR.31}
$$

These are the original mean-trace regularizer and half-skew convention.
The inverse has norm at most $\eta_{\mathcal E}^{-1}$, and its
derivative at most
$(1+\sqrt{\epsilon_{\mathcal E}})J_B/\eta_{\mathcal E}^2$.
Indeed $|\zeta_i'|\le\sqrt{\epsilon_{\mathcal E}}J_B$.
Hence $|\Omega_i'|\le P_1J+P_0V$.
The real-coordinate native LU solver rejects only a zero or nonfinite
pivot; the positive definite fitted matrix cannot have such a pivot.
No additional cutoff replaces this actual inverse.

For the native skew Cayley rotation
$\mathcal R=(I-b_{\rm C}\Omega)^{-1}(I+b_{\rm C}\Omega)$,
$\|\mathcal R\|=1$ and
$\|\mathcal R'\|\le2b_{\rm C}\|\Omega'\|$ because
$\|(I-b_{\rm C}\Omega)^{-1}\|\le1$.
The first quarter derivative has maximum
$L_\nu J+4\tau\nu C_wV$.
Rotate its output, of maximum at most $L_\nu V$, and apply the
second quarter, whose force is freshly evaluated. The resulting bound is

$$
 \Lambda J+8\tau\nu L_\nu C_wV
                 +2b_{\rm C}\Lambda V(P_1J+P_0V).
$$

Taking $V\le H_n$ proves (NAMHR.28).
Equation (NAMHR.25) gives
$\|W_{k+1}\|_{\max}\le D/t+\Lambda\|V_k\|_{\max}$.
Use (NAMHR.27) to get
$\|V_k\|_{\max}\le P_n$ throughout the horizon. Its derivative gives
$\|W_{k+1}'\|_{\max}\le D/(2t\rho_-)+J^U_k$.
Applying the just proved B-stage bound proves (NAMHR.29).
The regularizers, clamps and floors are continuous with the stated
one-sided derivatives; differentiating this finite recurrence is valid
at every one-sided branch. Its coefficients are polynomials in $R$,
not bounds on an unknown population law.
:::

:::{prf:theorem} Uncut history ridge/action variation at full interaction strength
:label: thm-namhr-full-force-ridge

In {prf:ref}`def-namhr-full-force`, the original unnormalized successful
action integral has both one-sided ridge derivatives at every interior
$\rho\in I$:

$$
\begin{split}
 \partial_\rho^\pm E_{\rm succ}H_n
 =E_D\int \sum_{\mathbf C}\Psi_{\rho,\mathbf C}(Z)
 \bigg[&(\widehat W_{\mathbf C})'^\pm\widehat H_{n,\mathbf C}
       +\widehat W_{\mathbf C}\widehat H_{n,\mathbf C}'^\pm\\
       &+\widehat W_{\mathbf C}\widehat H_{n,\mathbf C}
                       \sum_{k<n}\mathcal T_k^\pm\bigg]\,dZ,\\
 \mathcal T_k^\pm
 =-&{|G_k|^2-Na\over2\rho}
    +{1+c\over q}
       (\partial_\rho^\pm U_k-U_k/(2\rho))\cdot G_k ,
\end{split}                                             \tag{NAMHR.32}
$$

where $\Psi_{\rho,\mathbf C}$ is the path-specific Gaussian product in (NAMHR.26),
and $G_k=(\sqrt\rho Z_{k+1}-m_k)/s$.
In (NAMHR.32) each summand is integrated with its own conditional
product density; no common independent-position density is inserted.
The weight derivative is the exact native product derivative without
division by zero gate probabilities. The action derivative remains the
original terminal pulled-back derivative. All earlier native force,
curl, gate, rank, deletion and error branches are retained.

There is a fully primitive finite derivative budget. To specify it, set

$$
\begin{split}
 A&=\max(1,c\Lambda^2),\quad
 C_X=q[t+hn\Lambda^2A^n],\quad
 C_n={n\sqrt N\,C_X\over\sqrt{\rho_-}},\\
 B_{\rm full}(R)
 &={16NA_N\over s_c\ell_N^2}
       \left({J_*\over\epsilon_r}
             +{D(R)\over2\rho_-\epsilon_s}\right),\\
 J^U_{\max}(R)&=\max_{k<n}J^U_k(R).
\end{split}                                             \tag{NAMHR.33}
$$

Replace the finite maximum in the budget, if desired, by its polynomial
sum. If $\mathcal R_L$ has the original chi law with $L=nNa$ degrees
of freedom, one valid bound is

$$
\begin{split}
 |\partial_\rho^\pm E_{\rm succ}H_n|
 \le NJ_*+H_*E\bigg[
  &|\mathcal I_n|B_{\rm full}(C_n\mathcal R_L)\\
  &+n\left\{
     {\mathcal R_L^2+Na\over2\rho_-}
    +{(1+c)\sqrt N\over q}
       \left(J^U_{\max}(C_n\mathcal R_L)
                 +{\Lambda P_n(C_n\mathcal R_L)\over2\rho_-}\right)
                                  \mathcal R_L\right\}\bigg]<\infty .
\end{split}                                             \tag{NAMHR.34}
$$

Every term is an explicit finite polynomial Gaussian integral:
$E\mathcal R_L^j=2^{j/2}\Gamma((L+j)/2)/\Gamma(L/2)$.
The theorem includes the original reference $\nu=3,\beta_{\rm curl}=1$.
It does not posit a smooth native joint law or clip any innovation.
:::

:::{prf:proof}
In the fixed $Z$ coordinates the conditional Gaussian normalization
contributes $Na/(2\rho)$.
Differentiating its exponent, using
$m_k' = X_{C,k}/(2\rho)+t(1+c)U_k'$, gives exactly

$$
 {Na\over2\rho}
 -G_k\cdot(Z_{k+1}/(2\sqrt\rho)-m_k')/s
 =-{|G_k|^2-Na\over2\rho}
      +{1+c\over q}(U_k'-U_k/(2\rho))\cdot G_k .
$$

The final action has $|\widehat H'|\le NJ_*$ by Chapter NAMR.
Its actual geometric fitness depends on the current positions and their
sampled Euclidean distances; the changing reconstructed velocity does
not enter either EH reward or this distance tag. The proof of
(NAMHR.15)--(NAMHR.16) therefore applies at each fixed position-path
node with separation derivative at most $D(R)/(2\rho_-)$.
Summing the chronological product derivatives gives the exact bound
$|\mathcal I_n|B_{\rm full}(R)$, retaining the preceding accepted patterns.

We verify domination through the actual innovation law, rather than
assuming Gaussian positions. The maximum-norm B bound gives
$\|V_{k+1}\|_{\max}\le c\Lambda^2\|V_k\|_{\max}
                                  +\Lambda q|G_k|$.
Simultaneous copying does not increase position maxima and the two A
drifts give
$\|Y_{k+1}\|_{\max}\le\|Y_k\|_{\max}
                            +h\Lambda\|V_k\|_{\max}+tq|G_k|$.
Thus throughout the horizon
$\|Y_k\|_{\max}\le C_X\sum_{\ell<n}|G_\ell|$ and hence

$$
 |Z|\le C_n|G| .                                    \tag{NAMHR.35}
$$

This bound holds for every pattern and every $\rho\in I$, including
zero-mass patterns viewed as their conditional native updates.
The full Gaussian product density consequently has the common envelope

$$
 \Psi_{\rho,\mathbf C}(Z)\le
 \left({\rho_+\over2\pi s^2}\right)^{L/2}
                  \exp[-|Z|^2/(2C_n^2)] .           \tag{NAMHR.36}
$$

Conversely (NAMHR.25) and (NAMHR.28) bound each reconstructed residual
$|G_k|$ by a primitive linear polynomial in $|Z|$.
The derivative profiles in (NAMHR.28)--(NAMHR.29), the gate budget
and the differentiated Gaussian factors therefore have a common
integrable polynomial times (NAMHR.36) on $I$.
The sum over finitely many original patterns and matching contexts
introduces no unknown law bound. One-sided dominated differentiation
proves (NAMHR.32).

Under the original joint Gaussian innovations, (NAMHR.35) bounds these
nonnegative polynomial profiles by their evaluations at
$C_n\mathcal R_L$. Each block norm is at most $\mathcal R_L$.
For the action and Gaussian score terms the weights and their own
conditional densities give exactly the original joint history law.
For a gate derivative first expand its chronological product derivative
at a particular step. Bound the future action by $H_*$ and integrate
every subsequent conditional pattern and transition to one. The current
signed pattern derivatives have total mass at most
$B_{\rm full}(|Z_{\le k}|)$; their conditional Gaussian transitions
likewise integrate to one. The remaining past probability law is the
original one and (NAMHR.35) bounds its polynomial moment by the
displayed chi integral. Thus no derivative term is divided by a
zero gate and no different-pattern position densities are replaced by
a common one.
All three terms are bounded by exactly
(NAMHR.34), including the $\sqrt N$ conversion from a maximum row
derivative to an ambient array norm. This proves finiteness with every
parameter shown.

Native pre-O graph/resource errors have the same fixed scaled graph
branches in this coordinate proof. The real-coordinate fitted inverses
are everywhere defined by (NAMHR.31) and the skew Cayley bound.
A failed history remains its original stopped record and contributes
nothing to the unnormalized successful-action integral. For the
full-dimensional proof integral one may attach independent dummy
Gaussian position transitions after stopping; their integral is one.
Their density/moment bounds are weaker than the displayed ones and
their action contribution is zero. No successful-law normalization or
boundary interpretation is introduced.
:::

:::{prf:corollary} Full native viscosity and curl force-score action response
:label: cor-namhr-native-force-response

In the same complete absolute-metric register, vary either of the existing
fields $\zeta=\nu$ or $\zeta=\beta_{\rm curl}$ in a finite admissible
interval, while all metric, temperature, initial and other configuration
fields remain fixed. The two one-sided derivatives are those at interior
admissible coefficients; at coefficient zero only the admissible right
derivative is asserted. For this proof retain the chronological ambient
positions $Y_k$ in PHYSICAL coordinates rather than dilating them.
Then the original uncut successful-action response is

$$
 \partial_\zeta^\pm E_{\rm succ}H_n
 =E_{\rm succ}\left[
     H_n\,{1+c\over q}\sum_{k<n}
                        \partial_\zeta^\pm U_k\cdot G_k\right].
                                                               \tag{NAMHR.37}
$$

The original action, geometric gates, all rank and metric cutoff tests
have no direct $\zeta$ derivative at fixed positions.
All coupling through the actual changed force history is retained by
the recursively reconstructed $U_k$ and the displayed native mean score.
This includes the positive reference coefficients and their full
selection feedback; it does not identify the allocated action with the
pointwise likelihood score.

Explicit derivative profiles may be computed without a native-law
regularity hypothesis. Let $\nu_+,\beta_+$ be the finite interval
suprema, replace $L_\nu,\Lambda$ by their uniform bounds at $\nu_+$,
and put $b_{{\rm C},+}=\beta_+h/4$ in the following profiles.
For physical path norm $R=|Y|$, set

$$
 D_Y(R)=2(1+R),\qquad
 P_{n,Y}(R)={\Lambda D_Y(R)\over t}
                        \sum_{j=0}^{n-1}\Lambda^{2j},\qquad
 H_{n,Y}(R)=1+\Lambda P_{n,Y}(R)+D_Y(R)/t.
$$

One valid B-stage derivative budget $K_{1,Y}J+K_{0,\zeta,Y}$ is

$$
\begin{split}
 K_{1,Y}(R)
 &=\Lambda[1+8b_{{\rm C},+}\nu_+D_Y(R)H_{n,Y}(R)/\eta_{\mathcal E}],\\
 K_{0,\nu,Y}(R)
 &=4\tau L_\nu H_{n,Y}(R)
              +8b_{{\rm C},+}\Lambda D_Y(R)H_{n,Y}(R)^2/\eta_{\mathcal E},\\
 K_{0,\beta,Y}(R)
 &=8\tau\Lambda\nu_+D_Y(R)H_{n,Y}(R)^2/\eta_{\mathcal E}.
\end{split}                                             \tag{NAMHR.38}
$$

The actual velocity derivative recurrence is
$J_0=0$, $J^U_k=K_{1,Y}J_k+K_{0,\zeta,Y}$ and
$J_{k+1}=K_{1,Y}J^U_k+K_{0,\zeta,Y}$.
Put $C_{n,Y}=n\sqrt N\,q[t+hn\Lambda^2A^n]$ with
$A=\max(1,c\Lambda^2)$. The fully primitive finite response bound is

$$
 |\partial_\zeta^\pm E_{\rm succ}H_n|
 \le {H_*n(1+c)\sqrt N\over q}
       E[J^U_{\max}(C_{n,Y}\mathcal R_L)\mathcal R_L]<\infty .
                                                               \tag{NAMHR.39}
$$

The expectation is again the explicit original chi-polynomial integral.
:::

:::{prf:proof}
At fixed chronological physical positions every original geometric
refresh and its metric is fixed. The fitness reward and default sampled
Euclidean diversity depend only on those positions. Thus the pattern
probabilities have no direct viscosity/curl derivative in these
coordinates, even though their unconditional law changes when the
native dynamics changes.

The weights in each fitted B map are now fixed as well.
For viscosity differentiation the quarter-map derivative from a
parameter-independent velocity adds
$\tau\sum_jw_{ij}(v_j-v_i)$ of maximum at most $2\tau V$.
The curl derivative from that same direct coefficient change has
maximum at most $4D_YV/\eta_{\mathcal E}$; its derivative through an
input velocity of derivative maximum $J$ adds at most
$4\nu_+D_YJ/\eta_{\mathcal E}$.
Applying the two quarters and norm-one Cayley derivative gives exactly
the first and second lines of (NAMHR.38).

For curl-coefficient differentiation, the quarter maps have no direct
coefficient term. The Cayley scale's derivative is $\tau$, and the
original curl has norm at most $4\nu_+D_YV/\eta_{\mathcal E}$.
Its input-velocity response is the same just computed. The Cayley
derivative and both quarters give the third line of (NAMHR.38).
These bounds retain the original fitted inverse; they need no metric
or graph derivative because those actual positions and parameters are
fixed in this coordinate proof.

In (NAMHR.25), $W_{k+1}'=-U_k'$ at fixed positions.
This gives the stated recurrence. The differentiated Gaussian mean is
exactly $t(1+c)U_k'$, so its physical-position score is
$(1+c)U_k'\cdot G_k/q$, with no direct action or gate term.
The original maximum-norm path estimate now yields
$|Y|\le C_{n,Y}|G|$. The same polynomial/Gaussian domination used in
(NAMHR.35)--(NAMHR.36) proves differentiation of the complete
pattern-specific history density, including native stopped/error
outcomes in the unnormalized successful integral. Its score bound
is (NAMHR.39). This proves the original force/action weak response.
:::

## 6. Algorithm binding and remaining scope

The full thermal and original weak source responses hold at the
reference interaction coefficient $\nu=3$, with the original curl,
every scheduled positive gate, all earlier rank/deletion interfaces and
native error records. Their uncut action integration is closed by the
existing absolute-metric pointwise bound, without a stationary-law or
joint-regularity assumption.

The simple pure-ridge source formula is proved in the included $\nu=0$
origin-start regime. The full ambient-position chronology then extends
the original uncut ridge/action response to every finite normalized graph
coefficient and the actual reference $\nu=3$. At positive coefficient it
retains the additional original force/curl response in (NAMHR.32), rather
than assigning the zero-force source formula to that kernel.
Nonorigin initial data, temperature-dependent initial laws, altered
refresh schedules, nonnormalized weights, additional noise/caps,
different landscapes or metric error policies retain their actual
different coordinate maps and are not assigned these specific formulas.

Relative-trace configurations keep the previously proved literal
rank-one divergent action integral; they do not inherit the absolute
action bound or its uncut response. The original action, volume and
isotropic thermostat are retained throughout. None of these finite-history
identities identifies the action with an Einstein continuum action or
constructs a stationary gravitational limit.

The consumed implementation sources are the original preset
`variants/einstein_hilbert.rs`, the graph Boris chronology in `kinetic.rs`,
the frozen elastic identity component collision in `cloning.rs`, the
positive-epsilon sample standardizers and distance regularizer in
`fitness.rs`, and the native geometry/metric/reward implementations
already bound in Chapters NCMA and NAMR. All formal conclusions concern
their stated real-coordinate arithmetic tag with the actual comparison
constants retained.
