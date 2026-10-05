# Native sampled-measurement response of the complete stationary instrument

(sec-nmr-register)=
## 1. Original measured array and a frozen-bulk comparison

:::{prf:definition} Complete native measured-response register
:label: def-nmr-register

Retain the complete real-coordinate dense active count register of
{prf:ref}`def-npb-register`, including every original current distance
companion, its bounded squashed feature map, the original alive reward
statistics, sampled diversity moments, shifted logistic powers, current
clone donors, clipped gates, simultaneous copying and whole-component
Haar law. All original downstream jitter, both dense kicks, OU, cap,
terminal spatial noise and matched color masks remain those of that
register. The input law is the proved stationary $\mu_*$; no Gaussian
law for its entering population is assumed.

For conditional calculations freeze an original entering $S$ with
at least $a_fN$ alive rows, where $a_f\in(m_0,a_0)$ is the already
verified fraction in (NPB.20). Keep its original independently sampled
measurements $D_i$, diversities $s_i$ and redundant moments
$q_i=(s_i,s_i^2)$. Let $\bar m(S)=E[\widehat m\mid S]$ as in
(NPB.5). Only for the comparison below, evaluate the ORIGINAL fitness
formula $f_i(s_i;m)$ of (NPB.7) at $m=\bar m$.
The actual executed value is always $m=\widehat m$; restoring it is
proved below. This comparison changes no simulated gate or noise.

For a bounded compact-preparation test $f$, $|f|\le B$, write

$$
\mathcal C_{N,f}(S,D;m)
 =E_{\Omega,O}\left[\frac1N\sum_i f(W_i)\mid S,D;m\right]
 =\frac1N E_\Omega\sum_{C\in\Gamma} b_C,
\qquad
b_C=E_O\sum_{i\in C}f(W_i(O)).
\tag{NMR.1}
$$
All original donor/gate outcomes in $\Omega$ and Haar variables in $O$
are integrated in this exact conditional mean. Their original copied
positions, velocities and accepted masks remain in $W_i$.
No donor distribution is conditioned on a newly invented fitness law.
Finite-array and characteristic inequalities below hold for bounded
Borel $f$. Every weak-limit/bracket statement uses bounded CONTINUOUS
$f$ on the full compact marked preparation space, or a separately
proved convergence of its actual finite component Haar means.
The native compact color influence (NPB.34) is continuous and satisfies
this restriction.

Use the primitive single-component constants $\theta,M_{\rm comp}$
of (NPL.2), put $C_D=1/(\kappa_Ca_f)$ and $g=G_{\rm live}$, and set

$$
\begin{gathered}
r=2gC_D<1,\qquad
J_k=4C_D^2\left\{2^k+\frac{3^kk!}{(1-r)^{k+1}}\right\},\\
K_1=2BM_{\rm comp}/\theta,\qquad
K_2=4BM_{\rm comp}J_1/\theta .
\end{gathered}
\tag{NMR.2}
$$
The inherited $\varrho=2e g/(\kappa_Cm_0)<1$ implies $r<1$.
All higher-alive exceptions stay in the original chain and have their
proved exponentially small probability (NPB.20).
:::

:::{prf:lemma} Primitive component path bound for shared original donor addresses
:label: lem-nmr-path

Freeze any finite collection of measured fitness arrays which differ
only in the measured values of a fixed finite set of rows, with the
SAME bulk $m$, entering $S$ and original clone-donor distributions.
Couple all their original outcomes with the same clone-donor address
and the same acceptance uniform at each recipient. Their union graph
has at most one outgoing donor edge per row and the same mandatory
dead leaves. For every distinct pair of rows $i,j$ and integer $k\ge1$,

$$
E[|\mathcal U(i)|^k]\le k!M_{\rm comp}/\theta^k,
\qquad
E[|\mathcal U(i)|^k\mathbf1_{\{j\in\mathcal U(i)\}}]
 \le \frac{k!M_{\rm comp}J_k}{\theta^kN}.
\tag{NMR.3}
$$
The assertion also holds after deleting any specified outgoing choices.
The union can contain a live directed cycle; no false union-forest
assertion is needed.
:::

:::{prf:proof}
The union accepts a proposed live edge when the shared uniform is
below the maximum of its acceptance probabilities in these arrays.
That maximum is at most $g$. Thus every specified live edge has
probability at most $gC_D/N$, and recipient outcome blocks are still
independent after the arrays are frozen. The union has one donor
address at each recipient. Dead recipients always retain their one
current alive donor; no dead row is eligible as a donor.

A connected set of $\ell$ live vertices contains an undirected
spanning tree. Enumerating its orientations with distinct source rows
gives exactly the tree union bound (NPL.4), even if other union edges
create a cycle. An orientation requiring two outgoing edges from one
row has probability zero. After freezing the live graph, mandatory
dead donors have their same independent bounded probabilities.
The dead-leaf exponential-moment calculation of (NPL.2)--(NPL.4)
therefore applies without alteration, and gives the first inequality.
Deleting choices can only remove union edges and preserves that bound.

If $j\in\mathcal U(i)$, choose a simple undirected path from $i$ to
$j$ of length $\ell$. An internal vertex cannot be dead, because a
dead vertex has degree one. There are at most $N^{\ell-1}$ internal
label sequences and at most $2^\ell$ path orientations. For two live
endpoints the joint edge probability is at most
$(gC_D/N)^\ell$. With one dead endpoint remove one power of $g$;
with two dead endpoints remove two, and necessarily $\ell\ge2$.
All specified source addresses are distinct. Conditional on the path
choices, their remaining deleted graph uses independent untouched
recipient blocks. The entire component lies in the union of the
$(\ell+1)$ deleted-graph components of its path vertices. Jensen and
the first bound give its conditional $k$th moment at most
$(\ell+1)^k k!M_{\rm comp}/\theta^k$.
This remains true when some path source had its outgoing choice
already deleted: that path has probability zero.

Since $C_D\ge1$, the sum of all three endpoint cases is bounded by

$$
\frac{k!M_{\rm comp}}{\theta^kN}\,
 4C_D^2\left\{2^k+\sum_{n\ge0}(n+3)^k r^n\right\}.
$$
Finally $(n+3)^k\le3^k(n+1)^k
\le3^kk!\binom{n+k}{k}$, whose generating sum is
$(1-r)^{-k-1}$. This proves the second bound with the displayed
$J_k$. This is a path estimate for the original shared addresses;
it neither gives independent components nor removes correlations.
:::

:::{prf:theorem} Exact measured-row mixed differences of the original preparation mean
:label: thm-nmr-mixed

At fixed $S$ and $m=\bar m$, let $\Delta_i$ replace only the original
row-$i$ distance measurement by another allowed value. For $i\ne j$,

$$
|\Delta_i\mathcal C_{N,f}|\le K_1/N,\qquad
|\Delta_i\Delta_j\mathcal C_{N,f}|\le K_2/N^2.
\tag{NMR.4}
$$
These are deterministic bounds on the exact conditional expectation;
the original measurement values may share companion coordinates.
Dead-row measured values give zero difference because their executed
fitness is zero and their revival gate is independent of that value.
:::

:::{prf:proof}
Use the original address/uniform coupling of the preceding lemma for
the two or four compared measurement arrays. At fixed bulk moments,
changing measurement $i$ changes only fitness $i$. An altered accepted
edge consequently has $i$ as its recipient or as its chosen donor.
All changed preparations and component Haar means therefore lie
inside the union component of $i$. The summed Haar-mean difference
is at most $2B|\mathcal U(i)|$.
Its expectation and (NMR.3) give the first bound.

For two changes, all altered edges are incident to $i$ or $j$.
If their union components are disjoint, the preparation and Haar
mean contributions on the two components change separately; the
mixed difference cancels exactly. If they meet, the four summed
component contributions have mixed difference at most
$4B|\mathcal U(i)|$. The second inequality of (NMR.3) gives the
second bound. This cancellation uses the original simultaneous
copying and complete component Haar mean: it does not substitute
an independent-row collision rule.
:::

(sec-nmr-hajek)=
## 2. Actual independent-measurement projection and its native bracket

:::{prf:definition} Original measured-row projection
:label: def-nmr-projection

Conditional on entering $S$, let

$$
\begin{gathered}
Z_{N,f}=\sqrt N\{\mathcal C_{N,f}(S,D;\bar m)
                          -E_D\mathcal C_{N,f}(S,D;\bar m)\},\\
h_{i,N}(s)=N\{E[\mathcal C_{N,f}(S,D;\bar m)\mid s_i=s]
                             -E_D\mathcal C_{N,f}(S,D;\bar m)\},\\
Q_{{\rm meas},N}=\frac1N\sum_i
                           \operatorname{Var}_{Q_i^D}h_{i,N}(s_i).
\end{gathered}
\tag{NMR.5}
$$
The function $h_{i,N}$ integrates every other ORIGINAL measurement,
clone donor, gate and whole-component Haar mean; $|h_{i,N}|\le K_1$.
:::

:::{prf:theorem} Derived Gaussian law for the full frozen-bulk measured center
:label: thm-nmr-frozen-clt

Under the verified native component regime,

$$
Z_{N,f}=\frac1{\sqrt N}\sum_i h_{i,N}(s_i)+R_{N,f},
\qquad E[R_{N,f}^2\mid S]\le K_2^2/(4N).
\tag{NMR.6}
$$
For every real $u$,

$$
\left|E[e^{iuZ_{N,f}}\mid S]
                       -e^{-u^2Q_{{\rm meas},N}/2}\right|
\le\frac{|u|K_2}{2\sqrt N}
 +\frac{|u|^3K_1^3}{6\sqrt N}
 +\frac{|u|^4K_1^4}{8N}.
\tag{NMR.7}
$$
No conditional independent fitness assumption is made at the actual
sampled bulk moments; independence is used only for the ORIGINAL
measurements in this frozen-bulk comparison.
:::

:::{prf:proof}
For completeness use the finite independent-coordinate orthogonal
expansion. For each subset $A$ of row addresses apply
$\prod_{i\in A}(I-E_i)\prod_{j\notin A}E_j$ to the centered
$Z_{N,f}$, where $E_i$ integrates its original independent measurement.
The resulting functions $Z_A$ are pairwise orthogonal by conditioning
on a coordinate in their symmetric difference, and sum to $Z_{N,f}$.
Their singleton terms are exactly $h_{i,N}/\sqrt N$.
An independent replacement in each of coordinates $i,j$ gives

$$
E[(\Delta_i\Delta_j Z_{N,f})^2\mid S]
 =4\sum_{A\supset\{i,j\}}E[Z_A^2\mid S].
$$
Summing over unordered pairs bounds the nonsingleton residual variance
by one-quarter that sum. From (NMR.4),
$|\Delta_i\Delta_j Z_{N,f}|\le K_2/N^{3/2}$.
This gives the stated safe $K_2^2/(4N)$ bound. The singleton terms
are centered independent functions of the SAME actual donor draws.
Apply the elementary independent-sum characteristic bound (NPB.9)
with bound $K_1$, and use
$|e^{iu(x+r)}-e^{iux}|\le |u||r|$ and Cauchy--Schwarz.
This proves (NMR.7) directly from the finite original product law.
:::

:::{prf:definition} Exact original marked measured-response integral
:label: def-nmr-local-response

For an admitted limiting input law $\mu$, freeze only its actual
population reward statistics and diversity bulk $\bar m(\mu)$.
For a root entering row $z$, draw two ORIGINAL diversity values
$s,s'$ with current law $Q_\mu^D(z)$, sharing its physical entering
coordinates. All other rows retain their original sampled type marks.
Couple the two complete accepted graphs with the SAME original clone
donor and uniform at each recipient. Their finite rooted union limit
has the probability bounds and moment (NMR.3).
Let $\Psi_f(z,s,s')$ be the difference of the summed ORIGINAL
component Haar means on this root union, with $s$ minus $s'$.
Define

$$
\begin{gathered}
h_{\mu,f}(z,s)=E_{s',\text{native rooted union}}
                                     \Psi_f(z,s,s'),\\
Q_{\rm meas}^{\rm frozen}(\mu;f)
 =\int\operatorname{Var}_{Q_\mu^D(z)}h_{\mu,f}(z,s)\,d\mu(z).
\end{gathered}
\tag{NMR.8}
$$
The same sampled measurement drives the fitness and every incoming
or outgoing gate involving this root. Mandatory dead leaves and the
actual donor reweighting remain in the native rooted union.
:::

:::{prf:theorem} Deterministic native measured-response bracket
:label: thm-nmr-frozen-bracket

For original admitted FULL MARKED entering empirical laws
$\mu_N\to\mu$, with their original alive reward/diversity moment
and fitness statistics converging to the actual population statistics,
and bounded continuous $f$ on the full compact marked preparation
space,

$$
Q_{{\rm meas},N}\longrightarrow
                         Q_{\rm meas}^{\rm frozen}(\mu;f).
\tag{NMR.9}
$$
Consequently the original frozen-bulk conditional center has the
Gaussian limit with this bracket, stable relative to the full entering
state. In the actual stationary reset count regime its input law is
$\mu_*$, with the exponentially small higher-alive exception retained.
:::

:::{prf:proof}
Write the finite projection $h_{i,N}$ as the expectation of the summed
component difference obtained by replacing root measurement $s'$ by
$s$, with all other original measurements unchanged. This is precisely
the finite version of (NMR.8). Conditional on frozen marked row types,
incoming addresses have probabilities at most $gC_D/N$ and donor
probabilities at most $C_D/N$. Their finite marked exploration has the
native incoming Poisson limit, as proved in the full two-copy argument
(NPB.29)--(NPB.31). Here each recipient uses a SHARED single clone
donor address and its same acceptance uniform, so the root gate
changes preserve that correlation in the limiting point process.
The continuous positive donor kernels have denominator at least
$\kappa_Ca_f$; positive-part gates are continuous even at ties.
The independent measured row marks converge to their actual sampled
law by their bounded independent empirical variance. All self-excluded
addresses keep their $O((\kappa_Ca_fN)^{-1})$ correction.

At a fixed finite exploration size the original complete type kernels
therefore converge, including copied marks and component Haar means.
The uniform exponential component bound removes that restriction in
$L^2$. For squared projections use two independent copies of the
rest of the rooted experiment given the SAME root $z,s$; this is the
finite conditional square identity and retains shared root marks.
Dominated convergence on the original uniform-root input and donor
laws proves (NMR.9). The exact measurement moment convergence follows
the same argument jointly, with its original redundant $q$ retained.
Together with (NMR.7) this gives stable conditional convergence.
Stationary input identification and its true high-alive exception are
the already proved native stationary results, rather than a presumed
Gaussian input.
:::


(sec-nmr-normalizer)=
## 3. Restoring the ORIGINAL sampled global normalizer

:::{prf:definition} Primitive original gate-response budgets
:label: def-nmr-gate-response

Use the exact original fitness derivative constants $L_f,E_f$ in
(NPB.11), the positive denominator $E_0=F_*+\epsilon_c$, and the
ORIGINAL clone scale $s_c$. In the derived regime $g<1$, every
well-defined admissible common moment value has live gate

$$
\pi_{ij}(m)=\frac{(f_j(s_j;m)-f_i(s_i;m))_+}{s_c(f_i(s_i;m)+\epsilon_c)}.
\tag{NMR.10}
$$
The upper clipping branch is inactive by the already derived $g<1$;
the positive-part branch stays literal. Dead revival gates have no
moment derivative. Define the finite primitive budgets

$$
\begin{gathered}
G_1=\frac{2L_f}{s_cE_0}+\frac{gL_f}{E_0},\\
G_2=\frac{2E_f}{s_cE_0}+\frac{4L_f^2}{s_cE_0^2}
            +\frac{gE_f}{E_0}+\frac{2gL_f^2}{E_0^2},\\
J_1^+=4C_D^2\{4+5/(1-r)^2\},\qquad
K_R=4BM_{\rm comp}G_1/\theta,\\
K_{RR}=\frac{4BM_{\rm comp}}\theta G_2
 +\frac{16BM_{\rm comp}}\theta(J_1^++4C_D)G_1^2.
\end{gathered}
\tag{NMR.11}
$$
For strict live fitness order put

$$
\begin{aligned}
\dot\pi_{ij}(m)&=\mathbf1_{\{f_j>f_i\}}
 \left\{\frac{\nabla f_j-\nabla f_i}{s_c(f_i+\epsilon_c)}
       -\frac{(f_j-f_i)\nabla f_i}{s_c(f_i+\epsilon_c)^2}\right\},\\
D_{N,f}(S,D;m)&=\frac1N\sum_{i,j:\ a_i=a_j=1}Q_i^C(j)
       \dot\pi_{ij}(m)
 E_{\Omega_{-i}}[B_f(\Omega_{-i},i\to j)
                       -B_f(\Omega_{-i},i\to\varnothing)],\\
B_f(\Omega)&=\sum_{C\in\Gamma(\Omega)}b_C.
\end{aligned}
\tag{NMR.12}
$$
The displayed strict indicator defines the chosen zero branch at a
finite tie; no differentiability at every finite tied array is assumed.
Self proposals have identical fitness, gate zero and response zero.
The actual current clone-donor law $Q_i^C$ is unchanged by $m$.
:::

:::{prf:theorem} Complete primitive normalizer response with its original kink band
:label: thm-nmr-normalizer-response

Let $m,m+v$ be any two admissible moments joined by their admissible
segment. Define the original live ordered-pair band

$$
b_N(\varepsilon;m)=\frac1N\sum_{i,j:\ a_i=a_j=1}Q_i^C(j)
           \mathbf1_{\{|f_i(s_i;m)-f_j(s_j;m)|\le\varepsilon\}}.
\tag{NMR.13}
$$
Self proposals may be retained in this upper bound, though they have
zero response. For the exact preparation conditional mean,

$$
\begin{aligned}
&|\mathcal C_{N,f}(S,D;m+v)-\mathcal C_{N,f}(S,D;m)
                                      -D_{N,f}(S,D;m)\cdot v|\\
&\hspace{8mm}\le \frac12K_{RR}|v|^2
       +\frac{8BM_{\rm comp}G_1}\theta
                                  |v|b_N(2L_f|v|;m),\qquad
|D_{N,f}|\le K_R.
\end{aligned}
\tag{NMR.14}
$$
This finite-array inequality retains every original gate kink. It
neither replaces the kink by smoothing nor assumes an anti-concentration
profile for a stationary law.
:::

:::{prf:proof}
On a fixed positive branch the quotient rule gives $|\nabla\pi|\le G_1$
and Hessian norm at most $G_2$. On the zero branch both vanish.
If its fitness order changes on the segment, its initial fitness
difference has magnitude at most $2L_f|v|$. The Lipschitz quotient
bound on each side and its continuous value zero at the boundary
therefore give

$$
|\pi_{ij}(m+v)-\pi_{ij}(m)-\dot\pi_{ij}(m)\cdot v|
 \le\tfrac12G_2|v|^2
     +2G_1|v|\mathbf1_{\{|f_i(m)-f_j(m)|\le2L_f|v|\}}.
$$
This also holds at a finite tie by its defined strict indicator.

Delete recipient $i$'s original outcome. Inserting its actual edge
$i\to j$ instead of null changes the full copied/Haar mean only on
its two deleted-component seeds $i,j$. The expected total size is
at most $2M_{\rm comp}/\theta$, so the expected $B_f$ change has
magnitude at most $4BM_{\rm comp}/\theta$.
This proves the response norm and the direct quotient remainder term.

Here is the mixed product-law cost needed for simultaneous gate changes.
Delete distinct recipient outcomes $i,k$. Independently draw their
ORIGINAL donors $j,l$ with probabilities $Q_i^C(j),Q_k^C(l)$,
each bounded by $C_D/N$. Compare edge versus null at these two
recipients in the remaining original graph. If the deleted-component
seed groups $(i,j)$ and $(k,l)$ are disjoint, the mixed summed Haar
mean cancels exactly. Otherwise its magnitude is at most $4B D$,
where $D$ is the total size of the four deleted-component seeds.
For a specified connecting path of length $\ell$, all four seeds lie
in at most $\ell+3$ remaining components after its source choices are
deleted. The path proof of (NMR.3) consequently gives the same bound
with $J_1^+$ in place of $J_1$. There are four cross-group seed pairs.
A random donor coincides with the opposite fixed or random seed with
probability at most $C_D/N$; its four-component mean size is at most
$4M_{\rm comp}/\theta$. Thus

$$
E[D\mathbf1_{\{\text{seed groups meet}\}}]
 \le\frac{4M_{\rm comp}}{\theta N}(J_1^++4C_D).
$$
This includes random shared-donor coincidences. Other paths use their
literal mandatory dead endpoint factors, although the two inserted
recipients and donors here are live.

Interpolate the entire recipient product law linearly between its
original outcome laws at $m$ and $m+v$. Every interpolated accepted
live edge has probability at most $gC_D/N$, so all preceding path
bounds remain uniform. The second product derivative is the sum over
ordered pairs $i\ne k$ of their edge/null mixed differences, weighted
by $Q_i^C(j)Q_k^C(l)\Delta\pi_{ij}\Delta\pi_{kl}$.
Since $|\Delta\pi|\le G_1|v|$, divide by $N$ for the empirical mean
and sum the $N(N-1)$ source pairs. The half-integrated second cost is
at most
$8BM_{\rm comp}(J_1^++4C_D)G_1^2|v|^2/\theta$.
Combining it with the direct gate remainder proves (NMR.14), with
exactly (NMR.11). Every interpolation in this proof compares original
outcome distributions; it is not an extra update step.
:::

(sec-nmr-atomless)=
## 4. Primitive atomless live fitness in the actual positive-noise phase

:::{prf:lemma} Original squashed-feature live fitness has no atoms in the positive PC37 phase
:label: lem-nmr-live-atomless

For the actual real-coordinate $d=3$ PC37 phase, keep the configured
squashed feature map $\phi_R(x)=x/(1+|x|/R)$, regularized distance
$\delta_D>0$, quadratic reward $-\lambda|x|^2/2$ with $\lambda>0$,
positive terminal spatial noise $s>0$, and positive finite reward and
diversity scales. Its configured positive shifted logistic powers have
$\eta_r,\eta_s,A_r,A_s,p_r,p_s>0$.
At the proved actual limiting bulk standardizers, the eligible LIVE
measured-fitness law is atomless. The actual live ordered pair law of
root and current clone donor consequently has

$$
P_{\rm native}(|F-F'|\le\varepsilon)\longrightarrow0
                                     \quad(\varepsilon\downarrow0).
\tag{NMR.15}
$$
Dead fitness retains its original atom zero; no dead pair enters this
live gate statement. If both powers are disabled, fitness is constant
on the live rows, (NMR.15) fails, and the literal live gate and every
normalizer/measured-row response are instead zero.
:::

:::{prf:proof}
The ORIGINAL final position Gaussian is drawn after both kicks and
velocity capping. Conditional on the complete pre-position-noise
record and all terminal velocities, each final position has a positive
Gaussian density. Restricting to the actual live box preserves spatial
absolute continuity on its interior. The already proved mean-field
fixed point $\mu_*$ is the law of this actual output, so its live
spatial conditional law has this property without any assumed unknown
stationary smoothness. Current distance and clone donor kernels only
multiply and normalize positive bounded weights; their denominators
have the proved positive floors. Such reweighting preserves null sets.

Freeze a distance-companion position $y$ and velocities $v,w$.
For a unit vector $n$ perpendicular to $\phi_{R_x}(y)$ and $x=rn$,
the ORIGINAL regularized squashed phase distance has square

$$
\left(\frac{R_xr}{R_x+r}\right)^2
 +|\phi_{R_x}(y)|^2
 +\lambda_{\rm alg}|\phi_{R_v}(v)-\phi_{R_v}(w)|^2+\delta_D^2.
$$
Its increasing distance approaches its finite limit with a strictly
positive $r^{-1}$ deficit. The positive diversity logistic power has
a finite strictly positive derivative at that limit, so its factor
approaches its positive limit from below with a positive $r^{-1}$
deficit. The shifted quadratic-reward logistic power approaches
$\eta_r^{p_r}$ from ABOVE with $O(e^{-c r^2})$, $c>0$.
Their product is therefore below its positive limiting value for all
sufficiently large $r$ and cannot be constant. The positive additive
floors are retained here; the reward factor has not been incorrectly
replaced by a factor tending to zero.

The full formula is real analytic on the connected spatial stratum
$\mathbb R^3\setminus\{0\}$; the configured regularized square root
has positive argument. Thus its restriction to any nonempty open part
of the live box is nonconstant by analytic continuation. To verify the
null-level assertion directly, at a zero of a nonzero analytic function
some finite-order derivative is nonzero. A derivative of one lower
order vanishes there with nonzero gradient. Its regular zero set is a
local smooth hypersurface and is Lebesgue-null. Taking the countable
union over derivative multi-indices covers all zeros; a point where all
derivatives vanish would make the analytic function identically zero
on the connected stratum. This proves that every fixed fitness level
has spatial measure zero. The omitted point $x=0$ is itself null.

The native limiting distinct-row measured pair law is a positive
bounded reweighting of two original row/distance-companion type laws
with their common deterministic bulk statistics. The individual fitness
law just proved is atomless, so their equal-fitness set has product
measure zero and remains null under that reweighting. Any finite shared
companion identities are retained in the finite kernels and have their
vanishing bounded-address collision correction. Even the mutual
identity pattern has the same positive symmetric diversity factor at
both rows, and a tie then requires equality of their quadratic reward
values, $|x_i|^2=|x_j|^2$, a spatially null hypersurface. A self clone
proposal is the separate deterministic zero-gate branch.
Continuity from above gives (NMR.15). This proves the property from
original terminal noise and formula parameters, including the actual
constant-fitness counterregime.
:::

:::{prf:theorem} Actual sampled global normalizer contributes its full same-donor influence
:label: thm-nmr-measured-clt

At the actual stationary positive PC37 phase, or its derived positive
parameter regime above, put $m=\bar m(S)$ and restore the original
$v=\widehat m-\bar m$. There is a deterministic finite native response
$D_f(\mu_*)=\lim D_{N,f}(S,D;\bar m)$, and

$$
\begin{aligned}
&\sqrt N\{\mathcal C_{N,f}(S,D;\widehat m)
                          -E_D\mathcal C_{N,f}(S,D;\widehat m)\}\\
&\quad=\frac1{\sqrt N}\sum_i
 \left\{h_{i,N}(s_i)+\frac{a_i}{a_N}
            D_f(\mu_*)\cdot(q_i-\bar q_i)\right\}+o_{L^1}(1).
\end{aligned}
\tag{NMR.16}
$$
Its Gaussian bracket is the deterministic native integral

$$
Q_{\rm meas}(\mu_*;f)
 =\int \operatorname{Var}_{Q_{\mu_*}^D(z)}
   \left[h_{\mu_*,f}(z,s)+
       \frac{a(z)}{\mu_*(a=1)}D_f(\mu_*)\cdot(s,s^2)\right]d\mu_*(z).
\tag{NMR.17}
$$
The two terms use the SAME original sampled donor, and their covariance
is retained. The bracket is derived, finite and nonnegative; it is not
asserted positive for every preparation test. Convergence is stable
relative to the complete original entering state.
:::

:::{prf:proof}
First identify the response limit. Its finite expression (NMR.12) is
the original uniform-root/clone-donor average of a bounded gate derivative
and the expected component change from forcing that original edge.
Delete that recipient outcome and retain both physical source/donor
seeds. Its finite marked exploration converges to the original rooted
law by the same actual kernel argument as (NMR.9). Component moments
give uniform integrability of its summed Haar mean. Its only remaining
discontinuity is the strict live fitness-order indicator; (NMR.15) makes
its limiting boundary null. The exact independent measured-type
empirical convergence, with the actual normalizer $\bar m(S)$, therefore
gives a deterministic limit $D_f(\mu_*)$ for this bounded expression.
Its norm is at most $K_R$. This is an explicit rooted response integral:
replace the finite root/donor average in (NMR.12) by its actual limiting
row, measured-mark and donor laws and its deleted two-seed component.

The original bounded independent moments give
$E[|v|^2\mid S]\le (S_b^2+S_b^4)/(a_fN)$ and uniform bounded fourth
moments of $\sqrt N v$. Thus $v\to0$ in probability and $\sqrt N v$
is uniformly square-integrable. For each fixed $\varepsilon>0$, the
finite ordered-pair band converges, except at its endpoint levels, to
the native limiting band. This follows from the actual marked-pair
kernel limit and bounded independent measurement variance, retaining
all original shared-source corrections. By (NMR.15), subsequently
letting $\varepsilon\downarrow0$ gives
$b_N(2L_f|v|;\bar m)\to0$ in probability. Since it is at most one,
Cauchy--Schwarz and the displayed uniform fourth moments imply

$$
E[\sqrt N|v|b_N(2L_f|v|;\bar m)]\longrightarrow0.
$$
The quadratic term in (NMR.14) has scaled expected value $O(N^{-1/2})$.
Moreover $D_{N,f}\to D_f(\mu_*)$ in probability with uniform bound
$K_R$, so its product with $\sqrt Nv$ converges in $L^1$ to zero after
subtracting $D_f(\mu_*)$. We obtain

$$
\sqrt N\{\mathcal C_{N,f}(\widehat m)-\mathcal C_{N,f}(\bar m)}
             =D_f(\mu_*)\cdot\sqrt N(\widehat m-\bar m)+o_{L^1}(1).
$$
Centering by its exact conditional expectation changes its remainder
by at most its $L^1$ norm. The leading normalizer term has conditional
mean zero. Combine this with (NMR.6), and use
$\sqrt Nv=N^{-1/2}\sum_i(a_i/a_N)(q_i-\bar q_i)$.
This proves (NMR.16). The leading terms are independent centered
functions of each original measured donor. Their bounded row influence
is at most $K_1+K_R\sqrt{S_b^2+S_b^4}/a_f$ times a safe factor two.
Their original joint covariance converges by the two-copy-root argument
for squared conditional projections, preserving the same $q_i$ inside
$h_i$. The elementary product characteristic bound then proves the
Gaussian law and (NMR.17). The true higher-alive exception and complete
Doob comparison have their already proved exponential budgets and do
not change this bounded-record limit or its exact conditional center.
:::


(sec-nmr-full-color)=
## 5. Complete nonlinear color fluctuation after conditioning only on the entering state

:::{prf:definition} Primitive measured-array small-field budgets
:label: def-nmr-nonlinear-budget

Retain the EXACT Gaussian-averaged source functional $\mathcal T$ and
influence $\varphi_\eta$ of (NPL.7)--(NPL.8), (NPB.34).
Let

$$
\begin{aligned}
\bar\Theta_{N,D}(m)&=E_{\Omega,O}[\Theta_N\mid S,D;m],\\
\Theta_{0,N}(S)&=E_D\bar\Theta_{N,D}(\bar m),\qquad
\widehat\Theta_{N,D}=\bar\Theta_{N,D}(\widehat m).
\end{aligned}
\tag{NMR.18}
$$
These are conditional probability measures on the ORIGINAL compact
prejitter preparation $W=(Y,I,w)$; no redundant coordinate is discarded.
Use the already proved $S,C_N,r_N,J,Q,T_H,D_H,b_H$ of
(NPL.12), (NPB.34). The original measurement moment bound is
$K_q=2\sqrt{S_b^2+S_b^4}/a_f$. Define

$$
\begin{gathered}
K_1^{(1)}=2M_{\rm comp}/\theta,\qquad
K_R^{(1)}=4M_{\rm comp}G_1/\theta,\qquad
K_M=2K_1^{(1)}+2K_R^{(1)}K_q,\qquad
A_M=2(K_1^{(1)})^2,\\
C_{W,M}=\sqrt{2d}T_H
     +D_H\sqrt{2\,3^{2d}A_M}+2D_HK_R^{(1)}K_q,\\
p_N=\max\{2,\lceil\log[2JQ(N+1)^4]\rceil\},\qquad
\varepsilon_{M,N}=eK_M\sqrt{p_N/N}+4S^4/N^2,\\
E_{M,N}=C_N\sqrt N\left\{\varepsilon_{M,N}^2
     +\varepsilon_{M,N}C_{W,M}N^{-b_H}+N^{-4}\right\}
                   +C_N\sqrt{8dN}\,N^{-4}\longrightarrow0.
\end{gathered}
\tag{NMR.19}
$$
All original finite parameter budgets and unbounded Gaussian tails
remain in these constants, including the actual color/phase/taper
budgets inside $S$ and the full B1-propagated B2 influence.
:::

:::{prf:theorem} Original measured preparation closes the full nonlinear small-field remainder
:label: thm-nmr-full-measured-linearization

Conditional on original entering $S$ in the higher-alive class,

$$
\begin{aligned}
\mathcal T(\Lambda_{\widehat\Theta_{N,D}})
 -\mathcal T(\Lambda_{\Theta_{0,N}})
 &=\int\varphi_{\Theta_{0,N}}\,d(\widehat\Theta_{N,D}-\Theta_{0,N})
                                                  +R_{M,N},\\
E_D[\sqrt N|R_{M,N}|\mid S]&\le E_{M,N}.
\end{aligned}
\tag{NMR.20}
$$
The actual complete measured conditional center satisfies

$$
E_D\left[\sqrt N\left|E[H_N\mid S,D]
             -\mathcal T(\Lambda_{\widehat\Theta_{N,D}})\right|\mid S\right]
                      \le E_N+E_{H,N},
\tag{NMR.21}
$$
where the inherited $E_N,E_{H,N}\to0$ are exactly the full downstream
and correlated collision-preparation budgets (NPL.12), (NPB.34).
Thus (NMR.20) linearizes the ORIGINAL nonlinear measured center at
its true $\widehat m$, rather than merely the empirical mean fitness.
:::

:::{prf:proof}
At fixed $\bar m$, (NMR.4) applies to every compact bounded test.
The finite conditional-mean measure changed by one measurement has
total variation norm, with dual bounded by one, at most
$K_1^{(1)}/N$. The independent original measurement Doob martingale
therefore has range at most that quantity per row.
The elementary martingale exponential bound, obtained by centering a
variable in an interval of that length and integrating its convex
exponential upper secant, gives sub-Gaussian tails. Integrating them
gives the safe moment bound

$$
\|\bar\Theta_{N,D}(\bar m)f-\Theta_{0,N}f\|_{L^p(D\mid S)}
                  \le2B K_1^{(1)}\sqrt{p/N},\qquad p\ge2.
$$
The same bounded independent-moment estimate gives
$\|\widehat m-\bar m\|_p\le2K_q\sqrt{p/N}$.
Integrate the product-law first derivative used in (NMR.14), without
its signed linear term. The complete preparation mean has
$|\bar\Theta_{N,D}(m+v)f-\bar\Theta_{N,D}(m)f|
 \le B K_R^{(1)}|v|$, for all admissible segments, including ties.
Combining gives

$$
\|\widehat\Theta_{N,D}f-\Theta_{0,N}f\|_p
                                  \le B K_M\sqrt{p/N}.
$$
This is a bound for the TRUE sampled global normalizer and requires
no independent actual standardized fitness law.

For any compact cell partition, independent resampling and the same
conditional-mean total variation bound give
$\sum_A\operatorname{Var}[\bar\Theta_{N,D}(\bar m)(A)\mid S]
 \le A_M/N$. To verify the summed bound, use
$\sum_A|\Delta_i\bar\Theta(A)|\le K_1^{(1)}/N$ and
$\sum_A|\Delta_i\bar\Theta(A)|^2\le(K_1^{(1)}/N)^2$
in the original coordinate Efron--Stein argument, then sum its $N$
rows. Width $T_HN^{-b_H}$ gives the original compact histogram
transport bound with constant
$\sqrt{2d}T_H+D_H\sqrt{2\,3^{2d}A_M}$.
The true-normalizer shift has compact transport at most
$2D_HK_R^{(1)}|\widehat m-\bar m|$.
Consequently

$$
E_DW_1(\widehat\Theta_{N,D},\Theta_{0,N})
                                      \le C_{W,M}N^{-b_H}.
$$

Apply the $L^{p_N}$ bound to every Gaussian-integrated normalized
bounded source-query field and its first query derivatives in the
proved full NPL grid. Markov at $e$ times the moment bound and the
same grid union give failure probability at most $(N+1)^{-4}$ and
first-field scale $\varepsilon_{M,N}$.
The SAME future jitter and OU coupling carries the compact transport
bound to their extended source laws.
The full B1/B2 second variation then has precisely the proved
small-field times transport remainder: its mixed first-query
variations are bounded by that field scale times this transport cost,
and its remaining quadratic first fields by the field scale squared.
All finite primitive chain-rule constants and original Gaussian
comparison tails are the inherited $C_N,r_N$ of (NPL.12).
This proves (NMR.20) with (NMR.19), by the same complete variational
argument used in (NPB.35). Original innovations were not clipped.

Finally, conditional on $S,D$, the ORIGINAL outcome/Haar preparation
has exact mean $\widehat\Theta_{N,D}$. The NPB nonlinear-preparation
expansion has conditional mean linear term zero and scaled expected
remainder $E_{H,N}$. The original downstream Gaussian-integrated
expansion has scaled conditional mean discrepancy at most $E_N$.
Its terminal alive indicator is already integrated by the exact
$a_s$ in $\mathcal T$. Taking these conditional expectations proves
(NMR.21), including both force evaluations and original masks.
:::

:::{prf:theorem} Every original innovation after the entering state has its full Gaussian population law
:label: thm-nmr-complete-innovation-clt

For the actual positive PC37 count phase and its verified primitive
regime, let $\varphi_* =\varphi_{\Theta_*}$ be the exact full compact
color influence. Then

$$
\sqrt N\{H_N-E[H_N\mid S]\}
 \ \Longrightarrow\ \mathcal N(0,V_{\rm innov}),
\qquad
V_{\rm innov}=V_{\rm down}(\Theta_*)
 +Q_{\rm clone}(\mu_*;\varphi_*)
 +Q_{\rm Haar}(\mu_*;\varphi_*)
 +Q_{\rm meas}(\mu_*;\varphi_*).
\tag{NMR.22}
$$
Convergence is stable relative to the entire ORIGINAL entering state.
Finite vectors keep the polarized covariance of each bracket, with
all same-donor/normalizer and B1/B2 dense-source correlations retained.
For the actual diagonal projector vector,
$\operatorname{tr}V_{\rm innov}\ge c_{\rm down}>0$ of (NPB.2).

The complete instantaneous stationary fluctuation is therefore EXACTLY
this conditional Gaussian innovation plus
$\sqrt N\{E[H_N\mid S]-EH_N\}$, the still-retained entering-population
feedback. A stationary Gaussian law for that latter block is not an
assumption of (NMR.22).
:::

:::{prf:proof}
The already proved actual stationary compact preparation law converges
to $\Theta_*$. At fixed $S$ the frozen conditional mean
$\Theta_{0,N}$ has this same limit: the original global-normalizer
shift has expected compact transport $O(N^{-1/2})$ by (NMR.19),
and conditional Jensen carries the existing full preparation
transport to its outcome/Haar/measurement mean. Thus
$\Theta_{0,N}\to\Theta_*$ in probability and
$\varphi_{\Theta_{0,N}}\to\varphi_*$ uniformly on the compact marked
space by the proved original Gaussian dominated-continuity argument.

Apply (NMR.20)--(NMR.21) and center by the exact conditional entering
expectation. The scaled remainder tends to zero in $L^1$.
The remaining linear compact test is
$\mathcal C_{N,\varphi_{\Theta_{0,N}}}(S,D;\widehat m)$
minus its exact measurement expectation. The theorem (NMR.16)--(NMR.17)
therefore gives its Gaussian law with bracket
$Q_{\rm meas}(\mu_*;\varphi_*)$. Varying convergent compact tests cause
no scaled bias: their centered singleton projections and actual
normalizer responses have variance bounded by a parameter constant
times the squared sup-norm test difference, and the mixed-difference
residual is $O(N^{-1})$ times that difference squared. This follows
directly from (NMR.2)--(NMR.14), and retains the full same-donor term.

The complete postmeasurement CLT (NPB.36) is stable relative to
$S,D$. Its limiting covariance is deterministic and equals the first
three brackets in (NMR.22). Its conditional characteristic function
therefore factors asymptotically from the just-derived measured-center
characteristic function. This supplies an independent Gaussian
innovation in the limit under the actual chronology, and proves
(NMR.22). The primitive positive trace is already a positive
semidefinite downstream part of that full sum.
The true higher-alive exception and bounded full Doob record comparison
vanish exponentially. Their averaged disintegrated conditional-kernel
and scaled conditional-center errors are exactly the already proved
ones in (NPB.36), so the full selected-law assertion has the same
scope. The final entering-state term is the exact conditional-mean
complement, not a suppressed stationary fluctuation.
:::


(sec-nmr-density)=
## 6. A primitive quantitative limiting fitness band from the original spatial refresh

:::{prf:theorem} Derived bounded conditional measured-fitness density
:label: thm-nmr-fitness-density

In the same original positive stationary count phase, suppose the
executed dimension is $d\ge2$, terminal spatial noise is $s>0$,
bounded squashed position/velocity radii are $R_x,R_v>0$, the
regularized distance is $\delta_D>0$, and the diversity channel has
$p_s,A_s,\eta_s>0$. Let $a_*\ge a_f$ be any proved lower bound for
$\mu_*(a=1)$, for example the ORIGINAL row floor $a_0$.
Keep all original kernel weights and set

$$
\begin{gathered}
h_s=(2\pi s^2)^{-d/2},\qquad
j_x=(1+R_D/R_x)^{-d-1},\qquad
S_{\max}=\sqrt{4R_x^2+4\lambda_{\rm alg}R_v^2+\delta_D^2},\\
Z_s=S_b/\sigma_s,\qquad
\ell_{\min}=e^{-Z_s}/(1+e^{-Z_s})^2,\qquad
\Sigma_s=\sqrt{\sigma_s^2+S_b^2},\\
J_F=\eta_r^{p_r}\,p_sA_s
 \min\{\eta_s^{p_s-1},(\eta_s+A_s)^{p_s-1}\}
                            \ell_{\min}/\Sigma_s>0,\\
C_S=\frac{h_s}{\kappa_Da_*j_x}
     |\mathbb S^{d-1}|S_{\max}(2R_x)^{d-2},\qquad
C_F=C_S/J_F<\infty.
\end{gathered}
\tag{NMR.23}
$$
For a disabled reward channel its factor $\eta_r^{p_r}$ is exactly
one. Conditional on ANY actual live owner state $z$, its ORIGINAL
population distance-donor measured fitness has a density bounded
by $C_F$. Consequently the original distinct live root/clone-donor
measured pair, with its independent original distance-companion marks
and its actual clone-kernel reweighting, satisfies

$$
P_{\rm native}(|F-F'|\le\varepsilon\mid z,z')
                       \le\min\{1,2C_F\varepsilon\}.
\tag{NMR.24}
$$
This is a primitive population-limit bound. It is not assigned to the
conditional finite-$N$ discrete donor array. When the diversity power
is disabled this particular density mechanism is absent; the actual
zero measured-normalizer response in (NMR.16) still holds.
:::

:::{prf:proof}
Let $V$ denote the original final velocity before the independent final
spatial Gaussian is drawn, and let $Y$ be the actual pre-position-noise
position. Conditional on the entire preceding record, $X=Y+sG$ and
$G$ is an independent standard Gaussian. Hence conditional on $V$,
$X$ is a mixture of Gaussians of variance $s^2I_d$, each with density
at most $h_s$. Therefore, as measures,

$$
\mu_*(dx,dv,a=1)\le
        h_s\mathbf1_D(x)\,dx\,\mu_{*,V}(dv).
$$
The assertion uses the actual terminal-stage independence and allows
arbitrary dependence between $Y$ and $V$. No Gaussian velocity law,
unknown stationary density or independent complete row is presumed.
The current eligible distance-donor kernel has numerator at most one
and denominator at least $\kappa_Da_*$, so the same product domination
holds for its donor measure with factor $(\kappa_Da_*)^{-1}$.

On $|x|\le R_D$, the ORIGINAL radial squashing map is a one-to-one
map with radial derivative $(1+|x|/R_x)^{-2}$ and tangential derivative
$(1+|x|/R_x)^{-1}$. Its determinant is
$(1+|x|/R_x)^{-d-1}\ge j_x$; the origin does not affect change of
variables. Thus the donor feature $u=\phi_{R_x}(x)$ has conditional
Lebesgue density at most $h_s/(\kappa_Da_*j_x)$, dominated by the
same probability velocity measure. Its support lies in $|u|\le R_x$.

Fix the owner state and donor velocity. Its literal regularized
separation is $S=\sqrt{|u-u_0|^2+B}$ with
$B=\lambda_{\rm alg}|\phi_{R_v}(v)-\phi_{R_v}(v_0)|^2+\delta_D^2$.
Polar coordinates about $u_0$ give radius $r=\sqrt{S^2-B}$ and
volume element
$|\mathbb S^{d-1}|S r^{d-2}\,dS$.
Here $r\le2R_x$ and $S\le S_{\max}$. Since $d\ge2$, this
is bounded by the coefficient in $C_S$, uniformly in donor velocity.
Integrating its dominating probability velocity measure proves the
conditional separation density bound $C_S$. Its configured constant
shift does not change this density.

At fixed owner reward factor and bulk statistics, the positive diversity
map is strictly increasing in the shifted separation. Its standardized
score has $|z_s|\le Z_s$ and denominator at most $\Sigma_s$.
The derivative of $(\eta_s+A_s\operatorname{logistic}z_s)^{p_s}$
is bounded below by the factor in $J_F$ divided by its owner reward
factor, and that reward factor is at least $\eta_r^{p_r}$.
Changing the one-dimensional variable therefore gives fitness density
at most $C_S/J_F=C_F$.

Conditional on the two original owner states, their actual population
measurement draws are independent and each retains its native donor
kernel; the current clone kernel reweights only those owner states.
Integrating one conditional density over an interval of length
$2\varepsilon$ proves (NMR.24). This conditional independence is the
proved population marked-donor law, with finite shared-address
corrections treated separately above. It is not an assertion of
independent finite swarm output rows.
:::

:::{prf:corollary} Original PC37 has an explicit finite quantitative live band
:label: cor-nmr-fitness-density-witness

For the complete actual PC37 parameters and its ORIGINAL row floor
$a_0=.787033001485797\ldots$, the bound (NMR.23) evaluates to

$$
\begin{gathered}
j_x=(1+\sqrt3)^{-4}\simeq .01794919243112271,\qquad
C_S\simeq1278.018650888587,\\
J_F\simeq2.499999999940021\,10^{-7},\qquad
C_F\simeq5.112074603677\,10^9,\qquad
\log C_F\simeq22.35487114775.
\end{gathered}
\tag{NMR.25}
$$
It is conservative but finite. Thus the limiting live fitness-band
coefficient is derived from the original spatial Gaussian, actual
squashed features and original positive maps, including all floors.
No presumed linear anti-concentration condition is introduced.
:::

:::{prf:proof}
Use $d=3$, $s=1$, $R_x=R_v=2$, $R_D=2\sqrt3$,
$\lambda_{\rm alg}=1$, $\delta_D=.001$, $\kappa_D=1$,
$S_b=\sqrt{32+10^{-6}}-.001$, and the original positive logistic
fields $\eta_r=\eta_s=A_s=p_r=p_s=1$, $\sigma_s=10^6$.
The cap-$V_0$ row floor is exactly the minimum retained in (NPB.37),
not the unshifted Gaussian centered-box probability. Substitution
in (NMR.23) gives (NMR.25).
:::
