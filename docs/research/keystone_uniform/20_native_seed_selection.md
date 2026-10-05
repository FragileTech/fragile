# Native central-source amplification using the existing establishment method

This note instantiates the signed, reward-aware discovery sum of
{prf:ref}`prop-slc-reward-aware-discovery`. Its conclusions concern one
complete update from the declared entering classes. The configured force,
noise, cap, donor laws and component collisions retain their original values.

:::{prf:definition} Native central and resident source record
:label: def-kuns-native-record

Use the Rastrigin record of {prf:ref}`def-kurc-record`: dimension three,
$D=[-2,2]^3$, $h=.04$, $\gamma=b_O=\rho=1$, $\nu=.3$,
$\sigma_J=\sigma_x=.1$, $V=2$, restitution $.5$, and either actual
dense viscous normalization. The original force and reward are

$$
F(x)=-2x-20\pi\sin(2\pi x),\qquad R(x)=-U(x).
$$

The positive reward and diversity factors are
$G(z)=2/(1+e^{-z})+.1$, their exponents are one, and both standardization
floors are $.1$. The original feature radii and both companion widths
are two, the phase-space feature weight is one, the diversity distance
floor is $.001$, and the clone gate is

$$
a(f,g)=\min\{1,(g-f)_+/(f+10^{-6})\}.
$$

Every measurement is retained, with its actual common empirical
normalizers. For the full position update put

$$
t=.02,\quad c=e^{-.04},\quad b=t(1+c),\quad \eta=bt,\quad
q^2=(1-e^{-.08})/2,\quad \tau^2=t^2q^2+.0004,
$$

$$
g(x)=(1-2\eta)x-20\pi\eta\sin(2\pi x).
\tag{KUNS.1}
$$

Its coordinate derivative is positive, with
$g'(x)\ge1-\eta(2+40\pi^2)>0$. After the original complete preparation,
the actual positions are
$x_i^+=g(X_i)+bU_i+\tau Z_i$, where
$U=W_Xv^{\rm col}$ and the $Z_i$ are independent Gaussians conditional
on the whole jittered preparation. Both viscous B stages and the cap
remain in the full transition; B2 and the cap do not change this position
observable. The central landing core is $C=[-1/16,1/16]^3\Subset D$.
:::

:::{prf:proposition} Exact single-source sum and its population limit
:label: prop-kuns-exact-single-seed

Let $N\ge2$ and take an all-alive entering swarm with $N-1$ rows at
$e_1$, one row at zero, and every stored velocity zero. This is the
existing two-site discovery configuration embedded in three dimensions.
Write $K_B^c$ for the number of central frozen positions after copying,
before jitter. Its exact coefficients are

$$
w=e^{-1/18},\quad p_D=p_C=\frac{w}{N-2+w},\quad
\Delta=\sqrt{4/9+10^{-6}}-.001,
$$

$$
S_r=\sqrt{(N-1)/N^2+.01},\quad
H_A=G\!\left(-\frac1{NS_r}\right),\quad
H_B=G\!\left(\frac{N-1}{NS_r}\right).
$$

For $k=0,\ldots,N-1$ put

$$
t_k=(k+1)/N,\quad s_k=\sqrt{t_k(1-t_k)\Delta^2+.01},\quad
f_k^-=G(-t_k\Delta/s_k),\quad
f_k^+=G((1-t_k)\Delta/s_k).
$$

Then

$$
\begin{aligned}
\mathbb E K_B^c-1
=p_C\sum_{k=0}^{N-1}{N-1\choose k}p_D^k(1-p_D)^{N-1-k}
\big[&(N-1-k)a(H_Af_k^-,H_Bf_k^+)\\
&+k\,a(H_Af_k^+,H_Bf_k^+)\big].
\end{aligned}
\tag{KUNS.2}
$$

Both reverse terms in the existing reward-aware formula vanish here:
the central row has both the larger reward factor and the largest
realized diversity mark. For every $N\ge2$,

$$
\mathbb E K_B^c-1\ge
\frac{2w}{(1+w)^2}>.4996,
\qquad
\lim_{N\to\infty}(\mathbb E K_B^c-1)=w.
\tag{KUNS.3}
$$

*Proof.* The squashed feature distance between zero and $e_1$ is $2/3$,
so both cross-site Gaussian weights are $w$. The central row must
measure a resident. The number of resident far measurements is exactly
$\operatorname{Bin}(N-1,p_D)$. Its $k+1$ far marks generate exactly
the shared diversity mean and variance in the display. The reward values
are $-1$ and zero, giving the displayed reward standardizer. Freezing
this complete measured array and averaging its independent donor/gate
draws gives (KUNS.2).

The acceptance from a near-measured resident is one. Here is a finite
certificate for that assertion, rather than a fitness-order assumption.
For $N\ge8$, put $u=1/N\le1/8$. Then

$$
H_B\ge G\!\left(\frac{7/8}{\sqrt{7/64+.01}}\right),
\qquad H_A\le G(0)=1.1.
$$

Let $S_*=(\Delta^2/4+.01)^{1/2}$ and $z_*=\Delta/(2S_*)$.
For $t_k\le1/2$, $f_k^+/f_k^-\ge G(z_*)/G(0)$.
For $t_k\ge1/2$, it is at least $G(0)/G(-z_*)$, which is no
smaller because $G(z_*)+G(-z_*)=2G(0)$ and AM--GM applies.
Consequently $H_Bf_k^+/(H_Af_k^-)>2.4939$.
Since $H_Af_k^-\ge.01$, this exceeds the exact clipping threshold
$2+10^{-6}/(H_Af_k^-)$.
For $2\le N\le7$, substitute the finitely many $k$ values in
$H_Bf_k^+-2H_Af_k^- -10^{-6}$; its minimum is greater than
$1.5568$, using the elementary interval recipe below.
For the actual far resident at $N=2$, the corresponding margin is
$(H_B-2H_A)G(0)-10^{-6}>.2896$, so its gate is also one.

For $N\ge3$, dropping only the remaining nonnegative far-resident
gain therefore gives

$$
\mathbb E K_B^c-1
\ge p_C(N-1)(1-p_D)
=\frac{w(N-1)(N-2)}{(N-2+w)^2}.
$$

For $a=N-2\ge1$, $wa(a+1)/(a+w)^2$ increases in $a$:
its derivative has the sign of $(2w-1)a+w>0$.
Its minimum is $2w/(1+w)^2$. At $N=2$ the gain is exactly one.

For the limit, the number of far resident marks has expectation
$(N-1)p_D\le1$, and is consequently tight. At each fixed $k$ the
near-resident gate tends to one. Splitting its expectation at a finite
mark cutoff and then removing that cutoff proves that the near gain
tends to $w$. The far gain is at most $p_C\mathbb E k\to0$.
This proves (KUNS.3), without replacing any sampled mark by its mean.
$\square$
:::

:::{prf:corollary} Complete native central landing from the single source
:label: cor-kuns-native-single-seed-landing

Define the unrestricted Gaussian coefficients

$$
\ell_{R,\tau}(m)=\Phi((R-m)/\tau)-\Phi((-R-m)/\tau),\quad R=1/16,
$$

$$
s_0=\ell_{R,\tau}(0)^3,\qquad
s_J=\left[\int_{\mathbb R}\ell_{R,\tau}(g(.1z))\phi(z)\,dz\right]^3.
\tag{KUNS.4}
$$

The actual complete update, with either viscosity normalization, satisfies

$$
\mathbb E Y_C'\ge s_0+s_J(\mathbb E K_B^c-1)>1.0929
\quad(N\ge2).
\tag{KUNS.5}
$$

The defining formulas give diagnostically
$s_0\simeq.9935190769786460$, $s_J\simeq.2211436674241822$,
and $\mathbb E K_B^c-1\simeq.9460819373977110$ at $N=128$.
Thus (KUNS.5) has value about $1.202739$ there and tends to
$s_0+ws_J\simeq1.20271202316732$ as $N\to\infty$.
The certified inequality uses $s_0>.993$, $s_J>.2$ and
$2w/(1+w)^2>.4996$, rather than these rounded diagnostics.

*Proof.* Every actual component has zero entering velocity and therefore
zero collision velocity for every Haar matrix. The first viscous term
vanishes in both normalizations even though its weights depend on all
jitters. The central seed never accepts a donor and is consequently
not jittered: its final position is $\tau Z$, giving $s_0$.
Each copied central offspring receives its original independent jitter;
its final position is $g(.1Z^J)+\tau Z$, giving $s_J$.
The source pattern is decided before these independent innovations.
Linearity gives (KUNS.5); all resident-source arrivals into $C$ are
additional nonnegative contributions. B2, cap and final marking remain
the actual configured stages, and $C\Subset D$ makes every counted
landing alive. The finite Gaussian lower sum below certifies $s_J>.2$
and $s_0>.993$. $\square$
:::

:::{prf:definition} A narrow two-region entering class
:label: def-kuns-band-class

Let $N\ge8$, $1\le K\le N/8$ and $\varepsilon=.001$.
Exactly $K$ all-alive entering rows lie in $B(0,\varepsilon)$,
the other $n_A=N-K$ lie in $B(e_1,\varepsilon)$, and every stored
velocity has norm at most $\varepsilon$. The configured cap remains
$V=2$; this is a restriction on the entering state, not a new cap or a
noise restriction. The two positions remain separated after squashing.

The following constants are all functions of this entering geometry and
the original parameters. Put

$$
\begin{gathered}
u_*=1/8,\quad B_R=(1+20\pi^2)\varepsilon^2,\quad
A_R^-=(1-\varepsilon)^2,\quad A_R^+=1+2\varepsilon+B_R,\quad
d_R=A_R^+-A_R^-,\\
S_R^+=\sqrt{.01+u_*(1-u_*)(A_R^+)^2+(d_R^2+B_R^2)/4},\\
H_B^-=G(((1-u_*)A_R^- -B_R)/S_R^+),\quad
H_A^+=G(d_R/.1).
\end{gathered}
\tag{KUNS.6}
$$

For the diversity features define

$$
\begin{gathered}
D_n=2\sqrt2\varepsilon,\quad D_-=2/3-D_n,\quad D_+=2/3+D_n,\\
y_n^-=.001,\quad y_n^+=.003,\quad
y_f^-=\sqrt{D_-^2+10^{-6}},\quad y_f^+=\sqrt{D_+^2+10^{-6}},\\
d_f=y_f^+-y_f^-,\quad L_f=5d_f,\quad
f_f^-=G(-d_f/.1),\\
w_-=e^{-D_+^2/8},\quad w_+=e^{-D_-^2/8},\quad w_n=e^{-D_n^2/8},\\
q_B=\frac{u_*}{u_*+(1-u_*)w_-},\quad
q_A=\frac{u_*w_+}{(3/4)w_n+u_*w_+},\quad
a_g=\frac{H_B^- -H_A^+}{H_A^++10^{-5}}.
\end{gathered}
\tag{KUNS.7}
$$

Finally put

$$
\begin{gathered}
L_B=5B_R,\quad
g_{ff}=\frac{2.1(L_B+L_f)}{H_B^-f_f^-+10^{-6}},\\
\beta_+=(7/8)w_-(3/4)w_n(1-q_B)a_g,\quad
\beta_-=q_Bq_A,\quad
\beta_{BB}=q_B[q_B+(1-q_B)g_{ff}].
\end{gathered}
\tag{KUNS.8}
$$
:::

:::{prf:theorem} Signed native central-source gain with every measurement retained
:label: thm-kuns-band-selection

Every entering state in {prf:ref}`def-kuns-band-class` satisfies

$$
\mathbb E K_B^c-K\ge(\beta_+-\beta_-)K>.3813K.
\tag{KUNS.9}
$$

Its expected number of central recipients copying a resident is at most
$\beta_-K$. Its expected number accepting another central donor is at
most $\beta_{BB}K$. The incoming gain from residents copying central
donors is at least $\beta_+K$. All constants are independent of $N$.

*Proof.* The inequality $1-\cos z\le z^2/2$ gives
$0\le U(x)\le B_R$ in the central ball. Writing $x=e_1+y$ in the
resident ball gives $A_R^-\le U(x)\le A_R^+$.
Decomposing the reward variance by the two entering groups, its within
variances are at most $B_R^2/4,d_R^2/4$, and its between-group term
is at most $u(1-u)(A_R^+)^2$. Since $u=K/N\le1/8$,
its patched standard deviation is at most $S_R^+$.
Every central reward minus the actual global mean is at least
$(1-u_*)A_R^- -B_R$, while every resident reward minus that mean is
at most $d_R$. Thus their actual reward factors obey
$H_B\ge H_B^-$ and $H_A\le H_A^+$, with the common normalizer
retained.

The squash is 1-Lipschitz in position and velocity. Within-region
feature distance is at most $D_n$, and cross-region distance lies in
$[D_-,D_+]$. This gives the near/far raw intervals in (KUNS.7).
They apply to each actual independently drawn measurement, not its
conditional average.

Every central far-measured row has higher fitness than every resident
near-measured row: their diversity factors are ordered by their raw
marks under the same measured mean and standard deviation. Its gate
is at least $a_g$, because the near factor is at least $.1$.
Every central far-measured row also beats every resident far-measured
row. Indeed the diversity factors differ by at most $L_f$, and any
far factor is at least $f_f^-$. Consequently the fitness gap is at least

$$
(H_B^- -H_A^+)f_f^- -H_A^+L_f>.8581.
\tag{KUNS.10}
$$

For central and resident near-measured rows the analogous gap is at
least $(H_B^- -H_A^+).1-H_A^+(.01)>.0718$.
Hence a reverse central-to-resident edge can occur only when the
central recipient measured near and the resident donor measured far.

For each resident the probability of a near measurement is at least
$(n_A-1)w_n/(N-1)\ge(3/4)w_n$.
For each central row its near-measurement probability is at most
$(K-1)/(K-1+n_Aw_-)\le q_B$, and its far probability is therefore
at least $1-q_B$. These are independent individual measurement draws
conditional on the entering state. Their favorable pair event has
probability at least $(3/4)w_n(1-q_B)$, and the proved gate lower
bound holds for every assignment of all the other measurements.
Thus their common random global normalizers are not decoupled.
Each eligible cross donor has probability at least $w_-/(N-1)$.
Summing over the $n_AK$ directed resident/central pairs and using
$n_A/(N-1)\ge7/8$ gives the incoming lower bound $\beta_+K$.

A resident's far-measurement probability is at most
$Kw_+/[(n_A-1)w_n+Kw_+]\le q_A$.
For a central recipient and a distinct resident donor, independence
of their two measurement draws bounds the reverse event by $q_Bq_A$;
averaging the separate original cloning proposal cannot increase this
uniform bound. This gives $\beta_-K$.

For the within-central accepted edges retain their two mark groups.
The reward-factor oscillation between any two central rows is at most
$L_B=5B_R$, from the global denominator floor and $G'\le1/2$.
Two far-measured central rows have fitness difference at most
$2.1(L_B+L_f)$, and fitness at least $H_B^-f_f^-$, so their
gate is at most $g_{ff}$.
No far-measured central recipient accepts a near-measured central donor.
To verify this sign, set

$$
S_Y^+=\sqrt{(y_f^+-y_n^-)^2/4+.01},\quad
Z=(y_f^+-y_n^-)/.1,\quad m_G=2e^{-Z}/(1+e^{-Z})^2.
$$

Its diversity-factor gap is at least
$m_G(y_f^- -y_n^+)/S_Y^+$, and therefore its total fitness gap
is at least

$$
H_B^-m_G(y_f^- -y_n^+)/S_Y^+-2.1L_B>.0071.
\tag{KUNS.11}
$$

The central cloning-proposal probability of another central row is
at most $q_B$, just as for the measurement role because the configured
widths coincide. Near-measured recipients may accept with probability
one, but their probability is at most $q_B$. Far-measured recipients
accept a central donor with probability at most $g_{ff}$.
This gives $\beta_{BB}K$. The expected central position count changes
only on cross-region accepted edges; subtracting its reverse bound
from the incoming bound proves (KUNS.9). Component Haar matrices never
change the frozen position-source identity. $\square$
:::

:::{prf:theorem} Complete native spatial amplification on the same entering band
:label: thm-kuns-band-complete-update

Every entering state in {prf:ref}`def-kuns-band-class` satisfies

$$
\mathbb E Y_C'\ge1.0387K.
\tag{KUNS.12}
$$

This is a complete native spatial-count estimate for both normalization
tags. It includes every copy, persistence, component rotation, Gaussian
innovation, dense first kick and terminal mark. The actual second kick
and cap are retained and do not modify this spatial observable.

*Proof.* The all-row entering velocity bound is $\varepsilon$. Every
actual collision component consequently satisfies
$|v_i^{\rm col}|\le2\varepsilon$ for every shared Haar draw.
At the original $t\nu=.006$, both first-kick matrices are stochastic,
even after all jitters are exposed, so $|U_i|\le2\varepsilon$.
Set $R_e=R-2b\varepsilon>0$ and

$$
\begin{aligned}
\underline s_0&=\ell_{R,\tau}(g(\varepsilon)+2b\varepsilon)^3,\\
\underline s_J&=
\left[\int_{\mathbb R}
 \ell_{R_e,\tau}(g(\varepsilon+.1z))\phi(z)\,dz\right]^3.
\end{aligned}
\tag{KUNS.13}
$$

For a persistent central source, monotonicity and oddness of $g$
give landing probability at least $\underline s_0$.
For an accepted central source, erosion by the entire possible
$bU_i$ shift leaves the original target cube inside $C$.
Its lower conditional landing probability then depends only on that
row's own jitter. The convolution of the even decreasing function
$\ell_{R_e,\tau}(g(x))$ with a centered Gaussian is minimized over
$|\mu|\le\varepsilon$ at $|\mu|=\varepsilon$, as in
{prf:ref}`lem-kurc-monotone-map`. Independent own coordinate jitters
therefore give $\underline s_J$. This bound holds even though $U_i$
depends on the whole jitter array. The finite lower sums below give
$\underline s_0>.993$ and $\underline s_J>.2$.

Existing central recipients persist, clone centrally, or clone to a
resident. Their respective expected contributions are bounded below by

$$
K\left[.993(1-\beta_-)-\beta_{BB}(.993-.2)\right].
$$

The incoming resident recipients add at least $.2\beta_+K$.
Adding gives

$$
\mathbb E Y_C'/K
\ge .993(1-\beta_-)-.793\beta_{BB}+.2\beta_+
>1.0387.
\tag{KUNS.14}
$$

The inequalities use conditional row landing probabilities after the
complete preparation and then linearity; no independence between
unprepared or completed output rows is assumed. The unrestricted
Gaussian laws remain unchanged. $\square$
:::

:::{prf:remark} Explicit finite certification and the next establishment obligation
:label: rem-kuns-certificate-and-continuation

The primitive expressions above have the following outward-verifiable
enclosures:

$$
\begin{gathered}
1.95130<H_B^-<1.95131,\quad1.12098<H_A^+<1.12099,\\
.39927<\beta_+<.39928,\quad .01788<\beta_-<.01789,\quad
.02058<\beta_{BB}<.02059,\\
.38138<\beta_+-\beta_-<.38140,\qquad
1.03877<.993(1-\beta_-)-.793\beta_{BB}+.2\beta_+<1.03879.
\end{gathered}
\tag{KUNS.15}
$$

A finite lower integral sum certifies the accepted-row Gaussian
probability. Let $z_j=-8+j/20$, $j=0,\ldots,320$, and
$a_j=\max(|\varepsilon+.1z_{j-1}|,|\varepsilon+.1z_j|)$. Then

$$
\underline s_J\ge
\left[\sum_{j=1}^{320}
 [\Phi(z_j)-\Phi(z_{j-1})]
 \ell_{R_e,\tau}(g(a_j))\right]^3>.20599.
\tag{KUNS.16}
$$

Odd monotonicity of $g$ and even monotonicity of the Gaussian interval
probability justify each lower summand. The omitted tails are
nonnegative. Evaluate each Gaussian CDF with 401 integrated exponential
series terms, through degree $801$, with remainder below $10^{-200}$ for
$|x|\le8$, exactly as in (KURC.17)--(KURC.21). Outside this interval
use $0<Q(x)\le\phi(x)/x\le6.5\cdot10^{-16}$ and symmetry.
The first omitted integrated-series term at $8$ is below
$10^{-270}$; its terms decrease after index $32$.
The same recipe gives $\underline s_0>.99347$.
Square roots, exponentials and sines can be evaluated by elementary
outward interval arithmetic. The standalone
`docs/research/keystone_uniform/verify_native_seed.py` evaluates these
expressions at 85 decimal digits of outward interval precision and
asserts every strict fitness, gate, copy and landing margin used here.
No clipped-noise law is being simulated.
The displayed decimal endpoints, not rounded central diagnostics,
certify the strict inequalities used above.

These are positive source-establishment inputs, rather than a full
establishment-time theorem. Nonzero kinetic noise spreads the next
population beyond the two narrow source bands and changes its velocity
law. Thus (KUNS.12) cannot be iterated by silently retaining the same
entering hypothesis. An expected count alone also does not supply the
target-first probability required by {prf:ref}`thm-slc-hazard` or
{prf:ref}`thm-slch-competing-events`.

For a primitive establishment-time budget the next exact obligation is
to bound the actual complete transition into an increasing-central-mass
population class, uniformly over the whole reached class, including
its dispersed source positions, velocities and terminal marks. In
the notation of the existing hazard theorem this means explicitly
producing a block length $b_e(N)$ and probability $p_e(N)>0$ such that
every allowed discovery history has

$$
\Pr\{\tau_{C,K_N}\le\tau_{C,1}+b_e(N),\
       \tau_{C,K_N}<\tau_\dagger\wedge\sigma\mid
       \mathcal H_{\tau_{C,1}}\}\ge p_e(N),
\tag{KUNS.17}
$$

with its reached-source envelope and uncontrolled exit $\sigma$
declared and quantitatively bounded. A proof of (KUNS.17) must compute that
inequality from the signed source gains, within-source clone-jitter
charge, reverse measurement events, actual component velocity law and
both kinetic stages. Once it is proved, the existing discovery hazard
and stopping-time composition give the establishment budget without
requiring contraction of all distances between different phases.
The present note supplies the explicit native one-update producer,
including its quantitative positive sign, and leaves (KUNS.17) as its
precise continuation interface.
:::
