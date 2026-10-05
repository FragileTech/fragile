# Default source-boundary regularity and the exact delayed moment account

(sec-dsti-carrier)=
## 1. Actual carrier and the two boundary questions

:::{prf:definition} Default harmonic source-boundary record
:label: def-dsti-carrier

Use the actual count-normalized harmonic record of research 34, 37,
38 and 47:
$$
d=3,\quad L=2,\quad h=.04,\quad t=.02,\quad \nu=.3,\quad
a=t\nu=.006,\quad V=2,\quad V_c=4,\quad
\sigma_J=\sigma_x=.1,\quad \alpha_{\rm col}=.5.
$$
The force and raw reward remain $F(x)=-x$ and $R(x,v)=-|x|^2/2$.
Retain all current-frame alive-only sampled fitness statistics,
actual self-excluded roles, frozen position sources, mandatory
revival, component Haar marks on original slot velocities, and
all uncut Gaussian innovations. Terminal marks are
$a_{\rm mark}=\mathbf1_D(x)$, $D=[-L,L]^d$.
The current-frame restriction disables history, curl, elite and
geometry-feedback branches, as in (DMC.1).

Take any $0<\theta\le\theta_f$ of (DMC.3). Its proved actual alive
acceptance ceiling and incoming column bound obey
$$
a_*\le1/8,\qquad c_*=a_*/\kappa_C\le1/8,
\tag{DSTI.1}
$$
where $\kappa_C\in(0,1]$ is the configured donor floor.
The mandatory dead gate remains one. The population argument uses
the declared actual finite rooted components of research 37.
Set
$$
c=e^{-.04},\quad b=t(1+c),\quad a_x=1-tb,\quad
m_h=1-t^2,\quad q^2=(1-c^2)/2,\quad s^2=.0004,\quad
\tau^2=t^2q^2+s^2,
$$
$$
\sigma_1^2=\tau^2+a_x^2\sigma_J^2,\qquad
K_{\rm src}=\max\{1+c_*,1/\kappa_C\},\qquad
C_{\rm bd}=\frac{2d}{\tau\sqrt{2\pi}}.
\tag{DSTI.2}
$$
The actual stages, with identity operators distinguished from the
copying indicator $I$, are
$$
X=S+IJ,\quad P=v^{\rm p},\quad p_1=(\operatorname{Id}-aL_X)P,
\quad w=c(p_1-tX)+q\xi,
$$
$$
y=a_xX+bp_1+tq\xi,\quad z=(\operatorname{Id}-aL_y)w-ty,
\quad x^+=y+s\chi,\quad v^+=C_V(z).
\tag{DSTI.3}
$$
Every source $S$ is in $D$, but retained entering dead coordinates
have no imposed bound. Define
$$
\mathcal B_\delta=\{x\in D:|x|_\infty>L-\delta\},
\qquad 0<\delta<L.
$$
For a population law with alive mass $m_A>0$, put
$\beta_\delta=\mu(a_{\rm mark}=1,x\in\mathcal B_\delta)/m_A$.
For a surviving finite array put
$\beta_{\delta,N}=M^{-1}\sum_i\mathbf1_{\{x_i\in\mathcal B_\delta\}}$,
where $M$ is its alive count.
Let $a_{\rm ret}\in(0,1)$ and $e_N=(1-a_{\rm ret})^N$ be
the exact safe-return constants (DSA.3).

Boundary-layer regularity concerns $\delta\downarrow0$.
The stronger interior criterion concerns the fixed layer
$\delta=.5$ and the small source fraction in (RPT.12).
:::

(sec-dsti-boundary)=
## 2. A boundary-layer class preserved by the actual update

:::{prf:lemma} Gaussian boundary density and actual source columns
:label: lem-dsti-raw-boundary

For every actual surviving entering finite array, and every
consistent population input with nonzero alive mass, the raw
proposal satisfies
$$
\mathbb E\frac1N\sum_i\mathbf1_{\{x_i^+\in\mathcal B_\delta\}}
\ \text{or}\
\mu^+(a_{\rm mark}=1,x\in\mathcal B_\delta)
\le\min\{1,C_{\rm bd}\delta\}.
\tag{DSTI.4}
$$
For any fixed input the expected next-preparation source fraction
in $\mathcal B_\delta$ obeys
$$
B_{\rm src}(\delta)\le
\min\left\{1,\left[m_A(1+c_*)+
                  \frac{1-m_A}{\kappa_C}\right]\beta_\delta\right\}
\le\min\{1,K_{\rm src}\beta_\delta\}.
\tag{DSTI.5}
$$
For a finite input use its own $m_A=M/N$ and $\beta_{\delta,N}$.
:::

:::{prf:proof}
Freeze the full prepared $(X,P)$, before the fresh $\xi,\chi$.
Then $p_1$ is fixed. The exact position identity gives independent
row Gaussian positions with means $a_xX_i+bp_{1,i}$ and covariance
$\tau^2\operatorname{Id}$. The actual noisy second graph and cap
do not change positions. Every coordinate marginal density is
at most $(\tau\sqrt{2\pi})^{-1}$. The band is contained in the
union of $2d$ coordinate intervals of length $\delta$. Their
union bound proves (DSTI.4), then average the actual preparation.
This calculation is before survival or any moment restriction.

An alive persistent row contributes at most its own boundary
indicator. Each alive donor's expected accepted-alive incoming
column is at most $c_*$. A mandatory dead row selects each
alive donor with probability at most $1/(\kappa_CM)$.
Summing these actual source contributions gives
$[1+c_*+(1-m_A)/(\kappa_Cm_A)]m_A\beta_\delta$.
For a singleton alive pool its accepted-alive column is zero,
so the bound remains valid. The population conditional donor
formula gives the same charge. Finally
$m_A(1+c_*)+(1-m_A)/\kappa_C$ is a convex combination of the
two numbers defining $K_{\rm src}$.
:::

:::{prf:lemma} A population-size independent inverse alive-count moment
:label: lem-dsti-inverse-count

For the raw finite proposal from any surviving input, put
$W_N=(N/M^+)\mathbf1_{\{M^+>0\}}$. Then
$$
\mathbb EW_N^2<7/a_{\rm ret}^2.
\tag{DSTI.6}
$$
This averages the actual correlated proposal; it does not
condition the two count kicks on an independent alive count.
:::

:::{prf:proof}
Conditional on the discrete source/component plan before jitters,
the safe events (DSA.6) are independent and have probabilities at
least $a_{\rm ret}$. They imply actual alive marks for every
value of the jitter-dependent first count field. Independent
thinning uniforms therefore construct
$B\sim\operatorname{Bin}(N,a_{\rm ret})$ with $B\le M^+$
pathwise, preserving the actual proposal marginal.
For $k\ge1$, $k^{-2}\le6/[(k+1)(k+2)]$, and
$$
\mathbb E\frac1{(B+1)(B+2)}
=\int_0^1u(1-a_{\rm ret}u)^N\,du
\le\int_0^\infty u e^{-a_{\rm ret}Nu}\,du
=\frac1{a_{\rm ret}^2N^2}.
$$
Thus $\{B\ge1\}$ contributes at most $6/a_{\rm ret}^2$
to $\mathbb EW_N^2$.
On $\{B=0,M^+>0\}$ use $W_N^2\le N^2$. Its contribution is
at most
$$
N^2(1-a_{\rm ret})^N\le N^2e^{-a_{\rm ret}N}
\le4/(e^2a_{\rm ret}^2)<1/a_{\rm ret}^2.
$$
The maximum of $u^2e^{-u}$ is at $u=2$, and $e>2$.
Add both parts. Dependence of $B$ on the physical proposal
does not affect any of these inequalities.
:::

:::{prf:theorem} Preserved source-boundary envelopes without a particle floor
:label: thm-dsti-preserved-boundary

For the actual marked population recursion $\mu_n$ from any
capped consistent law with nonzero alive mass, all $n\ge1$ obey
$$
\beta_\delta(\mu_n)\le\min\{1,C_{\rm bd}\delta/a_{\rm ret}\},
\qquad
B_{\rm src}^{(n)}(\delta)\le
\min\{1,K_{\rm src}C_{\rm bd}\delta/a_{\rm ret}\}.
\tag{DSTI.7}
$$
For the actual current-survivor finite law
$\eta_n=\operatorname{Law}(S_n\mid\tau_N>n)$ from any
nonempty alive initial array, all $N\ge2,n\ge1$ obey
$$
\mathbb E_{\eta_n}\beta_{\delta,N}
\le\min\left\{1,\frac{3\sqrt{C_{\rm bd}\delta}}
 {a_{\rm ret}(1-e_N)}\right\}
\le\min\{1,3a_{\rm ret}^{-2}\sqrt{C_{\rm bd}\delta}\},
\tag{DSTI.8}
$$
$$
\mathbb E_{\eta_n}B_{\rm src}^{(n)}(\delta)
\le\min\{1,3K_{\rm src}a_{\rm ret}^{-2}
                               \sqrt{C_{\rm bd}\delta}\}.
\tag{DSTI.9}
$$
Here $B_{\rm src}^{(n)}$ averages the next actual preparation
from $\eta_n$. Restricting this next-preparation observable
to its own next survival introduces one further division by
$1-e_N$. There is no factor depending on the full elapsed time.
For the different convention that draws one uniform slot and
then conditions that slot to be alive, its current boundary
probability is at most $\min\{1,C_{\rm bd}\delta/a_{\rm ret}\}$.
:::

:::{prf:proof}
Population raw output alive mass is at least $a_{\rm ret}$ by
the safe-return theorem. Divide (DSTI.4) by this actual mass and
use (DSTI.5). These estimates hold for every input, giving (DSTI.7)
after every update without an entering-class assumption.

Start the finite raw next proposal from $\eta_{n-1}$ and
draw an independent uniform slot $J_0$. Let $p_n$ be its actual
whole-swarm next survival probability, so $p_n\ge1-e_N$.
The exact swarm-first ratio is
$$
\mathbb E_{\eta_n}\beta_{\delta,N}
=\frac{\mathbb E_{\rm raw}
 [W_N\mathbf1_{\{x_{J_0}^+\in\mathcal B_\delta\}}]}{p_n}.
$$
The inverse count remains inside the expectation.
Cauchy--Schwarz, (DSTI.4) and (DSTI.6) bound its numerator by
$\sqrt7\,\sqrt{C_{\rm bd}\delta}/a_{\rm ret}
<3\sqrt{C_{\rm bd}\delta}/a_{\rm ret}$.
Also $1-e_N\ge a_{\rm ret}$. This proves (DSTI.8).
Average (DSTI.5) under $\eta_n$ for (DSTI.9).
A nonnegative next-preparation observable restricted to next
survival is bounded by its raw mean divided by that actual next
probability, at least $1-e_N$.

For slot-first sampling, the chosen slot being alive already
implies whole-swarm survival, so the two current-survival factors
cancel in its conditional-alive ratio. Its raw numerator is at
most $C_{\rm bd}\delta$; its raw denominator is the expected
alive fraction, at least $a_{\rm ret}$.
:::

:::{prf:corollary} Preserved alive joint positional variation
:label: cor-dsti-alive-bv

The population conditional-alive phase law and the finite
slot-first conditional-alive phase law after any one update have
directional positional bounded variation satisfying
$$
\|D_{x_j}\alpha_n\|_{\rm TV}
\le\frac{2\sqrt{2/\pi}}{a_{\rm ret}s},
\qquad j=1,\ldots,d,\quad n\ge1.
\tag{DSTI.10}
$$
This includes the actual alive faces and all velocity test
functions of absolute value at most one. The swarm-first alive
law is not assigned an independent final Gaussian.
:::

:::{prf:proof}
Before $\chi$, the joint latent $(y,v^+)$ law retains every
actual OU/second-provider/cap correlation. Adding the independent
$s\chi$ gives a joint position--velocity mixture whose derivative
in direction $j$ has total variation at most
$\|\partial_j\varphi_s\|_1=\sqrt{2/\pi}/s$.
Each face $x_j=\pm L$ has full velocity-integrated trace at most
$1/(s\sqrt{2\pi})$. Multiplication by $\mathbf1_D$ adds both
face measures. Its variation is therefore at most
$\sqrt{2/\pi}/s+2/(s\sqrt{2\pi})=2\sqrt{2/\pi}/s$.
Divide by the actual raw root alive mass, at least $a_{\rm ret}$.
The finite slot-first current-survival ratio cancels as above.
:::

(sec-dsti-nonpreservation)=
## 3. The fixed interior criterion is not preserved by these hypotheses

:::{prf:proposition} A full-Gaussian source-class nonpreservation example
:label: prop-dsti-interior-nonpreservation

For the actual default population kernel and every positive
exponent interval (DSTI.1), a fresh-Gaussian consistent input
has stored velocity RMS $.55$ and prepared source boundary
fraction at width $\delta=.5$ less than $.001$, but its next
prepared source fraction at that width exceeds $.001$.
Its output stored velocity RMS is still less than $.545$.
All configured roles, raw reward, revival, cap and full noises
remain actual.
This disproves preservation of this scalar criterion under the
stated hypotheses; it does not disprove eventual burn into a
stronger delayed class or law-level convergence.
:::

:::{prf:proof}
Take
$$
x=(1.436,0,0)+.02Z,\qquad v=(.55,0,0),\qquad
a_{\rm mark}=\mathbf1_D(x),\qquad Z\sim N(0,\operatorname{Id}_3).
\tag{DSTI.11}
$$
This has unbounded retained dead positions, positive alive mass
and the stated RMS. All original velocities are identical, so
every component Haar collision leaves $P=v$, irrespective of
accepted and mandatory edges. The first count alignment is zero,
so $p_1=v$ for all source choices and jitters.

Put $\varepsilon=2^{-100}$. The upper inner face in coordinate
one is $(1.5-1.436)/.02=16/5$ standard deviations away.
Every other inner face is at least $75$ deviations away.
Mills' inequality and $\sqrt{2\pi}>5/2$ give
$$
\Phi(-16/5)<\tfrac18e^{-128/25}<.0008.
$$
The last step follows from the rational certificate
$\sum_{k=0}^{9}(128/25)^k/k!>625/4$.
The other five inner-face tails sum to less than $\varepsilon$.
Every outer face is more than $28$ deviations away, so the
input dead mass is also less than $\varepsilon$; the elementary
union bound $6e^{-28^2/2}<6\cdot2^{-392}<2^{-100}$ suffices.
Thus its conditional-alive band fraction is less than
$(.0008+\varepsilon)/(1-\varepsilon)<.00081$.
Its accepted-alive source columns contribute at most a factor
$1+c_*\le9/8$ to the full alive band mass. All mandatory-dead
source mass is at most $\varepsilon$. The input prepared source
band fraction is therefore less than
$(9/8)(.00081)+\varepsilon<.001$.
This uses the full dead mass directly, not a uniform donor role.

Restrict an entering alive root to $Z_1\in[0,1]$,
$|Z_2|,|Z_3|\le5$, and persistence. Its conditional persistence
probability is at least $7/8$ for every actual measured mark
and donor. Its own jitter indicator is zero. The exact terminal
position is then
$$
x^+=a_xx+bv+\tau H,\qquad H\sim N(0,\operatorname{Id}_3),
$$
with fresh $H$ independent of the entering root and preparation.
Restrict $H_1\in[2.2,2.7]$ and $|H_2|,|H_3|\le5$.
The default bounds $.0392<b<.04$ and $.02<\tau<.023$ give
$$
a_x(1.436)+b(.55)
=1.436+b(.55-.02\cdot1.436)>1.456434176.
$$
The first output coordinate exceeds
$1.456434176+.02(2.2)>1.5$.
Its entering first coordinate is at most $1.456$, so the output
is less than $1.456+.04(.55)+.023(2.7)<2$.
The two other coordinates have absolute value less than
$.1+.023(5)<2$. The event is alive and in $\mathcal B_{.5}$.

The event probability has strict rational lower bounds.
From $e^{-u}\ge1-u$ and $\sqrt{2\pi}<8/3$,
$$
\Pr(Z_1\in[0,1])>
\tfrac38\int_0^1(1-u^2/2)\,du=\tfrac5{16}.
$$
Also
$$
\Pr(H_1\in[2.2,2.7])
>\tfrac12\tfrac38 e^{-729/200}>\tfrac3{624}.
$$
The last inequality follows from $e^{729/200}<39$, with the
entirely rational Taylor/geometric certificate
$$
\sum_{k=0}^{5}\frac{(729/200)^k}{k!}
+\frac{(729/200)^6}{6!\,[1-(729/200)/7]}<39.
$$
The ratios beyond the sixth Taylor term are at most
$(729/200)/7<1$, which justifies this upper bound.
The four other Gaussian coordinate constraints jointly have
probability greater than $499/500$: their failure union is
$8e^{-25/2}<8/2^{12}<1/500$.
Independence of the entering/fresh coordinates and the pointwise
persistence lower bound give raw alive output band mass greater
than
$$
p=\tfrac78\,\tfrac5{16}\,\tfrac3{624}\,\tfrac{499}{500}.
$$
In the following actual preparation those alive roots persist
with conditional probability at least $7/8$, regardless of
their new actual fitness. Their source stays their own boundary
coordinate. Hence the following source band fraction is
strictly greater than
$$
\tfrac78p>.001,
\tag{DSTI.12}
$$
an exact rational comparison. The actual output velocity RMS
is at most $T(.55)<.545$ by (RVB.4) and its rational endpoint
certificate, which applies to every consistent input position law.
:::

(sec-dsti-delayed)=
## 4. Exact delayed signed position--velocity response

:::{prf:theorem} Actual covariance Duhamel identity retaining the cap residual
:label: thm-dsti-delayed-moments

Use normalized finite-array inner products or the actual
population root law. All raw expectations below include
preparation and both own count providers. Put
$$
z_v=c-tb,\quad r_H=t(c+a_x)=m_hb,\quad
H_h=\begin{pmatrix}a_x&b\\-r_H&z_v\end{pmatrix},
$$
$$
A_n=\begin{pmatrix}\mathbb E|x|^2&\mathbb E\langle x,v\rangle\\
\mathbb E\langle x,v\rangle&\mathbb E|v|^2\end{pmatrix},
\quad
A_{{\rm p},1,n}=\begin{pmatrix}
\mathbb E|X|^2&\mathbb E\langle X,p_1\rangle\\
\mathbb E\langle X,p_1\rangle&\mathbb E|p_1|^2
\end{pmatrix},\quad \Delta A_n=A_{{\rm p},1,n}-A_n.
$$
Define the actual noisy terms
$$
D_2=\langle w,L_yw\rangle,\quad E_2=\langle y,L_yw\rangle,
\quad e_{\rm cap}=z-C_V(z),\quad
D_{\rm cap}=|z|^2-|C_V(z)|^2\ge0,
$$
$$
N_h=d\begin{pmatrix}\tau^2&t m_hq^2\\
t m_hq^2&m_h^2q^2\end{pmatrix},
\quad
R_n=\begin{pmatrix}
0&-a\mathbb EE_2-\mathbb E\langle y,e_{\rm cap}\rangle\\
-a\mathbb EE_2-\mathbb E\langle y,e_{\rm cap}\rangle&
-2a\mathbb ED_2+2at\mathbb EE_2
+a^2\mathbb E|L_yw|^2-\mathbb ED_{\rm cap}
\end{pmatrix}.
\tag{DSTI.13}
$$
For population inputs with finite entering moments,
$$
A_{n+1}=H_hA_nH_h^{\mathsf T}
+H_h\Delta A_nH_h^{\mathsf T}+N_h+R_n.
\tag{DSTI.14}
$$
For finite current-survivor laws add one matrix $S_n$, the
actual next-survival moment defect relative to the raw proposal.
With
$$
g_4=[d(d+2)]^{1/4},\quad
K_4=(a_x\sqrt dL+bV_c+\sigma_1g_4+V)^4,
$$
it satisfies
$$
\|S_n\|_{\rm op}\le
2\sqrt{K_4}\sqrt{e_N}/(1-e_N).
\tag{DSTI.15}
$$
Set $S_n=0$ in the population case. For $n>k$ with finite $A_k$,
$$
\begin{split}
A_n={}&H_h^{n-k}A_k(H_h^{\mathsf T})^{n-k}\\
&+\sum_{j=k}^{n-1}H_h^{n-1-j}
[H_h\Delta A_jH_h^{\mathsf T}+N_h+R_j+S_j]
(H_h^{\mathsf T})^{n-1-j}.
\end{split}
\tag{DSTI.16}
$$
The survival-only contribution is uniformly in elapsed time
at most
$$
6.02\sqrt{K_4}\sqrt{e_N}/[(1-e_N)(1-c)].
\tag{DSTI.17}
$$
All needed moments are finite after one update even if retained
initial dead positions have no finite moment. This is a moment
response identity, not a law-level contraction.
:::

:::{prf:proof}
The exact affine pre-second-kick velocity is
$z_0=w-ty=-r_HX+z_vp_1+m_hq\xi$.
The fresh $\xi$ is centered and independent of the full
preparation and first count output. Thus
$$
\mathbb E|x^+|^2=a_x^2\mathbb E|X|^2
+2a_xb\mathbb E\langle X,p_1\rangle+b^2\mathbb E|p_1|^2+d\tau^2,
$$
$$
\begin{split}
\mathbb E\langle x^+,v^+\rangle={}&
-a_xr_H\mathbb E|X|^2
+(a_xz_v-br_H)\mathbb E\langle X,p_1\rangle
+bz_v\mathbb E|p_1|^2+dtm_hq^2\\
&-a\mathbb EE_2-\mathbb E\langle y,e_{\rm cap}\rangle .
\end{split}
$$
Here final $\chi$ is centered and independent of $v^+$.
Writing $v^+=z_0-aL_yw-e_{\rm cap}$ keeps the actual noisy
second-graph and cap cross terms, rather than centering them.
Similarly $z=z_0-aL_yw$ gives
$$
\begin{split}
\mathbb E|v^+|^2={}&r_H^2\mathbb E|X|^2
-2r_Hz_v\mathbb E\langle X,p_1\rangle
+z_v^2\mathbb E|p_1|^2+dm_h^2q^2\\
&-2a\mathbb ED_2+2at\mathbb EE_2
+a^2\mathbb E|L_yw|^2-\mathbb ED_{\rm cap}.
\end{split}
$$
These identities prove (DSTI.14). The matrix $\Delta A_n$
contains the actual source, revival, Haar and first-count
changes, without an assigned monotonicity sign.

For the finite statement let $F$ be the raw proposal's
two-by-two moment matrix before expectation. It is positive
semidefinite and
$\|F\|_{\rm op}\le N^{-1}\sum_i(|x_i^+|^2+|v_i^+|^2)$.
The source-box identity and the pathwise bounds $|p_1|\le V_c$,
$|v^+|\le V$ give by Minkowski
$\mathbb E(|x_i^+|^2+|v_i^+|^2)^2\le K_4$.
Indeed $a_xIJ+tq\xi+s\chi$, conditional on the pre-jitter
plan, has one of two Gaussian variances no greater than
$\sigma_1^2$, hence fourth-norm at most $\sigma_1g_4$.
Independence of this sum and $p_1$ is not required.
Jensen gives $\mathbb E\|F\|_{\rm op}^2\le K_4$.
For actual next extinction probability $p_{\rm ext}$,
$$
S_n=\frac{p_{\rm ext}\mathbb EF
-\mathbb E[F\mathbf1_{\{\rm extinction\}}]}{1-p_{\rm ext}}.
$$
Cauchy--Schwarz and $p_{\rm ext}\le e_N$ prove (DSTI.15).
This is precisely next-survival restriction starting from
$\eta_n$, without a full-history hazard.

Iterate the matrix identity for (DSTI.16). For its survival part
put
$$
\beta_h=\frac{1-c}{2b}=\frac{\tanh(.02)}{.04},\qquad
G_h=\begin{pmatrix}m_h&\beta_h\\\beta_h&1\end{pmatrix}.
$$
Direct multiplication using $r_H=m_hb$ and $a_x-z_v=1-c$
gives $H_h^{\mathsf T}G_hH_h=cG_h$ exactly.
Since $0<\tanh(.02)<.02$, $\beta_h<.5$,
$\lambda_{\min}(G_h)>.4996$ and
$\lambda_{\max}(G_h)<1.5$.
Their ratio is less than $3.01$, giving
$\|H_h^j\|_{\rm op}^2\le3.01c^j$.
Apply (DSTI.15) and sum the geometric series for (DSTI.17).
The auxiliary linear $G_h$ identity does not assert that
the cap contracts this cross metric: its full residual stays
in $R_n$. Source-box Gaussian moments and the stored cap give
finiteness after the first update.
:::

:::{prf:lemma} Explicit source, original-Haar and first-count entries
:label: lem-dsti-preparation-residual

In (DSTI.14), let $C$ be the actual frozen component containing
a uniform input root and let $\bar v_C$ be the mean of its
original slot velocities. Put
$D_1=\langle P,L_XP\rangle$ and
$E_1=\langle X,L_XP\rangle$.
The exact residual entries are
$$
\begin{split}
(\Delta A_n)_{xx}
={}&\mathbb E|S|^2-\mathbb E|x|^2
                 +d\sigma_J^2\mathbb EI,\\
(\Delta A_n)_{xv}
={}&\mathbb E\langle S,\bar v_C\rangle
                 -a\mathbb EE_1-\mathbb E\langle x,v\rangle,\\
(\Delta A_n)_{vv}
={}&\mathbb E\big[|\bar v_C|^2+
           \alpha_{\rm col}^2|v-\bar v_C|^2\big]
          -\mathbb E|v|^2-2a\mathbb ED_1
          +a^2\mathbb E|L_XP|^2 .
\end{split}
\tag{DSTI.18}
$$
For the actual measured mark and donor, write $g$ for its
alive acceptance probability, let $x_C$ denote that donor's
original alive position, and let $a_{\rm mark}$ be the
entering root mark. Then
$$
\mathbb E|S|^2=
\mathbb E\!\left[
a_{\rm mark}\{(1-g)|x|^2+g|x_C|^2\}
+(1-a_{\rm mark})|x_C|^2\right],
$$
$$
\mathbb EI=\mathbb E[a_{\rm mark}g+1-a_{\rm mark}].
\tag{DSTI.19}
$$
All expectations include their actual alive-only donor
normalizers, sampled fitness and the complete frozen forest.
In particular $\mathbb E\langle S,\bar v_C\rangle$ is a
mixed source/component term, not a product of its means.
:::

:::{prf:proof}
Conditional on the complete discrete forest and Haar marks,
$S,I,P$ are fixed before the independent centered $J$.
Hence $\mathbb E|X|^2=\mathbb E|S|^2+
d\sigma_J^2\mathbb EI$ and
$\mathbb E\langle X,P\rangle=\mathbb E\langle S,P\rangle$.
Conditional on the discrete forest alone the actual component
formula is
$P=\bar v_C+\alpha_{\rm col}O_C(v-\bar v_C)$.
The Haar rotation in dimension three has zero matrix mean
and preserves the centered norm. Its expectation therefore gives
$$
\mathbb E_{\rm Haar}\langle S,P\rangle
=\langle S,\bar v_C\rangle,\qquad
\mathbb E_{\rm Haar}|P|^2
=|\bar v_C|^2+\alpha_{\rm col}^2|v-\bar v_C|^2.
$$
These are the original frozen root velocity and its component
mean; donor velocities have not been copied.
The realized jitter-dependent first count output satisfies
$\langle X,p_1\rangle=\langle X,P\rangle-aE_1$ and
$|p_1|^2=|P|^2-2aD_1+a^2|L_XP|^2$ after the normalized
row average or population integral. Substitution gives (DSTI.18).
Finally condition on the actual root, measured mark and donor
before its acceptance uniform. An alive root persists with
probability $1-g$ and copies with probability $g$; a dead root
copies with probability one. This gives (DSTI.19). Every actual
source/component correlation remains inside the expectations.
:::

(sec-dsti-evidence)=
## 5. Completed endpoints and the remaining inference

:::{prf:remark} Exact status of the default source-tail task
:label: rem-dsti-status

(DSTI.7)--(DSTI.10) prove an actual boundary regularity class
preserved after one update, uniformly in time and population size.
They retain original velocities, mandatory revival, actual roles,
both own count kicks, cap and all Gaussian tails. The finite
swarm-first estimate keeps its own inverse alive count and has
no nonvanishing particle floor.

These envelopes do not imply the small fixed-width source
fraction required by (RPT.12); at $\delta=.5$ they can give
only the trivial bound one.
Proposition {prf:ref}`prop-dsti-interior-nonpreservation` proves
that the $.001$ source criterion is not preserved under the
velocity threshold and fresh-Gaussian regularity alone.
The exact delayed identity (DSTI.16) leaves the following
task: control the actual sum of
$H_h\Delta A_jH_h^{\mathsf T}+R_j$ and then the central
Gaussian functional (RPT.9), or prove a stronger delayed
shape/tail class implying that functional's smallness.
No Euclidean revival-energy monotonicity, averaged-Jacobian
factorization, alive-floor-to-small-dead inference, or nonlinear
own-provider law gap is used here. Previously completed
conservative, small-viscosity and default frozen-provider
law results retain their stated scopes.
:::
