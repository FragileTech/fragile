# Gaussian box geometry, rare killing and the remaining phase comparison

(sec-kurc-record)=
## 1. The unchanged record and the exact comparison being tested

:::{prf:definition} Actual landing map and the population energy budget
:label: def-kurc-record

Use the complete native reference record of
{prf:ref}`def-ku-conditioning-record`: $d=3$, $D=[-2,2]^3$,
$h=.04$, $\gamma=b_O=\rho=1$, $\nu=.3$, $\sigma_J=\sigma_x=.1$,
$V=2$ and $\alpha_{\rm col}=.5$. All measured rewards/diversities,
standardizations, companion roles, acceptance gates, mandatory revival,
component-Haar rotations, both viscous B stages, unbounded Gaussian
innovations, the original radial cap and terminal marking remain unchanged.
The configured force in this record is

$$
F_k(x)=-2x_k-20\pi\sin(2\pi x_k).
$$

Write $t=h/2$, $c=e^{-\gamma h}$, $b=t(1+c)$,
$\eta=bt$, $\tau^2=t^2q^2+s^2$, and

$$
g(x)=(1-2\eta)x-20\pi\eta\sin(2\pi x),\qquad
V_c=(1+2|\alpha_{\rm col}|)V=4.
\tag{KURC.1}
$$

Freeze the complete measured source/gate/component pattern and its Haar
rotations before jitter. Its actual frozen sources satisfy
$\mu_i\in\overline D$; if $A_i$ is its actual copy/revival indicator,
then $X_i=\mu_i+A_i\sigma_J Z_i^J$. Conditional on the complete
preparation, the actual final positions are

$$
X_i^+=g(X_i)+bU_i+\tau Z_i,\qquad
U=W_XV^{\rm col}.
\tag{KURC.2}
$$

The $Z_i$ are independent standard $d$-dimensional Gaussians. The first
viscous matrix uses every realized jitter. In both normalizations it is
stochastic at $t\nu=.006$, so $|U_i|\le V_c$. Both $U$ and its row
correlations remain in the actual law.

For every realization of preparation,

$$
\|U\|_{2,N}\le E_\nu,
\qquad E_\nu=\begin{cases}
V,&\text{count normalization},\\
\min\{V_c,V(1-t\nu+t\nu\sqrt C)\},&\text{row normalization},
\end{cases}
\tag{KURC.3}
$$

where $C$ is any proved upper bound for the nonself Gaussian-row column
mass. The singleton has zero viscosity and uses $E_\nu=V$ in either mode.
:::

:::{prf:proof}
The original two A drifts give $X^+=X+b[W_XV^{\rm col}+tF(X)]$
plus $tq\xi^O+s\xi^x$. B2 and the radial velocity cap leave these
positions unchanged; their actual effects on the retained velocity have
not been substituted. The independent row Gaussian sums have covariance
$\tau^2 I_d$, proving (KURC.2).

For any frozen connected component, write its entering velocities as
$\bar v+\widetilde v_i$. Orthogonality of its actual common Haar matrix
gives

$$
\sum_i|\bar v+\alpha_{\rm col}O\widetilde v_i|^2
=n|\bar v|^2+\alpha_{\rm col}^2\sum_i|\widetilde v_i|^2
\le\sum_i|v_i|^2.
$$

This is a component identity and therefore survives the entire measured
graph law. Every retained entering speed is capped by $V$, so the
collision energy is at most $V^2$. The count matrix contracts the
unweighted normalized second moment. For the row matrix,
$\|P_X\|_{2\to2}\le\sqrt C$ follows from its row sums one and column
sums at most $C$, by rowwise Jensen followed by column summation. Thus
$\|W_X\|_{2\to2}\le1-t\nu+t\nu\sqrt C$. The rowwise bound also
gives $\|U\|_{2,N}\le V_c$. This proves (KURC.3) without a degree floor,
independent collision rows, or a maximum Gaussian over $N$. $\square$
:::

:::{prf:lemma} Monotone Rastrigin landing geometry
:label: lem-kurc-monotone-map

The actual reference map is odd, globally strictly increasing, and maps
$[-2,2]$ into itself. More precisely,

$$
g'(x)\ge1-\eta(2+40\pi^2)=.6887959111964013>0,
\qquad g(2)=2-4\eta=1.9968627368973564.
\tag{KURC.4}
$$

Set $L=2$, $R=bV_c<L$ and define the unrestricted accepted-row floor

$$
p_1=\left[\int_{\mathbb R}
 \ell_{L-R,\tau}\bigl(g(L+\sigma_J z)\bigr)\phi(z)\,dz\right]^d,
\qquad
\ell_{a,\tau}(m)=\Phi((a-m)/\tau)-\Phi((-a-m)/\tau).
\tag{KURC.5}
$$

Every accepted row has an own-jitter-dependent lower landing probability
$G_i(Z_i^J)$ with conditional mean at least $p_1$, independently of the
other rows' jitters. This refers to lower indicators or lower conditional
probabilities; the actual landing probabilities need not be independent
before full preparation.
:::

:::{prf:proof}
Differentiate the actual sine formula to obtain (KURC.4). Oddness and
monotonicity, together with its endpoint value, give the asserted image
of the source interval. For each realized $U_i$, the cube
$[-L+R,L-R]^d$ translated by $bU_i$ is contained in $D$ because
$|bU_i|\le R$. Hence the actual row landing probability is at least

$$
G_i(Z_i^J)=\prod_{k=1}^d
\ell_{L-R,\tau}\bigl(g(\mu_{i,k}+\sigma_J Z_{i,k}^J)\bigr).
\tag{KURC.6}
$$

The function $x\mapsto\ell_{L-R,\tau}(g(x))$ is even and
nonincreasing on $x\ge0$. Its superlevel sets are centered intervals.
The probability that a Gaussian of fixed variance and mean $\mu$ lies
in any such interval is even and nonincreasing in $|\mu|$: differentiate
its two CDF endpoints. Layer-cake integration proves the same assertion
for the convolution in (KURC.5). Its minimum for $|\mu|\le L$ is
therefore attained at $\mu=L$. Conditional independent coordinate
jitters give $\mathbb E G_i\ge p_1$. No cutoff has been imposed on
either Gaussian, and the force at every jittered position is the actual
configured sine force. $\square$
:::

(sec-kurc-box-energy)=
## 2. A stronger energy-to-survival estimate from the Gaussian box

:::{prf:lemma} Convex energy profile for an unaccepted row
:label: lem-kurc-convex-box-profile

Put $k=b/\tau$, $Q(z)=1-\Phi(z)$ and

$$
\epsilon_D=\exp[-2L^2/\tau^2],\qquad
\psi(r)=(1-\epsilon_D)^d
 \left[Q\left(\frac{kr}{\sqrt d}\right)\right]^d.
\tag{KURC.7}
$$

For every unaccepted row of the actual prepared array,
its conditional probability of being alive is at least $\psi(|U_i|)$.
Moreover $e\mapsto\psi(\sqrt e)$ is decreasing and convex on
$[0,\infty)$.
:::

:::{prf:proof}
For this row $X_i=\mu_i\in\overline D$, so (KURC.4) gives
$|g(X_{i,k})|\le L$. Its conditional landing probability is consequently
at least the product of interval probabilities at absolute means
$L+b|U_{i,k}|$. If $z=k|U_{i,k}|$ and $A=2L/\tau$, that interval
probability equals $Q(z)-Q(z+A)$. Direct translation in the Gaussian
integral gives

$$
Q(z+A)=e^{-A^2/2}\int_z^\infty e^{-Au}\phi(u)\,du
\le e^{-A^2/2}Q(z),\qquad z\ge0.
\tag{KURC.8}
$$

For completeness the elementary identity

$$
Q(z)=\frac1\pi\int_0^{\pi/2}
 e^{-z^2/(2\sin^2\theta)}\,d\theta,
\qquad z\ge0,
\tag{KURC.9}
$$

follows by differentiating for $z>0$: substitute
$u=\cot\theta$ in the derivative to obtain $-\phi(z)$, then use
the value $1/2$ at zero. Thus $e\mapsto Q(k\sqrt e)$ is a positive
mixture of exponentials in $e$ and is log-convex. Jensen applied to
its logarithm, with $e_k=|U_{i,k}|^2$, gives

$$
\prod_k Q(k|U_{i,k}|)
\ge \left[Q(k|U_i|/\sqrt d)\right]^d.
$$

Together with (KURC.8) this proves the row floor. Finally the right
side of (KURC.7), as a function of $e=r^2$, is itself a positive
$d$-fold integral of exponentials in $e$, using (KURC.9). It is
therefore decreasing and convex, including at zero by continuity.
$\square$
:::

:::{prf:theorem} Population-uniform survival with every gate pattern retained
:label: thm-kurc-actual-global-survival

Define, with the continuous convention $H(0)=p_1$,

$$
H(u)=u\psi(E_\nu/\sqrt u)+(1-u)p_1,\qquad
a_* =\min_{0\le u\le1}H(u)>0.
\tag{KURC.10}
$$

For every nonextinct entering state, every population $N\ge1$,
and $\theta\ge0$, the actual terminal alive count obeys

$$
\mathbb E_S e^{-\theta M^+}
\le[1-(1-e^{-\theta})a_*]^N,
\qquad
\Pr_S(M^+=0)\le(1-a_*)^N,
\qquad \mathbb E_S(M^+/N)\ge a_*.
\tag{KURC.11}
$$

This is a Laplace-transform bound and its displayed consequences.
Full stochastic binomial domination is not inferred from this transform.
No lower bound on a fitness difference, acceptance frequency or source
phase mass is a premise.
:::

:::{prf:proof}
Condition on the complete pre-jitter information from (KURC.2), and
let $uN$ be its number of unaccepted rows. Conditional on every jitter,
the actual terminal indicators are independent Bernoulli trials with
parameters $p_i$. Put $T=1-e^{-\theta}$. The conditional transform is
$\prod_i(1-Tp_i)$.

For the unaccepted subset, (KURC.3), the previous lemma and energy
Jensen imply, pathwise in every jitter,

$$
\frac1{uN}\sum_{i:A_i=0}p_i
\ge \psi\!\left(
 \sqrt{\frac1{uN}\sum_{i:A_i=0}|U_i|^2}\right)
\ge\psi(E_\nu/\sqrt u).
\tag{KURC.12}
$$

Concavity of $\log(1-Tp)$ bounds their product by
$[1-T\psi(E_\nu/\sqrt u)]^{uN}$. This is a deterministic envelope
independent of all jitters, so it can be pulled out before integrating
the accepted-row factors. Replace each accepted factor by
$1-TG_i(Z_i^J)$ from (KURC.6). These replacement factors depend only
on their own independent jitters. Their product integrates to at most
$(1-Tp_1)^{(1-u)N}$, while every measured gate and Haar correlation
has remained frozen. Weighted AM--GM now gives

$$
\mathbb E[e^{-\theta M^+}\mid\text{pre-jitter information}]
\le(1-TH(u))^N\le(1-Ta_*)^N.
$$

Average the original measured pattern law. Letting $\theta\to\infty$
gives extinction, and the right derivative at zero gives the first
moment. Positivity of $a_*$ follows from continuity and positivity of
both endpoints and every interior value of $H$. $\square$
:::

:::{prf:corollary} Chernoff, inverse-alive and QSD survival consequences
:label: cor-kurc-conditioning-consequences

For any proved $0<a\le a_*$, put $c_0=(1-\log2)/2$. Then

$$
\Pr_S(M^+/N<a/2)\le e^{-c_0aN},\qquad
Q_N1(S)\ge1-(1-a)^N.
\tag{KURC.13}
$$

After conditioning on actual survival, for every $r>0$,

$$
\mathbb E_S[(N/M^+)^r\mid M^+>0]
\le (2/a)^r+\left(\frac r{e c_0a}\right)^r.
\tag{KURC.14}
$$

The same bounds hold after mixing an arbitrary entering survivor law.
Every compatible QSD has survival eigenvalue
$\alpha_N\ge1-(1-a)^N$, and the one-step survivor denominator obeys

$$
C_{{\rm surv},N}\le\frac1{1-(1-a)^N}\le\frac1a.
\tag{KURC.15}
$$
:::

:::{prf:proof}
Use (KURC.11) at $\theta=\log2$ and Markov's inequality;
$(1-a/2)^N\le e^{-aN/2}$ gives (KURC.13). If the actual extinction
probability is $\delta$, subtract it before conditioning:
$\Pr(0<M^+<aN/2)\le e^{-c_0aN}-\delta$.
Division by $1-\delta$ can only decrease the upper bound
$e^{-c_0aN}$. Split the inverse moment at $M^+=aN/2$ and use
$(N/M^+)^r\le N^r$ on its surviving complement. The supremum of
$N^re^{-c_0aN}$ over positive real $N$ is
$(r/(ec_0a))^r$. Integration against a QSD proves the eigenvalue
bound; (KURC.15) follows directly. $\square$
:::

(sec-kurc-evaluation)=
## 3. Explicit finite certificates and sharper diagnostic values

:::{prf:lemma} Reference coefficients certified without changing the noise
:label: lem-kurc-certified-reference

For the reference record one may use, for every $N\ge1$,

$$
\boxed{a_{
\rm count}=2\cdot10^{-6},\qquad
a_{\rm row}=7\cdot10^{-11}.}
\tag{KURC.16}
$$

The row singleton also admits $a_{\rm count}$. Thus the uniform
survivor multipliers in (KURC.15) are respectively at most
$500000$ and $10^{11}/7$. These conservative finite constants are
not claims that the true survivor probability is this small.

Here is a finite certificate for the accepted coefficient. Let
$z_j=-8+j/25$, $j=0,\ldots,400$, $L_*=L-bV_c$, and set

$$
\underline p_1=
\left[\sum_{j=1}^{400}
 [\Phi(z_j)-\Phi(z_{j-1})]
 \ell_{L_*,\tau}(g(L+\sigma_Jz_j))\right]^3.
\tag{KURC.17}
$$

Because $L+\sigma_Jz\in[1.2,2.8]$ on this interval, its integrand
apart from the Gaussian density is decreasing. Consequently
$p_1\ge\underline p_1>1.41317461517\cdot10^{-5}$.
Take $r_0=103/50$ and write

$$
\widetilde\psi(r)=[Q(kr/\sqrt3)]^3,\qquad
B_0=\widetilde\psi'(r_0)/(2r_0),\qquad
A_0=\widetilde\psi(r_0)-B_0r_0^2.
\tag{KURC.18}
$$

Explicit interval evaluations give

$$
\begin{aligned}
1.35529853339420\cdot10^{-5}<A_0
 &<1.35529853339421\cdot10^{-5},\\
2.05146788376870\cdot10^{-6}<A_0+4B_0
 &<2.05146788376872\cdot10^{-6}.
\end{aligned}
\tag{KURC.19}
$$

For row normalization, the same existing column-shell proof (KUK.7)
admits the sharper finite upper bound

$$
\begin{aligned}
C^{(M)}={}&2e^2+9^3\left[1+
 \sum_{m=0}^{M-1}
 \left(2e^{-a r^{2m}}+e^{-\beta r^m}\right)
 +2e^{-a r^{2M}}
       \left(1+\frac1{2a(\log r)r^{2M}}\right)
 +e^{-\beta r^M}
       \left(1+\frac1{\beta(\log r)r^M}\right)\right],\\
&a=35/128,\qquad\beta=3/8,\qquad r=5/4.
\end{aligned}
\tag{KURC.20}
$$

At $M=40$ this gives $C^{(40)}<7188$. Use the explicit larger
integer $C=7188$ in (KURC.3). Then

$$
E_\nu=2(.994+.006\sqrt{7188})=3.005384882922879,
\qquad \widetilde\psi(E_\nu)>7.45304452175\cdot10^{-11}.
\tag{KURC.21}
$$
:::

:::{prf:proof}
The right-endpoint sum (KURC.17) is a lower integral sum with exact
Gaussian masses, discarding only nonnegative contributions outside
$[-8,8]$. Convexity in energy gives the supporting line
$\widetilde\psi(\sqrt e)\ge A_0+B_0e$, with $B_0<0$.
For count normalization $E_\nu^2=4$. If $u$ is the unaccepted
fraction, the same energy argument therefore gives the lower rate

$$
(1-\epsilon_D)^3(A_0u+4B_0)+(1-u)\underline p_1
\ge(1-\epsilon_D)^3(A_0+4B_0),
$$

since $\underline p_1>(1-\epsilon_D)^3A_0$.
Also $\log\epsilon_D=-19259.624837630152<-200\log10$.
Equations (KURC.19) prove its lower bound $2\cdot10^{-6}$.

For (KURC.20), retain the first $M$ terms of the original column
series. Each remaining decreasing summand is bounded by its first
term plus its integral. Substitute $v=a r^{2x}$ or
$v=\beta r^x$, and use
$\int_z^\infty e^{-v}\,dv/v\le e^{-z}/z$.
This proves the exact expression before numerical evaluation.
For row normalization Jensen alone bounds the rate by
$u\psi(E_\nu/\sqrt u)+(1-u)\underline p_1$.
Its perspective term is convex, and its derivative at $u=1$ is
$\psi(E_\nu)-E_\nu\psi'(E_\nu)/2<1.42\cdot10^{-9}
<\underline p_1$. Hence the minimum is at $u=1$.
Equation (KURC.21) and $\epsilon_D<10^{-200}$ prove the stated
row coefficient.

All quoted enclosures can be checked by finite elementary interval
arithmetic. One explicit Gaussian-CDF recipe is the integrated
exponential series

$$
\Phi(x)=\frac12+\frac1{\sqrt{2\pi}}
 \sum_{n=0}^{400}\frac{(-1)^n x^{2n+1}}{(2n+1)2^n n!}
 +R_{400}(x),\qquad |R_{400}(x)|<10^{-200},\quad |x|\le8.
$$

The alternating remainder is bounded by its next term because
the summands decrease after $n\ge32$; that next term at $8$
is below $10^{-200}$. Outside $[-8,8]$, use
$0<Q(x)\le\phi(x)/x$ for $x\ge8$ and symmetry. This tail
bound is at most $6.4\cdot10^{-16}$ and suffices for every
enclosure displayed above. Evaluate the elementary coefficients
and arguments by interval exponentials/sines and exact integer
factorials. Thus the certificate concerns the original integrals,
not a clipped Gaussian simulation. $\square$
:::

The reproducible finite calculation is saved as
`docs/research/keystone_uniform/verify_rare_killing.py`.
Run `uv run python docs/research/keystone_uniform/verify_rare_killing.py`;
its interval outputs test the margins in (KURC.19)--(KURC.21).

:::{prf:remark} Values from the exact one-dimensional optimization
:label: rem-kurc-sharper-values

The unrestricted integral (KURC.5) is approximately
$p_1=1.619119267361721\cdot10^{-5}$. With the exact column-series
value $C_3\le7187.666564111235$, (KURC.10) gives approximately

$$
\begin{array}{c|c|c|c}
\text{normalization}&E_\nu&\text{minimizing }u&a_*\\\hline
\text{count}&2&.9634116560416744&2.17483625744411\cdot10^{-6}\\
\text{row}&3.005361285498921&1&7.455154672651456\cdot10^{-11}
\end{array}
\tag{KURC.22}
$$

These are diagnostic rounded values. The completely certified
constants used in consequences are (KURC.16). Count and row
remain distinct: the row column estimate gives a larger physical
velocity-energy envelope, not a conjectural uniform degree.
:::

(sec-kurc-phase-comparison)=
## 4. What the actual regional communication coefficients certify

:::{prf:proposition} Native regional killing and the direct discovery budget
:label: prop-kurc-phase-scales

Use the original regional cores $Q_z=z+[-R_0,R_0]^3$ with
$R_0=1/16$, jitter tag $J=1/64$, and stable Rastrigin roots
$z_k$ from {prf:ref}`def-klq-landscape-cells`. A boundary core
always means $Q_z\cap D$ for entering living sources and alive
landing. For sources in a core with coordinate source bound $m_0$,
put

$$
\ell=1-2\eta,\quad A=20\pi\eta,\quad
\sigma_*^2=\ell^2\sigma_J^2+\tau^2,\qquad
\delta_J(m_0)=
 Q\!\left(\frac{L-bV_c-\ell m_0-A}{\sigma_*}\right)
 +Q\!\left(\frac{L-bV_c+\ell m_0-A}{\sigma_*}\right).
\tag{KURC.23}
$$

For unaccepted rows the analogous bound is
$\delta_0(m_0)=
Q((L-bV_c-g(m_0))/\tau)+
Q((L-bV_c+g(m_0))/\tau)$.
Set $\delta_{\rm core}=
\min\{1,3\max(\delta_0(m_0),\delta_J(m_0))\}$.
Then the actual extinction hazard from any such entering core is
at most $\delta_{\rm core}^N$. For the all-zero and all-$z_1$
cores the certified exponents $-\log\delta_{\rm core}$ are at least,
respectively, $147$ and $28$. Direct CDF diagnostics give
$147.0096861$ and $28.0476946$.

The unchanged regional discovery formula uses

$$
\rho_z=1-\eta(2+20\sqrt2\pi^2),\quad
H=\rho_z(R_0+J)+bV_c=.21776050176732614,
\quad p_J=2\Phi(J/\sigma_J)-1=.1241640336165897.
$$

For an adjacent core differing in one coordinate, let
$\mathcal V_j=|Q_j\cap D|$ be its actual alive target volume.
Its proved per-row discovery coefficient is

$$
p_{ij}=p_J^3\mathcal V_j(2\pi\tau^2)^{-3/2}
 \exp\!\left[-\frac{
 (|z_j-z_i|+R_0+H)^2+2(R_0+H)^2}{2\tau^2}\right].
\tag{KURC.24}
$$

For $z_0\to z_1$, $-\log p_{01}\approx2150.1504804$;
For $z_1\to z_2$ with just one boundary coordinate, use
$\mathcal V_j=(2R_0)^2(R_0+2-z_2)$; the exponent is
approximately $2150.67852$. Every truncated target coordinate
is charged in $\mathcal V_j$.
Discovery of one row has probability at least $1-(1-p_{ij})^N$.
Direct all-row establishment has probability at least $p_{ij}^N$.
Both statements concern the original full-source law with alive
landing in the target; they do not assert subsequent establishment
by active selection from a single discoverer.
:::

:::{prf:proof}
The actual force gives the global pointwise inequalities
$\ell X-A\le g(X)\le\ell X+A$ in each coordinate.
The bounded physical offset obeys $|bU_k|\le bV_c$. For an accepted
row, its own jitter and kinetic Gaussian combine in
$\ell(\mu+\sigma_JZ^J)+\tau Z$ with variance $\sigma_*^2$.
The two one-sided comparisons therefore give (KURC.23), independent
of all other rows' jitters. For an unaccepted row, monotonicity and
oddness bound its mean by $g(m_0)+bV_c$, giving $\delta_0$.
Apply a coordinate union bound. Conditional independence after
complete preparation and the own-jitter replacement argument of
the survival theorem show that the product of the row death
probabilities has expectation at most $\delta_{\rm core}^N$.
This argument conditions on entering sources, never on restricted
future noise.

For $z=0$, take $m_0=R_0$. For $z=z_1$, take
$m_0=z_1+R_0$. The elementary bound $Q(x)\le\phi(x)/x$
for $x>0$, applied to (KURC.23), gives exponents greater than
$147.0062$ and $28.0294$, respectively; these prove the two
conservative integer certificates. Direct CDF evaluation gives
the displayed sharper diagnostics. In each case
$\delta_J$ is the larger one in each case. At the boundary core
$z_2$, $m_0=2$ and this particular sine-amplitude union bound is
vacuous; the global nonvacuous certificate (KURC.16) still applies.
This does not remove the boundary phase from the algorithm.

For discovery use exactly the original regional force profile on
the enlarged core and the tagged *single-row* jitter event.
Its pre-Gaussian mean is within $H$ of $z_i$ coordinatewise.
Infimize the independent kinetic Gaussian density over the original
target cube, then multiply by the volume of its actual intersection
with $D$, to obtain (KURC.24), as in
{prf:ref}`thm-slc-evaluated-basins`. The positive-viscosity first
kick has the same rowwise $V_c$ bound in both normalizations,
even while its matrix depends on every jitter. Own-jitter and
own-position-noise lower indicators prove the displayed count
and discovery consequences. No statement requires $H<R_0$;
that extra inequality is needed only for a useful retention
certificate. Substitution of the actual root values gives the
quoted diagnostics. $\square$
:::

:::{prf:remark} The retained mixing obligation
:label: rem-kurc-phase-mixing-obligation

The native signed regional variance coefficient is
$\lambda=.7989420407893824$, with explicit
$C_{\rm reg}=.20613357968212598$, as proved in
{prf:ref}`thm-kur-evaluated-regional-bound`. Its noise floor
$C_{\rm reg}/(1-\lambda)>1$ exceeds the maximal within-core
position variance $3R_0^2=.01171875$. It is still a valid signed
regional bound, but it does not certify full-law mixing in a
closed core at these parameters. Both unrestricted excursions
and exact between-phase flux remain in the original ledger.

The direct all-row transition budget (KURC.24) has exponent
about $2150$, whereas the certified global killing exponents
are $-\log(1-a_{\rm count})\ge2\cdot10^{-6}$ and
$-\log(1-a_{\rm row})\ge7\cdot10^{-11}$. The existing
direct event therefore does not establish that killing is
negligible on its certified all-row communication time.
Its exponent also exceeds the two improved interior-core
killing exponents above. These comparisons test the existing
certificates, not the actual optimal transition probabilities.

The one-row discovery coefficient is independent of $N$;
its probability $1-(1-p_{ij})^N$ is substantially stronger
than $p_{ij}^N$. Turning discovery into phase mixing still
requires an evaluated complete-update establishment and
residence estimate for a mixed population, including both
reward and diversity fitness, every measured normalizer,
donor destinations, Haar velocity law and both B stages.
The original diversity-only single-discoverer specialization
does not silently discharge that native combined-fitness
obligation. No equal-fitness or near-equal-fitness exclusion
was used to obtain the global survival estimates above.
:::

(sec-kurc-eigenfunction)=
## 5. The precise additional input for the optional eigenfunction route

:::{prf:theorem} A block comparison with a computed mortality cost
:label: thm-kurc-block-eigenfunction-comparison

Let $Q_Nh_N=\alpha_Nh_N$ be a compatible positive bounded
right eigenfunction with $\inf h_N>0$. Take any computed
block length $L_N\ge1$, and write

$$
m_N=[1-(1-a)^N]^{L_N},\qquad
B_N(S,\cdot)=Q_N^{L_N}(S,\cdot)/Q_N^{L_N}1(S).
\tag{KURC.25}
$$

If the actual survived block has a proved Dobrushin coefficient
$r_N<m_N$, then

$$
\frac{\sup h_N}{\inf h_N}
\le1+\frac{1-m_N}{m_N-r_N}.
\tag{KURC.26}
$$

This is a conditional block theorem: a regional positional
variance estimate or a discovery probability is not the
Dobrushin hypothesis for the complete survived block.

For example, a proved full-block estimate $r_N\le r<1$
and $L_N\le C e^{JN}$ suffices for a uniform bound on all
large $N$ when $J< -\log(1-a)$; the remaining finitely many
populations require their actual finite-block checks.
:::

:::{prf:proof}
The global survival floor gives $Q_N^{L_N}1\ge m_N$ and
$\alpha_N^{L_N}\ge m_N$. Normalize $\inf h_N=1$ and set
$H_N=\sup h_N$. Write $v_-=\inf B_Nh_N$ and
$v_+=\sup B_Nh_N$. The Dobrushin hypothesis gives
$v_+-v_-\le r_N(H_N-1)$. At states approaching the infimum
of $h_N$, the eigenfunction equation and the block survival
floor give $v_-\le\alpha_N^{L_N}/m_N$. At states approaching
its supremum, the same equation and survival at most one give
$\alpha_N^{L_N}H_N\le v_+$. Consequently

$$
(\alpha_N^{L_N}-r_N)(H_N-1)
\le\alpha_N^{L_N}(m_N^{-1}-1).
$$

The function $x/(x-r_N)$ decreases for $x>r_N$, and
$\alpha_N^{L_N}\ge m_N>r_N$. This proves (KURC.26).
Finally $1-m_N\le L_N(1-a)^N$, so the stated exponential
comparison makes $m_N\to1$. $\square$
:::

:::{prf:proposition} Phase-dependent killing can matter even when it is rare
:label: prop-kurc-two-phase-algebra

For an exactly lumped two-phase killed chain, suppose its actual
substochastic matrix is

$$
T_N=\begin{pmatrix}
1-\kappa_{0,N}-r_N&r_N\\
r_N&1-\kappa_{1,N}-r_N
\end{pmatrix},\qquad
0<\kappa_{0,N}<\kappa_{1,N},
\tag{KURC.27}
$$

with nonnegative entries. Its principal right eigenfunction
satisfies, with $\Delta_N=\kappa_{1,N}-\kappa_{0,N}$,

$$
\frac{h_{0,N}}{h_{1,N}}=
\frac{\Delta_N+\sqrt{\Delta_N^2+4r_N^2}}{2r_N}
\ge\frac{\Delta_N}{r_N}.
\tag{KURC.28}
$$

Thus if the *actual* phase data obey
$\Delta_N\ge c e^{-IN}$ and $r_N\le C e^{-JN}$ with
$J>I$, the eigenfunction ratio is at least
$(c/C)e^{(J-I)N}$, although both killing probabilities
can tend to zero exponentially. This statement is an algebraic
conditional phase example. Neither exact lumping nor these
actual relative rates has been established for the native gas
by the bounds in this record.
:::

:::{prf:proof}
The larger eigenvalue of (KURC.27) is
$1-(\kappa_{0,N}+\kappa_{1,N}+2r_N
-\sqrt{\Delta_N^2+4r_N^2})/2$. Its positive eigenvector
equations give (KURC.28) directly. Substitute the stated
relative rates. This example concerns phase-specific survival
weights, not a raw fixed-slot discrepancy or a violation of
the original alive-centered Keystone pressure. $\square$
:::

:::{prf:remark} Established output and unresolved inference
:label: rem-kurc-output-status

The complete primitive survival bounds (KURC.11)--(KURC.21)
are proved for the native Rastrigin force, both viscous
normalizations and all nonextinct entering configurations,
including mixed and boundary phases. They provide finite
population-independent survivor multipliers, inverse-alive
moments and exponentially small all-dead probabilities.
Their proof retains the original acceptance law rather than
using an acceptance-frequency premise.

The global eigenfunction oscillation conclusion remains
conditional on the full phase-resolved survived-block input
in (KURC.25)--(KURC.26), or on an alternative complete
renewal comparison with all retained source and velocity
information. Existing direct phase-transfer lower bounds
do not meet its mortality-versus-communication test.
This is the precise missing inference; no impossibility of
native phase establishment or global eigenfunction control
is inferred from the failed direct-event comparison.
The physical survivor entropy route may use (KURC.15)
directly and does not require introducing a Doob transform.
:::
