# Complete native stationary fluctuations, response and time covariance

(sec-nfs-register)=
## 1. The complete existing transition instrument

:::{prf:definition} Full stationary fluctuation register
:label: def-nfs-register

Retain every execution, algorithm, landscape, sampled normalization,
fitness, current donor, revival, simultaneous copy, component-Haar,
jitter, both kicks, cap, boundary, status, arithmetic, recording and
calibration field of {prf:ref}`def-nje-register`.
The active regime is its existing quadratic dense count phase,
original $q,s,\sigma_J>0$, $\kappa_A>0$,
$0<V\le\min\{V_0,1\}$, $\nu\le\nu_g$, and derived $r<1$.
Parameters are fixed independently of $N$. Smaller populations
retain their primitive finite-QSD common-part certificates.

Let $S$ be the full physical marked array $(x,v,a)$ and $R$
the complete original one-update source/innovation record.
Its surviving instrument $\Omega_N(S,dR,dT)$ has state marginal
$Q_N$. The actual existing Doob instrument is

$$
\Omega_N^e(S,dR,dT)
=\frac{e_N(T)}{\alpha_Ne_N(S)}\Omega_N(S,dR,dT),\qquad
P_N^e(S,dT)=\int\Omega_N^e(S,dR,dT).
\tag{NFS.1}
$$

Use $\pi_N=e_N\nu_N/\nu_N(e_N)$.
For $N\ge\widehat N_*$ use $T_N=\widehat T_N$ and $\Delta_N<1$
of (NJE.9). For the remaining admitted populations use
$T_N=1$, $\Delta_N=1-a_{F,N}<1$ from (NJE.24).
Every actual row pair of $(P_N^e)^{T_N}$ has TV distance at most
$\Delta_N$.

Let $H_N(S,R,T)\in\mathbb R^m$ be a fixed time-homogeneous
complete-update observation with $|H_N|\le B_N<\infty$.
It may consume original hard color masks, source/fitness/component
records, both force inputs and final geometry, provided its declared
projection is bounded. All their correlations remain in the instrument.
Absolute clock/generation labels stay in the record; a statistic
depending explicitly on a growing clock is not assigned stationary
time-homogeneous conclusions. Every consumed calibration and
observation parameter remains in $H_N$.
A bounded empirical statistic $N^{-1}\sum_i\psi_i$ has $B_N=B$;
its $\sqrt N$ version has $B_N=\sqrt N B$.
:::

(sec-nfs-poisson)=
## 2. Exact drift, Poisson response and full-update bracket

:::{prf:theorem} Complete native stationary martingale decomposition
:label: thm-nfs-full-decomposition

Define

$$
g_N(S)=\int H_N(S,R,T)\,\Omega_N^e(S,dR,dT),\qquad
\theta_N=\pi_Ng_N,\qquad
u_N=\sum_{k=0}^\infty(P_N^e)^k(g_N-\theta_N).
\tag{NFS.2}
$$

This series converges uniformly, has $\pi_Nu_N=0$, and solves
$u_N-P_N^eu_N=g_N-\theta_N$. Its primitive estimates are

$$
\left\|u_N-\sum_{k=0}^{K-1}(P_N^e)^k(g_N-\theta_N)\right\|_\infty
\le\frac{2B_NT_N}{1-\Delta_N}
                 \Delta_N^{\lfloor K/T_N\rfloor},\qquad
U_N:=\frac{2B_NT_N}{1-\Delta_N}\ge\|u_N\|_\infty.
\tag{NFS.3}
$$

On a stationary actual Doob record path put

$$
D_{N,n}=H_N(S_{n-1},R_n,S_n)-\theta_N
                 +u_N(S_n)-u_N(S_{n-1}),\qquad
M_{N,k}=\sum_{n=1}^kD_{N,n}.
\tag{NFS.4}
$$

Then $M_N$ is a complete-history martingale and

$$
\sum_{n=1}^k(H_{N,n}-\theta_N)
=M_{N,k}+u_N(S_0)-u_N(S_k).
\tag{NFS.5}
$$

Its exact complete conditional bracket is

$$
A_N(S)=\int DD^\top\,\Omega_N^e(S,dR,dT),\qquad
\langle M_N\rangle_k=\sum_{n=1}^kA_N(S_{n-1}),\qquad
\Sigma_N=\pi_NA_N.
\tag{NFS.6}
$$

Thus the actual full drift and covariance are identified by its
complete instrument. With $L_N=2B_N+2U_N$,
$|D_{N,n}|\le L_N$ and $\operatorname{tr}\Sigma_N\le L_N^2$.
:::

:::{prf:proof}
Common-mass block coupling contracts bounded-test oscillation by
$\Delta_N$. Its $\pi_N$ mean is zero, so
$\|(P_N^e)^k(g_N-\theta_N)\|_\infty
\le2B_N\Delta_N^{\lfloor k/T_N\rfloor}$.
Summing its geometric blocks proves (NFS.3) and uniform convergence.
Telescoping gives its Poisson equation and zero mean.
The conditional mean of (NFS.4) is
$g_N-\theta_N+P_N^eu_N-u_N=0$.
Its conditional second moment is exactly (NFS.6).
Summing the endpoint differences proves (NFS.5).
The norm bound follows from $|\theta_N|\le B_N$ and (NFS.3).
No stage independence is presumed.
:::

:::{prf:lemma} Each original stage contributes its exact conditional bracket
:label: lem-nfs-stage-brackets

Reveal original one-update blocks in their actual measurement,
donor/gate, addressed component-Haar, recipient-jitter, OU and
terminal-position order, including every auxiliary choice.
With entering $S$ fixed let
$\mathcal E_0\subset\cdots\subset\mathcal E_J$ include these
blocks and the deterministic stages they trigger. Set

$$
D^{(\ell)}=E_e[D\mid S,\mathcal E_\ell]
                   -E_e[D\mid S,\mathcal E_{\ell-1}].
\tag{NFS.7}
$$

Then $D=\sum_\ell D^{(\ell)}$ and

$$
A_N(S)=\sum_\ell E_e[D^{(\ell)}D^{(\ell)\top}\mid S].
\tag{NFS.8}
$$

Its raw-prefix Doob weight is
$E_{\rm raw}[e_N(T)\mathbf1_{\rm survive}
 \mid S,\mathcal E_\ell]/(\alpha_Ne_N(S))$.
Consequently every conditional expectation is an explicit original
draw integral with this weight, retaining its sampled normalization,
full component Haar variables and both coupled force evaluations.
It does not factorize the Doob-stage law into independent blocks.
:::

:::{prf:proof}
The final record makes $D$ measurable and its entering conditional
mean is zero. Its conditional-expectation martingale telescopes
into (NFS.7). Different increments are conditionally orthogonal
by the tower property, proving (NFS.8).
Integrate (NFS.1) over the unobserved original raw draws to obtain
the stated prefix weight.
:::

(sec-nfs-time-clt)=
## 3. A full native-time central limit theorem

:::{prf:theorem} Complete time fluctuations and a primitive characteristic bound
:label: thm-nfs-time-clt

For fixed admitted $N$ and native update count $n\to\infty$,

$$
\frac1{\sqrt n}\sum_{j=1}^n(H_{N,j}-\theta_N)
\Longrightarrow N(0,\Sigma_N).
\tag{NFS.9}
$$

The exact zero or singular covariance regimes are allowed.
For $v\in\mathbb R^m$, put $\ell_v=|v|L_N$,
$C_{{\rm erg},N}=1+8T_N/(1-\Delta_N)$ and

$$
\mathcal E_{N,n}(v)=\frac{2|v|U_N}{\sqrt n}
+e^{\ell_v^2}
\left[\frac{\ell_v^2\sqrt{C_{{\rm erg},N}}}{2\sqrt n}
+\frac{\ell_v^3}{6\sqrt n}+\frac{\ell_v^4}{8n}\right].
\tag{NFS.10}
$$

The characteristic-function error from its declared Gaussian is at
most $\min\{2,\mathcal E_{N,n}(v)\}$.
Its physical observation duration is $nt_*h$ and covariance rate
$\Sigma_N/(t_*h)$. This is a complete-update native-time CLT,
including original hard masks and every preparation stage.
It does not assert an instantaneous population CLT.
:::

:::{prf:proof}
For bounded state $f$, $|f|\le F$, stationarity and block coupling
give
$|\operatorname{Cov}(f(S_0),f(S_k))|
\le4F^2\Delta_N^{\lfloor k/T_N\rfloor}$.
Summing the covariance identity proves

$$
E\left|\frac1n\sum_{j=0}^{n-1}f(S_j)-\pi_Nf\right|^2
\le C_{{\rm erg},N}F^2/n.
\tag{NFS.11}
$$

For $Y_j=v\cdot D_{N,j}/\sqrt n$, its conditional variance $a_j$
is at most $\ell_v^2/n$. Taylor and its zero conditional mean give

$$
\left|e^{a_j/2}E[e^{iY_j}\mid\mathcal T_{j-1}]-1\right|
\le e^{\ell_v^2/(2n)}
 [\ell_v^3/(6n^{3/2})+\ell_v^4/(8n^2)].
$$

Indeed $|e^x(1-x)-1|\le x^2e^x/2$ for $x\ge0$ gives its
second term. Iterating the compensated exponential, whose modulus
is bounded by $e^{\ell_v^2/2}$, shows
$|Ee^{iv\cdot M_{N,n}/\sqrt n+\sum_ja_j/2}-1|$
is at most $e^{\ell_v^2}$ times the last two terms in (NFS.10).
Its bracket sum is the average of
$v^\top A_N(S_j)v\le\ell_v^2$. By (NFS.11) its $L^1$
discrepancy from $v^\top\Sigma_Nv$ is at most
$\ell_v^2\sqrt{C_{{\rm erg},N}/n}$.
The mean value bound for the real exponential gives the first
bracket term with the displayed larger $e^{\ell_v^2}$ factor.
The endpoint remainder in (NFS.5) costs at most
$2|v|U_N/\sqrt n$.

For fixed $N$ this error tends to zero. Martingale second moments
and its bounded remainder give tightness. Gaussian smoothing and
Fourier inversion identify each subsequential law as the stated
possibly singular Gaussian; removing that auxiliary smoothing proves
(NFS.9). No Gaussian preparation limit was assumed.
:::

:::{prf:theorem} The original QSD history conditioned once on long survival
:label: thm-nfs-survivor-time-clt

Start the original killed gas from $\nu_N$, retain its complete
records and condition once on survival through $n$.
For fixed admitted $N$, (NFS.9) holds under this original history
with the same complete drift and covariance.
The drift is its long-survival bulk value $\theta_N$;
a different one-update QSD mean cannot replace that centering.
:::

:::{prf:proof}
The actual history telescope (NUE.25) gives terminal density
$\nu_N(e_N)/e_N(S_n)$ relative to the stationary Doob path.
For fixed $N$, $m_N=\min e_N>0$.
Write $w=1/e_N$; then $\pi_Nw=1/\nu_N(e_N)$.
Discard the last $b_n=\lfloor n^{1/4}\rfloor$ observations.
Their centered normalized sum is at most $2B_Nb_n/\sqrt n\to0$.
The earlier sum is measurable at time $n-b_n$, while

$$
\|(P_N^e)^{b_n}w-\pi_Nw\|_\infty
\le2m_N^{-1}\Delta_N^{\lfloor b_n/T_N\rfloor}\to0.
$$

Condition the terminal tilt against its bounded characteristic test.
This factorizes with vanishing error and its normalization cancels
exactly. The stationary CLT and discarded remainder prove the
same Gaussian for the original once-conditioned history.
Every attached stage record remains in this telescope.
:::

:::{prf:theorem} Complete functional native-time central limit theorem
:label: thm-nfs-functional-time-clt

For fixed admitted $N$ and finite $T>0$, linearly interpolate
the original centered additive record path at the times $j/n$:

$$
X_{N,n}(t)=\frac1{\sqrt n}\left[
 \sum_{j=1}^{\lfloor nt\rfloor}(H_{N,j}-\theta_N)
 +(nt-\lfloor nt\rfloor)
             (H_{N,\lfloor nt\rfloor+1}-\theta_N)\right].
\tag{NFS.32}
$$

Under the stationary complete Doob instrument,

$$
X_{N,n}\Longrightarrow B_{\Sigma_N}
       \quad\text{in }C([0,T],\mathbb R^m),\qquad
E[B_{\Sigma_N}(s)B_{\Sigma_N}(t)^\top]
                 =\min(s,t)\Sigma_N.
\tag{NFS.33}
$$

The same functional limit holds for the original
$\nu_N$-started complete history conditioned once on survival
through $\lceil nT\rceil$. Zero and singular covariance regimes
are retained. In native physical duration the covariance rate
is $\Sigma_N/(t_*h)$.
:::

:::{prf:proof}
Let $Z_{N,n}$ be the linear interpolation of
$M_{N,j}/\sqrt n$ from (NFS.4). The interpolated Poisson
endpoint remainder is at most $2U_N/\sqrt n$ uniformly on
$[0,T]$, by (NFS.5). It therefore suffices to prove the
assertion for this actual complete-record martingale.

Its predictable bracket, divided by $n$, converges uniformly
in probability to $t\Sigma_N$ on $[0,T]$.
For each fixed time this follows from (NFS.11) applied
to $v^\top A_Nv$ and finite-coordinate polarization.
For uniformity first use a finite time grid.
Between two neighboring grid times each quadratic-form bracket
is increasing and its increment is at most
$|v|^2L_N^2$ times the grid length plus one mesh interval.
The deterministic limiting bracket has the same Lipschitz bound.
Let the grid spacing tend to zero after $n\to\infty$.
This proves the asserted uniform bracket convergence without
assuming independence of the preparation records.

For finite-dimensional distributions use arbitrary deterministic
vector coefficients, constant on finitely many disjoint time
intervals, against the increments of $M_N/\sqrt n$.
The Taylor conditional-characteristic calculation in the proof
of (NFS.10) applies with that interval's coefficient at each
original update. Its summed cubic remainder is $O(n^{-1/2})$
and quartic remainder $O(n^{-1})$, with constants depending
only on $T,L_N$ and the finite coefficient list.
The preceding bracket convergence gives limiting quadratic form
$\sum_j(t_j-t_{j-1})v_j^\top\Sigma_Nv_j$.
The compensated exponential consequently gives the Gaussian
characteristic function with these independent interval increments.
Its finite-dimensional laws are precisely those of
$B_{\Sigma_N}$, including a singular $\Sigma_N$.

For tightness, every block of $a$ original martingale increments
has the explicit fourth moment

$$
E\left|\sum_{j=b+1}^{b+a}D_{N,j}\right|^4
                   \le7L_N^4a^2.
\tag{NFS.34}
$$

To verify it put $M_l=\sum_{j=b+1}^{b+l}D_{N,j}$.
Expanding $|M_l+D|^4$ and conditioning removes the term
$4|M_l|^2M_l\cdot D$. The other increment terms are at most
$6L_N^2E|M_l|^2+4L_N^3E|M_l|+L_N^4$.
Martingale orthogonality gives $E|M_l|^2\le lL_N^2$,
so their sum over $0\le l<a$ is at most
$L_N^4[3a(a-1)+4a\sqrt a+a]\le7L_N^4a^2$.
This vector-norm calculation retains dependent cubic terms.

For $|t-s|\ge1/n$, split a linearly interpolated martingale
increment into its complete update block and its two partial
endpoint increments. The former has fourth moment at most
$7L_N^4|t-s|^2$, and the latter have combined norm at most
$2L_N/\sqrt n$. The fourth-power triangle inequality gives

$$
E|Z_{N,n}(t)-Z_{N,n}(s)|^4
                   \le184L_N^4|t-s|^2.
\tag{NFS.35}
$$

For $|t-s|<1/n$ there are at most two partial increments,
whose combined norm is at most $L_N\sqrt n|t-s|$;
their fourth power is at most $L_N^4|t-s|^2$.
Thus (NFS.35) holds for all times and all $n$.
For completeness its dyadic intervals of level $k$ on $[0,T]$
give, by Markov and the union bound, probability at most
$184L_N^4T^2\eta^{-4}2^{-k/2}$ that some interval increment
exceeds $\eta2^{-k/8}$.
Summing over $k\ge K$ and chaining the nested dyadic points
gives a common vanishing modulus bound as $K\to\infty$.
The interpolated paths are continuous and start at zero;
these bounds prove tightness in $C([0,T],\mathbb R^m)$.
Finite-dimensional identification then proves (NFS.33).

For the original once-conditioned history set
$a_n=\lceil nT\rceil$ and
$b_n=\lfloor n^{1/4}\rfloor$.
Freeze the interpolated additive path after
$(a_n-b_n)/n$. The uniform change is at most
$2B_N(b_n+1)/\sqrt n\to0$.
The frozen path is measurable before the last $b_n$ updates.
The exact terminal weight $\nu_N(e_N)/e_N(S_{a_n})$
therefore differs, on this entire earlier path law, from
its constant normalized mean by at most
$2\nu_N(e_N)m_N^{-1}\Delta_N^{\lfloor b_n/T_N\rfloor}$.
This is a TV bound for the earlier path, not a separate
coordinatewise replacement. It tends to zero for fixed $N$.
Remove the deterministic frozen-path error and apply the
stationary functional limit. The original histories have the
same full Brownian drift/bracket limit.
:::

(sec-nfs-green-kubo)=
## 4. Exact complete-record Green--Kubo covariance

:::{prf:theorem} Complete covariance series and its certified tail
:label: thm-nfs-green-kubo

Let $h_{N,n}=H_{N,n}-\theta_N$ and
$C_{N,k}=E_{\pi_N}[h_{N,1}h_{N,k+1}^{\top}]$. Then

$$
\Sigma_N=E[h_{N,1}h_{N,1}^{\top}]
+\sum_{k=1}^\infty(C_{N,k}+C_{N,k}^{\top}),\qquad
\|C_{N,k}\|_{\rm op}\le4B_N^2
                    \Delta_N^{\lfloor(k-1)/T_N\rfloor}.
\tag{NFS.12}
$$

The omitted matrix tail after $K$ is at most
$8B_N^2T_N\Delta_N^{\lfloor K/T_N\rfloor}/(1-\Delta_N)$.
In direction $v$, the covariance is zero exactly when
$v\cdot D_{N,1}=0$ almost surely under the complete stationary
instrument, equivalently the exact coboundary
$v\cdot h_{N,1}=v\cdot u_N(S_0)-v\cdot u_N(S_1)$.
Thus a nonzero terminal component alone is not substituted for
complete-update nondegeneracy.
:::

:::{prf:proof}
Condition the later update on its entering state.
Its centered mean is $g_N-\theta_N$; its prediction at the
earlier terminal state at lag $k$ is
$(P_N^e)^{k-1}(g_N-\theta_N)$.
Its norm is at most $2B_N\Delta_N^{\lfloor(k-1)/T_N\rfloor}$,
and the earlier centered record has norm at most $2B_N$.
This gives absolute summability and the stated tail.
Expand the second moment of $n$ centered records, divide by $n$
and use those summable lags. By (NFS.5) its martingale second
moment is $n\Sigma_N$ and its bounded remainder changes the
divided second moment by $O(n^{-1/2})$. Therefore the series
equals the complete bracket in (NFS.6).
Finally $\Sigma_N=EDD^\top$, so its directional form vanishes
precisely when that directional $D$ is zero almost surely.
:::


(sec-nfs-uniform-covariance)=
## 5. Population-uniform covariance for the actual Lipschitz state channel

:::{prf:definition} A consumed physical-state observation budget
:label: def-nfs-lipschitz-channel

Let $f_N(S)$ be a scalar physical-state empirical observation with
$|f_N|\le B$ and global Lipschitz coefficient at most $L$ in the
actual averaged marked metric $\overline d_\omega$, with $B,L$
fixed independently of $N$. This is an explicit observation test,
not regularity of an unknown stationary law.
For example $f_N=N^{-1}\sum_i\psi(x_i,v_i,a_i)$ qualifies with
the declared bounded row-test coefficients.
A smooth cutoff of an actual current-state source/ray descriptor
qualifies when its literal derivative and cutoff budget are finite.
A matched pre-terminal B2 color record is a complete update instrument;
it is not silently assigned this state-only Lipschitz budget.

Set $C=C_{\rm stat}L^2$, where $C_{\rm stat}$ is the complete
primitive constant of Chapter 30. Retain
$q_N\le\bar q=(1+r)/2$, $\zeta_N=e^{-uN}$,
$u=a_0/8$, $\varepsilon_N=(1-a_0)^N$,
$m_N=1-\widehat b_N\ge7/8$,
$d_N=\widehat b_N/(1-\widehat b_N)$ and
$K_N=\widehat T_N\le\widehat C_TN$.
All the same component, normalizer and kinetic parameters remain in
these already evaluated constants.
:::

:::{prf:lemma} A stopped semigroup has a genuine approximate Lipschitz extension
:label: lem-nfs-semigroup-extension

For $N\ge\widehat N_*$ and $k\ge1$ let $f_N=0$ at the actual
cemetery and $G_k=Q_N^kf_N$. There is a bounded globally Lipschitz
$G_k^\sharp$, $|G_k^\sharp|\le B$, with coefficient $Lq_N^k$
such that

$$
\|G_k-G_k^\sharp\|_{L^2(\nu_N)}
\le4Bk\zeta_N+2B\sqrt{\zeta_N}.
\tag{NFS.13}
$$

Consequently

$$
N|\operatorname{Cov}_{\nu_N}(f_N,Q_N^kf_N)|
\le Cq_N^k+
2B\sqrt{NC}(2k\zeta_N+\sqrt{\zeta_N}).
\tag{NFS.14}
$$
:::

:::{prf:proof}
For two entering states in the actual $H_N$, run the already
constructed $q_N$ full marked coupling until either path fails its
alive floor. The uniform output binomial bound gives failure
probability at most $2k\zeta_N$. Extinction is one such failure,
so no restart branch is inserted. On its surviving good part,
the expected distance is at most $q_N^k\overline d_\omega(S,S')$.
Therefore

$$
|G_k(S)-G_k(S')|
\le Lq_N^k\overline d_\omega(S,S')+4Bk\zeta_N .
$$

Take the infimum of
$G_k(T)+Lq_N^k\overline d_\omega(S,T)$ over $T\in H_N$,
then clip its value to $[-B,B]$. It is globally Lipschitz with
the displayed coefficient and differs from $G_k$ by at most
$4Bk\zeta_N$ on $H_N$. Its actual QSD has
$\nu_N(H_N^c)\le\zeta_N$ at the inherited $N\ge4/a_0$
threshold: the exact half-mean raw Chernoff exponent is
$c_0=(1-\log2)/2>1/8$, and QSD conditioning costs at most
$(1-e^{-a_0N})^{-1}$. For $x=a_0N\ge4$,
$-\log(1-e^{-x})\le2e^{-x}\le(c_0-1/8)x$,
so its factor is absorbed into that stronger exponent.
Outside $H_N$ pay $2B$.
This proves (NFS.13).
The stationary variance theorem gives
$\operatorname{Var}_{\nu_N}f_N\le C/N$ and
$\operatorname{Var}_{\nu_N}G_k^\sharp\le Cq_N^{2k}/N$.
Cauchy--Schwarz on this covariance and on its remainder proves
(NFS.14). Discrete statuses remain part of the true metric.
:::

:::{prf:theorem} Complete population-uniform Lipschitz Green--Kubo bound
:label: thm-nfs-uniform-green-kubo

For the stationary actual Doob state path put
$Z_{N,n}=\sqrt N[f_N(S_n)-\pi_Nf_N]$.
Its absolutely convergent covariance series satisfies

$$
\sigma_N^2=
N\operatorname{Var}_{\pi_N}f_N+
2N\sum_{k=1}^\infty
 \operatorname{Cov}_{\pi_N}(f_N(S_0),f_N(S_k))
\le\mathcal C_N,
$$
$$
\begin{aligned}
\mathcal C_N={}&C/m_N+\frac{2C\bar q}{1-\bar q}\\
&+4B\sqrt{NC}
 [\zeta_NK_N(K_N+1)+K_N\sqrt{\zeta_N}]\\
&+16NB^2[K_Nd_N+
             \varepsilon_NK_N(K_N+1)/2]\\
&+\frac{8NB^2K_N\Delta_N}{1-\Delta_N}.
\end{aligned}
\tag{NFS.15}
$$

It has the explicit asymptotic bound
$\limsup_N\mathcal C_N\le C+2C\bar q/(1-\bar q)$.
A primitive uniform upper bound for every $N\ge\widehat N_*$ is

$$
\begin{aligned}
\mathcal C_\infty={}&\frac87C+\frac{2C\bar q}{1-\bar q}\\
&+4B\sqrt C\,\widehat C_T(\widehat C_T+1)
                           M_{5/2}(u)
 +4B\sqrt C\,\widehat C_TM_{3/2}(u/2)\\
&+\left(\frac{256}7+\frac{256}5\right)
 B^2\widehat C_T\widehat C_\theta M_3(u)
 +8B^2\widehat C_T(\widehat C_T+1)M_3(a_0),
\qquad
M_s(v)=\left(\frac{s}{ev}\right)^s .
\end{aligned}
\tag{NFS.16}
$$

For the finitely many smaller admitted populations include
the finite maximum of $NB^2(1+8/a_{F,N})$.
This proves an actual population-uniform complete temporal
fluctuation covariance for this channel, with no stationary LSI
or presumed population Gaussian law.
:::

:::{prf:proof}
The entire stationary Doob history differs from the original
$\nu_N$-started killed path through lag $k$ by TV at most
$d_N+k\varepsilon_N$, by the terminal telescope and actual
survival bound. A covariance of two $[-B,B]$ tests changes by
at most $8B^2$ times this TV: its product moment and its two
means are bounded directly by that range.
Hence (NFS.14) implies, for every $k$,

$$
N|\operatorname{Cov}_{\pi_N}(f_N(S_0),f_N(S_k))|
\le Cq_N^k+2B\sqrt{NC}(2k\zeta_N+\sqrt{\zeta_N})
                  +8NB^2(d_N+k\varepsilon_N).
$$

Sum this only over $1\le k\le K_N$.
For the diagonal use the exact density bound
$\operatorname{Var}_{\pi_N}f_N\le C/(m_NN)$.
Beyond that finite window use the actual Doob block mixing,
which gives covariance at most
$4B^2\Delta_N^{\lfloor k/K_N\rfloor}$.
Its summed tail is at most
$4B^2K_N\Delta_N/(1-\Delta_N)$.
This proves (NFS.15). No constant small error has been summed
over an infinite time horizon.

For the explicit uniform formula use $K_N\le\widehat C_TN$,
$m_N\ge7/8$, $1-\Delta_N\ge5/7$,

$$
d_N\le(16/7)\widehat C_\theta Ne^{-uN},\qquad
\Delta_N\le(32/7)\widehat C_\theta Ne^{-uN},\qquad
\varepsilon_N\le e^{-a_0N}.
$$

The four remaining sums in (NFS.15) are bounded respectively by
their coefficients times $N^{5/2}e^{-uN}$,
$N^{3/2}e^{-uN/2}$,
$(256/7+256/5)B^2\widehat C_T\widehat C_\theta N^3e^{-uN}$
and $8B^2\widehat C_T(\widehat C_T+1)N^3e^{-a_0N}$.
Elementary maximization gives
$\sup_{x\ge0}x^se^{-vx}=M_s(v)$, proving (NFS.16).
Those remainders vanish as $N\to\infty$, proving its sharper
asymptotic bound.
For smaller populations their one-step Doob common part bounds
the same covariance sum by $NB^2(1+8/a_{F,N})$.
This is a finite maximum of primitive formulas.
:::

:::{prf:remark} What the uniform channel proves
:label: rem-nfs-uniform-channel-scope

For $H_N(S,R,T)=\sqrt N f_N(T)$, the complete time CLT above has
the covariance bounded by (NFS.15)--(NFS.16), uniformly over
the permitted populations. It includes every algorithmic stage
through the full state transition. For a native matched B2
color/source update instrument the exact full martingale and
time CLT remain valid, but this particular uniform state-channel
bound is used only after its actual observation variance and
conditional prediction continuity have been derived.
Hard descriptor masks are not assigned global Lipschitz budgets
from their qualitative null boundaries.

For the positive reference witness, $B=L=1$ gives
$C=C_{\rm stat}=3.025074197256319\ldots\,10^{30}$,
$\bar q=3/4$, and the leading uniform term is $(50/7)C$.
The remaining terms in (NFS.16) are explicit positive primitive
corrections. Substitution of the same exact witness gives
$\widehat C_T\simeq25.198328674$,
$\widehat C_\theta\simeq47.396657347$ and
$\mathcal C_\infty\simeq2.160767284\,10^{31}$,
$\log\mathcal C_\infty\simeq72.15060127$.
These decimals diagnose the displayed exact formula.
The unchanged larger-cap reference fails the
inherited contraction test and is not assigned this uniform
covariance or phase conclusion.
The complete instantaneous $N\to\infty$ fluctuation law and its
full mean-field linearization retain their distinct obligations;
the proved time CLT is not substituted for them.
:::


(sec-nfs-source-response)=
## 6. Physical-stage Gaussian-parameter response and original innovation shifts

:::{prf:definition} An existing Gaussian parameter changed for one update
:label: def-nfs-source-pulse

Change one consumed Gaussian parameter for one designated update,
keeping the original initial law and every other update at its
recorded parameters. This is the derivative of the existing
algorithm's source law at that update, not a new noise or a
permanently altered stationary algorithm.
The physical Gaussian-stage coordinates have their original
densities, and every subsequent gate, force, cap, mask, geometry
and calibration is recomputed from those coordinates.
For an amplitude or damping change, the density-score conclusion
uses a fixed measurable observation of these physical coordinates
and their original downstream maps, with no remaining explicit
parameter dependence. A retained standardized innovation is
reconstructed as $(z-cv_1)/q$ or displacement divided by its
amplitude; that reconstruction depends on the parameter. Its
actual derivative must be retained, as must every explicit
parameter, calibration, source-address metadata or propagated
history dependence of the readout. An arbitrary measurable
function of such reconstructions is not assigned an amplitude
response by a density score alone.

For the existing isotropic count register write
$c=e^{-\gamma h}$ and
$q=b_O[(1-c^2)/(2\gamma)]^{1/2}$, with $\gamma,b_O>0$.
The amplitude-coordinate scores are

$$
\begin{aligned}
\mathcal S_{\sigma_x}
 &=\sigma_x^{-1}\sum_i(|Z_i^{\rm pos}|^2-d),\\
\mathcal S_{\sigma_J}
 &=\sigma_J^{-1}\sum_i I_i(|Z_i^J|^2-d),\\
\mathcal S_{b_O}
 &=b_O^{-1}\sum_i(|\xi_i|^2-d),\\
\mathcal S_{\gamma\mid b_O}
 &=\sum_i\left[
 -\frac{hc}{q}v_{1i}\cdot\xi_i+
 \left(\frac{hc^2}{1-c^2}-\frac1{2\gamma}\right)
                       (|\xi_i|^2-d)\right].
\end{aligned}
\tag{NFS.17}
$$

They use the original addressed innovations; $I_i$ is the actual
pre-jitter application mark and $v_{1i}$ its actual first-kick
velocity. Unconsumed jitter need not be scored.
Here $b_O$ is the consumed derived diffusion amplitude; its
fixed-amplitude damping derivative is not automatically the
configured damping-knob derivative. For the executed thermostat
tags use the literal floor $\epsilon_{\rm th}=10^{-12}$:

$$
\beta_{\rm eff}=
\begin{cases}
\max\{\beta,\epsilon_{\rm th}\},&\text{manual},\\
\max\{2\max(\gamma,\epsilon_{\rm th})/
        \max(\sigma_v^2,\epsilon_{\rm th}),\epsilon_{\rm th}\},
                                      &\text{automatic},
\end{cases}
\quad
q^2=(1-c^2)/\beta_{\rm eff},\quad
b_O^2=2\gamma/\beta_{\rm eff}.
$$

On every strict executed floor branch, a configured parameter
$\vartheta$ therefore has its exact OU score

$$
\mathcal S_\vartheta^{O}
=\sum_i\left[
 (\partial_\vartheta c/q)v_{1i}\cdot\xi_i
 +\partial_\vartheta\log q\,(|\xi_i|^2-d)\right],
\quad
\partial_\vartheta\log q
=\mathbf1_{\{\vartheta=\gamma\}}\frac{hc^2}{1-c^2}
                         -\tfrac12\partial_\vartheta\log\beta_{\rm eff}.
$$

Manual $\beta>\epsilon_{\rm th}$ gives
$\partial_\beta\log q=-1/(2\beta)$ and
$\partial_\gamma\log q=hc^2/(1-c^2)$; its unused $\sigma_v$
has zero response. Automatic, outer-floor-inactive,
$\gamma>\epsilon_{\rm th}$ and $\sigma_v^2>\epsilon_{\rm th}$
give $\partial_\gamma\log q=hc^2/(1-c^2)-1/(2\gamma)$,
$\partial_{\sigma_v}\log q=1/\sigma_v$, and zero response
to unused manual $\beta$. An active inner floor removes that
inner derivative; an active outer floor removes both derivatives
of $\beta_{\rm eff}$. At a literal floor equality use its actual
one-sided derivatives; no two-sided derivative is assigned
unless they agree. These are the implemented thermostat branches.
Every other declared direction uses its actual chain rule.
Its raw square moment has primitive bound

$$
\begin{aligned}
E_{\rm raw}\mathcal S_{\sigma_x}^2&=2dN/\sigma_x^2,\\
E_{\rm raw}\mathcal S_{\sigma_J}^2&\le2dN/\sigma_J^2,\\
E_{\rm raw}\mathcal S_{b_O}^2&=2dN/b_O^2,\\
E_{\rm raw}\mathcal S_{\gamma\mid b_O}^2
&\le N\left[(hc/q)^2V_1^2+
2d\left(\frac{hc^2}{1-c^2}-\frac1{2\gamma}\right)^2\right],
\quad V_1=V_c+t\lambda(R_D+\sigma_J\sqrt d).
\end{aligned}
\tag{NFS.18}
$$

For a configured OU direction replace that last bound by
$N[(\partial_\vartheta c/q)^2V_1^2+
2d(\partial_\vartheta\log q)^2]$.

These formulas apply where the displayed original amplitudes
are positive. At a zero-amplitude branch the logarithmic density
score is not assigned a finite value or this regular response.
:::

:::{prf:theorem} Exact physical-record weak response and its native long-survival limit
:label: thm-nfs-full-source-response

Let $\mathcal S$ be one score above. Let $O_\vartheta$ be a bounded
history observation through horizon $n$ containing the pulse,
expressed in physical Gaussian-stage coordinates. Suppose its
actual explicit-coordinate derivative $\dot O$ exists in $L^1$
under a local integrable Gaussian-polynomial envelope; this
includes $\dot O=0$ for every fixed bounded measurable physical
readout with parameter-independent downstream map.
Under the original killed path conditioned once on survival,
the exact derivative is

$$
\left.\partial_\vartheta E_{\nu_N,\vartheta}
 [O_\vartheta\mid\text{survival through }n]\right|_{\vartheta=0}
=\operatorname{Cov}_{\nu_N,\rm survive\ n}(O,\mathcal S)
       +E_{\nu_N,\rm survive\ n}\dot O.
\tag{NFS.19}
$$

Thus fixed physical hard boundaries and status transitions are
included in its weak law derivative. If $O,\dot O$ consume only a fixed first
$K$ updates and the pulse is in that window, then as $n\to\infty$
this derivative tends to
$\operatorname{Cov}_{\pi_N}^e(O,\mathcal S)+E_{\pi_N}^e\dot O$.
No derivative of an assumed smooth descriptor density is used.

For a parameter-independent physical-record observation $H_N$
of Section 1, the density part of its stationary pulse causal
responses obeys

$$
\sum_{j=1}^\infty
 \operatorname{Cov}_{\pi_N}^e(H_{N,j},\mathcal S_1)
=E_{\pi_N}^e[
 \mathcal S_1(H_{N,1}-\theta_N+u_N(S_1))],
\tag{NFS.20}
$$

with absolute norm bounded by
$(2B_N+U_N)(E_{\pi_N}^e\mathcal S_1^2)^{1/2}$.
Its omitted future tail after $K$ is at most

$$
\frac{2B_NT_N}{1-\Delta_N}
 (E_{\pi_N}^e\mathcal S_1^2)^{1/2}
                 \Delta_N^{\lfloor(K-1)/T_N\rfloor}.
\tag{NFS.21}
$$

The score moment is bounded by its raw value in (NFS.18)
divided by $\alpha_N\min e_N$, a primitive positive denominator.
For $N\ge\widehat N_*$ it is at most the raw bound divided by
$a_0(1-\widehat b_N)$. This derives finite linear-response
budgets for actual jitter, OU, damping and terminal-noise
parameters, retaining every algorithm/landscape parameter
through the complete drift, covariance and Poisson response.
For explicitly reconstructed or calibrated observations add their
actual $E\dot H_{N,j}$ at each consumed lag. An infinite sum of
these extra terms is used only when its absolute convergence
has separately been derived from that readout's finite budgets.
It is a pulse response; the derivative of a permanently
parameter-changed QSD/Doob eigenpair is not inferred from it.
A bounded nonlinear test of an original standardized draw does
not qualify for the zero-explicit-derivative amplitude formula:
its standardized law is unchanged by an amplitude change,
and its reconstruction derivative can exactly cancel that
amplitude-density covariance. If its reconstruction is merely
measurable, this theorem makes no derivative assertion for it.
:::

:::{prf:proof}
Use physical coordinates for each scored Gaussian stage.
Its density derivative is the displayed score times its
unchanged density. For the OU row
$z=cv_1+q\xi$, differentiation of its mean and variance gives
$(c'v_1/q)\cdot\xi+(q'/q)(|\xi|^2-d)$.
At fixed derived $b_O$ the formulas give $c'=-hc$ and
$q'/q=hc^2/(1-c^2)-1/(2\gamma)$.
Jitter and position-amplitude scores follow from the same
Gaussian density derivative; actual unapplied jitter has no
physical density variation. The algorithm after each such
stage is its unchanged measurable map of physical stage
coordinates, so its hard gates, terminal marks and geometry
do not require pointwise derivatives.

Locally in each positive parameter the density derivative is dominated
by an integrable Gaussian times a polynomial. The first-kick
velocity has the primitive second moment $V_1^2$ and higher
Gaussian-jitter moments. Hence differentiation of a bounded
fixed physical-record observation and of its survival indicator
is justified. For $O_\vartheta$, its actual dominated coordinate
derivative adds $E\dot O$ before taking the quotient.
Dividing those differentiated expectations gives (NFS.19).
Orthogonality of distinct addressed
raw Gaussian scores, their zero conditional means and Gaussian
third-moment cancellation give (NFS.18).

For the long-survival limit use the exact terminal-tilt identity
as in the CLT proof, now with the fixed first-window integrable
observations $O\mathcal S$ and $\dot O$. After time $K$, block mixing of the
bounded terminal $1/e_N$ weight factorizes it. Its error is
bounded by a geometric mixing coefficient times $E|O\mathcal S|$.
The same argument handles each mean and the explicit derivative.
Thus its limit is the stationary complete Doob response.

For future $j\ge2$, condition on $S_1$: its centered prediction is
$(P_N^e)^{j-2}(g_N-\theta_N)$.
The score is measurable in the first complete record.
Summing the uniformly convergent series (NFS.2) gives
the $u_N(S_1)$ term in (NFS.20), while the first term is its
same-update covariance. The geometric bound yields (NFS.21).
Finally the stationary Doob one-update record law has density
$e_N(T)/[\alpha_N\nu_N(e_N)]$ relative to the original
$\nu_N$-started raw surviving record. It is at most
$1/[\alpha_N\min e_N]$. This proves the primitive score bound.
:::

:::{prf:corollary} The original addressed innovation-mean shift has a full source-record score
:label: cor-nfs-original-mean-source

Use only the existing Rust `QftExecutionConfig.innovation_shifts`
addresses of the declared Gaussian analytic execution: a finite
deterministic list $f$ shifts the original addressed standardized
innovations in one update by $\vartheta f$. The source recorded
by the existing provider is the shifted innovation itself.
Keep the original subsequent maps and every explicit observation
field fixed. For every bounded measurable function of that
complete stochastic source/stage record, its exact source score is

$$
\mathcal S_f=f\cdot G,\qquad
E_{\rm raw}\mathcal S_f^2=|f|^2.
\tag{NFS.31}
$$

Equations (NFS.19)--(NFS.21) hold with $\dot O=0$ for that fixed
full stochastic-record readout, including its original shifted
raw-innovation coordinates, hard gates and masks.
If a readout also consumes the explicitly changed configuration
metadata or reconstructs an unshifted seed draw, its actual
explicit derivative is again required by (NFS.19).
Unused but addressed innovations retain their original Gaussian
law and their full-record score when read; they have no physical
state effect. No new innovation address or noise is inserted.
The native stationary second-moment bound is
$|f|^2/[\alpha_N\min e_N]$.
:::

:::{prf:proof}
In the provider's recorded shifted innovation coordinates the
original density is $\varphi(G-\vartheta f)$.
Its logarithmic derivative at zero is $f\cdot G$.
All original downstream maps are unchanged functions of that
same coordinate. Gaussian differentiation with its integrable
linear score therefore applies to every fixed bounded measurable
complete stochastic-record readout. Its raw square moment is
the displayed original Gaussian value. Apply the finite-history
quotient, exact Doob telescope and future-prediction proof above.
These are the already implemented source-shift addresses,
restricted to their original continuous Gaussian execution.
:::

(sec-nfs-complete-nondegeneracy)=
## 7. A positive complete native color covariance on an actual source fiber

:::{prf:lemma} Complete conditional-record variance survives the Poisson correction
:label: lem-nfs-record-fiber-variance

For any bounded complete-update observation in Section 1,
with both endpoint states retained, its exact complete covariance obeys

$$
\Sigma_N\succeq
E_{\pi_N}^e\operatorname{Var}(H_N\mid S_0,S_1).
\tag{NFS.22}
$$

Thus a nonconstant original record observation on a positive-measure
physical endpoint fiber proves a positive component of the complete
time covariance. Its martingale and all preparation components need
not be independent.
:::

:::{prf:proof}
The Poisson endpoint correction in (NFS.4) is constant conditional
on $S_0,S_1$. Hence the conditional covariance of $D_{N,1}$ equals
that of $H_N$. The conditional second-moment decomposition of
$E D_{N,1}D_{N,1}^\top$ proves (NFS.22).
:::

:::{prf:definition} Primitive single-clone color-fiber tests
:label: def-nfs-color-fiber-tests

Retain the full register of Section 1, with $d=3$, $N\ge4$,
$\lambda,\nu,\sigma_J,q,s>0$, quadratic reward $-\lambda|x|^2/2$,
the original global population standardizers,
$\delta_D,\sigma_r,\sigma_s>0$, $p_r\ge0$, $p_s>0$,
and the configured strictly positive logistic-map floors and amplitudes.
The execution is the inherited real-coordinate, independent continuous
Gaussian, dense-count branch; a finite-precision fixed-seed execution
is not assigned these Jacobian and conditional-density conclusions.
Both original donor kernels must give positive mass to every eligible
current row on the bounded inputs below; the existing uniform and
finite-width Gaussian tags do so. There is no history donor,
elite freeze, geometry feedback or curl in this named canonical restriction.

Write $\ell(z)=(1+e^{-z})^{-1}$ and

$$
\begin{gathered}
u_b^-=\eta_b+A_b\ell(-1),\qquad
u_b^+=\eta_b+A_b\ell(1),\qquad b=r,s,\\
D_s^- =A_sp_s\frac{e}{(1+e)^2}
 \min\{(u_s^-)^{p_s-1},(u_s^+)^{p_s-1}\}(u_r^-)^{p_r},\\
D_r^+=\frac{A_rp_r}{4}
 \max\{(u_r^-)^{p_r-1},(u_r^+)^{p_r-1}\}(u_s^+)^{p_s},\\
G_{\rm gate}=
 \frac{D_s^-}{3\sqrt2\delta_D\sigma_s}
                  -\frac{D_r^+\lambda}{2\sigma_r}>0,\\
r_g=\min\{\delta_D,\sqrt{2\sigma_r/\lambda},
                    \sqrt{2\delta_D\sigma_s},L_D/2\}.
\end{gathered}
\tag{NFS.23}
$$

Set $D_r^+=0$ when $p_r=0$. Choose a declared $0<r_0<r_g$,
and retain $\alpha=1-t^2\lambda>0$ and
$\ell_K=e^{-1/2}/\rho$. Choose a declared $\varepsilon>0$ with

$$
\begin{gathered}
\kappa_B=\alpha-2t\nu-4t^2\nu\varepsilon\ell_K>0,\qquad
\vartheta=\frac{t\nu}{\alpha}<\frac12,\\
k_0=\exp[-(\alpha r_0+2t\varepsilon)^2/(2\rho^2)],\qquad
k_{\rm off}=k_0-
 \frac{\vartheta(3-2\vartheta)}
                  {(1-\vartheta)(1-2\vartheta)}>0,\\
0\le\delta_c<\tfrac12\nu\varepsilon k_0.
\end{gathered}
\tag{NFS.24}
$$

Here $\delta_c$ is the literal matched-color force threshold.
Consume any fixed finite phase $\kappa_c$ and the original normalized
matched B2 color
$c_i=(F_i/|F_i|)\odot e^{i\kappa_c z_i}$ on its available rows,
$P_i=c_ic_i^\dagger$, where $z$ is the original B2 input velocity.
Keep its literal matched clone-deletion mask: the accepted recipient
is unavailable to this matched readout. The test below uses the
original un-cloned row $k=N$; no unavailable color is filled in.
All other passive geometry and calibration coordinates retain their
declared original values. A different phase alignment or mask is not
identified with this declared observation by name alone.
:::

:::{prf:theorem} Strict complete time covariance of an original retained B2 color
:label: thm-nfs-complete-color-positive

Under (NFS.23)--(NFS.24), let $H_N$ be the eight real traceless
Hermitian coordinates of the actual retained matched B2 projector
$P_N$, and zero when that original readout is unavailable.
Its complete stationary covariance in (NFS.6) satisfies

$$
\operatorname{tr}\Sigma_N>0.
\tag{NFS.25}
$$

The same strict conclusion holds after multiplying this actual
tagged-row observation by $N^{-1}$ or $N^{-1/2}$, for each finite $N$.
It proves a nonzero complete native time fluctuation channel;
it does not assert an $N$-uniform lower bound, positivity of every
color direction, or nondegeneracy of the unweighted empirical color mean.
:::

:::{prf:proof}
**1. Produce the accepted clone with the original sampled fitness.**
Take an all-alive entering configuration with every velocity zero,
$x_2=r_0e_1$, and all other positions zero.
Every zero-position row chooses a different zero-position row
as its measurement donor; row two chooses a zero-position donor.
There are at least three zero-position rows, so this respects the
original no-self convention. This discrete donor pattern has positive
probability for the admitted donor tags.

Its reward values are $-\lambda r_0^2/2$ at row two and zero
elsewhere. Its sampled separations are $\delta_D+\Delta_s$ at
row two and $\delta_D$ elsewhere, where
$\Delta_s=\sqrt{\delta_D^2+r_0^2}-\delta_D$.
The actual two standardizer scales are exactly

$$
\sigma_{r,N}^2=\sigma_r^2+
       \frac{(N-1)\lambda^2r_0^4}{4N^2},\qquad
\sigma_{s,N}^2=\sigma_s^2+
       \frac{(N-1)\Delta_s^2}{N^2}.
$$

The difference of the two standardized reward arguments is
$-\lambda r_0^2/(2\sigma_{r,N})$, and that of the separation
arguments is $\Delta_s/\sigma_{s,N}$.
All arguments lie in $[-1,1]$ by the definition of $r_g$.
Also $\Delta_s\ge r_0^2/(3\delta_D)$,
$\sigma_{s,N}\le\sqrt2\sigma_s$, and
$\sigma_{r,N}\ge\sigma_r$.
The logistic-power product therefore gives, for any zero row $i$,

$$
F_2-F_i\ge
D_s^-\Delta_s/\sigma_{s,N}
      -D_r^+\lambda r_0^2/(2\sigma_{r,N})
\ge G_{\rm gate}r_0^2>0.
\tag{NFS.26}
$$

Choose the actual cloning donor $1\to2$, and accept this original
gate. Its conditional probability is at least
$\min\{1,G_{\rm gate}r_0^2/[s_c(F^*+\epsilon_c)]\}>0$.
Every other zero row chooses another zero row as cloning donor,
and row two chooses a zero row. Their literal gates are zero.
Thus there is exactly one accepted edge and no revival.
All component velocities are zero for every original Haar matrix.
At zero original jitters the post-copy positions $X$ are
$X_1=X_2=r_0e_1$ and all others zero.
Because the first input velocities vanish, the actual first viscous
kick contributes zero: its intermediate $U$ is zero,
$p=\alpha X$, and $v_1=-t\lambda X$.

**2. Keep every native color available at the same source point.**
Choose the original OU innovations, which have everywhere-positive
density, to give $z_i=\varepsilon e_1$ on the first
$\lceil N/2\rceil$ rows and $z_i=-\varepsilon e_1$ on the rest.
The B2 positions are $y=p+tz$, and their pairwise separations are
at most $\alpha r_0+2t\varepsilon$.
For this original same-record array define

$$
W_{ij}=\mathbf1_{i\ne j}K(y_i-y_j)/N,\qquad
D=\operatorname{diag}(\sum_jW_{ij}),\qquad L=D-W.
$$

This is its actual dense count Laplacian. It has zero row sums,
$0\preceq L\preceq I$, $D_{ii}\le1$, and
$W_{ij}\ge k_0/N$ for $i\ne j$.
The actual B2 force is $F=-\nu Lz$.
Every row sees at least $\lfloor N/2\rfloor$ opposite-sign rows,
so $|F_i|\ge\nu\varepsilon k_0/2>\delta_c$.
In particular row $N$ is un-cloned and available under the literal
matched deletion mask. The uncapped terminal velocity is

$$
w=\alpha z-t\lambda p-t\nu Lz.
\tag{NFS.27}
$$

The derivative in all OU velocity coordinates differs from
$\alpha I$ in maximum row norm by at most
$2t\nu+4t^2\nu\varepsilon\ell_K$.
The first term is the original velocity Laplacian derivative;
the second differentiates its same-record kernel through $y=p+tz$.
Thus (NFS.24) gives an invertible local derivative.
The configured cap $w\mapsto Vw/(V+|w|)$ is a $C^1$
diffeomorphism onto the open velocity ball with positive determinant.
Choose original final-position innovations to put every final
position at zero, strictly inside $D$. These are finite source
values with positive Gaussian density. Every final alive mark is one.

**3. A retained color changes while the complete endpoint is fixed.**
Vary the original applied jitter of recipient one in the transverse
$e_2$ direction. Its post-copy displacement is
$\dot X=\sigma_J e_1^{\rm row}\otimes e_2$.
The implicit-function theorem solves the entire original OU input
$z(X)$ locally to keep every $w_i$ in (NFS.27) constant.
Adjust each original terminal-position innovation to keep every
final position constant. Hence all physical endpoint coordinates,
capped velocities and alive marks remain exactly fixed.

At the collinear source point, the transverse derivative of every
Gaussian kernel is zero. Also the first viscosity remains zero
and $\dot p=\alpha\dot X$. With $A=\alpha I-t\nu L$,
the transverse derivatives are exactly

$$
\dot z=A^{-1}t\lambda\alpha\sigma_J e_1^{\rm row}\otimes e_2,
\qquad
\dot F=-\nu t\lambda\alpha\sigma_J
                  L(\alpha I-t\nu L)^{-1}
                         e_1^{\rm row}\otimes e_2.
\tag{NFS.28}
$$

This preserves the actual dependence of viscosity on its own
positions and both kinetic kicks.
For $k\ne1$, the original matrix in (NFS.28) obeys

$$
\big[L(\alpha I-t\nu L)^{-1}\big]_{k1}
\le-\frac{k_{\rm off}}{\alpha N}<0.
\tag{NFS.29}
$$

Indeed expand it as $\alpha^{-1}\sum_{m\ge0}\vartheta^mL^{m+1}$.
Its leading off-diagonal entry is $-W_{k1}/\alpha$.
In a product of $m+1$ factors $L=D-W$, the all-diagonal word
has zero off-diagonal entry; every other word has absolute
$(k,1)$ entry at most $1/N$, since $D,W$ are nonnegative
sub-stochastic in rows and columns and $\max W_{ij}\le1/N$.
There are $2^{m+1}-1$ such words. The remaining series costs at most

$$
\frac1{\alpha N}\sum_{m\ge1}\vartheta^m(2^{m+1}-1)
=\frac1{\alpha N}
 \frac{\vartheta(3-2\vartheta)}
                       {(1-\vartheta)(1-2\vartheta)}.
$$

This proves (NFS.29). In particular $\dot F_N$ has a nonzero
transverse component. At the original collinear point,
$F_N$ lies in the nonzero $e_1$ direction, and
$c_N=(e^{i\kappa_c z_{N,1}},0,0)$ up to its real sign.
Its $e_2$ derivative is $\dot F_{N,2}/|F_N|\ne0$.
The phase derivative multiplying a zero amplitude cannot remove it.
Consequently $\dot P_N$ has a nonzero off-diagonal Hermitian
entry and a nonzero traceless Hermitian coordinate, for every
finite consumed $\kappa_c$.

**4. The original stationary conditional fibers have positive mass.**
All gate, force-availability and inverse-derivative margins used
above are strict. The chosen single-edge event therefore persists
on an open entering-state neighborhood with an open gate-variable
interval, open Haar neighborhood and open Gaussian source
neighborhoods. Gates whose exact base value is zero remain refused
by choosing their original uniforms away from zero. The actual
finite-QSD minorization of {prf:ref}`thm-cgd-finite-n-qsd`
puts positive $\nu_N$, hence positive $\pi_N$, mass on the
chosen all-alive neighborhood: its analysis position radius
may be chosen larger than $r_0$ and smaller than $L_D$,
and its velocity target ball contains an open neighborhood of zero.

Conditional on such an entering state and the original discrete,
gate and Haar prefix, the map from original applied jitter,
all OU innovations and final-position innovations to
(applied jitter, full physical final state) is locally invertible.
Its output Jacobian factors the invertible $D_zw$, the positive
cap Jacobian and the original nonzero terminal Gaussian scale.
The original independent continuous source densities are strictly
positive in this local chart. Conditional jitter densities on
the resulting endpoint fibers are consequently positive there.
The nonzero projector derivative persists on a smaller open chart,
so the conditional variance of at least one actual projector
coordinate is strictly positive on a positive-measure endpoint set.
The actual Doob weight in (NFS.1) is positive and depends only
on the endpoints; it preserves this conditional source variation.
Equation (NFS.22) proves (NFS.25).
Multiplication by a positive finite $N$-dependent constant cannot
remove that strict conditional variance.
:::

:::{prf:corollary} Evaluated positive complete color-fiber witness
:label: cor-nfs-color-fiber-witness

Retain every original parameter of the positive count witness
(PC.37), including its exact positive cap, original uniform donor
tags, all fitness/standardizer/gate constants and unbounded noises.
Choose only the analysis values $r_0=5\,10^{-4}$,
$\varepsilon=1/2$, and consume the original matched B2
normalized projector with fixed finite $\kappa_c$ and original
threshold $\delta_c=10^{-12}$.
Then diagnostic evaluations of the primitive tests are

$$
\begin{gathered}
D_s^-\simeq0.2494890260,\quad D_r^+\simeq0.4327646447,
\quad G_{\rm gate}\simeq5.817237477\,10^{-5},\\
\frac{G_{\rm gate}r_0^2}{s_c(F^*+\epsilon_c)}
             \simeq3.635772514\,10^{-14}>0,\\
\vartheta\simeq0.01859140914,\quad
\kappa_B\simeq0.2559087681,\quad
k_0\simeq0.8824375616,\quad k_{\rm off}\simeq0.8241436127,\\
\tfrac12\nu\varepsilon k_0\simeq0.002206093904>\delta_c.
\end{gathered}
\tag{NFS.30}
$$

Thus for every finite $N\ge4$ in this actual positive phase,
at least one retained matched color has strictly positive
complete native time covariance. No force, gate, noise,
collision, cap, alive mark or unavailable row is changed.
:::

:::{prf:proof}
Here $r_g=10^{-3}$, $\lambda=4/(1+e^{-1})$,
$\alpha=(1+e)^{-1}$, $t=1/2$, $\nu=0.01$ and $\rho=1$.
Direct substitution in (NFS.23)--(NFS.24) gives the displayed
strict inequalities. The same exact expressions, rather than
rounded decimals, are their positive certificates.
Apply {prf:ref}`thm-nfs-complete-color-positive`.
:::

(sec-nfs-unbounded-source)=
## 8. Original unbounded source records have a complete time limit

:::{prf:theorem} Square-integrable complete-record Poisson and functional limit
:label: thm-nfs-l2-full-record

For fixed admitted $N$ retain the same actual complete transition
instrument, now with $H_N\in L^2(\pi_N\Omega_N^e)$ rather than
an imposed bounded source projection. Put
$h_N=H_N-EH_N$, $R_N=\|h_N\|_2$, and

$$
r_N^{(2)}=\sqrt{\Delta_N}<1,\qquad
A_N^{(2)}=\frac{T_N}{1-r_N^{(2)}},\qquad
J_N^{(2)}=1+2A_N^{(2)}.
\tag{NFS.36}
$$

Then the original Poisson series (NFS.2) converges in $L^2(\pi_N)$,
and its exact complete martingale decomposition, bracket and
Green--Kubo identity remain valid with

$$
\|u_N\|_2\le A_N^{(2)}R_N,\qquad
\|D_{N,1}\|_2\le J_N^{(2)}R_N,
$$
$$
\|C_{N,k}\|_{\rm op}\le R_N^2
 (r_N^{(2)})^{\lfloor(k-1)/T_N\rfloor},\qquad
\text{omitted tail after }K
\le2R_N^2 A_N^{(2)}
                  (r_N^{(2)})^{\lfloor K/T_N\rfloor}.
\tag{NFS.37}
$$

The functional CLT (NFS.33) holds for this actual unbounded
complete observation, with its exact $\theta_N,\Sigma_N$.
It also holds under the original $\nu_N$-started complete
history conditioned once on survival through $\lceil nT\rceil$.
The bounded-observation characteristic estimate (NFS.10) is not
assigned to an unbounded source without a further derived rate.
:::

:::{prf:proof}
The already proved native $L^2_0$ block inequality
{prf:ref}`thm-nsc-native-l2-block` gives
$\|(P_N^e)^k\|_{L^2_0}\le
(r_N^{(2)})^{\lfloor k/T_N\rfloor}$.
Its proof uses the actual KL block inequality and bounded-density
perturbations, followed by $L^2$ density; it is not an assumed
stationary entropy-production or gradient inequality.
Conditional Jensen gives
$\|g_N-\theta_N\|_2\le R_N$.
Summing its genuine complete-state block estimate proves the
$L^2$ convergence and bound for $u_N$.
The martingale and telescoping identities follow in $L^2$ by
their same conditional-expectation argument.
The two endpoint norms give the displayed bound for $D_N$.
Conditioning a later record on its entering state and applying
this $L^2$ prediction bound gives (NFS.37).
The endpoint remainder has $L^2$ norm at most $2\|u_N\|_2$,
so the normalized second-moment identity again identifies
this absolutely convergent covariance series with $EDD^\top$.

To prove the path limit, use bounded radial truncations
$H_N^{[M]}$ only as a proof approximation to $H_N$.
Their centered $L^2$ discrepancy tends to zero as $M\to\infty$.
The corresponding Poisson solutions and martingale increments
converge in $L^2$, by (NFS.37); hence their covariance matrices
converge to the exact $\Sigma_N$.
Each bounded approximation has the functional CLT already proved.
For the difference martingale, the original martingale maximum
inequality gives, with $a_n=\lceil nT\rceil$,

$$
E\max_{0\le j\le a_n}
 \left|\frac{M_{N,j}-M_{N,j}^{[M]}}{\sqrt n}\right|^2
\le4(T+1)(J_N^{(2)})^2
 \|[H_N-H_N^{[M]}]-E[H_N-H_N^{[M]}]\|_2^2.
$$

Its right side tends to zero uniformly in $n$.
For each fixed $M$, the difference endpoint solution is in $L^2$.
Stationarity and a union bound give for any $\epsilon>0$

$$
P\left(\max_{0\le j\le a_n}|u_N(S_j)|>
                                  \epsilon\sqrt n\right)
\le(a_n+1)P(|u_N|>\epsilon\sqrt n)\longrightarrow0,
$$

because $nP(|u_N|>\epsilon\sqrt n)$ is at most
$\epsilon^{-2}E[|u_N|^2\mathbf1_{|u_N|>\epsilon\sqrt n}]$.
The same argument handles each approximation and their difference.
Linear interpolation does not increase these maximum errors.
Thus bounded path approximations converge uniformly in probability
in the successive $n\to\infty$, $M\to\infty$ limits.
The limiting Gaussian covariances also converge, proving the
actual unbounded-observation functional CLT.

For the original once-conditioned path its density relative to
the stationary Doob path is exactly
$\nu_N(e_N)/e_N(S_{a_n})\le m_N^{-1}$.
Consequently every approximation-error probability above is
multiplied by at most this fixed positive primitive constant.
Each bounded approximation already has the exact original-history
functional CLT. Pass through those same successive approximation
limits to obtain the asserted unbounded source path law.
The source itself has never been clipped in the algorithm or
in the theorem's conclusion.
:::

:::{prf:corollary} Primitive original Gaussian-source moments and response scope
:label: cor-nfs-unbounded-gaussian-record

Let $G\in\mathbb R^{K_{G,N}}$, $K_{G,N}\ge1$, be a deterministic finite list of original addressed
Gaussian innovations consumed by the declared one-update source
readout, or such a list chosen independently of these innovations.
An innovation-dependent choice uses its fixed original candidate
address list and that full Gaussian norm, rather than declaring
the selected subset Gaussian. Unused addressed coordinates are
included when the declared readout or this candidate-list bound
consumes them. If its actual complete observation
satisfies the explicit growth test
$|H_N|\le B_H(1+|G|^r)$, $r\ge0$, then for every $p\ge1$

$$
E_{\pi_N}^e|H_N|^p
\le\frac{2^{p-1}B_H^p}{\alpha_N\min e_N}
\left[1+
2^{rp/2}\frac{\Gamma((K_{G,N}+rp)/2)}{\Gamma(K_{G,N}/2)}\right].
\tag{NFS.38}
$$

In particular all original linear and quadratic source scores,
their declared polynomial source tests and their complete
cross-covariances have finite primitive second moments.
Their actual unbounded functional time limits follow from
{prf:ref}`thm-nfs-l2-full-record`.
Every field determining $K_{G,N},B_H,r$ remains in the actual
readout; this test does not give an uncontrolled geometric
denominator an artificial moment bound.

The weak pulse formula (NFS.19) also extends to an unbounded
physical-coordinate observation when its original local
Gaussian-polynomial envelope and explicit reconstruction
derivative have the displayed finite integrable moments.
For a fixed first-$K$ record polynomial of those Gaussian
coordinates, its raw moments are explicit Gaussian integrals;
the stationary Doob moment costs at most
$[\alpha_N^K\min e_N]^{-1}$ times that raw moment.
The actual original once-survival-conditioned moment has the
same finite comparison up to its bounded terminal weight.
The covariance/response bound uses Cauchy--Schwarz with these
original unbounded moments, retaining the chain term in (NFS.19).
:::

:::{prf:proof}
The stationary one-update Doob density relative to the original
$\nu_N$-started raw record is at most
$1/(\alpha_N\min e_N)$, as proved in Section 6.
Under that original law $G$ is standard Gaussian, independently
of its entering state. Its exact radial moment is
$E|G|^{rp}=2^{rp/2}\Gamma((K_{G,N}+rp)/2)/\Gamma(K_{G,N}/2)$.
The power inequality proves (NFS.38).
For a first-$K$ record the exact telescope replaces the
one-update density bound by $1/(\alpha_N^K\min e_N)$.
Local Gaussian-polynomial domination justifies differentiation
and the same finite-history quotient proof. Its long-survival
limit uses the bounded terminal tilt against the integrable
observation, its product with the score and its actual explicit
derivative. This is the same dominated factorization used in
Section 6, with the primitive moments proving its integrability.
:::
