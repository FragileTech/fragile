# The native centroid-innovation process on the algorithmic time grid

(sec-ngp-record)=
## 1. Complete execution and causal Gaussian law

:::{prf:definition} Native process parameters and readout
:label: def-ngp-complete-process

Retain the full execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record` and all parameters of
{prf:ref}`def-ngf-parameter-ledger`. The finite identities below
use the actual dense Gaussian viscosity, either count or row normalization,
isotropic O innovation, independent final position noise, terminal-only
marking, and matched B2 force-input phase. All upstream fitness, donor,
acceptance, collision, potential, schedule and initial-law fields remain
in the actual preparation kernel. They are not replaced by independent
prepared rows. The finite weighted extension retains exactly the
centroid-invariant masks/weights in
{prf:ref}`thm-ngf-terminal-geometry-centroid`.

Set $t=h/2$, $c=e^{-\gamma h}$,
$q=b_O\sqrt{(1-e^{-2\gamma h})/(2\gamma)}$ with the original
$q=b_O\sqrt h$ convention at $\gamma=0$, $s=\sigma_x\sqrt h$, and

$$
\tau^2=t^2q^2+s^2,\quad \chi=s^2/\tau^2,\quad
\kappa=m\ell_0/\hbar_{\rm eff},\quad a=\sqrt3\,\kappa q\sqrt\chi.
$$

The conditional Gaussian regime has $h,q,s>0$. Physical algorithmic time
is its existing calibrated grid $\tau_\ell=\ell h$.
At update $\ell$, the complete actual preparation through B1/A1
has rows $(x_{1,i},v_{1,i})$. Retain all terminal positions $Y$ and
relative O velocities $r_i=z_i-N^{-1}\sum_jz_j$.
The next state still uses the actual B2 potential/viscous kick and cap.

Let $b^M_{N,\ell}$ be a declared matched determinant channel with its
actual relative coefficient $B^M(P_{N,\ell},Y,r)$ and proved bound
$|b^M|\le1$. This includes a unit-color determinant with terminal-alive
mask and normalized nonnegative averages of such determinants. For
other finite weights replace this bound by their actual absolute total.
After extinction append the cemetery state and assign zero to all later
readouts, preserving the original killed history and stopping convention.
:::

:::{prf:lemma} Exact causal reordering of the original update
:label: lem-ngp-causal-reordering

Conditional on the actual preparation, the original update is exactly
represented by first drawing its independent terminal positions
$Y_i\sim N(x_{1,i}+tcv_{1,i},\tau^2I_3)$, then the mean-zero residual
Gaussian array $\widetilde\eta$ with covariance
$I_N-N^{-1}{\bf1}{\bf1}^{\mathsf T}$ in each coordinate, and finally
an independent $Z_3\sim N(0,I_3)$. Set

$$
\begin{aligned}
z_i^0&=cv_{1,i}+\frac{tq^2}{\tau^2}
                 (Y_i-x_{1,i}-tcv_{1,i}),\\
z_i&=z_i^0+q\sqrt\chi\,(\widetilde\eta_i+N^{-1/2}Z_3).
\end{aligned}
\tag{NGP.1}
$$

Applying the actual B2, cap, marking and recording maps gives the original
complete next-state law. This changes a conditional probability
factorization, not the algorithm.

Let $\mathcal H_{N,\ell}$ contain the previous complete record,
current preparation, $Y$ and $r$. There is a scalar
$Z_\ell\sim N(0,1)$ independent of this sigma algebra and a measurable
$A_{N,\ell}$ with $|A_{N,\ell}|\le1$ such that

$$
b^M_{N,\ell}=A_{N,\ell}e^{iaZ_\ell/\sqrt N},\qquad
\mathbb E[b^M_{N,\ell}\mid\mathcal H_{N,\ell}]
=A_{N,\ell}e^{-a^2/(2N)}.
\tag{NGP.2}
$$

The $Z_\ell$ are mutually independent. Later coefficients may depend
on earlier innovations. On an absorbed history take $A_{N,\ell}=0$
and integrate an unused independent Gaussian.
:::

:::{prf:proof}
The original conditional noises give
$z_i=cv_{1,i}+q\xi_i$ and $Y_i=x_{1,i}+tz_i+s\zeta_i$.
Their joint Gaussian density factors exactly as in
{prf:ref}`thm-ngf-terminal-geometry-centroid`. Its posterior velocity
is $z_i^0+q\sqrt\chi\,\eta_i$ with independent standard $\eta_i$.
Orthogonal projection onto the relative row subspace and its centroid
gives independent $\widetilde\eta$ and $N^{-1/2}Z_3$, proving (NGP.1).
The original inputs are recovered by
$\xi_i=(z_i-cv_{1,i})/q$ and
$\zeta_i=(Y_i-x_{1,i}-tz_i)/s$. Every deterministic subsequent map
therefore has precisely its original joint law.

Put $g_Y=N^{-1}\sum_i z_i^0$ and
$A_{N,\ell}=B^M(P_{N,\ell},Y,r)e^{i\kappa{\bf1}\cdot g_Y}$.
The relative array is fixed before $Z_3$. B2 pair differences, its
force threshold, and the declared weights/masks are unchanged by a
common residual velocity shift. Every determinant acquires the same
factor $e^{i\kappa q\sqrt{\chi/N}{\bf1}\cdot Z_3}$.
Set $Z_\ell={\bf1}\cdot Z_3/\sqrt3$. Its characteristic function
proves (NGP.2).

The previous complete record is retained in the next preparation.
The fresh standardized centroid has the same conditional Gaussian
law for every such history, so conditional probability factorization
proves its independence from $\mathcal H_{N,\ell}$ and, inductively,
independence of the scalar sequence. Current extinction is determined
by $Y$, already drawn; unused independent variables after extinction
do not change the killed law. Nothing here asserts independent
completed walkers or independence of future coefficients from past noise.
:::

(sec-ngp-martingale)=
## 2. Native drift, bracket and uniform approximation

:::{prf:theorem} Exact native conditional-innovation martingale
:label: thm-ngp-exact-martingale-bracket

Define

$$
D_{N,\ell}=\sqrt N\{b^M_{N,\ell}-
\mathbb E[b^M_{N,\ell}\mid\mathcal H_{N,\ell}]\},
\qquad M_{N,L}=\sum_{\ell=1}^L D_{N,\ell}.
\tag{NGP.3}
$$

Its conditional drift is zero, and its exact complex variance and
pseudo-variance are

$$
\begin{aligned}
\mathbb E[|D_{N,\ell}|^2\mid\mathcal H_{N,\ell}]
 &=N|A_{N,\ell}|^2(1-e^{-a^2/N}),\\
\mathbb E[D_{N,\ell}^2\mid\mathcal H_{N,\ell}]
 &=NA_{N,\ell}^2(e^{-2a^2/N}-e^{-a^2/N}).
\end{aligned}
\tag{NGP.4}
$$

Write $\mathsf R(w)=(\Re w,\Im w)^{\mathsf T}$ and $u=a^2/N$.
The real conditional covariance is

$$
\begin{aligned}
Q_{N,\ell}={}&
\frac N2(1-e^{-u})^2
\mathsf R(A_{N,\ell})\mathsf R(A_{N,\ell})^{\mathsf T}\\
&+\frac N2(1-e^{-2u})
\mathsf R(iA_{N,\ell})\mathsf R(iA_{N,\ell})^{\mathsf T}.
\end{aligned}
\tag{NGP.5}
$$

In the filtration revealing $\mathcal H_{N,\ell}$ and then the centroid
Gaussian, $M$ is a martingale with bracket increments $Q_{N,\ell}$.
In the completed-update filtration $\mathcal F_{N,\ell}$, its bracket
increment is $\mathbb E[Q_{N,\ell}\mid\mathcal F_{N,\ell-1}]$.
The covariance rate on the configured time grid is this increment
divided by $h$; at the centroid reveal it is $Q_{N,\ell}/h$.

For fixed $p\ge1$ and $L<\infty$,

$$
\begin{aligned}
\|D_{N,\ell}-iaA_{N,\ell}Z_\ell\|_p
&\le\frac{a^2}{2\sqrt N}\{(\mathbb E|Z|^{2p})^{1/p}+1\},\\
\max_{j\le L}\left\|M_{N,j}-
\sum_{\ell=1}^j iaA_{N,\ell}Z_\ell\right\|_p
&\le\frac{La^2}{2\sqrt N}\{(\mathbb E|Z|^{2p})^{1/p}+1\}.
\end{aligned}
\tag{NGP.6}
$$

Uniform moments of all fixed orders follow from
$|D_{N,\ell}|\le|a||Z_\ell|+a^2/(2\sqrt N)$.
The limiting bracket approximation has the pathwise operator-norm rate

$$
\left\|Q_{N,\ell}-a^2
\mathsf R(iA_{N,\ell})\mathsf R(iA_{N,\ell})^{\mathsf T}\right\|_{\rm op}
\le\frac{3a^4}{2N}.
\tag{NGP.10}
$$

No mixing, joint-law LSI or stationary-chaos premise enters these identities.
:::

:::{prf:proof}
Insert (NGP.2). The Gaussian characteristic function gives
$\mathbb E e^{iaZ/\sqrt N}=e^{-u/2}$ and
$\mathbb E e^{2iaZ/\sqrt N}=e^{-2u}$. Expansion gives (NGP.4).
The radial/tangential variances of the centered exponential are
$\operatorname{Var}\cos(\sqrt u Z)=(1-e^{-u})^2/2$ and
$\operatorname{Var}\sin(\sqrt u Z)=(1-e^{-2u})/2$; their covariance
vanishes by parity. Multiplication by $\sqrt N A$ proves (NGP.5).

The previous completed record is contained in $\mathcal H_{N,\ell}$.
The tower property gives zero completed-update conditional drift and
the stated coarser bracket. In the finer reveal filtration the increment
is zero while preparation and relative data are revealed, and equals
$D_{N,\ell}$ at the centroid reveal. This is a filtration of the
original probability law, not another dynamics.

For real $v$, $|e^{iv}-1-iv|\le v^2/2$ and
$1-e^{-a^2/(2N)}\le a^2/(2N)$. The pointwise error is therefore
at most $a^2(Z_\ell^2+1)/(2\sqrt N)$. Take norms and sum for (NGP.6).
The inequality $|e^{iv}-1|\le|v|$ gives the domination by a Gaussian
polynomial. All its moments are finite, irrespective of dependence
created by cloning or the two graph evaluations.
Finally $(1-e^{-u})^2\le u^2$ bounds the radial coefficient in
(NGP.5) by $a^4/(2N)$, while
$|1-e^{-2u}-2u|\le2u^2$ bounds the tangential coefficient error
by $a^4/N$. Each displayed outer product has operator norm at most
one. Addition proves (NGP.10).
:::

(sec-ngp-limit)=
## 3. Nontrivial process limits for the unchanged reference

:::{prf:theorem} Limit of the native conditional-innovation process
:label: thm-ngp-native-process-limit

Fix the unchanged reference family of
{prf:ref}`def-uda-complete-parameters` with $\kappa\ne0$, either
normalization and the analytic Gaussian execution convention.
Start each killed process in its proved full QSD $\nu_N$ and retain
its raw history with the cemetery extension. Use the tagged matched
B2 determinant with its terminal-alive mask from
{prf:ref}`thm-uda-uniform-selected-amplitude`.

From every unbounded population sequence one can choose a subsequence
and a causal coefficient process $(A_\ell)_{\ell\ge1}$, $|A_\ell|\le1$,
and scalar Gaussian innovations $(Z_\ell)_{\ell\ge1}$ such that jointly
at every finite collection of update times,

$$
(A_{N,\ell},Z_\ell,D_{N,\ell},M_{N,L})
\ \Rightarrow\ (A_\ell,Z_\ell,iaA_\ell Z_\ell,M_L),
\quad M_L=\sum_{\ell=1}^L iaA_\ell Z_\ell.
\tag{NGP.7}
$$

All fixed mixed moments converge. Each $Z_\ell$ is independent of
$\sigma(A_1,\ldots,A_\ell,Z_1,\ldots,Z_{\ell-1})$.
Future coefficients may depend on earlier innovations.
In the interleaved coefficient/innovation filtration the limiting
drift is zero and its real bracket is

$$
\langle\mathsf R(M)\rangle_L
=a^2\sum_{\ell=1}^L
\mathsf R(iA_\ell)\mathsf R(iA_\ell)^{\mathsf T}.
\tag{NGP.8}
$$

The derived reference amplitude constant gives

$$
\mathbb E|A_\ell|^2\ge C_{\rm amp},\qquad
\mathbb E|M_L|^2=a^2\sum_{\ell=1}^L\mathbb E|A_\ell|^2
\ge La^2C_{\rm amp}>0.
\tag{NGP.9}
$$

Thus a nontrivial native conditional-innovation process is obtained
on the actual algorithmic time grid with its drift and bracket.
Its coefficient process need not be deterministic, Markov or unique
along all population sequences.
:::

:::{prf:proof}
Each finite coefficient prefix lies in a compact product of unit disks.
The scalar Gaussian sequence has its unchanged product law.
Joint prefix laws are tight: a compact disk product and a large
Gaussian box contain arbitrarily high mass.
Choose convergent subsequences successively for prefixes of length
$1,2,\ldots$ and diagonalize. The consistent finite laws give the process.

For bounded continuous prefix and scalar tests $f,g$, exact causal
independence gives

$$
\mathbb E[f(A_{N,1},\ldots,A_{N,\ell},Z_1,\ldots,Z_{\ell-1})g(Z_\ell)]
=\mathbb E[f]\mathbb E[g(Z)].
$$

Pass the equality to the weak limit. Such tests determine probability
laws, proving the asserted limit independence. It does not assert
independence of the entire coefficient array from the entire noise array.
Continuous multiplication and (NGP.6) give joint limits of $D$ and $M$.
Gaussian domination gives uniform moments of every higher order, hence
uniform integrability of each fixed polynomial and convergence of all
mixed moments.

The actual reference survival estimates
{prf:ref}`thm-ku-quadratic-binomial-survival` and
{prf:ref}`thm-ku-uniform-alive-inverse-moments` give a computed
$a_*>0$ with $\alpha_N\ge1-(1-a_*)^N$, so $\alpha_N\to1$.
The QSD identity makes the incoming law at update $\ell$, conditional
on survival through $\ell-1$, exactly $\nu_N$, with probability
$\alpha_N^{\ell-1}$. The terminal-alive determinant vanishes on current
extinction, and therefore

$$
\mathbb E_{\nu_N,\rm raw}|b_N^{\rm alive}|^2
=\alpha_N\mathbb E_{\mathbb P_N^{\rm sel}}|b_N^{\rm alive}|^2
\ge\alpha_N C_{\rm amp}.
$$

Its coefficient has that same modulus and is zero after absorption,
giving $\mathbb E|A_{N,\ell}|^2\ge\alpha_N^\ell C_{\rm amp}$.
Squared modulus is bounded continuous on the disk, so its expectation
passes to the prefix limit, proving the first part of (NGP.9).

Limit increments are conditionally centered at their innovation reveals,
with covariance $a^2\mathsf R(iA_\ell)\mathsf R(iA_\ell)^{\mathsf T}$.
This proves (NGP.8). Distinct martingale increments are orthogonal:
condition their product before the later innovation. Summing their
second moments proves (NGP.9). Every nonzero constant comes from the
original noises, calibration and proved native amplitude.
:::

:::{prf:corollary} Finite-horizon native selection preserves these limits
:label: cor-ngp-selected-process-limit

For fixed $L$, the raw history started in $\nu_N$ and the same history
conditioned on survival through $L$ have the same process limits
(NGP.7) and converging fixed moments.
:::

:::{prf:proof}
Actual survival has probability $\alpha_N^L\to1$.
Raw and conditioned laws differ in total variation by $1-\alpha_N^L$,
so bounded continuous expectations have identical limits.
For a fixed polynomial, Cauchy--Schwarz and its uniformly bounded
twice-order moment show the omitted expectation is
$O(\sqrt{1-\alpha_N^L})$; the normalization $\alpha_N^{-L}$ tends to one.
This also proves moment convergence.
At finite $N$, future survival can bias the current centroid.
The argument retains that actual selection and does not assert its
finite-horizon conditional independence.
:::

(sec-ngp-scope)=
## 4. Parameter regimes and remaining full-channel identification

:::{prf:remark} Native process scope
:label: rem-ngp-full-channel-scope

The finite identities retain every upstream parameter through the actual
preparation and next-state maps. The nonzero process theorem uses the
derived reference amplitude, its proved QSD and primitive survival floor;
both normalizations and the actual positive threshold are included.
At $\kappa=0$ or $q=0$ this centroid innovation is exactly zero.
At $s=0$, terminal positions determine O velocities when $tq\ne0$,
so the terminal-geometry conditional innovation is zero. These concern
this component and do not say that all other channels have zero variance.

The process above is the original channel's centered centroid-noise part.
The full tagged $\sqrt N$ determinant is non-tight by
{prf:ref}`prop-uda-fixed-tag-centering-nontightness`.
An empirical/growing-graph average needs its own native amplitude and
other preparation/relative-noise increments. Default B1 readers,
clone-exclusion masks, capped phases, triangle/curvature coordinates,
and future velocity-dependent weights retain their actual readouts.

Full physical gauge identification still requires the law/limit of all
channel coefficients, their remaining drift/bracket, applicable
centering/spatial weights and spacetime/action correspondence.
A continuous-time endpoint additionally needs an evaluated $h\to0$
joint schedule; the theorem above keeps the existing fixed-$h$ grid.
:::
