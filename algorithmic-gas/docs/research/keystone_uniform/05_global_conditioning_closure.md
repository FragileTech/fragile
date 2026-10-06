# Averaged-energy extinction estimates and direct QSD comparison

(sec-kugc-energy-hazard)=
## 1. A stronger actual-kernel estimate using averaged collision energy

:::{prf:definition} Quadratic landing and energy coefficients
:label: def-kugc-energy-coefficients

Use the complete reference kernel and notation in
{prf:ref}`def-ku-conditioning-record`, with the actual quadratic force
$F(x)=-\lambda x$, count normalization, $0\le t\nu\le1$ and
$|\alpha_{\rm col}|\le1$. Put

$$
A=|1-bt\lambda|,\quad R=bV_c<L_D,\quad
\tau_1^2=A^2\sigma_J^2+\sigma_h^2,
$$

$$
P_0=[\ell_{L_D,\sigma_h}(AL_D)]^d,\qquad
z_0=\Phi^{-1}(P_0),\qquad k=b/\sigma_h,
\qquad f(r)=\Phi(z_0-kr),
$$

$$
p_1=[\ell_{L_D-R,\tau_1}(AL_D)]^d,
\qquad D_V=f(V)-\frac V2 f'(V).
\tag{KUGC.1}
$$

Assume $P_0\le1/2$, so $z_0\le0$ and $f$ is decreasing and
convex on $[0,\infty)$. Every quantity is determined by the actual
primitive force, cap, collision, Gaussian amplitudes and timestep.
The finite feature, companion and fitness parameters remain in the
original pattern law and are not replaced or omitted.
:::

:::{prf:lemma} Frozen-component energy, including dead retained velocities
:label: lem-kugc-collision-energy

For every actual accepted connected component $C$ and its one Haar
matrix $O$, the exact collision satisfies

$$
\sum_{i\in C}|v_i^{\rm col}|^2
=|C||\bar v_C|^2+
\alpha_{\rm col}^2\sum_{i\in C}|v_i-\bar v_C|^2
\le\sum_{i\in C}|v_i|^2.
\tag{KUGC.2}
$$

Consequently, for every realized entire component pattern, every
jitter array and the actual first count-normalized alignment matrix,

$$
\frac1N\sum_i|U_i|^2
=\frac1N\sum_i|(W_Xv^{\rm col})_i|^2\le V^2.
\tag{KUGC.3}
$$

Retained dead velocities are included in both sums; revival does not
copy a donor velocity in this component convention.
:::

:::{prf:proof}
Write $v_i=\bar v_C+w_i$ and use $\sum_iw_i=0$ and
$|Ow_i|=|w_i|$. The cross term in the displayed square sum
vanishes exactly, proving (KUGC.2). Sum over the disjoint realized
components and use the input cap $|v_i|\le V$. Count-normalized
$W_X$ contracts the normalized Euclidean norm pathwise for
$t\nu\le1$, even when its weights depend on every jitter.
This proves (KUGC.3). $\square$
:::

:::{prf:theorem} Preparation-exact exponential alive-count transform
:label: thm-kugc-energy-alive-transform

If the explicitly computed inequality

$$
p_1\ge D_V
\tag{KUGC.4}
$$

holds, then every nonextinct entering state of this unchanged kernel
satisfies, for all $\theta\ge0$ and every $N\ge1$,

$$
\boxed{\quad
\mathbb E_S e^{-\theta M^+}
\le\exp[-N(1-e^{-\theta})f(V)],\qquad
\mathfrak h_N(S)\le e^{-Nf(V)},\qquad
\mathbb E_S M^+/N\ge f(V).
\quad}
\tag{KUGC.5}
$$

These bounds integrate every actual sampled-fitness, companion,
gate, mandatory-revival, component-Haar and Gaussian-jitter pattern.
There is no lower acceptance probability, assumed fitness gap,
independence of the alignment velocities, or all-row bounded-noise
event. The exponent uses the averaged $V$ instead of the rowwise
collision bound $V_c$.
:::

:::{prf:proof}
**1. Condition in the algorithm's order.** Freeze the actual
pre-jitter source array $Y_i\in\overline D$, gate pattern $I_i$ and
component rotations. Let $J_0=\{i:I_i=0\}$,
$J_1=\{i:I_i=1\}$, and $u=|J_0|/N$. Subsequently condition on
every jitter. Conditional landing indicators are independent, with
their actual probabilities $p_i$; hence their exponential transform
is exactly $\prod_i(1-Tp_i)$, where $T=1-e^{-\theta}\in[0,1)$.

**2. Keep the actual first-kick energy in the no-copy product.**
For a row in $J_0$, the unshifted position mean is $aY_i$,
$a=1-bt\lambda$. Its unshifted Gaussian landing probability is
at least $P_0$. The Gaussian shift comparison
{prf:ref}`lem-ku-gaussian-shift-landing`, with shift $bU_i$, gives
$p_i\ge f(|U_i|)$. Convexity and monotonicity of $f$, followed by
the actual energy identity (KUGC.3), imply for $u>0$

$$
\sum_{i\in J_0}p_i
\ge |J_0|f\left(|J_0|^{-1}\sum_{i\in J_0}|U_i|\right)
\ge uNf(V/\sqrt u).
$$

Thus the no-copy factors are bounded pointwise, including all
dependence on accepted-row jitters, by
$\prod_{i\in J_0}(1-Tp_i)\le
\exp[-TuNf(V/\sqrt u)]$.

**3. Integrate the accepted rows without discarding their jitters.**
For $i\in J_1$, let
$g_i(Z_i^J)=P_{D_R,\sigma_h}(a(Y_i+\sigma_JZ_i^J))$,
where $D_R=[-L_D+R,L_D-R]^d$.
The pointwise bounded shift $|bU_i|\le R$ gives
$p_i\ge g_i(Z_i^J)$. Therefore the complete conditional product
is at most

$$
\exp[-TuNf(V/\sqrt u)]
\prod_{i\in J_1}[1-Tg_i(Z_i^J)].
$$

Only after establishing this pointwise bound do we integrate the
independent own-row jitters. Their Gaussian convolution gives
$\mathbb Eg_i\ge p_1$, so the transform conditional on the
original frozen pattern is at most

$$
\exp[-TuNf(V/\sqrt u)](1-Tp_1)^{(1-u)N}.
\tag{KUGC.6}
$$

**4. Optimize over every possible accepted fraction.** For $T>0$,
the negative logarithm of (KUGC.6), divided by $N$, is

$$
\mathcal R_T(u)=Tu f(V/\sqrt u)
-(1-u)\log(1-Tp_1),
$$

with $u f(V/\sqrt u)=0$ at $u=0$. Its derivative for $u>0$ is

$$
\mathcal R_T'(u)=T\left[f(r)-\frac r2f'(r)\right]
+\log(1-Tp_1),\qquad r=V/\sqrt u\ge V.
$$

The bracket is decreasing in $r$: its derivative is
$f'(r)/2-rf''(r)/2\le0$. Since
$-\log(1-Tp_1)\ge Tp_1\ge TD_V$, (KUGC.4) gives
$\mathcal R_T'(u)\le0$. Its minimum on $[0,1]$ is therefore
$\mathcal R_T(1)=Tf(V)$. This proves the first inequality in
(KUGC.5) for each frozen pattern, and mixing over their entire
original law preserves it. Let $\theta\to\infty$ for the
extinction bound. The moment inequality follows from the derivative
at $\theta=0$: $M^+\le N$ permits differentiation of the exact
transform; equivalently subtract the transform from one, divide by
$\theta$ and pass to zero. $\square$
:::

(sec-kugc-reference)=
## 2. Explicit unchanged-reference coefficients and conditioning

:::{prf:corollary} Reference energy exponent and inverse alive mass
:label: cor-kugc-reference-energy-survival

For the unchanged count-normalized reference,

$$
\begin{aligned}
z_0&\simeq-1.0388108665073459,\qquad
k\simeq1.9241541991558555,\\
a_E:=f(V)&\simeq5.116105717745763\,10^{-7},\\
D_V&\simeq5.509877133155944\,10^{-6},\\
p_1&\simeq2.609549193618306\,10^{-4}>D_V.
\end{aligned}
\tag{KUGC.7}
$$

The exact values are (KUGC.1) with the unchanged coefficients
(KU.S10), not the displayed diagnostic decimals. In particular
(KUGC.5) provides a substantially stronger primitive extinction
exponent than the rowwise $a_*$ in (KU.S11).

Put $c_*=(1-\log2)/2$. At every current observation time $n\ge1$
under whole-horizon survivor conditioning, and under every QSD,

$$
\Pr(M/N<a_E/2)\le e^{-c_*a_EN},\qquad
\mathbb E(N/M)^r
\le(2/a_E)^r+\left(\frac{r}{ec_*a_E}\right)^r.
\tag{KUGC.8}
$$

Every compatible QSD eigenvalue has
$\alpha_N\ge1-e^{-a_EN}$.
These are population-independent quantitative estimates; their
worst-input floor remains conservative at $N=200$.
:::

:::{prf:proof}
The actual reference has $t\nu=.006\le1$,
$\alpha_{\rm col}=.5$, $R=4b<2$ and $P_0<1/2$.
Substitution in (KUGC.1) gives (KUGC.7). The strict inequality
required by (KUGC.4) can also be checked without trusting the
diagnostic decimals, as follows.

**Exact sufficient reference test.** Use
$0.0392\le b\le0.04$, $0.02\le\sigma_h<0.0204$,
$0.9992\le A\le1$, and $2.5<\sqrt{2\pi}<2.51$.
These follow from $1-x\le e^{-x}\le1$ and
$1-e^{-x}\le x$, with the stated primitive coefficients.
The interval upper endpoint is at most $.08$, so
$P_0\le(.5+.08/2.5)^3=.532^3<.151$.
The elementary inequality
$e^{-t^2/2}\le1-t^2/2+t^4/8$ for $0\le t\le1$
gives

$$
\Phi(-1)\ge\frac12-\frac{103}{120\sqrt{2\pi}}
>\frac12-\frac{103}{300}>.151.
$$

Hence $z_0<-1$ and
$x:=kV-z_0>1+2(.0392/.0204)>4.84$.
Also $kV/2\le2$. The Gaussian Mills bound therefore yields

$$
D_V=\Phi(-x)+(kV/2)\varphi(x)
\le(1/x+2)\varphi(x)<.9e^{-11.7}<10^{-5}.
$$

For the accepted-row floor,
$(L_D-R-AL_D)/\tau_1=-3.96b/\tau_1>-1.6$ and
$(-L_D+R-AL_D)/\tau_1<-30$, using
$\tau_1\ge.09992$ and $\tau_1<.103$.
Consequently its one-coordinate landing factor is at least
$\Phi(-1.6)-\Phi(-30)$.
Integration over $[1.6,2.2]$ and $[2.2,2.6]$ gives

$$
\Phi(-1.6)\ge .6\varphi(2.2)+.4\varphi(2.6)
>\frac{.6}{12(2.51)}+\frac{.4}{30(2.51)}>1/40.
$$

Mills' bound and $e^{450}>1+450+450^2/2$ give
$\Phi(-30)<10^{-6}$, so the coordinate factor exceeds $1/41$.
It follows that $p_1>1/41^3>10^{-5}>D_V$.
The three exponential comparisons just used have fully finite
rational checks. With

$$
L_m(x)=\sum_{j=0}^m\frac{x^j}{j!},\qquad
U_m(x)=L_m(x)+\frac{x^{m+1}}{(m+1)!}\frac1{1-x/(m+2)},
$$

the positive series and its geometric tail give
$L_m(x)\le e^x\le U_m(x)$ for $0<x<m+2$.
Direct rational arithmetic gives
$L_{32}(117/10)>100000$,
$U_{20}(121/50)<12$ and $U_{20}(169/50)<30$.
This verifies the strict reference test using the exact original
parameters.

Exponential Markov inequality with $\theta=\log2$ and the first
line of (KUGC.5) gives

$$
\Pr(M<a_EN/2)
\le e^{(\log2)a_EN/2}e^{-a_EN/2}=e^{-c_*a_EN}.
$$

This event includes extinction. Subtracting the actual extinction
probability before normalizing, exactly as in (KU.S15), proves
the same bound under the current survivor law with no cumulative
denominator. Splitting the inverse moment on this event and its
complement, then maximizing $N^re^{-c_*a_EN}$, proves (KUGC.8).
Integrating the extinction bound in (KUGC.5) against a QSD proves
its eigenvalue bound. $\square$
:::

:::{prf:corollary} A stronger row-normalized coefficient from the actual small kick
:label: rem-kugc-row-energy-scope

The same calculation is valid if a computed deterministic bound
$\|W_Xv^{\rm col}\|_{2,N}\le V_E$ is available for every jitter
and actual component pattern: replace $V$ by $V_E$ throughout.
The column bound in {prf:ref}`lem-kuk-gaussian-column` gives

$$
V_E=\min\left\{V_c,
V[(1-t\nu)+t\nu\sqrt{\overline C_d}]\right\}.
\tag{KUGC.11}
$$

At the unchanged row-normalized reference one may replace
$\overline C_3$ by its proved upper bound $17431$, giving

$$
V_E=2(.994+.006\sqrt{17431})<4,\qquad
a_{E,\mathrm{row}}=f(V_E)>0,
$$

with diagnostic evaluations
$V_E\simeq3.572318149867633$,
$a_{E,\mathrm{row}}\simeq1.2613357527748248\,10^{-15}$ and
$\log a_{E,\mathrm{row}}\simeq-34.30660511422914$.
All of (KUGC.5) and (KUGC.8) consequently hold for this actual
row-normalized kernel with $f(V)$ replaced by this
$a_{E,\mathrm{row}}$. This improves the earlier rowwise floor
without imposing a degree lower bound.

The separate weighted energy identity
$\sum_i d_i|U_i|^2\le\sum_i d_i|v_i^{\rm col}|^2$
is exact but uses degrees depending on every realized jitter.
It cannot be replaced by unweighted (KUGC.3). Its use in the
accepted-fraction optimization must retain those weights and their
actual Gaussian moment bounds.

*Proof.* The Gaussian column bound and row Jensen inequality give
$\|P_Xv^{\rm col}\|_{2,N}\le
\sqrt{\overline C_d}\|v^{\rm col}\|_{2,N}$.
Apply Minkowski's inequality to the actual decomposition
$W_X=(1-t\nu)I+t\nu P_X$, and then (KUGC.2), to obtain
$\|W_Xv^{\rm col}\|_{2,N}\le
V[(1-t\nu)+t\nu\sqrt{\overline C_d}]$.
Its rowwise convex bound also gives $\|W_Xv^{\rm col}\|_{2,N}\le V_c$;
take their minimum. These estimates are pathwise in every realized
jitter array, so the energy argument in (KUGC.6) applies unchanged.
For the reference direct squaring gives
$132.04^2=17434.5616>17431$, hence $\sqrt{17431}<132.04$;
thus $V_E<2(.994+.006(132.04))<4$.
Also $V_E\ge V$. Since $r\mapsto f(r)-rf'(r)/2$ is decreasing,
$D_{V_E}\le D_V<p_1$ by the exact reference test already proved.
Therefore the strict clone-fraction test passes for both
normalizations. $\square$
:::

(sec-kugc-renewal)=
## 3. Exact renewal identity and the information required to close it

:::{prf:lemma} Stopped eigenfunction identity for the original killed law
:label: lem-kugc-stopped-eigenfunction

Let $Q_Nh_N=\alpha_Nh_N$ with a bounded positive eigenfunction
associated with a QSD eigenvalue $\alpha_N\in(0,1)$.
For an actual-state set $C$ let
$\tau_C=\inf\{n\ge0:S_n\in C\}$ and let
$\tau_\dagger$ be extinction. For every integer $T\ge0$,

$$
\begin{aligned}
h_N(S)={}&\mathbb E_S[
\alpha_N^{-\tau_C}h_N(S_{\tau_C});
\tau_C\le T,\ \tau_C<\tau_\dagger]\\
&+\mathbb E_S[
\alpha_N^{-T}h_N(S_T);
T<\tau_C,\ T<\tau_\dagger].
\end{aligned}
\tag{KUGC.9}
$$

Every term is computed using the original trajectory, without
stepwise survivor rejection or restart. If the second term tends
to zero and
$0<h_{C,-}\le h_N|_C\le h_{C,+}$, then

$$
h_{C,-}\,\mathbb E_S[
\alpha_N^{-\tau_C};\tau_C<\tau_\dagger]
\le h_N(S)\le
h_{C,+}\,\mathbb E_S[
\alpha_N^{-\tau_C};\tau_C<\tau_\dagger].
\tag{KUGC.10}
$$
:::

:::{prf:proof}
Set the eigenfunction to zero at the cemetery state.
$\alpha_N^{-n}h_N(S_n)\mathbf1_{\{n<\tau_\dagger\}}$
is a martingale by the eigen-equation and the actual Markov
property. Stop it at $\min\{\tau_C,T\}$. This is a bounded
stopping time and its stopped variable is integrable, since
$h_N$ is bounded and $\alpha_N^{-T}<\infty$.
Partition by hitting $C$ and surviving before $T$, and by
surviving without that hit, to obtain (KUGC.9).
If the remainder vanishes, take the monotone limit of the
nonnegative first term and use the bounds on $h_N|_C$.
This proves (KUGC.10). $\square$
:::

:::{prf:remark} What a direct uniform comparison must still compute
:label: rem-kugc-direct-renewal-obligations

The stronger averaged-energy extinction estimate closes the alive
and normalization quantities without requiring full mixing. A
uniform full-array eigenfunction ratio through (KUGC.10) additionally
requires explicit control of the discounted hitting expectation
and of $h_{C,+}/h_{C,-}$ on the actual renewal set. An interior
mass or low average velocity bound alone does not compute this
second ratio: different spatial populations inside that set can
have different future survival hazards.

Copied-position ancestry does not remove every information source
in this algorithm. The source-choice probabilities depend on the
complete sampled diversity array and its population normalizer;
the previous velocities remain attached to rows instead of being
copied; collisions use shared matrices on accepted components;
both viscous kicks read the realized whole velocity population;
and the second kick reads uncapped innovations. Consequently a
backward donor-lineage coalescence estimate by itself is not a
refresh estimate for the full marked transition. It must also
control these retained measurement and velocity inputs.

Neither (KUGC.5) nor (KUGC.9) establishes the remaining uniform
eigenfunction comparison. No impossibility for the unchanged
quadratic reference is proved here. Equal or nearly equal fitness
is permitted throughout; the calculation includes the all-no-copy
pattern as the extremizing fraction in (KUGC.6).
:::
