# Polynomial excursions and normalized row derivatives

(sec-ku-polynomial-row)=
## Polynomial moments and row derivatives

:::{prf:lemma} Gaussian row defect with polynomial weights
:label: lem-ku-polynomial-row-defect

For the actual Gaussian row matrix $P_x$, put $C=\overline C_d$ from
{prf:ref}`lem-kuk-gaussian-column` and
$D_d=2C^{1/6}+4C^{1/3}+2C^{1/2}$. For any finite position arrays
$x,\widetilde x$, velocity array $v$ and $r=x-\widetilde x$,

$$
\|(P_x-P_{\widetilde x})v\|_{2,N}
\le\frac{D_d}{2\rho^2}(\|x\|_{6,N}+\|\widetilde x\|_{6,N})
\|r\|_{6,N}\|v\|_{6,N}.                         \tag{KUP.1}
$$

The singleton defect is zero. There is no minimum degree or exponential
position weight in this estimate.
:::

:::{prf:proof}
Interpolate $x_s=\widetilde x+sr$. Exact normalized-weight differentiation gives

$$
\frac d{ds}(P_{x_s}v)_i=\sum_{j\ne i}p_j[v_j-(P_{x_s}v)_i]
\frac{-(x_{s,i}-x_{s,j})\cdot(r_i-r_j)}{\rho^2}.
$$

Put $a_i=|x_{s,i}|$, $b_i=|r_i|$, $h_i=|v_i|$. The derivative norm is
bounded by $\rho^{-2}$ times the sum of eight nonnegative row arrays

$$
abPh+aP(bh)+bP(ah)+P(abh)+(Ph)ab
+(Ph)a(Pb)+(Ph)b(Pa)+(Ph)P(ab).
$$

For $q\ge1$, Jensen and the Gaussian column estimate give
$\|Pf\|_{q,N}\le C^{1/q}\|f\|_{q,N}$.
Hölder with $(6,6,6)$, $(6,3)$, or $2$ bounds the eight terms by
$\|a\|_6\|b\|_6\|h\|_6$ times, respectively,
$C^{1/6},C^{1/3},C^{1/3},C^{1/2},C^{1/6},C^{1/3},C^{1/3},C^{1/2}$.
Their sum is $D_d$. Integrate in $s$, using
$\|x_s\|_6\le(1-s)\|\widetilde x\|_6+s\|x\|_6$.
This proves (KUP.1) without inserting a degree lower floor. $\square$
:::

:::{prf:theorem} All uncapped moments of the unchanged Styblinski--Tang force
:label: thm-ku-cubic-uniform-moments

Keep the complete canonical preparation, $0\le t\nu\le1$, and both
actual dense normalization choices. The configured Styblinski--Tang
acceleration, as implemented in `benchmarks/src/lib.rs` and
`physics/fitness.rs::Objective::StyblinskiTang`, is

$$
F(x)=-2x^{\odot3}+16x-\tfrac52\mathbf1_d,
\quad U(x)=\tfrac12\sum_k(x_k^4-16x_k^2+5x_k).       \tag{KUP.2}
$$

No finite global derivative is assigned to this force. Put
$g_{d,p}=[2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2)]^{1/p}$ and compute

$$
\begin{aligned}
X_p&=R_D+\sigma_Jg_{d,p},\\
U_p&=V_c+t[2X_{3p}^3+16X_p+\tfrac52\sqrt d],\\
W_p&=cU_p+qg_{d,p},\qquad Y_p=X_p+bU_p+tqg_{d,p},\\
Z_p&=H_pW_p+t[2Y_{3p}^3+16Y_p+\tfrac52\sqrt d],\qquad
X_p^+=Y_p+sg_{d,p}.                                \tag{KUP.3}
\end{aligned}
$$

For every $p\ge1$, these bound the actual stages' normalized joint
$L^p$ norms. Here $H_p=1$ in count mode and
$H_p=[1+t\nu(C-1)]^{1/p}$ in row mode. The original output cap
still gives $|v_i^+|\le V$. B2 requires input Gaussian moments through
$9p$; the gamma formula evaluates every one. These bounds retain
positive viscosity and all unbounded innovations, uniformly in $N$.
:::

:::{prf:proof}
Since $|x^{\odot3}|\le|x|^3$,
$|F(x)|\le2|x|^3+16|x|+(5/2)\sqrt d$.
Minkowski on the product of normalized row measure and innovation law gives
$\|F(x)\|_{L^p}\le2\|x\|_{L^{3p}}^3+16\|x\|_{L^p}+(5/2)\sqrt d$.
Source coordinates are in $D$; accepted jitters are Gaussian, giving $X_p$.
The convex first matrix preserves the collision speed bound, giving $U_p$.
The actual OU and two drifts give $W_p,Y_p$. The pathwise Gaussian column
bound and the same cubic-force estimate at the actual second position give
$Z_p$. Final diffusion and the cap give the output budgets. $Y_{3p}$
contains $X_{9p}$, so no Gaussian tail has been removed. $\square$
:::

:::{prf:lemma} Complete paired row kicks for the actual cubic force
:label: lem-ku-cubic-paired-row

For prepared arrays put $r=x-\widetilde x$, $\zeta=v-\widetilde v$.
Their first-kick velocities satisfy

$$
\begin{aligned}
\|\Delta u\|_{2,N}\le{}&H_2\|\zeta\|_{2,N}+16t\|r\|_{2,N}\\
&+3t(\|x\|_{6,N}^2+\|\widetilde x\|_{6,N}^2)\|r\|_{6,N}\\
&+\frac{t\nu D_dV_c}{2\rho^2}
(\|x\|_{6,N}+\|\widetilde x\|_{6,N})\|r\|_{6,N}.
                                                        \tag{KUP.4}
\end{aligned}
$$

Under shared OU draws, $R=y-\widetilde y$ is deterministic conditional
on preparation. Compute its conditional Gaussian budgets

$$
\mathcal Y_6=\|x+bu\|_{6,N}+tqg_{d,6},\quad
\widetilde{\mathcal Y}_6=\|\widetilde x+b\widetilde u\|_{6,N}+tqg_{d,6},
\quad \widetilde{\mathcal W}_6=c\|\widetilde u\|_{6,N}+qg_{d,6}.
$$

The actual uncapped B2 difference $Z$ obeys

$$
\begin{aligned}
(\mathbb E[\|Z\|_{2,N}^2\mid\mathrm{prep}])^{1/2}
\le{}&cH_2\|\Delta u\|_{2,N}+16t\|R\|_{2,N}\\
&+3t\|R\|_{6,N}(\mathcal Y_6^2+\widetilde{\mathcal Y}_6^2)\\
&+\frac{t\nu D_d}{2\rho^2}\|R\|_{6,N}
(\mathcal Y_6+\widetilde{\mathcal Y}_6)\widetilde{\mathcal W}_6.
                                                        \tag{KUP.5}
\end{aligned}
$$

The original cap is nonexpansive. Outer preparation averages are bounded
explicitly by (KUP.3) and Hölder, with no exponential inverse degrees.
:::

:::{prf:proof}
The identity $a^3-b^3=(a-b)(a^2+ab+b^2)$ gives
$|\Delta F_i|\le16|r_i|+3(|x_i|^2+|\widetilde x_i|^2)|r_i|$.
Use normalized Hölder with three sixth moments, subtract B1 as
$A_x\zeta+t\Delta F+t\nu(P_x-P_{\widetilde x})\widetilde v$,
and apply (KUP.1) with $\|\widetilde v\|_6\le V_c$.
This proves (KUP.4). Shared OU gives the deterministic differences
$c\Delta u$ and $R=r+b\zeta+\eta f$. Subtract B2 in the same way,
using (KUP.1) at its actual $(y,\widetilde y,\widetilde w)$.
Conditional Minkowski gives the displayed sixth-moment budgets.
Jensen gives $\mathbb E\|y\|_6^4\le\mathcal Y_6^4$;
Cauchy--Schwarz bounds the graph product by
$(\mathcal Y_6+\widetilde{\mathcal Y}_6)\widetilde{\mathcal W}_6$.
These estimates prove (KUP.5). The already computed cap Jacobian gives
nonexpansiveness. Outer Hölder and (KUP.3) evaluate its finite polynomial
preparation moments. $\square$
:::


:::{prf:corollary} Evaluated outer cubic budgets for both normalized kicks
:label: cor-ku-cubic-outer-budget

With the actual paired preparation law, every outer factor in (KUP.5)
can be bounded by the already computed $X_p,U_p,W_p,Y_p$ of (KUP.3).
In particular, the row uncapped B2 discrepancy has the explicit bound

$$
(\mathbb E\|Z\|_{2,N}^2)^{1/2}
\le2cH_2U_2+32tY_2+12tY_6^3+
\frac{2t\nu D_d}{\rho^2}Y_6^2W_6.                \tag{KUP.6}
$$

For count normalization, the corresponding completely evaluated bound is

$$
(\mathbb E\|Z\|_{2,N}^2)^{1/2}
\le2cU_2+32tY_2+12tY_6^3+8t\nu\ell_\rho Y_4W_4,
\quad\ell_\rho=e^{-1/2}/\rho.                    \tag{KUP.7}
$$

Either may be replaced by the smaller marginal triangle bound $2Z_2$.
The original cap additionally bounds the output discrepancy by $2V$.
The conditional inequalities (KUP.4)--(KUP.5) retain actual small
pair discrepancies; these outer envelopes establish finiteness and
quantitative tail budgets and alone do not establish contraction.
:::

:::{prf:proof}
Normalized joint Minkowski gives
$\|R\|_{L^p(\mathrm{prep};\ell^p_N)}\le2Y_p$ and
$\|\Delta u\|_{L^p}\le2U_p$.
The outer sixth norms of each conditional budget $\mathcal Y_6$ and
$\widetilde{\mathcal W}_6$ are at most $Y_6,W_6$ by conditional
Minkowski and the triangle inequality. Hölder with three sixth norms
bounds the outer $L^2$ cubic-force term by
$3t(2Y_6)(Y_6^2+Y_6^2)=12tY_6^3$.
It bounds the graph term by
$[t\nu D_d/(2\rho^2)](2Y_6)(2Y_6)W_6$.
These give (KUP.6).

In count mode, subtract B2 as in
{prf:ref}`lem-kuk-paired-count-force`, replacing its external
$L_F\|R\|_2$ by the actual cubic-force sixth-moment term proved above.
Its graph term is $4t\nu\ell_\rho\|R\|_4\mathcal W_4$.
Outer Hölder and $\|R\|_{L^4}\le2Y_4$ give $8t\nu\ell_\rho Y_4W_4$,
proving (KUP.7). Marginal triangle and the actual cap prove the
remaining bounds. $\square$
:::


:::{prf:lemma} A stretched-exponential position tail for the actual cubic update
:label: lem-ku-cubic-stretched-tail

In the regime of {prf:ref}`thm-ku-cubic-uniform-moments`, set $r=2/3$,
$L_0=1+16\eta$, $B_0=bV_c+(5/2)\eta\sqrt d$ and
$C_x=L_0^r+(2\eta)^r$. For any $\delta>0$ satisfying

$$
4\delta C_x\sigma_J^2<1,\qquad
2\delta(tq)^r<1,\qquad2\delta s^r<1,
$$

put

$$
\begin{aligned}
C_0&=L_0^r+B_0^r+(tq)^r+s^r+2C_xR_D^2,\\
\mathcal B_\delta&=e^{\delta C_0}
(1-4\delta C_x\sigma_J^2)^{-d/2}
(1-2\delta(tq)^r)^{-d/2}(1-2\delta s^r)^{-d/2}.
\end{aligned}                                                   \tag{KUP.8}
$$

Every retained output row, for either positive-viscosity normalization,
satisfies $\mathbb E e^{\delta|x_i^+|^{2/3}}\le\mathcal B_\delta$ and
$\Pr(|x_i^+|>R)\le\mathcal B_\delta e^{-\delta R^{2/3}}$.
The constants are independent of $N$. A fully explicit admissible choice is
$\delta=[8C_x\sigma_J^2+4(tq)^r+4s^r]^{-1}$ when that denominator
is positive, and $\delta=1$ for the completely deterministic noise case.

The primitive binomial survivor bound of
{prf:ref}`lem-ku-coupled-binomial-survival` also applies to this unchanged
force when $\sigma_h^2>0$, by the computed local substitution
$\mathcal F_J=2(R_D+J)^3+16(R_D+J)+(5/2)\sqrt d$.
Thus current-time survivor rows and any existing QSD have the same
stretched-exponential bound multiplied by $1/[1-(1-a_J)^N]\le1/a_J$.
This conclusion does not require existence of a coupled cubic QSD as a premise
for the current-time survivor bound; its QSD statement applies when that law
has been established.
:::

:::{prf:proof}
The exact first-kick position formula and its convex velocity average give

$$
|x_i^+|\le L_0|X_i|+2\eta|X_i|^3+B_0+tq|\xi_i|+s|\chi_i|.
$$

For $0<r<1$, $(a+b)^r\le a^r+b^r$ for nonnegative $a,b$;
differentiate $(a+b)^r-a^r$ in $a$ to prove it.
Also $z^r\le1+z^2$ for $z\ge0$.
With $X_i=Y_i+I_i\sigma_JZ_i^J$ and $|Y_i|\le R_D$,
$|X_i|^2\le2R_D^2+2\sigma_J^2|Z_i^J|^2$. These pointwise
inequalities yield

$$
|x_i^+|^r\le C_0+2C_x\sigma_J^2|Z_i^J|^2
+(tq)^r|\xi_i|^2+s^r|\chi_i|^2.
$$

Conditional on the actual frozen preparation pattern, these three latent
row Gaussians are independent. Their squared-norm Gaussian integrals give
(KUP.8). The bounded viscous average may depend on all jitters; only its
pointwise bound was used. Exponential Markov gives the tail. The stated
$\delta$ makes every Gaussian integral coefficient at most one half.
Removing extinct outputs and using the proved next-step survival floor
transfers the same nonnegative envelope to survivor laws and QSDs.
$\square$
:::
