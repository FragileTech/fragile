# All-population discounted discovery and the precise regional eigenfunction reduction

(sec-kuac-finite-budget)=
## 1. A fixed quantitative budget for every remaining finite population

:::{prf:definition} Native finite-population analysis budget
:label: def-kuac-finite-budget

Keep the native real-coordinate Rastrigin record of
{prf:ref}`def-kurc-record`, with its original complete marked kernel.
Set $N_c=9000500000$ in count mode and
$N_r=257157142857143$ in row mode. Write $N_*$ for the appropriate
one of these integers. Use the proof-event radii

$$
J_0=.1,\quad J=29,\quad G=290,\quad
L_F=2+40\pi^2,\quad B_F=0,\quad R_D=2\sqrt3,
\quad V_c=4.
\tag{KUAC.1}
$$

In the primitive formulas of
{prf:ref}`def-cgd-analytic-force-profile` and
{prf:ref}`lem-cgd-two-update-density`, use the overestimate
$\nu=.3$ also for the singleton. It bounds that singleton's
configured zero viscous term; it does not change its kernel.
Put $p_b=(4\pi/3)(2\pi)^{-3/2}e^{-1/2}$ and evaluate
$B_0,B_1,B_x,Z_0,R'_2,H,R,\kappa_F,\beta_F,k_B$ with (KUAC.1).
Use $N=N_*$ in $Z(H),D_F(H)$, with the row $N>2$ formula.
Define the explicit positive number

$$
\begin{aligned}
A_*={}&-N_*\log p_b
+\frac{N_*[Z(H)+cB_1(J_0)]^2}{2q^2}\\
&+\frac{N_*[\sqrt3R+B_x(J_0)+tZ(H)]^2}{2s^2}\\
&+3N_*\left[\tfrac12\log N_*+\log D_F(H)
                          -\log\{(1-\beta_F)k_B\}\right]\\
&+N_*\log(4N_*^2)+13600+\log2 .
\end{aligned}
\tag{KUAC.2}
$$

All entries are primitive scalar formulas. In particular none is
an eigenvalue, eigenfunction minimum or estimated mixing time.
:::

:::{prf:lemma} Verified small-population primitive ratio budget
:label: lem-kuac-finite-eigenfunction-budget

For every actual $1\le N\le N_*$, the full-state right eigenfunction
$h_N$, normalized by $\max h_N=1$, satisfies

$$
e^{-A_*}\le h_N(S)\le1\qquad\hbox{for all survivor inputs}.
\tag{KUAC.3}
$$

The explicit conservative evaluations are

$$
A_{*,\rm count}<10^{28},\qquad A_{*,\rm row}<10^{23}.
\tag{KUAC.4}
$$

:::

:::{prf:proof}
The native force is real analytic with $B_F=0$ and the displayed
finite global derivative. All retained positivity and continuity
regularizers and $q,s>0$ are the original ones. Thus the qualitative
finite-$N$ QSD theorem applies. The worst coercivity margins exceed
$.82$ in both norms.

For the one-row primitive survival probability in (CGD.11), let
$A=L_D+J_0+t(1+c)B_1(J_0)$,
$z=(A-L_D)/\sqrt{t^2q^2+s^2}$ and
$u=(A+L_D)/\sqrt{t^2q^2+s^2}$. The Gaussian unit-ball probability
is at least $p_b$, by integrating its minimum density on that ball.
The two Mills inequalities therefore give the explicit lower bound

$$
a_F\ge p_b\left[
\phi(z)\frac{z}{1+z^2}-\frac{\phi(u)}u\right]^3>e^{-6800}.
\tag{KUAC.5}
$$

The bracket is positive; its logarithm together with $\log p_b$
is enclosed by $(-6774.346,-6774.345)$ after multiplication by
three as displayed. This uses the original unbounded tagged-noise
integral, not a modified force or a Gaussian truncation.
For every $N\le N_*$, the three original union-bound tails satisfy

$$
p_{\rm bad}\le18N_*e^{-290^2/6}
<\tfrac12e^{-13600}\le\tfrac12a_F^2.
\tag{KUAC.6}
$$

The first-drift tests, computed with the actual force, are
$\beta_{F,\rm count}<.160$ and $\beta_{F,\rm row}<.409$.
They hold on the stated proof event at all these populations.
Thus every hypothesis of
{prf:ref}`thm-cgd-primitive-eigenfunction` is verified.

For completeness one can bound all its ratios with a single scalar
evaluation, rather than a list of $N_*$ unknown spectral data.
Take its logarithmic ratio estimate (CGD.15). The Gaussian
prefactors cancel between $M_N$ and $\mathfrak l_N(R,H)$ because
its $\tau=\min(s,\sigma_J)=s$. Use $p_0\ge p_b$ and
$-2\log a_F<13600$. The resulting expression is (KUAC.2)
with $N$ in place of $N_*$. Every remaining summand is positive
and nondecreasing in $N$ when the overestimated $\nu=.3$ is held
fixed. In count mode $Z(H)$ also increases as $\sqrt N$;
in row mode the $N>2$ derivative formula overestimates the
pair and singleton cases. Its drift margins and event radii
are the same conservative ones throughout. Consequently the
expression is at most $A_*$, proving (KUAC.3).

`verify_discovery_finite_budget.py` evaluates the explicit Mills,
tail, drift and ratio expressions using 85-digit outward intervals.
It gives $A_{*,\rm count}<7.542\cdot10^{26}$ and
$A_{*,\rm row}<6.194\cdot10^{21}$, which imply the rounded
bounds (KUAC.4). No finite-population enumeration or uncomputed
Perron quantity is used. $\square$
:::

(sec-kuac-all-clock)=
## 2. All-population discovery and the remaining regional oscillation

:::{prf:theorem} Population-uniform actual discounted central discovery
:label: thm-kuac-all-population-discovery-clock

Let $G_N=\{S:K_C(S)\ge1\}$ with the actual central cube and
count of {prf:ref}`def-kupc-record`. For every survivor input
outside $G_N$ and every $N\ge1$,

$$
\mathbb E_S[\alpha_N^{-\tau_C};\tau_C<\tau_\dagger]
\le
\begin{cases}
e^{10^{28}},&\mathrm{count},\\
e^{10^{23}},&\mathrm{row}.
\end{cases}
\tag{KUAC.7}
$$

For $N\ge N_*$ the sharper bound is two, and actual discovery
before extinction has probability at least $2/3$.
These constants bound a physical discounted hitting moment.
They are not claimed to bound the global eigenfunction ratio at
populations above $N_*$.
:::

:::{prf:proof}
For $N\ge N_*$ use
{prf:ref}`thm-kupc-discounted-discovery` directly. For $N<N_*$,
(KUAC.3) gives $h_N\ge m=e^{-A_*}>0$.
The actual finite-$N$ Doob kernel is
$P_N^h(S,dT)=Q_N(S,dT)h_N(T)/(\alpha_Nh_N(S))$.
The global original discovery bound gives
$Q_N(S,G_N)\ge\lambda_N>0$ on every entering survivor state.
Hence $P_N^h(S,G_N)\ge m\lambda_N$, since
$\alpha_Nh_N(S)\le1$.

The Doob chain therefore hits $G_N$ almost surely with geometric
tail. Stop its equivalent original killed martingale. Its remainder
is exactly

$$
\mathbb E_S[\alpha_N^{-T}h_N(S_T);
 T<\tau_C,\ T<\tau_\dagger]
=h_N(S)\Pr_S^h(\tau_C>T)
\le(1-m\lambda_N)^T\longrightarrow0.
$$

Thus the stopped identity gives
$h_N(S)=\mathbb E_S[\alpha_N^{-\tau_C}h_N(S_{\tau_C});
\tau_C<\tau_\dagger]$. The lower bound $h_N\ge m$ on the
hit states implies that the discounted moment is at most
$h_N(S)/m\le e^{A_*}$. Apply (KUAC.4).
The finite-$N$ ratio is used only on this explicitly bounded
finite population range; no uniform large-$N$ ratio is assumed.
$\square$
:::

:::{prf:theorem} Exact reduction of the large-population ratio to the reached phase set
:label: thm-kuac-regional-eigenfunction-reduction

For $N\ge N_*$ set
$R_G(N)=\sup_{G_N}h_N/\inf_{G_N}h_N$. Then

$$
\frac{\sup h_N}{\inf h_N}\le3R_G(N).
\tag{KUAC.8}
$$

Consequently the global population-uniform ratio is reduced to
the full-state oscillation on the reached actual set $G_N$:

$$
\sup_N\frac{\sup h_N}{\inf h_N}
\le\max\left\{e^{L_*},\,3\sup_{N\ge N_*}R_G(N)\right\},
\qquad L_*=10^{28}\ \hbox{or}\ 10^{23}.
\tag{KUAC.9}
$$

:::

:::{prf:proof}
Outside $G_N$, the raw remainder in the stopped identity is
bounded by $[(1-\lambda_N)/\alpha_N]^T$, which tends to zero
because $\lambda_N>\delta_N\ge1-\alpha_N$ at this threshold.
The hit probability is at least $2/3$, and its discounted moment
is at most two. Thus
$h_N(S)\ge(2/3)\inf_{G_N}h_N$ and
$h_N(S)\le2\sup_{G_N}h_N$. The same weaker inequalities
hold inside $G_N$. Divide to obtain (KUAC.8), then combine the
already proved finite-population estimate (KUAC.3)--(KUAC.4).
The set $G_N$ retains all other rows, dead coordinates, velocities,
phase occupancies and source information. Its oscillation is not
identified with a one-row positional projection. $\square$
:::
