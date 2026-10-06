# Strict-gap revival and population normalization

:::{prf:theorem} Two strictly separated alive fitnesses exclude a global population-uniform fixed-slot coefficient
:label: thm-ku-strict-gap-revival-obstruction

Keep the complete native quadratic reference in dimension three, with either
viscosity normalization and every original preparation and kinetic parameter.
Write $a_h=1-bt$, $q_x=\alpha-\beta^2/\gamma_P>0$ for the positive definite
physical quadratic in (KU.O11), and allow any nonnegative status weight.
For $N\ge2$ and fixed $0<\delta\le1/10$, define two terminal-consistent inputs
$S_N,T_N$ with all entering velocities zero. Exactly rows $1,2$ are alive.
Their positions are

$$
x_1=-e_1,\quad x_2=(9/10)e_1\qquad(S_N),\qquad
\widetilde x_j=x_j+\delta e_2\quad(j=1,2)\qquad(T_N).
$$

Every dead row has position $3e_3$ in both inputs. In both populations the
actual two alive fitnesses and the optional acceptance probability are exactly

$$
z=\frac{19/200}{\sqrt{(19/200)^2+1/25}},\quad
F_1=\frac{11}{10}\left(\frac{2}{1+e^z}+\frac1{10}\right),\quad
F_2=\frac{11}{10}\left(\frac{2}{1+e^{-z}}+\frac1{10}\right),\quad
p=\frac{F_2-F_1}{F_1+10^{-6}}.
\tag{KU.R1}
$$

In particular $F_2-F_1>2/5$ and $0<p<1$; their diagnostic values are
$F_2-F_1\simeq0.4648530567$ and $p\simeq0.4755167716$.
They are independent of $N$ and $\delta$. The complete raw output laws satisfy

$$
\mathscr Q_{\rm mark}(S_N,T_N)=\frac{2\alpha\delta^2}{N},\qquad
\inf_{\pi\in\Pi(\widehat{\mathcal P}_N(S_N),
                       \widehat{\mathcal P}_N(T_N))}
\mathbb E_\pi\mathscr Q_{\rm mark}\ge q_xa_h^2\delta^2.
\tag{KU.R2}
$$

For the whole-update survivor-conditioned laws the same infimum has lower
limit at least $q_xa_h^2\delta^2$ as $N\to\infty$.
Thus no finite coefficient independent of $N$ bounds the complete output
quadratic by its input quadratic on all nonextinct states. An affine remainder
$r_N$ in such a bound must satisfy
$\liminf_Nr_N\ge q_xa_h^2\delta^2>0$.
This counterexample contains neither an equal nor a near-equal alive fitness.
:::

:::{prf:proof}
**Every measurement and normalizer.** Alive self-exclusion leaves exactly
one measurement companion per alive row, namely the other row. The two actual
smoothed separations are equal, even after the perpendicular translation and
the radial feature squashing. Their alive-only global standardized diversity
values are therefore exactly zero. The two oriented quadratic rewards are
$-(1+\delta^2)/2$ and $-(81/100+\delta^2)/2$ in $T_N$.
Global reward centering removes the common $-\delta^2/2$ exactly. Their
reward difference is $19/200$, their reward standard deviation with the
original floor is $\tfrac12\sqrt{(19/200)^2+1/25}$, and their standardized
values are $-z,z$. The actual logistic product gives (KU.R1).

For explicit strict separation, squaring positive quantities gives
$2/5<z<1/2$. Also $e^{2/5}>37/25$ from the first four terms of its
positive Taylor series. Hence
$F_2-F_1=(11/5)(e^z-1)/(e^z+1)>
(11/5)(12/62)>2/5$.
For the unsaturated gate, $e^{1/2}<7/4$ gives
$F_1>(11/10)(2/(1+7/4)+1/10)> (F_1+F_2)/3=121/150$,
where $F_1+F_2=121/50$. Thus $F_2<2F_1$ and $0<p<1$.

**Every donor and revival.** The alive cloning companion is also
deterministically the other alive row. Row two persists; row one optionally
copies row two with the computed probability $p$. Each dead row mandatorily
revives from one of rows $1,2$. Weighted revival retains the actual probabilities

$$
q_j(\delta)=\frac{\exp[-((6/5)^2+|S_2(x_j+\delta e_2)|^2)/8]}
 {\sum_{k=1}^2\exp[-((6/5)^2+|S_2(x_k+\delta e_2)|^2)/8]}.
\tag{KU.R3}
$$

This follows from the actual dead feature $S_2(3e_3)=(6/5)e_3$, zero
velocities and companion width two. No assertion that these probabilities
are unchanged by translation is required: every possible frozen alive source
has second coordinate zero in $S_N$, and $\delta$ in $T_N$.
The conditional source probabilities sum to one in each actual law.
Accepted rows add their original independent Gaussian jitter; persisting
rows add none. Thus each prepared row's second coordinate has mean zero or
$\delta$, respectively, conditional on every pre-jitter source pattern.
All frozen component velocities, their means and relative parts are zero;
every component Haar rotation therefore gives zero collision velocity,
including the retained velocities of dead rows.

**Both kicks, unrestricted innovations and the cap.** The first viscous
force is zero for both actual normalization matrices at every prepared array.
The first external kick is $-tX_i$. The two drifts, OU stage and final
position innovation give the exact retained position

$$
x_i^+=a_hX_i+tq\xi_i^O+s\xi_i^x.
\tag{KU.R4}
$$

The second external and dense viscous forces use this realized noisy array
and its uncapped OU velocities. They change only the final velocity.
The original cap changes only that velocity, and terminal classification
retains every physical position. Therefore their complete contribution is
accounted for by the pointwise Schur-complement inequality (KU.O11), valid
for every resulting velocity and every terminal mark. All Gaussians in
(KU.R4), including accepted-row jitter, remain unrestricted.

Writing $G_N=N^{-1}\sum_i x_i^+\cdot e_2$, integration of these independent
centered innovations gives
$\mathbb E_{S_N}G_N=0$ and $\mathbb E_{T_N}G_N=a_h\delta$.
For every coupling, (KU.O11), Jensen across slots and Jensen under the
coupling imply $\mathbb E\mathscr Q_{\rm mark}\ge
q_x|\mathbb E G_N-\mathbb E\widetilde G_N|^2$.
Only the two alive input positions differ, proving (KU.R2).

**Original survivor conditioning.** Put
$\sigma_*^2=\sigma_h^2+a_h^2\sigma_J^2$.
Conditional on any actual source pattern, the second-coordinate Gaussian
rows are independent, have the same conditional mean $0$ or $a_h\delta$,
and each has variance at most $\sigma_*^2$. The pattern mixture does not
add a variance of conditional means. Consequently

$$
\operatorname{Var}G_N\le\sigma_*^2/N,\qquad
\mathbb EG_N^2\le a_h^2\delta^2+\sigma_*^2/N.
\tag{KU.R5}
$$

The already proved complete-kernel extinction bound gives
$H_N\le e_N=(1-a_*)^N\to0$ in either population. Subtracting the actual
extinction expectation before dividing by the actual survival probability
gives the explicit bound

$$
|\mathbb E[G_N\mid\mathrm{survival}]-\mathbb EG_N|
\le\frac{e_N|a_h|\delta+
 \sqrt{e_N(a_h^2\delta^2+\sigma_*^2/N)}}{1-e_N}\longrightarrow0.
\tag{KU.R6}
$$

The two conditioned mean differences therefore tend to $a_h\delta$.
The same pointwise inequality and Jensen prove the survivor assertion.
Insert either lower bound into the proposed affine inequality and let
$N\to\infty$ to obtain the required positive remainder. $\square$
:::

:::{prf:corollary} Taking a concave power does not fix revival normalization
:label: cor-ku-strict-gap-revival-concave-obstruction

For every fixed $r>0$, neither the raw nor the survivor-conditioned complete
kernel admits a population-independent multiplicative bound for
$\mathscr Q_{\rm mark}^{r/2}$ on all nonextinct inputs. In the same strict-gap
construction, with $m=|a_h|\delta>0$, every raw output coupling satisfies

$$
\mathbb E\mathscr Q_{\rm mark}^{r/2}\ge
q_x^{r/2}(m/2)^r
\left(1-\frac{32\sigma_*^2}{Nm^2}\right)_+.
\tag{KU.R7}
$$

For survivor-conditioned couplings replace the last fraction by
$32\sigma_*^2/[Nm^2(1-e_N)]$.
The entering cost is $(2\alpha\delta^2/N)^{r/2}$ and tends to zero,
whereas these lower bounds tend to a strictly positive constant.
:::

:::{prf:proof}
Chebyshev and (KU.R5) give probability at most
$16\sigma_*^2/(Nm^2)$ for either averaged coordinate to differ from its own
raw mean by at least $m/4$. In any coupling the union of these two events
has probability at most their sum. On its complement the difference of the
averaged coordinates has absolute value at least $m/2$.
The pointwise completed-square inequality gives
$\mathscr Q_{\rm mark}\ge q_x|G_N-\widetilde G_N|^2$, proving (KU.R7).
Under original survivor conditioning each marginal's deviation probability
is at most its unconditioned value divided by its survival probability,
which is at least $1-e_N$. This proves the asserted conditional version
without using concave Jensen or presuming unchanged conditional means.
The limiting lower bounds exclude a finite coefficient independent of $N$.
$\square$
:::
