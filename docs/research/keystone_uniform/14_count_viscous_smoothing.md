# Gaussian smoothing through active preparation and dense count viscosity

This proof note establishes quantitative regularity of the actual conservative
finite-swarm update. The completed results are preparation bounds, an inverse
for the first count-viscous drift, and a joint density formula through the
second kick. A population-uniform contraction estimate for the complete active
viscous law still requires the mixed estimate stated in Section 5.

The nonviscous convergence results in
{prf:ref}`thm-slcw-finite-uniform-law` and
{prf:ref}`thm-slcw-alive-uniform-law` are unchanged.

(sec-kv-smoothing-register)=
## 1. The actual conservative law and its Gaussian input

:::{div} feynman-prose
The final position noise is especially convenient. At the beginning of the
next update, every position has a fresh Gaussian displacement, independent of
the already stored velocities. The positions and velocities may otherwise
share the entire history of the swarm. We will use this fresh displacement
without declaring the walkers independent.

Copying seems to spoil the argument: several recipients can use the same
donor position. The recipient jitters repair precisely this difficulty. To
move a donor coordinate while keeping its copied recipients fixed, compensate
their own jitters. The expected number of compensations is bounded by the
actual accepted donor probabilities.
:::

:::{prf:definition} Conservative count-viscous smoothing regime
:label: def-kv-smoothing-regime

Fix $N\ge2$, $d\ge1$, and $D=\mathbb R^d$. All rows are alive. Use the
canonical current-frame measurement, independent distinct donor draws,
simultaneous copying, connected-component Haar collisions, recipient Gaussian
jitter of variance $\sigma_J^2>0$, and smooth stored-velocity cap
$C_V(z)=Vz/(V+|z|)$, $V>0$. There is no history contribution, terminal killing,
or extra curl force. Before preparation $|v_i|\le V$; after the component
collision $|v_i^{\rm p}|\le V_c=(1+2|\alpha_{\rm col}|)V$.

Put
$$
t=h/2>0,\quad c=e^{-\gamma h}>0,\quad b=t(1+c),\quad
\eta=bt,\quad q>0,\quad s>0,\quad \lambda=\eta^{-1}.
$$
The force is $F(x)=-\lambda x$. The raw reward is either
$R(x)=-\lambda|x|^2/2$, or a declared $C^1$ bounded reward with oscillation
$R_{\rm osc}$ and $\sup|\nabla R|\le L_R$.

The comparison features are
$$
z_i=(S_{R_x}(x_i),\sqrt{\lambda_{\rm alg}}S_{R_v}(v_i)),\qquad
S_R(x)=Rx/(R+|x|),\qquad
D_*=2\sqrt{R_x^2+\lambda_{\rm alg}R_v^2}.
$$
For $a=D,C$, define $\kappa_a=e^{-D_*^2/(2\epsilon_a^2)}>0$,
the actual normalized Gaussian companion probabilities $P_a(j\mid i)$,
and $S_b=\sqrt{D_*^2+\delta_D^2}-\delta_D$. The measured raw diversity
has range length at most $S_b$. Reward and diversity use their actual sampled
empirical means and variances, with floors $\sigma_r,\sigma_s>0$.
The fitness functions, interval $[F_*,F^*]$, derivative bounds $H_r,H_s$,
and gate bounds $L_{\rm rec},L_{\rm don}$ are precisely those of
{prf:ref}`lem-ku-standardization` and {prf:ref}`lem-ku-gate`.
Set
$$
a_* =\min\left\{1,\frac{F^*-F_*}{s_c(F_*+\epsilon_c)}\right\},\qquad
c_*=a_*/\kappa_C,\qquad
\ell_\rho=e^{-1/2}/\rho.
$$
All floors, companion widths, and $\rho$ are positive. Fitness exponents
may be strictly positive. No lower bound on a realized fitness gap is imposed.

For an array $(X,v)$ after preparation, use the actual count operator
$$
\mathcal C_i(X,v)=\frac1N\sum_{j\ne i}
 e^{-|X_i-X_j|^2/(2\rho^2)}(v_j-v_i).
$$
The actual stages are
$$
u=v-t\lambda X+t\nu\mathcal C(X,v),\quad X_1=X+tu,\quad
w=cu+q\xi,\quad y=X_1+tw,
$$
$$
z=w-t\lambda y+t\nu\mathcal C(y,w),\qquad
X^+=y+s\zeta,\quad v^+=C_V(z).
$$
Within each own transition $\xi,\zeta$ have independent standard Gaussian
entries and are independent of the preceding preparation. The second count
matrix uses the actual noisy array $y$.
:::

:::{prf:definition} Normalized spatial variation of a correlated array law
:label: def-kv-array-bv

For a finite measure $\Lambda$ on
$\mathbb R^{Nd}\times\overline B_{V_c}^{,N}$, set
$$
\mathcal B_N^x(\Lambda)=\frac1N\sum_{i=1}^N\sum_{a=1}^d
 |D_{X_{i,a}}\Lambda|(\mathbb R^{2Nd}),
$$
provided the distributional derivatives are finite signed measures.
For probability measures use
$\operatorname{TV}(\mu,\nu)=\sup_A|\mu(A)-\nu(A)|$; the full variation
of their difference is $2\operatorname{TV}(\mu,\nu)$.
Write $G_p=\mathbb E|Z|^p=2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2)$ for
$Z\sim N(0,I_d)$, and $g_1=\mathbb E|Z_1|=\sqrt{2/\pi}$.
:::

:::{prf:lemma} The preceding actual step supplies Gaussian positions and moments
:label: lem-kv-current-gaussian

Suppose $0\le t\nu\le1$. After any complete step in
{prf:ref}`def-kv-smoothing-regime`, the next entering array admits
$$
x_i=Y_i+sZ_i,
$$
where, conditional on the entire latent array $(Y,v)$, the $Z_i$ are
independent standard Gaussians. The latent array can be correlated. For $p\ge1$,
$$
\mathbb E\frac1N\sum_i|Y_i|^p
\le(bV_c+tqG_p^{1/p})^p,
\qquad
M_p:=\mathbb E\frac1N\sum_i|x_i|^p
\le(bV_c+\tau G_p^{1/p})^p,
\quad\tau^2=t^2q^2+s^2.
\tag{KVS.1}
$$
In particular $M_5<\infty$, uniformly in $N$ and the preceding entering array.
:::

:::{prf:proof}
The matrix $A_X=I-t\nu L_X$ is row stochastic with nonnegative entries,
where $L_Xv=-\mathcal C(X,v)$. Therefore $|(A_Xv)_i|\le V_c$.
Since $1-\eta\lambda=0$, the exact landing position is
$$
y_i=b(A_Xv)_i+tq\xi_i.
$$
The stored velocity is computed before drawing $\zeta$. Conditional on
$(y,v^+)$, the final position noise remains an independent Gaussian array.
Minkowski on the probability space augmented by a uniformly sampled label
proves the first bound. Conditional on the preceding prepared array,
$tq\xi_i+s\zeta_i$ is Gaussian with variance $\tau^2I_d$; Minkowski
proves the second bound. No independence between prepared rows was used.
:::

(sec-kv-active-source-bv)=
## 2. Joint spatial variation through the actual active preparation

:::{div} feynman-prose
We need regularity of the whole correlated array, because its own positions
build the viscous matrix. A smooth sampled-row density alone would not give
this: fixing the random empirical matrix could select a nonsmooth conditional
law. The following calculation keeps every sampled measurement and graph
pattern inside its actual probability.
:::

:::{prf:theorem} Explicit joint-array BV bound for active preparation
:label: thm-kv-active-joint-bv

Let the entering array satisfy the Gaussian representation in
{prf:ref}`lem-kv-current-gaussian`, with stored velocities bounded by $V$ and
moment budgets $M_1,M_5$. Let $\Lambda_N^{\rm p}$ be its actual prepared
joint law. Define
$$
\begin{aligned}
B_a&=\frac{2D_*}{\epsilon_a^2}(1+\kappa_a^{-1}),\qquad a=D,C,\\
K_s&=\frac2{\sigma_s}+
          \frac{2S_b}{3\sqrt3\,\sigma_s^2},\\
J_r&=H_r\left(\frac{2\lambda M_1}{\sigma_r}
                        +\frac{\lambda^3M_5}{\sigma_r^3}\right),\\
J_s&=H_sK_s(1+\kappa_D^{-1}),\\
B_{\rm pat}&=B_D+2a_*B_C+
             2(L_{\rm rec}+\kappa_C^{-1}L_{\rm don})(J_r+J_s),\\
B_{\rm p}&=d\left[\frac{g_1}{s}
               +\frac{(a_*+2c_*)g_1}{\sigma_J}+B_{\rm pat}\right].
\end{aligned}
\tag{KVS.2}
$$
Then
$$
\boxed{\ \mathcal B_N^x(\Lambda_N^{\rm p})\le B_{\rm p}\ },
\tag{KVS.3}
$$
for every $N\ge2$. For a bounded reward, replace $J_r$ by
$$
J_r^{\rm bd}=H_rL_R\left(
\frac2{\sigma_r}+\frac{2R_{\rm osc}}{3\sqrt3\,\sigma_r^2}\right);
\tag{KVS.4}
$$
then the proof needs no entering positional moment bound.

The uniformly sampled prepared phase law
$\lambda_N^{\rm p}=N^{-1}\sum_i\operatorname{Law}(X_i,v_i^{\rm p})$
consequently satisfies, for all $\ell\in\mathbb R^d$,
$$
\operatorname{TV}(\tau_\ell\lambda_N^{\rm p},\lambda_N^{\rm p})
\le\tfrac12B_{\rm p}|\ell|_1.
\tag{KVS.5}
$$
Here $\tau_\ell$ translates positions and preserves velocities. All bounds
include exact and partial fitness ties.
:::

:::{prf:proof}
Condition first on the entire latent $(Y,v)$. The entering positions have
the product Gaussian density $\prod_i\varphi_s(x_i-Y_i)$, while the frozen
velocity array may be arbitrary in $\overline B_V^{\,N}$. Include the
measurement vector $m$, outgoing tokens $e$ (accepted donor or no edge),
and component Haar matrices in the preparation pattern. For fixed $m,e$
the collision output depends on $v,e$ and its Haar matrices, and does not
depend further on $x$. Recipient jitters are independent of this pattern.
We bound derivatives of its actual probability weight before using that fact.

Fix an input coordinate $x_{i,a}$. The map $S_R$ has derivative operator
norm at most one. A Gaussian comparison weight has logarithmic derivative
of absolute value at most $D_*/\epsilon_b^2$ when either endpoint changes.
Differentiating a normalized row gives
$$
\partial P_j=P_j(\partial\log w_j-\sum_kP_k\partial\log w_k).
$$
For row $i$ this bounds its full derivative variation by
$2D_*/\epsilon_b^2$. In another row only its candidate $i$ changes;
its full derivative variation is at most
$2(D_*/\epsilon_b^2)P_b(i\mid k)$.
Since $P_b(i\mid k)\le[(N-1)\kappa_b]^{-1}$, summing over the rows
gives $B_b$. Product differentiation proves the same bound for the full
measurement-vector law.

For fixed $m$, differentiation of the diversity array affects its own
row and rows measuring $i$. Its derivative magnitude in any such row is
at most one. The empirical $L^1$ standardizer bound in
{prf:ref}`lem-ku-standardization` therefore gives
$$
\sum_k|\partial_{x_{i,a}}F_k|_{\rm diversity}
\le H_sK_s(1+\#\{k\ne i:m_k=i\}).
$$
The conditional expected incoming measurement count is at most
$\kappa_D^{-1}$. Its contribution is $J_s$.

For the quadratic reward let $r_i=-\lambda|x_i|^2/2$,
$y_k=r_k-\bar r$, and $\sigma=(\operatorname{Var}_N(r)+\sigma_r^2)^{1/2}$.
The exact derivative of the empirical standardizer yields
$$
\sum_k|\partial_{r_i}Z_k|
\le\frac2{\sigma_r}
   +\frac{|y_i|\,N^{-1}\sum_k|y_k|}{\sigma_r^3}.
$$
Put $T_2=N^{-1}\sum_k|x_k|^2$. Then
$|y_i|\le(\lambda/2)(|x_i|^2+T_2)$,
$N^{-1}\sum_k|y_k|\le\lambda T_2$, and
$|\partial_{x_{i,a}}r_i|\le\lambda|x_i|$. Averaging the resulting
fitness derivative over $i$ gives
$$
\frac1N\sum_i\sum_k|\partial_{x_{i,a}}F_k|_{\rm reward}
\le H_r\left[\frac{2\lambda}{\sigma_r}\|x\|_{1,N}
 +\frac{\lambda^3}{2\sigma_r^3}
       (\|x\|_{3,N}^3T_2+\|x\|_{1,N}T_2^2)\right].
$$
Empirical Hölder bounds both degree-five products by
$N^{-1}\sum_i|x_i|^5$. Taking expectation gives $J_r$.
For bounded reward, the empirical $L^1$ standardizer bound directly gives
the replacement (KVS.4). This calculation uses the actual random global
mean and variance; it never freezes their values across compared inputs.

For fixed measurements, an accepted-token mass is
$P_C(j\mid k)\mathfrak a(F_k,F_j)$. The full derivative variation of a
row's complete token law is at most twice the sum of the derivatives of
its accepted masses: its no-edge mass is their complement. The direct
companion contribution, summed over rows, is at most $2a_*B_C$.
The gate contribution is at most
$$
2\left(L_{\rm rec}+\kappa_C^{-1}L_{\rm don}\right)
            \sum_k|\partial_{x_{i,a}}F_k|.
$$
The incoming normalized donor sum is at most $\kappa_C^{-1}$, which
explains that coefficient. Product differentiation of the independent
outgoing tokens and then the measurement law bounds the averaged full
variation of all discrete pattern derivatives by $B_{\rm pat}$.
Continuous piecewise differentiability of the gate suffices. In particular
there is no division by an acceptance probability at a tie.

Now translate prepared coordinate $X_{i,a}$ in a fixed pattern. If row $i$
copied a donor, translate its own jitter coordinate. The Gaussian derivative
has $L^1$ norm $g_1/\sigma_J$ and this branch has probability at most $a_*$.
If row $i$ did not copy, translate its entering coordinate $x_{i,a}$.
For every accepted recipient $k$ whose frozen donor is $i$, translate its
own jitter by the opposite amount. Then all other prepared positions stay
fixed. The prepared velocities also stay fixed, since the graph and Haar
matrices in this term are fixed.

Differentiation in this change of variables has three contributions. The
entering Gaussian density costs at most $g_1/s$. The compensation jitters
cost their count times $g_1/\sigma_J$. Conditional on the entering array
and measurements, that count has expectation at most
$\sum_{k\ne i}a_*/[(N-1)\kappa_C]=c_*$; the larger $2c_*$ in (KVS.2)
is an admissible bound. The derivative of the discrete pattern weight
costs its full variation, bounded above. Restricting to the no-copy branch
can only decrease the sum of absolute pattern derivatives.

For a smooth compactly supported test function of the entire prepared
array, these changes of variables prove the stated uniform bound on its
distributional coordinate derivative. Sum over coordinates, average over
labels and the latent array, and apply the representation theorem for
bounded linear functionals on continuous compactly supported tests. This
gives finite signed derivative measures and (KVS.3). One may first use
smooth approximations of the gate and then pass to its globally Lipschitz
limit; all bounds are uniform in that approximation.

Pushforward to a selected row decreases full variation. Averaging these
row derivative measures bounds the sum of spatial derivative variations
of $\lambda_N^{\rm p}$ by $B_{\rm p}$. Integrate successive coordinate
translations, or first spatially mollify and pass in variation, to obtain
(KVS.5). No product law for the prepared positions or velocities was used.
:::

:::{prf:corollary} A nonempty interval with both fitness exponents positive
:label: cor-kv-positive-small-exponents

Set $p_r=\theta\bar p_r$, $p_s=\theta\bar p_s$ with
$\bar p_r,\bar p_s>0$. Define
$$
M=\sum_{a=r,s}\bar p_a
       \max\{|\log\eta_a|,|\log(\eta_a+A_a)|\},\quad
\Delta=\sum_{a=r,s}\bar p_a\log\frac{\eta_a+A_a}{\eta_a}>0,
\quad a_0=\frac{e^M\Delta}{s_c\epsilon_c}.
$$
For $0<\theta\le\min\{1,(4a_0)^{-1}\}$, $a_*\le1/4$ and
$c_*\le\theta a_0/\kappa_C$. All constants in (KVS.2) are finite and
population independent. Each $H_a$ is bounded by
$\theta e^M\bar p_aA_a/(4\eta_a)$. The results apply at fitness ties.
:::

:::{prf:proof}
For $0<\theta\le1$, the logarithms of the fitness endpoints have
absolute value at most $M$ and their difference is $\theta\Delta$.
The mean-value theorem for the exponential gives
$F^*-F_*\le e^M\theta\Delta$. Bound the gate denominator below by
$s_c\epsilon_c$. Differentiating the positive logistic powers gives the
displayed bound for $H_a$. The regularizer bounds remain valid when every
raw reward or diversity value is equal.
:::

The expected common-component size used in the finite Hamming preparation
coupling remains $M_f=e^{8c_*}$ from
{prf:ref}`lem-slcw-finite-preparation`: after conditioning on unequal token
rows, a common accepted edge has probability at most $4c_*/N$ and strictly
increases the frozen fitness. The BV proof above instead counts the actual
incoming copied recipients in a fixed pattern. Both estimates use the actual
finite graph and retain its global measured normalizers.

(sec-kv-count-joint-density)=
## 3. The actual finite-array inverse and joint density

:::{div} feynman-prose
We now change coordinates on the entire array. This matters: the first
viscous force uses the same array whose density we are computing. After the
first drift, the OU noise supplies the missing velocity coordinates. The
second kick is linear in the full velocity list when its landing positions
are fixed. Its determinant is consequently a matrix determinant, rather than
a product of independent one-row determinants.
:::

:::{prf:theorem} Full-array count-viscous density from the prepared BV law
:label: thm-kv-count-joint-density

In {prf:ref}`def-kv-smoothing-regime`, set
$$
A=1-t^2\lambda=\frac{c}{1+c}>0,\quad
L_0=4dV_c\ell_\rho,\quad L_2=8dV_c/\rho^2,
$$
and require
$$
0<t\nu<\tfrac12,\qquad \Delta_A:=A-t^2\nu L_0>0.
\tag{KVS.6}
$$
Put $L_I=\Delta_A^{-1}$, $D_1=t\lambda+t\nu L_0$, and
$$
B_1=L_I B_{\rm p}
       +dL_I^2t^2\nu L_2
       +dcD_1L_Ig_1/q.
\tag{KVS.7}
$$
The actual joint law of $(X_1,w)$ has a Lebesgue density $f_N$ satisfying
$$
\frac1N\sum_{i,a}\|\partial_{X_{1,i,a}}f_N\|_1\le B_1,
\qquad
\frac1N\sum_{i,a}\|\partial_{w_{i,a}}f_N\|_1\le dg_1/q.
\tag{KVS.8}
$$
The actual pre-second-kick joint density is
$\widetilde f_N(y,w)=f_N(y-tw,w)$, with
$$
\mathcal B_N^y(\widetilde f_N)\le B_1,\qquad
\mathcal B_N^w(\widetilde f_N)\le dg_1/q+tB_1.
\tag{KVS.9}
$$
For the actual second-kick matrix $W_y=I-t\nu L_y$,
$$
g_N(y,z)=
\frac{\widetilde f_N(y,W_y^{-1}(z+t\lambda y))}
                  {(\det W_y)^d}
\tag{KVS.10}
$$
is the full-array pre-cap density. Here $\det W_y>0$,
$(\det W_y)^d\ge(1-t\nu)^{Nd}$, and
$\|W_y^{-1}\|_{1\to1},\|W_y^{-1}\|_{\infty\to\infty}
\le(1-2t\nu)^{-1}$ on the scalar row indices. These formulas retain
the actual empirical field and all correlations of $y,w$.
:::

:::{prf:proof}
For fixed prepared velocities, the first drift map on all $Nd$ position
coordinates is
$$
H_v(X)=AX+tv+t^2\nu\mathcal C(X,v).
$$
The kernel gradient norm is at most $\ell_\rho$ and its Hessian operator
norm is at most $\rho^{-2}$. For an individual pair $(i,j)$, the two
nonzero derivative blocks are rank-one matrices containing $v_j-v_i$.
Their coordinate row and column sums, summed over the incident pairs,
bound both coordinate matrix norms of $D_X\mathcal C$ by $L_0$.
For example, the off-diagonal block has norm at most
$2dV_c\ell_\rho/N$, and the diagonal block is the sum of the incident
blocks; adding the two contributions gives $4dV_c\ell_\rho$.

For a fixed coordinate $X_{i,a}$, differentiating this Jacobian involves
only pairs incident to $i$. Each contributes four matrix blocks, whose
total absolute coordinate-entry sum is at most $2dV_c/(N\rho^2)$ per
block. Consequently
$$
\sum_{j,k}\big|\partial_{X_{i,a}}
                    (D_X\mathcal C)_{jk}\big|\le L_2.
\tag{KVS.11}
$$
Here $j,k$ range over the scalar coordinates of the full array.

Given a target $X_1$, solve
$X=A^{-1}[X_1-tv-t^2\nu\mathcal C(X,v)]$. By (KVS.6) this is a
contraction in either coordinate $\ell^1$ or $\ell^\infty$ norm. It has
a unique solution. The smooth inverse function theorem and the Neumann
series show that $H_v$ is a global diffeomorphism and that its inverse
Jacobian has both coordinate matrix norms at most $L_I$.
Every eigenvalue of its Jacobian has positive real part, so its real
determinant is positive. The boundedness of $\mathcal C(X,v)$ also
gives properness directly.

First suppose the prepared measure is smooth in positions, allowing an
arbitrary velocity mixing measure. Its pushed density at $X_1$, conditional
on the prepared velocity coordinates, is its original spatial density
divided by $\det DH_v$. The transformed derivative of its density costs
at most $L_I\mathcal B_N^x(\Lambda_N^{\rm p})$.
The identity
$\partial\log\det DH_v=\operatorname{tr}[(DH_v)^{-1}\partial DH_v]$,
(KVS.11), and the inverse matrix bounds show that the averaged sum of
absolute logarithmic determinant derivatives in $X_1$ is at most
$dL_I^2t^2\nu L_2$.

Conditional on those coordinates, $w$ has product Gaussian density with
mean $\mu(X_1,v)=c[v-t\lambda X+t\nu\mathcal C(X,v)]$.
Both coordinate matrix norms of $D_{X_1}\mu$ are at most $cD_1L_I$.
A coordinate Gaussian derivative has integral $g_1/q$. Product
differentiation and the column-sum bound therefore add at most
$dcD_1L_Ig_1/q$ to the normalized spatial variation. Direct differentiation
in $w$ gives $dg_1/q$. Integration over the possibly correlated velocity
mixing measure decreases these variation bounds. This proves (KVS.8)
in the smooth case.

For general prepared input, convolve in the position coordinates only.
For each fixed finite $N$, the finite derivative variations proved in
{prf:ref}`thm-kv-active-joint-bv` imply convergence of these convolutions
to the input in full variation. Both the deterministic drift and the OU
transition decrease full variation. The resulting smooth-case densities
thus converge in $L^1$ to a density of the actual output. Distributional
variation is lower semicontinuous under this convergence, which proves
(KVS.8). This argument requires no regularity of a conditional empirical
provider. The shear $(X_1,w)\mapsto(y=X_1+tw,w)$ has determinant one;
the chain rule proves (KVS.9).

Finally, the scalar count Laplacian $L_y$ is symmetric positive
semidefinite and bounded above by the complete unit-weight count
Laplacian, whose spectrum is contained in $[0,1]$. Hence the eigenvalues
of $W_y$ lie in $[1-t\nu,1]$. Its inverse exists. Its diagonal entries
are at least $1-t\nu$, and its off-diagonal row and column sums are at
most $t\nu$. The Neumann series applied to $I-W_y$ bounds its inverse
coordinate norms by $(1-2t\nu)^{-1}$. At fixed $y$, the velocity map is
$z=W_yw-t\lambda y$, independently in each physical coordinate.
Changing variables proves (KVS.10). Smooth capping and the final independent
position noise are the original subsequent Markov operations.
:::

:::{prf:lemma} A mixed weighted derivative supplied directly by OU noise
:label: lem-kv-weighted-ou-score

For the density $f_N(X_1,w)$ in (KVS.8), set
$\mu=cu$ and $M_\mu=\mathbb E N^{-1}\sum_i|\mu_i|$. Then
$$
\frac1{N^2}\sum_{i,j,a}\int
        (1+|w_i|)|\partial_{w_{j,a}}f_N|\,dX_1\,dw
\le d\left[\frac{g_1}{q}(1+M_\mu)+G_1g_1+\sqrt d\right].
\tag{KVS.12}
$$
A sufficient explicit moment bound is
$$
M_\mu\le cV_c+ct\lambda M_{\rm p,1},\qquad
M_{\rm p,1}\le(1+c_*)M_1+\sigma_JG_1.
\tag{KVS.13}
$$
:::

:::{prf:proof}
Before mixing the prepared inputs, the absolute derivative in
$w_{j,a}$ is bounded by its Gaussian density times $|\xi_{j,a}|/q$.
Use $|w_i|\le|\mu_i|+q|\xi_i|$ and average over $i,j$.
For $i\ne j$, $\mathbb E|\xi_i||\xi_{j,a}|=G_1g_1$;
for $i=j$, Cauchy--Schwarz bounds it by $\sqrt d$.
The Gaussian array is independent of its own prepared mean. Integration
over the prepared law proves (KVS.12).
The first count kick is a convex velocity average before adding
$-t\lambda X$, which gives the first bound in (KVS.13).
For the second, a frozen input position can be retained at its own slot
or copied to recipients. Its conditional mean number of accepted incoming
copies is at most $c_*$. Sum their first moments and then the independent
recipient jitter moments. This gives the second bound.
:::

(sec-kv-explicit-profile)=
## 4. An explicit nonempty positive-viscosity smoothing regime

:::{div} feynman-prose
The inequalities above can all hold with viscosity and both selection
exponents positive. This checks that the regularity result has an actual
parameter regime. It is a regularity result: its constants have not yet
been turned into a contraction factor for the full active viscous law.
:::

:::{prf:example} A primitive smoothing profile
:label: ex-kv-count-smoothing-profile

Take $d=1$, $h=2$, $\gamma=0$, $q=s=1$, $V=10^{-3}$,
$\alpha_{\rm col}=1/2$, $\sigma_J=1/10$, $\rho=4$, and $\nu=1/4$.
Thus $t=c=1$, $b=\eta=2$, $\lambda=1/2$, $V_c=1/500$, and
$F(x)=-x/2$, with raw reward $R(x)=-x^2/4$.
The actual OU amplitude is $b_O=1/\sqrt2$ and final position diffusion
amplitude is $\sigma_x=1/\sqrt2$, using their continuous $\gamma=0$
definitions.

Use $R_x=R_v=1$, $\lambda_{\rm alg}=1$,
$\epsilon_D=\epsilon_C=4$, $\delta_D=10^{-3}$,
$\sigma_r=10^6$, $\sigma_s=1$, $s_c=\epsilon_c=1$,
$A_r=A_s=\eta_r=\eta_s=1$, and
$p_r=p_s=\theta$, with
$0<\theta\le\min\{1,(4a_0)^{-1}\}$ from
{prf:ref}`cor-kv-positive-small-exponents`.

Since $\ell_\rho<1/4$, $L_0<1/500$, and
$\Delta_A>1/2-1/2000>0$. Also $t\nu=1/4<1/2$.
All moment and derivative constants (KVS.1)--(KVS.13) are finite,
population independent, and use the unbounded raw quadratic reward.
This example supplies no contraction certificate for the complete active
viscous law.
:::

(sec-kv-mixed-closure)=
## 5. The mixed feedback estimate still needed for convergence

:::{div} feynman-prose
A density derivative measures the response to moving coordinates within one
law. Contraction asks a different question: how does that response change
when the entire entering law changes? An absolute bound on the first response
cannot be multiplied by a small discrepancy between two input laws. We need
an estimate for that difference itself.

There is a second issue. An OU velocity contains an unbounded force term,
even though the stored velocity is capped. The second viscous kick reads this
uncapped velocity. The previous Gaussian position noise controls its moments
in the regularized law class, and that control must remain attached to the
labels charged in a comparison.
:::

:::{prf:remark} Precise scope of the completed smoothing estimates
:label: rem-kv-mixed-feedback-open

Theorems {prf:ref}`thm-kv-active-joint-bv` and
{prf:ref}`thm-kv-count-joint-density` give absolute normalized derivative
bounds for each own actual correlated finite-swarm law. Formula (KVS.10)
also gives its exact second-kick density. They do not yet establish an
estimate of the form
$$
\mathcal D_N\big((\Pi-\widetilde\Pi)(P_\nu^2-P_0^2)\big)
\le C_{\rm mix}\nu\,\mathcal D_N(\Pi-\widetilde\Pi),
\tag{KVS.14}
$$
on a specified regularized law class, with $C_{\rm mix}$ independent of
$N$. Here $P_\nu$ denotes the complete active conservative kernel and
$\mathcal D_N$ must be a declared transport or signed-measure norm for
which the nonviscous two-update contraction is proved. Whole-array TV with
an $N$-dependent coefficient does not supply a population-uniform sampled
or empirical-law estimate.

For a deterministic population provider, the rowwise second-kick formula
is $z=(1-t\nu a(y))w-t\lambda y+t\nu m(y)$, where
$a(y)=\int K(y,y')\Lambda_2(dy',dw')$ and
$m(y)=\int K(y,y')w'\Lambda_2(dy',dw')$.
The velocity displacement under a provider variation contains
$t\nu[\Delta m(y)-w\Delta a(y)]$ divided by its positive coefficient.
This identifies the weighted velocity derivative required for that
population calculation. In a finite swarm, $W_y$ is the full random
matrix in (KVS.10); retaining this matrix and the joint law is essential.

The absolute OU estimate (KVS.12) controls an averaged weighted score. A
coupled discrepancy may select labels through their positions, sampled
fitness, and graph components. Establishing (KVS.14) additionally requires
a weighted derivative bound for that selected discrepancy, or a direct
joint-array coupling which keeps its Gaussian bridge marginals correct.
An unconditional fifth-moment bound and an unconditional BV bound do
not imply this selected-discrepancy estimate.

For bounded reward the preparation probabilities have bounded spatial
derivatives independent of entering positions. This simplifies that part
of a possible closure. It leaves the unbounded OU force and the two
correlated count matrices in the mixed kinetic estimate. The weak positive
exponent interval in {prf:ref}`cor-kv-positive-small-exponents` establishes
no value of $C_{\rm mix}$ and no full viscous contraction rate.
:::

:::{prf:proposition} Arbitrary-array Hamming sensitivity of the resonant count kick
:label: prop-kv-resonant-hamming-sensitivity

For the kinetic law in {prf:ref}`def-kv-smoothing-regime`, temporarily
disable cloning. Fix $0<t\nu<1$, $c,q,s,V>0$. Compare $S_N$ with all
entering positions and velocities zero to $T_{N,M}$ with only
$x_1=Me_1$ changed, all entering velocities still zero.
Their normalized entering Hamming distance is $1/N$. For each fixed
$N$, the uniformly sampled output velocity laws satisfy
$$
\lim_{M\to\infty}\mathbb E\bar v(T_{N,M}^+)=-Ve_1,
\qquad \mathbb E\bar v(S_N^+)=0,
$$
where $\bar v=N^{-1}\sum_i v_i$. Consequently the infimum, over all
couplings of the complete output laws, of expected normalized Hamming
distance has limit inferior at least $1/2$.
Thus an arbitrary-array Hamming Lipschitz estimate with a finite
population-independent coefficient is unavailable in this resonant
positive-count-viscosity regime. This proposition makes no claim about
optimal physical Wasserstein contraction or convergence from regularized
input laws.
:::

:::{prf:proof}
Couple the two kinetic computations with the same innovation arrays for
the purpose of computing their marginals. Resonance gives $y_i=tq\xi_i$
in both. In $T_{N,M}$ only the first OU velocity changes, by
$-ct\lambda Me_1$. The second count matrix is therefore identical in
the two computations. Its entry $(W_y)_{i1}$ is strictly positive for
$i\ne1$, because it is $t\nu K(y_i-y_1)/N$. Its entry $(W_y)_{11}$
is at least $1-t\nu>0$. Hence every pre-cap output velocity in
$T_{N,M}$ tends to $-\infty e_1$, and every capped velocity tends to
$-Ve_1$, almost surely for this marginal construction. Bounded convergence
gives its stated mean. The zero-input law is invariant under simultaneous
sign reversal of all Gaussian innovations; the cap is odd and both count
matrices preserve this symmetry. Its mean velocity is zero.

Under any output coupling, capped velocities differ by at most $2V$
at a mismatched row. Thus
$|\mathbb E\bar v(T^+)-\mathbb E\bar v(S^+)|\le
2V\mathbb E N^{-1}\sum_i\mathbf1_{\{S_i^+\ne T_i^+\}}$.
The marginal mean lower bound applies to the infimum over couplings.
It is incompatible with a fixed coefficient times the entering distance
$1/N$ as $N$ increases.
:::

The next proof step is therefore (KVS.14) on a regularized, weighted law
class, or a different complete coupling in a nonresonant confining regime.
The completed BV and density estimates above supply concrete primitives for
that step; they leave its signed mixed constant and resulting rate open.
