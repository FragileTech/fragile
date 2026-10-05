# Reference count viscosity: dissipation and a frozen law gap

(sec-rcb-retained)=
## 1. Retained state and exact target

:::{prf:definition} Reference count residual
:label: def-rcb-retained

The source checkout is `6107b67b9e85259581c1932565c1871e8a7e253a`,
with the working-tree arguments in research records 15, 19 and 24.
Their SHA-256 fingerprints at this proof's first complete draft are,
respectively,
`9191745f8df019e688373955c031756aedccbf7200bb9a6d1680863db241b255`,
`e9187b0cc87c3b4223071248a506023e3c1fe3ad651356ceadbd4856dd591e04`,
and
`d6d2e00f52df2e244318693ee68b6e7773d27eac4d9cfeef6e99669441ba678c`.
The accepted imports are the count-kernel estimates
{prf:ref}`lem-ku-count-noisy-pair-defect`, the actual incoming-column
moment calculation {prf:ref}`lem-pvb-eighth-moment`, and the frozen
root minorization/contraction
{prf:ref}`lem-pvb-frozen-minorization` and
{prf:ref}`thm-pvb-frozen-weighted-contraction`. Their domains remain
unchanged.

The target is a population-independent contraction of an actual law-level
block at $h=1/25$, $\nu=3/10$, preferably with active selection. Two
incoming arms are kept separate:

1. The **finite kinetic arm** disables cloning and death, has entering
   stored speed $V=2$, and retains the entire joint finite-array kernel.
2. The **active frozen population arm** retains the actual preparation
   $J_\mu$, including both positive fitness powers, recipient jitter and
   component collision. It freezes one population $\mu$ and its two
   actual deterministic viscous providers. The prepared speed envelope is
   $V_c=4$ for $\alpha_{\rm col}=1/2$.

The force is $F(x)=-x$, the count Gaussian width is $\rho=1$, and
$d=3$, $\gamma=b_O=1$, $\sigma_x=1/10$. There is no positional
cutoff. Every Gaussian innovation and the native smooth cap are retained.
The actual component collision uses original frozen slot velocities,
rather than donor-copied velocities. A singleton has zero count operator.
All constants below are independent of $N$.

The old absolute perturbation theorem closes a positive interval near
$10^{-6}$; it does not certify $\nu=3/10$. The alternative developed
here retains the negative alignment form before charging the graph
defect. No minimality or finite resource allocation is assumed; reuse
of a proved operator bound does not create an additional noise sample.
The whole-array second graph remains a function of the same OU noise
used in its velocities.

The finite kinetic arm closes dissipation and invariant existence.
The active frozen population arm closes an explicit weighted law gap.
The own-provider nonlinear block comparison remains the residual in
{prf:ref}`rem-rcb-first-missing-interface`. These are distinct
conclusions on distinct carriers.
:::

:::{prf:definition} Actual finite reference kernel
:label: def-rcb-kernel

Write $\langle a,b\rangle_N=N^{-1}\sum_i a_i\cdot b_i$,
$\|a\|_N^2=\langle a,a\rangle_N$, and
$$
t=\frac1{50},\quad c=e^{-1/25},\quad
b=t(1+c),\quad m=1-t^2,\quad a_x=1-tb,
$$
$$
q^2=\frac{1-c^2}{2}>0,\qquad s^2=\frac1{2500}>0,\qquad
\tau^2=t^2q^2+s^2,\qquad \ell=e^{-1/2}.
$$
For $N\ge2$ let
$$
(L_xv)_i=\frac1N\sum_{j\ne i}
 K(x_i-x_j)(v_i-v_j),\qquad K(r)=e^{-|r|^2/2},
\qquad A_x=I-t\nu L_x.
$$
Set $L_x=0$ for $N=1$. With $\nu=3/10$, the complete kinetic
update is
$$
v_A=A_xv,\quad u=v_A-tx,\quad
w=cu+q\xi,\quad y=x+bu+tq\xi,
$$
$$
z=(I-t\nu L_y)w-ty,\qquad
x^+=y+s\chi,\qquad v^+=C_V(z),\qquad
C_V(z)=\frac{Vz}{V+|z|},\quad V=2.
\tag{RCB.1}
$$
The entries of $\xi,\chi$ are independent standard $d$-Gaussians
within each own update. Paired updates may share corresponding entries.
Then $y-\widetilde y$ is deterministic conditional on the entering
arrays, but each noisy second graph depends on the actual shared
Gaussian array.
:::

(sec-rcb-noisy-defect)=
## 2. A sharper defect for the correlated noisy second graph

:::{prf:lemma} Frozen deterministic part and Gaussian Hessian remainder
:label: lem-rcb-noisy-defect

For general width $\rho>0$, put $\ell_\rho=e^{-1/2}/\rho$.
Let $y=y_0+\alpha\xi$ and
$\widetilde y=\widetilde y_0+\alpha\xi$, where $y_0,\widetilde
y_0$ are deterministic arrays and $R=y_0-\widetilde y_0$.
Then
$$
\left(\mathbb E\|(L_y-L_{\widetilde y})\xi\|_N^2\right)^{1/2}
\le \Gamma_\rho(\alpha)\|R\|_N,\qquad
\Gamma_\rho(\alpha)
=\sqrt{2d}\,\ell_\rho+
 \sqrt{8d(d+2)}\,\frac{|\alpha|}{\rho^2}.
\tag{RCB.2}
$$
No independence of a kernel weight and $\xi$ is imposed.
:::

:::{prf:proof}
Write $D_0=L_{y_0}-L_{\widetilde y_0}$ and
$d_{ij}=K_\rho(y_{0,i}-y_{0,j})-
K_\rho(\widetilde y_{0,i}-\widetilde y_{0,j})$.
The symmetric matrix $D_0$ has off-diagonal entries $-d_{ij}/N$
and diagonal entries $N^{-1}\sum_{j\ne i}d_{ij}$. Consequently
$$
\frac1N\operatorname{tr}(D_0^2)
=\frac1{N^3}\sum_i\left[
 \left(\sum_{j\ne i}d_{ij}\right)^2+
 \sum_{j\ne i}d_{ij}^2\right]
\le\frac1{N^2}\sum_{i,j}d_{ij}^2
\le2\ell_\rho^2\|R\|_N^2.
$$
The last step uses $|d_{ij}|\le\ell_\rho|R_i-R_j|$ and
$N^{-2}\sum_{i,j}|R_i-R_j|^2\le2\|R\|_N^2$.
Since $D_0$ is deterministic, independence of the Gaussian coordinates
gives $\mathbb E\|D_0\xi\|_N^2=d\,\operatorname{tr}(D_0^2)/N$.

The Gaussian kernel Hessian has operator norm at most $\rho^{-2}$.
Its tangential eigenvalue is $-\rho^{-2}e^{-u/2}$ and its radial
eigenvalue is $\rho^{-2}(u-1)e^{-u/2}$, where
$u=|r|^2/\rho^2$; both have absolute value at most $\rho^{-2}$.
Applying this bound to the difference of the two kernel increments
shows, pointwise,
$$
\begin{aligned}
&|K_\rho(d+\alpha(\xi_i-\xi_j))
 -K_\rho(d-R_i+R_j+\alpha(\xi_i-\xi_j))\\
&\hspace{8mm}-K_\rho(d)+K_\rho(d-R_i+R_j)|
\le\frac{|\alpha|}{\rho^2}|R_i-R_j|\,|\xi_i-\xi_j|.
\end{aligned}
$$
Here $d=y_{0,i}-y_{0,j}$. Rowwise Jensen, with the zero diagonal
included among the $N$ terms, and
$\mathbb E|\xi_i-\xi_j|^4=4d(d+2)$ give
$$
\mathbb E\|[(L_y-L_{\widetilde y})-D_0]\xi\|_N^2
\le8d(d+2)\alpha^2\rho^{-4}\|R\|_N^2.
$$
Minkowski in $L^2$ proves (RCB.2). For a singleton all operators are
zero, so the same conclusion holds. The remainder bound was pointwise
before Gaussian integration; it retains the kernel/noise correlation.
:::

:::{prf:corollary} Actual second-provider defect at the reference parameters
:label: cor-rcb-reference-second-defect

In paired clone-disabled updates (RCB.1), let
$R=y-\widetilde y$ and use shared $\xi$. Put
$$
A=\frac c{a_x},\qquad k=\frac{ct}{a_x},\qquad
\sigma=\frac{qm}{a_x},\qquad
\varepsilon_2=\nu[2AV\ell+2k+\sigma\Gamma_1(tq)].
$$
Then
$$
\left(\mathbb E\|\nu(L_y-L_{\widetilde y})
                     \widetilde w\|_N^2\right)^{1/2}
\le\varepsilon_2\|R\|_N,\qquad
\varepsilon_2<\frac{161}{200}=0.805.
\tag{RCB.3}
$$
:::

:::{prf:proof}
In each own update, elimination of the entering $x$ gives exactly
$$
w=A v_A-k y+\sigma\xi.
$$
Since $t\nu<1$, $A_x$ is a nonnegative row-stochastic matrix and
$|(v_A)_i|\le V$. The bounded-velocity count estimate gives
$\|(L_y-L_{\widetilde y})\widetilde v_A\|_N
\le2V\ell\|R\|_N$.
Also
$$
(L_y-L_{\widetilde y})\widetilde y
=[L_yy-L_{\widetilde y}\widetilde y]-L_yR,
$$
whose norm is at most $2\|R\|_N$, because $x\mapsto L_xx$
is 1-Lipschitz and $\|L_y\|\le1$. Apply (RCB.2) to the last
Gaussian term and multiply by $\nu$.

For the strict numerical upper bound, the alternating exponential bound
and $e^{-r}>1-r$ yield
$0.96<c<0.9608$, $a_x>0.9992$,
$A<0.962$, $k<0.01924$, $q<0.198$, and
$\sigma<0.199$. Furthermore $e>163/60$ implies $\ell<0.607$,
while $\sqrt6<2.45$ and $\sqrt{120}<11$. Thus
$$
\Gamma_1(tq)<2.45(0.607)+11(0.00396)=1.53071
$$
and
$$
\varepsilon_2<
0.3[4(0.962)(0.607)+2(0.01924)+0.199(1.53071)]
=0.803648187<0.805.
$$
All decimal bounds in this proof are terminating rational inequalities.
The exact diagnostic value is approximately $0.801335$; its decimal
evaluation is not used in the proof.
:::

(sec-rcb-signed)=
## 3. Signed coercivity on the stored velocity arm

:::{prf:lemma} A strict signed drift inequality at $\nu=3/10$
:label: lem-rcb-signed-field

For $\beta\in(0,1/2)$ define
$$
Q_\beta(r,\zeta)=\|r\|_N^2+2\beta\langle r,\zeta\rangle_N
                         +\|\zeta\|_N^2
$$
and the comparison drift field
$\Phi_N(x,v)=(v,-x-v-\nu L_xv)$.
For entering arrays with stored speed at most $V=2$, take $\beta=9/20$.
Then
$$
2\langle (r,\zeta),G_\beta[
       \Phi_N(x,v)-\Phi_N(\widetilde x,\widetilde v)]\rangle_N
\le-\frac2{29}Q_\beta(r,\zeta),
\tag{RCB.4}
$$
where $r=x-\widetilde x$, $\zeta=v-\widetilde v$ and
$G_\beta=\left(\begin{smallmatrix}1&\beta\\\beta&1\end{smallmatrix}\right)
\otimes I_d$.
This is an inequality for the displayed drift field on the stated
stored-speed domain; it is not a discretization theorem for (RCB.1).
:::

:::{prf:proof}
Let $L=L_x$ and $f=-\nu(L_x-L_{\widetilde x})\widetilde v$.
Then $\|f\|_N\le\varepsilon\|r\|_N$, with
$\varepsilon=2\nu V\ell$. Exact expansion gives
$$
\begin{aligned}
2\langle(r,\zeta),G_\beta(\zeta,-r-\zeta-\nu L\zeta+f)\rangle_N
={}&-2\beta\|r\|_N^2+(2\beta-2)\|\zeta\|_N^2
       -2\beta\langle r,\zeta\rangle_N\\
&-2\nu[\langle\zeta,L\zeta\rangle_N
                  +\beta\langle r,L\zeta\rangle_N]
 +2\langle\beta r+\zeta,f\rangle_N.
\end{aligned}
$$
Young's inequality bounds the last term by
$\|\beta r+\zeta\|_N^2+\varepsilon^2\|r\|_N^2$ and cancels
the displayed scalar cross term. Keep the sign of the alignment term:
$$
-2\nu\langle\zeta,L\zeta\rangle_N
-2\nu\beta\langle r,L\zeta\rangle_N
=-2\nu\|L^{1/2}(\zeta+\beta r/2)\|_N^2
 +\frac{\nu\beta^2}{2}\langle r,Lr\rangle_N
\le\frac{\nu\beta^2}{2}\|r\|_N^2.
$$
The two loss coefficients are consequently
$$
\kappa_r=2\beta-\beta^2-\varepsilon^2-\nu\beta^2/2,\qquad
\kappa_v=1-2\beta.
$$
Here $\varepsilon^2=36/(25e)<27/50$, since $e>8/3$.
At $\beta=9/20$,
$$
\kappa_r>
0.9-0.2025-0.54-0.030375
=0.127125>\frac1{10},\qquad \kappa_v=\frac1{10}.
$$
Finally $Q_\beta\le(1+\beta)(\|r\|_N^2+\|\zeta\|_N^2)$ and
$(1/10)/(1+\beta)=2/29$. This proves (RCB.4).
:::

:::{prf:corollary} Signed coercivity of the actual correlated stage defect
:label: cor-rcb-stage-signed

At the shared-OU stage in {prf:ref}`cor-rcb-reference-second-defect`,
put $\zeta=w-\widetilde w$; both $\zeta$ and $R=y-\widetilde y$
are deterministic conditional on the entering arrays. For the same
comparison field applied to $(y,w)$, take $\beta=12/25$. Then
$$
2\mathbb E\langle(R,\zeta),G_\beta[
 \Phi_N(y,w)-\Phi_N(\widetilde y,\widetilde w)]\rangle_N
\le-\frac1{37}Q_\beta(R,\zeta).
\tag{RCB.5}
$$
No bounded speed assumption on the noisy $w$ is added.
:::

:::{prf:proof}
Repeat the preceding expansion, now using the RMS defect bound
$\varepsilon_2<0.805$ in (RCB.3) after taking expectation.
The alignment form is nonnegative for each realization and
$\mathbb E\langle R,L_yR\rangle_N\le\|R\|_N^2$.
The positional loss coefficient is strictly greater than
$0.96-0.2304-0.648025-0.03456=0.047015>0.04$;
the velocity coefficient is $0.04$. Since $1+\beta=1.48$,
$0.04/(1+\beta)=1/37$. The artificial $-w$ in this comparison
field is not an extra operation in the actual second kick. Accordingly
(RCB.5) records a signed constraint on that stage's defect; integrating
it as though it were a continuous canonical update is not licensed.
:::

(sec-rcb-energy)=
## 4. Dissipation and invariant existence for the actual finite kernel

:::{prf:theorem} Exact count-alignment dissipation at the reference viscosity
:label: thm-rcb-finite-dissipation

Let $\mathcal E_m(S)=m\|x\|_N^2+\|v\|_N^2$,
$d_O=m(1-c^2)$ and $a_\nu=2-t\nu>0$.
For every finite entering array in the kinetic arm,
$$
\begin{aligned}
&\mathbb E\mathcal E_m(S^+)
 +\mathbb E[\|z\|_N^2-\|C_V(z)\|_N^2]
 +t\nu a_\nu\langle v,L_xv\rangle_N
 +d_O\|u\|_N^2\\
&\qquad+\frac{t\nu a_\nu}{2}
             \mathbb E\langle w,L_yw\rangle_N\\
&\le\mathcal E_m(S)+md(q^2+s^2)
                      +\frac{2t^3\nu}{a_\nu e}.
\end{aligned}
\tag{RCB.6}
$$
In particular both count kicks contribute genuine nonnegative
dissipation on the left. The right noise budget is independent of $N$.
:::

:::{prf:proof}
Put $z_0=w-ty$, so $z=z_0-t\nu L_yw$.
The uncapped harmonic/OU identity, evaluated with the actual
first-kick velocity $v_A$, is
$$
\mathbb E[m\|y\|_N^2+\|z_0\|_N^2]
=\mathcal E_m(S)-[\|v\|_N^2-\|v_A\|_N^2]
                    -d_O\|u\|_N^2+mdq^2.
$$
For completeness, its deterministic terms follow by inserting
$y=x+b(v_A-tx)$ and $z_0=c(v_A-tx)-ty$; their sum equals
$m\|x\|_N^2+\|v_A\|_N^2-d_O\|v_A-tx\|_N^2$.
The shared OU contribution is
$d q^2[mt^2+m^2]=mdq^2$.

Since $0\le L_x\le I$,
$$
\|v\|_N^2-\|A_xv\|_N^2
=2t\nu\langle v,L_xv\rangle_N
  -t^2\nu^2\|L_xv\|_N^2
\ge t\nu a_\nu\langle v,L_xv\rangle_N.
$$
For each realization of the actual noisy graph,
$$
\begin{aligned}
\|z\|_N^2-\|z_0\|_N^2
&\le-t\nu a_\nu\langle w,L_yw\rangle_N
            +2t^2\nu\langle y,L_yw\rangle_N\\
&\le-\frac{t\nu a_\nu}{2}\langle w,L_yw\rangle_N
            +\frac{2t^3\nu}{a_\nu}\langle y,L_yy\rangle_N.
\end{aligned}
$$
The second inequality is Young's inequality in the $L_y^{1/2}$
inner product. The pair formula and
$\sup_r |r|^2e^{-|r|^2/2}=2/e$ give
$$
\langle y,L_yy\rangle_N
=\frac1{2N^2}\sum_{i,j}K(y_i-y_j)|y_i-y_j|^2\le\frac1e.
$$
Subtract the exact nonnegative cap loss. The independent final
position Gaussian adds $mds^2$ to the modified energy.
Combining these statements proves (RCB.6), without a conditional
factorization at the noisy second kick.
:::

:::{prf:corollary} Actual finite invariants with a uniform fourth-moment budget
:label: cor-rcb-finite-invariant

The kinetic kernel at $\nu=3/10$ has an exchangeable invariant
probability $\Pi_N$ on $(\mathbb R^d\times\overline B_V)^N$
for every $N$. Put
$$
r_4=\frac{1+a_x^4}{2},\qquad
e_4=(r_4/a_x^4)^{1/3}-1,\qquad
D_4=bV+\tau[d(d+2)]^{1/4},\qquad
B_4=(1+e_4^{-1})^3D_4^4.
$$
Then an invariant can be chosen with, for every slot $i$,
$$
\int|x_i|^4\,d\Pi_N\le\frac{B_4}{1-r_4}.
\tag{RCB.7}
$$
At that invariant, the stationary expectation of the four nonnegative
loss terms in (RCB.6) is at most
$md(q^2+s^2)+2t^3\nu/(a_\nu e)$.
Neither uniqueness nor a convergence rate is asserted here.
:::

:::{prf:proof}
The actual position identity is
$x^+=a_xx+bA_xv+tq\xi+s\chi$.
The count average gives $|(A_xv)_i|\le V$.
Minkowski bounds the $L^4$ norm of its last three terms by $D_4$.
The scalar Young inequality
$(A+B)^4\le(1+e)^3A^4+(1+e^{-1})^3B^4$ therefore yields
$$
\mathbb E\left[\frac1N\sum_i|x_i^+|^4\right]
\le r_4\frac1N\sum_i|x_i|^4+B_4.
$$
Here $0<a_x<1$, so $e_4>0$ and $r_4<1$.
Start from the zero array. Its Cesaro occupation measures have averaged
fourth moment at most $B_4/(1-r_4)$.
For a fixed $N$ this gives tightness of the full position array;
the velocity ball is compact. The actual kinetic kernel is Feller:
for fixed innovations all finite count matrices, affine operations and
the native cap are continuous in the input, and dominated convergence
applies to bounded continuous tests. A weakly convergent occupation
subsequence therefore has an invariant limit. Truncation and lower
semicontinuity preserve the averaged fourth-moment bound.

The kernel is equivariant under simultaneous slot permutations.
Symmetrizing the invariant over the finite permutation group preserves
invariance and the averaged bound, and makes each slot moment equal
to that average. This proves (RCB.7).
Its second moment is finite. Integrate (RCB.6) against this invariant,
cancel its two modified-energy expectations and retain the nonnegative
loss terms to obtain the stationary budget.
:::

(sec-rcb-frozen)=
## 5. An explicit active frozen-provider law gap at $\nu=3/10$

:::{prf:definition} Positive active register and uniform provider envelopes
:label: def-rcb-active-register

Use the conservative current-frame population preparation of
{prf:ref}`def-pvb-record`, with any fixed bounded $C^1$ reward,
positive feature/role floors, diversity smoothing, regularized reward
and diversity standard deviations, and recipient noise $\sigma_J>0$.
Keep the actual global normalizers. For the two logistic factors fix
$f_b,A_b,\bar p_b>0$, $b=r,s$, and use the positive exponents
$p_b=\theta\bar p_b$. For fixed $s_c,\epsilon_c,\kappa_C>0$ put
$$
M=\sum_b\bar p_b\max\{|\log f_b|,|\log(f_b+A_b)|\},\quad
\Delta=\sum_b\bar p_b\log[(f_b+A_b)/f_b],
$$
$$
D_0=e^{-M}+\epsilon_c,\qquad
a_0=\frac{e^M\Delta}{s_cD_0},\qquad c_0=a_0/\kappa_C.
$$
The actual gate and incoming accepted donor density obey
$a_*\le\theta a_0$ and $c_*\le\theta c_0$, including equal-fitness
states. Set $V_c=4$ and
$$
r_8=(1+a_x^8)/2,\quad e_8=(r_8/a_x^8)^{1/7}-1,\quad
D_8=bV_c+\sqrt{a_x^2\sigma_J^2+\tau^2}\,G_8^{1/8},
$$
$$
B_8=(1+e_8^{-1})^7D_8^8,\qquad H_8=2B_8/(1-r_8),
\quad H_4=\sqrt{H_8},
$$
where $G_p=\mathbb E|Z|^p$ for a standard $d$-Gaussian.
The explicit positive interval and envelopes are
$$
\theta_0=\min\left\{1,\frac{\kappa_C}{8a_0},
             \frac{\kappa_C(1-r_8)}{2r_8a_0}\right\},\quad
\bar a=\theta_0a_0,\quad\bar c=\theta_0c_0.
\tag{RCB.8}
$$
For $0<\theta\le\theta_0$, $2c_*<1$,
the eighth-moment class $\mu|x|^8\le H_8$ is preserved, and the
actual joint second-stage provider has absolute velocity moment at most
$$
M_w=cV_c+ct[(1+\bar c)H_8^{1/8}+\sigma_JG_1]+qG_1.
\tag{RCB.9}
$$
The same eighth-moment drift holds for the average of every actual
finite swarm.
:::

:::{prf:proof}
The positive-base register gives
$e^{-M}\le F\le e^M$ and
$F^*-F_*\le\theta e^M\Delta$ for $\theta\le1$.
The clipped canonical gate is therefore bounded by $\theta a_0$;
the role floor gives incoming accepted density at most $\theta c_0$.
This does not divide by a realized reward or diversity variance.
The source moment calculation and Young inequality in
{prf:ref}`lem-pvb-eighth-moment` give
$M_8^+\le r_8(1+c_*)M_8+B_8$.
The last endpoint in (RCB.8) makes the coefficient at most
$(1+r_8)/2$ and proves the invariant sublevel.

The average source first moment is at most
$(1+c_*)\mu|x|+\sigma_JG_1$.
Collision and the first count average give the uniform prepared
velocity bound $V_c$ independently of all jitters.
Thus $w=c(U-tX)+q\xi$ satisfies (RCB.9) by the triangle inequality.
No factorization of the joint $(y,w)$ provider is introduced.
:::

:::{prf:theorem} Complete frozen whole-update contraction at the reference viscosity
:label: thm-rcb-frozen-gap

Freeze any $\mu$ in the class (RCB.8), its actual rooted preparation
kernel $J_\mu$ and both actual deterministic count providers. Let
$P_\mu$ be the resulting full root kernel on
$E=\mathbb R^d\times\overline B_V$.
It includes active preparation, both kicks, their joint OU stage,
the native cap and the final independent position Gaussian.
For every $0<\theta\le\theta_0$ and $\nu=3/10$, define
$$
r_4=(1+a_x^4)/2,\quad e_4=(r_4/a_x^4)^{1/3}-1,\quad
B_4=(1+e_4^{-1})^3
 [bV_c+\sqrt{a_x^2\sigma_J^2+\tau^2}G_4^{1/4}]^4,
$$
$$
B=1-r_4+r_4\bar cH_4+B_4,\quad
R=2+\frac{4B}{1-r_4},\quad R_x=(R-1)^{1/4},
$$
$$
R_1=mR_x+tV_c,\quad M_v=cV_c+ctR_x,\quad
Q=\frac{1+tR_1+t\nu M_w}{m-t\nu},
$$
$$
L_z=m+t\nu+t^2\nu\ell(Q+M_w),\quad
k_v=(2\pi q^2)^{-d/2}e^{-(Q+M_v)^2/(2q^2)}L_z^{-d},
$$
$$
k_x=(2\pi s^2)^{-d/2}e^{-(1+R_1+tQ)^2/(2s^2)},\quad
\epsilon=(1-\bar a)e^{-\bar c}v_d(1)^2k_vk_x>0,
$$
$$
\beta_H=\frac{\epsilon}{r_4R+2B},\qquad
w_H=1+\beta_H(1+|x|^4),\qquad
q_H=\max\left\{
\frac{2+\beta_H(r_4R+2B)}{2+\beta_H R},
1-\epsilon/2\right\}<1.
\tag{RCB.10}
$$
Then every zero-mass signed measure $\xi$ with finite $w_H$ norm obeys
$$
\|\xi P_\mu\|_{w_H}\le q_H\|\xi\|_{w_H}.
\tag{RCB.11}
$$
The constants are primitive, finite and uniform over the declared
population providers and positive exponent interval. In particular no
small-viscosity endpoint is imposed beyond $t\nu<m$, which the
reference $\nu=3/10$ satisfies.

For each fixed $P_\mu$ there is a unique invariant probability
$\rho_\mu$ of finite fourth moment, and
$$
\|\lambda P_\mu^n-\rho_\mu\|_{w_H}
\le q_H^n\|\lambda-\rho_\mu\|_{w_H},\qquad
\rho_\mu(1+|x|^4)\le B/(1-r_4).
\tag{RCB.12}
$$
For any fixed sequence of such actual frozen provider kernels,
$$
\|\lambda P_{\mu_0}\cdots P_{\mu_{n-1}}
       -\widetilde\lambda P_{\mu_0}\cdots P_{\mu_{n-1}}\|_{w_H}
\le q_H^n\|\lambda-\widetilde\lambda\|_{w_H}.
\tag{RCB.13}
$$
The compared laws in (RCB.13) share the same provider sequence.
:::

:::{prf:proof}
All moment and gate envelopes have already been checked. At the
reference parameters $t\nu=0.006<m=0.9996$ and $t\nu<1$.
The conditional source fourth moment at a root $z=(x,v)$ is at
most $|x|^4+c_*H_4$; donor position copying and recipient jitter
are actual operations. The bounded $V_c$ term may depend on that
jitter. Minkowski still bounds it before applying the Gaussian/Young
calculation, giving
$P_\mu W_4(z)\le r_4W_4(z)+B$, with $W_4=1+|x|^4$.

For completeness, the minorization uses the actual event that the root
has no outgoing accepted edge and no incoming accepted edge. Its
conditional probability is at least
$(1-\bar a)e^{-\bar c}$; its preparation then preserves the root.
At fixed input in $|x|\le R_x$, the OU velocity has Gaussian
mean norm at most $M_v$ and variance $q^2I_d$.
Its pre-cap second-kick map is
$$
Z(w)=[m-t\nu a_2(x_1+tw)]w-tx_1+t\nu m_2(x_1+tw).
$$
The actual second fields satisfy $0\le a_2\le1$ and
$|m_2|\le M_w$. Thus
$|Z(w)|\ge(m-t\nu)|w|-tR_1-t\nu M_w$.
It is proper, has degree one by homotopy to $mw-tx_1$, and
every preimage of the unit target ball lies in $B_Q$.
The Gaussian-kernel derivative bound gives $\|DZ\|\le L_z$ there.
The regular-value area formula, including the nonnegative singular
part, consequently gives a pre-cap velocity density at least $k_v$
on $B_1$. The independent final position Gaussian has mean bounded
by $R_1+tQ$ at every such preimage and density at least $k_x$ on
$B_1$. Pushing this lower measure through the actual cap gives the
common probability $\epsilon$ in (RCB.10), as in the complete
degree/area argument of {prf:ref}`lem-pvb-frozen-minorization`.

Since $R>2B/(1-r_4)$, the ratio
$[2+\beta_H(r_4u+2B)]/[2+\beta_Hu]$ decreases for $u\ge R$
and is strictly below one at $R$.
For two distinct inputs use ground cost
$2+\beta_H[W_4(z)+W_4(z')]$. Above this threshold the drift gives
the first contraction ratio in (RCB.10).
Below it use the common minorization coupling; its expected cost is
at most $2(1-\epsilon)+\beta_H(r_4R+2B)$, giving ratio at most
$1-\epsilon/2$. Identical inputs use the identical kernel draw.
The optimal cost for this ground metric is the full weighted variation:
match the common measure identically and charge each unmatched endpoint
its weight $w_H$. Integrating the kernel coupling and then scaling
the positive/negative parts proves (RCB.11).

Probabilities of finite $w_H$ norm form a complete closed subset of
the weighted finite-variation space. The drift makes $P_\mu$ map
that subset to itself, and (RCB.11) is a strict contraction.
Equivalently, successive iterates form a summable Cauchy sequence by
the contraction, their weighted limit is a probability, and passing
the bounded weighted operator through that limit yields invariance.
Strict contraction proves uniqueness and (RCB.12); integration of
the drift gives its moment bound. Iterating (RCB.11) with a prescribed
common sequence gives (RCB.13).
:::

:::{prf:corollary} Active finite invariant existence at the reference viscosity
:label: cor-rcb-active-finite-invariant

With the same active register and $0<\theta\le\theta_0$, the actual
finite conservative gas at $\nu=3/10$ has an exchangeable invariant
probability $\widehat\Pi_N$ for every $N$, with
$\widehat\Pi_N|x_i|^8\le H_8$ for each slot.
This is an invariant existence assertion for the actual finite-array
kernel, rather than an application of the frozen root contraction.
:::

:::{prf:proof}
The finite incoming-column bound gives the same drift
$M_8^+\le[(1+r_8)/2]M_8+B_8$ for the averaged positional moment.
The component collision uses original frozen velocities, and its
uniform speed bound $V_c$ is independent of the sampled forest and
jitters. The bounded first count average therefore justifies the
moment calculation without independent rows at the second kick.
Starting from the zero array gives uniformly bounded Cesaro moments.

For fixed finite $N$ the preparation is a finite mixture over the
measurement, donor and accepted-edge tokens. Their probabilities are
continuous because all role floors and variance regularizers are
strictly positive, the reward/features are continuous, and the clipped
gate is continuous. For each fixed discrete forest its Haar rotation
and Gaussian jitter integrations are Feller. Combining this with the
actual kinetic Feller property proves that the full finite kernel is
Feller. The tightness and symmetrization argument of
{prf:ref}`cor-rcb-finite-invariant` applies at moment order eight.
The invariant averaged moment is at most
$B_8/[1-(1+r_8)/2]=H_8$, and exchangeability converts it into the
stated individual slot bounds.
:::

(sec-rcb-source-box)=
## 6. A source-box frozen gap at the default killing boundary

:::{prf:theorem} Frozen default-box Doeblin contraction through every source outcome
:label: thm-rcb-source-box-frozen-gap

Let $D=(-L,L)^d$, $L=2$, and retain the actual mandatory revival
rule: every dead root copies an alive donor position; every alive root
retains its position or copies an alive donor. Assume nonzero alive
provider mass, and an actual well-defined marked rooted preparation
kernel with the native component collision bound $V_c=4$.
Freeze its providers, including the actual joint second-stage law.
No acceptance-smallness hypothesis is used in this theorem beyond any
condition required to define that population rooted kernel.

At every root outcome let $X=x_{\rm src}+I J$, where
$x_{\rm src}\in D$, $I\in\{0,1\}$ and the recipient jitter
$J\sim N(0,\sigma_J^2I_d)$ is independent of the discrete source
tokens. Keep the actual original-velocity collision, both count kicks,
all OU/final Gaussians and the smooth cap at the reference parameters.
The stored dead position may be arbitrarily far from $D$.
Define
$$
p_J=\Pr\{|\sigma_J Z|\le1\}>0,\qquad
R_X=\sqrt d\,L+1,\qquad R_1=mR_X+tV_c,\qquad
M_v=cV_c+ctR_X,
$$
$$
M_w^{D}=cV_c+ct(\sqrt d\,L+\sigma_JG_1)+qG_1,\qquad
Q_D=\frac{1+tR_1+t\nu M_w^{D}}{m-t\nu},
$$
$$
L_D^{Z}=m+t\nu+t^2\nu\ell(Q_D+M_w^{D}),\qquad
k_v^D=(2\pi q^2)^{-d/2}
 e^{-(Q_D+M_v)^2/(2q^2)}(L_D^Z)^{-d},
$$
$$
k_x^D=(2\pi s^2)^{-d/2}
 e^{-(1+R_1+tQ_D)^2/(2s^2)},\qquad
\epsilon_D=p_Jv_d(1)^2 k_v^Dk_x^D>0.
\tag{RCB.14}
$$
If $P_\mu^D$ denotes this actual frozen full root kernel with its
terminal mark $1_D(x^+)$, then for every consistent alive or dead
entering root
$$
P_\mu^D(z,\cdot)\ge\epsilon_D\vartheta_D,\qquad
\|\xi P_\mu^D\|_{\rm TV}
\le(1-\epsilon_D)\|\xi\|_{\rm TV}
\quad\text{when }\xi(E)=0.
\tag{RCB.15}
$$
Here full variation is used, and $\vartheta_D$ is uniform position
on $B_1$ times the cap-pushforward of uniform velocity on $B_1$,
with the alive mark. In particular the root-kernel floor holds at
$h=.04$, $\nu=.3$, $L=2$, and is uniform over all such frozen
providers. A fixed $P_\mu^D$ has a unique invariant probability and
converges to it in full variation at rate $(1-\epsilon_D)^n$.
The same rate holds for any common prescribed provider sequence.
:::

:::{prf:proof}
Every source lies in $D$, including sources selected for mandatory
revivals. Conditional on the discrete source/component outcomes,
$\Pr\{|I J|\le1\}\ge p_J$ and on this event $|X|\le R_X$.
The prepared velocity may depend on the component and other random
variables; its bound $V_c$ is uniform. The first deterministic
population count average is convex because $t\nu<1$. Thus
$|U|\le V_c$, $|x_1|\le R_1$, and the root OU mean is bounded
by $M_v$. These estimates are pointwise on the good-jitter event.
They do not require an isolated component or a persistent root.

Across the entire actual prepared provider,
$\mathbb E|X|\le\sqrt d L+\sigma_JG_1$, regardless of its entering
dead positions and its acceptance probabilities.
Consequently its actual joint stage law satisfies
$\Lambda_2|w|\le M_w^D$. Apply the proper-degree/area argument in
the proof of {prf:ref}`thm-rcb-frozen-gap` conditional on each
prepared phase on the good-jitter event. The same preimage bound
$Q_D$ and Jacobian bound $L_D^Z$ hold for every such phase.
The same $k_v^Dk_x^D$ joint density lower bound therefore holds on
the two unit balls. Average over the actual prepared law and multiply
by $p_J$. The common native cap and terminal mark are common
pushforwards, so preserve the lower measure inequality. Since
$B_1\subset D$, the reference output has the alive mark.

Write $P_\mu^D=\epsilon_D\vartheta_D+
(1-\epsilon_D)\widetilde P_\mu^D$ with a probability residual kernel.
For a zero-mass signed measure the common part cancels. Markov kernels
contract full variation, giving (RCB.15).
Probabilities are complete in total variation and the kernel is a
strict contraction on that space; Cauchy iteration gives its unique
invariant and the asserted rate. Iteration with any common sequence
of provider kernels gives the same bound. The argument compares each
root's own full frozen kernel. It never conditions a finite-array
second provider on its own OU innovations.
:::

(sec-rcb-interface)=
## 7. The remaining own-provider block interface

:::{prf:remark} Last justified implication and first missing transport
:label: rem-rcb-first-missing-interface

The proved implication is
$$
\text{actual reference parameters + fixed admissible providers}
\quad\Longrightarrow\quad
\text{(RCB.11)--(RCB.13), with }q_H<1.
$$
It does not imply
$\|\mathcal F_{\nu,\theta}\mu-\mathcal F_{\nu,\theta}\mu'\|
\le q_H\|\mu-\mu'\|$, because the second term changes its own
preparation and both of its own providers. The absolute weighted-BV
feedback proof in research record 19 bounds that additional term by a
constant proportional to $\nu$. Its completed endpoint does not
certify $\nu=3/10$. Finite-swarm empirical second providers cannot
be frozen conditional on their own OU array while retaining the
independent Gaussian root law used by the population minorization.

The new signed constraints (RCB.4)--(RCB.5) and actual energy balance
(RCB.6) retain useful alignment cancellation at the reference value.
They still need a transport estimate for a complete discrete block.
In particular ordinary Euclidean nonexpansion of the cap cannot be
used as signed-metric nonexpansion. For any $\beta>0$, take a nonzero
position difference $r$, pre-cap velocity difference $\zeta=-\beta r$,
and let the common velocity mean tend to infinity. The native cap
velocity difference tends to zero. The signed input cost is
$(1-\beta^2)|r|^2$, whereas the limiting output cost is $|r|^2$.
This proves only that this proposed intermediate metric inference
fails; it proves no obstruction to optimal law transport or to alive
swarm convergence.

The next precise obligation is therefore a cap-compatible block
estimate at $\nu=3/10$ that preserves the negative alignment forms,
charges the spatial graph defect using stored $V=2$ on untouched
rows, and charges the rare preparation-affected rows separately.
It must retain the joint noisy second graph and compare each law with
its own providers. Its block gain must exceed the actual active
preparation/provider error. No assumed positive pressure or realized
fitness-variance floor can substitute for this inequality.

Evidence status: (RCB.2)--(RCB.15) and the finite invariant results
are proved in the supplied arguments and independently reviewed by
the repair agent at the first complete source-box draft, SHA-256
`66469cc6600f481df6b0d402bf3860aa3c572f2568f49b32594ba6af602a8e32`.
The signed-field arm is a
productive reduction to the named discrete transport interface.
The full own-provider reference block is a bounded residual, rather
than closure or an impossibility theorem. The default survival floor
and recent-window conditioning estimates of research record 29 are
separate accepted inputs; they neither provide nor consume this
conservative block gap.
:::
