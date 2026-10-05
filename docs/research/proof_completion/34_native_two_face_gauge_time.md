# The original two-face temporal gauge channel and its complete path action

(sec-ntfg-register)=
## 1. Complete actual update and phase records

:::{prf:definition} Native two-face temporal register
:label: def-ntfg-complete-register

Retain every parameter, original raw-component ray threshold,
archive event/version/generation label, graph rule, Gaussian
stream, boundary mark, eligible mask, numerical convention,
cap and error policy in {prf:ref}`def-nsav-complete-register`.
Use the included $N=2,d=3$, constant quadratic reward and
force $\lambda=0$, $\sigma_x=0$, end-of-step boundary,
current unique nonself count-one companion regime of
{prf:ref}`def-nsav-two-face-regime`, with fixed
$x_1^*=e_1$, $x_2^*=(e_1+e_2)/\sqrt2$.
The domain has both points strictly in its interior.
It uses the ORIGINAL positive coupling condition

$$
t\nu=1\quad(\mathrm{row}),\qquad
t\nu K_0/2=1\quad(\mathrm{count}),\qquad
K_0=e^{-(2-\sqrt2)/(2\rho^2)},\quad t=h/2.
\tag{NTFG.1}
$$

Keep $h,\rho,\nu$, friction $\gamma\ge0$ and cap
fixed and let its original thermostat amplitude $q>0$
tend to zero as in (NSAV.12). Write $c=e^{-\gamma h}$.
The consumed thermostat is the existing Rust independent
isotropic constant diffusion-factor tag $b_O$, with
BAOAB friction validated at every $\gamma\ge0$.
Its O code uses $q=b_O\sqrt h$ at $\gamma=0$,
so the positive-noise $c=1$ endpoint below is an
included configuration of THIS tag. A thermostat
tag imposing $q^2=(1-c^2)/\beta_{\mathrm{eff}}$ retains
that constraint and has $q=0$ at $c=1$ for finite
positive $\beta_{\mathrm{eff}}$; it is not assigned
the positive-noise endpoint by this name.
Allowed entering velocities are $q w_0$ for a specified
finite deterministic six-vector $w_0$; the original
zero-velocity case is $w_0=0$. This specifies the
initial-law parameter, rather than averaging hidden
initial velocities into a new force rule.

At every actual update with both walkers alive, the
constant reward is equal and their symmetric unique
nonself distance gives equal diversity, even after
their positions or velocities have moved. Their
well-defined common fitness transforms give equal
fitness, so both actual living clone gates are zero.
The collision components are singletons with zero
relative velocity. Their configured jitter, restitution
and Haar marks remain in the record but do not change
this all-alive branch. Actual revival/extinction and
outside-domain paths retain their original operations
and masks; they are not removed from its law.

Let $X_{i,n}^q,u_{i,n}^q$ be the actual final state
after $n$ complete updates, $w_{i,n}^q=u_{i,n}^q/q$,
and $G_n=(G_{1,n},G_{2,n})$ its ORIGINAL independent
standard Gaussian O sources. Each actual archive
interaction face at update $n$ has vertices
$(\mathrm{pre}\ i,n,\mathrm{final}\ i,n,
  \mathrm{pre}\ j,n)$.
Use its original principal phase and availability
indicator in the explicitly declared masked coordinate
$\eta_{i,n}^q=a_{i,n}^q\Theta_{i,n}^q/q$.
The omitted-face value of the literal report is unchanged.
Let $S(w_1,w_2)=(w_2,w_1)$ and retain
$L_1=-e_2$, $L_2=(-e_1+e_2)/\sqrt2$.
The color calibration $\kappa$ is not consumed by
this ray instrument. A graph/covariance provider is
passive and does not replace either dense viscous kick.
This is explicitly the canonical real-coordinate DENSE
algorithm tag bound to Rust `QftExecutionConfig.viscosity`
and `recorded_force`, with its actual unregularized
nonself row mass. The Python $N=2,d=3$ empty CSR fallback
and its mass-floor variant retain their different update.
Finite-arithmetic Gaussian underflow/zero-mass branches
and rounded kicks keep their original comparison errors;
they are not assigned the global exact real-coordinate swap.
:::

(sec-ntfg-both-kick-limit)=
## 2. Both actual kicks and the complete finite-window phase process

:::{prf:theorem} Native repeated-update Gaussian limit
:label: thm-ntfg-finite-window-process

For every fixed finite $K$, the actual all-alive
probability through its first $K$ updates tends to
one. Jointly over these updates,
$X_n^q\to x^*$ and $w_n^q\to w_n$ in probability
under the original coupled source representation,
where the complete Gaussian state satisfies

$$
w_n=cw_{n-1}+SG_n,
\qquad
w_n=c^nw_0+\sum_{l=1}^nc^{n-l}SG_l.
\tag{NTFG.2}
$$

The original two-face phase coordinates converge
jointly, with every fixed polynomial moment, to

$$
\eta_{i,n}=L_i\cdot(w_{i,n}-w_{i,n-1}).
\tag{NTFG.3}
$$

The donor's own pre velocity contributes to each
individual edge and cancels in this actual triangle
sum; it is not omitted from the original edge map.
Every original measured defect satisfies jointly
$a_{i,n}^q W_{i,n}^q/q^2\to\eta_{i,n}^2/2$
with all fixed moments.

Relative to the complete limiting state filtration,
the exact native discrete conditional drift and
centered bracket per update are

$$
E[\Delta w_n\mid w_{n-1}]=(c-1)w_{n-1},
\quad E[M_nM_n^T\mid w_{n-1}]=I_6,
\quad M_n=\Delta w_n-(c-1)w_{n-1}=SG_n.
\tag{NTFG.4}
$$

The projected phase centered innovations are
$\epsilon_{i,n}=L_i\cdot(SG_n)_i$, independent
standard normals in both $i$ and $n$. Thus
$E[\eta_{i,n}\mid w_{n-1}]=(c-1)L_i\cdot w_{i,n-1}$
and their centered two-channel bracket is $I_2$
per update, or $I_2/(t_*h)$ per configured physical
clock interval $t_*h>0$. This is its derived discrete
Gaussian process, not an additional continuous-time
or continuum Yang--Mills limit.
:::

:::{prf:proof}
On the all-alive branch the first nonself row kick
is exactly $u\mapsto Su$ at $t\nu=1$.
For count normalization write its kick as
$B(K)u$, with diagonal $1-t\nu K/2$ and
off-diagonal $t\nu K/2$ in particle space.
At $K_0$, (NTFG.1) gives $B(K_0)=S$.
The Gaussian kernel is globally Lipschitz and the
fixed source-core positions move by $O_K(q)$;
therefore its actual B1 and separately reevaluated
B2 matrices are $S+O_K(q)$.
The original O step is $z=cB_1u+qG_n$.
The second kick gives $B_2z$, followed by the actual
cap. Consequently on each fixed compact source core

$$
u_n^q/q=cw_{n-1}^q+SG_n+O_K(q),
\quad
X_n^q-X_{n-1}^q
 =qt[(1+c)Sw_{n-1}^q+G_n]+O_K(q^2).
\tag{NTFG.5}
$$

In row normalization both swap identities are exact
before the cap; the finite cap has only its actual
$O(q^2|w|^2/V)$ velocity error. Induction on the
fixed $K$ proves the displayed state convergence,
with all original positions remaining inside the
strict domain neighborhood on that core. No living
gate is opened there. Increasing the original
Gaussian source core proves convergence in probability
for the full actual law.

For the phase calculation keep all THREE native
rays. At zero imaginary parts their three real
overlaps remain positive near the base points,
so the loop phase is identically zero for all
these real position perturbations. Its linear
imaginary-velocity terms at the base positions are

$$
\frac{x_i^*\cdot(w_{i,n}-w_{i,n-1})}{|x_i^*|^2}
+\frac{x_i^*\cdot w_{j,n-1}-x_j^*\cdot w_{i,n}}
       {x_i^*\cdot x_j^*}
+\frac{x_j^*\cdot w_{i,n-1}-x_i^*\cdot w_{j,n-1}}
       {x_i^*\cdot x_j^*}.
$$

They sum to precisely (NTFG.3). The old donor
terms cancel between its actual two incident
overlaps. This is the ordered interaction triangle,
not a triangle of new spatial sample points.
Taylor expansion of its original smooth phase
function on the strict ray/overlap neighborhood
gives joint convergence on each source core.

For moment control, use the original event
$\max_{n\le K}|G_n|\le q^{-1/8}$.
Deterministic finite-step kick bounds and
$|C_V(w)|\le|w|$ show all-alive scaled velocities
are at most $C_K(1+|w_0|+\sum_{n\le K}|G_n|)$.
Both positions and ray velocities stay within the
same strict neighborhood for small $q$; count
kick errors are bounded Gaussian polynomials times
$q$. The phase remainder divided by $q$ has a
vanishing fixed polynomial-moment bound on this
event. Its complement has Gaussian probability
at most $C_Ke^{-c_Kq^{-1/4}}$.
The actual available phase is bounded by $\pi$
on EVERY branch, and its declared masked coordinate
is zero when missing. Thus even actual revival,
later clone/jitter or extinction outcomes on this
exceptional event give a scaled-phase moment at
most $(\pi/q)^p C_Ke^{-c_Kq^{-1/4}}\to0$.
No rare algorithm branch or unbounded innovation
has been deleted. Cosine's fourth-order remainder
proves the defect moments by the same estimates.

The affine recursion gives (NTFG.2) and its
conditional Gaussian innovation $SG_n$.
The swap is orthogonal, so its covariance is $I_6$.
The two projected innovations consume different
original rows and each $L_i$ has norm one; they
are therefore independent standard normals.
Conditional expectation and the actual clock
normalization prove the drift/bracket assertions.
:::

(sec-ntfg-native-path-action)=
## 3. The native complete phase-path action and original source

:::{prf:theorem} Full finite-window native phase likelihood and measured-action comparison
:label: thm-ntfg-path-action

Put $v_{i,n}=L_i\cdot w_{i,n}$ and retain the specified
initial projection $v_{i,0}=L_i\cdot w_{i,0}$. The two
original limiting channels obey
$v_{i,n}=cv_{i,n-1}+\epsilon_{i,n}$ and
$\eta_{i,n}=v_{i,n}-v_{i,n-1}$.
Let $T_K$ be the $K\times K$ lower triangular matrix
with diagonal one and

$$
(T_K)_{nl}=-(1-c)c^{n-1-l}\quad(l<n),
\quad m_{i,n}=(c-1)c^{n-1}v_{i,0}.
\tag{NTFG.6}
$$

The COMPLETE native $2K$-phase support is full real
coordinate space and has mean $m$, independent channel
covariances $T_KT_K^T$, and the exact Gaussian path action

$$
S_K(\eta\mid v_0)=\frac12\sum_{i=1}^2\sum_{n=1}^K
 \left[\eta_{i,n}+(1-c)
        \left(v_{i,0}+\sum_{l<n}\eta_{i,l}\right)\right]^2
                         +K\log(2\pi).
\tag{NTFG.7}
$$

All past phases in these terms are retained, rather
than declaring the marginal phase channel Markov.
For the ORIGINAL simultaneous source shifts
$G_n\mapsto G_n+\theta f_n$, put
$g_{i,n}=L_i\cdot f_{j,n}$ with $j\ne i$.
Its induced path direction is $V_f=(T_Kg_1,T_Kg_2)$,
its volume divergence is zero, and its exact native score is

$$
j_f(\eta)=\sum_{i,n}g_{i,n}
 \left[\eta_{i,n}+(1-c)
        \left(v_{i,0}+\sum_{l<n}\eta_{i,l}\right)\right],
\qquad \int D O[V_f]e^{-S_K}d\eta
       =\int O j_fe^{-S_K}d\eta.
\tag{NTFG.8}
$$

The ORIGINAL finite-$q$ complete masked phase-path
law and its signed source-score measure converge
in total variation to this path density and score.
Their retained singular/missing mass tends to zero;
the actual absolutely continuous native path action
and score consequently converge in measure on every
compact phase-path coordinate set. The raw unscaled
phase density has its actual $2K\log q$ coordinate
volume correction.

The original sum of its unit face defects, divided
by $q^2$, converges with all fixed moments to
$Q_K(\eta)=\sum_{i,n}\eta_{i,n}^2/2$.
For $c<1$ and $K\ge2$, this differs from the native
path-action energy, even when $v_0=0$.
For $K=1$ it matches up to a constant exactly when
$(1-c)v_0=0$. For the existing zero-friction regime
$\gamma=0$ ($c=1$), it matches for EVERY finite $K$
and every finite specified $w_0$, with the full
native weak source first variation (NTFG.8).
Thus positive one-step matching does not silently
replace the complete temporal likelihood by a sum
of independently fitted face actions.
:::

:::{prf:proof}
Project the actual limiting recursion (NTFG.2).
Its two innovations are the independent unit normals
already derived from the original sources. Solving
the scalar recursion and subtracting its adjacent
values gives $\eta_i=m_i+T_K\epsilon_i$.
The triangular matrix has determinant one, so this
Gaussian law has full support and its complete
Jacobian has no unknown volume correction.
The inverse formula is explicitly
$\epsilon_{i,n}=\eta_{i,n}+(1-c)
(v_{i,0}+\sum_{l<n}\eta_{i,l})$.
Multiplying its original independent normal densities
proves (NTFG.7).

The original stream source changes each innovation
by $\theta g_{i,n}$ and therefore the path by
$\theta T_Kg_i$. The unobserved orthogonal
components of each original three-source row are
independent normals with mean-zero score.
The displayed invertible path reconstructs its
projected innovations, so the coarse conditional
mean of $\sum_n f_n\cdot G_n$ is exactly (NTFG.8).
Differentiating its Gaussian mean shift and integrating
by parts gives the weak identity and zero divergence.

The actual finite-$K$ source-core map from its
$6K$ original Gaussian coordinates to the $2K$
scaled phase path is $C^1$ close to this linear
map plus its deterministic mean. Both kicks,
separately recomputed kernels, the $C^1$ cap and
all three ray derivatives have this convergence
by the finite induction in Section 2. All masks
are one on each fixed core for small $q$.
Decompose the ORIGINAL source coordinates into
their $2K$ independent projected innovations
and the $4K$ orthogonal coordinates.
Augment the computed native phase map by those
orthogonal original coordinates, multiply its
phase part by $T_K^{-1}$ and subtract the known
mean. This is a $C^1$ near-identity injective
change of variables on each fixed convex source
cube. The determinant and inverse-density argument
of {prf:ref}`thm-nsav-face-density-source` therefore
applies in dimension $6K$ without any assumption
about an unknown full output density. Outside
the cube retain the original Gaussian mass and
$E[|\sum_n f_n\cdot G_n|\mathbf1_{\mathrm{outside}}]$;
both tend to zero when the core grows. This proves
the asserted total variation, signed-score and
Lebesgue-sector action/score convergence, with
every exceptional original branch retained.

The defect convergence was proved with moments
in Section 2. If $c<1$, already the first two
terms of (NTFG.7) with $v_0=0$ contain
$(1-c)\eta_{i,1}\eta_{i,2}
+(1-c)^2\eta_{i,1}^2/2$ beyond their literal unit
quadratics. More generally its last two time
coordinates have a nonzero mixed entry for every
$K\ge2$, so the full-support quadratic cannot
equal $Q_K$ up to a constant. For $K=1$ only
the linear shift $(1-c)v_0\cdot\eta_1$ remains,
giving its exact matching condition. At $c=1$
all those memory and initial-shift terms vanish:
the ORIGINAL phase innovations are independent
unit normals for every finite time window and
the native energy is exactly $Q_K$. Its actual
source derivative is the same (NTFG.8).
:::

(sec-ntfg-stationary-gaussian-channel)=
## 4. Stationary phase covariance and the complete additive variance

:::{prf:theorem} Derived stationary local Gaussian phase channel
:label: thm-ntfg-stationary-phase-law

For $0<c<1$ the Gaussian state (NTFG.2) has the
unique invariant law $N(0,(1-c^2)^{-1}I_6)$.
Its original two-face projection in that law is
a centered stationary Gaussian sequence with
independent channels and exact lag covariance

$$
R_0=\frac2{1+c},\qquad
R_k=-\frac{1-c}{1+c}c^{k-1}\quad(k\ge1).
\tag{NTFG.9}
$$

For every finite $K$, its native phase-path action
is

$$
S_K^{\mathrm{stat}}(\eta)
 =\frac12\sum_{i=1}^2\eta_i^T\mathsf R_K^{-1}\eta_i
       +K\log(2\pi)+\log\det\mathsf R_K,
\qquad (\mathsf R_K)_{nl}=R_{|n-l|}.
\tag{NTFG.10}
$$

The covariance is positive definite at every finite
$K$; the hidden initial velocity is integrated
with its DERIVED Gaussian law. Every original
finite source direction has the induced mean
$T_Kg_i$ and native score
$\sum_i(T_Kg_i)^T\mathsf R_K^{-1}\eta_i$.
It is not assigned the zero-initial-velocity action.
For $c<1$ its variance already differs from the
original unit action for $K=1$.

More explicitly, put $a_c=(1-c)/(1+c)$,
$A_K=T_K^{-1}$ and $z_i=A_K\eta_i$, whose actual
coordinates are
$z_{i,n}=\eta_{i,n}+(1-c)\sum_{l<n}\eta_{i,l}$.
The COMPLETE stationary action and source score are

$$
\begin{gathered}
\mathsf R_K=T_K(I_K+a_c\mathbf1\mathbf1^T)T_K^T,
\qquad \det\mathsf R_K=1+Ka_c,\\
S_K^{\mathrm{stat}}(\eta)
 =\frac12\sum_{i=1}^2
  \left[|z_i|^2-\frac{a_c}{1+Ka_c}
                (\mathbf1^Tz_i)^2\right]
  +K\log(2\pi)+\log(1+Ka_c),\\
j_f^{\mathrm{stat}}(\eta)
 =\sum_{i=1}^2
  \left[g_i^Tz_i-\frac{a_c}{1+Ka_c}
       (\mathbf1^Tg_i)(\mathbf1^Tz_i)\right].
\end{gathered}
\tag{NTFG.16}
$$

With the convention
$R_k=(2\pi)^{-1}\int_{-\pi}^{\pi}
 e^{ik\omega}\mathfrak s_c(\omega)\,d\omega$,
its exact phase spectral density is

$$
\mathfrak s_c(\omega)
 =\frac{2-2\cos\omega}{1+c^2-2c\cos\omega},
\qquad
\mathfrak s_c(\omega)
   =\frac{\omega^2}{(1-c)^2}+O(\omega^4)
                         \quad(\omega\to0).
\tag{NTFG.17}
$$

The vanishing temporal zero mode therefore follows
from the ORIGINAL adjacent-state phase, while
every finite native path density remains full rank.

Although its predictable centered innovation bracket
is exactly $I_2$ per step, the COMPLETE additive
variance of each stationary phase channel is zero:

$$
R_0+2\sum_{k\ge1}R_k=0,\qquad
\sum_{n=1}^K\eta_{i,n}=v_{i,K}-v_{i,0},\qquad
\frac1K\operatorname{Var}\left(\sum_{n=1}^K\eta_{i,n}\right)
 =\frac{2(1-c^K)}{K(1-c^2)}\longrightarrow0.
\tag{NTFG.11}
$$

These are properties of the actual derived ray
phase channel; a nonzero one-step bracket is
not identified with a nonzero complete additive
fluctuation coefficient. At $c=1$ its state is
a Gaussian random walk with no invariant
probability, while the phase increments are
independent unit normals and have additive
variance one per update.
:::

:::{prf:proof}
Iterating the affine state recursion gives its
normal mean $c^nw_0$ and covariance
$(1-c^{2n})(1-c^2)^{-1}I_6$.
They converge to the displayed invariant normal
law. Two chains driven by the same ORIGINAL
Gaussian sources have difference $c^n(w_0-w'_0)$;
their characteristic functions consequently
give uniqueness of an invariant probability,
or directly iterate the invariant characteristic
function and let $c^n\to0$.

In that state law
$\operatorname{Cov}(v_{i,0},v_{i,k})
=c^{|k|}/(1-c^2)$. Subtract adjacent values
at both times. Its zero lag is
$2(1-c)/(1-c^2)=2/(1+c)$, and its positive
lag is
$[2c^k-c^{k-1}-c^{k+1}]/(1-c^2)$,
which is (NTFG.9). Different projected channels
consume different independent state/source rows.
For a finite phase window, the conditional
innovation contribution $T_KT_K^T$ is positive
definite; adding the nonnegative initial-state
covariance preserves that property. This proves
the full density and (NTFG.10). A fixed original
stream source changes only its mean by $T_Kg$,
whose Gaussian derivative gives the stated
native source score.

In the derived invariant state, $v_{i,0}$ is an
independent centered normal of variance
$(1-c^2)^{-1}$. The inverse recursion gives
$A_K\eta_i=\epsilon_i-(1-c)v_{i,0}\mathbf1$.
Its covariance is $I_K+a_c\mathbf1\mathbf1^T$,
which proves the first formula in (NTFG.16).
This rank-one matrix has eigenvalue $1+Ka_c$
along $\mathbf1$ and eigenvalue one on its
orthogonal complement. Its determinant and inverse
$I_K-a_c\mathbf1\mathbf1^T/(1+Ka_c)$ follow
directly. Since $\det T_K=1$, substitution into
the already derived Gaussian density and original
source direction $T_Kg_i$ proves the remaining
explicit action and score formulas. The normal
hidden initial state is integrated, not treated
as a fitted extra innovation.

The summable stationary state covariance
$c^{|k|}/(1-c^2)$ has spectral density
$(1+c^2-2c\cos\omega)^{-1}$: summing its two
geometric series verifies the Fourier identity
directly. Taking an adjacent-state difference
multiplies that density by
$|e^{i\omega}-1|^2=2-2\cos\omega$.
This proves (NTFG.17), including its displayed
Taylor expansion because $c<1$ keeps the
denominator strictly positive.

The summable geometric covariance has the exact
zero sum in (NTFG.11). The same conclusion follows
without summing a covariance series: the original
phase is the adjacent-state difference, so its
complete partial sum telescopes. Evaluating its
endpoint variance gives the displayed formula.
Its compensated innovation remains $\epsilon$
with unit bracket, as already proved. At $c=1$
the recursion has covariance growing linearly
and its invariant characteristic-function equation
would require invariance under addition of a
nondegenerate centered Gaussian, impossible for
a probability. Its phase is exactly its independent
Gaussian innovation, proving that endpoint scope.
:::

(sec-ntfg-native-burnin)=
## 5. A derived growing native-time window with stationary local phases

:::{prf:theorem} Quantitative original branch control and local stationary phase limit
:label: thm-ntfg-growing-window

Keep every original parameter fixed with $0<c<1$
in the complete register. Let
$d_D=\min_i\operatorname{dist}(x_i^*,D^c)>0$,
$W_0=|w_0|$, and evaluate

$$
\begin{gathered}
C_W=W_0+\frac4{1-c},\qquad C_X=t(4C_W+1),\\
d_* =\min\left\{1,\sqrt{\frac{1+c}{2c}}-1\right\},\\
C_d=\begin{cases}
t\nu\sqrt2 C_X/(\rho\sqrt e),&\mathrm{count},\\
0,&\mathrm{row},
\end{cases}\\
\epsilon_*=
\min\left\{\frac{d_D}{2C_X},
 \frac{10^{-2}}{C_X+C_W},\frac{d_*}{C_d}\right\},\\
C_E=(3C_W+1)C_d+C_W^2/V,\\
C_\eta=10^7C_W(C_X+C_W)+\frac{2\sqrt2 C_E}{1-c}.
\end{gathered}
\tag{NTFG.12}
$$

Here $d_*/C_d=+\infty$ when $C_d=0$, and
$1/V=0$ when the original cap is `None`.
These constants have only original parameters and
the specified entering-state/domain data as inputs.
For $0<q<1$, $n\ge1$, put

$$
M_{q,n}=4\log((n+2)/q),\qquad
\delta_{q,n}=8n e^{-M_{q,n}^2/4}.
\tag{NTFG.13}
$$

If $qnM_{q,n}\le\epsilon_*$, the ORIGINAL history
has, with probability at least $1-\delta_{q,n}$,
both walkers alive through every update $1,\ldots,n$,
its actual zero gates and unique nonself face records,
and all those literal faces available. Under the
coupled ORIGINAL source representation, on this event

$$
\begin{gathered}
\max_{l\le n}|X_l^q-x^*|\le C_XqnM_{q,n},
\qquad\max_{l\le n}|w_l^q|\le C_WM_{q,n},\\
\max_{l\le n}|w_l^q-w_l|
 \le\frac{C_E}{1-c}qnM_{q,n}^2,\\
\max_{l\le n}|\eta_l^q-\eta_l|
                         \le C_\eta qnM_{q,n}^2 .
\end{gathered}
\tag{NTFG.14}
$$

No source conditioning is imposed on the algorithm;
the full exceptional branch probability is included
in this estimate. Choose any actual update horizon
$n(q)\to\infty$ with $qn(q)^2\to0$.
For every fixed $K$, the last $K$ original two-face
phases converge jointly, with every fixed polynomial
moment, to the DERIVED stationary Gaussian channel
(NTFG.9)--(NTFG.10). The last position arrays converge
to the original base points and survival probability
tends to one. This is an actual joint parameter/time
limit of the executed gas; no invariant law for its
whole physical position process is asserted.

For a bounded Lipschitz test $O$ on that last
$2K$ phase window, its primitive expectation budget is

$$
\begin{aligned}
|E O(\eta^q_{n-K+1:n})-E_{\mathrm{stat}}O|
\le{}&\operatorname{Lip}(O)\sqrt K C_\eta qnM_{q,n}^2
       +2\|O\|_\infty\delta_{q,n}\\
&+\operatorname{Lip}(O)\sqrt{2K}(1-c)c^{n-K}
       \left(W_0+\sqrt{\frac6{1-c^2}}\right).
\end{aligned}
\tag{NTFG.15}
$$

The actual once-survivor-conditioned last window
has the same limit, because its conditioning event
has probability tending to one with the displayed
survival budget. Fixed final-alive masks, original
cap, both force normalizations and complete phase
source records therefore retain their derived scope.
Increasing-window action or moment assertions at
$c=1$, graph feedback, different source pools,
nonconstant reward or a thermostat-constrained
unavailable endpoint retain their actual separate
parameter regimes.
:::

:::{prf:proof}
Use only the original Gaussian event
$\max_{l\le n}|G_l|\le M$, $M=M_{q,n}$.
For a six-dimensional standard Gaussian,
$E e^{|G|^2/4}=8$. Union and exponential Markov
give its complement probability at most
$8ne^{-M^2/4}=\delta_{q,n}$.
Every subsequent assertion is a deterministic
bound on this ORIGINAL event, followed by that
paid exceptional probability; the noise law is unchanged.

On an all-alive step, let $B_1,B_2$ be its actual
separately evaluated particle kick matrices.
For row normalization both equal $S$ exactly.
For count normalization
$\|B_j-S\|=t\nu|K_j-K_0|$.
The original kernel has global spatial gradient
bound $1/(\rho\sqrt e)$ and pair-difference map
norm $\sqrt2$. If every current/intermediate
position displacement is at most $C_XqnM$,
then $\|B_j-S\|\le C_d qnM\le d_*$.
Thus $\|B_j\|\le1+d_*$ and
$c\|B_2B_1\|\le c(1+d_*)^2\le(1+c)/2$.
The actual cap decreases Euclidean velocity norm,
so the scaled next velocity obeys
$|w_l^q|\le[(1+c)/2]|w_{l-1}^q|+2M$.
The radius $C_WM$ is preserved, including its
specified entering value.

Each actual pair of A drifts changes position
by at most
$qt[(1+c)(1+d_*)C_W+1]M\le C_XqM$.
Induction from $X_0=x^*$ gives the position
bound, also at A1 and at the separately evaluated
B2 force position. The condition with $d_D$
keeps every actual terminal row inside the domain,
so the asserted all-alive branch and its exact
zero clone gates are maintained at every step.
This closes the stopped induction: no alternative
gate/revival law was assumed after a possible exit.

The $10^{-2}$ condition keeps every pre and
final raw ray within the strict norm/overlap
neighborhood of the original base rays. At zero
perturbation its norm is one, its same-slot
overlap is one and its cross overlap is
$1/\sqrt2$. The displayed perturbation gives
norm at least $0.99$ and normalized overlaps
above $1/2$, larger than both original thresholds.
Thus these are the actual available recorded faces.

Subtract the original affine reference recursion
from the actual pre-cap update. With
$d=C_dqnM\le1$,
$\|B_2B_1-I\|\le2d+d^2\le3d$.
Its deterministic remainder from the two count
kicks is at most $(3C_W+1)dM$.
The actual finite-cap remainder is at most
$q C_W^2M^2/V$; it is zero for no cap.
Consequently
$|w_l^q-w_l|\le c|w_{l-1}^q-w_{l-1}|
 +C_E qnM^2$. Its geometric sum proves the
third bound of (NTFG.14).

For the phase estimate use its original raw
complex overlap product $Q$. On the indicated
position/velocity neighborhood, all three raw
ray norms are at most two, $|Q|\ge1/4$,
$|DQ|\le192$, $|D^2Q|\le480$.
These bounds follow by differentiating its
three bilinear overlaps: each factor has
modulus at most four, first derivative at most
four, and second derivative at most two.
Thus the operator second derivative of its
argument is bounded by
$480/(1/4)+192^2/(1/4)^2<10^6$.
Normalization by positive real ray norms
does not change this argument.

Crucially its phase is zero for EVERY nearby
configuration of real rays, not only at the
original base points. Taylor expand only in
imaginary velocities. The coefficient change
caused by the actual position displacements
costs at most a constant times $qnM^2$,
and its quadratic velocity remainder at most
a constant times $qM^2$ after division by $q$.
At the base point the complete linear coefficient
is exactly the already proved adjacent-velocity
expression (NTFG.3). Its difference between
actual and Gaussian velocities costs at most
$2\sqrt2 C_E qnM^2/(1-c)$ for the two-channel
vector. The larger stated $10^7$ polynomial
bound covers both phase Taylor terms, proving
the last estimate in (NTFG.14). This avoids
an erroneous quadratic cost from purely real
position changes whose phase is identically zero.

If $qn^2\to0$, eventually $n\le q^{-1/2}$,
so $M_{q,n}\le C\log(1/q)$.
Then $qnM_{q,n}\to0$ and
$qnM_{q,n}^2\to0$ by
$qn\le\sqrt q\sqrt{qn^2}$.
Moreover $\delta_{q,n}$ decreases faster than
every fixed power of $q$: its exponent is a
negative squared logarithm, whereas $n\le q^{-1/2}$.
The estimates therefore give convergence in
probability of the original last fixed phase
window to its Gaussian reference, with all
scaled-phase polynomial moments. On the exceptional
event the true masked phase is bounded by
$\pi/q$, so its $p$th moment contribution is
at most $C_{K,p}q^{-p}\delta_{q,n}\to0$;
the Gaussian reference has uniformly bounded
moments when $c<1$, and its exceptional contribution
vanishes by Cauchy--Schwarz and the same tail budget.

Couple that Gaussian reference with its already
derived invariant state using the same original
future sources. Their state difference at step
$l$ is $c^l(w_0-W_0^{\mathrm{stat}})$, and their
phase difference is
$(c-1)c^{l-1}L_i\cdot(w_{i,0}-W_{i,0}^{\mathrm{stat}})$.
The invariant Gaussian moment obeys
$E|W_0^{\mathrm{stat}}|\le\sqrt{6/(1-c^2)}$.
Combining that endpoint difference, the native
event estimates and bounded-test exceptional
budget gives (NTFG.15). Every fixed window has
the stationary limit and its stated moments.
Finally conditioning a bounded observation on
survival changes its expectation by at most
$2\|O\|_\infty P(\mathrm{nonsurvive})/
P(\mathrm{survive})$, which tends to zero by
the same original event. This proves the
once-conditioned scope without imposing an
unknown stationary law on physical positions.
:::

(sec-ntfg-scope)=
## 6. Complete parameter scope of the temporal result

:::{prf:remark} Temporal correspondence and actual remaining endpoints
:label: rem-ntfg-scope

This derives both actual viscous kicks, original
stream innovations, finite-window drift/bracket,
full native phase-path action/source and a growing
native-time stationary LOCAL phase limit. At $c<1$
its native action has memory and its complete
additive phase variance is zero despite its nonzero
innovation bracket. At the independently configured
zero-friction diffusion-factor tag, $c=1$ and positive
$q$ is attainable; each finite native window instead
has independent phase increments and original-unit
path-action matching. The inverse-temperature-linked
tag retains its $q=0$ endpoint and is not assigned
that positive-noise process.

The result uses the dense real-coordinate canonical
count/row gas. Its two-row algebra is not assigned
to Python empty CSR or floored row normalization,
or to finite arithmetic underflow at extreme
unbounded source inputs. Nonzero landscapes,
asymmetric/historical donor pools, unequal reward
feedback, accepted clones and different kinetic
forces retain their original laws. The original
survival and missing-ray branches have been paid
in the stated joint limit rather than suppressed.

This two-face temporal ray channel does not identify
the different native B2 non-Abelian color connection
or its physical local algebra. It does establish
the exact complete action and fluctuation behavior
of an existing native gauge readout in substantive
included parameter regimes, including which apparent
one-step action and bracket matches persist through
the actual repeated algorithm.
:::
