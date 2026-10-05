# Joint native phase energy with every kinetic innovation retained

(sec-kuje-full-noise)=
## 1. Conditional full-Gaussian potential and force moments

:::{prf:definition} Actual conditional kinetic register
:label: def-kuje-register

Use the native record of {prf:ref}`def-kurc-record`. Condition on the
complete source/acceptance/component-Haar preparation and all recipient
jitters. Write $X_i$ for the resulting positions and $U_i$ for the first
viscous velocities, in their configured normalization. These may depend
on every jitter and on the original shared Haar rotations. Put

$$
t=.02,\quad c=e^{-.04},\quad q^2=(1-e^{-.08})/2,
\quad s=.02,\quad b=t(1+c),\quad a=t\nu,
$$
$$
F(x)=-2x-20\pi\sin(2\pi x),\qquad
g(x)=x+btF(x),\qquad
m_i=g(X_i)+bU_i,\quad \bar z_i=c[U_i+tF(X_i)].
\tag{KUJE.1}
$$

The sine and $g$ are coordinatewise. For the original independent
standard Gaussian arrays $\xi,\chi$, the remaining update is exactly

$$
y_i=m_i+tq\xi_i,\quad z_i=\bar z_i+q\xi_i,
\quad w_i=z_i+tF(y_i)+aC_i(y,z),
$$
$$
v_i^+=\Psi_V(w_i),\qquad x_i^+=y_i+s\chi_i.
\tag{KUJE.2}
$$

Here $C_i$ is the original B2 Gaussian viscous average, including
its actual nonself denominator in row mode; $\Psi_V$ is the configured
radial cap. In particular $y_i$ and $z_i$ contain the same $\xi_i$.
The terminal alive mark is still determined by $x_i^+$.
:::

:::{prf:lemma} Exact native Gaussian energy and force work
:label: lem-kuje-gaussian-energy

Set $\omega=2\pi$, $A=20\pi$, $r^2=t^2q^2$, and
$\tau^2=r^2+s^2$. For a scalar coordinate with mean $m$, define

$$
B_h=e^{-\omega^2h/2},\qquad
u_h(m)=m^2+h+10-10B_h\cos(\omega m),
$$
$$
\begin{aligned}
\mathcal V_h(m)={}&2h^2+4m^2h
+100\left\{\tfrac12[1+e^{-2\omega^2h}\cos(2\omega m)]
                     -B_h^2\cos^2(\omega m)\right\}\\
&+20B_h[\omega^2h^2\cos(\omega m)
                         +2m\omega h\sin(\omega m)],
\end{aligned}
\tag{KUJE.3}
$$
$$
\begin{aligned}
f_h(m)&=-2m-AB_h\sin(\omega m),\\
j_h(m)&=-2-A\omega B_h\cos(\omega m),\\
f_{2,h}(m)&=4(m^2+h)
+4AB_h[m\sin(\omega m)+\omega h\cos(\omega m)]\\
&\hspace{14mm}+\tfrac12A^2[1-e^{-2\omega^2h}\cos(2\omega m)].
\end{aligned}
\tag{KUJE.4}
$$

Conditional on $(X,U)$, the complete final native potential
$\mathcal U(x)=\sum_k[x_k^2+10-10\cos(\omega x_k)]$ satisfies

$$
\mathbb E\mathcal U(x_i^+)=\sum_k u_{\tau^2}(m_{ik}),\qquad
\operatorname{Var}(\mathcal U(x_i^+))
=\sum_k\mathcal V_{\tau^2}(m_{ik}).
\tag{KUJE.5}
$$

The B2 force moments are $\mathbb EF(y_{ik})=f_{r^2}(m_{ik})$,
$\mathbb EF'(y_{ik})=j_{r^2}(m_{ik})$ and
$\mathbb E F(y_{ik})^2=f_{2,r^2}(m_{ik})$. Its complete kinetic
energy before the B2 viscous term and radial cap is

$$
\begin{aligned}
\mathcal K_i^0
&=\tfrac12\mathbb E|z_i+tF(y_i)|^2\\
&=\tfrac12\sum_k\left[
\bar z_{ik}^2+q^2
+2t\bar z_{ik}f_{r^2}(m_{ik})
+2t^2q^2j_{r^2}(m_{ik})
+t^2f_{2,r^2}(m_{ik})\right].
\end{aligned}
\tag{KUJE.6}
$$

These are exact identities, with no small-jitter or bounded-noise
approximation. Their conditional coefficients contain no $N$.
:::

:::{prf:proof}
For $Y\sim\mathcal N(m,h)$, differentiation of
$\mathbb Ee^{i\omega Y}=e^{i\omega m-\omega^2h/2}$ gives

$$
\mathbb E[Y\sin(\omega Y)]
=B_h[m\sin(\omega m)+\omega h\cos(\omega m)],
$$
$$
\mathbb E[Y^2\cos(\omega Y)]
=B_h[(m^2+h-\omega^2h^2)\cos(\omega m)
                       -2m\omega h\sin(\omega m)].
$$

Use $\sin^2 u=(1-\cos2u)/2$ and
$\cos^2u=(1+\cos2u)/2$. Expanding $F(Y)$, $F(Y)^2$ and
$Y^2+10-10\cos(\omega Y)$ yields (KUJE.3)--(KUJE.5).
Different final coordinates are independent conditional on $(X,U)$,
so their potential variances add. The preparation itself is not
declared independent across rows.

The shared OU draw in (KUJE.2) has covariance
$\operatorname{Cov}(z_{ik},y_{ik})=tq^2$.
Gaussian integration by parts therefore gives

$$
\mathbb E[z_{ik}F(y_{ik})]
=\bar z_{ik}f_{r^2}(m_{ik})+tq^2j_{r^2}(m_{ik}).
$$

Expanding the square proves (KUJE.6). Gaussian polynomial
integrability justifies each derivative and integration by parts;
the periodic derivatives are bounded. This explicitly retains
the force work caused by the common OU-position innovation.
$\square$
:::

(sec-kuje-graph-cap)=
## 2. Exact B2 graph work and cap loss

:::{prf:theorem} Whole-update joint-energy identity in both normalizations
:label: thm-kuje-complete-energy

Use $\langle f,g\rangle_N=N^{-1}\sum_i f_i\cdot g_i$ and
$\|f\|_N^2=\langle f,f\rangle_N$. Define the actual terms

$$
\mathcal G_N
=a\langle z+tF(y),C(y,z)\rangle_N
                         +\tfrac12a^2\|C(y,z)\|_N^2,
$$
$$
\mathcal D_{\Psi,N}
=\frac1{2N}\sum_i[|w_i|^2-|\Psi_V(w_i)|^2]\ge0.
\tag{KUJE.7}
$$

Then, conditional on the actual preparation,

$$
\mathbb E\frac1N\sum_i
  [\mathcal U(x_i^+)+\tfrac12|v_i^+|^2]
=\frac1N\sum_{i,k}u_{\tau^2}(m_{ik})
 +\frac1N\sum_i\mathcal K_i^0
 +\mathbb E\mathcal G_N-\mathbb E\mathcal D_{\Psi,N}.
\tag{KUJE.8}
$$

In count mode, with $0\le a\le1$ and the original
$C_i=N^{-1}\sum_jK_\rho(y_i-y_j)(z_j-z_i)$, the further
explicit upper bound is

$$
\mathbb E\mathcal G_N
\le\frac{at^2}{4(1-a/2)}
       \frac1N\sum_{i,k} f_{2,r^2}(m_{ik}).
\tag{KUJE.9}
$$

All row-mode graph work in (KUJE.7) is retained with its configured
denominator. The negative count Dirichlet identity is not silently
used in the unweighted row norm.
:::

:::{prf:proof}
Expand $|z+tF(y)+aC|^2/2$, then subtract the actual cap loss.
The original radial cap decreases magnitude pointwise, proving
the sign in (KUJE.7). Integrate the independent final-position
Gaussian for the potential, and use (KUJE.6) for the kinetic base.
This proves (KUJE.8) in either normalization without independence
between B2 graph weights and OU velocities.

For the count case let $L_y=-C(y,\cdot)$. The Gaussian weights
are symmetric and in $[0,1]$. Thus
$0\preceq L_y\preceq I$ in the normalized row inner product:

$$
\langle z,L_yz\rangle_N
=\frac1{2N^2}\sum_{i,j}K_\rho(y_i-y_j)|z_i-z_j|^2
\le\|z\|_N^2.
$$

Consequently $\|L_yz\|_N^2\le\langle z,L_yz\rangle_N$.
Write $D_z=\langle z,L_yz\rangle_N$ and
$D_F=\langle F(y),L_yF(y)\rangle_N$. The Dirichlet
Cauchy--Schwarz inequality yields

$$
\mathcal G_N
\le-a(1-a/2)D_z+at\sqrt{D_FD_z}
\le\frac{at^2}{4(1-a/2)}D_F
\le\frac{at^2}{4(1-a/2)}\|F(y)\|_N^2.
$$

The middle inequality maximizes the displayed quadratic in
$\sqrt{D_z}$. Now integrate the exact force square (KUJE.4).
All normalizations are averaged before estimating, so (KUJE.9)
has no growing population factor. The existing fourth-moment
budgets justify averaging (KUJE.8) over the original unbounded
jitters, source law and common Haar variables. $\square$
:::

:::{prf:lemma} Population-uniform quantitative radial-cap dissipation
:label: lem-kuje-cap-jensen

For the configured $\Psi_V(w)=Vw/(V+|w|)$, set

$$
H_V(B)=\frac{V^2B}{(V+\sqrt B)^2},\qquad H_V(0)=0.
\tag{KUJE.10}
$$

Whenever $B=\mathbb E\|w\|_N^2<\infty$ under the actual
whole-array law,

$$
\mathbb E\|\Psi_V(w)\|_N^2\le H_V(B),\qquad
\mathbb E\mathcal D_{\Psi,N}\ge\tfrac12[B-H_V(B)].
\tag{KUJE.11}
$$

No row independence is needed. In count mode, using the explicit
upper bound

$$
B\le B_*:=\frac2N\sum_i\mathcal K_i^0
+\frac{at^2}{2(1-a/2)}\frac1N\sum_{i,k}f_{2,r^2}(m_{ik}),
$$

the full conditional joint energy is at most

$$
\frac1N\sum_{i,k}u_{\tau^2}(m_{ik})+\tfrac12H_V(B_*).
\tag{KUJE.12}
$$

:::

:::{prf:proof}
For $B>0$, direct differentiation gives
$H_V'(B)=V^3/(V+\sqrt B)^3>0$ and
$H_V''(B)=-3V^3/[2\sqrt B(V+\sqrt B)^4]<0$.
The continuous extension at zero is concave. Choose a row
uniformly, independently of its original whole-array law, and
apply Jensen to its uncapped squared radius. This proves the first
inequality in (KUJE.11). Subtract it from the uncapped second
moment to get the actual expected cap-loss lower bound. In count
mode (KUJE.6) and (KUJE.9) imply $B\le B_*$, and monotonicity
of $H_V$ proves (KUJE.12). Every noise outcome is integrated before
Jensen; no truncation or independence assumption enters. $\square$
:::

:::{prf:remark} Phase information and the remaining producer
:label: rem-kuje-phase-producer

The same formulas apply after translating a phase centre. Native
Rastrigin has $F(x+n)=F(x)-2n$ and
$g(x+n)=g(x)+(1-2bt)n$ for integer $n$, so centre motion and
inter-well force offsets remain in (KUJE.1). They are not a
symmetry that identifies different equilibrium orbits.

The analytic moments are implemented by
`rastrigin_gaussian_energy` in
`src/fragile/fractalai/theory/landscape_phase.py`. An independent
joint Gaussian quadrature checks potential mean and variance and
the correlated kinetic force work. The formula does not assert
that a central basin is absorbing: actual Gaussian crossings,
normalizer changes, source flux and cap dissipation still enter
the history-uniform phase producer. It supplies their kinetic
energy calculation without changing the configured force or noise.
:::
