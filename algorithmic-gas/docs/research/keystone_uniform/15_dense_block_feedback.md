# Constructive harmonic kinetic contraction with positive dense count viscosity

:::{prf:definition} The unchanged harmonic count-viscous kinetic law
:label: def-ku-count-harmonic-register

Work on conservative, all-alive arrays in
$\mathbb R^d\times\overline B_V$, with the actual smooth output cap
$C_V(v)=Vv/(V+|v|)$ and harmonic external force $F(x)=-x$.
For $N\ge2$ use the actual count-normalized dense operator
$$
(L_xv)_i=\frac1N\sum_{j\ne i}
 e^{-|x_i-x_j|^2/(2\rho^2)}(v_i-v_j),\qquad A_x=I-t\nu L_x.
$$
The singleton operator is zero. Put
$$
t=h/2,\quad c=e^{-\gamma h},\quad B=t(1+c),\quad
m=1-t^2,\quad a_x=1-tB,
$$
$$
q^2=b_O^2(1-e^{-2\gamma h})/(2\gamma)>0,\qquad
s^2=\sigma_x^2h>0,
$$
with the usual continuous value at $\gamma=0$. Assume
$\gamma>0$, $m,a_x>0$, $V,\rho>0$ and $0\le t\nu\le1$.
The complete kinetic update is
$$
u=A_xv-tx,\qquad
w=cu+q\xi,\qquad y=x+Bu+tq\xi,
$$
$$
z=A_yw-ty,\qquad
x^+=y+s\chi,\qquad v^+=C_V(z).
$$
All entries of the two innovation arrays $\xi,\chi$ are independent
standard $d$-Gaussians within each own kernel. The two compared kernels
share corresponding innovations. Their dense second-kick matrices
depend on the actual noisy array $y$. No conditional independence of
the output rows is imposed.

Allow entering velocities bounded by $V_*\ge V$, as after an actual
cloning preparation; the final cap remains $V$. Define
$$
\mathcal Q_N(S,T)=\frac1N\sum_i
 [m|x_i-\widetilde x_i|^2+|v_i-\widetilde v_i|^2].
$$
This is the complete fixed-slot physical quadratic; it is not a centered
variance proxy.
:::

:::{prf:lemma} Uniform averaged contraction of the actual smooth cap
:label: lem-ku-noisy-cap-gap

For $V,\sigma>0$ and a standard $d$-Gaussian $\xi$, put
$$
\delta_{\rm cap}=\frac1{16}
 \left[1-\left(\frac{V}{V+\sigma}\right)^2\right]>0.
$$
For all deterministic vectors $b,\widetilde b$,
$$
\mathbb E|C_V(b+\sigma\xi)-C_V(\widetilde b+\sigma\xi)|^2
\le(1-\delta_{\rm cap})|b-\widetilde b|^2.
$$
Moreover $DC_V$ is globally $4/V$-Lipschitz.
:::

:::{prf:proof}
The radial and tangential derivative eigenvalues of $C_V$ are
$V^2/(V+|z|)^2$ and $V/(V+|z|)$, so
$\|DC_V(z)\|\le V/(V+|z|)$.
For any scalar shift $b_1$, the mass of an interval of length two
under a standard Gaussian is maximal when it is centered at zero:
differentiate that interval mass and use the monotonicity of its
density in the absolute coordinate. Consequently
$$
\Pr\{|b+\sigma\xi|\ge\sigma\}
\ge\Pr\{|\xi_1|\ge1\}
\ge2\int_1^2(2\pi)^{-1/2}e^{-u^2/2}\,du
\ge\frac{2e^{-2}}{\sqrt{2\pi}}>\frac1{16}.
$$
The final elementary inequality follows from $e<3$, $\pi<4$ and
$32>9\sqrt8$. Hence the averaged squared operator norm of the
cap derivative is at most $1-\delta_{\rm cap}$, uniformly over its
mean. Integrate the derivative along the segment from $b$ to
$\widetilde b$ and apply Jensen before taking Gaussian expectation.

For the last assertion, write
$$
DC_V(z)=f(r)I-H(z),\quad
f(r)=V/(V+r),\quad
H(z)=h(r)ee^T,\quad
h(r)=Vr/(V+r)^2,\quad e=z/r.
$$
For $r>0$, $|f'|,|h'|\le1/V$ and $h(r)/r\le1/V$.
The derivative of $ee^T$ has norm at most $2/r$. Thus
$\|D(DC_V)(z)\|\le4/V$ away from zero.
The Jacobian has continuous extension $DC_V(0)=I$; integration along
segments, including those crossing zero, proves the stated Lipschitz
bound. $\square$
:::

:::{prf:lemma} Count-kernel estimates retaining noisy second-kick dependence
:label: lem-ku-count-noisy-pair-defect

Write $\|r\|_{2,N}^2=N^{-1}\sum_i|r_i|^2$ and
$\ell=e^{-1/2}/\rho$. For arbitrary deterministic position arrays,
$$
\|L_x\|\le1,\qquad
\|(L_x-L_{\widetilde x})v\|_{2,N}
\le2V_*\ell\|x-\widetilde x\|_{2,N}
 \quad\text{if }\max_i|v_i|\le V_*,
$$
$$
\|L_xx-L_{\widetilde x}\widetilde x\|_{2,N}
\le\|x-\widetilde x\|_{2,N}.
$$
If $y-\widetilde y=R$ is deterministic while $y,\widetilde y$ may
depend on the same Gaussian array $\xi$, then
$$
\mathbb E\|(L_y-L_{\widetilde y})\xi\|_{2,N}^2
\le4d\ell^2\|R\|_{2,N}^2.
$$
For each individual row,
$$
|(L_yy)_i|\le\rho e^{-1/2},\qquad
\big(\mathbb E|(L_y\xi)_i|^2\big)^{1/2}\le2\sqrt d.
$$
Every constant is independent of $N$. None of these statements assumes
that the noisy positions and Gaussian velocities are independent.
:::

:::{prf:proof}
The count Laplacian is symmetric positive semidefinite. Its quadratic
form is at most that of the complete unit-weight count Laplacian,
whose norm is one. Hence $\|L_x\|\le1$.
Differentiate the kernel along a position segment and pair its two
orientations. For a test array $b$, the resulting bilinear form is
bounded by
$$
\frac{2V_*\ell}{2N^2}
\sum_{i,j}|b_i-b_j||r_i-r_j|
\le2V_*\ell\|b\|_{2,N}\|r\|_{2,N},
$$
using
$(2N^2)^{-1}\sum_{i,j}|r_i-r_j|^2
=\operatorname{Var}_N(r)\le\|r\|_{2,N}^2$.
Duality and integration prove the first defect estimate.
The pair map $r\mapsto e^{-|r|^2/(2\rho^2)}r$ has derivative
operator norm at most one: its eigenvalues are $e^{-u/2}$ and
$e^{-u/2}(1-u)$, $u=|r|^2/\rho^2$, whose absolute values are
at most one. The same paired bilinear estimate proves the second
defect estimate.

For the Gaussian statement, the global kernel-gradient bound gives
$|K(y_i-y_j)-K(\widetilde y_i-\widetilde y_j)|
\le\ell|R_i-R_j|$. Jensen over the $N$ terms in a count row,
followed by $\mathbb E|\xi_i-\xi_j|^2=2d$, gives
$$
\mathbb E\|(L_y-L_{\widetilde y})\xi\|_{2,N}^2
\le\frac{2d\ell^2}{N^2}\sum_{i,j}|R_i-R_j|^2
\le4d\ell^2\|R\|_{2,N}^2.
$$
The bound was applied before expectation, so correlations of kernel
weights and $\xi$ remain admissible. The two row bounds follow from
$\sup_r |r|e^{-|r|^2/(2\rho^2)}=\rho e^{-1/2}$ and
$|(L_y\xi)_i|\le|\xi_i|+N^{-1}\sum_j|\xi_j|$. $\square$
:::

:::{prf:theorem} An explicit positive-viscosity kinetic contraction interval
:label: thm-ku-count-small-viscosity-contraction

Use the register above and define
$$
z_r=-t(c+a_x),\quad z_v=c-tB,\quad
d_O=m(1-c^2),\quad \sigma_0=qm,
$$
$$
\delta=\frac1{16}\left[1-\left(\frac{V}{V+\sigma_0}\right)^2\right],
\qquad
T=d_O(1+t^2/m)+\delta(z_r^2/m+z_v^2),
$$
$$
\kappa_0=\min\left\{\frac12,\frac{d_O\delta t^2}{mT}\right\}>0,
\qquad q_0=\sqrt{1-\kappa_0}<1.
$$
For the dense perturbation put
$$
\begin{gathered}
\sigma_*=\frac{qm}{a_x},\quad A=\frac c{a_x},\quad
J=2AV_*\ell+\frac{ct}{a_x}+2\sigma_*\sqrt d\,\ell,\\
D_1=t(1+2V_*\ell/\sqrt m),\quad
R_*=\frac{a_x}{\sqrt m}+B,\quad
Z_*=\frac{|z_r|}{\sqrt m}+|z_v|,\\
C_p=|z_v|D_1+tA(1+D_1)+tJ(R_*+BD_1),\\
D_z=t[2|z_v|V_*+2AV_*+(ct/a_x)\rho e^{-1/2}
                                      +2\sigma_*\sqrt d],\\
C_\nu=\sqrt m\,BD_1+C_p+(4/V)D_zZ_*,\\
\nu_*=\min\{1,1/t,(1-q_0)/(2C_\nu)\}>0.
\end{gathered}
$$
For every $0\le\nu\le\nu_*$, the actual full count-viscous kinetic
updates under shared innovations satisfy
$$
\boxed{\quad
\mathbb E\mathcal Q_N(S^+,T^+)
\le\left(\frac{1+q_0}{2}\right)^2\mathcal Q_N(S,T),
\quad}
$$
for every $N$ and every pair of entering arrays with velocities bounded
by $V_*$. Both viscous kicks, the full Gaussian laws and the original
smooth cap remain in this estimate. No positions are truncated.
It is consequently an upper bound on optimal physical transport for
this kinetic kernel, rather than a lower bound on a chosen coupling.
:::

:::{prf:proof}
**The reference contraction.** For $\nu=0$ and a paired entering row
put $r=x-\widetilde x$, $\zeta=v-\widetilde v$ and
$U=\zeta-tr$. The exact uncapped differences are
$$
R_0=a_xr+B\zeta,\qquad Z_0=z_rr+z_v\zeta.
$$
The modified harmonic energy identity gives
$$
m|R_0|^2+|Z_0|^2
=m|r|^2+|\zeta|^2-d_O|U|^2.
$$
In each own reference row the pre-cap velocity has Gaussian noise
$qm\xi_i$, and its paired difference $Z_0$ is deterministic.
The noisy-cap lemma therefore subtracts the further loss
$\delta|Z_0|^2$. The loss matrix is
$D=d_O(-t,1)^T(-t,1)+\delta(z_r,z_v)^T(z_r,z_v)$.
The two row vectors have determinant $t$, since
$-tz_v-z_r=t(a_x+tB)=t$.
Thus $\det D=d_O\delta t^2$, and the trace after conjugating by
$\operatorname{diag}(m,1)^{-1/2}$ is exactly $T$.
The smaller eigenvalue of a positive two-dimensional matrix is at
least its determinant divided by its trace. Summing rows gives
the reference root-mean-square contraction factor $q_0$.
The shared final position noise cancels its paired difference.

**The first viscous kick.** Set $v_A=A_xv$,
$\widetilde v_A=A_{\widetilde x}\widetilde v$ and
$d=v_A-v$. Each count matrix is row-stochastic with nonnegative
entries when $t\nu\le1$, so $\max_i|(v_A)_i|\le V_*$.
The count estimates give
$$
\|d-\widetilde d\|_{2,N}
\le t\nu(\|\zeta\|_{2,N}+2V_*\ell\|r\|_{2,N})
\le\nu D_1\sqrt{\mathcal Q_N}.
$$
In a shared-noise pair the actual discrepancy $R=y-\widetilde y$
is deterministic given the entering arrays, and
$$
\|v_A-\widetilde v_A\|_{2,N}
\le(1+\nu D_1)\sqrt{\mathcal Q_N},\qquad
\|R\|_{2,N}\le(R_*+B\nu D_1)\sqrt{\mathcal Q_N}.
$$

**The actual noisy second kick.** Solving the first-stage affine
position identity gives, in every own swarm,
$$
w=\frac c{a_x}v_A-\frac{ct}{a_x}y+\frac{qm}{a_x}\xi.
$$
Apply the count estimates to its three terms, retaining the actual
dependence of $L_y$ on $\xi$. Minkowski gives
$$
\big(\mathbb E\|L_yw-L_{\widetilde y}\widetilde w\|_{2,N}^2\big)^{1/2}
\le A\|v_A-\widetilde v_A\|_{2,N}+J\|R\|_{2,N}.
$$
The individual row bounds also give
$$
\big(\mathbb E|(L_yw)_i|^2\big)^{1/2}
\le2AV_*+(ct/a_x)\rho e^{-1/2}+2\sigma_*\sqrt d.
$$
Let $e_i$ be the actual viscous pre-cap velocity minus its
nonviscous reference pre-cap velocity from the same entering state
and same innovations. Then exactly
$$
e=z_vd-t\nu L_yw.
$$
Since $|d_i|\le2t\nu V_*$, the individual bound is
$(\mathbb E|e_i|^2)^{1/2}\le\nu D_z$.
The paired bound, using $\nu\le1$, is
$$
\big(\mathbb E\|e-\widetilde e\|_{2,N}^2\big)^{1/2}
\le\nu C_p\sqrt{\mathcal Q_N}.
$$

**The cap perturbation.** For $F(z,e)=C_V(z+e)-C_V(z)$,
nonexpansion of $C_V$ and the Jacobian Lipschitz bound give
$$
|F(z,e)-F(\widetilde z,\widetilde e)|
\le|e-\widetilde e|+(4/V)|z-\widetilde z||\widetilde e|.
$$
The two reference pre-cap velocities differ by the deterministic
$Z_0$ under their shared innovations. The uniform individual
second-moment bound for $\widetilde e_i$ therefore proves that the
root-mean-square paired cap perturbation is at most
$\nu[C_p+(4/V)D_zZ_*]\sqrt{\mathcal Q_N}$.
The positional perturbation is exactly $B(d-\widetilde d)$.
Minkowski in the modified-energy norm bounds the complete paired
perturbation by $\nu C_\nu\sqrt{\mathcal Q_N}$.

Combining this perturbation with the reference contraction yields
$$
\big(\mathbb E\mathcal Q_N(S^+,T^+)\big)^{1/2}
\le(q_0+\nu C_\nu)\sqrt{\mathcal Q_N}.
$$
The displayed interval makes
$q_0+\nu C_\nu\le(1+q_0)/2<1$.
The constructed innovation coupling has exactly the actual marginal
law of each full dense kinetic update; its second-kick rows remain
correlated. Taking its squared cost proves the theorem. $\square$
:::

:::{prf:corollary} A nonempty interval at the original kinetic step size
:label: cor-ku-original-step-positive-count

Take $h=0.04$, $\gamma=b_O=1$, $\sigma_x=0.1$,
$V=2$, $\rho=1$, and $d=3$, retaining the actual harmonic force.
For inputs produced by the canonical component collision with
$\alpha_{\rm col}=1/2$, take $V_*=4$.
All conditions of the theorem hold, so its primitive formula gives
an explicitly positive interval $0<\nu\le\nu_*$ independent of $N$.
Diagnostic floating evaluation gives
$$
\kappa_0\simeq3.7795830\cdot10^{-6},\qquad
C_\nu\simeq0.886834735,\qquad
\nu_*\simeq1.06547095\cdot10^{-6}.
$$
These decimal values are diagnostics; positivity and the interval are
proved by the displayed exact primitive formulas.
:::

:::{prf:proof}
Here $t=0.02$, $0<c<1$, $m=1-0.02^2>0$ and
$a_x=1-0.02^2(1+c)>0$. Both noises are strictly positive,
and every denominator in the theorem is positive and finite.
The two independent energy-loss forms have strictly positive
coefficients, so $\kappa_0>0$, $q_0<1$ and $\nu_*>0$.
The collision-prepared velocity envelope is
$(1+2|\alpha_{\rm col}|)V=4$. No Gaussian maximum over the
population occurs in any constant. $\square$
:::

:::{prf:corollary} Population-uniform invariant and alive Wasserstein relaxation
:label: cor-ku-count-kinetic-invariant-law

Disable cloning and death, and let $K_N$ be exactly the conservative
count-viscous kinetic kernel above on
$E_N=(\mathbb R^d\times\overline B_V)^N$.
Use $V_*=V$ in the theorem, or any larger declared envelope, and fix
$0\le\nu\le\nu_*$. Put
$$
G_m=\operatorname{diag}(mI_d,I_d),\qquad
\mathfrak q=q_0+\nu C_\nu\le(1+q_0)/2<1,
$$
$$
A_{\rm anc}=\sqrt{md(t^2q^2+s^2)+V^2},\qquad
M_\infty=\frac{A_{\rm anc}}{1-\mathfrak q}.
$$
Let $\mathcal W_{2,N}$ be the Wasserstein distance on $E_N$ whose
squared ground distance is $\mathcal Q_N$, and let
$W_{2,G_m}$ be the corresponding one-particle Wasserstein distance.
Then $K_N$ has a unique invariant law $\Pi_N$ in $\mathcal P_2(E_N)$,
and
$$
\mathcal W_{2,N}(\mu K_N^n,\Pi_N)
\le\mathfrak q^n\mathcal W_{2,N}(\mu,\Pi_N)
\le\mathfrak q^n
\left[
\left(\int\mathcal Q_N(S,0)\,\mu(dS)\right)^{1/2}
+M_\infty
\right].
$$
The invariant law is permutation invariant and, for every slot $i$,
$$
\int [m|x_i|^2+|v_i|^2]\,\Pi_N(dS)\le M_\infty^2.
$$
In particular this is an individual, rather than merely averaged,
invariant second-moment bound independent of $N$.

Define the all-alive empirical map
$\mathcal E_N(S)=N^{-1}\sum_i\delta_{(x_i,v_i)}$, its random-measure
laws
$$
\mathscr L_{N,n}=(\mathcal E_N)_\#(\mu K_N^n),\qquad
\mathscr L_{N,\infty}=(\mathcal E_N)_\#\Pi_N,
$$
and the sampled-alive laws
$$
\lambda_{N,n}=\int\mathcal E_N(S)\,\mu K_N^n(dS),\qquad
\lambda_{N,\infty}=\int\mathcal E_N(S)\,\Pi_N(dS).
$$
Write $\mathcal W_{2,\mathrm{emp},G_m}$ for the Wasserstein distance
between random-measure laws with ground distance $W_{2,G_m}$.
Both actual alive law targets satisfy
$$
\begin{aligned}
\mathcal W_{2,\mathrm{emp},G_m}
 (\mathscr L_{N,n},\mathscr L_{N,\infty})
&\le
\mathfrak q^n
\left[
\left(\int\mathcal Q_N(S,0)\,\mu(dS)\right)^{1/2}
+M_\infty
\right],\\
W_{2,G_m}(\lambda_{N,n},\lambda_{N,\infty})
&\le
\mathfrak q^n
\left[
\left(\int\mathcal Q_N(S,0)\,\mu(dS)\right)^{1/2}
+M_\infty
\right].
\end{aligned}
$$
These are population-uniform convergence rates to the exact
conservative finite-population invariant law and its two alive
images. No mean-field replacement or finite-population error floor
is used.
:::

:::{prf:proof}
The complete physical metric space $(E_N,\sqrt{\mathcal Q_N})$
is a closed subset of a finite-dimensional Euclidean space.
Its Wasserstein space $\mathcal P_2(E_N)$ is consequently complete.
One can verify the latter directly: couple a subsequence of a
Wasserstein-Cauchy sequence with summable successive root-mean-square
distances, glue those couplings, and use completeness and Minkowski
to obtain an almost surely convergent sequence whose limit also
converges in mean square. The Cauchy property then gives convergence
of the full sequence.

The innovation construction is a measurable coupling for every
input pair. Integrating it against an optimal input coupling gives
$$
\mathcal W_{2,N}(\mu K_N,\widetilde\mu K_N)
\le\mathfrak q\,\mathcal W_{2,N}(\mu,\widetilde\mu).
$$
At the zero array the first viscous term is zero, so
$x_i^+=tq\xi_i+s\chi_i$ and $|v_i^+|\le V$. Hence
$$
\int\mathcal Q_N(S,0)\,K_N(0,dS)
\le md(t^2q^2+s^2)+V^2=A_{\rm anc}^2.
$$
The same coupling and Minkowski show that $K_N$ maps
$\mathcal P_2(E_N)$ into itself and that
$$
\left(\int\mathcal Q_N(S,0)\,\mu K_N(dS)\right)^{1/2}
\le
\mathfrak q
\left(\int\mathcal Q_N(S,0)\,\mu(dS)\right)^{1/2}
+A_{\rm anc}.
$$
Thus Banach's contraction argument applies to $\mu\mapsto\mu K_N$:
the iterates from $\delta_0$ are Cauchy, their limit $\Pi_N$ is
invariant by the contraction estimate, and two invariant laws in
$\mathcal P_2(E_N)$ must have zero distance. Iterating the last moment
bound from $\delta_0$ and passing to the limit gives
$$
\left(\int\mathcal Q_N(S,0)\,\Pi_N(dS)\right)^{1/2}
\le \frac{A_{\rm anc}}{1-\mathfrak q}=M_\infty.
$$
The kernel commutes with every permutation of slots: its dense count
operator and cap are equivariant and its innovation arrays have
permutation-invariant laws. A permuted invariant law is therefore
another invariant law in $\mathcal P_2(E_N)$, and uniqueness makes
$\Pi_N$ permutation invariant. All its coordinate second moments
equal their bounded average. This proves the individual bound.
Iterating the kernel contraction and using the triangle inequality
through $\delta_0$ proves the full-swarm bound.

For every pair of arrays the fixed-slot matching is an admissible
empirical transport plan, so
$$
W_{2,G_m}(\mathcal E_N(S),\mathcal E_N(T))^2
\le\mathcal Q_N(S,T).
$$
Push any swarm coupling through the two empirical maps and take
the infimum to obtain the random-empirical-law bound.
For the sampled-alive bound, choose a uniform slot $I$ independently
of that same swarm coupling and couple $(x_I,v_I)$ to
$(\widetilde x_I,\widetilde v_I)$. Its cost is exactly the expected
$\mathcal Q_N$, and its marginals are exactly the stated sampled-alive
laws. Infimizing proves that bound too. Every step retains the
correlated dense kinetic law. All constants depend only on the
primitive register and the initial averaged moment, and not on $N$.
$\square$
:::

:::{prf:lemma} Active count-viscous fourth-moment drift at the original step
:label: lem-ku-count-active-harmonic-drift

Retain $F=-x$, $m>0$, $0<a_x<1$, $q,s>0$, and $0\le t\nu\le1$, but do not
require $\nu\le\nu_*$. Disable death and use the unchanged frozen-frame
cloning, Gaussian jitter and component collision law.
Suppose the actual distinct-donor proposal weights satisfy
$\kappa_C\le w_C(i,j)\le1$, with $\kappa_C>0$, and every acceptance
gate is at most $a_*$. Let the output cap be $V$ and use the actual
collision envelope
$$
V_c=(1+2|\alpha_{\rm col}|)V.
$$
Define
$$
\rho_4=\frac{1+a_x^4}{2},\qquad
u=\left(\frac{\rho_4}{a_x^4}\right)^{1/3}-1>0,\qquad
\tau^2=t^2q^2+s^2,
$$
$$
D_4=BV_c+\sqrt{a_x^2\sigma_J^2+\tau^2}\,[d(d+2)]^{1/4},
\qquad b_4=(1+u^{-1})^3D_4^4.
$$
For $M_4(S)=N^{-1}\sum_i|x_i|^4$, the actual full update obeys
$$
\boxed{\quad
\mathbb E[M_4(S^+)\mid S]
\le\lambda_4M_4(S)+b_4,\qquad
\lambda_4=\rho_4(1+a_*/\kappa_C).
\quad}
$$
In particular, if
$$
a_*\le\frac{\kappa_C(1-\rho_4)}{2\rho_4},
\qquad H_4=\frac{2b_4}{1-\rho_4},
$$
then $\lambda_4\le(1+\rho_4)/2<1$ and the class
$\{\mu:\int M_4\,d\mu\le H_4\}$ is invariant for every $N$.
These statements apply to the original $h=0.04$, $F=-x$, cap $V=2$,
$\nu=0.3$ count dynamics because $t\nu=0.006<1$.

The gate condition has an explicit nonempty primitive realization.
For positive logistic base floors $\varepsilon_r,\varepsilon_s$,
positive amplitudes $A_r,A_s$, fixed reference powers
$\bar p_r,\bar p_s>0$, and actual powers
$(p_r,p_s)=\theta(\bar p_r,\bar p_s)$, put
$$
D=\bar p_r\log\frac{\varepsilon_r+A_r}{\varepsilon_r}
 +\bar p_s\log\frac{\varepsilon_s+A_s}{\varepsilon_s},
\qquad a_0=\frac{De^D}{s_c},
$$
where $s_c>0$ is the actual acceptance scale. The primitive interval
$$
0<\theta\le
\min\left\{1,\frac{\kappa_C(1-\rho_4)}{2\rho_4a_0}\right\}
$$
implies the displayed moment drift for every actual sampled
standardizer and every raw reward for which the implemented fitness
is defined. Thus the moment argument imposes no symmetric-fitness
exclusion and does not use a positive fitness-gap hypothesis.
:::

:::{prf:proof}
Freeze the input, actual measurement marks and source probabilities.
For $N\ge2$, write $b_{ij}=P_C(j\mid i)g_{ij}$ for the accepted
source token and $p_i=\sum_{j\ne i}b_{ij}$. The proposal bounds give
$$
b_{ij}\le\frac{a_*}{\kappa_C(N-1)},\qquad
\sum_{i\ne j}b_{ij}\le a_*/\kappa_C.
$$
The pre-jitter selected source $Y_i$ therefore has conditional moment
$$
\mathbb E|Y_i|^4
=(1-p_i)|x_i|^4+\sum_{j\ne i}b_{ij}|x_j|^4,
$$
and summing these identities gives
$$
\frac1N\sum_i\mathbb E|Y_i|^4
\le(1+a_*/\kappa_C)M_4(S).
$$
This uses the incoming donor columns. Replacing them by the outgoing
acceptance alone would not justify the bound. The singleton has no
distinct accepted source and satisfies the same inequality.

Now freeze the complete selected-source and collision pattern.
The prepared position is $X_i=Y_i+I_i\sigma_JZ_{J,i}$ and the
prepared velocity has norm at most $V_c$. The first count matrix
$A_X$ is row-stochastic when $t\nu\le1$, so
$|(A_XV^C)_i|\le V_c$ even though it depends on the actual jitters.
The actual position output is exactly
$$
x_i^+=a_xY_i+B(A_XV^C)_i
       +a_xI_i\sigma_J Z_{J,i}+tq\xi_i+s\chi_i.
$$
The sum of the last three Gaussian terms has covariance
$(a_x^2I_i\sigma_J^2+\tau^2)I_d$. Minkowski and the pointwise
velocity bound thus give its conditional fourth-root moment at most
$a_x|Y_i|+D_4$. No independence of $A_XV^C$ and the jitters is
assumed. The inequality
$$
(a+b)^4\le(1+u)^3a^4+(1+u^{-1})^3b^4
$$
and $(1+u)^3a_x^4=\rho_4$ imply
$\mathbb E|x_i^+|^4\le\rho_4\mathbb E|Y_i|^4+b_4$.
Sum and use the source estimate. The second viscous kick and cap
change velocity only; their full laws remain present in the
underlying update. This proves the moment inequality.
The proposed gate condition gives
$\lambda_4\le(1+\rho_4)/2$, whence
$\lambda_4H_4+b_4\le H_4$.

Finally each logistic base belongs to
$[\varepsilon_b,\varepsilon_b+A_b]$, regardless of its standardized
argument. The ratio of the maximum to minimum fitness is at most
$e^{\theta D}$. The canonical positive-part acceptance gate,
including its positive denominator regularizer and clipping at one,
is therefore at most
$$
\frac{e^{\theta D}-1}{s_c}
\le\theta\frac{De^D}{s_c}=\theta a_0
\qquad(0<\theta\le1).
$$
All constants are finite and strictly positive, and $a_x<1$ for
the displayed original step. Substitution proves the explicit
positive exponent interval. Its conclusion is a moment drift,
not an active-law contraction. $\square$
:::

:::{prf:remark} Remaining active-selection obligation
:label: rem-ku-count-active-feedback-obligation

This record proves the full kinetic contraction on arbitrary entering
arrays, including collision-prepared velocity envelopes, at the original
step size and positive count viscosity in an explicit interval.
It does not prove that the interval contains the reference $\nu=0.3$.

The theorem becomes a complete conservative finite-swarm relaxation
theorem when cloning is disabled. For active cloning it is a proved
kinetic component. The source-token quadratic coupling has a linear
near-tie mismatch cost, so its preparation estimate cannot be
multiplied into this quadratic kinetic coefficient.
A complete active-law proof must supply a compatible smoothed or
weighted transport estimate for the actual finite source arrays and
component collisions, or a separate weighted Harris block argument.
That estimate must preserve the correlated dense second-kick law.
Survival conditioning is a further distinct obligation.
:::
