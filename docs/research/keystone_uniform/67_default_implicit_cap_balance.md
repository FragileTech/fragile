# Actual implicit-cap friction, Gaussian injection and delayed inward balance

(sec-icb-register)=
## 1. The original complete update

:::{prf:definition} Implicit-cap default record
:label: def-icb-register

Retain the actual harmonic count record of research 34, 37,
38, 54 and 55:
$$
d=3,\quad L=2,\quad h=.04,\quad t=.02,\quad \nu=.3,\quad
a=t\nu=.006,\quad V=2,\quad V_c=4,\quad
\sigma_J=\sigma_x=.1,\quad\alpha_{\rm col}=.5.
$$
The original force and raw reward are $F=-x$,
$R=-|x|^2/2$. Keep current-frame sampled alive fitness and
its actual global normalizers, all actual eligible roles,
simultaneous frozen positional sources, mandatory revival,
original-velocity component Haar readout, recipient jitter,
both count kicks, original independent OU/final Gaussians,
native cap and terminal mark. The positive exponent interval
(DMC.3) supplies the declared finite actual rooted components.
No historical, curl, elite or geometry-feedback branch is
enabled in this restriction.

Set
$$
c=e^{-.04},\quad b=t(1+c),\quad a_x=1-tb,\quad
m_h=1-t^2,\quad q^2=(1-c^2)/2,\quad s^2=.0004,
$$
$$
z_v=c-tb,\qquad r_H=t(c+a_x)=m_hb.
$$
The actual stages are
$$
X=S+IJ,\quad P=v^{\rm p},\quad
U=(\operatorname{Id}-aL_X)P,
$$
$$
w=c(U-tX)+q\xi,\quad y=a_xX+bU+tq\xi,\quad
z=w-ty-aL_yw,\quad v^+=C_V(z),\quad x^+=y+s\chi.
\tag{ICB.1}
$$
All sources are alive in $D$, $|P|,|U|\le V_c$, but
retained entering dead positions have no imposed bound.
All finite inner products and traces below are normalized
row averages. Population versions use the actual root and
its independent stage-law environment in the count field.
All expectations in Sections 2--4 are raw proposals; their
finite entering law can already be its own current-survivor law.
The fresh OU remains independent at this precise stage.
:::

(sec-icb-resolvent)=
## 2. Exact native-cap friction and count alignment

:::{prf:lemma} Native cap as a convex-friction resolvent
:label: lem-icb-resolvent

On the open velocity ball $|v|<V$, put $u=|v|/V$ and
$$
\Psi_V(v)=V^2[-u-u^2/2-\log(1-u)],\qquad
e_V(v)=\nabla\Psi_V(v)=\frac{|v|}{V-|v|}v.
\tag{ICB.2}
$$
Extend $\Psi_V$ by $+\infty$ outside the open ball.
It is proper, lower semicontinuous and convex, and
$C_V=(\operatorname{Id}+\nabla\Psi_V)^{-1}$.
For $v=C_V(z)$ the exact identities are
$$
z=v+e_V(v),\qquad
\phi_V(v):=\langle v,e_V(v)\rangle=\frac{|v|^3}{V-|v|},
$$
$$
|z|^2-|v|^2=2\phi_V(v)+|e_V(v)|^2.
\tag{ICB.3}
$$
The cap is firm:
$$
\langle v-v',z-z'\rangle\ge|v-v'|^2.
\tag{ICB.4}
$$
Its Jacobian $D_C=DC_V(z)$ obeys
$$
\operatorname{tr}D_C=(d-1)(1-u)+(1-u)^2
=d-(d+1)u+u^2.
\tag{ICB.5}
$$
:::

:::{prf:proof}
For radius $r\in(0,V)$ the radial derivative of $\Psi_V$
is $r^2/(V-r)$ and its second radial derivative is
$r(2V-r)/(V-r)^2\ge0$. The radial derivative is nonnegative,
so this radial function is convex on the ball. It is zero
at the origin and diverges to infinity at the sphere, proving
the asserted extended lower semicontinuity and convexity.
The strictly convex proximal objective
$|v-z|^2/2+\Psi_V(v)$ has its unique minimizer inside the
ball. Its stationarity equation is $z=v+e_V(v)$.
Taking its radius solves $r=V|z|/(V+|z|)$, giving exactly
the configured native cap. Expansion proves (ICB.3).
Monotonicity of the gradient gives (ICB.4).
Finally the tangential and radial cap eigenvalues are
$V/(V+|z|)=1-u$ and $V^2/(V+|z|)^2=(1-u)^2$.
This proves (ICB.5), including $z=0$ by continuity.
:::

:::{prf:lemma} Alignment survives the complete nonlinear cap
:label: lem-icb-capped-alignment

For every realized actual second graph, write
$$
D_v=\langle v^+,L_yv^+\rangle,\quad
D_w=\langle w,L_yw\rangle,\quad
D_y=\langle y,L_yy\rangle.
$$
Then, pointwise in the full Gaussian outcomes,
$$
-a\langle v^+,L_yw\rangle
\le-\frac a4D_v+\frac{at^2}2D_y+a^3D_w
\le-\frac a4D_v+\frac{at^2}{2e}+a^3D_w.
\tag{ICB.6}
$$
The same inequality holds for the actual population stage law.
No graph/noise independence or cap/graph commutation is needed.
:::

:::{prf:proof}
Apply firm cap monotonicity (ICB.4) to each actual pair
$(z_i,z_j)$ and multiply by its nonnegative actual conductance.
Pair symmetrization gives
$\langle v^+,L_yz\rangle\ge D_v$.
Use the exact $z=w-ty-aL_yw$ to obtain
$$
\langle v^+,L_yw\rangle
\ge D_v+t\langle v^+,L_yy\rangle
       +a\langle v^+,L_y^2w\rangle.
$$
The actual count operator is self-adjoint with $0\le L_y\le I$.
Cauchy--Schwarz in its quadratic form gives
$|\langle v^+,L_yy\rangle|\le\sqrt{D_vD_y}$.
Also
$|\langle v^+,L_y^2w\rangle|
\le\|L_yv^+\|\|L_yw\|\le\sqrt{D_vD_w}$.
Young's inequalities
$at\sqrt{D_vD_y}\le aD_v/2+at^2D_y/2$
and
$a^2\sqrt{D_vD_w}\le aD_v/4+a^3D_w$
prove the first bound. Pair symmetrization and
$\sup_{r\ge0}r^2e^{-r^2/2}=2/e$ give $D_y\le1/e$.
All statements concern the realized joint $(y,w,z,v^+)$;
integration gives the population version.
:::

(sec-icb-stein)=
## 3. Exact OU injection with its actual graph trace

:::{prf:theorem} Gaussian cap-energy identity retaining both providers
:label: thm-icb-stein-cap

At each actual root put $D_C=DC_V(z)$ and let $d_y$ be
its count degree. For a finite array,
$$
d_{y,i}=\frac1N\sum_{j\ne i}K(y_i-y_j).
$$
Define
$$
\mathcal J=
\mathbb E\left[(m_h-ad_y)\operatorname{tr}D_C
 +at\,\mathcal T_C\right],\qquad
\mathcal T_{C,i}=\frac1N\sum_{j\ne i}
 K(y_i-y_j)(y_i-y_j)\cdot D_{C,i}(w_i-w_j).
\tag{ICB.7}
$$
Population $d_y,\mathcal T_C$ use their actual independent
stage-law integrals. Then
$$
\mathbb E\langle\xi,v^+\rangle=q\mathcal J
\tag{ICB.8}
$$
and the exact nonlinear cap-energy account is
$$
\begin{split}
\mathbb E|v^+|^2+\mathbb E\phi_V(v^+)
={}&z_v\mathbb E\langle U,v^+\rangle
-r_H\mathbb E\langle X,v^+\rangle\\
&+m_hq^2\mathcal J
-a\mathbb E\langle v^+,L_yw\rangle .
\end{split}
\tag{ICB.9}
$$
All source/component, first-provider and cap correlations
remain in these expectations.
:::

:::{prf:proof}
Condition before the independent OU array on the complete
actual preparation and its first count output. For $j\ne i$,
differentiate the actual row field with respect to its own
original OU vector:
$$
\partial_{\xi_i}(L_yw)_i
=q\frac1N\sum_{j\ne i}K(y_i-y_j)
\left[I-t(w_i-w_j)(y_i-y_j)^{\mathsf T}\right].
$$
The self term is identically zero and has derivative zero.
Since $\partial_{\xi_i}(w_i-ty_i)=m_hqI$,
$$
\partial_{\xi_i}v_i^+
=qD_{C,i}\left[(m_h-ad_{y,i})I+
at\frac1N\sum_{j\ne i}
 K(y_i-y_j)(w_i-w_j)(y_i-y_j)^{\mathsf T}\right].
$$
This derivative includes the OU-induced spatial graph response,
not only the Laplacian with conductances held fixed.
Gaussian integration by parts for each own OU coordinate
and then summing its trace proves (ICB.8).
It is valid for the full finite noise array even though the
second graph couples those rows. In the population version
the own root noise does not vary the deterministic provider
law; differentiating its actual environment integral gives
the same formula.

The integration by parts can first be performed with a smooth
Gaussian cutoff and then passed to the limit. The cap and
its Jacobian are bounded. Conditional on the preparation,
all uncapped stages are affine in the OU vectors, and the
kernel and its first derivative are bounded. Their derivatives
are Gaussian-integrable. Source-box Gaussian preparation gives
the required integrated moments before removing this conditioning.
Section 4's explicit pair bound also proves uniform integrability
of the graph trace, without any cutoff in the conclusion.

The exact affine pre-second-kick velocity is
$w-ty=z_vU-r_HX+m_hq\xi$.
Substitute $z=v^++e_V(v^+)$ from (ICB.3), take its
inner product with $v^+$, and use (ICB.8).
This proves (ICB.9) with the complete actual second field.
:::

(sec-icb-primitive)=
## 4. Complete primitive bound and post-burn endpoint

:::{prf:lemma} Full-Gaussian bound for the remaining cap graph trace
:label: lem-icb-trace-bound

Put
$$
C_{\rm pair}=
\frac{3(c/a_x)^2(2V_c)^2}{e}
+\frac{12(ct/a_x)^2}{e^2}
+\frac{6dq^2(m_h/a_x)^2}{e}.
\tag{ICB.10}
$$
The complete actual trace obeys
$$
|\mathbb E\mathcal T_C|\le\sqrt{C_{\rm pair}}<8.5.
\tag{ICB.11}
$$
This is finite-array and population, uniformly in size and
in the unbounded prepared position displacement.
:::

:::{prf:proof}
Drop only the pointwise cap operator norm at most one:
$$
|\mathcal T_C|
\le\text{actual count average of }K(Y)|Y||W|.
$$
Cauchy--Schwarz in the actual normalized pair measure,
whose total mass is at most one, bounds its expected value
by the square root of
$\mathbb E[|Y|^2e^{-|Y|^2}|W|^2]$.
Freeze the whole prepared pair before both OU draws.
Then
$$
Y=a_xD+bH+tq\Xi,\quad W=c(H-tD)+q\Xi,\quad
\Xi\sim N(0,2I_d),\quad |H|\le2V_c.
$$
The exact identity
$W=(c/a_x)H-(ct/a_x)Y+(qm_h/a_x)\Xi$
and the three-term square inequality yield (ICB.10) from
$\sup r^2e^{-r^2}=1/e$,
$\sup r^4e^{-r^2}=4/e^2$ and
$\mathbb E|\Xi|^2=2d$. Finite self pairs are zero;
every distinct actual pair has this conditional Gaussian law.
No independence of $D_C,Y,W$ has been invoked.

The retained rational intervals imply
$c/a_x<1$, $ct/a_x<.02$, $m_h/a_x<1.001$,
$q^2<.0392$, $e^{-1}<.368$, $e^{-2}<.136$.
Thus
$$
C_{\rm pair}<
3(64)(.368)+12(.0004)(.136)
+18(.0392)(1.001)^2(.368)
=70.9168331812608<8.5^2.
$$
Every comparison is an exact terminating-rational inequality.
:::

:::{prf:theorem} Actual inward cap balance with a uniform primitive remainder
:label: thm-icb-primitive-balance

Let $M_w^2=\mathbb E|w|^2$ be the actual averaged uncapped
OU moment. Define
$$
\delta_{\rm cap}=
\frac{at^2}{2e}+a^3M_w^2+
m_h a t q^2\sqrt{C_{\rm pair}}.
\tag{ICB.12}
$$
Then the complete raw proposal obeys
$$
\begin{split}
\mathbb E|v^+|^2+\mathbb E\phi_V(v^+)+\frac a4\mathbb ED_v
\le{}&z_v\mathbb E\langle U,v^+\rangle
-r_H\mathbb E\langle X,v^+\rangle\\
&+m_h^2q^2\mathbb E\operatorname{tr}D_C
 +\delta_{\rm cap}.
\end{split}
\tag{ICB.13}
$$
For every actual entering law whose averaged stored velocity
RMS is at most $.56$, including an already current-survivor
finite entering law, the bound sharpens to the primitive
$$
M_w^2<.49,\qquad \delta_{\rm cap}<.000041.
\tag{ICB.14}
$$
No root-local velocity/displacement product is replaced by
this averaged RMS.

Writing $r_v=(\mathbb E|v^+|^2)^{1/2}<V$, a further
fully justified scalar consequence is
$$
\begin{split}
\left(1+\frac{dm_h^2q^2}{V^2}\right)r_v^2
+\frac{r_v^3}{V-r_v}+\frac a4\mathbb ED_v
\le{}&z_v\mathbb E\langle U,v^+\rangle
-r_H\mathbb E\langle X,v^+\rangle\\
&+dm_h^2q^2+\delta_{\rm cap}.
\end{split}
\tag{ICB.15}
$$
The mixed inward terms on the right are actual; no sign
for their combination is asserted.
:::

:::{prf:proof}
In (ICB.7) the degree is nonnegative and the cap trace is
nonnegative. Hence
$\mathcal J\le m_h\mathbb E\operatorname{tr}D_C+
at\sqrt{C_{\rm pair}}$.
Use this estimate and (ICB.6) in the exact (ICB.9),
and use $\mathbb ED_w\le\mathbb E|w|^2=M_w^2$.
This proves (ICB.12)--(ICB.13).

The actual component readout contracts the normalized original
velocity energy, and the first count average contracts that
prepared energy. Thus $\|U\|_2\le.56$ under the stated entering
budget. The source and recipient Gaussian identity gives
$\mathbb E|X|^2\le d(L^2+\sigma_J^2)=12.03<3.469^2$,
irrespective of retained dead positions.
Fresh OU centering before survival gives
$$
M_w^2=c^2\mathbb E|U-tX|^2+dq^2
\le c^2(.56+.02\sqrt{12.03})^2+dq^2
<(.9608)^2(.56+.02(3.469))^2+3(.0392)<.49.
$$
This uses the averaged energy only for the affine OU moment,
where the $L^2$ triangle is valid. It is not used in the
cap graph trace or a local correlated displacement product.
The exact remaining rational estimate is
$$
\delta_{\rm cap}<
\frac{(.006)(.0004)(.368)}2
+(.006)^3(.49)+(.006)(.02)(.0392)(8.5)
=.00004053144<.000041.
$$
This proves (ICB.14).

For (ICB.15), $0\le u<1$ implies
$\operatorname{tr}D_C\le d(1-u)$ by (ICB.5).
Also $|v^+|\ge|v^+|^2/V$, so
$\mathbb E\operatorname{tr}D_C\le
d(1-r_v^2/V^2)$.
The function of $b_0=|v|^2$
$$
f(b_0)=\frac{b_0^{3/2}}{V-\sqrt{b_0}}
=\sum_{k\ge0}\frac{b_0^{(3+k)/2}}{V^{k+1}}
$$
is convex on $[0,V^2)$: its nonnegative summands are convex,
and the increasing pointwise series preserves convexity.
Jensen therefore gives
$\mathbb E\phi_V(v^+)\ge r_v^3/(V-r_v)$.
Its expectation is finite because $\phi_V(v^+)\le V|z|$
and the actual source/OU stage has finite second moment.
Substitute these two bounds into (ICB.13).
:::

(sec-icb-delayed)=
## 5. Exact cross-flux interface and actual next survival

:::{prf:corollary} The retained cross moment in the nonlinear cap balance
:label: cor-icb-cross-flux

Let $H^+=\mathbb E\langle x^+,v^+\rangle$ under the
same actual raw proposal. Then
$$
H^+=a_x\mathbb E\langle X,v^+\rangle
+b\mathbb E\langle U,v^+\rangle+tq^2\mathcal J,
$$
$$
\mathbb E|v^+|^2+\mathbb E\phi_V(v^+)
+\frac{r_H}{a_x}H^+
=\frac c{a_x}\mathbb E\langle U,v^+\rangle
+\frac{m_hq^2}{a_x}\mathcal J
-a\mathbb E\langle v^+,L_yw\rangle .
\tag{ICB.16}
$$
These exact identities can be inserted into (DSTI.16)--(DSTI.19),
which keep the full mixed source/Haar/first-count account.

For a finite current-survivor input $\eta_n$, the nonnegative
left side of (ICB.13) under its own next-survivor proposal is
bounded by the raw right side of (ICB.13), computed from
$\eta_n$, divided by $1-e_N$. This is one actual next-survival
factor; no full-horizon hazard or fresh Gaussian identity on
the already restricted output is asserted.
:::

:::{prf:proof}
The final $\chi$ is independent of the stored $v^+$, so its
cross mean vanishes. The exact expression for $y$ and (ICB.8)
give the first identity. Eliminate $\mathbb E\langle X,v^+\rangle$
from (ICB.9). The elementary identities
$a_xz_v+br_H=c$ and
$m_h+r_Ht/a_x=m_h/a_x$ give (ICB.16).
Neither mixed term has been assigned a monotonicity sign.

For finite next survival the three raw observables
$|v^+|^2,\phi_V(v^+),D_v$ are pointwise nonnegative.
Their restricted sum is at most its unconditioned expectation
divided by the actual next survival probability, at least
$1-e_N$. Bound that raw expectation by (ICB.13).
The right side's mixed terms and Stein trace remain their
raw expectations; their noises have not been relabeled
independent after survival.
:::

:::{prf:remark} Completed primitive bound and missing signed inward inference
:label: rem-icb-scope

(ICB.6), (ICB.8)--(ICB.16) are complete actual finite and
population cap/OU/alignment accounts, with a population-size
independent primitive remainder less than $.000041$ whenever
the entering averaged velocity RMS is at most $.56$.
The proved population six-step velocity burn supplies this
budget. Finite uses may invoke only their already proved
current-survivor budget or the explicit entering budget
in the theorem, not a Gaussian law conditioned on a future
empirical-moment event.

The exact nonlinear friction and trace injection preserve the
inward mixed $X\cdot v^+$ term instead of dropping it into
an absolute phase moment. Their delayed use still needs a bound
on that mixed flux together with the original source/Haar
increment in (DSTI.18)--(DSTI.19).
This record proves neither entry into the fixed-width small-source
class nor a default full nonlinear law gap. The cap is unchanged;
its convex-friction representation is an identity, not an
altered integrator or additional reset.
:::
