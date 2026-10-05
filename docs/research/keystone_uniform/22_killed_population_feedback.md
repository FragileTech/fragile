# Large-box marked population feedback for the active viscous gas

This note keeps the terminal mark and mandatory revival in the population
map. It concerns the deterministic rooted-component population law. A
stationary marked revival law and its current-time alive restriction are
different objects from the QSD of a finite swarm killed at extinction.
The conservative positive-viscosity theorem in research record 19 remains
unchanged. The estimates below use its harmonic kinetic register and its
proved two-provider density comparison, while replacing its all-alive
preparation calculation.

(sec-kpf-regime)=
## 1. The actual marked register

:::{div} feynman-prose
A dead slot contributes its stored velocity to a collision, but its stored
position is replaced by an alive donor's position before the force kicks.
Keep these two facts separate. They explain why a dead slot can be charged
in total variation when comparing the environment, even though its retained
position can be arbitrarily large.

The extra mark cost can also be kept under control. If a landing position
is dead, its norm is at least the box radius. A fourth-moment weight already
pays for that event. Enlarging the box lets us charge a substantial price
for dead input types without paying that price often at the output.
:::

:::{prf:definition} Consistent marked population and Gaussian class
:label: def-kpf-marked-register

Use the component-Haar population map of
{prf:ref}`def-mean-field-rooted-collision`, the actual alive-only
measurement and cloning donor laws, bounded $C^1$ reward, and both positive
fitness powers $p_b=\theta\bar p_b$. The harmonic count-viscous kinetic
register is exactly {prf:ref}`def-pvb-record`: $F(x)=-x$, $h>0$,
$m=1-t^2>0$, $a_x=1-t^2(1+c)\in(0,1)$ and $q,s,\sigma_J,V,\rho>0$.
There is no history and no change to the component collision rule.

For $L\ge1$ put $D_L=[-L,L]^d$ and use types
$z=(x,v,a)$, where $|v|\le V$ and $a=\mathbf1_{D_L}(x)$.
For a marked probability $\mu$ write
$$
\mu_A=\mu\restriction\{a=1\},\quad
\mu_D=\mu\restriction\{a=0\},\quad
m_A=\mu_A1,\quad e_D=\mu_D1,\quad
\alpha_\mu=\mu_A/m_A.
\tag{KPF.1}
$$
Only $\alpha_\mu$ enters the actual reward/diversity means and regularized
variances. It is also the base probability for each donor law. An alive
root has the actual fitness acceptance gate; a dead root accepts its
revival donor with probability one. The entire tree retains its original
frozen-slot velocities, including those of dead vertices.

Let $\mathfrak G_{8,s,L}$ consist of the laws
$$
(x,v,a)\overset{\rm law}=
 (Y+sZ,v,\mathbf1_{D_L}(Y+sZ)),\quad
Z\sim N(0,I_d)\text{ independent of }(Y,v),\quad
|v|\le V,\quad \mu|x|^8\le H_8,
\tag{KPF.2}
$$
where $H_8$ is the explicit constant in
{prf:ref}`lem-pvb-eighth-moment`. Use full weighted variation
$\|\xi\|_w=\int w\,d|\xi|$ throughout. In particular $W_p=1+|x|^p$.
The fullmarked update $\mathcal F_L\mu$ stores every landing coordinate
and applies the terminal mark. No extinction-conditioned finite-swarm
transition is substituted for this population map.
:::

:::{prf:lemma} Revival source moments and alive mass
:label: lem-kpf-marked-moments

Fix $0<\epsilon_0\le1/4$, $m_0=1-\epsilon_0$, and suppose
$e_D\le\epsilon_0$. The actual sampled frozen source, before recipient
jitter, satisfies for $1\le p\le8$
$$
\mathbb E|x_{\rm src}|^p
\le\left[1+\frac{a_*}{\kappa_C}
              +\frac{\epsilon_0}{\kappa_Cm_0}\right]\mu|x|^p.
\tag{KPF.3}
$$
If
$$
\frac{a_*}{\kappa_C}+\frac{\epsilon_0}{\kappa_Cm_0}
       \le\frac{1-r_8}{2r_8},\qquad L^8\ge H_8/\epsilon_0,
\tag{KPF.4}
$$
then $\mathfrak G_{8,s,L}$ is invariant under $\mathcal F_L$ and every
law in this class has $m_A\ge m_0$. The terminal dead mass obeys
$e_D\le H_8/L^8$.
:::

:::{prf:proof}
A normalized cloning donor row has density at most $1/\kappa_C$ relative
to $\alpha_\mu$. Alive recipients have total accepted mass at most
$m_Aa_*$. Their averaged copied-source moment is therefore at most
$(a_*/\kappa_C)\mu_A|x|^p$. Their retained-source moment is at most
$\mu_A|x|^p$. Mandatory dead recipients have total mass $e_D$; their
averaged donor moment is at most
$e_D\mu_A|x|^p/(\kappa_Cm_A)$. Adding these nonnegative bounds gives
(KPF.3). A dead retained position never enters this source moment.

The collision speed bound is still $V_c$. For $t\nu\le1$ the first
count velocity average is convex and has the same bound. Repeat the
Gaussian/Young calculation of {prf:ref}`lem-pvb-eighth-moment`, including
recipient jitter, to get
$$
M_8^+\le
r_8\left[1+\frac{a_*}{\kappa_C}
                +\frac{\epsilon_0}{\kappa_Cm_0}\right]M_8+B_8.
$$
The coefficient in (KPF.4) is at most $(1+r_8)/2$. Since
$H_8=2B_8/(1-r_8)$, this preserves the moment sublevel. Final Gaussian
noise is independent of the stored capped velocity and the preceding
landing coordinate, so (KPF.2) is preserved. Finally a dead position has
$|x|\ge L$, whence $e_D\le M_8/L^8\le\epsilon_0$.
:::

(sec-kpf-boundary)=
## 2. Boundary variation belongs to preparation

:::{prf:lemma} Uniform large-box alive mass after every actual update
:label: lem-kpf-large-box-survival

Put $\sigma_*^2=a_x^2\sigma_J^2+\tau^2$ and
$\Delta_L=(1-a_x)L-bV_c$. For every consistent entering marked
population with nonzero alive mass, if $\Delta_L>0$, then
$$
\mathcal F_L\mu\{a=0\}
\le 2d\exp[-\Delta_L^2/(2\sigma_*^2)].
\tag{KPF.15}
$$
This bound requires no entering moment bound and is uniform in the
population alive fraction. An output has finite moments of every fixed
order and has the representation (KPF.2) without its moment-sublevel
restriction. In particular its eighth moment is at most
$$
M_{8,\rm box}=2^7\left[(a_x\sqrt dL+bV_c)^8+\sigma_*^8G_8\right].
$$
The corresponding actual finite-array raw update has the
same one-row dead-probability bound conditional on its frozen preparation
choices; it does not by itself give a QSD or a survival-conditioned
mixing bound.
:::

:::{prf:proof}
Every frozen source is the position of an entering alive slot, either
the root itself or its chosen donor. It lies in $D_L$, even when a
mandatory revival came from a far outside retained dead slot. The exact
landing identity is
$$
x^+=a_xx_{\rm src}+bU+a_x I J+tq\xi+s\zeta,
\qquad |U|\le V_c.
$$
The bound on $U$ follows from the count-convex first velocity average
and the original-slot collision bound. The indicators $I$ and frozen
sources are chosen before recipient jitters and kinetic innovations.
Conditional on those choices, each coordinate of
$a_x I J+tq\xi+s\zeta$ is centered Gaussian of variance at most
$\sigma_*^2$. The term $U$ may depend on these noises; its pointwise
bound is sufficient and no independence from $U$ is asserted.
If $|a_x I J+tq\xi+s\zeta|_\infty\le\Delta_L$, the landing is
inside the box. The Gaussian coordinate tail bound and a union bound
give (KPF.15). Bounded source and bounded $U$ also give finite moments
of every fixed order. More explicitly, the Euclidean source norm is
at most $\sqrt dL$ and the Gaussian vector has eighth moment at most
$\sigma_*^8G_8$. Apply $(A+B)^8\le2^7(A^8+B^8)$ to obtain the
displayed uniform eighth-moment bound. The independent final position Gaussian gives
the stated representation. The same identities hold row by row in the
actual finite array.
:::

:::{div} feynman-prose
The new boundary derivative occurs when an alive input is kept at its own
position. That input density is a Gaussian mixture restricted to the box,
so differentiating it includes the density on each face. Copied positions
have their own recipient Gaussian and do not create this term.

The terminal decision itself is a fixed map applied after the kinetic
density comparison. It does not need to be differentiated in that
comparison. Confusing these two places would either omit an input boundary
term or invent an unnecessary output regularity assumption.
:::

:::{prf:lemma} Explicit weighted Gaussian boundary trace
:label: lem-kpf-boundary-trace

For a Gaussian-mixture input (KPF.2), $1\le p<8$, and the sum over all
box faces of its spatial trace, define
$$
B_p=2^{2p-2},\quad A_p(L)=1+2^{p-1}L^p+B_ps^pG_p.
$$
The trace weighted by $W_p$ is at most
$$
\mathcal T_p(L)=\frac{2d}{\sqrt{2\pi}s}
\left[
 e^{-L^2/(8s^2)}\{A_p(L)+B_p(L/2)^p\}
 +\frac{2^8H_8A_p(L)}{L^8}
 +\frac{2^{8-p}B_pH_8}{L^{8-p}}
\right].
\tag{KPF.5}
$$
In particular $\mathcal T_5(L)=O(L^{-3})$. A uniform bound for all
$L\ge1$ is obtained by replacing the first bracket by
$$
A_{p,0}+(2^{p-1}+B_p2^{-p})(4ps^2)^{p/2}e^{-p/2},\quad
A_{p,0}=1+B_ps^pG_p,
$$
and replacing $A_p(L)/L^8$ and $L^{p-8}$ by
$A_{p,0}+2^{p-1}$ and $1$. Denote that finite bound by
$\overline{\mathcal T}_p$.
:::

:::{prf:proof}
Conditional Jensen gives $\mathbb E|Y|^8\le H_8$. On the face
$x_a=L$, the Gaussian trace conditional on $(Y,v)$ is the one-coordinate
density $e^{-(L-Y_a)^2/(2s^2)}/(\sqrt{2\pi}s)$ times the remaining
Gaussian law. On that remaining law,
$$
\mathbb EW_p(L,Y_{-a}+sZ_{-a})
\le A_p(L)+B_p|Y|^p.
$$
For $|Y|\le L/2$, use the Gaussian factor
$e^{-L^2/(8s^2)}$ and $|Y|^p\le(L/2)^p$. On the complement,
$\mathbb P(|Y|>L/2)\le2^8H_8/L^8$ and
$\mathbb E[|Y|^p;|Y|>L/2]\le2^{8-p}H_8/L^{8-p}$.
The face $x_a=-L$ has the same bound. Sum the $2d$ faces to obtain
(KPF.5). Finally
$\sup_{L\ge0}L^pe^{-L^2/(8s^2)}=(4ps^2)^{p/2}e^{-p/2}$,
which proves the displayed uniform envelope.
:::

:::{prf:lemma} Weighted BV for the actual marked population preparation
:label: lem-kpf-marked-preparation-bv

Freeze an environment $\mu\in\mathfrak G_{8,s,L}$ with $e_D\le\epsilon_0$.
Let $\lambda=\mu J_\mu$ be its actual all-alive prepared phase law, with
the actual component-Haar collision. Put
$$
\ell_b=(\epsilon_b\sqrt e)^{-1},\quad
L_F=H_rL_R/\sigma_r+H_s/\sigma_s,\quad
L_D=2\ell_D/\kappa_D,
$$
$$
C_{\rm out}=2a_*\ell_C/\kappa_C+L_{\rm rec}L_F,
\qquad
C_{\rm in}=\frac{a_*\ell_C+L_{\rm don}L_F
+\epsilon_0\ell_C/m_0}{\kappa_C},
\quad C_r=L_D+C_{\rm out}+2C_{\rm in},
$$
$$
r_J=a_*+\epsilon_0,\quad
c_J=\frac{a_*+\epsilon_0}{\kappa_Cm_0},\quad
M_p=H_8^{p/8}.
$$
For $1\le p<8$ define
$$
\begin{aligned}
L_{G,p}&=\frac{g_1}{s}(1+2^{p-1}M_p)
                     +2^{p-1}s^{p-1}G_{p+1},\\
L_{J,p}&=\frac{g_1}{\sigma_J}(r_J+2^{p-1}c_JM_p)
                    +2^{p-1}r_J\sigma_J^{p-1}G_{p+1},\\
B_{p,L}^{\rm src}
 &=d[L_{G,p}+L_{J,p}+C_r(1+M_p)]+\mathcal T_p(L).
\end{aligned}
\tag{KPF.6}
$$
Then
$$
\sum_j\int W_p(X)|D_{X_j}\lambda|\le B_{p,L}^{\rm src}.
\tag{KPF.7}
$$
These bounds retain the actual original-slot collision velocities and
the actual alive-only normalizers. They do not replace a finite empirical
provider by an independent one-row provider.
:::

:::{prf:proof}
Split the exact sampled-root preparation into its copied and persistent
subprobabilities. On the copied branch, the prepared position is the
frozen alive donor position plus the root's independent recipient jitter.
Collision and the decision to copy are independent of that jitter.
The total copied mass is at most $r_J$ and its averaged donor measure
is dominated by $c_J\mu_A$. Differentiating this jitter Gaussian and
integrating its actual source-weighted score gives $L_{J,p}$ per axis,
by $|y+\sigma_JZ|^p\le2^{p-1}(|y|^p+\sigma_J^p|Z|^p)$.

A persistent root is alive, has no outgoing accepted edge, and keeps
$X=x$. Conditional on its root measurement mark, only two quantities
at this root change when $x_j$ changes: its probability of no outgoing
edge and its incoming Poisson intensity. Beyond that first incoming
generation, a child has consumed its outgoing edge and its descendants
have their original environment laws. Its velocities stay frozen when
we differentiate $x_j$. Thus matching those first primitives matches
the full recursive component and its collision velocity.

The normalized measurement row has full variation derivative at most
$L_D$. With global normalizers fixed, the root fitness derivative is
at most $L_F$: the feature map and regularized sampled distance have
spatial derivative at most one. The normalized outgoing proposal row
has full variation derivative at most $2\ell_C/\kappa_C$. Its gate
is at most $a_*$ and has root-fitness derivative at most
$L_{\rm rec}$, giving $C_{\rm out}$.
An incoming alive intensity has target-position derivative at most
$(a_*\ell_C+L_{\rm don}L_F)/\kappa_C$. Dead incoming children
contribute at most $\epsilon_0\ell_C/(\kappa_Cm_0)$, have no
fitness gate, and have no incoming children of their own. For an
intensity derivative $h$, the Poisson law has full variation derivative
at most $2\int|h|$: differentiate its likelihood and bound the
compensated count by the sum of its two nonnegative expectations.
This proves the full variation derivative envelope $C_r$ for the
persistent-root collision kernel, including its no-edge weight.

Differentiate the current Gaussian position conditional on its actual
latent $(Y,v)$. Its weighted score gives $L_{G,p}$. Restriction to $D_L$
adds the face derivative measures bounded by (KPF.5). The derivative
of the persistent-root kernel adds at most $C_r(1+M_p)$ per axis.
Adding copied and persistent terms gives (KPF.7). Integrating the latent
law and then disintegrating by collision velocity preserves these
variation bounds. The same argument with smooth mollifiers yields the
weak derivative statement if the conditional velocity laws are singular.
:::

(sec-kpf-feedback)=
## 3. Exact alive forest and mandatory dead leaves

:::{div} feynman-prose
First draw the accepted forest among alive types. Then attach incoming
dead leaves to each exposed alive vertex. This is the same population
tree as the original rule: dead types cannot be donors and have already
used their one outgoing edge. The order of this exposure does not change
its law.

Now a change of alive environment has two small effects. Active alive
edges have probability proportional to the positive fitness powers.
Dead leaves have mean proportional to the dead fraction. A common dead
root has a mandatory edge, so its own comparison is not small; averaging
over roots makes that contribution small because few roots are dead.
:::

:::{prf:lemma} Conditional alive normalization and dead-leaf intensities
:label: lem-kpf-provider-normalization

Let $\mu,\mu'$ have $m_A,m'_A\ge m_0$, $e_D,e'_D\le\epsilon_0$,
and eighth moments at most $H_8$. Set
$$
H_A=H_8/m_0,\quad B_A=1+\sqrt{H_A},\quad
\delta_A=\|\alpha_\mu-\alpha_{\mu'}\|_{W_4},\quad
\delta_D=\|\mu_D-\mu'_D\|_1.
$$
For $w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}$,
$$
\delta_A\le\left(\frac1{\beta m_0}
                       +\frac{B_A}{m_0\omega}\right)
                           \|\mu-\mu'\|_w,
\qquad \delta_D\le\omega^{-1}\|\mu-\mu'\|_w.
\tag{KPF.8}
$$
At a common alive vertex with fixed physical type, its incoming dead
Poisson intensities can be coupled with failure hazard at most
$$
H_D=\epsilon_0 h_A\delta_A+h_D\delta_D,
\quad h_A=(\kappa_C^2m_0)^{-1},\quad
h_D=(\kappa_Cm_0)^{-1}
                   +\epsilon_0(\kappa_Cm_0^2)^{-1}.
\tag{KPF.9}
$$
Their individual mean is at most
$d_0=\epsilon_0/(\kappa_Cm_0)$.
:::

:::{prf:proof}
Subtract the two normalized alive measures, first dividing their
unnormalized difference by $m_A$, then changing the denominator in
$\mu'_A/m_A$. Its weighted norm is at most
$m_0^{-1}\|\mu_A-\mu'_A\|_{W_4}
 +(B_A/m_0)|m_A-m'_A|$. Use
$|m_A-m'_A|=|e_D-e'_D|\le\delta_D$ and the two parts of $w$ to obtain
(KPF.8). Conditional Jensen gives $\alpha_\mu|x|^8\le H_A$.

For a fixed alive target $z$, the exact dead-leaf intensity is
$$
\mu_D(du)\,
\frac{w_C(z_u,z)}{m_A Z_C(\alpha_\mu;z_u)},\qquad
\kappa_C\le Z_C(\alpha_\mu;z_u)\le1.
$$
The retained feature of $u$ and its stored velocity remain part of this
formula. Subtract first the conditional-alive denominator, then the dead
measure and the factor $m_A^{-1}$. The corresponding full intensity
norms are at most
$\epsilon_0\delta_A/(\kappa_C^2m_0)$,
$\delta_D/(\kappa_Cm_0)$ and
$\epsilon_0\delta_D/(\kappa_Cm_0^2)$. Independent Poisson processes
can be coupled by their common intensity, with failure probability at
most the remaining intensity mass. This proves (KPF.9).
:::

:::{prf:proposition} Marked common-root preparation feedback
:label: prop-kpf-marked-preparation-feedback

Evaluate the complete constants $C_J^{\rm env},L_J=B_AC_J^{\rm env}$
of {prf:ref}`thm-kuhw-frozen-provider-fourth-feedback` at $H_A$ above,
using the actual conditional-alive primitive register. Require
$0<\theta\le\min(1,\kappa_C/(8a_0))$. Put
$$
G=e^{1/4}<3,\qquad
C_*=64D_JB_A^2G(1+\kappa_D^{-1})(1+\kappa_C^{-2}),
$$
$$
\begin{aligned}
C_A&=L_J+C_*
 \left[\frac1{\kappa_Cm_0}+(1+\epsilon_0)h_A+1+L_J\right],\\
C_D&=C_*(1+\epsilon_0)h_D.
\end{aligned}
\tag{KPF.10}
$$
For every common consistent marked root law $\zeta$ with
$\zeta|x|^8\le H_8$ and $\zeta\{a=0\}\le\epsilon_0$,
$$
\|\zeta J_\mu-\zeta J_{\mu'}\|_{W_4}
\le C_A(\theta+\epsilon_0)\delta_A+C_D\delta_D.
\tag{KPF.11}
$$
The norm on the left is for the actual prepared all-alive physical phase
law. The different providers on the right are the actual alive-only
measured law and the full retained dead law; they are not replaced by a
common unconditioned donor pool.
:::

:::{prf:proof}
Expose the ordered accepted forest among alive vertices before attaching
dead leaves. Its incoming alive intensity has density at most
$c_*=a_*/\kappa_C$ relative to the conditional alive marked type law.
Every accepted alive edge strictly increases fitness. The ordered-arm
count of {prf:ref}`thm-kuhw-frozen-provider-fourth-feedback` bounds the
mean number of alive vertices by $e^{2c_*}\le G$ from a free root and
by $2G$ after one exposed edge. Additional dead children are independent
Poisson leaves, with mean at most $d_0$ at each alive vertex. They do not
alter this alive forest or its alive vertex count. A known mandatory
incoming dead child has consumed its outgoing edge; all additional dead
children retain their original Palm Poisson law.

Expose the root's source before the rest of its component. A persistent
alive root uses its own position; a copied root, alive or dead, uses its
selected alive donor. The prepared position is that source plus its own
recipient jitter when copied. For every prescribed tree and Haar mark,
$\mathbb E W_4(x_{\rm src}+I J)\le D_JW_4(x_{\rm src})$.
Only velocity depends on the rest of the component. Consequently every
component-discrepancy charge may be bounded using this already exposed
source, without multiplying a donor moment by a dependent failure event.

For an alive root, use the exact source-first coupling in the cited
provider theorem for the alive forest. Complete each own first marginal
without conditioning on future successful matches. Its proof applies to
the present conditional alive laws with moment bound $H_A$. The readout
may include any attached matched dead leaves: its absolute source-weight
bound is unchanged, and a fully matched component uses the same Haar
matrix. The conservative proof therefore charges the alive-forest
discrepancies by $\theta L_J\delta_A$ after averaging the common alive
root sublaw. Here that sublaw has mean $W_4$ at most $B_A$; the cited
pointwise bound, not a normalized-root substitution, is used.

There are two extra charges. A failed root measurement coupling has
probability at most $\delta_A/\kappa_D$. Subtract the common isolated-root
baseline as in the cited proof. The contribution from a nontrivial alive
forest is already included in its $\theta L_J$ alive-edge charge. On the
remaining isolated-alive-forest branch, only the root can receive a dead
leaf. Adding dead leaves changes that branch's increment only if such a
leaf is present, with probability at most $d_0$. Its
averaged source-weighted contribution is at most
$2D_JB_A d_0\delta_A/\kappa_D$.
On matched root marks and matched exposed source, couple the dead-leaf
processes at every queried common alive vertex using (KPF.9). The queried
vertices are a pathwise subset of a completed own alive forest with at
most $2G$ expected vertices. The failure hazard is uniform in every
exposed source type. The averaged source weight on an outgoing alive
edge is at most $c_*B_A$, while on a persistent root it is at most its
own $W_4$. Thus this additional component charge is at most
$8D_JB_A(1+c_*B_A)G H_D$.
Both extra charges are bounded by $C_*[d_0\delta_A+H_D]$.
This argument retains all dead original-slot velocities; no dead position
becomes a copied source.

For a common dead root, expose its mandatory outgoing alive donor first.
The normalized donor kernels satisfy the weighted bound
$$
\|P_C(\cdot\mid z,\alpha_\mu)
       -P_C(\cdot\mid z,\alpha_{\mu'})\|_{W_4}
\le(\kappa_C^{-1}+B_A\kappa_C^{-2})\delta_A.
$$
This follows by subtracting the normalized numerators; the feature of the
common retained dead root is kept fixed. Couple the donor physical types
by their common weighted measure, then their measurement companions by
the actual companion coupling. A donor or measurement mismatch has
source-weighted readout cost bounded by
$4D_JB_A(1+\kappa_C^{-2})(1+\kappa_D^{-1})\delta_A$.
On matched donor types, the remaining alive component has one known
incoming dead child. It has at most $2G$ mean alive vertices and the same
source-first conditional bounds. Its alive-environment charge is at most
$8D_JB_AG(1+\kappa_C^{-2})\theta C_J^{\rm env}\delta_A$;
its additional dead-leaf charge is at most
$8D_JB_AG(1+\kappa_C^{-2})H_D$.
These three displayed charges are bounded by
$C_*[(1+\theta L_J)\delta_A+H_D]$.
Average them over the common dead-root sublaw, whose mass is at most
$\epsilon_0$. Its retained position does not enter the source weight.

The resulting full variation bound is
$$
\theta L_J\delta_A+
C_*[d_0\delta_A+H_D]+
\epsilon_0C_*[(1+\theta L_J)\delta_A+H_D].
$$
Insert (KPF.9), $\theta\le1$ and $d_0=\epsilon_0/(\kappa_Cm_0)$.
The constants (KPF.10) bound its coefficients by
$C_A(\theta+\epsilon_0)$ and $C_D$ respectively. This proves
(KPF.11) for all tests $|\varphi|\le W_4$ and hence in full variation.
Every exposure used the prescribed own conditional law; dead leaves were
not assigned a weak fitness gate.
:::

(sec-kpf-frozen)=
## 4. Frozen marked drift and the kinetic provider comparison

:::{prf:lemma} A dead output cost is controlled by the position weight
:label: lem-kpf-marked-output-weight

For the consistent terminal mark and
$w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}$,
$$
w(x,v,\mathbf1_{D_L}(x))
\le1+(\beta+\omega/L^4)W_4(x).
\tag{KPF.12}
$$
Consequently the actual kinetic density comparison of
{prf:ref}`thm-pvb-viscous-feedback` applies to marked outputs after
replacing its $W_4$ postprocessing factor by
$1+\beta+\omega/L^4$. No derivative of the terminal indicator is
taken in this use.
:::

:::{prf:proof}
A dead point has $|x|\ge L$. Thus
$\mathbf1_{\{a=0\}}\le|x|^4/L^4$. The terminal-mark map is the same
measurable injective map in both output laws. Pushforward variation
under that map equals the physical output variation with its pulled-back
weight. Bound that weight by (KPF.12), and then by
$(1+\beta+\omega/L^4)W_4$. The weighted density proof needs only this
pointwise estimate, so discontinuity of the terminal indicator causes
no additional output term.
:::

:::{prf:lemma} Uniform frozen marked Harris estimate
:label: lem-kpf-frozen-marked-harris

Assume the uniform moment and dead-mass bounds above and $0\le t\nu\le1$.
Set $H_4=\sqrt{H_8}$ and use $r_4,B_4$ of
{prf:ref}`thm-pvb-frozen-weighted-contraction`. Put
$$
B_g=1-r_4+r_4H_4/(\kappa_Cm_0)+B_4,\quad
u_0=(1-r_4)/(2r_4),\quad r_H=(1+r_4)/2,\quad
C_B=(1+u_0)B_g.
$$
Choose $R>\max(2,2C_B/(1-r_H))$, and suppose
$\omega>\beta R$ and $\omega/(\beta L^4)\le u_0$.
On $T=W_4+(\omega/\beta)\mathbf1_{\{a=0\}}$,
$$
P_\mu T\le r_HT+C_B.
\tag{KPF.13}
$$
For $T(z)+T(z')<R$ both roots are alive. Use the kinetic minorization
of {prf:ref}`lem-pvb-frozen-minorization` with
$R_x=(R-1)^{1/4}$, $r<L$, and isolated-root probability
$(1-a_*)e^{-(c_*+d_0)}$. Let its resulting common floor be
$\epsilon_H>0$. For
$$
0<\beta<\frac{2\epsilon_H}{r_HR+2C_B},\quad
q_H=\max\left\{
\frac{2+\beta(r_HR+2C_B)}{2+\beta R},\quad
1-\epsilon_H+\frac\beta2(r_HR+2C_B)\right\}<1,
\tag{KPF.14}
$$
every frozen fullmarked kernel satisfies
$\|\xi P_\mu\|_w\le q_H\|\xi\|_w$ for zero-mass signed $\xi$.
:::

:::{prf:proof}
For an alive root the conditional frozen source fourth moment is at
most $|x|^4+a_*H_4/(\kappa_Cm_0)$. For a dead root it is at most
$H_4/(\kappa_Cm_0)$, with no retained dead position. Young's inequality
and the Gaussian noise give $P_\mu W_4\le r_4W_4+B_g$ for either root.
Also $T$ at an output is at most $(1+u_0)W_4$ by (KPF.12).
This proves (KPF.13).

The small-$T$ set contains no dead root because $\omega/\beta>R$.
For an alive root, incoming alive and dead processes are independent
and have means at most $c_*$ and $d_0$. Its persistent isolated event
therefore has probability at least the stated product. On that event
the original physical input is retained. All fields and OU means have
the same moment bounds as in the proper-degree kinetic minorization;
that proof applies with the present bound on its actual joint second
provider. Its common final position ball is contained in $D_L$, so its
common marked law has mark one.

Repeat the elementary large/small weight coupling proof of
{prf:ref}`thm-pvb-frozen-weighted-contraction`, now with the energy $T$,
drift $r_H,C_B$ and common floor $\epsilon_H$. The diagonal-zero cost
is $w(z)+w(z')$ off the diagonal. Its optimal transport equals full
$w$ variation. The two ratios are exactly (KPF.14).
:::

(sec-kpf-positive-assembly)=
## 5. Explicit positive endpoints and marked population convergence

:::{div} feynman-prose
There are four small charges in the comparison: active preparation,
mandatory revival through a small dead fraction, changes of dead provider
types, and viscosity. The dead-type weight makes the third charge small;
the large box keeps that weight compatible with drift. The final endpoints
below assign each charge part of a fixed mixing margin. They are positive
numbers computed from the primitive data, even though they can be very
small.

At stationarity, the gas still contains a small dead population which is
revived on the following update. Conditioning this stationary law on its
current alive mark gives the alive population law. Conditioning an entire
finite swarm on survival through many updates is a separate change of
measure and requires the finite-swarm argument.
:::

:::{prf:definition} Primitive large-box positive parameter interval
:label: def-kpf-positive-endpoints

Use the positive linear register $a_0,c_0,J_b,G_0$ of
{prf:ref}`lem-kuhw-positive-preparation-register` and compute $r_8,B_8,H_8$
from the harmonic primitives as above. Choose
$$
\epsilon_0=\min\left\{\frac14,
                     \frac{\kappa_C(1-r_8)}{8r_8}\right\},
\quad m_0=1-\epsilon_0,
\quad
\theta_0=\min\left\{1,\frac{\kappa_C}{8a_0},
                \frac{\kappa_C(1-r_8)}{4r_8a_0}\right\},
$$
$$
\bar a=\theta_0a_0,\quad \bar c=\theta_0c_0,
k_0=\bar c+\frac{\epsilon_0}{\kappa_Cm_0},\quad
H_{{\rm p},p}
=\left[((1+k_0)H_8^{p/8})^{1/p}+\sigma_JG_p^{1/p}\right]^p
\quad(p=1,5),
$$
$$
M_2=cV_c+ctH_{{\rm p},1}+qG_1.
\tag{KPF.16}
$$
The moment condition (KPF.4) holds for these envelopes. Use the strictly
positive $\bar\nu$ in {prf:ref}`lem-pvb-joint-scores`. Compute the uniform
source BV envelope (KPF.6) at $p=5$, replacing its trace by
$\overline{\mathcal T}_5$, its moment by $H_8^{5/8}$, its gate by
$\bar a$, its fitness slopes by $\theta_0J_b$, and its gate derivatives
by $G_0$. Compute the explicit $M_*,S_x,S_w$ of that same joint-score
lemma at $\bar\nu$ and $H_{{\rm p},5}$ with this source BV envelope.
Then compute the physical $W_4$ kinetic feedback envelope
$\overline C_{\rm fb}$ using
{prf:ref}`thm-pvb-viscous-feedback`. All its increasing denominators and
factors use $\bar\nu$; the estimate holds uniformly at every smaller
nonnegative viscosity. These operations use only the displayed Gaussian
moments and derivative formulas, not an assumed Sobolev constant.

Use $r_4,B_4,B_g,u_0,r_H,C_B$ in
{prf:ref}`lem-kpf-frozen-marked-harris`. Put
$R=2+4C_B/(1-r_H)$ and $R_x=(R-1)^{1/4}$.
Fix $u=r=1$ in the proper-degree minorization and compute its common
floor $\bar\epsilon_H$ with $\bar a,\bar c,\bar\nu,M_2,R_x$,
using the isolated-root factor
$(1-\bar a)\exp[-\bar c-\epsilon_0/(\kappa_Cm_0)]$.
The floor is positive and at most one because it minorizes a probability.
Set
$$
\beta=\frac{\bar\epsilon_H}{r_HR+2C_B},\qquad
q_H=\max\left\{
\frac{2+\beta(r_HR+2C_B)}{2+\beta R},
                         1-\bar\epsilon_H/2\right\}<1,
\quad g_H=1-q_H.
\tag{KPF.17}
$$
Compute $C_A,C_D,B_A$ in (KPF.10) with this $H_8,m_0,\epsilon_0$,
and set
$$
C_J=1+D_JB_A/\kappa_C,\quad
C_K=1+\beta(1+u_0)(1+B_4),\quad
C_{\rm fb}^w=[1+\beta(1+u_0)]\overline C_{\rm fb}.
$$
Choose
$$
\omega=\max\left\{1,\beta R+1,\beta B_A,
                                  \frac{8C_KC_D}{g_H}\right\},
$$
$$
\boxed{\begin{aligned}
\theta_*&=\min\left\{\theta_0/2,
                     \frac{g_H\beta m_0}{16C_KC_A}\right\}>0,\\
\epsilon_*&=\min\left\{\epsilon_0/2,
                     \frac{g_H\beta m_0}{16C_KC_A}\right\}>0,\\
\nu_*&=\min\left\{\bar\nu,
 \frac{g_H}{8C_{\rm fb}^w
 [C_J/\beta+2C_A(\theta_0+\epsilon_0)/(\beta m_0)+C_D]}\right\}>0,\\
r_*&=(1+q_H)/2<1.
\end{aligned}}
\tag{KPF.18}
$$
Finally take one fixed box radius satisfying
$$
\boxed{\quad
L>\max\left\{1,R_x,\left(H_8/\epsilon_*\right)^{1/8},
 \left(\omega/(\beta u_0)\right)^{1/4},
 \frac{bV_c+\sigma_*\sqrt{2\log(2d/\epsilon_*)}}{1-a_x}
                       \right\}.\quad}
\tag{KPF.19}
$$
This defines a nonempty parameter interval and a finite box from the
primitive register. Its proof does not certify any smaller prescribed
radius or larger prescribed viscosity.
:::

:::{prf:theorem} Complete large-box active marked population contraction
:label: thm-kpf-large-box-population-convergence

Use {prf:ref}`def-kpf-marked-register` and
{prf:ref}`def-kpf-positive-endpoints`. If
$0<\theta\le\theta_*$ and $0<\nu\le\nu_*$, then for
$w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}$ the exact population update
satisfies
$$
\|\mathcal F_L\mu-\mathcal F_L\mu'\|_w
\le r_*\|\mu-\mu'\|_w
\quad(\mu,\mu'\in\mathfrak G_{8,s,L}).
\tag{KPF.20}
$$
The class is complete and invariant. It has a unique stationary marked
population law $\pi_L$ and
$$
\|\mu_n-\pi_L\|_w\le r_*^n\|\mu_0-\pi_L\|_w
\quad(\mu_0\in\mathfrak G_{8,s,L}).
\tag{KPF.21}
$$
Every consistent capped entering marked probability with positive alive
mass reaches this class at the uniform finite update number
$$
n_{\rm box}=1+\left\lceil
\frac{\log\max\{1,3M_{8,\rm box}/H_8\}}
     {-\log\lambda_{\rm burn}}\right\rceil,
\qquad \lambda_{\rm burn}=(1+3r_8)/4<1.
$$
In particular, for $n\ge n_{\rm box}$ its distance to stationarity is
at most
$$
[2+2\beta(1+\sqrt{H_8})+2\omega\epsilon_*]\,
                     r_*^{\,n-n_{\rm box}}.
$$
Every stationary
population law with positive alive mass belongs to it. These statements
concern the revival population map; an added all-dead absorbing state is
separate.
:::

:::{prf:proof}
The moment and Gaussian-class hypotheses hold by
{prf:ref}`lem-kpf-marked-moments` and the box condition. The sharper
large-box estimate (KPF.15) gives dead mass at most $\epsilon_*$ after
every update, and the moment sublevel gives the same bound on every
input in the class by (KPF.19). Thus the actual common-root feedback
argument (KPF.11) can use this smaller dead-mass envelope in its charges
while retaining the uniform constants computed at $\epsilon_0$.

Put $D=\|\mu-\mu'\|_w$ and use $\omega\ge\beta B_A$ in (KPF.8).
The prepared environment difference with common root law $\mu'$ is
bounded by
$$
\|\mu'J_\mu-\mu'J_{\mu'}\|_{W_4}
\le A_JD,\qquad
A_J=\frac{2C_A}{\beta m_0}(\theta+\epsilon_*)+\frac{C_D}{\omega}.
\tag{KPF.22}
$$
The root's prepared $W_4$ operator is bounded by $C_JW_4$: a persistent
root keeps its own position, whereas any copied root has conditional
donor $W_4$ at most $B_A/\kappa_C$ followed by its Gaussian jitter.
Consequently, for the actual prepared providers
$\lambda=\mu J_\mu$ and $\lambda'=\mu'J_{\mu'}$,
$$
\|\lambda-\lambda'\|_{W_4}
\le(C_J/\beta+A_J)D.
\tag{KPF.23}
$$
They have speed bound $V_c$, moments bounded by $H_{{\rm p},p}$, and
the completed weighted BV bound (KPF.7). The joint-score hypotheses
therefore hold, including the actual own conditional OU mean and the
actual correlated second provider. Source BV is proved before velocity
disintegration; the fibrewise first-field comparison used in the
joint-score proof remains valid. Its first-field interpolation and
second-provider pullback have the same positive inverse bounds as in
research record 19.

Freeze the full provider of $\mu$, obtaining its actual marked root
kernel $P_\mu$. The uniform frozen Harris estimate applies with the
values (KPF.17). The terminal dead weight is controlled by
$\omega/L^4<\beta u_0$. Thus
$$
\mathcal F_L\mu-\mathcal F_L\mu'
=(\mu-\mu')P_\mu+\mu'(P_\mu-P_{\mu'}).
$$
Its first term has norm at most $q_HD$. In its second term, first change
the preparation provider and keep both kinetic fields fixed. Its
postprocessing operator is bounded by $C_KW_4$ from (KPF.12) and the
conditional kinetic fourth-moment drift; this costs at most $C_KA_JD$.
Next keep the own prepared law $\lambda'$ fixed and change the two
actual kinetic providers. The proved weighted density comparison,
with (KPF.12), costs at most
$\nu C_{\rm fb}^w(C_J/\beta+A_J)D$.

The endpoints give
$C_KA_J\le g_H/4+g_H/8=3g_H/8$.
For the other charge, use
$A_J\le2C_A(\theta_0+\epsilon_0)/(\beta m_0)+C_D$ in (KPF.18)
to get at most $g_H/8$. The total coefficient is therefore
$q_H+g_H/2=r_*<1$, proving (KPF.20).

Weighted finite-measure variation is Banach. Probabilities, the velocity
cap, and the eighth-moment sublevel are closed under its convergence.
For the Gaussian-mixture and consistent-mark condition, take latent laws
$\eta_j=\operatorname{Law}(Y_j,v_j)$. Conditional Jensen gives
$\eta_j|Y_j|^8\le H_8$, so they have a tight subsequence. Its Gaussian
convolution has a position density and assigns zero mass to box faces.
The terminal-mark map is therefore continuous almost surely for that
limit. Its marked pushforward is the given norm limit. Hence
$\mathfrak G_{8,s,L}$ is closed and complete. It is nonempty because
$\operatorname{Law}(sZ,0,\mathbf1_{D_L}(sZ))$ has eighth moment at
most $H_8$; this follows from the positive Gaussian contribution in
$B_8,H_8$. Moment invariance and the final independent Gaussian make
the map invariant on this class. Banach's contraction proof gives
$\pi_L$ and (KPF.21).

For an arbitrary consistent input with positive alive mass, every source
is in the fixed box. The first output has eighth moment
$M_{8,1}\le M_{8,\rm box}$,
the Gaussian representation, and dead mass at most $\epsilon_*$. At
all subsequent updates, $\theta\le\theta_0/2$ and
$\epsilon_*\le\epsilon_0/2$ give
$$
\frac{a_*}{\kappa_C}
+\frac{\epsilon_*}{\kappa_C(1-\epsilon_*)}
\le\frac{1-r_8}{4r_8}.
$$
Indeed their two respective bounds are
$(1-r_8)/(8r_8)$ and $(1-r_8)/(12r_8)$, using $m_0\ge3/4$.
Thus $\lambda_{\rm burn}=(1+3r_8)/4<1$ and
$$
M_{8,1+j}\le\lambda_{\rm burn}^jM_{8,1}+2H_8/3.
$$
The displayed $n_{\rm box}$ ensures
$\lambda_{\rm burn}^{n_{\rm box}-1}M_{8,\rm box}\le H_8/3$ and
proves uniform entry. At entry both this law and $\pi_L$ have
$w$ moment at most $1+\beta(1+\sqrt{H_8})+\omega\epsilon_*$;
their difference norm is at most the sum of those moments. Apply the
contraction for the subsequent updates to get the stated uniform bound.
A stationary law with positive alive mass is an output, so has finite
eighth moment and the representation. The same drift gives its moment
at most $2H_8/3$. It consequently belongs to the class and equals
$\pi_L$. No claim is made about a population with zero alive mass.
:::

:::{prf:corollary} Current-time alive population relaxation
:label: cor-kpf-current-alive-relaxation

Let $\alpha_n=(\mu_n)_A/\mu_n\{a=1\}$ and
$\pi_L^A=(\pi_L)_A/\pi_L\{a=1\}$ in the preceding theorem.
For $\mu_0\in\mathfrak G_{8,s,L}$ and TV defined by $\sup_B$,
$$
\operatorname{TV}(\alpha_n,\pi_L^A)
\le\min\left\{1,\frac{r_*^n}{2m_0}
                                   \|\mu_0-\pi_L\|_w\right\}.
\tag{KPF.24}
$$
For a positive physical phase matrix $G$,
$$
W_{2,G}(\alpha_n,\pi_L^A)^2
\le4\lambda_{\max}(G)(dL^2+V^2)
      \min\left\{1,\frac{r_*^n}{2m_0}
                                   \|\mu_0-\pi_L\|_w\right\}.
\tag{KPF.25}
$$
The physical-time $W_2$ rate is $-\log r_*/(2h)$. The bounds are
population bounds with no particle number. They do not identify
$\pi_L^A$ with a finite-swarm QSD or prove conditioning on survival
through a future horizon.
:::

:::{prf:proof}
For probabilities $\mu,\mu'$ with alive masses at least $m_0$, subtract
their normalized alive restrictions. In unweighted full variation,
$$
\|\alpha_\mu-\alpha_{\mu'}\|_1
\le m_0^{-1}\{
\|\mu_A-\mu'_A\|_1+|m_A-m'_A|\}
\le m_0^{-1}\|\mu-\mu'\|_1.
$$
The second inequality uses
$|m_A-m'_A|=|e_D-e'_D|\le\|\mu_D-\mu'_D\|_1$.
TV is half full variation and $w\ge1$, so (KPF.21) gives (KPF.24).
Alive physical phase lies in $D_L\times\overline B_V$, whose squared
$G$ diameter is at most $4\lambda_{\max}(G)(dL^2+V^2)$.
Match the common part of the two probabilities and couple the residual
parts arbitrarily to obtain $W_{2,G}^2\le\operatorname{diam}_G^2
\operatorname{TV}$. This proves (KPF.25) and the square-root rate.
:::

:::{prf:corollary} Nonempty original-step active marked regime
:label: cor-kpf-original-step-marked-regime

Use the original harmonic step and bounded-reward profile of
{prf:ref}`cor-pvb-original-step-active-viscosity`, with all its displayed
positive comparison, standardization, collision and noise primitives.
Compute the constants of {prf:ref}`def-kpf-positive-endpoints` from
that profile, choose $p_r=p_s=\theta_*/2$, $\nu=\nu_*/2$, and choose
one fixed $L$ satisfying (KPF.19). Then the exact active marked population
map has the unique attracting law and current-time alive relaxation above.
Complete fitness ties and zero raw fitness variance are allowed.
The default box radius $2$ and viscosity $0.3$ are not certified by these
endpoints.
:::

:::{prf:proof}
That primitive profile has $h=1/25$, $F=-x$, bounded reward
$R=-\tanh(|x|^2/2)$, positive Gaussian amplitudes and a radial velocity
cap. Its globally bounded comparison features give positive donor floors
independent of $L$. Every Gaussian moment, boundary trace envelope,
regularized fitness derivative, inverse bound and minorization floor
used here is finite; each denominator in (KPF.18) is strictly positive.
Hence $\theta_*,\epsilon_*,\nu_*>0$ and the right side of (KPF.19)
is finite. These values satisfy the displayed assumptions. The tree and
density proofs use Lipschitz positive-part gates at ties and never require
a positive realized fitness gap or variance.
:::
