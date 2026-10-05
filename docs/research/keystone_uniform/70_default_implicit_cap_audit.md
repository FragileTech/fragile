# Independent audit of the implicit-cap count balance

(sec-ica70-source)=
## 1. Frozen source and actual parameter scope

:::{prf:definition} Reviewed implicit-cap source
:label: def-ica70-source

This independent audit concerns
`67_default_implicit_cap_balance.md` at SHA-256
`1541e58b1f499e2528e9358651c2f701af651761f6fa9b05a3968ec43c3fd821`.
Its register {prf:ref}`def-icb-register` retains the actual harmonic
count update at $h=.04$, $\nu=.3$, $d=3$, $L=2$, $V=2$,
$V_c=4$, positive recipient jitter and both full original kinetic
Gaussians. The force is $F=-x$ and the raw reward is $-|x|^2/2$.
Sources, current alive-only sampled fitness normalizers, mandatory
revival and original-slot component velocities remain unchanged.

The active population preparation is used in the positive exponent
interval (DMC.3), namely $0<\theta\le\theta_f$ of
{prf:ref}`def-dmc-register`. Unit reference fitness powers are not
certified by this audit. The primitive numerical remainder additionally
requires an entering averaged stored-velocity RMS at most $.56$.
It is an average over all slots, not a pointwise or alive-normalized
speed bound.

The source proves cap-energy, alignment, Gaussian-injection and mixed
cross-flux identities and inequalities. It does not assign a favorable
sign to the remaining mixed flux or prove a full law contraction.
:::

(sec-ica70-resolvent)=
## 2. Resolvent, firmness and actual graph alignment

:::{prf:lemma} Nonlinear friction and alignment verification
:label: lem-ica70-resolvent-alignment

The convex resolvent, Jacobian trace and capped alignment estimates
in {prf:ref}`lem-icb-resolvent` and
{prf:ref}`lem-icb-capped-alignment` are valid with the unchanged
native cap and the realized own count graph.
:::

:::{prf:proof}
For $r=|v|<V$, differentiating the source's radial potential gives
$\Psi'_V(r)=r^2/(V-r)$ and
$\Psi''_V(r)=r(2V-r)/(V-r)^2\ge0$. Its radial derivative is
nonnegative, its value at zero is zero, and it diverges at the sphere.
Extension by $+\infty$ is therefore proper, lower semicontinuous
and convex. The unique proximal minimizer is interior and satisfies
$$
z=v+e_V(v),\qquad e_V(v)=\frac{|v|}{V-|v|}v.
$$
Solving this radius equation gives exactly $v=Vz/(V+|z|)$.
Expansion proves the cap energy loss, and monotonicity of the convex
gradient proves firmness. Its tangential and radial Jacobian eigenvalues
are $1-u$ and $(1-u)^2$, with $u=|v|/V$, so (ICB.5) follows.

For every realized graph, pairwise firmness with its nonnegative
conductances gives $\langle v^+,L_yz\rangle\ge D_v$.
Substitution of $z=w-ty-aL_yw$ yields
$$
\langle v^+,L_yw\rangle
\ge D_v+t\langle v^+,L_yy\rangle
                  +a\langle v^+,L_y^2w\rangle.
$$
The Gaussian count operator is self-adjoint and obeys $0\le L_y\le I$,
because its kernel operator is positive semidefinite and its degree
operator is at most $I$. Consequently the two error terms are bounded
by $\sqrt{D_vD_y}$ and $\sqrt{D_vD_w}$. The source's two Young
inequalities leave precisely $-aD_v/4$, $at^2D_y/2$ and $a^3D_w$.
Finally
$$
D_y=\tfrac12\text{actual pair average of }
                 |y-y'|^2e^{-|y-y'|^2/2}\le1/e.
$$
This proves (ICB.6) pointwise in the complete stage law. The proof
uses no cap/graph commutation and does not average a noisy graph
before applying firmness.
:::

(sec-ica70-stein)=
## 3. Own-row Gaussian differentiation and the complete trace

:::{prf:lemma} Actual finite and population Stein identities
:label: lem-ica70-stein

The derivative, self exclusion and normalizations in
{prf:ref}`thm-icb-stein-cap` are correct for each finite array and
for the deterministic population provider.
:::

:::{prf:proof}
Condition on the complete actual preparation and its first count
output. Then $\partial_{\xi_i}w_i=qI$ and
$\partial_{\xi_i}y_i=tqI$, whereas $w_j,y_j$ do not depend
on $\xi_i$ for $j\ne i$. Differentiating the entire second count
row, including the spatial conductance derivative, gives
$$
\partial_{\xi_i}(L_yw)_i
=\frac qN\sum_{j\ne i}K(y_i-y_j)
       [I-t(w_i-w_j)(y_i-y_j)^{\mathsf T}].
$$
The self field and its derivative vanish identically; no independent
self noise or diagonal degree is added. Since
$\partial_{\xi_i}(w_i-ty_i)=m_hqI$, multiplication by the
actual cap Jacobian gives the source's full own-row derivative.
The trace of its outer-product term is
$(y_i-y_j)\cdot D_{C,i}(w_i-w_j)$, with the positive sign
and coefficient $at$ in (ICB.7).

Gaussian integration by parts in every own row coordinate, followed
by normalized row summation, therefore proves
$\mathbb E\langle\xi,v^+\rangle=q\mathcal J$.
The other rows may be coupled to $v_i^+$ through the graph; integration
by parts does not require them to be independent of that output.
It only uses the independent original OU coordinates before the field
is evaluated. All derivatives are Gaussian-integrable: $D_C$ is
bounded, kernel derivatives are bounded, and the uncapped pair velocities
grow at most affinely in these Gaussians after conditioning. The source
moment bounds and the following trace estimate justify removing the
conditioning and any temporary cutoff.

For the population identity, the provider is its deterministic actual
joint $(y,w)$ law. Varying one tagged root noise does not vary this
law. Differentiation of its environment integral thus has the same
own-root formula and trace. Its position and velocity are not replaced
by independent marginals.

Finally $w-ty=z_vU-r_HX+m_hq\xi$ pointwise. Take its inner product
with $v^+$, subtract the actual $aL_yw$, use
$z=v^++e_V(v^+)$ and insert the Stein identity. This proves every
coefficient in the exact energy identity (ICB.9).
:::

(sec-ica70-tail)=
## 4. Conditional full-Gaussian consumer and rational remainder

:::{prf:lemma} Uniform trace and post-burn arithmetic verification
:label: lem-ica70-bounds

The all-tail trace bound (ICB.10)--(ICB.11) and the primitive
post-burn remainder (ICB.14) hold without a local displacement or
individual velocity reduction.
:::

:::{prf:proof}
For each distinct pair, condition before both fresh OU coordinates.
The actual $D=X-X'$ and $H=U-U'$ are then fixed, with
$|H|\le2V_c$, and
$$
Y=a_xD+bH+tq\Xi,\qquad W=c(H-tD)+q\Xi,
\qquad \Xi\sim N(0,2I_3).
$$
The exact identity
$W=(c/a_x)H-(ct/a_x)Y+(qm_h/a_x)\Xi$ follows from
$a_x+ct^2=m_h$. Three-term Young and the two Gaussian-kernel
pointwise maxima give exactly the coefficient $C_{\rm pair}$.
Its upper rational certificate is
$$
3(64)(.368)+12(.0004)(.136)
 +18(.0392)(1.001)^2(.368)
=70.9168331812608<72.25=8.5^2.
$$
Using only $\|D_C\|\le1$, Cauchy--Schwarz in the normalized
pair measure then bounds the actual expected trace by
$\sqrt{C_{\rm pair}}$. Finite self terms are zero. This retains
the correlation of cap, graph, positions and velocities and integrates
every OU outcome.

The original component collision contracts its own frozen-slot velocity
energy, and the first count average contracts that prepared energy.
Thus an entering RMS at most $.56$ gives $\|U\|_2\le.56$.
Conditional centering of recipient jitter and its alive source box give
$\mathbb E|X|^2\le12.03$, irrespective of retained dead positions.
Independent fresh OU centering then proves
$$
M_w^2=c^2\mathbb E|U-tX|^2+3q^2
\le c^2(.56+t\sqrt{12.03})^2+3q^2
<(.9608)^2(.56+.02(3.469))^2+.1176<.49.
$$
This is a valid global $L^2$ triangle, with no factoring of local
velocity/displacement products. The trace degree term is nonpositive,
so $\mathcal J\le m_h\mathbb E\operatorname{tr}D_C
+at\sqrt{C_{\rm pair}}$. Combining this with (ICB.6),
$D_w\le\|w\|^2$ and the exact energy identity yields (ICB.13).
Since $m_h<1$, the remaining rational bound is
$$
\frac{(.006)(.0004)(.368)}2
 +(.006)^3(.49)+(.006)(.02)(.0392)(8.5)
=.00004053144<.000041.
$$
The six-step population burn supplies the stated entering budget.
For finite current-survivor laws, the separate finite burn applies
when its own extinction bound is at most $.01$; alternatively the
theorem may use an explicitly given entering RMS budget. Neither use
conditions a Gaussian stage on a future empirical-moment event.
:::

(sec-ica70-flux)=
## 5. Scalar friction, cross flux and own next survival

:::{prf:lemma} Nonlinear scalar and normalization verification
:label: lem-ica70-flux-survival

The scalar cap-friction consequence (ICB.15), the exact cross-flux
identity (ICB.16), and the one-next-step survivor bound of
{prf:ref}`cor-icb-cross-flux` have their stated marginals and scope.
:::

:::{prf:proof}
The trace formula gives $\operatorname{tr}D_C\le d(1-u)$.
Since $0\le u<1$, $\mathbb Eu\ge\mathbb Eu^2=r_v^2/V^2$.
Thus the scalar injection upper bound in (ICB.15) follows. The
function $b_0\mapsto b_0^{3/2}/(V-\sqrt{b_0})$ is a pointwise
increasing sum of nonnegative convex powers on $[0,V^2)$.
Jensen proves its displayed friction lower bound. Its expectation
is finite because $\phi_V(v^+)\le V|z|$ and the source and
full OU stages have finite second moments.

The final position Gaussian is independent of the stored velocity
under the raw proposal, so its cross mean vanishes. The exact landing
formula and Stein give
$$
H^+=a_x\mathbb E\langle X,v^+\rangle
 +b\mathbb E\langle U,v^+\rangle+tq^2\mathcal J.
$$
Eliminating the $X$ cross term uses exactly
$a_xz_v+br_H=c$ and
$m_h+r_Ht/a_x=m_h/a_x$. This proves (ICB.16) without
assigning a sign to either retained mixed expectation.

For a finite entering current-survivor law, the next proposal first
has the raw Gaussian identities just proved. The left side of
(ICB.13) is pointwise nonnegative. Restricting it to its own next
survival event and dividing by that event's probability gives at
most its raw expectation divided by $1-e_N$. The raw right side
is at least this nonnegative expectation, so the same division is
valid for the stated bound. Its mixed terms and Stein trace remain
raw expectations under that entering law. No Gaussian independence
is claimed after output restriction, no common survival event between
two histories is used, and no full-horizon denominator is introduced.
:::

(sec-ica70-scope)=
## 6. Accepted intermediate and unproved delayed inference

:::{prf:remark} Independent acceptance and remaining mixed flux
:label: rem-ica70-scope

The frozen source passes this independent review. Its convex cap
resolvent, realized-graph alignment, exact finite and population Stein
identities, full-Gaussian trace consumer, post-burn primitive arithmetic,
cross-flux elimination and own-next-survival normalization are complete.
Exact rational checks confirm every displayed numerical endpoint.

The remaining $U\cdot v^+$ and $X\cdot v^+$ terms are actual
mixed fluxes. Their combination is not assigned a favorable sign
by this record. An inward source account, an invariant comparison
family and a general delayed nonlinear law gap do not follow from
the cap balance alone. In particular this audit asserts neither a
default full active-update contraction nor finite quasi-stationary
mixing. The resolvent is an identity for the existing cap, not a
new integrator or a modified algorithm.
:::
