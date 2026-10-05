# Quantitative preparation estimates for the marked canonical gas

(sec-ku-selection-record)=
## Exact configuration and probability record

This proof record concerns the measurement, cloning, revival and component
collision stages of the unchanged real-coordinate canonical Viscous Euclidean
Gas. Viscosity acts in the subsequent B stages and does not alter the
preparation law here. The source definitions are `alg-euclidean-gas`,
`def-cgd-parameter-register`, `def-cgd-existing-reference`, the measurement
operator in Chapter 3, and the complete marked ledger in Chapter 6a. The
implemented branches are `Standardizer::Global`, `PositiveMap::Logistic`,
independent Gaussian companion sampling, `CloneOperator::plan`, and
`accepted_current_components` followed by `apply_component_rotations` in
`algorithmic-gas/crates/algorithmic-gas/src/`.

The proof is for the real-arithmetic law. It does not identify independent
mathematical innovations with a finite-state pseudorandom generator. All
fitness values below are retained sampled values, including complete and
partial ties. No noise truncation is made.

The parameter-to-source mapping is explicit. In `variants/euclidean.rs`,
`GasConfig::euclidean` sets both squashing radii to $2$, phase-space weight
to $1$, both Gaussian companion widths to $2$, absorbing-box endpoints to
$\pm2$, clone jitter to $0.1$, restitution to $0.5$, both global floors
to $0.1$, both logistic amplitudes/floors to $2/0.1$, weighted revival
to true, final position diffusion to $0.1$ and cap radius to $2$.
`FitnessModule::default` supplies exponents $1$ and separation floor
$10^{-3}$; `CloneDecision::default` supplies acceptance regularizer
$10^{-6}$, saturation $1$ and period $1$; `CloneTransform::default`
supplies Haar rotations. `FitnessModule::evaluate` computes the
separation as `sqrt(distance * distance + floor * floor)` before
standardizing. The default independent current donor law and self
exclusion come from `DonorModule::default` and its independent branch.
`variants/viscous_euclidean.rs` extends this configuration in the two
kinetic kicks; it does not replace any measurement or cloning field.

:::{prf:definition} Primitive preparation record
:label: def-ku-preparation-record

Let $S=((x_i,v_i,a_i))_{i=1}^N$, $a_i\in\{0,1\}$, have a nonempty
alive set $A$ of cardinality $M$. All velocities, including dead-slot
velocities, obey $|v_i|\le V$. Alive positions lie in the terminal box
$D\subset B(0,B_x)$; dead positions have no imposed bound. Set

$$
z_i=(S_{R_x}(x_i),\sqrt{\lambda_{\rm alg}}S_{R_v}(v_i)),\qquad
S_R(x)=\frac{Rx}{R+|x|},\qquad
R_*^2=R_x^2+\lambda_{\rm alg}R_v^2,\quad D_*=2R_*.
$$

For role $b=D,C$, define

$$
w^b_{ij}=\exp[-|z_i-z_j|^2/(2\epsilon_b^2)],\qquad
\kappa_b=\exp[-D_*^2/(2\epsilon_b^2)].
$$

An alive row has eligible distinct companions $A\setminus\{i\}$
when $M\ge2$. A dead row has eligible cloning donors $A$. The alive
singleton measurement uses its declared self convention; the alive
singleton cloning law persists. On a nonempty eligible set $E_i^b$,

$$
P_b(j\mid i,S)=\frac{w^b_{ij}}{Z^b_i},\qquad
Z^b_i=\sum_{k\in E_i^b}w^b_{ik},\qquad
\kappa_b|E_i^b|\le Z_i^b\le |E_i^b|.
\tag{KU.1}
$$

For independent alive measurement draws $m_i\sim P_D(\cdot\mid i,S)$,

$$
r_i=R(x_i,v_i),\qquad s_i=\sqrt{|z_i-z_{m_i}|^2+\delta_D^2},
\quad i\in A.
$$

For $b=r,s$, let

$$
\bar b=M^{-1}\sum_{i\in A}b_i,\quad
\sigma_b(b)=\sqrt{M^{-1}\sum_{i\in A}(b_i-\bar b)^2+\sigma_{b,\min}^2},
\quad Z_{b,i}=(b_i-\bar b)/\sigma_b(b),
$$

$$
G_b(t)=\frac{A_b}{1+e^{-t}}+\eta_b,\qquad
F_i=G_r(Z_{r,i})^{p_r}G_s(Z_{s,i})^{p_s}.
\tag{KU.2}
$$

The floors $\eta_b,\sigma_{b,\min},\epsilon_c$ are strictly positive;
$A_b>0$, $p_b\ge0$, and $s_c>0$. Define the primitive fitness interval

$$
F_*=\eta_r^{p_r}\eta_s^{p_s},\qquad
F^*=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},
$$

and the actual gate

$$
\mathfrak a(f,g)=\min\left\{1,\frac{(g-f)_+}{s_c(f+\epsilon_c)}\right\}.
\tag{KU.3}
$$

The measurement-vector law is exactly
$W_S(m)=\prod_{i\in A}P_D(m_i\mid i,S)$, with the singleton factor
equal to one. Products in this exact law are used only to average its
actual marks; no product event is used to obtain a minorization constant.
:::

(sec-ku-normalization)=
## Sharper normalization and acceptance estimates

:::{prf:lemma} Population-independent global-standardization operator bounds
:label: lem-ku-standardization

On a fixed alive pool of any size $M\ge1$, write
$\|b\|_{q,M}=(M^{-1}\sum_i|b_i|^q)^{1/q}$. For the actual global
standardizer $\mathcal Z_\sigma(b)=(b-\bar b)/\sqrt{\operatorname{Var}_M(b)+\sigma^2}$,

$$
\|\mathcal Z_\sigma(b)-\mathcal Z_\sigma(\widetilde b)\|_{2,M}
\le\sigma^{-1}\|b-\widetilde b\|_{2,M}.
\tag{KU.4}
$$

If the union of the two raw ranges has length at most $R$, then

$$
\|\mathcal Z_\sigma(b)-\mathcal Z_\sigma(\widetilde b)\|_{1,M}
\le\left(\frac2\sigma+\frac{2R}{3\sqrt3\,\sigma^2}\right)
\|b-\widetilde b\|_{1,M}.
\tag{KU.5}
$$

Consequently, for the retained fitness arrays on this same pool,

$$
\|F-\widetilde F\|_{2,M}
\le\sum_{b=r,s}\frac{H_b}{\sigma_{b,\min}}
                      \|b-\widetilde b\|_{2,M},
\tag{KU.6}
$$

$$
\|F-\widetilde F\|_{1,M}
\le\sum_{b=r,s}H_b\left[
\frac2{\sigma_{b,\min}}+
\frac{2R_b}{3\sqrt3\,\sigma_{b,\min}^2}\right]
\|b-\widetilde b\|_{1,M},
\tag{KU.7}
$$

where $H_b=0$ if $p_b=0$, and otherwise

$$
H_b=\frac{A_bp_b}{4}
\max\{\eta_b^{p_b-1},(A_b+\eta_b)^{p_b-1}\}
(A_{b'}+\eta_{b'})^{p_{b'}},\qquad b'\ne b.
$$

These bounds include constant raw arrays, arbitrarily small variances,
and equal fitness. They require no positive variance or fitness gap.

*Proof.* Let $P=I-\mathbf1\mathbf1^T/M$, $y=Pb$, and
$s=(\|y\|_{2,M}^2+\sigma^2)^{1/2}$. Direct differentiation gives

$$
D\mathcal Z_\sigma(b)=\frac1s
\left[P-\frac{yy^T}{Ms^2}\right].
$$

The eigenvalue in the constant direction is zero. The eigenvalue along
$y$ is $\sigma^2/s^3$ and every orthogonal centered direction has
eigenvalue $1/s$. Thus the Euclidean operator norm is at most
$1/\sigma$; the same is true for the normalized empirical norm.
Integrating along the segment proves (KU.4), including $y=0$.

For the empirical $L^1$ norm, $\|P\|_{1\to1}\le2$ and the rank-one
part has norm at most
$\max_i|y_i|\,\|y\|_{1,M}/s^3\le R\sqrt v/(v+\sigma^2)^{3/2}$,
where $v=\|y\|_{2,M}^2$. Differentiation in $v$ shows that the last
factor has maximum $2/(3\sqrt3\,\sigma^2)$ at $v=\sigma^2/2$.
Every array on the segment has range at most $R$. Integration proves
(KU.5). The derivative of $G_b$ is at most $A_b/4$. Differentiating
the two positive powers and their product gives the displayed $H_b$.
Apply the mean-value theorem and then Minkowski to obtain (KU.6)--(KU.7).
$\square$
:::

:::{prf:lemma} Saturation-aware acceptance sensitivity
:label: lem-ku-gate

On $[F_*,F^*]^2$ the gate (KU.3) satisfies

$$
|\mathfrak a(f,g)-\mathfrak a(\widetilde f,\widetilde g)|
\le L_{\rm rec}|f-\widetilde f|+L_{\rm don}|g-\widetilde g|,
$$

$$
L_{\rm don}=\frac1{s_c(F_*+\epsilon_c)},\qquad
L_{\rm rec}=\frac{1+s_c}{s_c(F_*+\epsilon_c)}.
\tag{KU.8}
$$

*Proof.* On the positive unsaturated branch,
$\partial_g\mathfrak a=1/[s_c(f+\epsilon_c)]$ and
$|\partial_f\mathfrak a|=(g+\epsilon_c)/[s_c(f+\epsilon_c)^2]$.
The same branch imposes $g-f<s_c(f+\epsilon_c)$, hence
$g+\epsilon_c<(1+s_c)(f+\epsilon_c)$. This proves both bounds there.
On the zero and saturated branches the derivatives are zero.
For fixed one coordinate the gate is continuous and piecewise $C^1$;
integrate its bounded derivative across the junctions. Change the
coordinates consecutively to obtain the two-coordinate estimate.
At $g=f$ the positive-part function is continuous, so ties cause no
exception. $\square$
:::

:::{prf:lemma} Gaussian companion sensitivity without an inverse weight-floor prefactor
:label: lem-ku-companion

Consider two feature arrays $z,\widetilde z$ in $B(0,R_*)$, on a
common eligible pool, and put $\delta_i=|z_i-\widetilde z_i|$.
For an all-alive pool of size $M\ge2$, or when averaging only over
its eligible alive measurement rows, set
$E_z=M^{-1}\sum_i\delta_i^2$. The total variation convention is
$\|P-Q\|_{\rm TV}=\tfrac12\sum_j|P_j-Q_j|$. Then

$$
\overline{\operatorname{TV}}_b
\le\min\left\{1,C_b\sqrt{E_z}\right\},\qquad
C_b=\frac{R_*+D_*\kappa_b^{-1/2}}{2\epsilon_b^2}.
\tag{KU.9}
$$

The scalar-array bound (KU.9) uses the actual normalized Gaussian
law, rather than a derivative bound for an unnormalized kernel divided
twice by its denominator.

*Proof.* Interpolate the two feature arrays linearly; the interpolation
remains in the convex ball. If $\ell_j(t)=\log w_{ij}(t)$, then
$P'_j=P_j(\ell'_j-\sum_kP_k\ell'_k)$. Consequently

$$
\frac12\sum_j|P'_j|\le\frac12
 \sqrt{\operatorname{Var}_{P_i(t)}(\ell'_j)}.
$$

Writing $\Delta z_j=\widetilde z_j-z_j$, the common term
$-z_i\cdot\Delta z_i/\epsilon_b^2$ disappears from this variance.
The remaining two terms are
$z_j\cdot\Delta z_i/\epsilon_b^2$ and
$(z_i-z_j)\cdot\Delta z_j/\epsilon_b^2$. The variance norm is
bounded by the $L^2$ norm. Minkowski therefore gives

$$
\sqrt{\operatorname{Var}(\ell')}
\le\epsilon_b^{-2}
\left[R_*\delta_i+D_*\sqrt{\sum_jP_b(j\mid i,t)\delta_j^2}\right].
$$

For distinct alive donors, (KU.1) gives
$\sum_iP_b(j\mid i,t)\le1/\kappa_b$; there are exactly $M-1$
eligible recipient rows for each $j$. Average the preceding display
and use Cauchy--Schwarz twice. Integrating in $t\in[0,1]$ proves
(KU.9). $\square$
:::

:::{prf:corollary} Raw-measurement and accepted-source envelopes
:label: cor-ku-measurement-source

Use the common all-alive pool and a rowwise maximal coupling of the
two actual measurement companion laws, independently across rows.
Put

$$
R_s=\sqrt{D_*^2+\delta_D^2}-\delta_D.
$$

The actual raw diversity arrays obey

$$
\mathbb E\|s-\widetilde s\|_{1,M}
\le(1+\kappa_D^{-1/2})\sqrt{E_z}
                         +R_s\overline{\operatorname{TV}}_D,
\tag{KU.10}
$$

$$
\mathbb E\|s-\widetilde s\|_{2,M}^2
\le2(1+\kappa_D^{-1})E_z
                         +R_s^2\overline{\operatorname{TV}}_D.
\tag{KU.11}
$$

For a configured reward $R(x,v)=-U(x)-\lambda_{\rm vel}|v|^2$, let
$L_U=\sup_{x\in D}|\nabla U(x)|$, a computed landscape profile.
For example $L_U\le B_F+L_FB_x$ when the recorded global force
profile is $|F(x)|\le B_F+L_F|x|$. With
$E_{xv}=M^{-1}\sum_i[|x_i-\widetilde x_i|^2+
\lambda_{\rm alg}|v_i-\widetilde v_i|^2]$,

$$
\|r-\widetilde r\|_{2,M}
\le\sqrt{L_U^2+(2\lambda_{\rm vel}V)^2/\lambda_{\rm alg}}
\sqrt{E_{xv}}.
\tag{KU.12}
$$

For actual frozen fitness arrays, set
$b_{ij}=P_C(j\mid i)\mathfrak a(F_i,F_j)$,
$\widetilde b_{ij}=\widetilde P_C(j\mid i)
\mathfrak a(\widetilde F_i,\widetilde F_j)$ and
$\ell_i=\sum_j|b_{ij}-\widetilde b_{ij}|$. Then

$$
\bar\ell\le\min\left\{2,
2\overline{\operatorname{TV}}_C+
(L_{\rm rec}+\kappa_C^{-1/2}L_{\rm don})
\|F-\widetilde F\|_{2,M}\right\},
\tag{KU.13}
$$

and also

$$
\bar\ell\le\min\left\{2,
2\overline{\operatorname{TV}}_C+
(L_{\rm rec}+\kappa_C^{-1}L_{\rm don})
\|F-\widetilde F\|_{1,M}\right\}.
\tag{KU.14}
$$

Average (KU.13) using (KU.6), (KU.11)--(KU.12), or average
(KU.14) using (KU.7), (KU.10), and (KU.12). These provide complete
primitive-parameter mismatch bounds. The exact retained arrays may
always be used instead of these envelopes.

*Proof.* On a common companion $j$, the map
$u\mapsto\sqrt{|u|^2+\delta_D^2}$ is $1$-Lipschitz, so the
measurement difference is at most $\delta_i+\delta_j$. On a
mismatched companion it is at most $R_s$, since each realized raw
measurement lies in $[\delta_D,\sqrt{D_*^2+\delta_D^2}]$.
Average the common part and use the donor column bound from the
previous proof. Cauchy--Schwarz gives (KU.10); squaring on the
common part and using $(a+b)^2\le2(a^2+b^2)$ gives (KU.11).
The segment between two box positions stays inside the convex box;
integrate $\nabla U$ and use
$||v|^2-|\widetilde v|^2|\le2V|v-\widetilde v|$ to get (KU.12).
Split $b-\widetilde b=(P-\widetilde P)\mathfrak a+
\widetilde P(\mathfrak a-\widetilde{\mathfrak a})$. The first
summed term is at most twice the candidate TV. Apply (KU.8) to the
second. Its donor-weighted average is at most
$\kappa_C^{-1/2}\|F-\widetilde F\|_{2,M}$ by Cauchy--Schwarz,
or $\kappa_C^{-1}\|F-\widetilde F\|_{1,M}$ directly. Each accepted
measure has mass at most one, yielding the clipping at two.
$\square$
:::

(sec-ku-marked-source)=
## Mandatory revival and the exact marked source plan

:::{prf:theorem} Exact single-population cloning variance with retained dead coordinates
:label: thm-ku-marked-one-population

For any nonextinct entering marked swarm, let
$\bar x=N^{-1}\sum_i x_i$, $e_i=|x_i-\bar x|^2$, and
$W_N=N^{-1}\sum_i e_i$, including the retained positions of dead slots.
Condition on the actual sampled alive fitnesses and define

$$
b_{ij}=\begin{cases}
P_C(j\mid i)\mathfrak a(F_i,F_j),&a_i=1,\ M\ge2,\ j\in A\setminus\{i\},\\
P_C(j\mid i),&a_i=0,\ j\in A,\\
0,&\text{otherwise}.
\end{cases}
$$

Let $p_i=\sum_jb_{ij}$, so every dead slot has $p_i=1$.
Set

$$
t_i=\sum_jb_{ij}(x_j-x_i),\quad
\sigma_i^2=\sum_jb_{ij}|x_j-x_i|^2-|t_i|^2+d\sigma_J^2p_i,
$$

$$
A_{\rm rec}=\frac1N\sum_ip_ie_i,\quad
D_{\rm donor}=\frac1N\sum_{ij}b_{ij}e_j,\quad
\bar p=\frac1N\sum_ip_i,\quad\bar t_x=\frac1N\sum_it_i.
$$

Then $\sigma_i^2\ge0$ and, for the unchanged actual copying/jitter proposal,

$$
\mathbb E[W_N(S^C)-W_N(S)\mid S,F]
=-A_{\rm rec}+D_{\rm donor}+d\sigma_J^2\bar p
-|\bar t_x|^2-\frac1{N^2}\sum_i\sigma_i^2.
\tag{KU.14a}
$$

Thus mandatory revival removes the dead recipient's exact retained
radial square and inserts its selected alive donor's radial square.
No arbitrary reset coordinate or bounded-dead-coordinate substitution
is present. The outer average over $W_S(m)$ gives the actual
unconditional preparation drift.

*Proof.* Each proposal row is its frozen position with probability
$1-p_i$, or its frozen donor position plus its own Gaussian jitter
with subprobability $b_{ij}$. This description applies to the mandatory
revival branch because $p_i=1$ there. Its mean is $x_i+t_i$ and its
variance trace is exactly $\sigma_i^2$, proving nonnegativity.
The proposal rows are independent conditional on the complete frozen
fitnesses. Therefore the output barycenter's variance trace is
$N^{-2}\sum_i\sigma_i^2$. Compute the second moment about the
entering $\bar x$, whose change is
$-A_{\rm rec}+D_{\rm donor}+d\sigma_J^2\bar p$, and subtract the
new barycenter's squared displacement and variance. This gives
(KU.14a). Collision changes only velocities. The input dead coordinates
were finite but otherwise arbitrary, so no tail restriction is used
in this identity. $\square$
:::

:::{prf:definition} Source tokens that retain the jitter and edge marks
:label: def-ku-source-token

A source token is $\tau=(e,j)$, where $e\in\{0,1\}$ records copying
and $j$ records the frozen source label. Given $S$ and its complete
retained fitness vector, the row-token law is

$$
q_i(0,i)=1-p_i,\qquad
q_i(1,j)=P_C(j\mid i,S)\mathfrak a(F_i,F_j),\quad j\in A\setminus\{i\},
\quad a_i=1, M\ge2,
\tag{KU.15}
$$

where $p_i=\sum_jq_i(1,j)$. For an alive singleton,
$q_i(0,i)=1$. For each dead row,

$$
q_i(1,j)=P_C(j\mid i,S),\quad j\in A,\qquad q_i(0,j)=0.
\tag{KU.16}
$$

In particular $e=1$ surely on every revived row. Couple two row-token
laws by

$$
\lambda_i(\tau)=\min\{q_i(\tau),\widetilde q_i(\tau)\},\qquad
t_i=1-\sum_\tau\lambda_i(\tau),
$$

$$
\Pi_i(\tau,\widetilde\tau)
=\lambda_i(\tau)\mathbf1_{\tau=\widetilde\tau}
+\frac{[q_i(\tau)-\lambda_i(\tau)]
        [\widetilde q_i(\widetilde\tau)-\lambda_i(\widetilde\tau)]}{t_i},
\tag{KU.17}
$$

with the fraction zero when $t_i=0$. Use these paired plans
independently across rows conditional on the two retained fitness
vectors. Share one row jitter $\zeta_i\sim N(0,I_d)$ across swarms,
independently of all row plans. The actual prepared position difference is

$$
r_i=x_j-\widetilde x_k+\sigma_J(e-\widetilde e)\zeta_i,
\qquad \tau_i=(e,j),\ \widetilde\tau_i=(\widetilde e,k).
\tag{KU.18}
$$

The accepted graph in each marginal uses exactly the edges $(i,j)$
of tokens $(1,j)$. This token construction applies to unequal alive
sets and opposite marks at a label. In the unchanged canonical
configuration every accepted donor differs from its recipient:
live self cloning has zero gate and a dead slot cannot donate to
itself. Thus $e=\mathbf1_{j\ne i}$ and the acceptance mark can also
be recovered from its source label. The explicit token notation
records that fact together with the revival branch and jitter law;
it does not postulate an additional same-source jitter mismatch.
:::

:::{prf:theorem} Exact marked positional preparation and source load
:label: thm-ku-marked-preparation

Let $d_i=x_i-\widetilde x_i$, $\bar d=N^{-1}\sum_i d_i$ and
$D=N^{-1}\sum_i|d_i-\bar d|^2$. For the token law (KU.17), define

$$
\mu_i=\sum_{\tau=(e,j),\widetilde\tau=(\widetilde e,k)}
                  \Pi_i(\tau,\widetilde\tau)(x_j-\widetilde x_k),
\qquad h_i=\mu_i-d_i,
$$

$$
V_i=\sum_{\tau,\widetilde\tau}\Pi_i(\tau,\widetilde\tau)
       [|x_j-\widetilde x_k|^2+d\sigma_J^2(e-\widetilde e)^2]
       -|\mu_i|^2\ge0.
$$

The complete uncentered positional balance is

$$
\frac1N\sum_i\mathbb E|r_i|^2-\frac1N\sum_i|d_i|^2
=\frac1N\sum_{i,\tau,\widetilde\tau}\Pi_i
 [|x_j-\widetilde x_k|^2-|d_i|^2
                   +d\sigma_J^2(e-\widetilde e)^2].
\tag{KU.19}
$$

The centered balance is the right side of (KU.19), minus

$$
2\bar d\cdot\bar h+|\bar h|^2+\frac1{N^2}\sum_iV_i,
\qquad \bar h=N^{-1}\sum_i h_i.
\tag{KU.20}
$$

All source positions appearing in (KU.18) are alive frozen positions.
Thus, without bounding any retained dead position,

$$
\frac1N\sum_i\mathbb E|r_i|^2
\le4B_x^2+d\sigma_J^2\bar t,
\qquad \bar t=N^{-1}\sum_it_i.
\tag{KU.21}
$$

If $Q_{ij}=\sum_eq_i(e,j)$ denotes the actual marginal source
matrix, then for each eligible alive label $j$,

$$
\sum_iQ_{ij}\le1+\frac1{\kappa_C}
                         +\frac{N-M}{\kappa_CM}
=1+\frac{N}{\kappa_CM}.
\tag{KU.22}
$$

For $M=1$, the sharper exact column sum is $N$. With
$P=N^{-1}\sum_i|d_i|^2$ and
$\Lambda=\min\{1+N/(\kappa_CM),1+N/(\kappa_C\widetilde M)\}$,

$$
\frac1N\sum_i\mathbb E|r_i|^2
\le\Lambda P+(4B_x^2+d\sigma_J^2)\bar t.
\tag{KU.23}
$$

Both (KU.21) and (KU.23) are valid; use their minimum. The exact
identity (KU.19)--(KU.20) is sharper when negative donor flux and
barycenter corrections matter.

*Proof.* The marginal sums of (KU.17) are $q_i$ and
$\widetilde q_i$; its residual supports cannot intersect at a common
token. Conditional Gaussian averaging in (KU.18) proves (KU.19)
and $V_i\ge0$. The row differences are conditionally independent,
so the output empirical mean has second moment
$|\bar d+\bar h|^2+N^{-2}\sum_iV_i$. Subtract this from the
uncentered square and subtract the entering mean square to obtain
(KU.20). The subtraction keeps the actual shared-normalizer
correlations through the outer measurement average.

Every persistent row is alive, and every accepted source is alive.
Consequently $|x_j-\widetilde x_k|\le2B_x$. Common tokens have
$e=\widetilde e$ and only residual mass $t_i$ can incur jitter cost.
This proves (KU.21) with unbounded Gaussian jitter integrated exactly.
For (KU.22), persistence contributes at most one. For $M\ge2$,
the other $M-1$ alive recipients contribute at most $1/\kappa_C$
in total, by their $(M-1)$-donor denominator. Every dead recipient
contributes at most $1/(\kappa_CM)$. When $M=1$, the alive row
persists and all $N-1$ dead rows necessarily copy it. For (KU.23),
common-token positions equal $d_j$ and have incoming column mass
bounded by each marginal's (KU.22). Residual pairs have total mass
$t_i$ and squared positional cost at most $4B_x^2$. Sum these
terms and their exact jitter costs. $\square$
:::

(sec-ku-collision)=
## Shared-component Haar rotation and graph sensitivity

:::{prf:theorem} Exact marked collision moments and sharper population-independent path bound
:label: thm-ku-marked-collision

For each paired token plan, build the two actual undirected accepted
graphs. In each marginal let $C_i$ be row $i$'s component, interpreting
an isolated row as a singleton. Let

$$
m_i=|C_i|^{-1}\sum_{j\in C_i}v_j,\qquad b_i=v_i-m_i,
$$

and define $\widetilde m_i,\widetilde b_i$ similarly, always from
the frozen slot velocities, including retained dead-slot velocities.
Couple identical components with the same Haar orthogonal matrix;
use independent Haar matrices for all unmatched components. Then,
with $z_i^C=V_i^C-\widetilde V_i^C$ and restitution $\alpha_c$,

$$
\mathbb E_Rz_i^C=m_i-\widetilde m_i,
$$

$$
\mathbb E_R|z_i^C|^2=|m_i-\widetilde m_i|^2+
\alpha_c^2[|b_i|^2+|\widetilde b_i|^2
-2\mathbf1_{C_i=\widetilde C_i}b_i\cdot\widetilde b_i],
\tag{KU.24}
$$

$$
\mathbb E_R[r_i\cdot z_i^C]=r_i\cdot(m_i-\widetilde m_i).
\tag{KU.25}
$$

The exact component mismatch probability is the finite-plan expression

$$
\theta_N=\sum_{\boldsymbol\tau,\widetilde{\boldsymbol\tau}}
\left[\prod_i\Pi_i(\tau_i,\widetilde\tau_i)\right]
\frac1N\sum_i\mathbf1_{C_i\ne\widetilde C_i}.
\tag{KU.26}
$$

Set

$$
T_M=\begin{cases}1,&M=1,\\e^{2/\kappa_C},&M\ge2,\end{cases}
\qquad \mathcal C_M=T_M\left[1+\frac{N-M}{\kappa_CM}\right].
$$

Without a positive alive-fraction assumption,

$$
\theta_N\le\min\{1,6(\mathcal C_M+\mathcal C_{\widetilde M})\bar t\}.
\tag{KU.27}
$$

For two all-alive inputs this sharpens the earlier universal path
envelope to $\theta_N\le\min\{1,6e^{2/\kappa_C}\bar t\}$.
For $A_c=\max(1,\alpha_c^2)$ and
$V_c=(1+2|\alpha_c|)V$,

$$
\frac1N\sum_i\mathbb E|z_i^C|^2
\le A_c\frac1N\sum_i|v_i-\widetilde v_i|^2+4V_c^2\theta_N.
\tag{KU.28}
$$

The large exponential in (KU.27) is an upper envelope; (KU.26)
is the exact quantity and includes the correlation between a
mismatched row and the components it changes.

*Proof.* The component rule is
$V_i^C=m_i+\alpha_cR_{C_i}b_i$. Haar invariance gives
$\mathbb ER_C=0$ and $R_C^TR_C=I$. Independent unmatched
rotations have zero cross expectation, whereas a shared matched
rotation preserves $b_i\cdot\widetilde b_i$. Positions and
jitters are independent of rotations. Expansion gives (KU.24)--(KU.25).
This also holds in dimension one, where Haar $O(1)$ is a uniform
sign, and for singletons where $b_i=0$.

Condition on retained fitnesses and on one row's mismatched token
pair. Other rows remain independent in each marginal. Each accepted
alive edge points to strictly larger fitness and each alive vertex
has at most one outgoing edge. The alive graph is a forest. On a
simple undirected path the orientations must first rise in fitness
and then fall: a valley would give its vertex two outgoing edges.
For a seed alive vertex, a path with $l$ other alive vertices and
rising-leg length $a$ has at most
$\binom{M-1}{l}\binom la$ candidate label arrangements, since
fitness fixes the order on each leg. Tied fitnesses can only reduce
this count. Its $l$ distinct recipient rows have joint edge
probability at most $[\kappa_C(M-1)]^{-l}$. Hence the expected
path count is at most

$$
\sum_{l=0}^{M-1}\sum_{a=0}^l
\frac{\binom{M-1}{l}\binom la}{[\kappa_C(M-1)]^l}
\le\sum_{l=0}^\infty\frac{(2/\kappa_C)^l}{l!}
=e^{2/\kappa_C}.
$$

If the fixed mismatched row contributes an edge, seed both of its
endpoints. A path to any other vertex can then avoid that fixed
edge, so its remaining rows still have the independent probabilities
just counted. Every dead vertex is a leaf attached to an alive
donor. Conditional on an alive component with $L$ vertices,
unfixed dead rows attach to it with total expected count at most
$L(N-M)/(\kappa_CM)$. This proves $\mathcal C_M$ as a seeded
component bound. If $M=1$, there are no alive edges and the same
bound follows directly for the revival star.

A token mismatch at row $i$ changes at most the two accepted
edges $(i,j),(i,k)$. Their endpoints comprise at most three
labels. Every vertex whose component changes lies in a component,
in one marginal, of one of these endpoints. Seed those endpoints
in both graphs and apply the preceding conditional bound. A dead
seed whose donor row is not the fixed mismatched row has its own
random donor. Its expected component size is at most
$1+T_M[1+(N-M-1)/(\kappa_CM)]\le2\mathcal C_M$:
the seed contributes one, its donor's alive component contributes
at most $T_M$, and other dead leaves contribute the remaining term.
For a fixed dead row its donor endpoint is already among the seeds;
the same upper bound is still valid. Thus each of the three
arbitrary seeds costs at most $2\mathcal C_M$ in a marked graph.
When both graphs are all alive, each seed instead costs $T_N$,
giving the sharper $6e^{2/\kappa_C}$ coefficient stated above.
Union bounding across mismatch rows, then
using $\sum_i\mathbb P(\tau_i\ne\widetilde\tau_i)=N\bar t$,
proves (KU.27). No mismatch/component-size expectations have
been multiplied. Finally, on matched components, sum (KU.24)
and use the orthogonal mean/deviation decomposition to get
$A_c\sum|v_i-\widetilde v_i|^2$. On unmatched components
$|V_i^C|\le V_c$, so the discrepancy is at most $2V_c$.
Average to prove (KU.28). $\square$
:::

(sec-ku-signed-positive)=
## Signed pressure, computable inward regimes, and a non-tie diagnostic

:::{prf:theorem} Exact marked quadratic preparation ledger with Keystone pressure retained
:label: thm-ku-preparation-keystone

Choose metric coefficients $\alpha>0$, $\gamma_P>0$,
$\alpha\gamma_P>\beta^2$ and $\lambda_a\ge0$. For
$d_i=x_i-\widetilde x_i$, $u_i=v_i-\widetilde v_i$, put

$$
\mathscr Q_{\rm mark}(S,\widetilde S)=\frac1N\sum_i
[\alpha|d_i|^2+2\beta d_i\cdot u_i+\gamma_P|u_i|^2
 +\lambda_a(a_i-\widetilde a_i)^2].
$$

For a paired token plan let $\delta_i=x_{j_i}-\widetilde x_{k_i}$,
$\Delta m_i=m_i-\widetilde m_i$ and define

$$
\mathcal V_i=|\Delta m_i|^2+\alpha_c^2
[|b_i|^2+|\widetilde b_i|^2
-2\mathbf1_{C_i=\widetilde C_i}b_i\cdot\widetilde b_i].
$$

Then, before kinetics and terminal marking, the exact conditional
preparation increment equals

$$
\begin{aligned}
\mathscr P_N={}&\sum_{\boldsymbol\tau,\widetilde{\boldsymbol\tau}}
\left[\prod_i\Pi_i(\tau_i,\widetilde\tau_i)\right]\frac1N\sum_i
\Big\{\alpha[|\delta_i|^2-|d_i|^2
             +d\sigma_J^2(e_i-\widetilde e_i)^2]\\
&\qquad+2\beta[\delta_i\cdot\Delta m_i-d_i\cdot u_i]
 +\gamma_P[\mathcal V_i-|u_i|^2]
 -\lambda_a(a_i-\widetilde a_i)^2\Big\}.
\end{aligned}
\tag{KU.28a}
$$

Here $e_i$ in the token-plan formula is the copying indicator,
not the radial error used below. The states after preparation are
all alive, hence the last term is the exact removal of the entering
status mismatch. Conditional jitter is unbounded but contributes
exactly its displayed second moment and has zero cross expectation
with collision velocity.

Let $\mathcal A_{11}$ be the actual common-alive error-weighted
activity of `thm-slc-marked-keystone-port`, computed from the same
source arrays and alive-centered errors. After averaging (KU.28a)
over its actual paired measurement law, define the computed finite
remainder
$\mathscr R_N^C=\mathbb E_m\mathscr P_N+\alpha\mathcal A_{11}/2$.
For the source Keystone constants $\chi_*,W_0,B_*$, exactly as
computed in `thm-keystone-discharged-averaged-pressure`,

$$
\mathbb E\Delta\mathscr Q_{\rm mark}
\le-\frac\alpha2\chi_*(W-W_0)
 +\frac{\alpha B_*}{2N^2}+\mathscr R_N^C.
\tag{KU.28b}
$$

This places the discharged population-independent Keystone pressure
inside the complete marked preparation computation. Its remainder
is a prescribed finite weighted sum, not an assumed contraction
constant. Composing with the signed kinetic and terminal ledger
adds those actual computed increments to the same $\mathscr R_N^C$.

*Proof.* First integrate the shared row jitter in (KU.18): the
positional second moment is $|\delta_i|^2+
d\sigma_J^2(e_i-\widetilde e_i)^2$. Integrate the component Haar
rotations with (KU.24)--(KU.25). The jitter's mean is zero and it
is independent of rotations, so the cross term becomes exactly
$\delta_i\cdot\Delta m_i$. Subtract the entering metric, including
its status mismatch, and average the independent conditional
row-token plans. This is (KU.28a). Its outer measurement law
retains the complete sampled normalizers. The definition of
$\mathscr R_N^C$ is then an exact addition and subtraction of
$\alpha\mathcal A_{11}/2$. Apply the source Keystone lower bound
$\mathcal A_{11}\ge\chi_*(W-W_0)-B_*/N^2$ to get (KU.28b).
The source errors are alive-centered; none has been silently
replaced by the full-slot positional or velocity metric.
$\square$
:::

:::{prf:proposition} A complete inward radial donor calculation
:label: prop-ku-inward-donor

For an all-alive input, set $e_i=|x_i-\bar x|^2$ and use the actual
retained fitness array. The exact signed radial donor contribution is

$$
\mathcal D(F)=\frac1N\sum_{i,j}P_C(j\mid i)
                          \mathfrak a(F_i,F_j)(e_j-e_i).
\tag{KU.29}
$$

This is the donor term minus recipient pressure; no absolute values
are taken. If $e_i>e_j$, define the computed radial fitness slopes
$m_{ij}=(F_j-F_i)/(e_i-e_j)$ and
$m_F=\min_{e_i>e_j}m_{ij}$. If no such pairs exist, put $m_F=0$.
When its computed value is nonnegative, with
$E_e=\max_i e_i-\min_i e_i$ and
$K_F=\max\{s_c(F^*+\epsilon_c),m_FE_e\}$,

$$
\mathcal D(F)\le-\frac{\kappa_Cm_F}{K_F}
\frac{N}{N-1}\operatorname{Var}_N(e).
\tag{KU.30}
$$

The actual cloning variance drift consequently satisfies

$$
\mathbb E[W_N(S^C)-W_N(S)\mid S,F]
\le-\frac{\kappa_Cm_F}{K_F}\operatorname{Var}_N(e)
+d\sigma_J^2\bar p-|\bar t_x|^2-
\frac1{N^2}\sum_i\sigma_i^2,
\tag{KU.31}
$$

where $\bar t_x,\sigma_i^2$ are the actual mean displacement and
row variance from the exact Chapter 6a balance. All coefficients
are population independent. Formula (KU.29) remains the applicable
test when $m_F<0$; the proof does not posit monotone fitness.
The measurement average of (KU.29), rather than a fitness vector
averaged in advance, is the unconditional donor calculation.

*Proof.* If $e_i>e_j$ and $m_F\ge0$, then $F_j\ge F_i$,
so only the inward gate can be positive. The acceptance is at least
$m_F(e_i-e_j)/K_F$: both its unsaturated expression and one
exceed this quantity by the definition of $K_F$.
Use $P_C(j\mid i)\ge\kappa_C/(N-1)$, group unordered pairs,
and apply
$\sum_{i<j}(e_i-e_j)^2=N^2\operatorname{Var}_N(e)$.
This proves (KU.30). The exact variance balance is
$\mathcal D(F)+d\sigma_J^2\bar p-|\bar t_x|^2-
N^{-2}\sum_i\sigma_i^2$, proving (KU.31).
All-alive singleton cloning has zero drift and is handled separately.
$\square$
:::

:::{prf:proposition} Positive sampled activity does not remove signed donor cancellation
:label: prop-ku-two-site-diagnostic

Take canonical parameters, even $N=2n\ge4$, zero input velocities,
and $n$ walkers at each of $-ae_1,+ae_1$, where $0<a\le2$.
Use the unchanged quadratic objective $U(x)=|x|^2/2$ and reward
$-U$. Set

$$
\Delta_z=\frac{4a}{2+a},\quad w=\exp[-\Delta_z^2/8],\quad
Z=n-1+nw,\quad\vartheta_N=\frac{nw}{Z},\quad
\Delta_s=\sqrt{\Delta_z^2+10^{-6}}-10^{-3}.
$$

The diversity high/low measurement indicators are independent
Bernoulli $\vartheta_N$. If their empirical high fraction is $T$,
the shared raw scale and the two standardized values are exactly

$$
\widehat\sigma_s(T)=\sqrt{T(1-T)\Delta_s^2+0.01},\quad
z_H(T)=\frac{(1-T)\Delta_s}{\widehat\sigma_s(T)},\quad
z_L(T)=\frac{-T\Delta_s}{\widehat\sigma_s(T)}.
$$

All rewards standardize to zero, so their factor is $1.1$.
For $G(z)=2/(1+e^{-z})+0.1$, define

$$
A(T)=\min\left\{1,
\frac{G(z_H(T))-G(z_L(T))}{G(z_L(T))+10^{-6}/1.1}\right\}.
$$

The exact measurement-averaged cloning activity is the finite sum

$$
\bar p_N=\vartheta_N(1-\vartheta_N)
\sum_{k=0}^{N-2}\binom{N-2}{k}\vartheta_N^k
(1-\vartheta_N)^{N-2-k}A((k+1)/N)>0.
\tag{KU.32}
$$

For every measurement assignment, $D_{\rm donor}=A_{\rm rec}$,
because all $e_i=a^2$. Nevertheless the actual gate is positive on
mixed high/low marks. The cloning variance obeys the explicit lower bound

$$
\mathbb E\Delta W_N\ge d\sigma_J^2\bar p_N
-\frac{a^2w^2N\vartheta_N(1-\vartheta_N)}{Z^2}
-\frac{(4a^2+d\sigma_J^2)\bar p_N}{N}.
\tag{KU.33}
$$

For the unchanged reference $N=200,d=3,a=1,\sigma_J=0.1$, direct
evaluation gives

$$
\vartheta_N=0.4471551225838822,\quad
\bar p_N=0.2472074189308783,\quad
\mathbb E\Delta W_N\ge0.0014464219616448>0.
$$

This is a preparation-stage diagnostic with positive averaged
selection, not an obstruction based on equal or near-equal fitness.
The subsequent quadratic force and viscosity can restore the complete
update; their signed kinetic calculation must be included before a
complete-update sign is asserted.

*Proof.* In either site, same-site donors have feature weight one
and opposite-site donors weight $w$. This gives the displayed
Bernoulli probability and independent high/low marks. Conditional
on one recipient being low and one specified donor being high,
the remaining $N-2$ indicators have their original independent
Bernoulli law. Both these labels are distinct, and the donor law
is independent of measurement draws. Summing its normalized row
weights gives (KU.32), retaining the shared normalizer through
$A((k+1)/N)$. Every summand is positive because $\Delta_s>0$;
the all-equal mark assignments remain in the law with zero activity.

Let $H_+,H_-$ be the high counts at the two sites. The conditional
mean center displacement is exactly
$\bar t_x=awA(T)(H_+-H_-)/Z$. Since $A(T)\le1$ and
$\mathbb E(H_+-H_-)^2=N\vartheta_N(1-\vartheta_N)$,
its square expectation is at most the middle term in (KU.33).
The row variance is at most
$\sigma_i^2\le(4a^2+d\sigma_J^2)p_i$. In the exact variance
balance the radial donor and recipient terms cancel identically.
Substitute these two upper bounds for the negative center terms.
The numerical values follow from the finite binomial sum in
(KU.32), without sampling or replacing the realized standardizer
by its limit. $\square$
:::

(sec-ku-reference-ledger)=
## Reference parameter evaluation and scope

:::{prf:definition} Canonical quantitative preparation ledger
:label: def-ku-reference-ledger

For the original $d=3$ reference, $B_x=2\sqrt3$, $V=2$,
$R_x=R_v=2$, $\lambda_{\rm alg}=1$, $\epsilon_D=\epsilon_C=2$,
$\sigma_{r,\min}=\sigma_{s,\min}=0.1$,
$A_r=A_s=2$, $\eta_r=\eta_s=0.1$, $p_r=p_s=1$,
$s_c=1$ and $\epsilon_c=10^{-6}$. The following are direct
evaluations of the proved primitive formulas:

| Quantity | Formula | Value |
|---|---|---:|
| Feature diameter | $D_*=2\sqrt8$ | $5.656854249492381$ |
| Weight floor | $\kappa_D=\kappa_C=e^{-4}$ | $0.01831563888873418$ |
| Fitness interval | $[F_*,F^*]$ | $[0.01,4.41]$ |
| Logistic-product derivative | $H_r=H_s$ | $1.05$ |
| Fitness empirical $L^2$ coefficient per raw channel | $H_b/\sigma_b$ | $10.5$ |
| Gate donor slope | $L_{\rm don}$ | $99.99000099990002$ |
| Gate recipient slope | $L_{\rm rec}$ | $199.98000199980004$ |
| Gaussian averaged TV coefficient | $C_D=C_C$ | $5.578405064714954$ |
| Diversity raw range | $R_s$ | $5.655854337880727$ |
| Reward raw range for the quadratic box objective | $R_r=2d$ | $6$ |
| Reward empirical $L^1$ coefficient | (KU.7) | $263.4871130596428$ |
| Diversity empirical $L^1$ coefficient | (KU.7) | $249.5786317130925$ |
| Gate $L^2$ incoming coefficient | $L_{\rm rec}+\kappa_C^{-1/2}L_{\rm don}$ | $938.8117287201932$ |
| Revived-position residual cost | $4B_x^2+d\sigma_J^2$ | $48.03$ |
| Collision output envelope | $V_c$ | $4$ |
| Universal alive component exponent | $2/\kappa_C$ | $109.1963000662885$ |
| Base-ten logarithm of the component envelope | $2/(\kappa_C\log10)$ | $47.4233505630408$ |

The component envelope is finite and population independent but is
quantitatively pessimistic. Actual finite-plan component sums (KU.26)
retain their correlations and can be much smaller.

For random entering alive pools, the revival terms are the actual
expectations $\mathbb E[(N/M)\bar t;M>0]$ and analogous paired
quantities. They must be bounded with the proved inverse-alive-mass
estimates of the same terminal law. A marginal estimate for
$\mathbb E[N/M]$ cannot be multiplied by an estimate for
$\mathbb E\bar t$ without an additional justified conditional
bound. No deterministic alive-fraction floor has been assumed here.
:::

:::{prf:remark} What these estimates discharge
:label: rem-ku-selection-scope

The actual measurement and source arrays, global normalization,
nonlinear gate, mandatory revival, conditional Gaussian jitter,
frozen dead velocities, full accepted components, component Haar
correlation, barycenter variance and self-exclusion denominators
have been explicitly retained. These estimates insert into the
existing marked Keystone full-update identity; they do not replace
its recipient pressure by an unsigned global selection Lipschitz
constant. A positive complete-update regime is established only
after its actual donor flux, force, viscous kicks, second-force
Gaussian excursions, cap and terminal indicators have been evaluated
in that identity.

The universal normalization and graph coefficients do not prove a
global small-discrepancy contraction. In particular the coarser
diversity $L^2$ envelope has a fourth-root modulus after measurement
mismatch is averaged; (KU.7), (KU.10) provide a first-order modulus
at the price of larger explicit coefficients. Neither upper envelope
can be given a favorable sign merely by renaming it a remainder.

The estimates concern a preparation kernel or one complete update
when composed with the kinetic record. They do not themselves prove
stationary dependence, the full-swarm QSD eigenfunction ratio, a
population-uniform entropy constant, an indefinitely invariant
all-alive phase, or uniqueness of a population stationary law.
Full-swarm conditioning and fixed sampled-row conclusions remain
different targets. No failure or extra exclusion has been introduced
for ties: all ties are kept, and their gates are computed as zero.
:::
