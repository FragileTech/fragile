# Physical-time correlations of the existing edge Hamiltonian

(sec-native-pt-parameters)=
## Complete parameters and the two existing evolutions

:::{prf:definition} Complete physical-time data
:label: def-native-pt-ledger

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`. In particular, retain all
three position coordinates, the executed update order, masks, fitness and
donor rules, both viscous force evaluations and their normalization, every
landscape provider, cloning and collision parameters, the uncapped noise law,
configured caps, arithmetic and seed convention, geometry feedback, initial
law and recording stages. No position coordinate is converted into time.

For a constant executed step $h>0$, the algorithmic clock is $\tau_n=nh$.
If the declared time calibration is $c_\tau>0$, its physical step is
$a=c_\tau h$. For a recorded subsequence use its actual absolute step
numbers: the interval is
$a(n_{j+1}-n_j)$, including a short final recording interval. These data are
stored as `delta_t`, `recorded_steps` and `record_every` in `RunHistory`.
For a variable step schedule retain its actual accumulated clock instead.

The existing finite edge readout of
{prf:ref}`def-lqft-edge-spectral-parameters` supplies, for a realized record
$Y$, the finite mode space $E(Y)$, its declared basis, coefficient recipe
$K(Y)$, units and Hermitian matrix
$h_{\rm e}(Y)=i(K(Y)-K(Y)^{\mathsf T})$. Retain every field of the actual
readout configuration; in the implemented Dirac diagnostic these include
`mc_time_index`, `epsilon_clone`, `kernel_mode`, `epsilon_c`, `lambda_alg`,
`h_eff`, `include_phase`, `n_generations`, `color_threshold`,
`min_sector_size`, `svd_top_k`, `time_average`, `time_range`,
`warmup_fraction` and `max_avg_frames`, with their actual frame and alive
selection. Any denominator patch and projected-sector recipe are retained.
The real `fitness_ratio` matrix is skew whenever its returned entries are
finite: its numerator is antisymmetric and its patched denominator symmetric.
A different returned matrix must satisfy its own actual Hermitian test.
The results below concern the already declared Hermitian edge operator.

Write

$$
H=d\Gamma(h_{\rm e}),\qquad
H_0=H-E_{\rm sea}I,\qquad T(t)=e^{-tH_0},
$$

with the existing filled ground space and projection $P_0$. Choose a
normalized fixed-parity vector $\Omega_0\in\operatorname{Ran}P_0$.
This is an existing ground vector of $H$, not the empty exterior vacuum
unless that vacuum is a ground. $H_0$ has units of inverse physical time
after the declared conversion; the physical energy operator is
$\hbar_{\rm eff}H_0$ when $\hbar_{\rm eff}>0$.

The complete native conservative transition, when a conservative stationary
law has been established for the chosen execution, remains
$C=P|_{L^2_0(\pi)}$. A finite survival window instead uses its actual
time-dependent selected transitions. Neither is defined here by the edge
matrix. The statements about $T$ hold for every well-defined finite
Hermitian edge readout, without an LSI, stationary chaos or extra premise
about the sampling law.
:::

(sec-native-pt-ground-correlations)=
## Filled-ground correlations in algorithmic time

:::{prf:theorem} Positive-energy edge correlations and time reflection
:label: thm-native-pt-edge-reflection

Fix the complete data and realized edge operator of
{prf:ref}`def-native-pt-ledger`. For bounded even CAR operators
$A_1,\ldots,A_r$ and grid times
$0\le t_1\le\cdots\le t_r$, $t_j=a n_j$, define the positive-time word
and its boundary vector by

$$
W=((t_1,A_1),\ldots,(t_r,A_r)),\qquad
\Psi(W)=T(t_1)A_1T(t_2-t_1)A_2\cdots
                    T(t_r-t_{r-1})A_r\Omega_0.
$$

For any finite list $W_i$, the edge correlation obtained by reflecting
the clock $t\mapsto-t$, reversing the operator order and taking adjoints
on the reflected word is exactly

$$
Q^{\rm e}_{ij}=\langle\Psi(W_i),\Psi(W_j)\rangle.
$$

It is a positive Hermitian matrix. For single insertions this gives the
explicit algorithmic-time Hankel matrix

$$
Q^{\rm e}_{ij}
=\langle\Omega_0,A_i^*T(a(n_i+n_j))A_j\Omega_0\rangle\succeq0.
$$

The same construction with an additional interval $a\ge0$ across the
reflection plane is positive:
$\langle\Psi(W_i),T(a)\Psi(W_j)\rangle\succeq0$.
Thus both reflection through a time slice and through the middle of a time
link hold for these existing edge correlations.

The cyclic Hilbert space
$\mathcal H_{\rm e}=\overline{\mathfrak A_{\rm ev}(E)\Omega_0}$ is
the fixed-parity block of $\Omega_0$. It is invariant under the existing
unitaries $U(t)=e^{itH_0}$, fixes $\Omega_0$, and implements
$\alpha_t(A)=e^{itH}Ae^{-itH}$ there. Its energy spectrum is nonnegative.
For every bounded $A,B$,

$$
G_{A,B}(t)=\langle A\Omega_0,T(t)B\Omega_0\rangle
=\int_{[0,\infty)}e^{-tE}\,d\eta_{A,B}(E),
$$

where $\eta_{A,B}(B')=\langle A\Omega_0,
\mathbf1_{B'}(H_0)B\Omega_0\rangle$. For every finite observable list,
$[\eta_{A_i,A_j}(B')]$ is positive Hermitian for each Borel set $B'$.

When $h_{\rm e}$ has nonzero spectrum, the connected correlation above
the complete ground space has the exact bound

$$
\left|\langle A\Omega_0,T(t)(I-P_0)B\Omega_0\rangle\right|
\le e^{-\delta_{\rm edge}t}
\|(I-P_0)A\Omega_0\|\|(I-P_0)B\Omega_0\|.
$$

Subtracting only the expectation in a chosen ground vector does not remove
other zero-energy ground components when the ground is degenerate.
:::

:::{prf:proof}
The previously proved occupation spectrum gives $H_0\ge0$,
$T(s)T(t)=T(s+t)$, $T(t)^*=T(t)$, $\|T(t)\|\le1$ and
$T(t)\Omega_0=\Omega_0$. Expand the adjoint of $\Psi(W_i)$.
The operators on its left are in reverse order and adjointed, and each
time interval is reflected. At the join, the two adjacent transfers are
$T(t_{i,1})T(t_{j,1})=T(t_{i,1}+t_{j,1})$. This is precisely the
time-ordered reflected correlation. All operators are even, so this
adjoint and gluing calculation introduces no odd permutation sign.

For coefficients $c_i$, the corresponding quadratic form is
$\|\sum_i c_i\Psi(W_i)\|^2$. Inserting a link across the join gives
$\|T(a/2)\sum_i c_i\Psi(W_i)\|^2$. Both are nonnegative and prove
the two stated reflections on the actual algorithmic-time grid.

The already proved even-algebra matrix units show that the cyclic space
of a nonzero fixed-parity vector is its whole parity block.
The even Hamiltonian preserves that block. Since
$H_0\Omega_0=0$, its unitary group fixes the vector, and
the scalar ground-energy shift cancels in conjugation of operators.
Its restriction therefore implements the existing edge dynamics with
nonnegative generator on this cyclic space.

Diagonalize the finite Hermitian $H_0$. Each spectral projection is
positive, hence its matrix on the vectors $A_i\Omega_0$ is a Gram
matrix. The finite spectral expansion gives the Laplace formula and
support on nonnegative energies. On $\operatorname{Ran}(I-P_0)$, the
same expansion has every energy at least $\delta_{\rm edge}$.
Its transfer norm is $e^{-\delta_{\rm edge}t}$.
Cauchy--Schwarz proves the stated connected bound.
The zero-energy term in the spectral expansion is $P_0$, which explains
why centering against one vector alone is insufficient in a degenerate
ground space.
:::

:::{prf:corollary} Explicit nonconstant even edge correlation
:label: cor-native-pt-edge-nonconstant

For a nonzero realized real-edge matrix, the positive-energy construction
in {prf:ref}`thm-native-pt-edge-reflection` has a nonconstant even
correlation. With the filled negative modes and empty zero modes as
$\Omega_0$, there is an even CAR polynomial $A$ such that

$$
\langle A\Omega_0,T(t)A\Omega_0\rangle=e^{-\gamma t},
\qquad\gamma>0.
$$

If there are no zero modes, one can take
$\gamma=d_1+d_2=2\delta_{\rm edge}$ for real-edge coefficients.
If there is a zero mode, one can take $\gamma=\delta_{\rm edge}$.
These rates are realized-record rates with the declared time units.
No lower bound uniform over records or populations is asserted.
:::

:::{prf:proof}
In an orthonormal eigenbasis of $h_{\rm e}$, set
$b_j^\dagger=a_j^\dagger$ when $\lambda_j\ge0$, and
$b_j^\dagger=a_j$ when $\lambda_j<0$.
The adjoints $b_j$ annihilate the filled ground with empty zero modes.
The CAR for the $b_j$ follow directly from those for the $a_j$.
Changing one occupation has cost $|\lambda_j|$.

With no zero mode, choose the two distinct indices of smallest absolute
eigenvalues and put $A=b_j^\dagger b_k^\dagger$.
Its ground vector has norm one by the CAR and energy $d_1+d_2$.
Real-edge spectral pairing gives $d_1=d_2=\delta_{\rm edge}$.
With a zero mode, choose one smallest nonzero-cost index and one zero
index. The same even word gives energy $\delta_{\rm edge}$.
The claimed exponential is its exact eigenvector matrix element.
:::

(sec-native-pt-particle-hole)=
## Exact excitation coordinates and native correlation error

:::{prf:lemma} Particle--hole transfer of the same edge Hamiltonian
:label: lem-native-pt-particle-hole

Choose the canonical filled-negative, empty-zero ground $\Omega_0$ and
the eigenbasis and $b_j^\dagger$ of the preceding proof. Regard the
label space $E$ with that basis as the excitation space and write
$g=|h_{\rm e}|$. The map

$$
\mathcal U(e_{j_1}\wedge\cdots\wedge e_{j_k})
=b_{j_1}^\dagger\cdots b_{j_k}^\dagger\Omega_0,
\qquad\mathcal U\Omega=\Omega_0,
$$

is unitary and satisfies

$$
H_0\mathcal U=\mathcal U d\Gamma(g),\qquad
T(t)\mathcal U=\mathcal U\Gamma_-(e^{-tg}).
$$

Thus the filled-ground excitation $k$-sector correlation is

$$
\left\langle\mathcal U(f_1\wedge\cdots\wedge f_k),
T(t)\mathcal U(g_1\wedge\cdots\wedge g_k)\right\rangle
=\det[\langle f_i,e^{-t|h_{\rm e}|}g_j\rangle]_{i,j=1}^k.
$$

The operator $|h_{\rm e}|$ here is the excitation-coordinate expression
of the existing filled-ground $H_0$. It is not a replacement sampling
generator and does not replace $h_{\rm e}$ in its original CAR dynamics.
:::

:::{prf:proof}
For every excitation label, $b_j\Omega_0=0$.
The CAR consequently make the displayed excitation wedges orthonormal.
Every original occupation is obtained uniquely by toggling its differing
occupations relative to the filled negative modes. There are $2^{\dim E}$
such wedges, so the map is unitary onto the original finite Fock space.
The energy of a toggle set is the sum of its $|\lambda_j|$ costs,
including zero costs. This proves the generator identity on its complete
occupation basis. Exponentiation and the exterior inner product prove
the two subsequent identities.
:::

:::{prf:theorem} Quantitative comparison with the complete native transition
:label: thm-native-pt-transfer-error

Retain the actual conservative stationary law and $C=P|_{L^2_0(\pi)}$
when invoking its stationary CAR reconstruction. Suppose that the
declared edge mode selection is an orthonormal finite set of actual
centered record modes, with isometric inclusion
$j:E\to\mathcal H=L^2_0(\pi)$.
This condition is a readout definition to check on the selected modes;
it is not asserted for arbitrary walker labels or a fitted spectral matrix.
For the realized fixed edge readout define its actual discrepancy

$$
B=e^{-a|h_{\rm e}|},\qquad
\varepsilon_{\rm tr}(\mathfrak P,Y)
=\|Cj-jB\|_{E\to\mathcal H}.
$$

Both $C$ and $B$ are contractions. For every $n\ge0$,

$$
\|C^nj-jB^n\|\le n\varepsilon_{\rm tr},\qquad
\|\Lambda^k(C^n)\Lambda^k(j)-\Lambda^k(j)\Lambda^k(B^n)\|
\le kn\varepsilon_{\rm tr}.
$$

Consequently the native recorded determinant correlation and the
filled-ground excitation correlation differ by at most

$$
kn\varepsilon_{\rm tr}
\|f_1\wedge\cdots\wedge f_k\|
\|g_1\wedge\cdots\wedge g_k\|.
$$

The native determinant is computed with $\langle jf_i,C^n jg_j\rangle$;
the edge determinant is computed with
$\langle f_i,B^n g_j\rangle$ and the particle--hole map of
{prf:ref}`lem-native-pt-particle-hole`.

For a finite transition word with creation and annihilation modes in
$jE$, let $q$ bound the number of occupied excitation modes at every
intermediate step, and let its transfer intervals be $n_1,\ldots,n_s$.
Replacing its native transfers by the same edge excitation transfers
changes its vacuum matrix element by at most

$$
q\varepsilon_{\rm tr}\Big(\sum_{\ell=1}^s n_\ell\Big)
\prod_{\text{insertions }f}\|f\|.
$$

The exact zero-defect regime is $Cj=j e^{-a|h_{\rm e}|}$ for the
actual selected modes. In that regime all these native replica correlations
equal the existing filled-ground excitation correlations at every grid
time. If a specified family has
$\varepsilon_{\rm tr}/a\to0$, the same comparison vanishes on fixed
physical horizons and finite words. The theorem does not assert that
either test has been proved for the viscous reference.
:::

:::{prf:proof}
Telescope the actual one-particle products:

$$
C^nj-jB^n
=\sum_{\ell=0}^{n-1}C^{n-1-\ell}(Cj-jB)B^\ell.
$$

Every outside factor has norm at most one, proving the first estimate.
For two contractions $R,S:E\to\mathcal H$, telescope
$R^{\otimes k}-S^{\otimes k}$ one tensor factor at a time. There are
$k$ terms, each of norm at most $\|R-S\|$.
Restriction to antisymmetric tensors proves the exterior estimate with
$R=C^nj$ and $S=jB^n$.
Apply this operator estimate between the two wedges. Their inner products
are exactly the two determinants by the existing replica reconstruction
and the particle--hole identity.

Creation or annihilation with mode $jf$ intertwines the Fock inclusion
$\Gamma_-(j)$ with the corresponding finite excitation insertion of $f$.
For annihilation this follows because its contraction coefficients are
preserved by the isometry; for creation it follows by insertion into a
wedge. Their norms equal $\|f\|$. At a transfer acting on at most
$q$ occupied modes, the proved sector estimate is bounded by
$qn_\ell\varepsilon_{\rm tr}$.
Telescope the finite word using the intertwining at each insertion and
the transfer estimate on its finite sector sum. Norms of all other
transfers are at most one; insertion norms supply the product in the
statement. This proves the word estimate. Exact equality and the stated
fixed-horizon consequence follow directly, with
$\sum n_\ell\le T/a$ for total physical duration at most $T$.
:::

:::{prf:corollary} Native temporal reflected-matrix error
:label: cor-native-pt-native-reflection-error

Use the stationary data and the discrepancy of
{prf:ref}`thm-native-pt-transfer-error`. For wedges
$u_i\in\Lambda^kE$ and integers $n_i\ge0$, the actual stationary
whole-swarm replica time-reflection matrix is

$$
Q^{\rm rec}_{ij}
=\langle\Lambda^k(j)u_i,
 \Lambda^k(C^{n_i+n_j})\Lambda^k(j)u_j\rangle.
$$

Here the reflected mode is the same mode evaluated at the past clock
slice; any additional configured reversal of its marks must also be
retained in that mode. Stationarity identifies the displayed expression
with the actual covariance between clock slices $-n_i$ and $n_j$.
The already constructed independent complete-swarm replica law realizes
its exterior sectors; independence inside a swarm is not used.

Its edge excitation counterpart $Q^{\rm e}$ is positive Hermitian, and
for every coefficient vector $c$,

$$
\left|c^*(Q^{\rm rec}-Q^{\rm e})c\right|
\le 2k\varepsilon_{\rm tr}
 \left(\sum_i|c_i|n_i\|u_i\|\right)
 \left(\sum_i|c_i|\|u_i\|\right).
$$

An additional native step across the cut adds
$k\varepsilon_{\rm tr}(\sum_i|c_i|\|u_i\|)^2$ to this bound;
the corresponding edge link matrix is positive as well. For even $k$
these edge vectors belong to the ground-parity observable sector.
Thus a proved $\varepsilon_{\rm tr}/a\to0$ on the actual modes gives
vanishing temporal reflected-form error on fixed physical horizons for
these native replica sectors. This is a quantitative consequence of that
estimate; the estimate itself is not proved for the reference gas here.
:::

:::{prf:proof}
The complete stationary Markov law gives
$\mathbb E[\overline{f(S_{-n_i})}g(S_{n_j})]
=\langle f,C^{n_i+n_j}g\rangle$. A finite window started in $\pi$
and shifted to a nonnegative clock interval proves this identity without
requiring observations before the start of a run. Applying the established
whole-swarm replica exterior realization gives the displayed matrix.
The edge matrix is the Gram matrix of
$\Gamma_-(B^{n_i})u_i$ and $\Gamma_-(B^{n_j})u_j$.
Its link version inserts the positive $\Gamma_-(B)$ between them.

The preceding theorem bounds each entry difference by
$k(n_i+n_j)\varepsilon_{\rm tr}\|u_i\|\|u_j\|$.
Sum the absolute entry bounds against $|c_i||c_j|$ and factor the two
sums. With one additional interval replace $n_i+n_j$ by
$n_i+n_j+1$, which gives the stated additional term.
:::

:::{prf:remark} The discrepancy retains leakage and the actual law
:label: rem-native-pt-discrepancy

With $\Pi=jj^*$, the defect has the orthogonal decomposition

$$
\|(Cj-jB)f\|^2
=\|j^*Cjf-Bf\|^2+\|(I-\Pi)Cjf\|^2.
$$

Thus a finite compression match alone does not identify the transfer when
the actual update leaks out of the selected mode space. Every entry of
$j^*Cj$ is the actual whole-swarm two-time covariance; every squared
leakage is an actual conditional-prediction norm minus its finite projected
norm. No independent-walker approximation is used. These profiles retain
all fields of $\mathfrak P$ through $P$, $\pi$, $j$ and the readout.
Their definition supplies a quantitative error, not a discharge that the
error is zero or small.

More explicitly, in the selected orthonormal basis define
$A=j^*Cj$ and $M=(Cj)^*(Cj)$. Their entries are the actual
covariances $A_{ij}=\langle je_i,Cje_j\rangle$ and prediction
products $M_{ij}=\langle Cje_i,Cje_j\rangle$. Then

$$
\varepsilon_{\rm tr}^{\,2}
=\lambda_{\max}\big(M-A^*B-BA+B^2\big),\qquad
\| (I-\Pi)Cj\|^2=\lambda_{\max}(M-A^*A).
$$

Indeed these are the matrices of $(Cj-jB)^*(Cj-jB)$ and the
orthogonal leakage operator. The prediction products use the complete
ordered native kernel, or equivalently two conditionally independent
complete native updates from the same present state. This representation
computes a prediction norm; it does not replace two successive native
updates by independent updates.

When $Y$ itself is random, the edge formulas hold for its realized matrix.
Using an unconditional native $C$ with its realized frozen mode functions
is a statement about those deterministic functions under the named law.
It is not automatically the correlation of the same random mode functions
evaluated on that same history. A conditional application must use the
actual conditional law and its conditional-prediction operators; future
information in $Y$ cannot be discarded in that conditioning.
:::

(sec-native-pt-changing-record)=
## Changing readouts and positive record averaging

:::{prf:proposition} Ordered comparison for the actual changing mode data
:label: prop-native-pt-changing-transfer

At consecutive native clock slices use their actual marginal mode spaces
$\mathcal H_\ell=L^2_0(\mu_\ell)$ and conditional-expectation contractions
$C_\ell:\mathcal H_{\ell+1}\to\mathcal H_\ell$.
For a Markov state these compose to its true two-time prediction operator.
A finite survival window uses its existing selected transitions and
selected marginals. If a common finite label space $E$ and isometries
$j_\ell:E\to\mathcal H_\ell$ are already specified by the readout,
retain them. Let $g_\ell=|h_{{\rm e},\ell}|$ be the existing Hermitian
edge excitation matrices in those specified coordinates, and
$B_\ell=e^{-a_\ell g_\ell}$, $a_\ell\ge0$.
Then

$$
\left\|C_0\cdots C_{n-1}j_n
-j_0B_0\cdots B_{n-1}\right\|
\le\sum_{\ell=0}^{n-1}
\|C_\ell j_{\ell+1}-j_\ell B_\ell\|.
$$

For a fixed comparison matrix $g_*\ge0$ in the same actual label
coordinates, its additional changing-edge error is bounded by

$$
\|B_0\cdots B_{n-1}-e^{-(\sum a_\ell)g_*}\|
\le\sum_{\ell=0}^{n-1}a_\ell\|g_\ell-g_*\|.
$$

Both bounds extend to the $k$th exterior sector with an extra factor $k$.
Dimension changes, absent mode transports and discarded directions are
not identified by these bounds. They require the actually declared
comparison maps and their separate remainder.
The ordered changing-edge product is not asserted self-adjoint or
reflection positive as the native transfer.
:::

:::{prf:proof}
Conditional Jensen gives each $C_\ell$ norm at most one, even when the
marginals vary. Telescope the two ordered products at their successive
isometries; all surrounding factors are contractions. This gives the
first sum.

For positive matrices $g_\ell,g_*$, differentiating
$e^{-(a_\ell-s)g_\ell}e^{-sg_*}$ on $0\le s\le a_\ell$ and
integrating gives the Duhamel bound
$\|e^{-a_\ell g_\ell}-e^{-a_\ell g_*}\|
\le a_\ell\|g_\ell-g_*\|$.
Telescope the products again. The constant comparison factors commute
with one another and multiply to the displayed exponential.
The antisymmetric tensor telescoping used in the previous theorem gives
the exterior bounds. No reversed order or self-adjointness of a varying
ordered product was used.
:::

:::{prf:corollary} Positive-energy edge correlations averaged over native records
:label: cor-native-pt-random-edge-positive

Take the actual probability law of complete records for $\mathfrak P$,
including its configured survival normalization when present. At each
realized Hermitian edge matrix use the canonical existing ground density
$\rho_0(Y)=P_0(Y)/\operatorname{Tr}P_0(Y)$.
For bounded measurable even operator words define their reflected edge
entries using

$$
\Psi_Y(W)
=T_Y(t_1)A_1(Y)\cdots
 T_Y(t_r-t_{r-1})A_r(Y)\rho_0(Y)^{1/2}
$$

and the Hilbert--Schmidt inner product. If the squared boundary norms
are integrable, then

$$
\overline Q_{ij}^{\rm e}
=\mathbb E_{\mathfrak P}
 \operatorname{Tr}\big(\Psi_Y(W_i)^*\Psi_Y(W_j)\big)\succeq0.
$$

Normalized finite CAR words have boundary norm at most the product of
their mode norms, giving integrability without a tail assumption when
those norms are uniformly bounded. Averaged two-point edge correlations
are Laplace transforms of positive measures supported on $[0,\infty)$.
For uniformly bounded boundary norms, their diagonal functions are
completely monotone for $t>0$, even without energy-moment bounds.
This preserves the actual law of edge records. It does not identify these
averaged edge correlations with the native ordered transition correlations.
:::

:::{prf:proof}
The finite spectral projection onto the ground is a measurable function
of the Hermitian matrix, including multiplicity changes. Its normalized
density is positive, has trace one and is killed by $H_0$.
The reflection-gluing proof therefore holds column by column with
$\rho_0^{1/2}$ in place of a pure ground vector.
Each realized matrix is positive Hermitian; integrating its nonnegative
quadratic forms against the actual normalized record law proves the
claim. The transfer contractions and bounded insertion norms give the
stated norm bound.

Each realized spectral measure is positive and supported on nonnegative
energies. Integrating those measures gives a positive measure with the
same support; its total mass is bounded by the integrable boundary norm.
Tonelli's theorem gives the averaged Laplace formula. For $t>0$,
$E^k e^{-tE}$ is bounded on $[0,\infty)$ for every integer $k\ge0$.
Local domination on any interval bounded away from zero permits
differentiation under this finite measure, and
$(-1)^k G^{(k)}(t)=\int E^k e^{-tE}d\eta(E)\ge0$.
:::

(sec-native-pt-history-reflection)=
## Exact time-cut factorization for the original selected history

:::{prf:proposition} Native temporal reflection defect with survival retained
:label: prop-native-pt-history-cut

Use a complete Markov history $S_0,\ldots,S_K$ of the existing algorithm
and its actual selected law, conditioned on survival through $K$ when
that is the specified experiment. Let $k$ be a clock cut and $\mu_k$
its actual marginal. Let $F_i$ be bounded future cylinders and let $G_i$
be their declared time-reflected past cylinders, reflecting the existing
clock about $kh$ and retaining the configured action on their marks.
No fourth position coordinate is introduced. Put

$$
A_i(S_k)=\mathbb E[F_i\mid S_k],\qquad
B_i(S_k)=\mathbb E[G_i\mid S_k],\qquad D_i=B_i-A_i.
$$

Then the original native reflected matrix is exactly

$$
Q^{\rm nat}_{ij}=\mathbb E[\overline{G_i}F_j]
=\langle B_i,A_j\rangle_{L^2(\mu_k)}.
$$

For $A_c=\sum c_iA_i$ and $D_c=\sum c_iD_i$,

$$
\operatorname{Re}(c^*Q^{\rm nat}c)
=\|A_c+D_c/2\|^2-\|D_c\|^2/4
\ge-\|D_c\|^2/4.
$$

Writing $A,D$ for the maps from the finite coefficient space to
$L^2(\mu_k)$, the skew-Hermitian defect also obeys

$$
\left\|\frac{Q^{\rm nat}-(Q^{\rm nat})^*}{2}\right\|
\le\|A\|\|D\|.
$$

For QSD initial data with survival through $K$, all these expectations
use exactly
$\mu_k=h_{K-k}\nu_N/\alpha_N^{K-k}$ and
$R_{\ell,K}(s,ds')=Q_N(s,ds')h_{K-\ell-1}(s')/h_{K-\ell}(s)$.
Their past expectations use the actual reverse conditional kernels of
this same selected path law. A positive-energy edge estimate cannot
silently replace either set of conditional expectations.
:::

:::{prf:proof}
The selected history is Markov with its actual time-dependent selected
kernels. For finite-horizon survival this follows from the survival-factor
cancellation already proved in
{prf:ref}`prop-ym-qsd-history-identification`; the analogous formula with
the actual initial law gives the same Markov property when the initial
law is not a QSD. A complete state retains every variable consumed by
future stages, including any donor history, cached geometry and clock
residue. Conditional on this state at the cut, past and future cylinders
are independent. Therefore
$\mathbb E[\overline{G_i}F_j\mid S_k]=\overline{B_i}A_j$.
Integrating proves the factorization.

Since $B_c=A_c+D_c$, expansion and completion of the square give the
quadratic-form identity. In matrix form
$Q^{\rm nat}=A^*A+D^*A$, so
$Q^{\rm nat}-(Q^{\rm nat})^*=D^*A-A^*D$.
The operator triangle inequality gives the claimed bound.
The displayed selected marginal and kernels are the exact previously
proved survival formulas. Reverse conditional kernels are determined
by the same two-time joint laws, not by an imposed detailed-balance law.
:::

(sec-native-pt-remaining)=
## Discharged and remaining physical-time statements

:::{prf:remark} Physical-time discharge register
:label: rem-native-pt-status

For every existing finite Hermitian edge readout, the filled-ground theory
now has positive-energy even correlations, and these are nonconstant for a
nonzero real-edge matrix. It also has temporal
reflection positivity on $\tau=nh$, both slice and link reflection,
positive spectral measures, and positive averaging over the native record
law. These conclusions use the already constructed Hamiltonian and its
actual ground space. They do not require reversing or modifying the gas.

The particle--hole intertwiner gives the exact positive excitation transfer
to compare with native record replica correlations. The complete transition
comparison retains its full-law discrepancy, leakage, changing-edge
variation, finite-window marginal law and survival normalization.
The native time-cut factorization separately bounds any negative reflected
form by the actual forward/backward conditional-prediction discrepancy.

For the unchanged viscous reference, the correspondence of these edge
correlations with the native physical gauge correlations remains open.
The missing estimate is a derived small transfer/word discrepancy on the
actual selected observable family, or a direct temporal reflected-form sign
estimate for that family. The formulas here neither establish that estimate
nor exhibit a violating reference-gas temporal word. Spatial locality,
Lorentz covariance, a unique continuum vacuum and a target-uniform physical
gap retain their separate existing obligations. The construction of positive
finite edge dynamics itself is discharged; its native transfer and field
identification is the residual.
:::
