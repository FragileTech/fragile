# Native joint-kick reflection and the preceding-kick physical sector

(sec-njk-register)=
## 1. Complete original algorithm, record and reflection

:::{prf:definition} Joint native kick dictionary
:label: def-njk-register

Retain EVERY original parameter, source address, mask and arithmetic
restriction of {prf:ref}`def-nir-register` and
{prf:ref}`def-nkr-register`. The original dense real-coordinate ROW
Gaussian viscosity has $N=2,d=3$, $h>0$, $t=h/2$, $q>0$ and

$$
c=e^{-\gamma h}\in(1/3,1),\quad
u=t^2\lambda=\frac12,\quad
\alpha=\frac{1-c}{2c},\quad
t\nu=\frac{1-\alpha}{2}>0,\quad s=\sigma_x\sqrt h=0 .
\tag{NJK.1}
$$

Retain the existing unbounded/all-alive boundary, harmonic force/reward,
`velocity_cap=None`, constant fitness exponents $p_r=p_s=0$ and
absence of graph/curl/metric/history feedback. Every original companion,
standardizer, logistic, donor, gate-period, unapplied jitter and collision
parameter remains configured. Accepted clone gates are zero and components
are singletons. The positive Gaussian mass of the unique nonself companion
cancels in Rust's unfloored dense row normalization. The Python
empty-CSR/floored-row force and finite-arithmetic underflow branches
retain their distinct laws.

On the SAME complete trajectory evaluate the existing viscous-color arms
`MatchedKick { B1 }` and `MatchedKick { B2 }`, with their fixed supplied
finite original phase $\kappa>0$ and common strict force threshold
$0\le\delta<\infty$. Their projector entries and availability marks form
the within-update joint dictionary. Both are evaluations of recorded
input states, forces and phase velocities. No trajectory, coefficient
or Gaussian innovation is added. Their joint evaluation is an instrument
on that record even when one spectroscopy plan selects one alignment.

Let $S_n$ be the entering B1 state and $Y_n$ the post-A2, pre-B2 state of
update $n$. Use the existing subsequence $n=4\ell k$, $\ell\ge1$ integer,
with physical interval $4\ell t_*h>0$. Retain every intermediate update.
The book's reflection is $\theta_0k=-k$ or $\theta_{1/2}k=1-k$, with scalar
complex conjugation and the original B1/B2 labels preserved. This is
the reflection in (NIR.13) and (NKR.11); those definitions contain no
stage exchange or additional phase transformation.
:::

(sec-njk-cross-covariance)=
## 2. Exact original cross-stage covariances

:::{prf:lemma} Forward and backward native stage covariance
:label: lem-njk-cross-covariance

For $a=1,\alpha$, use the already derived matrices $A_a=L_aK_a$,
$C_a,\widehat C_a$, and $b=(t,1)^T$. Put $r_a=c^2a^4$. For $\ell\ge1$,

$$
\begin{split}
\operatorname{Cov}(S_{a,0},Y_{a,4\ell})
 &=r_a^\ell C_aK_a^T,\\
\operatorname{Cov}(Y_{a,0},S_{a,4\ell})
 &=r_a^\ell\widehat C_aK_a^{-T},\\
\widehat C_aK_a^{-T}
 &=C_aK_a^T+q^2bb^TK_a^{-T}.
\end{split}
\tag{NJK.2}
$$

Let $V=C_{1,22}=q^2/[2(1-c^2)]$,
$f_a=(C_aK_a^T)_{22}$ and
$\widetilde f_a=(\widehat C_aK_a^{-T})_{22}$. Then

$$
C_{1,12}=0,\quad C_{1,11}=4t^2V,\quad
\widehat C_{1,22}=2V,\quad f_1=cV,\quad\widetilde f_1=V/c .
\tag{NJK.3}
$$

The relative formulas are

$$
\begin{split}
C_{\alpha,22}
 &=-\frac{q^2(17c^2-18c+5)}{(1+c)^2(c^2-6c+1)},\\
\widehat C_{\alpha,22}
 &=-\frac{q^2(25c^2-6c+1)}{(1+c)^2(c^2-6c+1)},\\
f_\alpha
 &=\frac{q^2(c^3-15c^2+13c-3)}{(1+c)^2(c^2-6c+1)}>0,\\
\widetilde f_\alpha-f_\alpha&=\frac{q^2}{1-c}>0 .
\end{split}
\tag{NJK.4}
$$

In particular

$$
0<\rho_\ell
 =\frac{r_\alpha^\ell f_\alpha}
        {\sqrt{C_{\alpha,22}\widehat C_{\alpha,22}}}
<\widetilde\rho_\ell
 =\frac{r_\alpha^\ell\widetilde f_\alpha}
        {\sqrt{C_{\alpha,22}\widehat C_{\alpha,22}}}<1 .
\tag{NJK.5}
$$

These are covariances of the original stationary trajectory and original
orthogonally transformed Gaussian arrays.
:::

:::{prf:proof}
The original identities are $Y_n=K_aS_n+qb\xi_n$ and
$S_{n+1}=L_aY_n$, because $s=0$. Since $A_a^{4\ell}=r_a^\ell I$,
the first covariance follows from independence of $\xi_{4\ell}$.
For the reverse covariance use $S_{4\ell}=A_a^{4\ell-1}L_aY_0$
plus subsequent independent original innovations, and
$A_a^{4\ell-1}L_a=A_a^{4\ell}K_a^{-1}$.
The stationary identity $\widehat C_a=K_aC_aK_a^T+q^2bb^T$
gives the third equality.

The actual inverse satisfies
$K_a^{-1}b=(-t/c,1/(2ca))^T$ at $u=1/2$. Thus
$\widetilde f_a-f_a=q^2/(2ca)$. Substitution in (NIR.3) gives
(NJK.3)--(NJK.4). Here $c^2-6c+1<0$. For the remaining sign set
$z=c-1/2\in(-1/6,1/2)$. Direct expansion gives

$$
3-13c+15c^2-c^3
=\frac18+\frac54z+\frac{27}{2}z^2-z^3
\ge13(z+5/104)^2+\frac{79}{832}>0 .
$$

This proves $f_\alpha>0$. The intervening original OU innovations have
the nonzero controllability determinant (NIR.6); their positive definite
two-update covariance makes the separated-time joint covariance
nonsingular. Therefore neither normalized cross-correlation is one.
:::

(sec-njk-angular)=
## 3. Bounded actual projector reflection test

:::{prf:lemma} Exact angular correlation and primitive strict coefficient
:label: lem-njk-angular

For different actual component indices $a,b\in\{1,2,3\}$ let
$Q(G)=G_a^2G_b^2/|G|^4$ for a standard three-Gaussian $G$.
Its null zero set can be assigned any value. For original standard
Gaussian vectors $G,H$ with coordinate correlation $\rho\in[0,1)$,

$$
\mathcal B(\rho)=E[Q(G)Q(H)]
=\frac1{225}+\sum_{j\ge1}\beta_{2j}\rho^{2j},\quad
\beta_{2j}\ge0,\quad\beta_2=\frac4{3675}.
\tag{NJK.6}
$$

The series converges absolutely and consequently

$$
\mathcal B(\widetilde\rho)-\mathcal B(\rho)
\ge\frac4{3675}(\widetilde\rho^2-\rho^2)>0
\quad(0\le\rho<\widetilde\rho<1).
\tag{NJK.7}
$$
:::

:::{prf:proof}
Expand the bounded $Q$ in the orthonormal standard Gaussian Hermite basis.
The actual isotropic Gaussian coupling is its Mehler kernel, so each
degree contributes its squared coefficients times $\rho^{\rm degree}$.
Evenness removes odd degrees and the original sphere mean is $1/15$.
The degree-two coefficients for $(G_i^2-1)/\sqrt2$ are
$2/(105\sqrt2)$ for $i=a,b$ and $-4/(105\sqrt2)$ for the remaining
index. Indeed $E|G|^2=3$ and the sphere moments are
$E[n_a^4n_b^2]=1/35$ and
$E[n_a^2n_b^2n_c^2]=1/105$. Off-diagonal degree-two coefficients
vanish by coordinate parity. Their squares sum to $4/3675$.
Parseval and $0\le Q\le1/4$ give convergence and the bound.
:::

:::{prf:theorem} Same-update B1/B2 color history fails native reflection
:label: thm-njk-joint-reflection

At $\delta=0$, every parameter in (NJK.1), every finite original
$\kappa>0$ and every stride $4\ell$ give strict failure of both site
and link reflection positivity for the actual fixed-label joint
projector history.

The failure extends to an explicitly computed positive-threshold
interval. Define

$$
\begin{split}
Z_\ell&=\frac4{3675}(\widetilde\rho_\ell^2-\rho_\ell^2)>0,\\
\sigma_1^2&=2C_{\alpha,22},\qquad
\sigma_2^2=2\widehat C_{\alpha,22},\qquad
c_{\rm ball}=\frac{\sqrt{2/\pi}}3,\\
\delta_\ell^*
 &=\nu\left[\frac{8Z_\ell}
 {c_{\rm ball}(\sigma_1^{-3}+\sigma_2^{-3})}\right]^{1/3}.
\end{split}
\tag{NJK.8}
$$

Link reflection fails for $0\le\delta<\delta_\ell^*$.
Site reflection also fails if $\delta<\delta_{2\ell}^*$.
These are sufficient primitive regimes; the estimate assigns no sign
outside them. The interval is independent of supplied phase because
the tested diagonal entries cancel the original phase.
:::

:::{prf:proof}
For ACTUAL row-one projectors at the two stages put

$$
Q_{1,n}=(P^{(1)}_{1,n})_{aa}(P^{(1)}_{1,n})_{bb},\qquad
Q_{2,n}=(P^{(2)}_{1,n})_{aa}(P^{(2)}_{1,n})_{bb}.
\tag{NJK.9}
$$

Unavailable projectors retain their original zero readout and their
availability labels. These are bounded ACTUAL color-only functions.
The literal phases cancel in their diagonal entries. At $\delta=0$
their almost surely available values are $Q(\Delta V_n)$ and
$Q(\Delta Z_n)$. The original relative covariance gives

$$
E[Q_{1,0}Q_{2,4\ell}]=\mathcal B(\rho_\ell),\qquad
E[Q_{2,0}Q_{1,4\ell}]=\mathcal B(\widetilde\rho_\ell).
\tag{NJK.10}
$$

For the bounded insertion $F=Q_1+iQ_2$ at coarse future time one,
the actual link diagonal is $E[\overline{F_0}F_1]$. Its imaginary
part is $\mathcal B(\rho_\ell)-\mathcal B(\widetilde\rho_\ell)
\le-Z_\ell<0$. A positive form must have real diagonals.
For the site diagonal the lag is $8\ell$; the same argument with
$2\ell$ proves strict failure. No zero-measure state or assigned
Hamiltonian is used.

At positive threshold the actual products acquire exactly
$1_{\nu|\Delta V|>\delta}$ and $1_{\nu|\Delta Z|>\delta}$.
Their unmasked product is at most $1/16$. Each cross expectation
changes by at most $(p_1+p_2)/16$, where $p_i$ are the ACTUAL
three-Gaussian lower-ball probabilities with variance $\sigma_i^2$.
The cross difference thus changes by at most $(p_1+p_2)/8$.
The maximum of the original Gaussian density gives

$$
p_i(\delta)\le c_{\rm ball}
                     \left(\frac{\delta}{\nu\sigma_i}\right)^3 .
\tag{NJK.11}
$$

The displayed threshold makes this loss strictly smaller than $Z_\ell$.
This retains actual availability and all original Gaussian tails.
:::

(sec-njk-witness)=
## 4. Evaluated witness and original phase-sensitive cross test

:::{prf:corollary} Exact native witness
:label: cor-njk-witness

At $h=1,c=\alpha=1/2,q=1,s=0,\lambda=2,\nu=1/2$ retain any
finite original $\kappa>0$. The original covariances are

$$
C_+=\frac23I,\quad
C_-=\frac1{63}\begin{pmatrix}17&-2\\-2&4\end{pmatrix},\quad
\widehat C_+=\frac23\begin{pmatrix}1&1\\1&2\end{pmatrix},\quad
\widehat C_-=\frac1{63}\begin{pmatrix}17&30\\30&68\end{pmatrix}.
\tag{NJK.12}
$$

At stride four,

$$
\begin{gathered}
f_+=1/3,\quad\widetilde f_+=4/3,\quad
f_-=2/63,\quad\widetilde f_-=128/63,\\
\rho_1=\frac1{128\sqrt{17}},\quad
\widetilde\rho_1=\frac1{2\sqrt{17}},\quad
Z_1=\frac{39}{2437120}.
\end{gathered}
\tag{NJK.13}
$$

Thus $\delta_1^*=0.01389782837\ldots$. In particular the ACTUAL
effective threshold $10^{-15}$ is in the proved link-defect interval
if this configured joint projector readout consumes that clamp.
No numerical branch is replaced by threshold zero.

For the phase-sensitive native functions

$$
D_{j,n}=\operatorname{Im}
 \{(P^{(j)}_{1,n})_{ab}(P^{(j)}_{2,n})_{ab}\},\qquad j=1,2,
\tag{NJK.14}
$$

the zero-threshold cross difference is

$$
\begin{split}
&E[D_{2,0}D_{1,4}]-E[D_{1,0}D_{2,4}]\\
&=e^{-4\kappa^2}
 \{\mathcal B(\widetilde\rho_1)\sinh(4\kappa^2/3)
              -\mathcal B(\rho_1)\sinh(\kappa^2/3)\}\\
&\ge\frac{e^{-4\kappa^2}}{225}
       \{\sinh(4\kappa^2/3)-\sinh(\kappa^2/3)\}>0 .
\end{split}
\tag{NJK.15}
$$

At $\kappa=1$ this primitive lower bound is
$0.0001160393261\ldots$; its positive-threshold extension is
$\delta<0.02690003090\ldots$ by the same original mask budget.
The decimals evaluate exact strict formulas.
:::

:::{prf:proof}
Substitution in (NIR.3), (NKR.3) and (NJK.4) gives the matrices
and correlations; $r_+=1/4,r_-=1/64$. For (NJK.14) the literal
row phases give
$D_j=Q(\Delta_j)\sin[2\kappa(M_{j,a}-M_{j,b})]$.
The original centroid and relative processes are independent.
The two centroid differences have variances $2/3,4/3$ and
cross covariances $1/12,1/3$ in their two orders.
Their original Gaussian characteristic function yields the two
factors $e^{-4\kappa^2}\sinh(\kappa^2/3)$ and
$e^{-4\kappa^2}\sinh(4\kappa^2/3)$.
Multiply by (NJK.10). Since
$\mathcal B(\widetilde\rho_1)\ge\mathcal B(\rho_1)\ge1/225$,
the strict lower bound follows. The original positive-threshold
loss is again at most $(p_1+p_2)/8$, because $|D_j|\le1/4$.
:::

(sec-njk-preceding)=
## 5. Positive full joint reconstruction of the existing preceding-kick arm

:::{prf:theorem} Current B1 and preceding B2 physical sector
:label: thm-njk-preceding-positive

Retain (NJK.1), every finite original $\delta\ge0$, fixed supplied
$\kappa>0$ and stride $4\ell$. Consume the ACTUAL joint dictionary
of current `MatchedKick { B1 }` and `PrecedingKick { B2 }`,
with the previous CONTIGUOUS actual frame retained. Both native
colors and projectors are functions of the current entering state:

$$
\mathcal D_n^\leftarrow
 =(\mathcal D_{\rm B1}(S_n),\mathcal D_{\rm B2}(L^{-1}S_n)).
\tag{NJK.16}
$$

Its complete joint history has positive site and link forms under the
original time-index reflection. Its physical space is the native
closed multiplication/transfer sector

$$
\mathcal K_\leftarrow
=\overline{\operatorname{span}}\{
 M_{f_0}P_{4\ell}M_{f_1}\cdots P_{4\ell}M_{f_r}1:
 f_j\text{ bounded functions of }\mathcal D^\leftarrow\}.
\tag{NJK.17}
$$

The restricted actual transfer is positive and injective, its vacuum
is unique and its log Hamiltonian with original interval $4\ell t_*h$
has EXACT gap

$$
\operatorname{gap}H_\leftarrow=\frac{\gamma}{2t_*}.
\tag{NJK.18}
$$

This is a positive full JOINT color history with both native viscous
kicks and all original innovations. It is a different EXISTING record
alignment from the same-update joint dictionary.
:::

:::{prf:proof}
At zero terminal position diffusion the actual update gives
$S_n=LY_{n-1}$, with both $L_a$ invertible. Thus
$Y_{n-1}=L^{-1}S_n$ exactly. The previous-stage force and its OWN
previous-stage phase velocity are precisely those consumed by
`PrecedingKick`. They are not replaced by the current phase velocity
of the separate `PrecedingForce` arm. Zero accepted clones and unchanged
incarnations satisfy the actual matching masks, proving (NJK.16).

The original $S_{4\ell k}$ law is the reversible positive Mehler chain
$P_{4\ell}=P_4^\ell$, with mode eigenvalues $r_+^\ell,r_-^\ell$
and $0<r_-<r_+<1$. For any finite future joint cylinder put
$JF=E[F\mid S_0]$. Native Markov conditioning gives exactly the
dense products in (NJK.17). Its invariant closed space is reducing
because $P_{4\ell}$ is selfadjoint. Original conditional independent
past and future give the ACTUAL forms

$$
E[\overline{F\circ\theta_0}G]=\langle JF,JG\rangle,\qquad
E[\overline{F\circ\theta_{1/2}}G]
                      =\langle JF,P_{4\ell}JG\rangle .
\tag{NJK.19}
$$

Both are positive and site-null quotient completion gives
$\mathcal K_\leftarrow$. The actual full Mehler transfer has only the
constant eigenvector at one, no zero eigenvector, and zero as a
spectral accumulation point. Restriction therefore gives the stated
unique vacuum and positive injective transfer.

This dictionary contains the original current B1 projector dictionary.
Its bounded product witness (NIR.14) has a nonzero degree-one centroid
coefficient for EVERY $\kappa>0$ and finite $\delta$, with its exact
positive original force-availability probability. Its spectral
projection gives a degree-one centroid vector in $\mathcal K_\leftarrow$.
That eigenvalue is $r_+^\ell=c^{2\ell}$, while all nonconstant
full-state eigenvalues are at most $r_+^\ell$.
Its exact energy is $-\log(c^{2\ell})/(4\ell t_*h)=\gamma/(2t_*)$.
:::

(sec-njk-scope)=
## 6. Exact algorithm and physical scope

The nonreal form concerns the ACTUAL within-update B1/B2 component
projector instrument and the book's declared reflection. It does not
infer a defect for the smaller common-frame scalar-orbit algebra,
another physical reflection or a continuum Yang--Mills limit.
A stage-exchanging involution is absent from the consumed record
implementation and from (NIR.13)/(NKR.11); that identification would
require its own actual observable map and tested form. Separate
positive stage clocks alone do not supply it.

The positive preceding-kick theorem consumes the original contiguous
frame rule. An archive containing only frames spaced by $4\ell$
fails `previous.step + 1 == frame.step`; its previous stored frame
does not become the actual predecessor. The complete intermediate
record, or the original accumulator ingesting those updates, supplies
the required predecessor. Both alignments retain their original
source labels and clocks.

Positive terminal position diffusion removes the deterministic identity
$Y_{n-1}=L^{-1}S_n$ and is not assigned (NJK.16). Positive selection,
other force normalizations, Gaussian underflow, caps, adaptive graphs,
landscapes and nonresonant clocks retain their actual separate laws.
Random frozen/history calibration retains its independently proved
static ground-sector effects in Chapter NKR. This finite $N=2$ color
gap is not asserted to be the population-uniform local continuum
Yang--Mills gap.

