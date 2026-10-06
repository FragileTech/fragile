# Native physical spectrum of the complete spatial-action limit

(sec-nasr-register)=
## 1. Existing action, native clock and complete fluctuation law

:::{prf:definition} Complete action reconstruction register
:label: def-nasr-register

Retain every algorithm, landscape, source, mask, arithmetic, graph,
phase and observation parameter of {prf:ref}`def-nspf-register`.
In particular this is the existing matched B1 harmonic stationary
branch, exact book Delaunay/CSR composite action, positive finite
phase family $\kappa_N/r_N\to\eta<\infty$, zero terminal spatial
noise and its proved resonant recorded clock. Its physical interval is

$$
 \Delta=mt_*h,\qquad r=c^{m/2},\qquad
 g_0=-\log r/\Delta=\gamma/(2t_*).
 \tag{NASR.1}
$$

For a real continuous strict-chart test $f$, use the COMPLETE
executed action fluctuation $T_{N,k}(f)$ in (NSPF.1).
The limiting process $T_k(f)$ is the Gaussian limit already proved
in {prf:ref}`thm-nspf-full-action`. No stage, force, graph, noise,
source or finite-population action is replaced.

Write $D_\eta^f(x,\Pi)=f(x)D_\eta^1(x,\Pi)$ for the actual
local add-one response (NSPF.2), and put

$$
 d_f(x)=E_\Pi D_\eta^f(x,\Pi),\quad
 m_f=\int p(x)d_f(x)\,dx,\quad h_f=d_f-m_f,\quad
 L_f(x)=f(x)\overline B_\eta(x).
 \tag{NASR.2}
$$

All these quantities are the original finite Poisson/Gaussian
integrals, with their primitive moment bounds. In particular they
are not unspecified stationary-law coefficients.
Let $[\,\cdot\,]_\ell$ denote orthogonal projection onto total
degree $\ell$ Hermites of $N(0,XI_3)$, with Frobenius norm for
matrix-valued functions. Define

$$
 a_\ell(f)=2V^2\|[L_f]_\ell\|_{L^2(p)}^2,\qquad
 d_\ell(f)=\|[h_f]_\ell\|_{L^2(p)}^2\quad(\ell\ge1).
 \tag{NASR.3}
$$

The existing site reflection is
$\Theta T_k(f)=T_{-k}(f)$, followed by complex conjugation of
polynomial coefficients. The physical dictionary in this chapter
consists of polynomials of these ORIGINAL action fluctuations at
nonnegative recorded times. It does not enlarge that dictionary
by intermediate stages or assign native spatial regions a
spacelike interpretation.
:::

(sec-nasr-complete-spectrum)=
## 2. Exact covariance spectrum and the retessellation remainder

:::{prf:theorem} Complete native action covariance spectrum
:label: thm-nasr-covariance-spectrum

For every test in the register the complete covariance is

$$
\begin{split}
 C_f(k)
 &=b_f\,{\bf1}_{\{k=0\}}+\sum_{j\ge1}w_j(f)r^{j|k|},\\
 w_j(f)&=d_j(f)+{\bf1}_{\{j\ge2\}}a_{j-2}(f),\\
 b_f&=\mathcal V_\eta(f)-\sum_{\ell\ge0}a_\ell(f)
       +\mathcal Q_\eta(f)-\|h_f\|_{L^2(p)}^2\ \ge0 .
\end{split}
 \tag{NASR.4}
$$

Every series is absolutely convergent. For distinct tests use
the bilinear versions of all displayed coefficients. The matrices
of $b$ and each $w_j$ on every finite test collection are positive
semidefinite.

The formula includes both the original velocity marks and
position/retessellation center. The contribution $b_f$ has
zero covariance at EVERY strictly positive recorded lag.
:::

:::{prf:proof}
The original Mehler formula (NMAF.2) and Hermite orthogonality
expand (NMAF.12) into
$\sum_{\ell\ge0}a_\ell r^{(\ell+2)|k|}$ at $k\ne0$.
They expand (NSPF.8) into
$\sum_{\ell\ge1}d_\ell r^{\ell|k|}$.
The complete same-time variance is
$\mathcal V_\eta+\mathcal Q_\eta$.

Equation (NSPF.3), with its single common/private mixing value,
proves $\mathcal Q_\eta\ge\|h_f\|_2^2$:
conditional on the common Poisson background, the two
private backgrounds are independent and identically distributed,
so their response product is a conditional mean square.
For the mark contribution, expand the leading score in the
orthogonal Hermites of its ORIGINAL row velocities. Its complete
single-address term is (NMAF.4). Its variance limit is

$$
 2V^2\int f(x)^2p(x)
             E_\Pi\operatorname{tr}B_\eta(x,\Pi)^2\,dx
 \ \ge\ 2V^2\int p(x)\operatorname{tr}L_f(x)^2\,dx
 =\sum_{\ell\ge0}a_\ell .
 \tag{NASR.5}
$$

The omitted, multiple-address Hermite terms have nonnegative
squared norms. Original four-ring moments and the local
two-root averaging calculation of (NMAF.12) justify the first
limit and remove its shields. Thus both remainders in $b_f$
are nonnegative.

The same calculation applies to every real linear combination
of finitely many tests. This proves positive semidefiniteness
of all the coefficient matrices. Such linear combinations also
have the position cavity moments and concentrated brackets
of {prf:ref}`lem-nspf-bracket`, so the complete Gaussian
finite-history law holds jointly for the tests. Parseval gives
$\sum a_\ell<\infty$, $\sum d_\ell<\infty$ and proves
absolute convergence, including their bilinear versions.
:::

(sec-nasr-white-mode)=
## 3. A strictly positive native zero-transfer mode

:::{prf:theorem} Primitive positive equal-time remainder at every positive critical phase
:label: thm-nasr-positive-white-mode

If $f\ge0$ is nonzero and $\eta>0$, then

$$
 b_f\ \ge\
 \frac{\eta^8V^4}{108}
 \int f(x)^2p(x)G(n)^2
             E_{\Pi_{p(x)}}|\mathcal T_0|\,dx
 \ >0,\qquad
 G(n)=6n_1^2n_2^2n_3^2 .
 \tag{NASR.6}
$$

This is a property of the COMPLETE original action law.
It does not presume that its local marks, faces or temporal
addresses are independent.
:::

:::{prf:proof}
For an original vertex mark $v_a$, put

$$
 H_a=\frac{|v_a|^2-3V}{\sqrt6\,V}.
$$

It has mean zero and norm one. For distinct addresses,
the products $H_aH_b$ form an orthonormal collection.
They are multiple-address fourth-degree Hermites.

The face commutator decomposition (NMAF.6) shows that only
its pure-phase quartic term can have an $H_aH_b$ coefficient.
Set $D=Q\operatorname{diag}(n)$ and $S=DD^T$. The quartic
commutator on a face is the sum of the three alternating
bilinear pair commutators. Terms involving a third mark have
zero $H_aH_b$ coefficient by oddness in a remaining mark.
For the pair $(a,b)$ itself,

$$
 E\|[B_{iDv_a},B_{iDv_b}]\|_F^2
 =2V^2\{(\operatorname{tr}S)^2-\operatorname{tr}S^2\}
 =2V^2G(n).
$$

Moreover
$E[v_av_a^TH_a]=(2V/\sqrt6)I_3$.
Replacing BOTH Gaussian second moments by this last
identity multiplies the preceding squared-commutator mean
by $2/3$. Consequently the coefficient of EACH pair on
one original ordered face in $\xi_i$ is exactly

$$
 E[\xi_{i,\mathrm{face}}H_aH_b]
       =\eta^4V^2G(n_i)/18 .
 \tag{NASR.7}
$$

This is independent of the face orientation, because its
action is the squared norm. It is nonnegative. The geometric
and mixed terms have velocity degree at most two and cannot
contribute to (NASR.7).

For the complete normalized leading action, sum the
coefficients of every face using a given pair. Since $f_i\ge0$,
the square of that sum is at least the sum of the squares of
its individual face contributions. Sum over all unordered
address pairs. Each ordered face uses three such pairs.
The squared norm of this orthogonal fourth-degree projection
is therefore at least

$$
 \frac{\eta^8V^4}{108N}
       \sum_i f(x_i)^2G(n_i)^2|\mathcal T_i^N|.
 \tag{NASR.8}
$$

The actual critical phase sequence gives the same limit with
$\eta_N\to\eta$. Original local Poisson convergence, protected
four-ring count moments and two-root averaging give the
integral in (NASR.6) as the limit of (NASR.8).
The centered exact Wilson action and this leading score have
the already proved vanishing $L^2$ comparison. Subtracting
the single-address variance in (NASR.5) from the complete
mark variance leaves ALL these fourth-degree multiple-address
norms. Subtracting the smaller averaged single-address
variance $\sum a_\ell$ only increases the remainder.
The nonnegative position remainder then proves (NASR.6).

The root of the full-dimensional Poisson Delaunay graph has
a bounded Voronoi cell almost surely by the original shielding
proof, hence a nonempty triangular star. Thus
$E|\mathcal T_0|>0$. Also $p>0$ everywhere and $G(n)>0$
away from the three coordinate planes. A nonzero continuous
nonnegative chart test is positive on an open set. These
facts prove the strict inequality without an additional
geometric or covariance hypothesis.
:::

(sec-nasr-reconstruction)=
## 4. Native reflection positivity and the actual reconstructed transfer

:::{prf:theorem} Full action site-reflection reconstruction and exact transfer spectrum
:label: thm-nasr-action-reconstruction

The COMPLETE Gaussian action history is reflection positive
for the site's ORIGINAL recorded reflection and the declared
polynomial dictionary. For one test, its reconstructed
one-particle space is the closure of vectors

$$
 u_t=\left(\sqrt{b_f}{\bf1}_{\{t=0\}},
             \{\sqrt{w_j(f)}r^{jt}\}_{j:\,w_j(f)>0}\right),
 \qquad t=0,1,\ldots .
 \tag{NASR.9}
$$

Its actual one-step transfer is

$$
 \mathsf T_1=0\ \hbox{on the first coordinate},\qquad
 \mathsf T_1=r^j\ \hbox{on coordinate }j .
 \tag{NASR.10}
$$

An absent coefficient means the corresponding coordinate
is absent. The full polynomial reconstruction is symmetric
Fock space over this one-particle space, with transfer
$\mathsf T=\Gamma(\mathsf T_1)$ and unique vacuum.

For $f\ge0$, $f\ne0$, $\eta>0$, the first coordinate is
present by (NASR.6). Therefore the FULL reconstructed
transfer has a nontrivial kernel. It cannot be
$e^{-\Delta\mathsf H}$ for a densely defined self-adjoint
Hamiltonian on that same full space. In particular its
site reconstruction has no strongly continuous positive-time
Hamiltonian interpolation on the full declared dictionary.

The NONZERO transfer spectrum on the vacuum complement has
lowest native frequency

$$
 g_f=
 \begin{cases}
   g_0,& \displaystyle
       \int x f(x)E_\Pi D_\eta^1(x,\Pi)p(x)\,dx\ne0,\\[4pt]
   2g_0,& \displaystyle
       \int x f(x)E_\Pi D_\eta^1(x,\Pi)p(x)\,dx=0 ,
 \end{cases}
 \quad
 \|\mathsf T|_{\Omega^\perp}\|=e^{-\Delta g_f}.
 \tag{NASR.11}
$$

The energy calibration is $\hbar_{\rm eff}g_f$.
These are the exact frequency and discrete transfer estimates
for the native limit. They do not identify its noninjective
full transfer with a continuous-time target Hamiltonian.
:::

:::{prf:proof}
For times $t,s\ge0$, (NASR.4) gives the reflected covariance

$$
 E[T_{-t}(f)T_s(f)]
 =b_f{\bf1}_{\{t=s=0\}}
       +\sum_jw_j(f)r^{j(t+s)}
 =\langle u_t,u_s\rangle .
$$

Thus its first-chaos reflection matrix is positive.
Use Wick polynomials with the ORIGINAL Gaussian history
covariance at the positive times. Wick's pairing identity
cancels contractions internal to either side; the reflected
inner product between equal-degree Wick monomials is the
sum over pairings of products of the displayed cross
inner products. Different degrees are orthogonal.
This proves positivity for the entire polynomial dictionary
and identifies its completion with symmetric Fock space.
It also follows by passing the existing finite-$N$ positive
full-state reflected form to the limit using the uniform
all-action moments, but no positivity is needed beyond
the explicit displayed Gram matrix.

To identify the one-particle closure, the vectors with
$t\ge1$ span all the nonzero coordinates densely.
Indeed a vector orthogonal to them defines a finite
signed measure on the points $r^j$ with every positive
integer moment zero. Finiteness follows from
$\sum w_j<\infty$ and Cauchy--Schwarz.
Polynomial approximation then makes that measure supported
at zero, where it has no atom. It is zero.
For completeness this conclusion can be obtained using
polynomials vanishing at zero: they approximate every
continuous function vanishing at zero on $[0,r]$, and
the remaining possible measure is a multiple of the
point mass at zero. Hence the orthogonal vector vanishes.
Subtracting the nonzero-coordinate part of $u_0$ then
isolates the first coordinate when $b_f>0$.

Time shift sends $u_t$ to $u_{t+1}$ and has exactly the
coordinate action (NASR.10). Wick time shift is its
second quantization, which gives the full transfer.
Every nonvacuum product of nonzero coordinates has
eigenvalue a product of $r^j<1$; any factor of the
first coordinate gives eigenvalue zero. The vacuum
is therefore the unique eigenvector with eigenvalue one.

For a densely defined self-adjoint Hamiltonian,
the spectral theorem makes $e^{-\Delta\mathsf H}$
injective: its spectral multiplier is strictly positive
at every finite spectral value. The positive first
coordinate in (NASR.10) disproves that identification
on the actual full reconstructed space.
Likewise the coordinate values of any proposed
interpolation at $t>0$ would be zero on this coordinate;
they cannot approach identity as $t\downarrow0$.
Restricting the dictionary or quotienting out this
coordinate would be a different reconstruction choice;
no such change is used in the theorem.

Finally $w_1=d_1$, and its value is exactly

$$
 d_1(f)=X^{-1}\left|
       \int x f(x)E_\Pi D_\eta^1(x,\Pi)p(x)\,dx
                         \right|^2 .
 \tag{NASR.12}
$$

The constant subtraction $m_f$ has no first Hermite
coefficient. Also $w_2\ge a_0>0$ at the stated
nonnegative positive-phase regime: (NMAF.4) is positive
semidefinite, and its primitive pure-phase radial
coefficient is strictly positive on the same open set
used in (NASR.6). Hence
$\int p f\overline B_\eta$ is a nonzero positive
semidefinite matrix. This proves $a_0>0$.
The least nonzero one-particle degree is consequently
one precisely when (NASR.12) is nonzero, and otherwise
two. Products cannot lower that degree.
Equations (NASR.1) and (NASR.10) give (NASR.11).
:::

(sec-nasr-spatial-covariance)=
## 5. Actual equal-time spatial covariance and its scope

:::{prf:corollary} Native complete spatial covariance on disjoint chart tests
:label: cor-nasr-disjoint-covariance

For real chart tests $f,g$ with disjoint supports,

$$
 E[T_0(f)T_0(g)]=-m_fm_g .
 \tag{NASR.13}
$$

Thus the actual fixed-population limit retains its
global rank-one position correction whenever the two
explicit add-one integrals $m_f,m_g$ are nonzero.
If either is zero this equal-time cross covariance
vanishes. This is an exact characterization, not an
assumed local covariance law.

Neither (NASR.13) nor reflection positivity identifies
these native position regions with physical spacelike
regions, or proves a relativistic local algebra.
:::

:::{prf:proof}
The same-time mark bilinear covariance is the
polarization of (NMAF.5); its local limit contains
$f(x)g(x)$ and is zero for disjoint supports.
The position bilinear covariance is the polarization
of (NSPF.3). Its local common/private response product
also contains $f(x)g(x)$, so only the exact
$-m_fm_g$ fixed-population subtraction remains.
Every integral is justified by the existing add-one
moments. This proves (NASR.13) on the original
Gaussian population, without imposing compact
population support or changing the graph.
:::

The existing action limit therefore has an explicit
positive recorded-time transfer, a unique vacuum,
a fully characterized nonzero spectrum and a
strictly positive zero-transfer sector at positive
critical phase. The remaining continuous-time and
spacetime identifications must accommodate these
actual features. A positive covariance or a
finite-step gap does not remove the zero-transfer
sector from the declared physical dictionary.

