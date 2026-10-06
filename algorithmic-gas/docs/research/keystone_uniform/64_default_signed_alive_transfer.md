# Actual alive-law transfer of the signed inward kinetic family

(sec-siat-register)=
## 1. Retained kinetic endpoint and its exact target

:::{prf:definition} Balanced prepared inputs and their actual alive laws
:label: def-siat-register

Retain every primitive and the actual count kinetic stages of
{prf:ref}`def-dsg-register`, including the unchanged harmonic
$F=-x$, $h=.04$, $\nu=.3$, $d=3$, $V=2$, $L=2$, $\rho=1$,
both full Gaussian innovations and the native cap.
Let $N\ge2$ be even. For $z\in[.2,.3]$, use a deterministic
prepared array with $N/2$ rows at each of
$(ze_1,-ze_1/2)$ and $(-ze_1,ze_1/2)$, with a fixed common
sign assignment when comparing two arrays. Run the actual
kinetic kernel and its terminal classification. Its positions
and stored velocities are not resampled after classifying.

Write $E_z=\{M_z^+>0\}$, where $M_z^+$ is its terminal alive
count, and define
$$
\mathcal H_N(z)=
\operatorname{Law}\left(
  \frac1{M_z^+}\sum_{i:a_i^+=1}
          \delta_{(x_i^+,v_i^+)}\ \middle|\ E_z\right).
\tag{SIAT.1}
$$
The outer metric $\mathbb W_{2,G}$ has inner physical phase
$W_{2,G}$ as ground, with the same fixed
$G=G_\beta$, $\beta=.04$, of
{prf:ref}`thm-dsg-inward-family`.
Let $\ell_N^{\rm swarm}(z)$ sample a uniformly alive slot after
this swarm's own survival conditioning. Let
$\ell_N^{\rm slot}(z)$ first sample an original slot uniformly
and condition that slot to be alive. Adding the swarm's own
nonextinction conditioning before this latter slot draw gives
the same slot-conditioned law.

This is a theorem about the actual kinetic transition from the
stated prepared inputs. It does not assert that a full active
preparation preserves or produces those exact inputs from
arbitrary entering swarms. No preparation, fitness or revival
operator is modified to make that happen.
:::

(sec-siat-transfer)=
## 2. Bounded alive readouts preserve the signed physical gain

:::{prf:lemma} Common-alive plan with a mean-square physical carrier
:label: lem-siat-bounded-transfer

Let two coupled proposed arrays have a physical phase ground
quadratic $G$ and a common bounded alive phase set of squared
diameter at most $D_*^2$. Suppose
$$
\mathbb E\frac1N\sum_i q_G(Z_i-\widetilde Z_i)\le k^2 w_0^2,
\quad
\mathbb E(N-M)/N,\
\mathbb E(N-\widetilde M)/N\le\epsilon.
\tag{SIAT.2}
$$
On extinction assign the auxiliary readout a point mass inside
the alive set, without changing either killed transition.
The raw auxiliary empirical-law distributions $\mathcal B,
\widetilde{\mathcal B}$ satisfy
$$
\mathbb W_{2,G}(\mathcal B,\widetilde{\mathcal B})^2
\le k^2w_0^2+9D_*^2\epsilon.
\tag{SIAT.3}
$$
If each own extinction probability is at most $\epsilon$, their
separately normalized actual alive empirical laws obey
$$
\mathbb W_{2,G}(\mathcal H,\widetilde{\mathcal H})
\le kw_0+5D_*\sqrt\epsilon.
\tag{SIAT.4}
$$
This is an upper bound from an admissible physical alive plan,
not a status-cost comparison or a prescribed-coupling lower bound.
:::

:::{prf:proof}
On the event $M,\widetilde M\ge N/2$, pair each common-alive
index with mass $1/M_{\max}$,
$M_{\max}=\max(M,\widetilde M)$.
Every such row pair has physical cost at most $D_*^2$.
The cost of these common pairs is at most
$$
\begin{split}
\frac1{M_{\max}}\sum_{i\in A\cap\widetilde A}
               q_G(Z_i-\widetilde Z_i)
&\le\frac1N\sum_i q_G(Z_i-\widetilde Z_i)
   +D_*^2\frac{N-M_{\max}}N.
\end{split}
\tag{SIAT.5}
$$
Indeed the excess coefficient is
$(N-M_{\max})/(NM_{\max})$, and the intersection has
size at most $M_{\max}$.
Complete the residual empirical marginals by any transport plan.
Its mass is at most
$|A\mathbin\triangle\widetilde A|/M_{\max}
\le2|A\mathbin\triangle\widetilde A|/N$,
and its cost per unit mass is at most $D_*^2$.

The expectation of the excess term in (SIAT.5) is at most
$D_*^2\epsilon$.
The expected symmetric-difference fraction is at most
$2\epsilon$, so the residual contributes at most
$4D_*^2\epsilon$. Markov's inequality gives probability at most
$4\epsilon$ that either alive count is below $N/2$.
On that event the two auxiliary measures still lie in the
bounded phase set and cost at most $D_*^2$.
Use (SIAT.2) for the remaining nonnegative carrier cost.
This gives (SIAT.3).

Each raw auxiliary law is the mixture of its own survivor law
and its own extinct auxiliary law. Maximal mixture coupling
shows its outer distance from its survivor law is at most
$D_*\sqrt\epsilon$. Apply this separately to both marginals.
The metric triangle and
$\sqrt{k^2w_0^2+9D_*^2\epsilon}
 \le kw_0+3D_*\sqrt\epsilon$
prove (SIAT.4). There is no joint-survival normalization.
All transport plans preserve each actual empirical alive weight.
:::

(sec-siat-alive)=
## 3. Exact size-independent alive transport for the signed family

:::{prf:theorem} Alive empirical-law and both sampled-law kinetic estimates
:label: thm-siat-signed-alive-laws

Take $z_0,z_1\in[.2,.3]$ with $|z_1-z_0|\ge1/100$.
For every even $N\ge2$,
$$
\mathbb W_{2,G}(\mathcal H_N(z_0),\mathcal H_N(z_1))
\le\frac{249}{250}\,
       W_{2,G}(\lambda_{z_0},\lambda_{z_1}),
\tag{SIAT.6}
$$
where the two deterministic input empirical probabilities are
$\lambda_z$ in (DSG.5).
The same coefficient holds for each of
$\ell_N^{\rm swarm}$ and $\ell_N^{\rm slot}$.
The coefficient is independent of $N$ and has no particle floor.
Every law uses its own actual alive or nonextinction denominator.
This is one kinetic endpoint estimate, not a default full-update
or repeated-time convergence assertion.
:::

:::{prf:proof}
Use the full actual paired-noise comparison in
{prf:ref}`thm-dsg-inward-family`.
Its normalized labeled physical cost is at most
$(99/100)w_0^2$, where
$w_0=\sqrt{121/100}\,|z_1-z_0|\ge1/100$.
The aligned sign assignment is preserved in this comparison;
permutation equivariance gives the same empirical output laws
for any other assignments.

Every actual terminal position is an independent Gaussian of
variance $\tau^2=t^2q^2+s^2<.0005$.
Its first-coordinate mean is
$\pm z[a_x-b(1-ae^{-2z^2})/2]$, of absolute value less than $.3$;
the other means are zero.
Each mean is at least $1.7$ from every box face.
Consequently, with $\epsilon=2^{-100}$,
$$
\Pr(a_i^+=0)\le6e^{-1.7^2/(2(.0005))}
 =6e^{-2890}<\epsilon,\qquad
\Pr(E_z^c)\le\epsilon^N.
\tag{SIAT.7}
$$
This bounds the complete Gaussian tails without removing them.
Independence of positions, unaffected by the actual second
field and cap, proves the extinction bound.
Linearity gives the expected dead-fraction assumptions in
(SIAT.2); no independence of velocities and marks is asserted.

All alive positions lie in $[-2,2]^3$, and stored speed is at
most two. Euclidean phase squared diameter is at most64.
Since $q_G\le(1+\beta)|\cdot|^2$, its $G$ squared diameter
is at most $1.04(64)<67=:D_*^2$.
The origin belongs to this set and is a valid auxiliary ghost.
Thus (SIAT.4) gives
$$
\mathbb W_{2,G}(\mathcal H_N(z_0),\mathcal H_N(z_1))
\le\sqrt{.99}\,w_0+5\sqrt{67}\,2^{-50}
<.995w_0+42\,2^{-50}.
$$
The integer comparison $42(100000)<2^{50}$, together with
$w_0\ge1/100$, bounds the last term by $.001w_0$.
This proves (SIAT.6). Sampling an optimal inner alive transport
pair from each coupled survivor empirical pair preserves each
own swarm-first sampled law. Averaging those plans proves its
distance is no larger than the outer distance.

For slot-first sampling, draw a common uniform index from the
two raw paired arrays. Replace a dead selected row by the
bounded origin in an auxiliary phase law, for comparison only.
On two alive selected rows its cost is the original carrier
cost; if either selected row is dead, its new cost is at most
$D_*^2$. Hence the coupled auxiliary phase laws have squared
distance at most
$(.99)w_0^2+2D_*^2\epsilon$.
Each auxiliary marginal is the mixture of its own
slot-conditioned alive law and its ghost, with ghost mass at
most $\epsilon$. Each of its two conditioning repairs costs
at most $D_*\sqrt\epsilon$.
The phase metric triangle therefore gives
$$
W_{2,G}(\ell_N^{\rm slot}(z_0),\ell_N^{\rm slot}(z_1))
\le\sqrt{.99}\,w_0+(\sqrt2+2)D_*\sqrt\epsilon
<.995w_0+42\,2^{-50}.
$$
The same exact integer margin proves its stated coefficient.
Conditioning a raw selected row to be alive already implies
that swarm's nonextinction, so adding the own-swarm survivor
conditioning before the slot choice cancels exactly in the
final conditional ratio. The two sampling orders have not
been identified with each other.
:::

:::{prf:remark} Default proof scope after terminal transfer
:label: rem-siat-full-update-scope

The signed varying-velocity family now has an actual finite,
size-independent one-update kinetic alive empirical-law estimate,
including the noisy own provider, cap, terminal marks, both
sampling conventions and each own survival or alive denominator.
Unlike the physical all-slot theorem, this estimate pays the
entire Gaussian tail and uses the stated minimum separation.

A complete active update also applies its actual preparation to
the entering swarm. Membership of its resulting prepared law
in this exact balanced family is a separate hypothesis; this
note does not assert it for arbitrary swarms or after another
noisy update. Neither this coefficient nor (DSG.6) can therefore
be iterated as the unchanged default active convergence rate.
The general shape/velocity, source/component and delayed marked
feedback obligations remain in
{prf:ref}`thm-dlb-delayed-response` and
{prf:ref}`thm-dsti-delayed-moments`.
:::

