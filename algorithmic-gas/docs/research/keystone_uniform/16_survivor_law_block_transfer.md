# Whole-horizon survivor transfer from a complete raw block

This note separates a proved normalization calculation from its remaining raw
coupling premise. It does not claim that the dense viscous gas has already
discharged that premise. The calculations use the original absorbing kernel,
with all retained dead coordinates, sampled global fitness normalizers, donor
and collision dependence, both viscous kicks and Gaussian innovations present.
No restart or stepwise survivor-rejection kernel replaces it.

## 1. The original absorbing kernel and its exact block normalization

Let $\mathcal X_N$ be the complete physical marked array space, let
$E_N=\{M>0\}$ be its nonextinct subset, and let $P_N$ be the original
absorbing Markov kernel. Its killed restriction is
$Q_N(S,B)=P_N(S,B\cap E_N)$ for $S\in E_N$. An all-dead state stays
absorbed; its retained coordinates need not be identified with a single point.
Write $\tau_\dagger$ for the first all-dead update.

Assume a discharged uniform row floor $a\in(0,1)$, as in
{prf:ref}`thm-ku-uniform-alive-inverse-moments`, and put

$$
e_N=(1-a)^N,\qquad
u_N(m)=1-(1-e_N)^m\le me_N.
\tag{KUST.1}
$$

Then $P_N(S,E_N^c)\le e_N$ for every nonextinct input. Define

$$
\Phi_{N,m}(\mu)=\frac{\mu Q_N^m}{\mu Q_N^m1},\qquad
\eta_n=\frac{\eta_0Q_N^n}{\eta_0Q_N^n1}.
\tag{KUST.2}
$$

Every denominator is positive. The Markov property and absorption give

$$
\eta_n=\Phi_{N,m}(\eta_{n-m}),\qquad n\ge m,
$$
$$
\|\Phi_{N,m}(\mu)-\mu P_N^m\|_{\rm TV}
=1-\mu Q_N^m1\le u_N(m).
\tag{KUST.3}
$$

Indeed, $\mu P_N^m$ is the mixture of its surviving and extinct conditional
laws on the disjoint marked events $E_N$ and $E_N^c$. Its distance to the
surviving conditional law is exactly the extinct mixture mass. Iterating the
one-update lower survival probability gives
$\mu Q_N^m1\ge(1-e_N)^m$. The first identity follows by cancelling the
normalizer of $\eta_{n-m}$.

In particular this is a whole-block statement. If
$q_m(S)=Q_N^m1(S)$ and
$H_m(S,\cdot)=Q_N^m(S,\cdot)/q_m(S)$, then exactly

$$
\Phi_{N,m}(\mu)=\frac{q_m\mu}{\mu q_m}\,H_m.
\tag{KUST.4}
$$

The input tilt is the future survival probability for the entire remaining
block. Replacing this formula by $\mu H_m$ discards that tilt.

## 2. A quantitative conditional transfer theorem

Let $\rho_N\le1$ be a measurable metric on the complete marked array space,
and let $D_N=\mathcal W_{\rho_N}$ be its law transport distance. In
particular $D_N\le\|\cdot\|_{\rm TV}$ by the coupling inequality.
The raw coupling below is required to be measurably available as a function
of its two input arrays, as is the case for an explicit coupled-update proof.

Let $G_N\subset E_N$ be an actual input class. Suppose current-time survivor
laws and an actual QSD $\nu_NQ_N=\alpha_N\nu_N$ satisfy

$$
\eta_n(G_N^c)\le\delta_N\quad(n\ge1),\qquad
\nu_N(G_N^c)\le\delta_N.
\tag{KUST.5}
$$

For $G_N=\{M/N\ge a/2\}$ the cited alive-floor theorem discharges this
with
$\delta_N=\exp[-(1-\log2)aN/2]$. It does not discharge (KUST.5) for a
particular spatial well, population phase or sign-constrained velocity class.
Such a class needs its own current-law and QSD membership estimates.

The substantive raw premise is a fixed integer $m\ge1$, a fixed $q\in(0,1)$
independent of $N$, and an explicit defect $d_N\ge0$ such that

$$
D_N(\delta_SP_N^m,\delta_TP_N^m)
\le q\rho_N(S,T)+d_N\qquad(S,T\in G_N).
\tag{KUST.6}
$$

This is a complete-array law statement for the actual raw block, including
extinct outputs. A bound for only one sampled alive row does not imply it.
The metric must retain every physical coordinate and terminal mark needed
to control the next actual update. No autonomous Markov evolution of a
projected alive observation is assumed.

Under these hypotheses, define

$$
\mathcal E_N=d_N+2\delta_N+2u_N(m),\qquad
k(n)=\left\lfloor\frac{n-1}{m}\right\rfloor\quad(n\ge1).
$$

Then

$$
\boxed{\quad
D_N(\eta_n,\nu_N)
\le B_{N,n}:=
\min\left\{1,q^{k(n)}+
       \frac{\mathcal E_N[1-q^{k(n)}]}{1-q}\right\},\qquad n\ge1.
\quad}
\tag{KUST.7}
$$

To prove it, couple two entering laws $\mu,\zeta$ within an arbitrary
$\epsilon$ of their $D_N$ distance. On $G_N\times G_N$, use the raw
coupling (KUST.6); on the remaining input pairs use any coupling at cost
at most one. The probability of such a remaining pair is at most
$\mu(G_N^c)+\zeta(G_N^c)\le2\delta_N$. Integrating the constructed
coupling and taking $\epsilon\downarrow0$ gives

$$
D_N(\mu P_N^m,\zeta P_N^m)
\le qD_N(\mu,\zeta)+d_N+2\delta_N.
$$

Apply (KUST.3) separately to the two marginals and use the triangle
inequality. This proves

$$
D_N(\Phi_{N,m}\mu,\Phi_{N,m}\zeta)
\le qD_N(\mu,\zeta)+\mathcal E_N.
\tag{KUST.8}
$$

For the actual QSD, $\Phi_{N,m}\nu_N=\nu_N$. Write
$n=r+km$ with $1\le r\le m$. Both $\eta_r$ and $\nu_N$ satisfy
(KUST.5), their initial distance is at most one, and every later current
survivor input again satisfies that same bound. Iterate (KUST.8) exactly
$k=k(n)$ times to obtain (KUST.7). Nothing in this argument conditions a
paired process on both swarms surviving. Each normalization correction
uses its own original marginal survival event.

If (KUST.6) holds globally on $E_N$, set $\delta_N=0$, start at
$r\in\{0,\ldots,m-1\}$, and replace $k(n)$ by $\lfloor n/m\rfloor$.
The global conclusion then also applies at $n=0$.

The time rate in (KUST.7) is independent of $N$. Its particle floor
vanishes whenever $d_N\to0$ and the actual discharged $\delta_N\to0$.
In the alive-floor class the normalization and coverage terms are exponential
in $N$. The physical block time is $mh$. Choosing $m=m_N\to\infty$
would instead lose this particular population-independent physical time
rate unless a finer raw time estimate is proved.

The additive defect does not itself prove finite-$N$ uniqueness of the QSD.
Existence and any claimed uniqueness must come from the actual finite-particle
QSD theorem. The conclusion above applies to every QSD satisfying (KUST.5).

## 3. Exact normalized-alive observations for a Hamming raw certificate

For this section take

$$
\rho_N(S,T)=\frac1N\sum_{i=1}^N
\mathbf1_{\{(x_i,v_i,a_i)\ne(\widetilde x_i,
                                      \widetilde v_i,\widetilde a_i)\}},
\qquad m_*=a/2,
$$
$$
\mu_S^a=\frac1{M_S}\sum_{i:a_i=1}\delta_{(x_i,v_i)}
\quad(S\in E_N).
\tag{KUST.9}
$$

Let $A$ be the unequal full-row labels. Let $C$ be the number of labels
outside $A$ that are alive in both arrays, and set
$M_{\max}=\max(M_S,M_T)$. There are at least $M_{\max}-|A|$ such
common alive labels. Couple their identical phase points with mass
$1/M_{\max}$ per label. That mass is no greater than either required
single-label probability, and the residual marginals have the same total
mass. Any coupling of those residual marginals completes a valid transport
of the two separately normalized alive empirical measures. Hence

$$
\|\mu_S^a-\mu_T^a\|_{\rm TV}
\le1-\frac{C}{M_{\max}}
\le\min\left\{1,\frac{|A|}{M_{\max}}\right\}.
\tag{KUST.10}
$$

For $S,T\in G_N=\{M/N\ge m_*\}$ this is at most
$\rho_N(S,T)/m_*$. This coupling preserves each own uniform-alive
sampling law. It is not uniform slot sampling conditional on a shared
or joint alive event.

For any coupling of two array laws supported on $E_N$ with bad-alive
masses at most $\delta_N$, the same construction gives

$$
\mathbb E\|\mu_S^a-\mu_T^a\|_{\rm TV}
\le\min\{1,\mathbb E\rho_N(S,T)/m_*+2\delta_N\}.
\tag{KUST.11}
$$

The common-label mass in (KUST.10) explicitly couples the own normalized
alive observations conditionally on the two arrays. Averaging this coupling
proves the same upper bound for the TV distance between the mean alive
measures. Pushing the array coupling to its two empirical measures also
bounds transport between their laws with the ground cost given by TV.
Take array couplings within an arbitrary $\epsilon>0$ of $D_N(\mu,\zeta)$
and then let $\epsilon\downarrow0$. Both observation distances are therefore
at most $\min\{1,D_N(\mu,\zeta)/m_*+2\delta_N\}$; no existence of an
optimal coupling is required.

If the original terminal region is $D=[-L_D,L_D]^d$, alive phase points
are in $D\times\overline B_V$. For fixed positive definite $G$ its
squared diameter in the phase metric is at most

$$
\mathcal B_G=4\lambda_{\max}(G)(dL_D^2+V^2).
\tag{KUST.12}
$$

No restriction on retained dead positions is used in this bound.
On this alive phase space,
$W_{2,G}^2(\xi,\zeta)\le\mathcal B_G\|\xi-\zeta\|_{\rm TV}$:
keep the common part on the diagonal and transport the residual mass
at the diameter bound. Combining (KUST.7) and (KUST.11), let

$$
b_{N,n}=\min\{1,B_{N,n}/m_*+2\delta_N\}.
$$

Then both the mean alive law and the law of the actual normalized alive
empirical measure satisfy

$$
\|\lambda_{N,n}^a-\lambda_N^{a,\nu}\|_{\rm TV}\le b_{N,n},
\qquad
W_{2,G}^2(\lambda_{N,n}^a,\lambda_N^{a,\nu})
\le\mathcal B_Gb_{N,n},
$$
$$
\mathcal W_{W_{2,G}}^2(\operatorname{Law}_{\eta_n}(\mu_S^a),
                       \operatorname{Law}_{\nu_N}(\mu_S^a))
\le\mathcal B_Gb_{N,n}.
\tag{KUST.13}
$$

Here $\lambda_{N,n}^a=\int\mu_S^a\eta_n(dS)$ and
$\lambda_N^{a,\nu}=\int\mu_S^a\nu_N(dS)$ are their own mean alive
probabilities. These conclusions retain the actual alive-count denominators.
If the raw block uses another marked metric, (KUST.10) must be replaced by
its proved observation modulus. A quadratic metric without terminal-status
control cannot silently substitute for Hamming in these inequalities.

## 4. Future-horizon conditioning requires a separate survival-weight bound

For $0\le j\le T$ the original law at the earlier observation time,
conditioned on survival through the later horizon, is exactly

$$
\eta_{j\mid T}(dS)=
\frac{Q_N^{T-j}1(S)\eta_j(dS)}{\eta_jQ_N^{T-j}1}.
\tag{KUST.14}
$$

The block recurrence (KUST.7) concerns $\eta_j$, whose horizon is the current
observation time. Rare one-step extinction alone bounds the difference in
(KUST.14) by $u_N(T-j)$, which need not vanish uniformly in an arbitrarily
long future horizon.

A separate *full-array* actual survivor-block coefficient

$$
\beta_N(m')=\sup_{S,T\in E_N}
\left\|\frac{Q_N^{m'}(S,\cdot)}{Q_N^{m'}1(S)}
      -\frac{Q_N^{m'}(T,\cdot)}{Q_N^{m'}1(T)}\right\|_{\rm TV}
$$

with $\beta_N(m')+2u_N(m')<1$ controls this additional tilt. The completed
finite-future normalization calculation gives, for every $L\ge0$,

$$
\frac{\sup Q_N^L1}{\inf Q_N^L1}
\le R_N:=\frac{1-\beta_N(m')-u_N(m')}
                    {1-\beta_N(m')-2u_N(m')},
$$
$$
\|\eta_{j\mid T}-\eta_j\|_{\rm TV}
\le\chi(R_N):=\frac{\sqrt{R_N}-1}{\sqrt{R_N}+1}.
\tag{KUST.15}
$$

For completeness, write $K=Q_N^{m'}$, $u=u_N(m')$, $\beta=\beta_N(m')$
and $t(f)=\operatorname{osc}(f)/\inf f$. The normalized rows of $K$
have Dobrushin coefficient $\beta$, while their masses are in $[1-u,1]$.
For every bounded positive $f$,

$$
\operatorname{osc}(Kf)\le\beta\operatorname{osc}(f)+u\sup f,
\qquad\inf Kf\ge(1-u)\inf f.
$$

Thus
$t(Kf)\le[(\beta+u)/(1-u)]t(f)+u/(1-u)$.
The coefficient is strictly below one, with fixed value
$u/(1-\beta-2u)$. For $L=km'+r$ with $0\le r<m'$, the starting
function $Q_N^r1$ has $t\le u/(1-u)$, no larger than that fixed value.
Iteration proves the first inequality of (KUST.15).

For the sharp reweighting bound, let $a\le w\le b$, $a>0$, and set
$z=\mu w$. If $a=b$, reweighting leaves the law unchanged. Otherwise
the chord bound
$(w-z)_+\le(b-z)(w-a)/(b-a)$ gives

$$
\|w\mu/\mu w-\mu\|_{\rm TV}
=\frac{\mu(w-z)_+}{z}
\le\frac{(b-z)(z-a)}{z(b-a)}.
$$

This expression is maximized at $z=\sqrt{ab}$, giving
$(\sqrt{b/a}-1)/(\sqrt{b/a}+1)$. Apply it to the actual future weight
$w=Q_N^{T-j}1$ to finish (KUST.15).

Alive empirical observations and uniform-alive sampling cannot increase this
TV bound. Their squared physical Wasserstein comparisons are therefore at
most $\mathcal B_G\chi(R_N)$. Combining those observation comparisons with
(KUST.13) for $j\ge1$ gives an upper bound
$\mathcal B_G\min\{1,b_{N,j}+\chi(R_N)\}$ toward the QSD alive
observations, using the triangle inequality first in transport with the
ground cost TV. This avoids an unnecessary factor two from squaring a
Wasserstein triangle inequality.

When $\sup_N\beta_N(m'_N)<1$ and
$u_N(m'_N)/(1-\beta_N(m'_N))\to0$, the future-weight error vanishes
uniformly in $T-j$. These are quantitative requirements on an actual
full-array block. The block $m'_N$ used only for this tilt estimate may
grow with $N$; the current-time rate in (KUST.7) still uses its own fixed
raw block length $m$.

## 5. What is proved and what remains a dense-kernel premise

The original uniform row floor, inverse alive-mass bounds and current-time
survivor moments are discharged in Chapter 18a Section 6. The entire-block
normalization identities (KUST.2)--(KUST.4), the transfer of a supplied raw
complete block, the exact alive-denominator coupling and the finite-future
weight calculation above need no additional joint-survival assumption.

The dense viscous raw contraction (KUST.6) remains a separate mathematical
task. {prf:ref}`thm-slcw-finite-uniform-law` discharges a related complete
Hamming calculation for its conservative, nonviscous, bounded-reward,
force-center regime. It cannot be imported as a theorem for the absorbing
dense kernel without verifying its different revival, kick and mark terms.
An averaged energy drift or a sampled-alive-row estimate also does not close
the full-state premise merely by having population-independent coefficients.

The landscape-resolved phase identities in
`11_landscape_resolved_qsd.md` retain the actual crossing scale, centered
killing correction and eigenfunction phase weights. The full-state block
coefficient in (KUST.15) is not supplied by those identities or by symmetry
of individual well orbits. A local phase input class must charge its actual
exit and current-law membership errors in (KUST.5); its local comparison
cannot replace the global supremum in $\beta_N(m')$.
