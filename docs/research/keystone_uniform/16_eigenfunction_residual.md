# Actual singleton renewals and the retained eigenfunction residual

(sec-kuer-retained)=
## 1. Exact retained endpoint and the two routes

:::{prf:definition} Retained actual-kernel state and endpoint
:label: def-kuer-retained-state

The source revision is 6107b67b9e85259581c1932565c1871e8a7e253a.
The retained mathematical objects are the real-arithmetic canonical
terminal-box kernels of Chapter 18, with its complete measurement,
companion, fitness, acceptance, mandatory-revival, component-Haar,
clone-jitter, two-viscous-kick, thermostat, position-noise, cap and
terminal-mark record. The force/reward alternatives are separately
tagged: the unchanged quadratic reference, and the native Rastrigin
potential and reward. They are not combined as simultaneous force
hypotheses. Both configured count and Gaussian row normalizations are
retained. All primitive parameters other than the population size are
fixed. A passive recording extension has no feedback.

Write \(P_N\) for the raw transition and \(Q_N\) for its restriction to
nonextinction. Its one-step extinction probability is
\(\kappa_N(S)=1-Q_N1(S)\). For each finite population whose established
QSD and positive bounded eigenfunction share the eigenvalue, use
\[
\nu_NQ_N=\alpha_N\nu_N,\qquad Q_Nh_N=\alpha_Nh_N.
\]
The requested endpoint is
\[
\sup_{N\ge1}
\frac{\sup_{k\in\mathcal K_N}h_N(k)}
     {\inf_{k\in\mathcal K_N}h_N(k)}<\infty,
\tag{KUER.1}
\]
on the actual effective-input compactification \(\mathcal K_N\).
Alive positions and every retained velocity enter this object.
Dead positions are reset before being used as a source or force
coordinate; their entering measurement and companion features still
enter the preparation law and are retained through its established
feature compactification. Dead velocities are retained. Only at a
singleton does the unique living donor make its source choice
independent of those dead-position features. The full-array eigenfunction
is not replaced by an empirical observable or a tagged statistic.
No equilibrium phase is declared closed.

The accepted inputs are the component-energy identity and normalized
Gaussian-column bounds; the actual alive-count transforms, current-time
survivor moments and stopped identity; the exact phase flux/killing
balance; and the existing finite-\(N\) spectral certificates at their
stated parameter domains. Snapshot hashes of the principal incoming
research proofs are:

- Conditioning: 052ac7c157c18d4e2835649a1b9cf072094689598f99510c8370da0dc884a92e.
- Averaged energy: 52e67566171abc2abd2666da988c4fae259bcdce36a4212e22fc45655be0a3bf.
- Phase balance: c112d06fe4c12fdbeef704e837c6fcdad226ba6e11a38f2a0fe7e867e68ea602.

These identify the imported arguments, including their explicit
conditional hypotheses. They do not promote an assumed within-phase
eigenfunction bound to an accepted input.
:::

:::{prf:remark} The actual structural tension and its direct consumer
:label: rem-kuer-route-choice

The global-ratio route asks whether actual renewal and communication
erase survival-relevant input information before killing distinguishes
that information. The existing positive full-array density constants
alone do not compare those two clocks uniformly in \(N\). The new move
below instead compares original killing directly with an original
near-extinction event, followed by the already configured mandatory
revival. It uses neither arbitrary-swarm attraction nor a modified
conservative algorithm.

The serious alternative is the existing direct survivor route.
Current-time alive moments, physical finite-horizon limits and QSD
population-map invariance already have exact normalized-kernel
interfaces without a right eigenfunction change of measure. Chapter
15's complete survivor entropy chain rule has the same property.
The bounded Doob entropy transfers (15.40)--(15.41), however, still
require their actual bounded reweighting cost to be uniform. Completing
a direct physical-law consumer does not prove (KUER.1).
:::

(sec-kuer-relative-renewal)=
## 2. Original near-extinction events are more frequent than killing

:::{prf:lemma} Dimension-independent logarithmic bound for Gaussian death
:label: lem-kuer-death-log-lipschitz

Let \(D=[-L_D,L_D]^d\), \(\sigma>0\), and
\[
q_D(m)=\Pr(m+\sigma Z\notin D),\qquad
H_D=\frac{\phi(L_D/\sigma)}
          {\sigma\Phi(-L_D/\sigma)}.
\]
Then, for all \(m,m'\in\mathbb R^d\),
\[
|\log q_D(m')-\log q_D(m)|\le H_D|m'-m|,
\qquad
H_D\le \frac{L_D}{\sigma^2}+\frac1{\sigma}.
\tag{KUER.2}
\]
When \(L_D/\sigma\ge1\), the sharper elementary upper bound
\(H_D\le L_D/\sigma^2+1/L_D\) also applies.
:::

:::{prf:proof}
For one coordinate put
\(q_r(u)=\Phi((u-L_D)/\sigma)+\Phi((-u-L_D)/\sigma)\).
By symmetry it suffices to take \(u\ge0\). With
\(z=(L_D-u)/\sigma\le L_D/\sigma\),
\[
0\le q_r'(u)
\le \phi(z)/\sigma,\qquad q_r(u)\ge\Phi(-z).
\]
The Gaussian hazard \(r(z)=\phi(z)/\Phi(-z)\) is increasing.
Indeed \(r'(z)=r(z)[r(z)-z]>0\): for \(z>0\), integrating
\(t/z>1\) over \(t>z\) gives \(\Phi(-z)<\phi(z)/z\);
for \(z\le0\) positivity is immediate. Thus
\(|q_r'|\le H_Dq_r\).

For the box, \(q_D=1-\prod_r(1-q_r)\). Consequently
\[
\sum_r|\partial_r\log q_D|
\le H_D
\frac{\sum_r q_r\prod_{s\ne r}(1-q_s)}
     {1-\prod_r(1-q_r)}
\le H_D.
\]
The numerator is the probability of exactly one coordinate escape;
the denominator is the probability of at least one. The Euclidean
gradient norm is no larger than this sum. Integration along a line
segment proves the first claim.

For \(z\ge1\), integration by parts gives
\(\Phi(-z)\ge z\phi(z)/(z^2+1)\), hence
\(r(z)\le z+1/z\le z+1\).
On \(0\le z\le1\), \(r(z)<z+1\) follows from the same differential
equation: it holds at zero and at one, and at any hypothetical
crossing its difference from \(z+1\) has derivative \(z>0\), so a
crossing cannot return to the negative value at one.
This proves both stated primitive upper bounds. \(\square\)
:::

:::{prf:lemma} Changing one own jitter has bounded total influence on other means
:label: lem-kuer-one-jitter-influence

Freeze the actual pre-jitter source, gate and component-Haar pattern,
including its retained collision velocities \(v_i^{\rm col}\).
Let \(a=t\nu\in[0,1]\), \(U=W_Xv^{\rm col}\), and let
only the prepared position \(X_k\) change. Every other row's final
Gaussian mean changes only by its actual first viscous velocity term.
The sum of its changes is bounded by
\[
\sum_{i\ne k}|M_i'-M_i|
\le\Lambda_N,\qquad
\Lambda_{\rm count}=2baV_c,\qquad
\Lambda_{\rm row}=4baV_c C_d,
\tag{KUER.3}
\]
where \(C_d\) is any independently proved dimension-only incoming
column envelope for the actual Gaussian row matrix.
Under \(|\alpha_{\rm col}|\le1\), the complete component-energy
identity improves the count bound to
\[
\Lambda_{\rm count}=ba(V_c+V).
\tag{KUER.4}
\]
For \(N=1\) the actual sum is zero.
:::

:::{prf:proof}
In count normalization, for \(i\ne k\),
\[
U_i'-U_i=\frac aN(K'_{ik}-K_{ik})
(v_k^{\rm col}-v_i^{\rm col}).
\]
Here \(|K'_{ik}-K_{ik}|\le1\). Summation and
\(|v_i^{\rm col}|\le V_c\) prove the first count bound.
The stronger one uses
\[
\frac1N\sum_{i\ne k}|v_k^{\rm col}-v_i^{\rm col}|
\le |v_k^{\rm col}|+\|v^{\rm col}\|_{2,N}
\le V_c+V.
\]
The last inequality is the actual frozen-component energy identity,
including every retained dead velocity.

In row normalization only one unnormalized entry changes in each
other row. Its normalized row changes in total variation norm by
\(2|\omega'_{ik}-\omega_{ik}|
\le2(\omega'_{ik}+\omega_{ik})\).
Multiplying by \(aV_c\) and summing both actual incoming columns
gives \(4aV_cC_d\). The force at another row's own \(X_i\) does
not change. Multiply both bounds by \(b\).
No Gaussian degree floor, jitter cutoff or independence of
\(W_X\) from the jitter is used. \(\square\)
:::

:::{prf:theorem} Primitive singleton-core probability relative to actual extinction
:label: thm-kuer-singleton-vs-extinction

Let \(C\subset D\) be a nonempty box and let
\[
\mathcal R_{C,N}=\{S:\ M(S)=1,\ x_{\rm survivor}\in C\}.
\]
Use the actual conditional position variance
\(\sigma_h^2=t^2q^2+s^2>0\).
Suppose \(a_{0,C}>0\) is a uniform own-row \(C\)-landing floor for
an uncopied row, and \(a_{1,C}>0\) is a uniform own-jitter-integrated
floor for a copied row after the proved bound
\(|bU_k|\le R=bV_c\).
Both floors must be computed from the actual configured force.
Then every entering nonextinct state satisfies
\[
\boxed{\quad
P_N(S,\mathcal R_{C,N})
\ge N c_C\kappa_N(S),\qquad
c_C=\min\{a_{0,C},a_{1,C}e^{-H_D\Lambda_N}\}>0.
\quad}
\tag{KUER.5}
\]
The constants are independent of \(N\) when the displayed envelope
for \(\Lambda_N\) is. The singleton events for different surviving
indices are disjoint, so the factor \(N\) has no multiplicity loss.
:::

:::{prf:proof}
Freeze the complete actual pre-jitter pattern. Conditional on all its
jitters, the final position Gaussians are independent. Let \(E_k(C)\)
be the event that exactly row \(k\) survives and lands in \(C\). Write
\[
T_k=\prod_{i\ne k}q_D(M_i),\qquad
\kappa_{\rm pat}=\mathbb E[q_D(M_k)T_k].
\]
For an uncopied row, its \(C\)-probability is at least \(a_{0,C}\)
pointwise, regardless of all other jitters. Since \(q_D(M_k)\le1\),
\(\Pr(E_k(C)\mid{\rm pat})\ge a_{0,C}\kappa_{\rm pat}\).

For a copied row replace only its own independent jitter by a
second independent copy integrated against exactly the same Gaussian
law. By (KUER.2)--(KUER.4), for all two realizations of that own jitter
and the same remaining inputs,
\[
T_k(X_k')\ge e^{-H_D\Lambda_N}T_k(X_k).
\]
The pointwise shift bound in the theorem supplies a function of the
new own jitter whose integrated \(C\)-landing probability is at least
\(a_{1,C}\), independently of the other jitters. Multiply the displayed
comparison by that function and integrate the two own-jitter copies.
It follows that
\[
\Pr(E_k(C)\mid{\rm pat})
\ge a_{1,C}e^{-H_D\Lambda_N}\mathbb ET_k
\ge a_{1,C}e^{-H_D\Lambda_N}\kappa_{\rm pat}.
\]
Every shared donor and component rotation was frozen before the
independent own jitters. Their later mixture preserves the bounds.
Sum the disjoint \(E_k(C)\) and average the entire original pattern
law. This proves (KUER.5). \(\square\)
:::

:::{prf:corollary} Stronger extinction-weighted energy coefficient
:label: cor-kuer-optimized-energy-singleton

For \(C=[-L_C,L_C]^d\), let \(G_0\) be a proved bound on the
absolute coordinates of the zero-jitter force-drift map. Put
\[
z_C=\Phi^{-1}\!\left(
 [\ell_{L_C,\sigma_h}(G_0)]^d\right),\qquad
f_C(r)=\Phi(z_C-br/\sigma_h),\qquad
g_{1,C}=a_{1,C}e^{-H_D\Lambda_N}.
\]
Assume \(z_C\le0\). Let \(V_E\) be the actual first-kick normalized
energy envelope: \(V_E=V\) in count normalization, and, in row
normalization, the proved Minkowski envelope
\[
V_E=\min\{V_c,V[1-a+a\sqrt{C_d}]\}.
\]
Then (KUER.5) remains true with the larger explicit coefficient
\[
c_{E,C}=\inf_{0\le u\le1}
\left\{u f_C(V_E/\sqrt u)+(1-u)g_{1,C}\right\},
\tag{KUER.6}
\]
where the first term is zero at \(u=0\).
The additional actual row bound permits replacing
\(V_E/\sqrt u\) by \(\min\{V_c,V_E/\sqrt u\}\), if useful.

For the expression as written, set
\(D_r=f_C(r)-rf_C'(r)/2\).
It is strictly convex in \(u\). If \(g_{1,C}\ge D_{V_E}\), its
minimum is \(f_C(V_E)\). Otherwise its unique minimizer is
\[
D_{r_*}=g_{1,C},\qquad u_*=(V_E/r_*)^2.
\tag{KUER.7}
\]
These statements give a primitive one-dimensional optimization;
they do not assume independent alignment velocities.
:::

:::{prf:proof}
For a fixed pattern with uncopied fraction \(u\), divide each
uncopied singleton probability integrand by its conditional
extinction integrand. Its ratio is \(p_{i,C}/q_D(M_i)\ge p_{i,C}\).
The Gaussian shift lemma and the pathwise energy certificate give
\[
\sum_{I_i=0}p_{i,C}
\ge uNf_C(V_E/\sqrt u).
\]
This is pointwise in the entire jitter array, so it remains valid
when that array is tilted by its original extinction product.
The copied rows supply \((1-u)Ng_{1,C}\kappa_{\rm pat}\) by
(KUER.5). This proves (KUER.6) after every pattern is averaged.
Since \(z_C\le0\), \(f_C\) is decreasing and convex on
\([0,\infty)\). Differentiation gives
\[
\frac d{du}[uf_C(V_E/\sqrt u)]=D_{V_E/\sqrt u},\qquad
D_r'=\tfrac12f_C'(r)-\tfrac r2f_C''(r)<0.
\]
The asserted convexity, endpoint case and unique root follow.
The pointwise row cap gives the additional strengthened argument.
\(\square\)
:::

:::{prf:example} Actual quadratic and native Rastrigin coefficients
:label: ex-kuer-original-reference-singleton

Retain the unchanged kinetic, noise, cap, viscosity and fitness
record, with \(d=3\), \(L_D=2\), \(h=.04\), \(\nu=.3\),
\(\sigma_J=.1\), \(V=2\), \(V_c=4\).
Let \(\eta=bt\), \(R=bV_c\).

For the quadratic alternative,
\[
A_Q=|1-\eta|,\quad G_0=2A_Q,\quad
a_{1,C}=
\ell_{L_C-R,\sqrt{A_Q^2\sigma_J^2+\sigma_h^2}}(2A_Q)^3.
\tag{KUER.8}
\]
For native Rastrigin use its actual coordinate map
\[
\psi(x)=A_Rx-K_R\sin(2\pi x),\qquad
A_R=1-2\eta,\quad K_R=20\pi\eta.
\]
Here \(A_R-2\pi K_R>0\), so the map is increasing on the whole
line. Its exact zero-jitter range on \([-2,2]\) is
\([-2A_R,2A_R]\), because the endpoint sine is zero.
Thus \(G_0=2A_R\), without a superfluous periodic-force penalty.
A conservative actual copied floor is
\[
a_{1,C}=
\ell_{L_C-R-K_R,\sqrt{A_R^2\sigma_J^2+\sigma_h^2}}(2A_R)^3
\tag{KUER.9}
\]
when the eroded width is positive. This integrates the complete
unbounded jitter: the periodic term is charged by its actual bounded
amplitude while the linear part is convolved exactly.
For smaller boxes use the strictly positive original integral
of (KLQ.5a)--(KLQ.5c), or a finite replacement-jitter proof event
with its exact Gaussian probability. A failed eroded width is not
identified with a zero original landing probability.

With the exact \(H_D\) and the stronger count influence bound,
the diagnostic evaluations of the proved formulas are
\[
\begin{array}{c|c|c}
\text{force}&L_C&c_{E,C}\\\hline
\text{quadratic}&2&8.85143246722\,10^{-8}\\
\text{quadratic}&1.99&1.01768960879\,10^{-8}\\
\text{native Rastrigin}&2&6.45405748199\,10^{-9}\\
\text{native Rastrigin}&1.99&2.12386955231\,10^{-9}
\end{array}
\]
Here \(H_D\simeq4815.40610562224\) and
\(H_D\Lambda_{\rm count}\simeq6.79823815473627\).
The simpler explicit upper bound \(L_D/\sigma_h^2+1/\sigma_h\)
also yields positive coefficients, respectively about
\(8.36778\,10^{-8}\), \(1.01769\,10^{-8}\),
\(6.05642\,10^{-9}\), and \(2.00446\,10^{-9}\).
The exact formulas, not rounded decimals, define the constants.

The finite Gaussian-column series of research draft 17, with its
proved 40th truncation \(C_3<7188\), can now be imported.
With this envelope and the coarser explicit \(H_D\) above, the
row certificate is positive but very weak: for \(C=D\), its
direct \(g_{1,C}\) has log approximately \(-131631.111290\)
for the quadratic alternative and \(-131634.153993\) for
Rastrigin. The complete noise law has not been clipped to obtain
this improvement. The two normalization alternatives remain separate.
:::

(sec-kuer-stopped-compression)=
## 3. A proved compression of the minimum and the exact remaining return kernel

:::{prf:corollary} Explicit coverage of the central source class
:label: cor-kuer-central-singleton-coverage

Take \(C=[-.01,.01]^3\) and \(r_x=.001\).
For either configured force, the copied-row floor in (KUER.5) can
be taken as the exact positive primitive expression
\[
 a_{1,C}^{\rm core}
 =\ell_{r_x,\sigma_J}(L_D)^3
       \ell_{.01,\sigma_h}(r_x+R)^3.
\tag{KUER.31}
\]
The uncopied floor can be taken as
\[
 a_{0,C}=\ell_{.01,\sigma_h}(G_0+R)^3.
\tag{KUER.32}
\]
Thus the central singleton cutset in the native comparison has
an explicit coefficient
\[
 c_C=\min\{a_{0,C},
          a_{1,C}^{\rm core}e^{-H_D\Lambda}\}>0
\]
for every original population and all original effective inputs.
The reference bounds are, conservatively,
\[
 c_{C,\rm count}>e^{-17100},\qquad
 c_{C,\rm row}>e^{-133000}.
\tag{KUER.33}
\]
They establish coverage, not a practically useful return time.
:::

:::{prf:proof}
The event \(|X_{kr}|\le r_x\) in all three coordinates has
probability at least \(\ell_{r_x,\sigma_J}(L_D)^3\), uniformly
over the original donor source. On this event the quadratic map
and the native monotone map both satisfy
\(|\psi(X_{kr})|\le r_x\). The actual viscous velocity offset
has each coordinate at most \(R\), so the full independent final
Gaussian gives the second factor in (KUER.31). This lower event
integrates the complete unchanged Gaussian law; it does not
condition or modify the algorithm. The no-jitter mean bound gives
(KUER.32).

For the coarse displayed evaluations, use
\(R<.158\), \(G_0<2\),
\(.000415<\sigma_h^2\), \(\sigma_h<.021\), and
\((2\pi)^{-1/2}>1/3\). The minimum Gaussian density over each
interval gives \(a_{0,C}>e^{-17100}\) and
\(a_{1,C}^{\rm core}>e^{-724}\). The stronger count influence
has \(H_D\Lambda<7\). In row mode the imported \(C_3<7188\)
and coarser death-gradient bound give
\(H_D\Lambda<131624\). The two minima prove (KUER.33).
All these inequalities involve scalar probabilities and bounds
independent of \(N\). \(\square\)
:::

:::{prf:theorem} First singleton hit before original extinction
:label: thm-kuer-singleton-first-hit

Let \(c_C>0\) be any proved coefficient above, \(r_N=Nc_C\),
and \(\mathcal R=\mathcal R_{C,N}\). From every state outside
\(\mathcal R\), the original chain satisfies
\[
\Pr_S(\tau_{\mathcal R}<\tau_\dagger)
\ge\frac{r_N}{1+r_N}.
\tag{KUER.10}
\]
The first return from a state in \(\mathcal R\), with its time
required to be at least one, has the same lower survival probability.
For the associated eigenfunction,
\[
\boxed{\quad
\inf_{\mathcal K_N}h_N
\ge\frac{r_N}{1+r_N}\inf_{\mathcal R}h_N.
\quad}
\tag{KUER.11}
\]
Consequently the global minimum is controlled by its restriction
to actual singleton states with an \(N\)-independent loss
at most \((1+c_C)/c_C\). This is a productive reduction of one
half of (KUER.1), not its closure.
:::

:::{prf:proof}
For every Gaussian mean the death probability is at least
\[
q_*=
1-[2\Phi(L_D/\sigma_h)-1]^d>0.
\]
Conditional final rows are independent, hence
\(\kappa_N(S)\ge q_*^N\) for every effective input.
On each step before hitting \(\mathcal R\) or death, the
probability of the former is at least \(r_N\) times that of the
latter. Their combined probability is at least
\(\lambda_N=(1+r_N)q_*^N>0\).
Therefore the stopping time is almost surely finite, with
tail at most \((1-\lambda_N)^n\).
Sum the disjoint stopped hitting and killing probabilities over
their actual preceding path law. The stepwise comparison gives
\(\Pr(\tau_{\mathcal R}<\tau_\dagger)
\ge r_N\Pr(\tau_\dagger<\tau_{\mathcal R})\).
Their sum is one, proving (KUER.10). The same argument starts
immediately after time zero for a first return.

The stopped positive eigenfunction martingale gives, at every
finite horizon,
\[
h_N(S)\ge
\mathbb E_S[\alpha_N^{-\tau_{\mathcal R}}
h_N(S_{\tau_{\mathcal R}});
\tau_{\mathcal R}\le T,\ \tau_{\mathcal R}<\tau_\dagger].
\]
Monotone convergence and \(\alpha_N^{-\tau_{\mathcal R}}\ge1\)
give (KUER.11). A state already in \(\mathcal R\) satisfies the
same inequality trivially. No terminal remainder has been declared
zero to obtain this one-sided bound. \(\square\)
:::

:::{prf:proposition} Exact first-hit and first-return operators without eigenfunction weights
:label: prop-kuer-unweighted-return-residual

Partition the nonextinct effective states into
\(\mathcal R\) and \(\mathcal B=\mathcal K_N\setminus\mathcal R\).
Write the actual subkernel in these blocks. At every finite
population with its established bounded positive eigenfunction
bounded away from zero, define
\[
\begin{aligned}
\mathcal F_N(\alpha)
&=\sum_{n\ge0}\alpha^{-(n+1)}
 Q_{\mathcal BB}^{\,n}Q_{\mathcal BR},\\
\mathcal L_N(\alpha)
&=\alpha^{-1}Q_{\mathcal RR}
 +\sum_{n\ge0}\alpha^{-(n+2)}
 Q_{\mathcal RB}Q_{\mathcal BB}^{\,n}Q_{\mathcal BR}.
\end{aligned}
\tag{KUER.12}
\]
At \(\alpha=\alpha_N\), the series are the actual discounted
first-hit and first-return kernels and
\[
h_N|_{\mathcal B}=\mathcal F_N(\alpha_N)h_N|_{\mathcal R},
\qquad
h_N|_{\mathcal R}=\mathcal L_N(\alpha_N)h_N|_{\mathcal R}.
\tag{KUER.13}
\]
These operators contain only the original kernel and the actual
eigenvalue. They do not contain unknown target eigenfunction weights
in their entries.

Writing
\[
F_N^+=\max\{1,\sup_{\mathcal B}\mathcal F_N(\alpha_N)1\},
\qquad
R_N^{\mathcal R}=
\frac{\sup_{\mathcal R}h_N}{\inf_{\mathcal R}h_N},
\]
the exact comparison is
\[
\frac{\sup_{\mathcal K_N}h_N}{\inf_{\mathcal K_N}h_N}
\le
\frac{1+r_N}{r_N}F_N^+R_N^{\mathcal R}.
\tag{KUER.14}
\]
Thus a genuine uniform closure through this compression must compute
\(\sup_NF_N^+\) and \(\sup_NR_N^{\mathcal R}\).
The unweighted first-return kernel has row survival mass at least
\(r_N/(1+r_N)\), but its discounted version in (KUER.12) also
retains the actual random physical waiting time. Those two kernels
cannot be interchanged.
:::

:::{prf:proof}
The Doob chain for the existing finite-\(N\) eigenfunction reaches
\(\mathcal R\) with a positive uniform one-step probability:
\[
\widehat Q_N(S,\mathcal R)
\ge \frac{\inf h_N}{\alpha_N\sup h_N}
r_Nq_*^N>0.
\]
Its probability of avoiding \(\mathcal R\) tends geometrically
to zero. This proves disappearance of the stopped eigenfunction
remainder at this fixed population and convergence of (KUER.12).
Equivalently the outside block has spectral radius strictly below
\(\alpha_N\). Expanding by the number of consecutive outside
transitions gives the two displayed series and the eigenfunction
identities. This use of the finite-\(N\) bound establishes the exact
interface only; it supplies no uniform constant.

The first identity gives
\(\sup_{\mathcal K_N}h_N\le F_N^+\sup_{\mathcal R}h_N\).
Combine it with (KUER.11) to get (KUER.14).
The unweighted return mass assertion is (KUER.10), starting its
clock at one. \(\square\)
:::

:::{prf:lemma} Exact matched-source harmonic mean-velocity response
:label: lem-kuer-harmonic-mean-velocity-response

For a single actual kinetic update suppose its two complete prepared
position arrays are identical and its collision velocity arrays differ
by the same vector \(z\) in every row. Couple the original thermostat
and final position Gaussians identically. For \(F(x)=-\lambda x\),
both configured Gaussian viscosity normalizations give exactly
\[
\Delta u_i=z,\qquad
\Delta w_i=cz,\qquad
\Delta y_i=bz,\qquad
\Delta v_i^{\rm uncapped}=(c-t\lambda b)z.
\tag{KUER.18}
\]
Thus the original nonexpansive radial cap gives
\[
\|\Delta v^+\|_{2,N}\le |c-t\lambda b|\,|z|.
\]
At the unchanged quadratic reference,
\(c-tb=.9600051233766623\) diagnostically. Every innovation remains
unbounded, and the second viscous kick has been retained.
This is a response estimate for the stated matched-source input;
it is not a claim that the capped mean mode remains invariant
under repeated complete updates or survival conditioning.
:::

:::{prf:proof}
The first Gaussian kick matrix is identical and row stochastic,
so it sends the constant difference to \(z\). The identical force
inputs cancel. The shared OU gives \(cz\), and the two drifts give
\(bz\). All second-stage positions differ by the same translation,
so their Gaussian pairwise distances and both second-kick matrices
are exactly equal. That common matrix sends \(cz\) to \(cz\),
and the harmonic force difference is \(-\lambda bz\).
This proves (KUER.18); cap nonexpansiveness proves the bound.
The rowwise cap may convert a constant uncapped difference into
different row differences, so no repeated mean-mode invariance
has been inferred. \(\square\)
:::

:::{prf:lemma} Original count-mode matched-source velocity comparison
:label: lem-kuer-matched-count-velocity

For one kinetic update, freeze a common source, accepted-component
and Haar pattern before the original independent clone jitters.
Suppose both prepared arrays use the same source point \(\mu\),
the same own-row jitter indicators \(I_i\), and the same jitters:
\[
 X_i=\mu+I_i\sigma_J Z_i.
\]
The two entering collision-velocity arrays have row norms at most
\(V_c\). Write their difference as \(e_i\), which is measurable
before these jitters. Use the original count viscosity, shared OU
innovations, and original radial cap. Define
\[
 a=t\nu,\quad \ell_\rho=e^{-1/2}/\rho,\quad
 b=t(1+c),\quad \eta=bt,\quad
 W_X=I-aL_X.
\]
The normalized count Laplacian has spectrum in \([0,1]\);
\(W_X\) is a positive symmetric stochastic contraction when
\(0\le a\le1\). The first-kick difference is exactly
\[
 \delta u=W_Xe=e+r,\qquad
 \|r\|_{2,N}\le a\|e\|_{2,N},\qquad
 |\delta u_i|\le |e_i|+a\|e\|_{1,N}.
\tag{KUER.19}
\]
Suppose each actual first-kick row satisfies
\(\mathbb E|\widetilde u_i|^2\le U_2^2\), where this
expectation includes its own clone jitter. Put
\(W_2^2=c^2U_2^2+dq^2\). The graph-dependent part of B2 obeys
\[
 \left(\mathbb E\|
       (W_y-W_{\widetilde y})\widetilde w
       \|_{2,N}^2\right)^{1/2}
 \le 4a\ell_\rho b(1+a)W_2\|e\|_{2,N}.
\tag{KUER.20}
\]
No independence between the first viscous weights and own-row
jitter, or between B2 weights and OU innovations, is assumed.
:::

:::{prf:proof}
The force inputs are identical, so \(\delta u=W_Xe\).
The spectral assertion follows from the symmetric kernel weights
\(0\le K_{ij}\le1\), since
\[
 \langle z,L_Xz\rangle_N
 =\frac1{2N^2}\sum_{ij}K_{ij}|z_i-z_j|^2
 \le\|z\|_{2,N}^2-|\bar z|^2.
\]
The diagonal coefficient of \(W_X\) is nonnegative and at most
one, and its off-diagonal row sum is at most \(a\); this proves
the last assertion of (KUER.19). Shared OU innovations give
\(\delta y=b\delta u\) and \(\delta w=c\delta u\).

Let \(d_i=|\delta y_i|\) and
\(f_i=|e_i|+a\|e\|_{1,N}\), so
\(d_i\le bf_i\) pointwise and
\(\|f\|_{2,N}\le(1+a)\|e\|_{2,N}\).
The Gaussian gradient bound gives
\[
 |[(W_y-W_{\widetilde y})\widetilde w]_i|
 \le\frac{a\ell_\rho}{N}\sum_j
 (d_i+d_j)(|\widetilde w_i|+|\widetilde w_j|).
\]
Jensen, followed by
\((d_i+d_j)^2(|w_i|+|w_j|)^2
 \le4(d_i^2+d_j^2)(|w_i|^2+|w_j|^2)\), yields
\[
 \mathbb E\| (W_y-W_{\widetilde y})\widetilde w\|_{2,N}^2
 \le8(a\ell_\rho)^2
 \left[c^2\mathbb E\{
 \overline{d_i^2|\widetilde u_i|^2}
 +\overline{d_i^2}\,\overline{|\widetilde u_i|^2}\}
 +2dq^2\mathbb E\overline{d_i^2}\right].
\]
Here \(d_i\) is fixed before the shared OU innovations, so the
conditional identity
\(\mathbb E[|\widetilde w_i|^2\mid X]
 =c^2|\widetilde u_i|^2+dq^2\) is valid. The error envelope
\(bf_i\) is fixed even before the jitters. Consequently each of
the two mixed terms is at most
\(b^2(1+a)^2U_2^2\|e\|_{2,N}^2\), and the last term is at most
\(b^2(1+a)^2\|e\|_{2,N}^2\). Taking a square root proves
(KUER.20). \(\square\)
:::

:::{prf:corollary} Full harmonic matched-source count estimate
:label: cor-kuer-harmonic-count-velocity

For \(F(x)=-\lambda x\), let
\[
 X_2=\sqrt{d(L_D^2+\sigma_J^2)},\qquad
 U_2=V_c+t\lambda X_2,\qquad W_2^2=c^2U_2^2+dq^2.
\]
If \(0\le\eta\lambda\le c(1-a)\), the two original kicks
and original cap satisfy
\[
 (\mathbb E\|\delta v^+\|_{2,N}^2)^{1/2}
 \le L_{v,\rm quad}\|e\|_{2,N},\qquad
 L_{v,\rm quad}=c-\eta\lambda
              +4a\ell_\rho b(1+a)W_2.
\tag{KUER.21}
\]
At the unchanged quadratic reference,
\(L_{v,\rm quad}<.963\).
:::

:::{prf:proof}
The pointwise bound \(|(W_X\widetilde v^{\rm col})_i|\le V_c\)
and Minkowski give the stated per-row \(U_2\), without an
independence assumption. The uncapped B2 difference is
\[
 (cW_y-\eta\lambda I)\delta u
 +(W_y-W_{\widetilde y})\widetilde w.
\]
The first operator is symmetric with nonnegative spectrum bounded
above by \(c-\eta\lambda\). Apply (KUER.19)--(KUER.20) and
cap nonexpansiveness. At the reference,
\(X_2=\sqrt{12.03}\), and direct substitution gives
\(W_2<3.925\),
\(4a\ell_\rho b(1+a)W_2<.002255\), and
\(c-\eta=.9600051234\ldots\). These numerical bounds can also
be obtained from the elementary finite-series procedure below.
\(\square\)
:::

:::{prf:lemma} Native Rastrigin central-source count estimate
:label: lem-kuer-rastrigin-central-count-velocity

Keep the native force
\(F_r(x)=-2x_r-20\pi\sin(2\pi x_r)\) and every configured
reference parameter. In the matched-source update of
{prf:ref}`lem-kuer-matched-count-velocity`, assume only
\(|\mu_r|\le\epsilon=.01\) for every coordinate. No clone,
OU, or final position innovation is bounded. Define
\[
 \begin{split}
 \psi(x)&=(1-2\eta)x-20\pi\eta\sin(2\pi x),\qquad R=bV_c,\\
 D_0&=c-2\eta,\qquad A_0=40\pi^2\eta,\qquad M_0=D_0+A_0,\\
 s_0&=\frac{\sin(2\pi bV_c)}{2\pi bV_c},\qquad
 a_O=e^{-2\pi^2t^2q^2},\quad a_{O,2}=e^{-8\pi^2t^2q^2}.
 \end{split}
\]
For \(q_c\in[-1,1]\) and \(s_c\in[s_0,1]\), put
\[
 \mathcal V(q_c,s_c)
 =D_0^2-2D_0A_0s_ca_Oq_c
 +\frac{A_0^2s_c^2}{2}
       (1-a_{O,2}+2a_{O,2}q_c^2).
\tag{KUER.22}
\]
The following explicit nondecreasing envelope is defined for all
\(r\ge0\):
\[
 \mathcal W(r)=
 \max_{s_c\in\{s_0,1\}}
 \mathcal V\!\left(
    \cos[2\pi\min\{\psi(r)+R,\tfrac12\}],s_c\right).
\tag{KUER.23}
\]
Use \(r_j=j/100\), \(0\le j\le50\), and the exact Gaussian
interval mass \(\ell_{r,\sigma}(\mu)
 =\Phi((r-\mu)/\sigma)-\Phi((-r-\mu)/\sigma)\). Set
\[
 \begin{split}
 B_{50}^{\rm jit}={}&\sum_{j=1}^{50}\mathcal W(r_j)
       [\ell_{r_j,\sigma_J}(\epsilon)
        -\ell_{r_{j-1},\sigma_J}(\epsilon)]\\
 &+M_0^2[1-\ell_{.5,\sigma_J}(\epsilon)],\\
 B_{50}={}&\max\{\mathcal W(\epsilon),B_{50}^{\rm jit}\},\\
 X_{2,c}={}&\sqrt{d(\epsilon^2+\sigma_J^2)},\qquad
 U_{2,c}=V_c+t(2X_{2,c}+20\pi\sqrt d),\\
 W_{2,c}^2={}&c^2U_{2,c}^2+dq^2,\\
 L_{v,\rm Rast}={}&\sqrt{B_{50}}+aM_0+ac
                  +4a\ell_\rho b(1+a)W_{2,c}.
 \end{split}
\tag{KUER.24}
\]
These coefficients are primitive finite expressions, and
\[
 B_{50}<.838,\qquad
 L_{v,\rm Rast}<.934,
 \qquad
 (\mathbb E\|\delta v^+\|_{2,N}^2)^{1/2}
       \le L_{v,\rm Rast}\|e\|_{2,N}.
\tag{KUER.25}
\]
The inequalities hold for every population size and arbitrary retained
collision-velocity arrays satisfying the original cap envelope.
At an actual singleton the revived star supplies the common component
and common donor source. Average its original Haar law after the
conditional comparison; its centered velocity error already contracts
by \(|\alpha_{\rm col}|\) before this kinetic estimate. The
persistent living donor has \(I_i=0\), and its separate term is
retained in the maximum defining \(B_{50}\).
:::

:::{prf:proof}
Put \(\delta u=W_Xe\), \(\delta y=b\delta u\), and
\(y^{\rm mid}=(y+\widetilde y)/2\). The exact force secant in
coordinate \(r\) is
\[
 t[F_r(y_i)-F_r(\widetilde y_i)]
 =-\left[2\eta+40\pi^2\eta
     \operatorname{sinc}(\pi b\delta u_{ir})
       \cos(2\pi y^{\rm mid}_{ir})\right]\delta u_{ir}.
\]
All collision rows have norm at most \(V_c\), so
\(|\delta u_{ir}|\le2V_c\), and the sinc lies in
\([s_0,1]\). The midpoint position is exactly
\[
 y^{\rm mid}_{ir}=\psi(X_{ir})
       +b[W_X(v^{\rm col}+\widetilde v^{\rm col})/2]_{ir}
       +tq\xi_{ir}^O.
\]
Its bounded velocity offset has magnitude at most \(R\), although
it depends on all own jitters. The OU coordinate is independent of
that complete preparation. If \(T_{ir}\) denotes the diagonal
secant coefficient
\[
 T_{ir}=D_0-A_0
       \operatorname{sinc}(\pi b\delta u_{ir})
                  \cos(2\pi y^{\rm mid}_{ir}),
\]
integrating the original OU Gaussian gives exactly (KUER.22) at
the actual \(s_c\) and
\(q_c=\cos(2\pi[\psi(X_{ir})+\text{velocity offset}])\).
The expression is convex in \(s_c\). Its derivative in \(q_c\)
is negative, since
\(D_0a_O>A_0a_{O,2}\), a direct reference inequality.
For \(|X_{ir}|=r\), the lowest possible cosine is bounded below
by \(\cos[2\pi\min\{\psi(r)+R,1/2\}]\).
Indeed \(\psi\) is odd and strictly increasing, because
\(\psi'\ge1-2\eta-40\pi^2\eta>.688\), and before the clipped
endpoint every possible argument has absolute value at most
\(2\pi(\psi(r)+R)\le\pi\). Thereafter the bound \(-1\) is
valid. Thus (KUER.23) bounds the conditional OU square, and is
nondecreasing.

The law of \(|\mu_r+\sigma_JZ|\) is stochastically increasing in
\(|\mu_r|\), by differentiating its exact symmetric-interval
probability. The right endpoint sum in (KUER.24), including its
complete tail, therefore bounds the expectation of
\(\mathcal W(|X_{ir}|)\) for every accepted row. For an uncopied
row, \(|X_{ir}|\le\epsilon\), which contributes the separate
\(\mathcal W(\epsilon)\) term in the displayed maximum.
Consequently \(\mathbb E T_{ir}^2\le B_{50}\).
This does not make \(T\) independent of \(\delta u\).
Instead (KUER.19) gives the exact decomposition
\[
 \delta z^{\rm uncapped}
 =Te+Tr-caL_y\delta u
             +(W_y-W_{\widetilde y})\widetilde w.
\]
Since \(e\) is frozen before own jitter and OU,
\(\mathbb E\|Te\|_{2,N}^2\le B_{50}\|e\|_{2,N}^2\).
The two middle terms have \(L^2\) norms bounded by
\(aM_0\|e\|_{2,N}\) and \(ac\|e\|_{2,N}\), respectively.
The actual force envelope gives the per-row \(U_{2,c}\) in
(KUER.24); (KUER.20) supplies the final graph defect. Minkowski
and cap nonexpansiveness prove the asserted comparison. The
final position Gaussian is shared and has no velocity feedback,
so its complete unbounded law is still present. \(\square\)
:::

:::{prf:lemma} Finite elementary certificate for the native coefficient
:label: lem-kuer-native-finite-numeric-certificate

The exact finite expression (KUER.24) admits the rational enclosures
\[
 \begin{split}
 .6793196683989&<\mathcal W(\epsilon)<.6793196683990,\\
 .8376367364890&<B_{50}^{\rm jit}=B_{50}<.8376367364891,\\
 5.9507605102685&<W_{2,c}<5.9507605102687,\\
 .0034173968014&<4a\ell_\rho b(1+a)W_{2,c}<.0034173968015,\\
 .9320202359870&<L_{v,\rm Rast}<.9320202359872.
 \end{split}
\tag{KUER.26}
\]
These are scalar integrals of the actual unbounded innovations,
not empirical concentration assertions.
:::

:::{prf:proof}
A finite elementary enclosure procedure suffices. Compute \(\pi\)
from Machin's identity
\(\pi=16\arctan(1/5)-4\arctan(1/239)\), using the first
161 alternating odd-power terms and the first omitted term as
a signed error enclosure. For each required exponential and
trigonometric value, use its degree-160 Taylor sum with its
remainder enclosure; all exponential arguments here have
absolute value below one, and sine/cosine arguments have absolute
value at most \(\pi\). For Gaussian masses the only normal
arguments have absolute value at most \(5.1\). Use
\[
 \Phi(x)=\tfrac12+\frac1{\sqrt{2\pi}}
 \sum_{n=0}^{160}\frac{(-1)^nx^{2n+1}}
                      {(2n+1)2^nn!}+E_{160}(x),\qquad
 |E_{160}(x)|\le
 \frac{|x|^{323}}{323\,2^{161}161!\sqrt{2\pi}}.
\tag{KUER.27}
\]
The alternating tail is decreasing from this truncation onward.
Evaluate each of the 50 nonnegative bin masses and its endpoint
weight, and then the displayed full-tail term, with outward rational
rounding at 40 decimal places. Square roots are bracketed by
squaring their rational endpoints. The resulting enclosures are
(KUER.26); their width is far smaller than the stated strict
margin. This supplies a finite computation of every coefficient
in (KUER.25). \(\square\)
:::

The outward interval evaluation is reproduced by
`verify_central_velocity.py` in this research directory. An independent
rational implementation using the displayed Taylor sums, 40-place
outward rounding and integer square-root bracketing verifies all five
enclosures in (KUER.26).

:::{prf:lemma} Primitive discounted-return comparison required for closure
:label: lem-kuer-discounted-return-comparison

Let \(\mathcal L_N=\mathcal L_N(\alpha_N)\) be the actual
unweighted-in-\(h\) discounted return kernel in (KUER.12).
Suppose a block length \(k_N\), a probability measure \(\pi_N\)
on the actual singleton set, and primitive bounds satisfy
\[
 \mathcal L_N^{k_N}(S,\cdot)\ge\beta_N\pi_N(\cdot),\qquad
 \ell_N\le\mathcal L_N^{k_N}1(S)\le u_N,
 \qquad \beta_N>u_N-1.
\tag{KUER.28}
\]
Then
\[
 R_N^{\mathcal R}
 \le\frac{1-\ell_N+\beta_N}{1-u_N+\beta_N},\qquad
 \frac{\sup_{\mathcal K_N}h_N}{\inf_{\mathcal K_N}h_N}
 \le\frac{1+r_N}{r_N}F_N^+
       \frac{1-\ell_N+\beta_N}{1-u_N+\beta_N}.
\tag{KUER.29}
\]
The coefficients in (KUER.28) involve the original source, Haar,
jitter, thermostat, force, cap, terminal flags and physical return
times. No eigenfunction appears in their entries. In particular a
contraction for a raw matched-source update cannot be substituted
for this stopped and discounted block comparison.
:::

:::{prf:proof}
Since \(\mathcal L_Nh_N=h_N\) on the singleton set, its powers
also fix that restriction. Write \(M=\sup_{\mathcal R}h_N\),
\(m=\inf_{\mathcal R}h_N\), and
\(A=\beta_N\pi_N(h_N)\). Removing the common positive
submeasure in (KUER.28) gives
\[
 M\le A+(u_N-\beta_N)M,\qquad
 m\ge A+(\ell_N-\beta_N)m.
\]
Here one may replace \(\ell_N\) by
\(\max\{\ell_N,\beta_N\}\), so its residual mass bound is
nonnegative. Consequently
\[
 (1-u_N+\beta_N)M
 \le A\le(1-\ell_N+\beta_N)m.
\]
The denominator is positive by the strict comparison in (KUER.28).
Divide and use (KUER.14). \(\square\)
:::

:::{prf:remark} Exact conditioning cost of the new local comparison
:label: rem-kuer-local-return-conditioning-cost

For a raw kinetic coupling and an actual next-output event \(E\),
including a specified singleton position and every terminal flag,
the tilted second moment is exactly
\[
 \mathbb E[\|\delta v^+\|_{2,N}^2\mid E]
 =\frac{\mathbb E[
     \|\delta v^+\|_{2,N}^2\,\Pr(E\mid\mathscr H)]}
         {\Pr(E)},
\tag{KUER.30}
\]
where \(\mathscr H\) contains the complete preparations, jitters
and OU outputs, and the remaining conditional probability integrates
the original final-position Gaussians. It is the actual Gaussian
product of the alive/dead interval masses conditional on those
outputs. The elementary estimate obtained from (KUER.25) alone is
\(L_{v,\rm Rast}^2\|e\|_{2,N}^2/\Pr(E)\); it is not uniform
for rare singleton outcomes. Further, in a coupling the two marked
events may differ, so common-event transport must also quantify
the unmatched marked mass.

The native secant has pointwise magnitude at most
\(M_0=1.2688562648\ldots>1\), although its fully integrated
central coefficient is below one. Conditioning changes the scalar
OU and clone-jitter averages in that proof. Thus neither the raw
coefficient nor the Haar centered-energy factor can be used as a
proved coefficient for \(\mathcal L_N\). A constructive next
inference is to evaluate its death-tilted scalar secant integrals
jointly with the final-position marks, and their graph-dependent
mixed moments, followed by the exact physical waiting-time series
in (KUER.12). In parallel, a proved phase-lineage establishment
estimate can replace entire-array phase replacement without making
phase basins closed. Both are productive actual-kernel routes to
the primitive bounds (KUER.28); neither bound is presumed here.
:::

:::{prf:remark} The retained velocities and clock are actual unresolved outcomes
:label: rem-kuer-return-outcomes

At a singleton, the next mandatory revival makes all zero-jitter
sources the same living position. It does not copy that donor's
velocity. The accepted component contains the revived star, and its
one Haar rotation contracts the frozen centered velocity energy by
\(\alpha_{\rm col}^2\), while retaining the actual mean velocity.
Both viscous kicks, the unbounded thermostat, and B2's force at
its original noisy A2 positions still read that velocity information.
The effective singleton set consequently has a living position,
its surviving label, and an entire array of retained capped velocities.

The matched-source velocity comparisons below retain the harmonic
force and, separately, native Rastrigin curvature and Gaussian
excursions. A velocity estimate for
successive matched-source revival steps does not by itself prove an
estimate for the first-return kernel: physical waiting trajectories
and the discounted factor \(\alpha_N^{-\tau_{\mathcal R}}\) must also
be transported. They are the two exact residuals in (KUER.14).

The crude tail bound above would give
\[
F_N^+\le
\frac{\lambda_N}{\lambda_N-(1-\alpha_N)}
\quad\text{if }1-\alpha_N<\lambda_N.
\tag{KUER.15}
\]
Indeed the stopping time is dominated by a geometric variable of
parameter \(\lambda_N\), whose \(\alpha_N^{-1}\)-moment is that
fraction. With the supplied upper mortality estimate
\(1-\alpha_N\le e^{-Na_E}\), this sufficient comparison is not
verified: \(q_*^N\) contains the very small central Gaussian death
tail. Its failure is a failure of this comparison, not a proof
that the actual discounted moment or global ratio diverges.

Likewise an arbitrary algebraic two-phase killing matrix is not an
actual canonical-kernel counterexample. Any negative conclusion about
(KUER.1) requires a verified family of this unchanged algorithm,
including its mandatory revival and complete Gaussian inputs.
:::

(sec-kuer-direct-consumer)=
## 4. Direct normalized estimates and evidence status

:::{prf:proposition} The already available direct entropy interface
:label: prop-kuer-direct-entropy-interface

Let \(k_N=Q_N1\), \(P\ll\nu_N\), \(f=dP/d\nu_N\), and let
\(P^+=PQ_N/P(k_N)\). At the integrability domain of the existing
complete survivor chain-rule theorem, exactly
\[
D(P^+\Vert\nu_N)-D(P\Vert\nu_N)
=
\frac{\operatorname{Cov}_P(k_N,\log f)}{P(k_N)}
+\log\frac{\alpha_N}{P(k_N)}
-\mathcal B_N(P,\nu_N).
\tag{KUER.16}
\]
The backward information \(\mathcal B_N\ge0\) uses the whole
actual marked update. If the retained alive transform gives
\(\kappa_N(S)\le e^{-Na_E}\), then
\[
\frac1{P(k_N)}
\le\frac1{1-e^{-Na_E}}
\le\frac1{1-e^{-a_E}},
\tag{KUER.17}
\]
which is independent of the population. No global eigenfunction
ratio is used. A signed estimate on the covariance and backward
information is still required for a strict entropy contraction;
the normalization bound alone does not claim one.
:::

:::{prf:proof}
Apply the existing complete-kernel chain rule with its QSD
reference, for which \(R^+=R=\nu_N\) and \(R(k_N)=\alpha_N\).
The integrated one-step survivor floor gives (KUER.17).
Every sampled feature, gate, Haar component, retained coordinate
and noise has already been integrated by the joint forward/backward
laws. \(\square\)
:::

:::{prf:remark} Final local evidence record
:label: rem-kuer-evidence-status

The new original-kernel results (KUER.2)--(KUER.11) are proved
quantitative reductions, with independent review of the own-jitter
replacement and extinction-weighted energy optimization. Their
displayed floating singleton coefficients are diagnostic evaluations
of the explicit positive analytic formulas. The central cutset
coverage (KUER.31)--(KUER.33) has explicit analytic lower bounds.
The count-mode velocity comparisons (KUER.19)--(KUER.25) are
additional proved uniform estimates, including the unchanged
harmonic reference and native central Rastrigin sources. The native
coefficients (KUER.26) have independent mathematical review and
outward interval and rational-series certificates reproduced by
the linked verification script.

The global eigenfunction endpoint (KUER.1) remains open in this
record. The global-minimum compression (KUER.11) and the raw
central-source retained-velocity comparison (KUER.25) are genuine
uniform implications at their stated domains. The first missing inference is the
independent bound on the actual discounted singleton-return clock
and retained-velocity eigenfunction oscillation in (KUER.14).
The primitive sufficient comparison (KUER.28)--(KUER.29) states
exactly how a complete stopped-kernel estimate would discharge it.
The raw coefficient has not been passed through the rare-event
conditioning in (KUER.30) or extended to row normalization.
The weaker ordinary hitting and return probabilities are not
silently substituted for that inference. The direct physical-law
route (KUER.16)--(KUER.17) is a serious separate consumer with a
closed normalization interface; it is not a claim that the
bounded-Doob reweighting endpoint has been achieved.
:::
