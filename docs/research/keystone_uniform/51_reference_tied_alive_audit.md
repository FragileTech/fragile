# Independent audit of exact tied-fitness alive transport

(sec-tata-source)=
## 1. Frozen source and audit endpoint

The audited source is
`49_reference_tied_alive_transport.md` at SHA-256
`d446b6fb97c4e8b9d6a429af51f2e48833dd02f18bb83d8319d268dc192a5e81`.
Its precision correction makes the final product inequality in
(TAT.1) non-strict, so it also holds at zero center displacement.
The scalar coefficient certificate remains strict.
The separated class in the final theorem is unchanged.

The audit is **pass** for the actual one-update law statement:
for both dense count and dense row normalization, all $N\ge1$,
centers in $[-.5,.5]^3$ separated by at least $1/4$, and the
specified unchanged harmonic restriction, the physical alive
empirical-law and swarm-first uniformly alive-sampled $W_2$
distances contract by $1-1/160000$.
Each law uses its own actual alive count and its own survival event.
There is no particle floor.

This is not an iterated convergence result. The collapsed class
is not invariant, and its within-swarm exact ties make acceptance
zero even when both configured fitness powers are positive.
Neither a full active-selection block nor Rastrigin or QSD
convergence is certified by this audit.

(sec-tata-preparation)=
## 2. Actual tied preparation, including the singleton

All original rewards in one $S(m)$ equal $-|m|^2/2$.
Every measured phase pair has identical endpoints, so each
diversity value is identical, including its configured smoothing.
The sampled alive means and regularized variances consequently
produce identical within-swarm fitness values. Positive regularizer
floors keep the zero-variance calculation finite.
The original clipped difference gate is exactly zero, independently
of the sign or absolute size of the common raw reward.
No input slot is dead.

The actual code agrees with the required edge cases:
`companion_selection.py` returns the sole alive index in its
singleton branch; global fitness variance divides by the alive
count with its positive standardizer floor; and `cloning.py`
computes zero difference score and zero clipped gate for equal
fitness. Thus $N=1$ introduces no unverified self-exclusion draw.
The declared singleton kinetic convention has zero viscous force.
For all $N$, the original velocities are zero, so any executed
component readout is zero. Position copies and copied-recipient
jitters do not occur. Hence the actual prepared array is exactly
$X_i=m,P_i=0$, even though the configured fitness and cloning
channels remain present.

The exact input empirical phase measure is the point mass at
$(m,0)$. Its distance to the corresponding measure at
$(\widetilde m,0)$ is $|\widetilde m-m|$.

(sec-tata-providers)=
## 3. Own noisy providers and the cap preserve the displayed comparison

The first viscous force vanishes, giving the actual stages
$$
w_i=-ctm+q\xi_i,\qquad y_i=a_xm+tq\xi_i.
$$
Sharing each original OU and final Gaussian between the two
compared arrays preserves both marginals: within either marginal
these full Gaussian arrays have their original independent laws.
For $\delta=\widetilde m-m$,
$$
\Delta w_i=-ct\delta,\qquad \Delta y_i=a_x\delta.
$$
All pairwise second-stage position differences agree between the
two arrays, so every count weight and every self-excluded row
degree agrees. All pairwise velocity differences also agree.
Therefore each actual second viscous force agrees, without
freezing its provider or making it independent of the OU noise.
The harmonic second force gives
$$
\Delta z_i=-t(c+a_x)\delta=-r_H\delta.
$$
The original cap is globally nonexpansive. The shared final
position Gaussian cancels in the difference. Consequently,
for every original Gaussian outcome and every row,
$$
|\Delta x_i^+|^2+|\Delta v_i^+|^2
\le(a_x^2+r_H^2)|\delta|^2
\le(1-1/40000)|\delta|^2.
$$
The exact rational upper certificate is
$$
.999216^2+.03920032^2
=.9999692797441024<.999975=1-1/40000.
$$
The product bound is non-strict at $\delta=0$.
The second graph may correlate all capped velocities; row-output
independence is not used in this transport comparison.

(sec-tata-tails)=
## 4. Exact terminal tails and the survival marginals

Because the second kick and cap change velocity only,
terminal positions are independent across rows with law
$$
x_i^+\sim N(a_xm,\tau^2I_3),\qquad
\tau^2=t^2q^2+s^2<.0005.
$$
This is an unconditional statement for the actual fixed input,
before survival restriction. The mean of every coordinate has
absolute value at most $.5$, so every box face is at distance
at least $1.5$. A Gaussian exponential bound and union over six
faces give
$$
\Pr(a_i^+=0)\le6e^{-2250}<2^{-100}=:\epsilon.
$$
The elementary certificate $e>2$ suffices for the last inequality,
since $6<2^{2150}$. Independence of terminal positions proves
$\Pr(E_m^c)\le\epsilon^N$, and linearity proves the expected
dead fraction bound $\epsilon$. No Gaussian tail is removed.
Independence is not asserted after conditioning on $E_m$.

The auxiliary point mass at $(0,0)$ on extinction defines only
a raw observation variable. It does not change or restart the
finite killed chain. Every nonextinct alive observation, and
this auxiliary observation, has support in
$$
K=[-2,2]^3\times\overline B_2.
$$
The physical squared diameter is at most $48+16=64<67=D^2$.
Therefore any two empirical measure observations in this support
have inner $W_2$ distance at most $D$, and their law space has
the same squared-diameter transport bound.

(sec-tata-weights)=
## 5. Matching common alive labels with their actual weights

On the coupled event $M,\widetilde M\ge N/2$, let
$C=A\cap\widetilde A$ and assign common-index pairs mass
$1/\max(M,\widetilde M)$. This mass is no greater than either
actual per-row weight $1/M$ or $1/\widetilde M$.
Subtracting these pairs from the two empirical marginals leaves
two nonnegative residual measures of equal mass
$$
\rho=1-\frac{|C|}{\max(M,\widetilde M)}.
$$
They can be coupled, for instance by their normalized product
when $\rho>0$. Thus the construction preserves both actual
alive-normalized marginal measures. Its common-label cost is
at most $k^2|\delta|^2$, where $k^2=1-1/40000$.
Its residual cost is at most $D^2\rho$.

Since
$\max(M,\widetilde M)-|C|\le|A\mathbin\triangle\widetilde A|$,
$$
\rho\le2|A\mathbin\triangle\widetilde A|/N.
$$
The symmetric-difference set is included in the union of the
two dead-label sets. Its expected fraction is at most
$2\epsilon$ under the exhibited raw common-innovation coupling.
Each low-alive event has probability at most $2\epsilon$ by
Markov applied to its own expected dead fraction; their union
has probability at most $4\epsilon$.
On this exceptional event both auxiliary observations still lie
in the bounded observation space, so cost at most $D^2$.
The total raw expected inner squared cost is consequently at
most
$$
k^2|\delta|^2+4D^2\epsilon+4D^2\epsilon
=k^2|\delta|^2+8D^2\epsilon.
$$
This proves (TAT.4) for the outer law metric. It also covers
$N=1$: its good event has both slots alive, and its bad event
uses the same auxiliary comparison bound.

(sec-tata-normalizers)=
## 6. Both own survival normalizers and the multiplicative coefficient

For each center separately, its raw auxiliary law is exactly
$$
\mathcal B_N(m)
=\Pr(E_m)\mathcal A_N(m)
 +\Pr(E_m^c)\delta_{\delta_{(0,0)}}.
$$
Couple this mixture with $\mathcal A_N(m)$ by taking a common
survivor observation with probability $\Pr(E_m)$, and using
the auxiliary observation and a survivor sample on the remaining
branch. The marginals are the displayed raw and survivor laws.
The mismatch probability is at most $\epsilon^N$, giving
$$
\mathbb W_2(\mathcal B_N(m),\mathcal A_N(m))
\le D\sqrt{\epsilon^N}\le D\sqrt\epsilon.
$$
The same construction is performed separately for $\widetilde m$.
Their survival events and probabilities are not identified.
The outer metric triangle gives
$$
\mathbb W_2(\mathcal A_N(m),\mathcal A_N(\widetilde m))
\le k|\delta|+(\sqrt{8D^2}+2D)\sqrt\epsilon
<k|\delta|+40\,2^{-50}.
$$
An exact upper certificate for the constant is
$$
(\sqrt{8D^2}+2D)^2
=67(12+8\sqrt2)
<67(12+80/7)=10988/7<1600.
$$
Here $\sqrt2<10/7$ follows by squaring.
The elementary bound $\sqrt{1-u}\le1-u/2$ gives
$k\le1-1/80000$. Finally,
$$
40\cdot640000<2^{50},\qquad |\delta|\ge1/4
$$
imply $40\,2^{-50}<|\delta|/160000$.
This proves the claimed coefficient $1-1/160000$ independently
of $N$. The fixed separation absorbs the complete tiny tail
budget into a multiplicative coefficient; the statement does
not claim the same coefficient at arbitrarily small separation.

(sec-tata-sampling)=
## 7. Swarm-first alive sampling and final scope

Each surviving empirical measure already has weights equal to
the reciprocal of its own alive count. Its barycenter as a random
probability law is precisely the swarm-first uniformly alive-sampled
phase distribution. Coupling a pair of empirical measures and then
coupling their normalized inner phase distributions preserves these
two sampled marginals. Integrating the conditional squared phase
cost bounds sampled $W_2^2$ by the outer expected inner $W_2^2$.
For these finite empirical observations a measurable optimal inner
plan can be selected from its finite transportation linear program,
using a deterministic tie rule; alternatively measurable
approximating plans give the same infimum. Thus the sampled-law
assertion uses an actual coupling with both own survival
normalizers and actual alive weights.

The construction compares optimal transport between actual alive
observations. It neither charges fixed-slot status differences as
the requested distance nor assumes a prescribed source pairing
is optimal. All providers are each transition's own joint providers,
all Gaussian outcomes remain present, and the actual tied gate
is zero without a variance lower bound.
The completed claim is a single update on the explicit collapsed
and separated harmonic class. That class immediately becomes noisy
and need not retain tied fitness, so the coefficient cannot be
iterated to assert unrestricted default convergence.

(sec-tata-scope-revision)=
## 8. Final source scope revision

The final source SHA-256 is
`cda8b6ec493e85c2561249c60d6b77b2c8a7e089183edfa0ca8ceaa875d88262`.
This scope-only revision is accepted while preserving the independently
reviewed historical source hash in Section 1.
It explicitly uses the current-frame harmonic restriction with
history, curl, elite and geometry-feedback branches disabled,
as in the DMC kinetic register. That is the carrier consumed by
the proof reviewed above.

The positive fitness powers need not satisfy the DMC weak-selection
interval in this particular theorem: exact within-swarm measured
ties make every original alive acceptance gate zero. The positive
regularizer floors, actual measured statistics and singleton
convention remain retained. Neither the transport constants nor
the one-update, separated-input scope changed.
The independent final-source review in research 52 also accepts
this revision. The audit result remains pass.
