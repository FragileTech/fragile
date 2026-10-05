# Independent audit of the signed family's actual alive-law transfer

(sec-siaa-source)=
## 1. Exact reviewed sources and carrier

:::{prf:definition} Signed alive transfer review
:label: def-siaa-source

This independent review reads
`64_default_signed_alive_transfer.md` at SHA-256
`2e0066d61ec2d340707d78b3ef33640ea2b36fab0caca590da9051dd9ad6507f`.
Its consumed kinetic carrier is the explicit paired-noise proof in
`61_default_signed_gaussian_tensor.md` at SHA-256
`bb2a271f9905c42a3edf9ac3881a7990bf77c8e264fdb3f37d01785670776425`.
The carrier is the actual kinetic transition from balanced
deterministic prepared inputs with half the rows at
$(ze_1,-ze_1/2)$ and half at $(-ze_1,ze_1/2)$,
$z\in[.2,.3]$, even $N\ge2$.

Both actual count kicks retain denominator $N$, their own
providers, the complete correlated OU second stage, the native
cap and final positional Gaussian. The carrier proof constructs
a valid full paired-noise coupling whose labeled mean-square
physical cost is at most
$$
\frac{99}{100}w_0^2,\qquad
w_0^2=\frac{121}{100}|z_1-z_0|^2.
\tag{SIAA.1}
$$
This review checks its transfer to each output's own survival
normalization, rather than interpreting a physical-law upper
bound as a chosen-coupling bound.
:::

(sec-siaa-plan)=
## 2. Common-alive plan and the complete normalization correction

:::{prf:lemma} Independent verification of the universal bounded transfer
:label: lem-siaa-plan-review

The common-alive plan in (SIAT.3)--(SIAT.5) is admissible and
its correction $9D_*^2\epsilon$ is valid under only the stated
mean-square physical carrier and expected dead fractions.
The separately normalized survivor correction is
$5D_*\sqrt\epsilon$ as in (SIAT.4).
No rowwise physical contraction or joint-survival denominator
is required.
:::

:::{prf:proof}
Write $A,\widetilde A$ for the actual alive sets,
$M=|A|$, $\widetilde M=|\widetilde A|$ and
$M_{\max}=\max(M,\widetilde M)$.
On $\{M,\widetilde M\ge N/2\}$ pair every common-alive
index with weight $1/M_{\max}$. Each weight is at most
both its true empirical marginal weights $1/M$ and
$1/\widetilde M$, so this partial plan is admissible.
If $k=|A\cap\widetilde A|$, its common-pair cost is bounded by
$$
\frac1N\sum_i q_G(Z_i-\widetilde Z_i)
+D_*^2\frac{N-M_{\max}}N,
$$
because its excess coefficient is
$(N-M_{\max})/(NM_{\max})$ and $k\le M_{\max}$.
The full labeled carrier includes all row costs; those outside
the common alive intersection remain nonnegative.

Both residual marginals have equal mass $1-k/M_{\max}$.
This is
$(M_{\max}-k)/M_{\max}\le|A\triangle\widetilde A|/M_{\max}
\le2|A\triangle\widetilde A|/N$.
Any coupling of the residuals costs at most its mass times
$D_*^2$, since both reside in the same bounded alive phase set.
The expected normalization excess is at most $D_*^2\epsilon$.
Also
$|A\triangle\widetilde A|\le(N-M)+(N-\widetilde M)$,
so its expected residual cost is at most $4D_*^2\epsilon$.

By Markov, either alive count falling below $N/2$ has joint
probability at most $4\epsilon$.
On this event any coupling of the two auxiliary alive
readouts costs at most $D_*^2$, including the declared
bounded ghost in an extinct marginal.
Together these charges give $1+4+4=9$ times $D_*^2\epsilon$.
The carrier cost restricted to the good event is bounded by
its unconditioned nonnegative expectation.
This proves the raw auxiliary-law transport estimate.

Each raw auxiliary law has its own mixture decomposition into
its actual own survivor law and its own extinct auxiliary law.
Couple the survivor component identically to a draw from that
survivor law; on the extinct mixture component use any coupling
within the bounded alive set. Its cost is at most
$D_*^2$ times that marginal's extinction probability, hence
at most $D_*^2\epsilon$.
The two separate repairs each cost at most $D_*\sqrt\epsilon$.
The outer metric triangle and
$\sqrt{k^2w_0^2+9D_*^2\epsilon}
\le kw_0+3D_*\sqrt\epsilon$ give the claimed factor five.
No common survival event or unnormalized alive empirical weight
is used at any step.
:::

(sec-siaa-default)=
## 3. Original Gaussian tails, exact providers and coefficient margin

:::{prf:lemma} Default carrier and full-tail certificates
:label: lem-siaa-default-review

The balanced original finite arrays satisfy every hypothesis
of the bounded-transfer lemma with
$k=\sqrt{.99}$, $D_*^2=67$, $\epsilon=2^{-100}$.
For $|z_1-z_0|\ge.01$ the resulting coefficient is at most
$249/250$, independent of the even array size and with no
additive particle floor.
:::

:::{prf:proof}
In the actual first graph, same-sign pair velocity differences
vanish and opposite-sign pairs have mass exactly one half.
Thus $L_1P=e^{-2z^2}P$, including zero self contributions
and the actual denominator $N$. The explicit derivative proof
in source 61 shares each row's original Gaussian noises and
integrates the complete OU-induced second graph, with no
independent provider substitution. Its differential bound
integrates by Minkowski to the specific carrier (SIAA.1).
Diagonal and same-sign differential pair terms vanish, so the
finite pair calculation imports its stated constants exactly.

The prepared array is deterministic. Its first output is fixed
before the independent rowwise OU and final noises. Each
terminal position therefore is an independent Gaussian with
variance $\tau^2<.0005$ and first-coordinate mean
$$
\pm z\{a_x-b[1-ae^{-2z^2}]/2\},
$$
of absolute value below $.3$; other coordinate means vanish.
The actual second graph and cap do not change these positions.
Every mean is at least $1.7$ from every terminal box face.
Consequently each full dead tail is at most
$$
6e^{-1.7^2/(2(.0005))}=6e^{-2890}<2^{-100}.
$$
The last inequality is elementary: $e>2$ and
$6\cdot2^{-2890}<2^{-100}$.
All Gaussian outcomes remain in the actual law.
Independence of positions gives extinction probability at most
$\epsilon^N$. Linearity alone suffices for expected dead fractions.
No independence of the stored velocities and terminal marks
has been asserted.

Alive position squared diameter is $48$ and capped velocity
squared diameter is at most $16$. Therefore the physical
$G$ squared diameter is at most $1.04(48+16)=66.56<67$.
The origin is a permissible bounded auxiliary ghost.
The separate-survivor lemma gives
$$
\mathbb W_{2,G}(\mathcal H_N(z_0),\mathcal H_N(z_1))
\le\sqrt{.99}\,w_0+5\sqrt{67}\,2^{-50}
<.995w_0+42\,2^{-50}.
$$
The exact rational comparisons $.995^2>.99$ and
$25(67)<42^2$ justify these strict coefficient replacements.
Finally $w_0=1.1|z_1-z_0|\ge.01$ and
$42(100000)<2^{50}$ give
$42\,2^{-50}<.001w_0$. Hence the coefficient is less
than $.996=249/250$, proving the non-strict theorem endpoint.
This absorption explicitly requires the stated minimum
separation; it is not a small-displacement Lipschitz claim.
:::

(sec-siaa-sampling)=
## 4. Both actual sampling orders and their distinct normalizers

:::{prf:lemma} Sampled-law transport and slot-first cancellation
:label: lem-siaa-sampling-review

Both sampled-law conclusions in (SIAT.6) are valid with the
same coefficient. Each law retains its own sampling weights
and own alive denominator.
:::

:::{prf:proof}
For swarm-first sampling, draw a physical pair from an optimal
inner alive transport for each coupled surviving empirical
pair. Its two marginals are the respective uniform-alive
empirical probabilities. Averaging the plans therefore gives
exactly the two actual swarm-first sampled laws, with cost at
most the outer empirical-law cost.

For slot-first sampling, draw a common uniform original index
from the raw paired arrays. Keep an alive selected phase and
replace a dead one by the bounded origin only in the auxiliary
comparison measure. On two alive selected rows the cost is
the original labeled carrier cost. If either row is dead, the
new cost is bounded by $D_*^2$. Therefore its raw coupled cost
is at most $.99w_0^2+2D_*^2\epsilon$.
Each auxiliary marginal is the mixture of that marginal's own
slot-conditioned alive phase law and its ghost, with dead
mixture mass at most $\epsilon$.
The two separate mixture repairs give total distance at most
$$
\sqrt{.99}\,w_0+(\sqrt2+2)D_*\sqrt\epsilon
<.995w_0+42\,2^{-50}.
$$
The same exact integer margin proves the slot-first coefficient.

An original selected row being alive implies its swarm's
nonextinction. Thus conditioning that swarm first to survive,
then drawing a uniform original index and conditioning that
index to be alive, gives the same final ratio as the raw
slot-first construction: the own survival probability cancels
between its numerator and denominator. This cancellation
does not identify the slot-first ratio with the swarm-first
random inverse-count ratio. Both conventions are checked
separately in the reviewed source.
:::

(sec-siaa-result)=
## 5. Accepted endpoint and scope

:::{prf:remark} Independent review result
:label: rem-siaa-result

The frozen source specified in Section 1 passes this complete
independent review. No correction to source 64 is required.
It proves an actual separately surviving alive empirical-law
kinetic transport estimate and both actual sampled-law estimates
for every even $N\ge2$, with coefficient $249/250$ and no
particle floor on the explicitly separated prepared family.

The proof includes the full Gaussian tails and actual finite
count providers. Its universal alive-plan correction uses only
the mean-square physical carrier, not a per-row contraction.
It assumes the stated deterministic prepared inputs; a full
active preparation, class preservation, another noisy update
or default general convergence does not follow from this
one-update kinetic theorem.
:::
