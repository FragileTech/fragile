# QFT Calibration: Channel Knobs and Mass Plateaus

:::{div} feynman-prose
**TLDR.** Calibration begins with a precisely defined observable, its recorded
frame and mask, and the correlator that the analysis actually computes. A
stable fitted exponential supplies an operational channel decay scale.
Identification with a physical particle requires the corresponding theoretical
model and independent validation.

This chapter connects the Fractal Set operators to the calibration code and
separates simulation parameters from analysis parameters. Some proposed
channels vanish identically under their current masks; some names denote
scalar phase proxies. The exact implementation identities below determine
which signals can be fitted and what those fits measure.

Prerequisites: {doc}`01_fractal_set`, {doc}`03_lattice_qft`,
{doc}`04_standard_model`, and {doc}`05_yang_mills_noether`.
Recorded experiments and the calibration notebook are in
{doc}`06_empirical_validation`, {doc}`07_qft_calibration_report`, and
{doc}`08_qft_calibration_notebook`.
:::

(sec-qft-calibration-correlators)=
## From correlators to mass plateaus

Channel decay rates are extracted from time correlators of frame observables
along the executed algorithm. The lag is the step index of the recorded chain
multiplied by an assigned unit $\Delta\tau$; it is algorithm time, not a
Euclidean-time coordinate of a field configuration. The Schwinger functions
of a specified Euclidean field law are the objects of
{prf:ref}`def-euclidean-correlator-fg`; reading a chain correlator as one of
them is the additional hypothesis {prf:ref}`assm-qft-positive-transfer`. For
practical calibration we measure two-point correlators and their connected
variants ({prf:ref}`def-sm-direct-correlations`,
{prf:ref}`def-two-point-connected`) and fit an exponential decay in the lag.

The link to theory is the correlation length relation
{prf:ref}`def-correlation-length` and the scale hierarchy
{prf:ref}`thm-mass-scales`. A stable exponential decay defines the channel
decay rate $m_\chi$ of {prf:ref}`def-qft-channel-decay-rate`. It is called a
channel mass only under {prf:ref}`assm-qft-positive-transfer` together with
the unit conventions of {prf:ref}`def-sm-direct-correlations`;
{prf:ref}`prop-qft-decay-rate-scope` states what holds with and without that
hypothesis.

Implementation note: the generic correlator utilities and effective-mass extraction live in
`src/fragile/physics/new_channels/correlator_channels.py` and
`src/fragile/physics/aic/correlator_channels.py`. The active electroweak dashboard route then
assembles electroweak-specific operators and fit inputs in
`src/fragile/physics/app/electroweak_correlators.py` and
`src/fragile/physics/app/electroweak_mass_tab.py`.

### Frame observables, decay rates, and the scope of the word mass

:::{div} feynman-prose
Let me tell you what this section is really about, because everything else in
the chapter leans on it. We run an algorithm. It produces a sequence of
recorded frames. On each frame we compute a number — a colour bilinear, a
phase average, whatever — and we ask how fast that number forgets itself as
the step index grows. If it forgets exponentially, we have a rate. That rate
is a fact about the algorithm, and we can measure it.

Calling that rate a *mass* is a different move entirely, and it is worth
being honest about the gap. A mass, in the Euclidean-field sense, is the
lowest energy carrying spectral weight, and the whole picture of "lowest
energy" only makes sense when the correlator is the Laplace transform of a
*positive* measure: $C(\ell)=\int\lambda^{\ell}\,d\nu(\lambda)$ with
$\nu\ge0$. That representation is what a self-adjoint positive transfer
operator buys you. Our kernel clones and kills walkers. Nobody has shown it
is self-adjoint in the relevant inner product, and for a kernel that is not,
a correlator can go *negative* — you will see an explicit three-state
counterexample below. When that happens there is no positive spectral
measure, no "lowest energy", and the effective rate is not even defined at
the offending lag.

So here is the bookkeeping we adopt. The rate is always available and always
means something. The word "mass" is a promotion, and it costs a stated
hypothesis. Nothing in this chapter is weakened by saying so; you simply know
which claim you are entitled to.
:::

:::{prf:definition} Channel, channel correlator and decay rate
:label: def-qft-channel-decay-rate

Let $(R_n)_{n\ge0}$ be the complete recorded-state chain of
{prf:ref}`thm-effective-twistor-spectral-meaning`, with transition kernel $K$
and an invariant law $\pi$. A **channel** $\chi$ consists of a local operator
$O$ with $c\ge1$ real components, its element selection, masks, weights and
colour alignment ({prf:ref}`def-sm-color-alignment`), and one of the frame
normalizations $\mathcal A_t$ or $\mathcal A_t^{N}$ of
{prf:ref}`def-sm-direct-observable-law`. These data define a frame observable
$f_\chi=(f_\chi^{1},\ldots,f_\chi^{c})$, a function of the recorded state.
Assume $f_\chi^{k}\in L^2(\pi)$. The **channel correlator** is the contracted
connected autocorrelation

$$
C_\chi(\ell)=\sum_{k=1}^{c}
 \operatorname{Cov}_\pi\bigl(f_\chi^{k}(R_0),f_\chi^{k}(R_\ell)\bigr),
\qquad \ell=0,1,2,\ldots
$$

For an assigned time $\Delta\tau>0$ per lag, the effective rate is

$$
m_\chi(\ell)=-\frac1{\Delta\tau}\log\frac{C_\chi(\ell+1)}{C_\chi(\ell)},
$$

defined at the lags where both correlator values are positive. The channel
has the **decay rate** $m_\chi\in[0,\infty]$ when $C_\chi(\ell)>0$ for all
sufficiently large $\ell$ and $m_\chi=\lim_{\ell\to\infty}m_\chi(\ell)$
exists. A fitted plateau is an estimate of $m_\chi$. A channel whose
correlator vanishes at every nonzero lag, or changes sign at arbitrarily large
lags, has no decay rate.

The pair $(K,\pi)$ is a time-homogeneous Markov kernel with an invariant law:
a conservative executed kernel, or the kernel of the established
Doob-transformed process, as declared under
{prf:ref}`def-sm-direct-observable-law`. For a killed chain with almost sure
extinction every invariant law of the killed kernel is carried by the
cemetery state, where all frame observables vanish, so that kernel defines no
channel. For a finite recorded law or a survival-conditioned history the
correlator is the two-time function $C_O(t,s)$ of
{prf:ref}`def-sm-direct-correlations`, and the first identity of
{prf:ref}`thm-effective-twistor-spectral-meaning` replaces item 1 of
{prf:ref}`prop-qft-decay-rate-scope`.

The same effective-rate formula and limit, applied to a source-frozen pair
correlator of {prf:ref}`def-effective-twistor-correlators`, define the
**source-frozen decay rate** of the operator. It is a different quantity from
the decay rate of the frame channel, and
{prf:ref}`prop-qft-decay-rate-scope` is not asserted for it.

In this chapter the symbol $m_\chi$ and the words heavier and lighter refer to
a decay rate in one of these two senses, with the estimator declared
({prf:ref}`rem-qft-declared-conventions`). The name **channel mass** is
reserved for the decay rate $m_\chi$ of a frame channel under
{prf:ref}`assm-qft-positive-transfer`, in the calibrated units required by
{prf:ref}`def-sm-direct-correlations`.
:::

:::{div} feynman-prose
Notice how much of that definition is *bookkeeping* rather than mathematics.
The operator, the mask, the weights, the alignment, the normalization — all
of it is part of the channel. Change any one and you are measuring a
different thing, and you have no right to be surprised when the number moves.
This is not pedantry. Most of the confusion in calibration work comes from
comparing two rates that were never measurements of the same channel.

One clause deserves a second look: the insistence that $(K,\pi)$ be a
conservative kernel with an honest invariant law. Why not just use the killed
chain, the thing the algorithm literally does? Because if the walkers die out
with probability one, the only invariant law of the killed kernel sits on the
cemetery — the state where every observable is zero. Its correlator is
identically zero and its decay rate is meaningless. You must either declare
the conservative kernel, or condition on survival and use the Doob transform.
The definition forces you to say which.
:::

:::{prf:assumption} Positive transfer representation of a channel
:label: assm-qft-positive-transfer

For the channel $\chi$ and each component $k$ there is a finite positive Borel
measure $\nu_\chi^{k}$ on $[0,1]$ with

$$
\operatorname{Cov}_\pi\bigl(f_\chi^{k}(R_0),f_\chi^{k}(R_\ell)\bigr)
 =\int_{[0,1]}\lambda^{\ell}\,d\nu_\chi^{k}(\lambda),
\qquad\ell\ge0.
$$

With $\lambda=e^{-E\Delta\tau}$ this is the representation
$C_O(t)=\int e^{-Et}d\nu_O(E)$ of {prf:ref}`def-sm-direct-correlations`.
It holds when $K$ is self-adjoint and positive on $L^2(\pi)$, in particular
under the hypotheses of {prf:ref}`cor-effective-twistor-positive-transfer`.
No result of this volume establishes it for a gas variant. It is a hypothesis
on the channel, to be tested through the necessary conditions of
{prf:ref}`prop-qft-decay-rate-scope`.
:::

:::{prf:proposition} What a fitted rate measures
:label: prop-qft-decay-rate-scope

Let $\widetilde f^{k}=f_\chi^{k}-\pi f_\chi^{k}$ and let $L_0^2(\pi)$ be the
centred subspace.

1. **Without further hypotheses.**
   $C_\chi(\ell)=\sum_k\langle\widetilde f^{k},K^{\ell}\widetilde f^{k}\rangle_{L^2(\pi)}$
   and $|C_\chi(\ell)|\le\|K^{\ell}\|_{L_0^2(\pi)}\,C_\chi(0)$. Every decay
   rate of a channel is a rate of the semigroup of the executed algorithm on
   the cyclic subspace of its frame observable.
2. **Under {prf:ref}`assm-qft-positive-transfer`.** $C_\chi(\ell)\ge0$ and
   $C_\chi(\ell+1)^2\le C_\chi(\ell)\,C_\chi(\ell+2)$ for all $\ell$. If
   $C_\chi(1)=0$ then $C_\chi(\ell)=0$ for all $\ell\ge1$. If $C_\chi(1)>0$
   then $C_\chi(\ell)>0$ for all $\ell$, the effective rate
   $m_\chi(\ell)$ is nonincreasing, and

   $$
   m_\chi=\lim_{\ell\to\infty}m_\chi(\ell)
        =-\frac1{\Delta\tau}\log\lambda_\chi^{*},
   \qquad
   \lambda_\chi^{*}=\max\operatorname{supp}\nu_\chi,\quad
   \nu_\chi=\sum_k\nu_\chi^{k}.
   $$

   Thus the decay rate exists, the effective rate approaches it from above,
   and $m_\chi$ is the smallest energy carrying spectral weight of the
   channel.
3. **The hypothesis can fail for a non-reversible kernel.** On
   $\mathbb Z/3\mathbb Z$ with uniform $\pi$ let $(Pg)(x)=g(x+1)$,
   $K=(1-a)I+aP$ with $0<a<1$, and $f(x)=\sqrt2\cos(2\pi x/3)$. Then
   $C(\ell)=|\lambda|^{\ell}\cos(\ell\varphi)$ with
   $\lambda=1-\tfrac32a+i\tfrac{\sqrt3}{2}a=|\lambda|e^{i\varphi}$,
   $0<\varphi<\pi$. The correlator is negative at some lag, no positive
   representing measure exists, and the effective rate is undefined there,
   although $|C(\ell)|\le|\lambda|^{\ell}$.
4. **Normal form of the conclusion.** A fitted rate of a channel is a decay
   rate of the executed chain in that channel. It is a channel mass when
   {prf:ref}`assm-qft-positive-transfer` holds for that channel. Negativity
   of $C_\chi$ beyond its statistical error, failure of log-convexity, or an
   effective rate that increases with the lag refutes the hypothesis for
   that channel.

None of these statements is asserted for the source-frozen pair correlators
of {prf:ref}`def-effective-twistor-correlators`, which are ratios of sums
with a lag-dependent valid-pair denominator and pair a source observable with
a different sink observable.
:::

:::{prf:proof}
**Item 1.** The proof of {prf:ref}`thm-effective-twistor-spectral-meaning`
uses the Markov property and square integrability of the frame observable
only; applied to each component and summed it gives the identity. The bound
is Cauchy--Schwarz with $\|\widetilde f^{k}\|^2$ summed to $C_\chi(0)$.

**Item 2.** A sum of positive measures is positive, so
$C_\chi(\ell)=\int\lambda^{\ell}d\nu_\chi\ge0$. Writing
$\lambda^{\ell+1}=\lambda^{\ell/2}\lambda^{(\ell+2)/2}$, Cauchy--Schwarz in
$L^2(\nu_\chi)$ gives the log-convexity inequality. If $C_\chi(1)=0$ then
$\nu_\chi$ is carried by $\{0\}$ and all later values vanish. If
$C_\chi(1)>0$ then $\nu_\chi((0,1])>0$ and every
$C_\chi(\ell)\ge\int_{(0,1]}\lambda^{\ell}d\nu_\chi>0$. Log-convexity makes
$\rho_\ell=C_\chi(\ell+1)/C_\chi(\ell)$ nondecreasing, and
$C_\chi(\ell+1)\le\lambda_\chi^{*}C_\chi(\ell)$ gives
$\rho_\ell\le\lambda_\chi^{*}$; let $\rho_\infty$ be its limit. From
$C_\chi(\ell)\le C_\chi(1)\rho_\infty^{\ell-1}$ one gets
$\limsup C_\chi(\ell)^{1/\ell}\le\rho_\infty$, while
$C_\chi(\ell)^{1/\ell}=\|\lambda\|_{L^{\ell}(\nu_\chi)}
 \to\|\lambda\|_{L^{\infty}(\nu_\chi)}=\lambda_\chi^{*}$. Hence
$\rho_\infty=\lambda_\chi^{*}$, and $m_\chi(\ell)=-\Delta\tau^{-1}\log\rho_\ell$
decreases to the stated limit. With $\lambda=e^{-E\Delta\tau}$ the maximum of
the support in $\lambda$ is the minimum in $E$.

**Item 3.** The characters $e_{\pm1}(x)=e^{\pm2\pi ix/3}$ are orthonormal in
$L^2(\pi)$, $Pe_{\pm1}=e^{\pm2\pi i/3}e_{\pm1}$, and
$f=(e_1+e_{-1})/\sqrt2$ is centred. Therefore
$K^{\ell}f=(\lambda^{\ell}e_1+\overline\lambda^{\ell}e_{-1})/\sqrt2$ and
$C(\ell)=\operatorname{Re}\lambda^{\ell}$. Since $\operatorname{Im}\lambda>0$,
$0<\varphi<\pi$; steps smaller than $\pi$ cannot jump over the arc
$(\pi/2,3\pi/2)$, so $\cos(\ell\varphi)<0$ for some $\ell$. A positive
representing measure would force $C\ge0$.

**Item 4.** This collects items 1 and 2; the three refutation criteria are
the contrapositives of the three necessary conditions in item 2. $\square$
:::

:::{div} feynman-prose
Item 2 is the one to carry around in your head. Under the positivity
hypothesis, the correlator is a mixture of pure decaying exponentials with
nonnegative weights. Mix exponentials and the slowest one always wins in the
end — so the ratio of successive values can only *rise* toward the slowest
$\lambda$, which means the effective rate can only *fall* toward the true
rate. That is why the plateau in a well-behaved channel is approached from
above, and why an effective-mass curve that drifts *upward* with the lag is
not noise you should average away. It is telling you the hypothesis is wrong
for that channel.

Now, item 3 is a small, concrete, completely explicit machine that breaks the
hypothesis, and I want you to take it seriously rather than filing it under
"pathological". Three states on a ring; at each step you stay with
probability $1-a$ or step forward with probability $a$. Nothing exotic. But
the motion has a *direction*, and direction means complex eigenvalues, and
complex eigenvalues mean the correlator oscillates as it decays. Take
$a=0.4$: the correlator runs $1,\ 0.4,\ 0.04,\ -0.08,\ldots$ It goes
negative at lag three. You cannot take the log of a negative number, and no
positive measure on $[0,1]$ can produce it. A reversible chain — one obeying
detailed balance — has real spectrum and cannot do this. The gas is not known
to be reversible. That is the whole content of the warning.

Let me also say what item 1 does *not* say. It does not say your fit is
meaningless without the hypothesis. It says the rate you fit is a decay rate
of the algorithm's own semigroup on the subspace your observable generates —
a genuine dynamical quantity, comparable across runs, responsive to knobs.
You just cannot call it the lowest energy of a spectrum until you have earned
the spectrum.
:::

:::{prf:proposition} The two frame normalizations
:label: prop-qft-frame-normalizations

Let $O$ be a local operator with
$\sup_I|O_I|<\infty$, or more generally with both frame observables in
$L^2(\pi)$.

1. $\mathcal A_t(O)$, including its zero-denominator value, and
   $\mathcal A_t^{N}(O)=(W_t/N)\mathcal A_t(O)$ are functions of the recorded
   state $R_t$. Item 1 of {prf:ref}`prop-qft-decay-rate-scope` holds for each
   of them with the same kernel $K$.
2. Their correlators are
   $\operatorname{Cov}(\mathcal A_0,\mathcal A_\ell)$ and
   $N^{-2}\operatorname{Cov}(W_0\mathcal A_0,W_\ell\mathcal A_\ell)$. They are
   proportional for every operator when $W_t$ is almost surely constant, and
   need not be proportional otherwise. A decay rate, and under
   {prf:ref}`assm-qft-positive-transfer` a spectral weight, is a property of
   the channel including its normalization.
3. A series from which the frames with $W_t=0$ have been removed is a
   function of the recorded state on $\{W>0\}$ only, sampled at
   state-dependent times. When $\pi(W=0)=0$ it is almost surely the full
   series and item 1 of {prf:ref}`prop-qft-decay-rate-scope` applies to it
   with the kernel $K$. When $\pi(W=0)>0$ it is a function of the trace chain
   of $R$ on $\{W>0\}$, whose kernel is the first-return kernel
   $K_{W>0}(x,\cdot)=\mathbb P_x(R_{\tau}\in\cdot)$,
   $\tau=\min\{n\ge1:W(R_n)>0\}$, with invariant law
   $\pi(\cdot\mid W>0)$; its lag counts retained frames, not steps, and its
   correlator is not $\langle\widetilde f,K^{\ell}\widetilde f\rangle_{L^2(\pi)}$
   in general. Frames that the record cannot
   evaluate for a reason independent of the state, such as the first frame of
   a segment under $\mathsf A_{\mathrm{PK}}$, are missing data and not
   zeros.

The chapter-04 average $\mathcal A_t$ is the primary normalization;
$\mathcal A_t^{N}$ is the alternative of
{prf:ref}`thm-effective-twistor-spectral-meaning`. Every reported rate states
which one it uses.
:::

:::{prf:proof}
$W_t$ and the numerator are finite sums of functions of the recorded fields
of frame $t$, and the zero-denominator branch is a measurable case
distinction; this gives item 1 together with the cited proof. Item 2 is the
definition of the two series; if $W_t=W$ almost surely the second covariance
is $(W/N)^2$ times the first. For the converse direction take $O\equiv1$ and
a law with $W_t>0$ almost surely and $\operatorname{Var}_\pi W>0$: then
$\mathcal A_t(O)=1$ has the zero correlator, while $\mathcal A_t^{N}(O)=W_t/N$
has $C(0)=N^{-2}\operatorname{Var}_\pi W>0$. For item 3, the retained series
is $\mathcal A(R_{n_j})$ along the random times $n_j$ with $W_{n_j}>0$; when
$\pi(W=0)=0$ these are almost surely all times. When $0<\pi(W>0)<1$, the
stationary chain started in $\{W>0\}$ returns to that set almost surely by
the Poincaré recurrence theorem, the strong Markov property at the successive
return times makes $(R_{n_j})_j$ a Markov chain with the first-return kernel,
and $\pi(\cdot\mid W>0)$ is invariant for it. A position in the record
fixed before the run does not depend on the state, so its removal leaves the
chain law of the retained frames unchanged. $\square$
:::

:::{div} feynman-prose
You might think dividing by the number of valid pairs instead of by the fixed
$N$ is a cosmetic choice — a constant, near enough, that cancels out of any
ratio. It is not, and item 2 says exactly why. The valid-pair count $W_t$ is
itself a fluctuating function of the state. Dividing by it does not rescale
the series; it *multiplies the series by a second random observable*, and the
correlator of a product is not the product of correlators. The two
normalizations agree only in the degenerate case where $W_t$ never moves.

Item 3 is subtler and it catches people. Suppose you throw away the frames
where nothing was valid. You have not cleaned your data — you have sampled
your chain at times chosen by the chain itself. The retained frames follow the
first-return kernel, not $K$, and a lag now counts retained frames instead of
steps, so the rate you fit belongs to a different chain. It is harmless only when those
frames are almost never there to begin with. And there is one honest
exception worth separating out: a frame that cannot be evaluated for a reason
fixed before the run — the first frame of a segment, say — is missing data.
Do not record it as a zero. A zero is a measurement; a gap is not.
:::

:::{prf:proposition} Component contraction versus component mean
:label: prop-qft-component-contraction

Let $A_t\in\mathbb R^{d}$ be the component series of a channel, with
$C_{kl}(\ell)=\operatorname{Cov}(A_0^{k},A_\ell^{l})$. Under an orthogonal
change of the component basis $A_t\mapsto RA_t$, $R\in O(d)$:

1. the contracted correlator $\sum_kC_{kk}(\ell)=\operatorname{tr}C(\ell)$ is
   invariant;
2. the correlator of the component mean
   $\bar A_t=d^{-1}\sum_kA_t^{k}$ is
   $d^{-2}\,\mathbf 1^{\mathsf T}C(\ell)\mathbf 1$ and becomes
   $d^{-2}(R^{\mathsf T}\mathbf 1)^{\mathsf T}C(\ell)(R^{\mathsf T}\mathbf 1)$.
   For $d\ge2$ it is invariant under all of $O(d)$ if and only if the
   symmetric part of $C(\ell)$ is a multiple of the identity, in which case
   it equals $d^{-2}\operatorname{tr}C(\ell)$.

The component mean is the projection of the vector series on the fixed
direction $\mathbf 1/d$ of the recorded basis. Vector channels are therefore
correlated by contraction, as in {prf:ref}`def-sm-direct-correlations` and
{prf:ref}`def-effective-twistor-correlators`. The statement concerns the
component index. Covariance of the underlying observable under rotations of
the particle system is a separate property, which the componentwise colour
encoding does not have ({prf:ref}`thm-sm-su3-emergence`).
:::

:::{prf:proof}
$\operatorname{Cov}(RA_0,RA_\ell)=RC(\ell)R^{\mathsf T}$ and the trace is
cyclic. The mean is $d^{-1}\mathbf 1^{\mathsf T}A_t$, which gives the
quadratic form; only the symmetric part of $C(\ell)$ contributes to it. The
orbit of $\mathbf 1/\sqrt d$ under $O(d)$ is the unit sphere, and a quadratic
form that is constant on the unit sphere is a multiple of the identity; the
constant is $\operatorname{tr}C(\ell)/d$. $\square$
:::

:::{div} feynman-prose
Here is a habit worth breaking. You have a three-component vector channel,
and the tempting move is to average the three components into one number and
correlate that. Don't. Averaging the components is *dotting your vector
series with the fixed direction* $(1,1,1)/3$ — a direction that has no
meaning except that it is where your array indices happened to point. Rotate
the component basis and that direction moves with the labels, and your
correlator changes.

Contract instead: correlate each component with itself and add the results.
That is a trace, and a trace does not care what basis you wrote the matrix
in. The two agree only when the symmetric part of the correlation matrix is
already a multiple of the identity — that is, only when the channel had no
preferred direction to begin with, which is precisely the assumption you were
trying to avoid making.

And now the caveat, because an analogy is about to run away with itself. This
is a statement about the *component index*, not about physical rotations. The
colour encoding is not covariant under rotating the particle system; see
{prf:ref}`thm-sm-su3-emergence`. Contraction buys you independence from how
you labelled three slots in an array. It does not buy you a rotational
quantum number. Those are different claims, and only the first one is proved.
:::

:::{prf:remark} Conventions that a reported rate declares
:label: rem-qft-declared-conventions

The definitions of this volume leave the following choices open. Each is part
of the channel of {prf:ref}`def-qft-channel-decay-rate`, and a reported rate
states them.

1. The colour alignment of {prf:ref}`def-sm-color-alignment`; the primary one
   is $\mathsf A_{\mathrm{PK}}$.
2. The frame normalization, $\mathcal A_t$ or $\mathcal A_t^{N}$, and the
   treatment of frames with $W_t=0$
   ({prf:ref}`prop-qft-frame-normalizations`).
3. The estimator: frame-average correlator or source-frozen pair correlator
   ({prf:ref}`def-effective-twistor-correlators`). Only the former is covered
   by {prf:ref}`prop-qft-decay-rate-scope`.
4. The centring: one empirical mean of the series
   ({prf:ref}`def-sm-direct-correlations`) or separate means of the two lag
   windows. The two differ at finite record length.
5. For the colour-gamma form, the sign pattern of $\Gamma_5$ and the recorded
   part; for the determinant channel, the recorded part of $b$.
6. The scales $h_S$ and $\hbar_{\text{eff}}$ of the score and fitness phases
   ({prf:ref}`rem-qft-ew-ranges`), and the amplitude
   $\sqrt{w}$ of this chapter versus the normalized $\sqrt{P_i(k)}$ of
   {prf:ref}`thm-sm-u1-emergence`.
7. The time unit $\Delta\tau$ per lag ({prf:ref}`thm-qft-ratio-rescale`).
:::

:::{div} feynman-prose
Seven items, and every one of them is a place where two honest people using
the same code can produce two different numbers and both be right. That is
not a defect in the framework; it is what it looks like when a measurement is
specified completely enough to be reproduced. A rate reported without them is
not wrong so much as unfalsifiable — nobody can rerun it.

The practical advice is boring and I will give it anyway: write the seven down
next to the number, in the file, every time. It costs you a line. It is the
difference between a result and an anecdote.
:::

(sec-qft-calibration-couplings)=
## Couplings and interaction ranges

:::{div} feynman-prose
The following coupling assignments use the normalization conventions and hypotheses of the cited results. They organize parameter comparisons within those models. A scalar dashboard phase or a channel name alone does not identify a matrix-valued gauge transport or its physical coupling.
:::



$$
g_1^2 = \frac{\hbar_{\text{eff}}}{\epsilon_d^2}\,\mathcal{N}_1(T,d)
$$
({prf:ref}`thm-sm-g1-coupling`)

$$
g_2^2 = \frac{2\hbar_{\text{eff}}}{\epsilon_c^2}\,\frac{C_2(2)}{C_2(d)}
$$
({prf:ref}`thm-sm-g2-coupling`)

$$
g_d^2 = \frac{\nu^2}{\hbar_{\text{eff}}^2}\,\frac{d(d^2-1)}{12}\,\langle K_{\text{visc}}^2\rangle_{\text{QSD}}
$$
({prf:ref}`thm-sm-g3-coupling`)

$$
e_{\text{fitness}}^2 = \frac{m}{\epsilon_F}
$$
({prf:ref}`thm-u1-coupling-constant`)

:::{div} feynman-prose
With the other factors fixed, these expressions give the displayed inverse-range and amplitude scalings. In a new simulation, changing a parameter can also change the QSD statistics, including the kernel average. The response of a fitted channel mass must therefore be measured; it does not follow from a prefactor alone.
:::



The scale conventions and separation regime in {prf:ref}`thm-mass-scales` are:

$$
m_{\text{clone}} = 1/\epsilon_c,\quad
m_{\text{MF}} = 1/\rho,\quad
m_{\text{gap}} = \hbar_{\text{eff}}\lambda_{\text{gap}},\quad
m_{\text{friction}} = \gamma.
$$

:::{div} feynman-prose
These characteristic scales describe the regime of the cited model. Applying a result that assumes their hierarchy requires checking that hierarchy. The actual channel decay also depends on the observable and its overlap with the evolving modes.
:::



(sec-qft-calibration-channels)=
## Channel sensitivity map (theory to knobs)

Channel operators are built from Fractal Set ingredients:

- Companion kernels and algorithmic distance ({prf:ref}`def-fractal-set-companion-kernel`,
  {prf:ref}`def-fractal-set-alg-distance`).
- Two-channel fitness and cloning score ({prf:ref}`def-fractal-set-two-channel-fitness`,
  {prf:ref}`def-fractal-set-cloning-score`).
- Viscous coupling and color state ({prf:ref}`def-fractal-set-viscous-force`,
  {prf:ref}`thm-sm-su3-emergence`).
- Colour triangle products for glueball channels
  ({prf:ref}`def-sm-direct-color-contractions`,
  {prf:ref}`prop-sm-direct-triangle-projectors`).

The table below summarizes which knobs primarily move which channel families. The suggested directions are sweep hypotheses. Their signs and magnitudes require validation for the chosen observable, generating run, and fit window.

| Channel family | Fractal Set ingredient | Primary knobs | Expected qualitative effect (operational) |
| --- | --- | --- | --- |
| Meson / pseudoscalar (color bilinear) | Color state from viscous force ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, $\gamma$, $\beta$, $\Delta t$ | Shorter $\rho$ or larger $\nu$ increases color coupling, typically shortening correlators (heavier masses). |
| Baryon / nucleon (color determinant) | SU(3) invariant of three color vectors ({prf:ref}`thm-sm-su3-emergence`) | $\nu$, $\rho$, neighbor selection | Trilinear color invariants are sensitive to color coherence; adjust $\nu$ and $\rho$ first. |
| Glueball / gauge channel | Colour triangle product; alternative: viscous force norm ({prf:ref}`def-sm-direct-color-contractions`, {prf:ref}`def-fractal-set-viscous-force`) | $\nu$, $\rho$ | Stronger viscous coupling or shorter $\rho$ tends to increase glueball mass scales. |
| Cloning/diversity-dominated channels | Companion kernel + cloning score ({prf:ref}`def-fractal-set-companion-kernel`, {prf:ref}`def-fractal-set-cloning-score`) | $\epsilon_c$, $\epsilon_d$, $\lambda_{\text{alg}}$, $\epsilon_{\text{clone}}$, $p_{\max}$ | Decreasing $\epsilon_c$ or $\epsilon_d$ strengthens the corresponding coupling and can shift correlator decay. |
| Fitness/U(1) phase channels | Phase potential and fitness coupling ({prf:ref}`def-fractal-set-phase-potential`, {prf:ref}`thm-u1-coupling-constant`) | $\epsilon_F$, fitness weights $(\alpha,\beta)$ | Larger $\epsilon_F$ weakens the fitness coupling, softening phase-driven oscillations. |

(sec-qft-calibration-channel-derivations)=
## Channel operators and calibration parameters

The electroweak correlator and mass tabs wired by `src/fragile/physics/app/dashboard.py` report
**Extracted Masses** from Euclidean two-point correlators. The generic correlator machinery lives in
`src/fragile/physics/new_channels/correlator_channels.py` and
`src/fragile/physics/aic/correlator_channels.py`, while the electroweak-specific assembly happens
in `src/fragile/physics/app/electroweak_correlators.py` and
`src/fragile/physics/app/electroweak_mass_tab.py`. For any channel operator $O_\chi$,

$$
C_\chi(\tau) = \langle O_\chi(\tau)\,O_\chi(0)\rangle_{\text{conn}}
$$
({prf:ref}`def-euclidean-correlator-fg`, {prf:ref}`def-two-point-connected`).

For a channel whose decay rate exists ({prf:ref}`def-qft-channel-decay-rate`), write the asymptotic form and effective-rate estimator as:

$$
C_\chi(\tau) \sim Z_\chi e^{-m_\chi \tau},
\qquad
m_\chi(\tau) = -\frac{1}{\Delta \tau}\log\frac{C_\chi(\tau+\Delta\tau)}{C_\chi(\tau)},
\qquad
\xi_\chi = \frac{1}{m_\chi}
$$

using the correlation-length definition {prf:ref}`def-correlation-length` and the mass-scale
hierarchy {prf:ref}`thm-mass-scales`. Here $\xi_\chi=1/m_\chi$ is a correlation time in units of
$\Delta\tau$; it is a length only under the Euclidean rotation identification stated in
{prf:ref}`def-correlation-length`. The AIC-weighted plateau in the Channels tab is an
implementation of this $m_\chi(\tau)$ extraction, so its output depends on the operator, sampling, and fit window. The operator formulas identify parameters to investigate; they do not establish universal monotonic tuning rules.

:::{div} feynman-prose
For a fixed correlator sequence indexed by frame lag, changing the assigned
time unit rescales every fitted decay rate by the inverse factor. Changing
$\Delta t$ in a new simulation changes its transition kernel, while changing
recording stride changes the sampled data. Neither operation is merely a
change of units. Compare dynamical sweeps at controlled time resolution and
check discretization and recording effects separately.

The scale hierarchy is a hypothesis of the corresponding continuum or
spectral model. Use it when applying those results, alongside checks of the
observed fit stability and uncertainty.
:::

Implementation note: the active electroweak route exposes analysis knobs such as `h_eff`,
`mass`, `ell0`, `ell0_method`, `max_lag`, `use_connected`, and the Bayesian fit settings in the
mass tab. These belong to the **measurement** map, not the underlying swarm dynamics. The first
three appear directly in the color-state and spinor constructions
({prf:ref}`thm-sm-su3-emergence`, {prf:ref}`def-lqft-chiral-projectors`), so set them consistently
with the run. Changing them can move extracted masses without changing the simulation itself.

The table below makes the knob-to-parameter correspondence explicit.

| Knob (symbol) | Algorithm parameter name | Location (code) | Role in calibration |
| --- | --- | --- | --- |
| $\nu$ | `nu` | `KineticOperator` (`src/fragile/physics/fractal_gas/kinetic_operator.py`) | Viscous coupling strength (color/gauge sector) |
| $\rho$ | `viscous_length_scale` | `KineticOperator` | Localization range of viscous kernel |
| $\gamma$ | `gamma` | `KineticOperator` | Friction mass scale ($m_{\text{friction}}$) |
| $\beta$ | `beta` | `KineticOperator` | Inverse temperature (noise scale) |
| $\Delta t$ | `delta_t` | `KineticOperator` | Integrator step; changes the dynamics when rerunning |
| $\epsilon_F$ | `epsilon_F` | `KineticOperator` | Fitness/U(1) coupling scale |
| $\epsilon_c$ | `companion_selection_clone.epsilon` | `RunHistory.params` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Clone-companion interaction range used by electroweak operators |
| $\epsilon_d$ | `companion_selection.epsilon` | `RunHistory.params` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Diversity-companion interaction range used by electroweak operators |
| $\lambda_{\text{alg}}$ | `lambda_alg` | Fixed inside `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | Pinned to `0.0` in the active electroweak pipeline |
| $\epsilon_{\text{clone}}$ | `epsilon_clone` | `ElectroweakCorrelatorSettings` in `src/fragile/physics/app/electroweak_correlators.py`, falling back to `CloneOperator` (`src/fragile/physics/fractal_gas/cloning.py`) via `RunHistory.params` | Cloning-score regularization entering SU(2) and chirality operators |
| $p_{\max}$ | `p_max` | `CloneOperator` (`src/fragile/physics/fractal_gas/cloning.py`) | Max cloning probability in the recorded dynamics |
| Fitness weights $(\alpha,\beta,\eta,A,\rho)$ | `alpha`, `beta`, `eta`, `A`, `rho` | `FitnessOperator` (`src/fragile/physics/fractal_gas/fitness.py`) | Fitness coupling shape |
| $\hbar_{\text{eff}}$ | `h_eff` | `ElectroweakCorrelatorSettings` (`src/fragile/physics/app/electroweak_correlators.py`) | Measurement: phase scale for chirality and Dirac-spinor operators |
| $m$ (phase mass) | `mass` | `ElectroweakCorrelatorSettings` | Measurement: color-state phase factor in the spinor path |
| $\ell_0$ | `ell0` | `ElectroweakCorrelatorSettings` | Measurement: color-state length scale in the spinor path |
| $\ell_0$ method | `ell0_method` | `ElectroweakCorrelatorSettings` | Measurement: automatic estimator for the spinor path when `ell0` is blank |
| Connected correlator | `use_connected` | `ElectroweakCorrelatorSettings` | Measurement: connected vs raw $C(t)$ |
| Max lag | `max_lag` | `ElectroweakCorrelatorSettings` | Measurement: correlator window length |
| Warmup fraction | `warmup_fraction` | `ElectroweakCorrelatorSettings` | Measurement: drop transient steps |
| Covariance / prior fit controls | `covariance_method`, `nexp`, `tmin`, `tmax`, `svdcut`, `use_log_dE`, `use_fastfit_seeding`, `effective_mass_method`, `include_multiscale` | `ElectroweakMassSettings` (`src/fragile/physics/app/electroweak_mass_tab.py`) | Bayesian mass-extraction and plateau-fitting controls |

The older electroweak UI in `src/fragile/physics/app/electroweak.py` retains additional knobs such
as `knn_k`, `knn_sample`, and `window_widths_spec`. Those belong to that legacy/alternate
interface, not to the active `dashboard.py` route documented in this chapter.

Below, each channel is tied to its operator, the Fractal Set ingredients that define it, and the
parameters that control its correlator decay.

:::{div} feynman-prose
A normalization can remove an apparent tuning parameter from a fixed frame.
In the displayed color encoding, multiplying every component of a nonzero
viscous force vector by the same positive number leaves its normalized color
state unchanged, with the phase held fixed. Varying viscous strength in a new
run can still change the trajectory. Distinguish that dynamical effect from
recomputing an observable on the same recorded frame.
:::

### Scalar channel (label σ)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger color coupling → shorter correlator → heavier $m_\sigma$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter localization → heavier $m_\sigma$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster velocity relaxation → heavier $m_\sigma$ (keep hierarchy). |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → weaker phase winding → slightly lighter $m_\sigma$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\sigma$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\sigma$. |

**Operator (primary, {prf:ref}`def-sm-direct-color-contractions`):**

$$
O_{\sigma}(t)=\mathcal A_t\bigl(\operatorname{Re}q_{i\,c(i)}\bigr),
\qquad q_{ij}=c_i^\dagger c_j,
$$

with $c$ the selected companion map. It is even under pair exchange and under
inversion ({prf:ref}`cor-sm-direct-exchange-parity`).

The color state $c_i$ is built from the viscous force and momentum-phase encoding
({prf:ref}`thm-sm-su3-emergence`):

$$
\tilde{c}_i^{(\alpha)} = F_\alpha^{(\text{visc})}(i)\,
\exp\!\left(i\,p_i^{(\alpha)}\ell_0/\hbar_{\text{eff}}\right),
\quad
c_i^{(\alpha)} = \frac{\tilde{c}_i^{(\alpha)}}{\|\tilde{c}_i\|}.
$$

The pairing of the force with the phase velocity is an alignment of
{prf:ref}`def-sm-color-alignment`.

Therefore the scalar correlator is controlled by the viscous force
({prf:ref}`def-fractal-set-viscous-force`) and the $SU(d)$ coupling
({prf:ref}`thm-sm-g3-coupling`), with the mean-field range $\rho$ and friction $\gamma$ setting the
dominant decay scales ({prf:ref}`thm-mass-scales`).

:::{div} feynman-prose
Look at what the scalar channel actually is once the fog clears. Each walker
carries a unit complex vector $c_i$ — its colour. You pick its companion,
form the overlap $q_{ij}=c_i^\dagger c_j$, and take the real part. That is
the cosine of the angle between two colours, in the complex sense. Average it
over the frame and you have a single number per step: how aligned the gas is
with itself, right now.

Two details in that construction are easy to skate past, and both matter.
First, $\operatorname{Re}q$ and not $q$: the overlap is a complex number, and
a complex number is not an observable. You must say which real part of it you
record, and the answer here is the real part, which is symmetric under
swapping the two walkers and unchanged under inversion. Second, the colour
itself is built by pairing a *force* with a *phase velocity*
({prf:ref}`def-sm-color-alignment`). There is more than one defensible way to
line those two up in time, they give genuinely different numbers, and the
choice travels with the channel. The alignment is not a detail of the code;
it is part of what you measured.
:::

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen the viscous coupling and shorten the scalar
  correlation length (heavier scalar mass).
- Decrease $\nu$ or increase $\rho$ to soften the coupling and lengthen the plateau (lighter
  scalar mass).
- Increasing $\gamma$ raises $m_{\text{friction}}$ and typically shortens scalar plateaus; keep the
  hierarchy $m_{\text{friction}} \ll m_{\text{gap}}$ intact.

### Pseudoscalar channel (label π)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase dispersion → lighter $m_\pi$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → more phase winding → heavier $m_\pi$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → more phase winding → heavier $m_\pi$. |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → lifts overall meson scale → heavier $m_\pi$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter coupling → heavier $m_\pi$. |

**Operator (primary, {prf:ref}`def-sm-direct-color-contractions`):**

$$
O_{\pi}(t)=\mathcal A_t\bigl(\operatorname{Im}q_{i\,c(i)}\bigr).
$$

At $\kappa=0$ the colours are real and $\operatorname{Im}q_{ij}=0$: this
channel is generated entirely by the momentum phase
$\exp(i\,p_i^{(\alpha)}\ell_0/\hbar_{\text{eff}})$ of the color state
({prf:ref}`thm-sm-su3-emergence`), which makes $\kappa=m\ell_0/\hbar_{\text{eff}}$
the knob that separates it from the scalar channel without changing the
viscous coupling. It is odd under pair exchange and under inversion. On a
mutual pairing its frame series is identically zero
({prf:ref}`cor-sm-direct-exchange-parity`); a decay rate is then available
only from an orientation-weighted average or from a source-frozen pair
correlator ({prf:ref}`rem-exchange-odd-scope`).

**Alternative operator (colour-gamma form):** $O_{\pi}^{\Gamma_5}$ of
{prf:ref}`def-qft-color-gamma-operators`. Its real part is even under
inversion ({prf:ref}`prop-qft-color-gamma-parities`).

:::{div} feynman-prose
The scalar took the real part of the overlap; the pseudoscalar takes the
imaginary part. That is the whole difference, and it is a beautiful one,
because the imaginary part has nowhere to come from except the phase. Turn
$\kappa=m\ell_0/\hbar_{\text{eff}}$ down to zero and every colour becomes a
real vector, every overlap becomes a real number, and this channel is
identically nothing. So $\kappa$ is a knob that moves the pseudoscalar while
leaving the viscous coupling — and therefore the scalar's main driver —
alone. That is exactly the kind of lever you want in a calibration.

Now the warning, and it is a sharp one. $\operatorname{Im}q$ is *odd* under
exchanging the two walkers of a pair. If your companion map is mutual — $i$
points to $j$ and $j$ points right back at $i$ — then every pair contributes
twice with opposite signs, and the frame average is zero. Not small. Not
noisy. Algebraically zero, at every step, for every run. You can fit an
exponential to that series all day and the number you get will be a fit to
floating-point dust.

So if you want a pseudoscalar rate, you must break the cancellation on
purpose: weight the pair by an orientation, or freeze the source and use the
pair correlator of {prf:ref}`rem-exchange-odd-scope`. Either is fine. Doing
neither and reporting a number is not.
:::

**Sweep hypotheses to check:**
- Increase $m$ or $\ell_0$, or decrease $\hbar_{\text{eff}}$, to increase phase winding and shorten
  the pseudoscalar correlation length (heavier pseudoscalar).
- Decrease $m$ or $\ell_0$, or increase $\hbar_{\text{eff}}$, to reduce phase dispersion (lighter
  pseudoscalar).
- Sweep $\nu$ and $\rho$ to measure whether the scalar and pseudoscalar scales move together
  through their dependence on the viscous-force coupling.

### Vector channel (label ρ)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger alignment → heavier $m_\rho$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → tighter alignment → heavier $m_\rho$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster decay of coherent modes → heavier $m_\rho$. |
| Phase scale | $\hbar_{\text{eff}}$ | `CompanionCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → less phase winding → slightly lighter $m_\rho$. |
| Phase mass | $m$ | `CompanionCorrelatorSettings.mass` | Increase $m$ → stronger phase winding → slightly heavier $m_\rho$. |
| Phase length | $\ell_0$ | `CompanionCorrelatorSettings.ell0` | Increase $\ell_0$ → stronger phase winding → slightly heavier $m_\rho$. |

**Operator (primary, {prf:ref}`def-sm-direct-color-contractions`):**

$$
O_{\rho}^{k}(t)=\mathcal A_t\bigl(\operatorname{Re}q_{i\,c(i)}\,r_{i\,c(i)}^{k}\bigr),
\qquad k=1,\ldots,d,
\qquad C_\rho(\ell)=\sum_{k}C_{\rho,kk}(\ell),
$$

with the contraction of {prf:ref}`prop-qft-component-contraction`. It is odd
under pair exchange and under inversion; on a mutual pairing every component
of its frame series is identically zero
({prf:ref}`cor-sm-direct-exchange-parity`). The axial companion
$\operatorname{Im}q_{ij}\,r_{ij}$ is even under both and is not constrained.

**Alternative operator (colour-gamma form):** $O_{\rho}^{\Gamma,\mu}$ of
{prf:ref}`def-qft-color-gamma-operators`, contracted over $\mu$. The series
$d^{-1}\sum_\mu O_{\rho}^{\Gamma,\mu}$ is its component mean in the sense of
{prf:ref}`prop-qft-component-contraction`.

:::{div} feynman-prose
The vector channel is the scalar overlap with the separation vector attached:
$\operatorname{Re}q_{ij}$ times $r_{ij}^{k}$, the $k$-th component of the
displacement between the pair. So it does not just ask whether two walkers
have aligned colours; it asks whether they have aligned colours *and which
way one lies from the other*. That is what earns it the word "vector".

Two consequences follow immediately from that extra factor. The displacement
flips sign when you swap the pair, so the whole thing is exchange-odd, and
the mutual-pairing cancellation of the pseudoscalar bites here too — every
component, identically zero. And because it now carries a free index, you
must decide what to do with $d$ series rather than one. Contract them: sum
the $d$ self-correlators. Do not average the components first.
{prf:ref}`prop-qft-component-contraction` explains why, and the alternative
colour-gamma form below is exactly a case where the averaged version has been
used and deserves its own name.
:::

The vector projection emphasizes **directional coherence** in the color state, which is driven by
velocity alignment in the viscous force ({prf:ref}`def-fractal-set-viscous-force`) and damped by
friction ($m_{\text{friction}}=\gamma$; {prf:ref}`thm-mass-scales`).

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen alignment and shorten the vector correlator
  (heavier vector mass).
- Increase $\gamma$ to speed velocity relaxation, which typically shortens vector plateaus.
- Keep $\Delta t$ fixed when comparing vector masses across runs (see the time-scale normalization
  rule above).

### Nucleon channel (baryon, color determinant)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → tighter color coherence → heavier $m_N$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → stronger local binding → heavier $m_N$. |
| Clone temperature | $\epsilon_c$ | Recorded in `RunHistory.params["companion_selection_clone"]["epsilon"]` | Decrease $\epsilon_c$ in the generating run → stronger clone locality in the recorded companion graph → typically heavier $m_N$. |
| Diversity temperature | $\epsilon_d$ | Recorded in `RunHistory.params["companion_selection"]["epsilon"]` | Decrease $\epsilon_d$ in the generating run → tighter distance locality → typically heavier $m_N$. |
| Alg. distance weight | $\lambda_{\text{alg}}$ | Recorded run parameter when present | Larger $\lambda_{\text{alg}}$ strengthens velocity-weighted locality in the generating run and can raise $m_N$. |
| Pair selection | — | `CompanionCorrelatorSettings.pair_selection` | Measurement: choose distance pairs, clone pairs, or both when building local triplets; this changes the estimator, not the recorded dynamics. |
| Multiscale locality | — | `CompanionCorrelatorSettings.n_scales`, `kernel_type`, `edge_weight_mode` | Measurement: changes neighborhood weighting and plateau stability for baryon correlators without changing the run itself. |

**Operator (primary, {prf:ref}`def-sm-direct-color-contractions`):**

$$
b_{ijk}=\det\!\big[c_i,c_j,c_k\big],\qquad
(i,j,k)=(i,c^{D}(i),c^{C}(i)),
$$

read through $\operatorname{Re}b$, $\operatorname{Im}b$, or the complex
source-frozen correlator $\operatorname{Re}(\overline{B_s}B_t)$ of
{prf:ref}`prop-sm-baryon-exterior-correlator`. The determinant is invariant
under common $SU(3)$ frame changes
({prf:ref}`thm-sm-direct-color-invariants`) and changes sign when the two
companion roles are exchanged. When the two roles are exchangeable, the frame
series of $\operatorname{Re}b$ and of $\operatorname{Im}b$ are centred and
uncorrelated at every nonzero lag ({prf:ref}`prop-sm-direct-role-swap`); the
decay rate $m_N$ is then defined through the source-frozen correlator only.
The phase-blind readout obeys, for unit colours,

$$
|b_{ijk}|^2=1-|q_{ij}|^2-|q_{jk}|^2-|q_{ki}|^2+2\operatorname{Re}\Pi_{ijk},
$$

because $|\det C|^2=\det(C^\dagger C)$ is the determinant of the Gram matrix
of the three columns. It is even under role exchange, and its correlator is a
combination of pair and triangle correlators. The readout $|b|$ of the
reference operator module is a separately specified function, as stated in
{prf:ref}`prop-sm-baryon-exterior-correlator`.

:::{div} feynman-prose
The determinant of three unit colour vectors measures how much *volume* they
span. Three colours pointing nearly the same way give a determinant near
zero; three mutually orthogonal ones give modulus one. That is a genuine
three-body quantity — you cannot build it out of pairs — and it is invariant
under a common $SU(3)$ rotation of all three, which is why it deserves the
baryon slot.

But the determinant is complex, and antisymmetric, and this is where you have
to be careful. Swap the two companion roles and $b$ changes sign. If nothing
in the algorithm distinguishes those two roles — if they are exchangeable —
then $\operatorname{Re}b$ and $\operatorname{Im}b$ are centred and, worse,
uncorrelated at every nonzero lag. Uncorrelated at every lag is a correlator
that is zero everywhere except at the origin. There is no exponential in
that. There is no plateau. There is no $m_N$.

The escape is to freeze the source: correlate $\overline{B_s}B_t$ with the
triplet identity fixed at the source frame, which is a different estimator and
survives the antisymmetry. And if instead you take the modulus and throw the
phase away, you get something real and role-even — but look at the Gram
identity above and see what you have actually bought. $|b|^2$ is one, minus
the three pair overlaps, plus twice the triangle invariant. It is not an
independent channel at all; it is a fixed combination of the pair and
glueball channels wearing a baryon's name.
:::

This channel is an $SU(3)$-invariant trilinear built from the same color state
({prf:ref}`thm-sm-su3-emergence`). It probes **three-body color coherence**, which depends both on
the viscous coupling (for color alignment) and on the companion/IG structure that determines which
triplets are local ({prf:ref}`def-fractal-set-companion-kernel`,
{prf:ref}`def-fractal-set-cloning-score`).

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to tighten color coherence and increase nucleon masses.
- Adjust $\epsilon_c$, $\epsilon_d$, and $\lambda_{\text{alg}}$ to modify local companion structure
  and triplet availability; this changes baryon plateaus without altering the color definition.
- Implementation constraint: the nucleon channel requires $d=3$ and at least two neighbors; if the
  Channels tab reports `n/a`, verify that the run dimension is three and that neighbor sampling is
  adequate.

### Glueball channel (label G)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Viscous coupling | $\nu$ | `KineticOperator.nu` | Increase $\nu$ → stronger force fluctuations → heavier $m_G$. |
| Viscous range | $\rho$ | `KineticOperator.viscous_length_scale` | Decrease $\rho$ → shorter-range force → heavier $m_G$. |
| Friction | $\gamma$ | `KineticOperator.gamma` | Increase $\gamma$ → faster damping → heavier $m_G$. |

**Operator (primary, {prf:ref}`def-sm-direct-color-contractions`):**

$$
O_{G}(t)=\mathcal A_t\bigl(\operatorname{Re}\Pi_{ijk}\bigr),
\qquad \Pi_{ijk}=q_{ij}q_{jk}q_{ki},
$$

or $1-\operatorname{Re}\Pi_{ijk}$, which has the same connected correlator.
$\Pi_{ijk}=\operatorname{Tr}(P_iP_jP_k)$ is the three-vertex invariant of
{prf:ref}`prop-sm-direct-triangle-projectors`: it is invariant under
independent rephasings and common $U(3)$ frame changes, it is conjugated by
role exchange, so that $\operatorname{Re}\Pi_{ijk}$ is role-even
({prf:ref}`prop-sm-direct-role-swap`), and its factors are rank-one
projectors, not unitary comparison links. It is a different object from the plaquette of
{prf:ref}`def-fractal-set-plaquette` and the holonomy of
{prf:ref}`def-fractal-set-wilson-loop`.

**Alternative operator (force norm):**

$$
O_{G}^{F}(t)=\sum_i\left\|F^{(\text{visc})}(i,t)\right\|^2 .
$$

It contains no colour phase, carries the units of a squared force and scales
as $\nu^2$, and is unbounded, so its correlator requires a finite second
moment under the sampled law. Its correlator tracks how quickly the force
magnitude decorrelates under the viscous coupling
({prf:ref}`def-fractal-set-viscous-force`).

:::{div} feynman-prose
The glueball channel is the product of three overlaps around a closed
triangle: $i$ to $j$ to $k$ and back to $i$. Go around a loop and every
walker's arbitrary phase appears once with a bar and once without, so it
cancels. What survives is a phase that belongs to the *loop* and not to any
walker — which is precisely the structure that makes gauge-invariant
observables gauge-invariant.

Now, I want to head off an analogy before it does damage. It is tempting to
call $\Pi$ a Wilson loop, because a Wilson loop is also a product of things
around a closed path with the phases cancelling. The similarity is real and
it is where the intuition comes from. But the analogy breaks, and it breaks at
a place that matters: a Wilson loop multiplies *unitary* comparison links,
which is why the whole loop is unitary and why the plaquette has the
expansion in field strength that gives it its meaning. Our factors are
rank-one projectors. They are not unitary, the product is not a holonomy, and
$\Pi$ has modulus at most one for reasons of shrinkage, not of phase. It is a
perfectly good invariant. It is not the plaquette of
{prf:ref}`def-fractal-set-plaquette`, and the two must not be conflated in
either direction.

The force norm is kept as a named alternative because it is a real thing the
code computes, but notice how different an animal it is. No phase anywhere —
it cannot see colour at all. It carries units, so it is not dimensionless.
And it is unbounded, which means its correlator does not even exist unless
the force has a finite second moment under the sampled law. That is a
hypothesis, and it is one you should check rather than assume.
:::

**Sweep hypotheses to check:**
- Increase $\nu$ or decrease $\rho$ to strengthen gauge-field fluctuations and shorten the glueball
  correlator (heavier glueball mass).
- Use $\gamma$ only to fine-tune decay speed while preserving the mass-scale hierarchy.

### Alternative colour-gamma operators

:::{div} feynman-prose
Here is a place where notation has done real damage, so let us take it apart
slowly. Somewhere in the pipeline there are matrices called $\gamma_5$ and
$\gamma_\mu$, and they are sandwiched between two colour vectors in exactly
the way a Dirac bilinear sandwiches gamma matrices between two spinors. The
names, the placement, the shape of the formula — everything invites you to
read these as Dirac matrices.

They are not. Dirac matrices are $4\times4$ and act on a spinor index, and
their entire content is the Clifford relation
$\{\gamma^\mu,\gamma^\nu\}=2\eta^{\mu\nu}$. These are $d\times d$, they act on
the *colour* index, and they satisfy no Clifford relation at all — you can
check below that $\Gamma_0^2$ is not even invertible. There is a genuine
Dirac lift later in this chapter ({prf:ref}`def-qft-dirac-lift`); it is a
different construction and the two must never be mixed.

The notation is unfortunate, but the operators are real and the code computes
them, so we define them honestly and work out their symmetries. And the
symmetries hold a surprise: the thing named "pseudoscalar" in this family is
even under inversion. It is a second scalar channel. That is not a small
correction to a label; it is the opposite sign.
:::

:::{prf:definition} Colour-gamma operators
:label: def-qft-color-gamma-operators

For $d\ge3$ and colour components indexed by $a=0,\ldots,d-1$ define the
Hermitian $d\times d$ matrices

$$
\Gamma_5=\operatorname{diag}\bigl((-1)^{a}\bigr)_{a=0}^{d-1},\qquad
(\Gamma_\mu)_{ab}=i\,(\delta_{a\mu}\delta_{b\nu}-\delta_{a\nu}\delta_{b\mu}),
\quad\nu=\mu+1\bmod d,
$$

and the pair contractions

$$
g_{ij}=c_i^\dagger\Gamma_5c_j,\qquad
h_{ij}^{\mu}=c_i^\dagger\Gamma_\mu c_j
 =i\bigl(\overline{c_i^{\mu}}c_j^{\nu}-\overline{c_i^{\nu}}c_j^{\mu}\bigr).
$$

The colour-gamma channels are
$O_{\pi}^{\Gamma_5}=\mathcal A_t(\operatorname{Re}g)$,
$O_{\pi,-}^{\Gamma_5}=\mathcal A_t(\operatorname{Im}g)$,
$O_{\rho}^{\Gamma,\mu}=\mathcal A_t(\operatorname{Re}h^{\mu})$ and
$O_{a}^{\Gamma,\mu}=\mathcal A_t(\operatorname{Im}h^{\mu})$, the last two
with $d$ components contracted as in
{prf:ref}`prop-qft-component-contraction`. These matrices act on the colour
index. They are not Dirac matrices: $\Gamma_0^2=\operatorname{diag}(1,1,0,\ldots,0)\ne I$,
so they satisfy no Clifford relation. For $d=3$,
$h^{\mu}=i\,(\overline{c_i}\times c_j)_{\mu+2\bmod3}$.
:::

:::{prf:proposition} Symmetries of the colour-gamma operators
:label: prop-qft-color-gamma-parities

1. $g_{ji}=\overline{g_{ij}}$ and $h_{ji}^{\mu}=\overline{h_{ij}^{\mu}}$.
   Under the inversion of {prf:ref}`prop-sm-direct-parity`,
   $g\mapsto\overline g$ and $h^{\mu}\mapsto-\overline{h^{\mu}}$. Hence

   | Channel | $X$ | $P$ | On a mutual pairing |
   |---|---|---|---|
   | $\operatorname{Re}g$ | $+$ | $+$ | not constrained |
   | $\operatorname{Im}g$ | $-$ | $-$ | identically zero |
   | $\operatorname{Re}h^{\mu}$ | $+$ | $-$ | not constrained |
   | $\operatorname{Im}h^{\mu}$ | $-$ | $+$ | identically zero |

   In particular $O_{\pi}^{\Gamma_5}$ is even under inversion: in the sense of
   {prf:ref}`prop-sm-direct-parity` it is a second scalar channel,
   $g_{ij}=q_{ij}-2\sum_{a\ \mathrm{odd}}\overline{c_i^{a}}c_j^{a}$. The
   colour-gamma vector has the opposite exchange behaviour to the primary
   vector channel $\operatorname{Re}q\,r$.
2. $g$ is invariant under a common $A\in U(d)$ if and only if $A$ commutes
   with $\Gamma_5$, that is
   $A\in U(\lceil d/2\rceil)\times U(\lfloor d/2\rfloor)$. For $d=3$, under a
   common real rotation $R\in SO(3)$ of the colour components the vector
   $\overline{c_i}\times c_j$ rotates with $R$, so
   $\sum_\mu(\operatorname{Re}h^{\mu})^2$ and the contracted correlators are
   invariant; $h$ is not invariant under $SU(3)$.
3. For $d=2$ the same formula gives $\Gamma_1=-\Gamma_0$, so
   $\sum_\mu h^{\mu}=0$ and the component mean vanishes identically; this is
   why the definition requires $d\ge3$.
4. For $d=3$ the antisymmetric colour bilinear
   $\operatorname{Re}(\overline{c_i^{\mu}}c_j^{\nu})
    -\operatorname{Re}(\overline{c_i^{\nu}}c_j^{\mu})$, $\mu<\nu$, equals
   $\operatorname{Im}h^{0}$, $\operatorname{Im}h^{1}$ and
   $-\operatorname{Im}h^{2}$ for $(\mu\nu)=(01),(12),(02)$. It is the vector
   $\operatorname{Re}(\overline{c_i}\times c_j)$ up to a relabelling: three
   components, even under inversion, odd under exchange. It contains no
   symmetric traceless part and is not a spin-two object.
:::

:::{prf:proof}
**Item 1.** $\Gamma_5$ and $\Gamma_\mu$ are Hermitian, so
$c_j^\dagger\Gamma c_i=\overline{c_i^\dagger\Gamma c_j}$. Under
$c\mapsto-\overline c$ a contraction $c_i^\dagger Mc_j$ becomes
$c_i^{\mathsf T}M\overline{c_j}=\overline{c_i^\dagger\overline Mc_j}$.
$\Gamma_5$ is real, which gives $\overline g$; $\Gamma_\mu$ is purely
imaginary, $\overline{\Gamma_\mu}=-\Gamma_\mu$, which gives
$-\overline{h^{\mu}}$. The table follows, and its last column is
{prf:ref}`prop-exchange-odd-cancellation` applied as in
{prf:ref}`cor-sm-direct-exchange-parity`. The identity for $g$ is
$\Gamma_5=I-2\sum_{a\ \mathrm{odd}}e_ae_a^{\mathsf T}$.

**Item 2.** $(Ac_i)^\dagger\Gamma_5(Ac_j)=c_i^\dagger A^\dagger\Gamma_5Ac_j$
for all unit vectors forces $A^\dagger\Gamma_5A=\Gamma_5$, which for unitary
$A$ is $[A,\Gamma_5]=0$; the commutant of a diagonal matrix with two
eigenvalues is block unitary on its eigenspaces. For real $R\in SO(3)$,
$(R\overline{c_i})\times(Rc_j)=R(\overline{c_i}\times c_j)$. The matrix
$A=\operatorname{diag}(i,-i,1)\in SU(3)$ multiplies both
$\overline{c_i^{0}}c_j^{1}$ and $\overline{c_i^{1}}c_j^{0}$ by $-1$, so
$h^{0}\mapsto-h^{0}$ and $h$ is not invariant.

**Item 3.** For $d=2$, $\mu=1$ has $\nu=0$ and the displayed formula gives
$(\Gamma_1)_{10}=i=-(\Gamma_0)_{10}$.

**Item 4.** $\operatorname{Im}[i(z-w)]=\operatorname{Re}z-\operatorname{Re}w$
with $z=\overline{c_i^{\mu}}c_j^{\nu}$, $w=\overline{c_i^{\nu}}c_j^{\mu}$;
the sign for $(02)$ comes from the cyclic convention $\nu=\mu+1\bmod3$, which
orders that pair as $(2,0)$. An antisymmetric $3\times3$ array has three
independent components and is dual to a vector. $\square$
:::

:::{div} feynman-prose
Item 1 is worth dwelling on. Why is $\operatorname{Re}g$ inversion-*even*
when the matrix is called $\Gamma_5$? Because inversion here means
$c\mapsto-\overline c$ — complex conjugation with a sign — and conjugation
sends $c_i^\dagger Mc_j$ to the conjugate of $c_i^\dagger\overline Mc_j$. So
everything turns on whether the matrix is *real* or *imaginary*, not on
whether it anticommutes with something. $\Gamma_5$ is real. Its real part
therefore comes back unchanged. The $\Gamma_\mu$ are purely imaginary, and
their real parts flip.

That is the whole mechanism, and the identity in item 1 makes it concrete:
$g$ is just $q$ with the odd colour components subtracted twice over. It is a
reweighted overlap. A reweighted scalar is still a scalar.

Item 4 closes off a second tempting mislabel. The reference code records an
antisymmetric colour bilinear and calls it a tensor channel, with the
implication of spin two. But an antisymmetric $3\times3$ array has three
independent entries, and three entries dual to a vector are a vector. A
spin-two object would be the *symmetric traceless* part — five components —
and nothing here constructs it. The antisymmetric pieces are $\pm$ the three
$\operatorname{Im}h^{\mu}$, no more.
:::

:::{prf:remark} What a channel label asserts
:label: rem-qft-channel-labels

The symmetry content established for the channels of this chapter consists
of two signs: $X$, under exchange of the two walkers of a pair or of the two
companion roles of a triplet, and $P$, under the inversion of
{prf:ref}`prop-sm-direct-parity`, valid under the equivariance hypotheses
stated there. A total spin $J$ is not defined, because the colour encoding is
not covariant under rotations ({prf:ref}`thm-sm-su3-emergence`) and the lift
of {prf:ref}`def-qft-dirac-lift` is not equivariant. A charge-conjugation
sign $C$ is not defined, because no charge conjugation acts on the record.
The labels $\sigma$, $\pi$, $\rho$, $N$, $G$ name measurement channels, as in
{prf:ref}`def-sm-direct-color-contractions`.

For the Dirac-lift bilinears the continuum quantum numbers of a fermion
bilinear $\bar q\Gamma q$ are quoted as analogues only:

| $\Gamma$ | Continuum analogue | $P$ of the lifted bilinear |
|---|---|---|
| $I$ | $0^{++}$ | $+$ |
| $\gamma^5$ | $0^{-+}$ | $-$ |
| $\gamma^{k}$ | $1^{--}$ | $-$ |
| $\gamma^5\gamma^{k}$ | $1^{++}$ | $+$ |
| $\sigma^{jk}$ | $1^{+-}$ | $+$ |
| $\sigma^{0k}$ | $1^{--}$ | $-$ |

An antisymmetric $\sigma^{\mu\nu}$ has $6=3+3$ components, two spin-one
multiplets; it contains no spin-two part.
:::

:::{div} feynman-prose
Ask yourself what a label like $0^{-+}$ actually claims. It claims three
things: a total spin $J$, a parity $P$, and a charge-conjugation eigenvalue
$C$. Now ask which of the three we have earned here.

Parity, yes — there is an honest inversion on the record and the channels have
definite signs under it. Exchange, yes, and we track it as $X$. Spin? Spin
requires an action of the rotation group under which the observable
transforms in a definite representation, and the colour encoding does not have
one; {prf:ref}`thm-sm-su3-emergence` is explicit about that. Charge
conjugation? There is no charge conjugation acting on the record at all.
Nothing to take an eigenvalue of.

So two of the three superscripts in $J^{PC}$ are simply not defined for our
channels, which is why the headings above carry plain labels. The names
$\sigma$, $\pi$, $\rho$, $N$, $G$ are not claims about particles; they are
names for measurements, kept because the measurements were built by analogy
with those particles' operators. The table of continuum analogues is offered
in exactly that spirit — this is what the corresponding bilinear would be in
a relativistic field theory — and it is quoted, not derived.
:::

### Empirical calibration status (zero-reward baseline)

The baseline QFT calibration runs in `QFT_CALIBRATION_REPORT.txt` (zero reward, viscosity-only,
200 walkers, 300 steps, Channels-tab analysis) report ratios of fitted decay rates from the
Channels-tab pipeline, with its operators, component treatment, frame normalization and fit
settings, against the reference ratios of {prf:ref}`def-qft-reference-ratios`. They are
selection-stage measurements in the sense of
{prf:ref}`rem-qft-reference-ratio-selection`:

- **Closest $R_{\rho\pi}$**: $\;R_{\rho\pi}\approx 5.437$ (thr=0.9, pen=1.1, $\beta=0.5$), but
  $R_{N\pi}\approx 0.592$ (nucleon suppressed).
- **Closest $R_{N\pi}$**: $\;R_{N\pi}\approx 6.171$ (weak\_potential\_fit1\_aniso\_stable2), but
  $R_{\rho\pi}\approx 3.186$ (rho too light).
- **Nucleon\_abs2** can raise $R_{N\pi}$ (≈7.55) but collapses $\pi$ and explodes $R_{\rho\pi}$.
- **Threshold sensitivity**: high neighbor thresholds (≈0.9) are the only tested lever that moves
  $R_{\rho\pi}$ near target, but they suppress $R_{N\pi}$ in the baseline.
- **Numerical stability**: curl + anisotropic diffusion runs are currently unstable (NaN noise at
  step 1), so those results are not admissible for calibration.

**Empirical conclusion.** Within the current viscosity-only baseline and neighbor-threshold/penalty
parameter space, no configuration achieves both ratios within ±2% of the reference ratios. High
companion thresholds move $R_{\rho\pi}$ toward target but suppress $R_{N\pi}$; stable anisotropic
settings recover $R_{N\pi}$ but leave $R_{\rho\pi}$ low. These findings are measurement-based and do
not override the theoretical ratio-sieve constraints below; they instead flag where the current
baseline does not realize the reference ratios.

(sec-qft-calibration-electroweak)=
## Electroweak dashboard calibration

:::{div} feynman-prose
The dashboard combines three observable families: labels derived from recorded
cloning roles, projected Dirac-spinor bilinears, and legacy scalar-phase or
doublet proxies. They share correlator utilities but have different algebra
and masks. The two realization propositions below specify the recorded frames,
normalizations, and scalar factors actually used by the implementation.

Start with those definitions before interpreting a fitted mass. In particular,
a channel that is identically zero has no exponential amplitude to fit, and
a scalar phase applied to a bilinear does not acquire matrix-valued gauge
transport merely through its channel name.
:::

Implementation note: the active electroweak correlator path is
`src/fragile/physics/electroweak/electroweak_channels.py`, with chirality classification in
`src/fragile/physics/electroweak/chirality.py`, projector-based spinor operators in
`src/fragile/physics/electroweak/electroweak_spinors.py`, the channel-selection UI in
`src/fragile/physics/app/electroweak_correlators.py`, the mass-extraction tab in
`src/fragile/physics/app/electroweak_mass_tab.py`, and the top-level tab wiring in
`src/fragile/physics/app/dashboard.py`.

### Walker-role chirality observables

The baseline electroweak matter observables are defined from the recorded clone events. At each
frame, alive walkers are partitioned into

$$
\Delta_t,\qquad \mathrm{SR}_t,\qquad \mathrm{WR}_t,\qquad \mathrm{P}_t,
$$

exactly as in {prf:ref}`def-sm-walker-role-partition`, with left- and right-handed sectors

$$
L_t = \Delta_t \cup \mathrm{SR}_t,
\qquad
R_t = \mathrm{WR}_t \cup \mathrm{P}_t.
$$

The chirality label is

$$
\chi_i(t)=
\begin{cases}
+1,& i\in L_t,\\
-1,& i\in R_t,\\
0,& i\notin A_t.
\end{cases}
$$

Writing $N$ for the recorded walker count per frame, the dashboard-computed chirality channels are
then

$$
\chi_{\mathrm{mean}}(t)=\frac{1}{N}\sum_{i=1}^{N}\chi_i(t),
\qquad
f_L(t)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}_{\{i\in L_t\}},
$$

$$
f_{\Delta\to R}(t)=\frac{1}{|\Delta_t|}
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}},
$$

and the complex left-right transfer observable

$$
M_{LR}(t)=
\frac{1}{N_{\Delta\to R}(t)}
\sum_{\substack{i\in\Delta_t\\c_c(i,t)\in R_t}}
\exp\!\left(i\frac{F_{c_c(i,t)}(t)-F_i(t)}{\hbar_{\mathrm{eff}}}\right).
$$

Dead walkers contribute $0$ to $\chi_i$, so the averages above are taken over the full recorded
walker count exactly as in the implementation. The conventions are
$f_{\Delta\to R}(t)=0$ when $|\Delta_t|=0$ and $M_{LR}(t)=0$ when
$N_{\Delta\to R}(t)=0$.

Operationally, the electroweak correlator tab exposes these as `chi_mean`, `left_fraction`,
`lr_fraction`, and `lr_coupling_mag`. 

:::{div} feynman-prose
Under the same-frame role partition, every alive target of a cloning walker
belongs to the left set. Thus the intersection defining the right-target
transfer is empty: `lr_fraction` and `lr_coupling_mag` are exactly zero under
the proposition's conventions. Their zero correlators contain no mass signal.
The nonzero role observables remain diagnostics of the recorded population;
identifying them with a physical chiral interaction requires additional model
structure.
:::



:::{prf:proposition} Current Chirality-Channel Realization
:label: prop-qft-ew-chirality-realization

**Rigor Class:** F (Implementation-Exact)

Let

$$
t \in \{t_{\mathrm{start}},\dots,t_{\mathrm{end}}-1\},
\qquad
t_{\mathrm{start}}=\max(1,\lfloor n_{\mathrm{recorded}}\,f_{\mathrm{warm}}\rfloor),
\qquad
t_{\mathrm{end}}=\max(t_{\mathrm{start}}+1,\lfloor n_{\mathrm{recorded}}\,f_{\mathrm{end}}\rfloor).
$$

For each such frame, let $\chi_i(t)$, $L_t$, $R_t$, and $\Delta_t$ be the walker-role chirality
objects of {prf:ref}`def-sm-walker-chirality`, computed from the recorded slices
`will_clone[t-1]`, `companions_clone[t-1]`, `fitness[t-1]`, and `alive_mask[t-1]`. Then the
implemented chirality channels in `src/fragile/physics/electroweak/electroweak_channels.py` are
exactly

$$
\mathrm{chi\_mean}(t)=\frac{1}{N}\sum_{i=1}^{N}\chi_i(t),
\qquad
\mathrm{left\_fraction}(t)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}_{\{i\in L_t\}},
$$

$$
\mathrm{lr\_fraction}(t)=
\frac{1}{\max(|\Delta_t|,1)}
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}},
$$

$$
\mathrm{lr\_coupling\_mag}(t)=
\left|
\frac{1}{\max(N_{\Delta\to R}(t),1)}
\sum_{\substack{i\in\Delta_t\\c_c(i,t)\in R_t}}
\exp\!\left(i\frac{F_{c_c(i,t)}(t)-F_i(t)}{h_{\mathrm{eff}}}\right)
\right|,
$$

where

$$
N_{\Delta\to R}(t):=
\sum_{i\in\Delta_t}\mathbf{1}_{\{c_c(i,t)\in R_t\}}.
$$

The same-frame partition also gives the exact identities

$$
N_{\Delta\to R}(t)=0,\qquad
\mathrm{lr\_fraction}(t)=\mathrm{lr\_coupling\_mag}(t)=0,\qquad
\mathrm{chi\_mean}(t)=2\,\mathrm{left\_fraction}(t)-|A_t|/N.
$$

These follow for the recorded companion-role definitions of
{prf:ref}`prop-sm-walker-role-partition`; selecting cloning frames preserves
them.

If `cloning_frames_only=True`, these four series are further restricted to the subfamily of frames
with at least one cloning event.
:::

:::{prf:proof}
By {prf:ref}`prop-sm-walker-role-partition`, a clone companion of a
walker in $\Delta_t$ lies in the left role set, so the cross mask is empty.
The conventions in the displayed denominators then give both zero channels.
Since $A_t=L_t\sqcup R_t$ and dead walkers have chirality zero,
$N^{-1}(|L_t|-|R_t|)=2|L_t|/N-|A_t|/N$.

In `_compute_chirality_series`, the recorded tensors are sliced on
`[t_start-1:t_end-1]` and passed to `classify_walkers_vectorized`. By construction of that helper,
`classification.chi` is the tensor $\chi_i(t)$ with dead walkers assigned the value $0$, and
`classification.left_handed` is the indicator of $L_t$. The assignments

$$
\texttt{series["chi_mean"]} = \texttt{classification.chi.mean(dim=1)},
\qquad
\texttt{series["left_fraction"]} =
\texttt{classification.left_handed.float().mean(dim=1)}
$$

therefore produce the two averages above over the full recorded walker count $N$.

Next, the code forms `comp_idx = companions_clone.clamp(0,N-1)`,
`comp_is_right = gather(classification.right_handed, comp_idx)`, and
`cross_mask = classification.delta & comp_is_right`. Hence `cross_mask[t,i]` is true exactly when
$i\in\Delta_t$ and $c_c(i,t)\in R_t$. The lines

$$
\texttt{delta_count = delta_mask.float().sum(dim=1).clamp(min=1)},
\qquad
\texttt{cross_count = cross_mask.float().sum(dim=1)}
$$

give $\max(|\Delta_t|,1)$ and $N_{\Delta\to R}(t)$ respectively, so
`series["lr_fraction"] = cross_count / delta_count` is exactly the stated formula with the
zero-delta convention built in.

For the phase-transfer channel, the code computes
`phase = (comp_fitness - fitness) / h_eff`,
`phase_exp = exp(1j * phase)`, and

$$
\texttt{lr_complex}
=
\frac{
\sum_i e^{i(F_{c_c(i,t)}-F_i(t))/h_{\mathrm{eff}}}\,\mathbf{1}_{\{i\in\Delta_t,\,
c_c(i,t)\in R_t\}}
}{
\max(N_{\Delta\to R}(t),1)
}.
$$

Taking `abs()` yields the displayed $\mathrm{lr\_coupling\_mag}(t)$. Finally, if
`cloning_frames_only=True`, the code restricts all four series to
`frame_has_cloning = will_clone.any(dim=1)`, which is exactly the stated frame filter. $\square$
:::

### Dirac-spinor electroweak operator layer

The second electroweak layer maps recorded color states to four-component
vectors $\psi_i \in \mathbb{C}^4$ using the lift of
{prf:ref}`def-qft-dirac-lift`.

:::{div} feynman-prose
We are about to take a three-component colour vector and make a
four-component Dirac spinor out of it. Before we do, let us be clear-eyed
about what such a map can and cannot be.

A real three-vector has three numbers; a Weyl spinor has two complex numbers,
so four real ones. You might hope for a map that *respects rotations* — rotate
the vector and the spinor rotates with it by the spin-$\tfrac12$
representation. That is the map you would want, and item 8 below proves it
does not exist. Not "is hard to construct": does not exist, and the argument
is two lines. So whatever we build will be a *coordinate* construction — a
definite recipe in a definite basis — and it will be covariant only about one
distinguished axis.

That is not a reason to refuse to build it. The code builds it, it produces
series, and those series have honest symmetry properties worth knowing. It is
a reason to write the recipe down explicitly, and to keep the word "spinor"
from smuggling in covariance that was never there.
:::

:::{prf:definition} Dirac lift of a colour state
:label: def-qft-dirac-lift

Let $d=3$ and fix the threshold $\delta_c$ of
{prf:ref}`def-sm-direct-observable-law`. For $w\in\mathbb R^3\setminus\{0\}$
put

$$
E(w)=\frac{1}{\sqrt{\|w\|}}\begin{pmatrix}w_1+iw_2\\ w_3\end{pmatrix}
\in\mathbb C^2,
$$

and for a valid colour $c$ with $\|\operatorname{Re}c\|>\delta_c$ and
$\|\operatorname{Im}c\|>\delta_c$ define

$$
\psi(c)=\begin{pmatrix}E(\operatorname{Im}c)\\E(\operatorname{Re}c)\end{pmatrix}
\in\mathbb C^4 .
$$

Colours failing either inequality have no lift and are masked. The numerical
Clifford matrices are the declared $\widehat\gamma^\mu$ of signature
$(+,-,-,-)$ in the Dirac representation
({prf:ref}`thm-sm-ew-operator-layers`),

$$
\widehat\gamma^0=\begin{pmatrix}I&0\\0&-I\end{pmatrix},\quad
\widehat\gamma^{k}=\begin{pmatrix}0&\sigma_k\\-\sigma_k&0\end{pmatrix},\quad
\gamma^5=i\widehat\gamma^0\widehat\gamma^1\widehat\gamma^2\widehat\gamma^3
        =\begin{pmatrix}0&I\\I&0\end{pmatrix},\quad
\sigma^{\mu\nu}=\tfrac i2[\widehat\gamma^\mu,\widehat\gamma^\nu],
$$

and $\bar\psi=\psi^\dagger\widehat\gamma^0$. The **Dirac-lift bilinears** of
a pair are $D_{ij}^{\Gamma}=\bar\psi_i\Gamma\psi_j$; the recorded parts are
$D^{S}=\operatorname{Re}D^{I}$, $D^{P}=\operatorname{Im}D^{\gamma^5}$,
$D^{V,k}=\operatorname{Re}D^{\gamma^k}$,
$D^{A,k}=\operatorname{Re}D^{\gamma^5\gamma^k}$,
$D^{T,jk}=\operatorname{Re}D^{\sigma^{jk}}$ and
$D^{T,0k}=\operatorname{Re}D^{\sigma^{0k}}$, with three-component families
contracted as in {prf:ref}`prop-qft-component-contraction`. These are
alternative operators; the primary channels are those of
{prf:ref}`def-sm-direct-color-contractions`.
:::

:::{prf:proposition} Properties of the Dirac lift
:label: prop-qft-dirac-lift-properties

Write $u=\operatorname{Im}c$, $w=\operatorname{Re}c$, $a=E(u)$, $b=E(w)$.

1. $E(w)^\dagger E(w)=\|w\|$ and $E(-w)=-E(w)$. For a unit colour
   $\psi^\dagger\psi=\|u\|+\|w\|\in[1,\sqrt2]$; the lift does not preserve
   norms.
2. Under the inversion $c\mapsto-\overline c$ of
   {prf:ref}`prop-sm-direct-parity`, $\psi(-\overline c)=\widehat\gamma^0\psi(c)$,
   hence $D_{ij}^{\Gamma}\mapsto D_{ij}^{\widehat\gamma^0\Gamma\widehat\gamma^0}$.
   The signs $P$ in {prf:ref}`rem-qft-channel-labels` are those of
   $\widehat\gamma^0\Gamma\widehat\gamma^0=\pm\Gamma$.
3. If $\widehat\gamma^0\Gamma$ is Hermitian then
   $D_{ji}^{\Gamma}=\overline{D_{ij}^{\Gamma}}$: the real part is even and the
   imaginary part odd under pair exchange. This is the case for
   $\Gamma\in\{I,\gamma^k,\gamma^5\gamma^k,\sigma^{\mu\nu}\}$. For
   $\Gamma=\gamma^5$ the matrix $\widehat\gamma^0\gamma^5$ is anti-Hermitian,
   $D_{ji}=-\overline{D_{ij}}$, and the imaginary part is the even one. Every
   recorded part listed in {prf:ref}`def-qft-dirac-lift` is exchange-even.
4. $\widehat\gamma^0P_{L,R}=\tfrac12(\widehat\gamma^0\mp\widehat\gamma^0\gamma^5)$
   is neither Hermitian nor anti-Hermitian:
   $\operatorname{Re}(\bar\psi_iP_{L,R}\psi_j)
    =\tfrac12D^{S}_{ij}\mp\tfrac12\operatorname{Re}D^{\gamma^5}_{ij}$, and the
   second term is exchange-odd. On a mutual pairing the frame averages of the
   left and the right scalar bilinear both equal $\tfrac12\mathcal A_t(D^{S})$.
   The projected currents $\widehat\gamma^0\gamma^kP_{L,R}$ are Hermitian and
   their real parts are exchange-even.
5. The upper and lower component pairs of $\psi$ are the eigenspaces of
   $\widehat\gamma^0$ with eigenvalues $+1$ and $-1$, the inversion-even and
   inversion-odd components of item 2. They are not chirality eigenspaces:
   $P_L(\xi,0)^{\mathsf T}=\tfrac12(\xi,-\xi)^{\mathsf T}$. The chiral
   components of $\psi=(a,b)^{\mathsf T}$ are
   $P_{L}\psi=\tfrac12(a-b,\,b-a)^{\mathsf T}$ and
   $P_{R}\psi=\tfrac12(a+b,\,a+b)^{\mathsf T}$.
6. $D^{P}_{ij}=-D^{T,03}_{ij}$ identically. The recorded pseudoscalar part is
   one component of the family $D^{T,0k}$ and is not an independent channel.
7. The bilinears are not invariant under the common phase
   $c\mapsto e^{i\alpha}c$, hence not under $U(3)$ or $SU(3)$ frame changes:
   $\alpha=\pi/2$ maps $D^{S}\mapsto-D^{S}$.
8. There is no nonzero map $E:\mathbb R^3\to\mathbb C^2$ with
   $E(Rw)=\pm U(R)E(w)$ for the spin-$\tfrac12$ representation $U$. The lift
   above is covariant only under rotations about the third colour axis,
   $E(R_z(\phi)w)=\operatorname{diag}(e^{i\phi},1)E(w)$.
:::

:::{prf:proof}
**Item 1.** $|w_1+iw_2|^2+w_3^2=\|w\|^2$, divided by $\|w\|$; oddness is
immediate. For a unit colour $\|u\|^2+\|w\|^2=1$ with both norms
nonnegative, so their sum lies between $1$ and $\sqrt2$.

**Item 2.** $-\overline c$ has imaginary part $u$ and real part $-w$, so
$\psi(-\overline c)=(E(u),E(-w))=(a,-b)=\widehat\gamma^0\psi(c)$. Then
$\bar\psi_i'\Gamma\psi_j'
 =\psi_i^\dagger\widehat\gamma^0\widehat\gamma^0\Gamma\widehat\gamma^0\psi_j
 =\bar\psi_i(\widehat\gamma^0\Gamma\widehat\gamma^0)\psi_j$.

**Item 3.** For $M=\widehat\gamma^0\Gamma$,
$\psi_j^\dagger M\psi_i=\overline{\psi_i^\dagger M^\dagger\psi_j}$. With
$(\widehat\gamma^0)^\dagger=\widehat\gamma^0$,
$(\widehat\gamma^k)^\dagger=-\widehat\gamma^k$ and
$(\gamma^5)^\dagger=\gamma^5$, anticommutation gives $M^\dagger=M$ for the
listed $\Gamma$ and
$(\widehat\gamma^0\gamma^5)^\dagger=\gamma^5\widehat\gamma^0
 =-\widehat\gamma^0\gamma^5$.

**Item 4.** Linearity and item 3; the cancellation is
{prf:ref}`prop-exchange-odd-cancellation`. For the currents,
$(\widehat\gamma^0\widehat\gamma^k\gamma^5)^\dagger
 =\gamma^5(-\widehat\gamma^k)\widehat\gamma^0
 =\widehat\gamma^0\widehat\gamma^k\gamma^5$ after three anticommutations.

**Item 5.** $\widehat\gamma^0=\operatorname{diag}(I,-I)$ and
$P_{L,R}=\tfrac12\bigl(\begin{smallmatrix}I&\mp I\\\mp I&I\end{smallmatrix}\bigr)$.

**Item 6.** $\widehat\gamma^0\gamma^5=\bigl(\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}\bigr)$
gives $D^{\gamma^5}_{ij}=a_i^\dagger b_j-b_i^\dagger a_j$, and
$\widehat\gamma^0\sigma^{03}=i\bigl(\begin{smallmatrix}0&\sigma_3\\-\sigma_3&0\end{smallmatrix}\bigr)$
gives $D^{\sigma^{03}}_{ij}=i(a_i^\dagger\sigma_3b_j-b_i^\dagger\sigma_3a_j)$.
The second components of $a$ and $b$ are real, so
$a^\dagger b$ and $a^\dagger\sigma_3b$ differ by a real number and have equal
imaginary parts. Hence
$\operatorname{Re}D^{\sigma^{03}}=-\operatorname{Im}(a_i^\dagger\sigma_3b_j-b_i^\dagger\sigma_3a_j)
 =-\operatorname{Im}D^{\gamma^5}$.

**Item 7.** $ic$ has imaginary part $w$ and real part $-u$, so
$\psi(ic)=(E(w),-E(u))$ and
$D^{I}=a_i^\dagger a_j-b_i^\dagger b_j$ becomes
$b_i^\dagger b_j-a_i^\dagger a_j$.

**Item 8.** Rotations about $\hat w$ fix $w$, so $E(w)$ would be an
eigenvector of $\exp(-i\phi\,\hat w\cdot\sigma/2)$ with eigenvalue $\pm1$
for every $\phi$; its eigenvalues are $e^{\mp i\phi/2}$. The covariance
under $R_z(\phi)$ follows from
$(w_1+iw_2)\mapsto e^{i\phi}(w_1+iw_2)$ with $w_3$ and $\|w\|$ fixed.
$\square$
:::

:::{div} feynman-prose
Item 5 is the one that will save you from a real mistake, so let me spell it
out. The spinor is built as (upper) $=E(\operatorname{Im}c)$, (lower)
$=E(\operatorname{Re}c)$, and it is almost irresistible to say "upper is
left-handed, lower is right-handed". In the Dirac representation used here,
that is false. The upper and lower pairs are the eigenspaces of
$\widehat\gamma^0$ — they are the *parity* blocks, which is exactly why item 2
comes out so cleanly. Chirality is the eigenbasis of $\gamma^5$, and in this
representation $\gamma^5$ is off-diagonal, so a chirality eigenvector mixes
upper and lower in equal measure: $P_L\psi=\tfrac12(a-b,\,b-a)$. Take a
purely-upper spinor and project it left and you get half of it, spread across
both blocks. The two decompositions are as different as two orthogonal
bases can be.

Item 6 is a different kind of surprise: the recorded pseudoscalar part is not
an independent measurement at all. $D^{P}=-D^{T,03}$, identically, walker by
walker and frame by frame. If you fit both and report two rates, you have
reported one rate twice with a sign flip, and if you count them as two
agreeing channels you have double-counted your evidence.

And item 7 should temper any talk of gauge invariance in this layer. Multiply
every colour by a common phase — the most harmless $U(1)$ transformation you
can imagine — and the scalar bilinear can flip sign outright at
$\alpha=\pi/2$. The reason is structural: the lift reads
$\operatorname{Re}c$ and $\operatorname{Im}c$ separately, and a common phase
rotates them into each other. Whatever these channels measure, it is not
invariant under the colour frame.
:::

The matrix bilinears of this layer use the
chiral projectors from {prf:ref}`def-lqft-chiral-projectors`. The implementation
constructs the following measurement channels:

$$
J_L^\mu = \bar\psi\gamma^\mu P_L\psi,
\qquad
J_R^\mu = \bar\psi\gamma^\mu P_R\psi,
\qquad
J_V^\mu = \bar\psi\gamma^\mu\psi,
$$

$$
O_L=\bar\psi P_L\psi,
\qquad
O_R=\bar\psi P_R\psi,
\qquad
O_{LR}=\bar\psi P_L\psi\ \text{on }L\!\to\!R\text{ pairs}.
$$

Multiplication by the implemented scalar phases gives channels carrying the following historical labels:

$$
J_{U(1)}^\mu,\qquad J_{L,U(1)}^\mu,\qquad J_{L,SU(2)}^\mu,\qquad J_{R,SU(2)}^\mu.
$$

In code, these are recorded as real bilinears,
$\operatorname{Re}(\bar\psi_i\Gamma P\psi_j)$ and
$\operatorname{Re}(U_{ij}\bar\psi_i\Gamma P\psi_j)$, before time correlators are constructed.

In code, these appear as
`j_vector_L`,
`j_vector_R`,
`j_vector_V`,
`o_scalar_L`,
`o_scalar_R`,
`j_vector_walkerL`,
`j_vector_walkerR`,
`j_vector_L_walkerL`,
`j_vector_R_walkerR`,
`o_yukawa_LR`,
`o_yukawa_RL`,
`j_vector_u1`,
`j_vector_L_u1`,
`j_vector_L_su2`,
`j_vector_R_su2`,
`parity_violation_dirac`,
and `parity_violation_walker`.

The implementation interprets these channels as follows:

- `j_vector_L_su2`: left-current bilinear multiplied by the scalar returned by `compute_su2_gauge_link`, used as a W-like proxy.
- `j_vector_u1`: vector bilinear multiplied by the fitness-difference phase, used as a photon-like proxy.
- `j_vector_L_u1`: left-current bilinear multiplied by that phase, used as a neutral-current proxy.
- `o_yukawa_LR`: cross-chirality scalar bilinear, used as the Dirac/Yukawa mass proxy.
- `parity_violation_dirac` and `parity_violation_walker`: asymmetry diagnostics comparing left and
  right sectors at the projector and walker-role levels.

:::{div} feynman-prose
The routine named `compute_su2_gauge_link` returns one complex number of unit
modulus. Its phase uses an absolute fitness difference, so reversing the edge
leaves it unchanged. The determinant and orientation calculations in the next
proposition explain why it is a scalar modulation rather than an implemented
$SU(2)$ connection. The bilinear and parity diagnostics can still be computed
and compared under their stated definitions.
:::



:::{prf:proposition} Current Dirac-Spinor Realization
:label: prop-qft-ew-spinor-realization

**Rigor Class:** F (Implementation-Exact)

Assume $d=3$ so that the color states admit the lift
$c_i(t)\mapsto \psi_i(t)\in\mathbb{C}^4$ of {prf:ref}`def-qft-dirac-lift`.
For each retained frame $t$ and walker index $i$, let

$$
j=c_d(i,t)
$$

be the recorded distance companion, let $\chi_i(t)\in\{+1,-1,0\}$ be the walker-role chirality
computed from the clone companion data, and define the validity mask

$$
V_t(i):=
\mathbf{1}_{\{\mathrm{spinor\_valid}_i(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{spinor\_valid}_j(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{alive}_i(t)\}}
\cdot
\mathbf{1}_{\{\mathrm{alive}_j(t)\}}
\cdot
\mathbf{1}_{\{j\neq i\}},
$$

where $\mathrm{spinor\_valid}$ is colour validity together with the two
inequalities of {prf:ref}`def-qft-dirac-lift`.

Let the pair classes be

$$
LL_t=\{i:V_t(i)=1,\ \chi_i(t)>0,\ \chi_j(t)>0\},
\qquad
RR_t=\{i:V_t(i)=1,\ \chi_i(t)<0,\ \chi_j(t)<0\},
$$

$$
LR_t=\{i:V_t(i)=1,\ \chi_i(t)>0,\ \chi_j(t)<0\},
\qquad
RL_t=\{i:V_t(i)=1,\ \chi_i(t)<0,\ \chi_j(t)>0\}.
$$

With unit edge weights, define for any mask $M_t\subseteq\{1,\dots,N\}$ and any pair observable
$B_t(i)$

$$
\operatorname{Avg}_{M_t}[B]
:=
\frac{
\sum_{i=1}^{N}\mathbf{1}_{\{i\in M_t\}}\,B_t(i)
}{
\max(|M_t|,10^{-12})
}.
$$

Further define the real bilinears

$$
B_{\Gamma,P}(i,t):=
\operatorname{Re}\!\bigl(\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
$$

$$
B_{\Gamma,P}^{U(1)}(i,t):=
\operatorname{Re}\!\bigl(U_{ij}^{(1)}(t)\,\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
\qquad
U_{ij}^{(1)}(t)=
\exp\!\left(i\frac{F_j(t)-F_i(t)}{h_{\mathrm{eff}}}\right),
$$

$$
B_{\Gamma,P}^{SU(2)}(i,t):=
\operatorname{Re}\!\bigl(U_{ij}^{(2)}(t)\,\psi_i(t)^\dagger\gamma^0\Gamma P\,\psi_j(t)\bigr),
\qquad
U_{ij}^{(2)}(t)=
\exp\!\left(
i\,
\frac{|F_j(t)-F_i(t)|}{|F_j(t)-F_i(t)|+\epsilon_{\mathrm{clone}}}
\cdot
\frac{\pi}{2h_{\mathrm{eff}}}
\right).
$$

Here $U_{ij}^{(2)}$ is the scalar phase returned by
`compute_su2_gauge_link`, as distinguished in
{prf:ref}`thm-sm-ew-operator-layers`. Its absolute fitness difference makes
$U_{ji}^{(2)}=U_{ij}^{(2)}$, whereas inverse-oriented transport would require
$U_{ji}^{(2)}=(U_{ij}^{(2)})^{-1}$. Multiplication by this scalar is the
implemented bilinear modulation; it is not a matrix-valued $SU(2)$ link.
Indeed, representing it as $U_{ij}^{(2)}I_2$ gives determinant
$(U_{ij}^{(2)})^2$, which is generally not one.

Then the current Dirac-spinor pipeline computes exactly the operator series

$$
j_{\mathrm{vector},L}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}\right],
\qquad
j_{\mathrm{vector},R}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}\right],
$$

$$
j_{\mathrm{vector},V}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
\qquad
o_{\mathrm{scalar},L}(t)=\operatorname{Avg}_{V_t}[B_{I,P_L}],
\qquad
o_{\mathrm{scalar},R}(t)=\operatorname{Avg}_{V_t}[B_{I,P_R}],
$$

$$
j_{\mathrm{vector},\mathrm{walkerL}}(t)=
\operatorname{Avg}_{LL_t\cup LR_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
\qquad
j_{\mathrm{vector},\mathrm{walkerR}}(t)=
\operatorname{Avg}_{RR_t\cup RL_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}\right],
$$

$$
j_{\mathrm{vector},L,\mathrm{walkerL}}(t)=
\operatorname{Avg}_{LL_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}\right],
\qquad
j_{\mathrm{vector},R,\mathrm{walkerR}}(t)=
\operatorname{Avg}_{RR_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}\right],
$$

$$
o_{\mathrm{yukawa},LR}(t)=\operatorname{Avg}_{LR_t}[B_{I,P_L}],
\qquad
o_{\mathrm{yukawa},RL}(t)=\operatorname{Avg}_{RL_t}[B_{I,P_R}],
$$

$$
j_{\mathrm{vector},U(1)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,I}^{U(1)}\right],
\qquad
j_{\mathrm{vector},L,U(1)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}^{U(1)}\right],
$$

$$
j_{\mathrm{vector},L,SU(2)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_L}^{SU(2)}\right],
\qquad
j_{\mathrm{vector},R,SU(2)}(t)=
\operatorname{Avg}_{V_t}\!\left[\frac{1}{3}\sum_{k=1}^{3}B_{\gamma^k,P_R}^{SU(2)}\right],
$$

and the parity diagnostics

$$
\mathrm{pv}_{\mathrm{dirac}}(t)=
\frac{j_{\mathrm{vector},L}(t)^2-j_{\mathrm{vector},R}(t)^2}
{j_{\mathrm{vector},L}(t)^2+j_{\mathrm{vector},R}(t)^2+\varepsilon_{\mathrm{pv}}},
$$

$$
\mathrm{pv}_{\mathrm{walker}}(t)=
\frac{j_{\mathrm{vector},\mathrm{walkerL}}(t)^2-j_{\mathrm{vector},\mathrm{walkerR}}(t)^2}
{j_{\mathrm{vector},\mathrm{walkerL}}(t)^2+j_{\mathrm{vector},\mathrm{walkerR}}(t)^2+\varepsilon_{\mathrm{pv}}},
\qquad
\varepsilon_{\mathrm{pv}}=10^{-30}.
$$

The same routine also records the pair-count diagnostics

$$
n_{\mathrm{valid}}(t)=|V_t|,
\qquad
n_{LL}(t)=|LL_t|,
\qquad
n_{RR}(t)=|RR_t|,
\qquad
n_{LR}(t)=|LR_t|.
$$
:::

:::{prf:proof}
The helper `_compute_dirac_spinor_channels` first resolves the retained frame interval, computes
color states on that interval, reads the clone companions for walker classification, and reads the
distance companions for the spinor pairing. It then sets
`sample_indices = [0,\dots,N-1]` and `neighbor_indices = companions_distance.unsqueeze(-1)`, so
the pair for walker $i$ is exactly $(i,c_d(i,t))$.

Inside `compute_electroweak_spinor_operators`, the validity mask is
`valid = v_i & v_j & (first_nb != sample_indices)`, with `v_i` and `v_j` requiring spinor
validity, which includes color validity, and `alive`. This is precisely $V_t(i)$. The chirality masks `both_L`, `both_R`,
`cross_LR`, and `cross_RL` are exactly the four sets $LL_t$, $RR_t$, $LR_t$, and $RL_t$ above.

The helper `_compute_chiral_bilinear` builds the matrix
$M=\gamma^0\Gamma$ and, when present, right-multiplies by the chiral projector $P_L$ or $P_R$.
It evaluates $\psi_i^\dagger M\psi_j$, multiplies by the requested gauge link if present, and
returns `bilinear.real.float()`. Hence every recorded spinor operator is the real part of the
corresponding complex bilinear.

The helper `_vector_current` sums the three spatial gamma-matrix bilinears and divides by $3$;
`_scalar_op` uses $\Gamma=I$. Both helpers average over the requested mask through `_avg`, whose
denominator is the masked weight sum clamped below by $10^{-12}$. In the active dashboard
`sample_edge_weights` is not supplied, so all weights are $1$ and `_avg` becomes the stated masked
arithmetic mean. The named assignments in the function body are exactly the displayed formulas for
`j_vector_L`, `j_vector_R`, `j_vector_V`, `o_scalar_L`, `o_scalar_R`,
`j_vector_walkerL`, `j_vector_walkerR`, `j_vector_L_walkerL`,
`j_vector_R_walkerR`, `o_yukawa_LR`, `o_yukawa_RL`,
`j_vector_u1`, `j_vector_L_u1`, `j_vector_L_su2`, and `j_vector_R_su2`.
The `_count` helper simultaneously returns the displayed cardinalities
`n_valid_pairs`, `n_valid_pairs_LL`, `n_valid_pairs_RR`, and `n_valid_pairs_LR`.

Finally, the function squares the already averaged current series and inserts them into the two
rational expressions defining `parity_violation_dirac` and `parity_violation_walker`, with the
regularizer `eps_pv = 1e-30`. This proves the claim. $\square$
:::

:::{prf:remark} Component means and exchange-mixed parts of the recorded spinor series
:label: rem-qft-ew-spinor-series

Each current series of {prf:ref}`prop-qft-ew-spinor-realization` is the
component mean $\tfrac13\sum_kB_{\gamma^k,P}$ of a three-component family. By
{prf:ref}`prop-qft-component-contraction` its correlator depends on the
recorded component basis; the basis-independent statistic of the same family
is the contracted correlator of the three component series. By
{prf:ref}`prop-qft-dirac-lift-properties`, on a mutual distance pairing
$o_{\mathrm{scalar},L}$ and $o_{\mathrm{scalar},R}$ are the same series
$\tfrac12\operatorname{Avg}_{V_t}[B_{I,I}]$, while the unsplit current series
are exchange-even. Role-restricted averages use masks that differ at the two
ends of a pair and are not constrained.
:::

:::{div} feynman-prose
Two things worth noticing about the series this pipeline actually records.

The currents all carry that $\tfrac13\sum_k$ out front — a component mean,
the very thing {prf:ref}`prop-qft-component-contraction` warned about. Their
correlators are not statistics of the three-component family; they are
statistics of one projection of it, onto a direction fixed by how the array
indices were laid out. The basis-independent alternative is right there: keep
the three series, correlate each with itself, add. It costs nothing but
bookkeeping.

The second is sharper. On a mutual distance pairing, the left-projected and
right-projected scalar bilinears are *the same series*. Not similar, not
close — equal, because the part that distinguishes them is exchange-odd and
cancels in the frame average, leaving both equal to half the unprojected
scalar. So a left-right asymmetry built from those two is identically zero,
and no amount of running will make it nonzero. If you want a parity
diagnostic with content, it has to come from the role-restricted averages,
whose masks genuinely differ at the two ends of a pair, and those are not
constrained by this argument either way.
:::

### Legacy phase/doublet proxy construction

The older U(1)/SU(2) phase and doublet channels are retained for continuity, comparison, and gauge
coherence diagnostics. They remain valid observables, but they should be read as a legacy proxy
family rather than the primary electroweak matter-sector story.

:::{prf:theorem} Active Electroweak Mass-Fit Domain
:label: thm-qft-ew-active-pipeline

**Rigor Class:** F (Implementation-Exact)

In the current dashboard pipeline, the electroweak mass fitter acts only on correlator keys
present in `state["electroweak_correlator_output"].correlators`. Consequently, the fitted
electroweak masses are extracted only from the user-selected legacy electroweak channels together
with the user-selected chirality channels and, when enabled, the user-selected Dirac-spinor
channels. No additional clustering observable or latent-dimension proxy enters the mass fit unless
it has first been materialized as a correlator key in that pipeline result.
:::

:::{prf:proof}
The electroweak correlator tab first collects the user-selected channel names from the U(1), SU(2),
mixed, symmetry-breaking, parity-velocity, and chirality selectors. It passes that list to
`compute_electroweak_channels(history, channels=selected_channels, config=cfg)`, converts the
output to a `PipelineResult`, and stores it as `state["electroweak_correlator_output"]`.

If `enable_dirac_spinors=True`, the helper `_compute_dirac_spinor_channels` iterates only over the
user-selected entries of the Dirac-spinor selector. For each selected key `ch_name` that matches a
field of `ElectroweakSpinorOutput`, it inserts exactly two objects into the same `PipelineResult`:
the operator time series `result.operators[ch_name]` and its FFT correlator
`result.correlators[ch_name]`. No unselected spinor key is inserted.

The electroweak mass tab then reads
`pipeline_result = state["electroweak_correlator_output"]` and forms channel groups solely from
`list(pipeline_result.correlators.keys())`. The widget selectors in that tab can only remove keys
from those groups; they cannot introduce new ones. After this filtering, the code calls
`extract_masses(pipeline_result, config)`. Therefore the fit domain is exactly the set of retained
correlator keys already present in `pipeline_result.correlators`.

In particular, the mass fitter has no direct access to any independent Higgs-clustering observable,
to any latent-dimension label, or to any undocumented diagnostic outside the stored correlator map.
Only realized correlator channels are fitted. For the identically zero
same-frame channels proved in {prf:ref}`prop-qft-ew-chirality-realization`,
the exact correlator contains no nonzero exponential signal from which a
mass can be identified. Availability of a channel key does not alter that
algebraic fact. $\square$
:::

Let $c_d(i)$ be the **distance** companion and $c_c(i)$ the **clone** companion of walker $i$. The
legacy U(1) and SU(2) phases are constructed from the fitness differences as

$$
\phi_i^{(U1)} = -\frac{F_{c_d(i)} - F_i}{\hbar_{\text{eff}}}, \qquad
\phi_i^{(SU2)} = \frac{F_{c_c(i)} - F_i}{(|F_i| + \epsilon_{\text{clone}})\,h_S}.
$$

The companion-localized amplitudes use the algorithmic distance
({prf:ref}`def-fractal-set-companion-kernel`):

$$
D_{d,i}^2 = \|x_i - x_{c_d(i)}\|^2 + \lambda_{\text{alg}}\|v_i - v_{c_d(i)}\|^2,
\qquad
D_{c,i}^2 = \|x_i - x_{c_c(i)}\|^2 + \lambda_{\text{alg}}\|v_i - v_{c_c(i)}\|^2,
$$

$$
w_{d,i} = \exp\!\left(-\frac{D_{d,i}^2}{2\epsilon_d^2}\right), \qquad
w_{c,i} = \exp\!\left(-\frac{D_{c,i}^2}{2\epsilon_c^2}\right),
$$

and amplitudes $A_{d,i}=\sqrt{w_{d,i}}$, $A_{c,i}=\sqrt{w_{c,i}}$. The dashboard computes
correlators from these complex phase series and extracts masses using the same effective-mass
relation and correlation-length definition {prf:ref}`def-correlation-length`.

:::{prf:remark} Ranges, regularizer and phase scales of the proxy family
:label: rem-qft-ew-ranges

The three numbers $\epsilon_d$, $\epsilon_c$ and $\epsilon_{\text{clone}}$ are
distinct. $\epsilon_d$ and $\epsilon_c$ are the ranges of the distance and
cloning companion kernels, in the units of the algorithmic distance; they are
the amplitude widths $\ell_d$, $\ell_c$ of
{prf:ref}`def-sm-direct-companion-doublet`. $\epsilon_{\text{clone}}$ has the
units of a fitness and enters the score denominator only. When a companion
law has no range, as for the uniform matchings of the Einstein–Hilbert Gas
({prf:ref}`rem-variants-measurability`), the run does not determine the
amplitude width; the analysis declares either the modulus one or an explicit
width, and the coupling estimates $g_1^{\text{est}}$, $g_2^{\text{est}}$ below
are undefined.

The score phase uses the dimensionless scale $h_S$ of
{ref}`(SM.U1) <eq-fg-sm-u1>`. The dashboard sets $h_S=\hbar_{\text{eff}}$
numerically; this identification is a declared nondimensionalization. For
positive fitness $|F_i|=F_i$.

On a mutual distance pairing the imaginary parts of $O_{u1}$ and $O_{u1,d}$
vanish identically and the measured observables are
$\langle\cos\phi^{(U1)}\rangle$ and $\langle A_d\cos\phi^{(U1)}\rangle$
({prf:ref}`cor-sm-direct-exchange-parity`). On a mutual cloning pairing with
pair-symmetric weights the frame average of the difference doublet vanishes
and the frame average of the sum doublet is twice that of the component
({prf:ref}`cor-sm-paired-doublet-cancellation`).
:::

:::{div} feynman-prose
Three epsilons, and they are not three names for one idea. Two of them —
$\epsilon_d$ and $\epsilon_c$ — are *ranges*: they live in the units of the
algorithmic distance and they set how far a companion kernel reaches. The
third, $\epsilon_{\text{clone}}$, lives in the units of a *fitness* and never
leaves the denominator of a score. They cannot be traded off against each
other, they do not have the same dimensions, and the fact that they share a
letter is an accident of naming.

The denominator deserves one more look: it is $|F_i|+\epsilon_{\text{clone}}$,
with the absolute value. If the fitness can go negative and you drop those
bars, the phase flips sign on exactly the walkers where it matters most, and
worse, the denominator can pass through zero and take the whole score with
it. When fitness is positive the two forms agree, which is precisely why the
difference is easy to miss and expensive to find.

Finally, the rangeless case. Some variants match companions uniformly, with
no kernel width at all. Then $\epsilon_d$ is not a small number — it is not a
number. The amplitude envelope is undetermined, and the coupling estimates
that divide by it have nothing to divide by. Declare a width or declare the
modulus one, and say which you did; do not let a missing parameter default
silently to something.
:::

The dashboard proxies are computed from phase dispersion (a diagnostic for phase coherence, not a
direct measurement of the physical couplings in {doc}`07_qft_calibration_report`):

$$
g_1^{\text{proxy}} = \operatorname{std}(\phi^{(U1)}), \qquad
g_2^{\text{proxy}} = \operatorname{std}(\phi^{(SU2)}),
$$

$$
\sin^2\theta_W^{\text{proxy}} = \frac{(g_1^{\text{proxy}})^2}{(g_1^{\text{proxy}})^2+(g_2^{\text{proxy}})^2},
\qquad
\tan\theta_W^{\text{proxy}} = \frac{g_1^{\text{proxy}}}{g_2^{\text{proxy}}}.
$$

The coupling estimates displayed for this proxy family follow directly from Volume 2:

$$
g_1^{\text{est}} = \sqrt{\frac{\hbar_{\text{eff}}}{\epsilon_d^2}}, \qquad
g_2^{\text{est}} = \sqrt{\frac{2\hbar_{\text{eff}}}{\epsilon_c^2}\frac{C_2(2)}{C_2(d)}}.
$$

**Calibration cross-check.** The dashboard label `g1_est (N1=1)` corresponds to the simplified
$\mathcal{N}_1(T,d)=1$ normalization. To compare with the calibration report, rescale via
$g_1 = g_1^{\text{est}}\sqrt{\mathcal{N}_1(T,d)}$ and use the report's $g_2$ directly.

**Measurement note.** The active `electroweak_channels.py` path resolves $\epsilon_d$ and
$\epsilon_c$ from `RunHistory.params`, keeps $\lambda_{\text{alg}}=0$ in that path, and uses
$\hbar_{\text{eff}}$ and $\epsilon_{\text{clone}}$ as the main analysis-level controls. The
projector-based spinor path additionally uses `mass`, `ell0`, and `ell0_method` when constructing
color states and Dirac spinors. The UI merges chirality, spinor, and legacy proxy correlators into
a single electroweak result object before passing them to the mass-extraction tab.

**Legacy proxy reference mapping.**

| Electroweak channel | Proxy reference (GeV) | Dashboard mapping |
| --- | --- | --- |
| `u1_phase` | 0.000511 | electron |
| `u1_dressed` | 0.105658 | muon |
| `su2_phase` | 80.379 | $W$ boson |
| `su2_doublet` | 91.1876 | $Z$ boson |
| `ew_mixed` | 1.77686 | tau |

These references are dashboard anchors for visual comparison of the legacy proxy masses. The
chirality and projector layers are not constrained to this five-channel mapping; the mass tab fits
whatever electroweak channels are selected. The calibration inversion in
{doc}`07_qft_calibration_report` uses measured couplings
$(\alpha_{\text{em}}, \sin^2\theta_W, \alpha_s)$ at a chosen scale instead of these proxy masses.

### Legacy U(1) phase channel (`u1_phase`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → typically smaller phase winding → often lighter $m_{u1}$. |
| Distance companion selection | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → tighter locality → often heavier $m_{u1}$ (validate by sweep). |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Time step | $\Delta t$ | `KineticOperator.delta_t` | Changes the generating dynamics; relabeling a fixed analysis time unit instead rescales rates uniformly. |

**Operator (U(1) phase mean):**

$$
O_{u1}(t) = \left\langle e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Increasing $\hbar_{\text{eff}}$ tends to reduce phase dispersion and lengthen the correlator.
- Decreasing $\epsilon_d$ in the generating run typically tightens companion locality and shortens
  the correlator; confirm empirically.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting and can shorten the correlator. The active dashboard route keeps this term off.
- Fitness coupling parameters ($\epsilon_F$, fitness weights) can shift the fitness differences and
  therefore the U(1) phase spread.

### Legacy U(1) dressed channel (`u1_dressed`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Distance temperature | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → sharper amplitude localization → often heavier $m_{u1,d}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as stronger velocity weighting. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{u1,d}$. |

**Operator (amplitude-weighted U(1) phase):**

$$
O_{u1,d}(t) = \left\langle A_{d,i}\,e^{i\phi_i^{(U1)}(t)} \right\rangle_{\text{alive}},
\qquad A_{d,i}=\sqrt{w_{d,i}}.
$$

**Sweep hypotheses to check:**
- Use $\epsilon_d$ from the recorded run to control the locality of the U(1) amplitude envelope;
  tighter locality often shortens the plateau.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it adds
  velocity weighting to the same envelope. The active dashboard route keeps this term fixed at zero.
- Use $\hbar_{\text{eff}}$ to control the overall phase winding without changing locality.

### Legacy SU(2) phase channel (`su2_phase`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{su2}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → smaller score → often lighter $m_{su2}$. |
| Clone companion selection | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → tighter clone locality → often heavier $m_{su2}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier SU(2) proxy. |

**Operator (SU(2) phase mean):**

$$
O_{su2}(t) = \left\langle e^{i\phi_i^{(SU2)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Decreasing $\epsilon_{\text{clone}}$ tends to increase the phase score magnitude and shorten the
  correlator (confirm by sweep).
- Decreasing $\epsilon_c$ in the generating run typically tightens clone pairing and increases
  $m_{su2}$.
- In legacy alternate implementations where $\lambda_{\text{alg}}$ is exposed, increasing it can
  also raise the proxy mass. The active dashboard route keeps this term off.
- Adjust $\hbar_{\text{eff}}$ to rescale phase winding without changing clone topology.

### Legacy SU(2) doublet channel (`su2_doublet`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| Clone temperature | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → tighter pairing → often heavier $m_{su2,d}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → often lighter $m_{su2,d}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier SU(2) doublet proxy. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{su2,d}$. |

**Operator (clone-paired doublet):**

$$
O_{su2,d}(t) = \left\langle A_{c,i}e^{i\phi_i^{(SU2)}(t)}
+ A_{c,c(i)}e^{i\phi_{c(i)}^{(SU2)}(t)} \right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Tightening clone locality (smaller $\epsilon_c$) often sharpens the doublet and shortens the
  plateau; verify with sweeps.
- Use $\epsilon_{\text{clone}}$ to regulate phase-score magnitude without changing the pairing
  graph.

### Legacy mixed electroweak channel (`ew_mixed`)

| Parameter | Symbol | Code parameter | Sweep hypothesis for the observed mass |
| --- | --- | --- | --- |
| U(1) locality | $\epsilon_d$ | `RunHistory.params["companion_selection"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_d$ in the generating run → often heavier $m_{\text{EW}}$. |
| SU(2) locality | $\epsilon_c$ | `RunHistory.params["companion_selection_clone"]["epsilon"]` resolved by `src/fragile/physics/electroweak/electroweak_channels.py` | Decrease $\epsilon_c$ in the generating run → often heavier $m_{\text{EW}}$. |
| Clone regularizer | $\epsilon_{\text{clone}}$ | `ElectroweakCorrelatorSettings.epsilon_clone` with fallback to `RunHistory.params["cloning"]["epsilon_clone"]` / `CloneOperator.epsilon_clone` | Increase $\epsilon_{\text{clone}}$ → often lighter $m_{\text{EW}}$. |
| Algorithmic distance weight | $\lambda_{\text{alg}}$ | `_resolve_electroweak_params` in `src/fragile/physics/electroweak/electroweak_channels.py` | In the active dashboard route this is pinned to `0.0`; older proxy analyses interpreted larger $\lambda_{\text{alg}}$ as a heavier mixed proxy. |
| Phase scale | $\hbar_{\text{eff}}$ | `ElectroweakCorrelatorSettings.h_eff` | Increase $\hbar_{\text{eff}}$ → often lighter $m_{\text{EW}}$. |

**Operator (U(1) × SU(2) phase product):**

$$
O_{\text{EW}}(t) = \left\langle A_{d,i}A_{c,i}\,e^{i(\phi_i^{(U1)}(t)+\phi_i^{(SU2)}(t))}
\right\rangle_{\text{alive}}.
$$

**Sweep hypotheses to check:**
- Use $\epsilon_d$ and $\epsilon_c$ to control the relative U(1) vs. SU(2) localization; the mixed
  channel is typically the most sensitive to simultaneous changes in both.
- Adjust $\hbar_{\text{eff}}$ and $\epsilon_{\text{clone}}$ to shift phase winding without changing
  the companion graphs; validate shifts with the Electroweak tab fits.

### Legacy empirical tuning status (QFT baseline)

The electroweak tuning runs in `electroweak_tuning_report.md` (zero reward, viscosity-only,
analysis-level Electroweak tab) support the theoretical mapping for **coupling estimates** but not
for **mass ratios**:

- **Couplings**: adjusting $\epsilon_d$ and $\epsilon_c$ moves $g_1^{\text{est}}$ and
  $g_2^{\text{est}}$ as predicted, and defaults land within $\sim$3.5% of the $M_Z$ targets.
- **Ratios**: observed proxy ratios remain $\mathcal{O}(1)$ across analysis-level sweeps. The best
  $m_{\text{su2\_doublet}}/m_{\text{u1\_dressed}}$ achieved $\sim 5.83$ (target $\sim 863$), and
  $m_{\text{u1\_phase}}/m_{\text{u1\_dressed}}$ stays orders of magnitude above the observed
  electron/muon ratio.
- **Interpretation**: the electroweak channels are therefore **phase-coherence diagnostics** in the
  current baseline, not a calibrated reproduction of electroweak mass hierarchies. This aligns with
  the coupling inversion workflow in {doc}`07_qft_calibration_report`, which calibrates couplings
  directly rather than through proxy mass ratios.

(sec-qft-calibration-ratio-sieve)=
## Ratio-sieve theorems (symbolic constraints)

Define the Channels-tab mass ratios (symbolic targets):

$$
R_{\sigma\pi} := \frac{m_\sigma}{m_\pi}, \qquad
R_{\rho\pi} := \frac{m_\rho}{m_\pi}, \qquad
R_{G\pi} := \frac{m_G}{m_\pi}, \qquad
R_{N\pi} := \frac{m_N}{m_\pi}.
$$

:::{prf:definition} Reference ratios and the hadron-label hypothesis
:label: def-qft-reference-ratios

The **reference ratios** are the external numbers

$$
R_{\rho\pi}^{\mathrm{ref}}=5.5,\qquad R_{N\pi}^{\mathrm{ref}}=6.7,
$$

the two-digit truncations of the measured mass ratios
$m_\rho/m_{\pi^\pm}=5.5546$ and $m_p/m_{\pi^\pm}=6.7226$. The **hadron-label
hypothesis** is the statement that the decay rates of the channels labelled
$\pi$, $\rho$, $N$ of a gas variant stand in these ratios. It is a hypothesis
about a labelling; {prf:ref}`def-sm-direct-color-contractions` assigns no
particle to a channel.

The reference ratios enter this chapter in two places only: as the hypothesis
of {prf:ref}`cor-qft-ratio-numeric-bounds` and of item 4 of
{prf:ref}`cor-qft-parameter-sieve`, and as comparison values for measured
ratios. They enter no operator definition, no estimator, no fit prior and no
fit window. A ratio $R_{\chi\pi}$ is defined only when both channels have a
decay rate ({prf:ref}`def-qft-channel-decay-rate`) obtained with one
estimator, one frame normalization and one time unit.
$R_{\sigma\pi}$ and $R_{G\pi}$ remain symbolic until measured.
:::

:::{prf:remark} Selection and evidence
:label: rem-qft-reference-ratio-selection

A parameter set retained because its measured ratios lie near the reference
ratios has been selected on that outcome; its agreement with them is not
evidence for the hadron-label hypothesis. Evidence requires ratios measured
on runs and seeds that played no part in the selection, with the channel,
estimator, normalization, fit window and priors fixed beforehand, and with
the number of compared ratios stated. On a mutual pairing the primary $\pi$
and $\rho$ frame series vanish identically
({prf:ref}`cor-sm-direct-exchange-parity`) and the $N$ frame series has no
decay rate under exchangeable companion roles
({prf:ref}`prop-sm-direct-role-swap`); the ratios are then undefined for
frame-average estimators.
:::

:::{div} feynman-prose
Where do $5.5$ and $6.7$ come from? Not from the gas. They are
$m_\rho/m_{\pi^\pm}$ and $m_p/m_{\pi^\pm}$ from the particle data tables, cut
to two digits. That is a perfectly respectable thing to compare against — but
notice that comparing against them presumes something substantial: that the
channel we call $\pi$ should be read as a pion, $\rho$ as a rho, $N$ as a
nucleon. Nothing in the definition of those channels says so. They were built
by analogy, and the analogy is the hypothesis, not a result.

Now the part that requires real discipline. Suppose you sweep a thousand
parameter sets, keep the ones whose $R_{\rho\pi}$ lands near $5.5$, and then
report that the survivors have $R_{\rho\pi}$ near $5.5$. You have discovered
nothing except that your filter works. This is not a subtle statistical
point; it is the whole point. Selection on an outcome destroys that outcome's
value as evidence for the hypothesis that motivated the selection.

What would count as evidence? Fix everything first — channel, estimator,
normalization, fit window, priors — then measure on runs and seeds that took
no part in the selection, and say how many ratios you compared. That last bit
matters too: compare enough quantities and one of them will land on target by
luck. There is nothing wrong with using the reference ratios as a sieve. Just
do not then hand the sieve's output back as a confirmation.
:::

:::{div} feynman-prose
These numbers are chosen calibration targets. The following algebra supplies necessary constraints within a specified scale and coupling model. It does not establish that every selected dashboard channel has an asymptotic mass, or that satisfying the constraints reproduces the targets.
:::



:::{prf:theorem} Ratio invariance under relabeling a fixed time unit
:label: thm-qft-ratio-rescale

Fix the correlator values $C_\chi[n]$ at integer lags and assign a time unit
$\Delta\tau>0$ per lag. Wherever the effective mass is defined,

$$
m_\chi[n]=-\frac1{\Delta\tau}\log\frac{C_\chi[n+1]}{C_\chi[n]}.
$$

Replacing only the assigned unit by $s\Delta\tau$, $s>0$, sends
$m_\chi[n]$ to $m_\chi[n]/s$ and leaves ratios unchanged.
:::

:::{prf:proof}
The correlator quotient remains fixed while the prefactor is divided by $s$.
The common factor cancels in a ratio. A new generating time step or recording
stride can change the correlator sequence itself and is outside this
unit-relabeling statement.
:::

:::{div} feynman-prose
Changing only the unit label cannot tune ratios. Hold the integrator step and recording settings controlled during parameter sweeps, and treat changes to either as changes to the experiment.
:::



:::{prf:corollary} Dimensionless Reduction of Ratio Dependence
:label: cor-qft-ratio-dimensionless

In a model covariant under its declared changes of units, a mass ratio is a function of dimensionless inputs. The combinations in {prf:ref}`thm-dimensionless-ratios` and the coupling conventions provide the following useful coordinates.
The displayed combinations provide reduced coordinates for the declared scale model:

$$
(\sigma_{\text{sep}}, \eta_{\text{time}}, \kappa; \; g_1, g_2, g_3; \; N, d; \; \phi),
$$
where $\phi := m\ell_0/\hbar_{\text{eff}}$ is the phase-winding combination from
{prf:ref}`thm-sm-su3-emergence`.
:::

:::{prf:proof}
{prf:ref}`thm-dimensionless-ratios` enumerates the fundamental dimensionless
ratios built from $(m,\tau,\rho,\epsilon_c)$; the gauge couplings are themselves dimensionless
({prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`), and
the displayed phase factor contains $\phi$. Other dimensionless regularizers, operator settings, or kernel parameters must be included if they vary; dimensionlessness alone does not make this coordinate list exhaustive.
$\square$
:::

:::{prf:theorem} A clustering envelope bounds an asymptotic decay exponent
:label: thm-qft-channel-gap-bound

Suppose the specified channel correlator has a clustering bound
$|C_\chi(t)|\leq A e^{-m_{\mathrm{gap}}t}$ with $A<\infty$, and its
nonzero asymptotic exponential rate $m_\chi$ exists. Then
$m_\chi\geq m_{\mathrm{gap}}$. In particular, if
$C_\chi(t)=Z_\chi e^{-m_\chi t}(1+o(1))$ with $Z_\chi\ne0$, the result
applies. When the model identifies
$m_{\mathrm{gap}}=\hbar_{\mathrm{eff}}\lambda_{\mathrm{gap}}$, this is the
corresponding lower bound in that normalization.
:::

:::{prf:proof}
At times with $C_\chi(t)\ne0$, take logarithms of the envelope:

$$
-\frac1t\log|C_\chi(t)|\geq m_{\mathrm{gap}}-\frac{\log A}{t}.
$$

Taking the lower limit proves the claim. For the stated leading exponential,
the left side tends to $m_\chi$. An envelope alone does not bound every
finite-lag logarithmic ratio; a fitted plateau estimates an asymptotic rate
only with control of competing contributions and fit error.
:::

:::{prf:corollary} Ratio-Driven Bounds on $\lambda_{\text{gap}}, \eta_{\text{time}}, \kappa$
:label: cor-qft-ratio-gap-bounds

Let

$$
m_{\min} := \min(m_\pi, m_\sigma, m_\rho, m_G, m_N)
       = m_\pi \cdot \min(1, R_{\sigma\pi}, R_{\rho\pi}, R_{G\pi}, R_{N\pi}).
$$
Then

$$
\lambda_{\text{gap}} \leq \frac{m_{\min}}{\hbar_{\text{eff}}},
\qquad
\eta_{\text{time}} = \tau \lambda_{\text{gap}} \leq \tau \frac{m_{\min}}{\hbar_{\text{eff}}},
\qquad
\kappa = \frac{1}{\rho \hbar_{\text{eff}} \lambda_{\text{gap}}} \geq \frac{1}{\rho m_{\min}}.
$$

These are necessary bounds when the chosen channels possess the asymptotic rates and common clustering envelope of the preceding theorem. Applying them to fitted values must retain fit and approximation uncertainty.
:::

:::{prf:proof}
Apply the preceding theorem to every included nonzero asymptotic channel rate and take their minimum. Divide by $\hbar_{\mathrm{eff}}>0$, multiply by $\tau>0$, and invert the positive inequality for $\rho\hbar_{\mathrm{eff}}\lambda_{\mathrm{gap}}$. These operations give the three bounds.
:::


:::{prf:corollary} Explicit Pruning Bounds for $R_{\rho\pi}=5.5$, $R_{N\pi}=6.7$
:label: cor-qft-ratio-numeric-bounds

Assume the hadron-label hypothesis of {prf:ref}`def-qft-reference-ratios`,
$R_{\rho\pi}=R_{\rho\pi}^{\mathrm{ref}}=5.5$ and
$R_{N\pi}=R_{N\pi}^{\mathrm{ref}}=6.7$, for channels that possess decay
rates. Then

$$
m_\rho = 5.5\,m_\pi, \qquad m_N = 6.7\,m_\pi,
$$
and

$$
m_{\min} = m_\pi \cdot \min(1, R_{\sigma\pi}, R_{G\pi})
$$
because both $5.5$ and $6.7$ exceed $1$. Therefore the ratio-sieve bounds become

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\eta_{\text{time}} \leq \tau \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\kappa \geq \frac{1}{\rho\,m_\pi\,\min(1, R_{\sigma\pi}, R_{G\pi})}.
$$
In particular, if measurements give $R_{\sigma\pi} \geq 1$ and
$R_{G\pi} \geq 1$, then

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}, \qquad
\eta_{\text{time}} \leq \tau \frac{m_\pi}{\hbar_{\text{eff}}}, \qquad
\kappa \geq \frac{1}{\rho\,m_\pi}.
$$
:::

:::{prf:proof}
Substitute $m_\rho=5.5m_\pi$ and $m_N=6.7m_\pi$ into the minimum. Since both multipliers exceed one, neither lowers the minimum. Apply the previous corollary and simplify; if the remaining ratios are also at least one, the minimum multiplier is one.
:::


:::{prf:definition} Candidate constraints in the declared calibration model
:label: cor-qft-parameter-sieve

Within the declared hierarchy and coupling model, define the algebraically admissible candidate set by the following constraints. They are necessary for matching the specified asymptotic masses in that model; they are not sufficient for agreement of measured plateaus:

1. **Hierarchy constraint** ({prf:ref}`thm-mass-scales`):

$$
m_{\text{friction}} \ll m_{\text{gap}} < m_{\text{MF}} < m_{\text{clone}}.
$$

2. **Dimensionless ratios** ({prf:ref}`thm-dimensionless-ratios`):

$$
\sigma_{\text{sep}} = \frac{\epsilon_c}{\rho}, \quad
\eta_{\text{time}} = \tau\lambda_{\text{gap}}, \quad
\kappa = \frac{1}{\rho \hbar_{\text{eff}} \lambda_{\text{gap}}}.
$$

3. **Gap lower bound (all channels)** ({prf:ref}`thm-qft-channel-gap-bound`):

$$
m_\chi \geq \hbar_{\text{eff}} \lambda_{\text{gap}} \quad \text{for } \chi \in \{\pi,\sigma,\rho,G,N\},
$$

for each listed channel that possesses a decay rate.

4. **Ratio-sieve bounds under the hadron-label hypothesis**
({prf:ref}`def-qft-reference-ratios`, {prf:ref}`cor-qft-ratio-numeric-bounds`):

$$
R_{\rho\pi} = 5.5, \qquad R_{N\pi} = 6.7,
$$

$$
\lambda_{\text{gap}} \leq \frac{m_\pi}{\hbar_{\text{eff}}}\,\min(1, R_{\sigma\pi}, R_{G\pi}),
$$

$$
\kappa \geq \frac{1}{\rho\,m_\pi\,\min(1, R_{\sigma\pi}, R_{G\pi})}.
$$

5. **Coupling inversion manifold** ({prf:ref}`cor-qft-coupling-inversion-manifold`):

$$
\epsilon_c = \sqrt{\frac{2\hbar_{\text{eff}}C_2(2)}{C_2(d)\,g_2^2}}, \quad
\rho = g_2\sqrt{\frac{2\hbar_{\text{eff}}}{m^2}}, \quad
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}.
$$

A candidate failing a required model constraint is excluded from that declared regime. Finite-data estimates require uncertainty margins before they can justify exclusion.

:::

### Pruning procedure (pre-sweep)

Use the checklist above as a deterministic filter before running large parameter sweeps.

1. **Fix absolute time scale**: choose $\Delta t$ (and `record_every`) and hold fixed for all
   runs so ratios are comparable ({prf:ref}`thm-qft-ratio-rescale`).
2. **Invert couplings**: for chosen $(g_1,g_2,g_3)$ and QSD statistics, solve for
   $(\epsilon_d,\epsilon_c,\nu,\epsilon_F,\rho,\tau)$ using
   {prf:ref}`cor-qft-coupling-inversion-manifold`. Discard any candidate that violates the
   hierarchy in {prf:ref}`thm-mass-scales`.
3. **Check dimensionless diagnostics**: compute
   $(\sigma_{\text{sep}}, \eta_{\text{time}}, \kappa)$ from
   {prf:ref}`thm-dimensionless-ratios`. Discard candidates outside the stable regime indicated by
   prior calibrated runs.
4. **Pilot estimate of $m_\pi$**: run a short QSD‑valid trajectory and extract the decay rate
   $m_\pi$ of a declared pseudoscalar channel — operator, companion map, alignment, estimator and
   normalization — whose correlator is not identically zero
   ({prf:ref}`cor-sm-direct-exchange-parity`) ({prf:ref}`def-euclidean-correlator-fg`,
   {prf:ref}`def-two-point-connected`, {prf:ref}`def-correlation-length`).
5. **Apply ratio bounds**: enforce {prf:ref}`cor-qft-ratio-numeric-bounds` using the pilot
   estimate of $m_\pi$ (and symbolic $R_{\sigma\pi}, R_{G\pi}$ if still unanchored). Discard
   candidates that violate the inequalities. These bounds are consequences of the hadron-label
   hypothesis; they do not test it ({prf:ref}`rem-qft-reference-ratio-selection`).

:::{div} feynman-prose
Treat a pilot fit as an estimate with uncertainty. Exclusion by an asymptotic spectral constraint is justified only when its hypotheses and the error margin hold for that channel. A missing plateau or an identically zero observable supplies no mass estimate.
:::



:::{prf:corollary} Coupling-Inversion Manifold (Symbolic Constraints)
:label: cor-qft-coupling-inversion-manifold

For fixed positive normalization factors and QSD statistics, the coupling assignments constrain the corresponding ranges and amplitudes. The displayed inversion additionally adopts the indicated relations for $\rho$ and $\tau$; a fitness coupling must be specified to fix $\epsilon_F$. These are algebraic constraints within the chosen model, using:
{prf:ref}`thm-sm-g1-coupling`, {prf:ref}`thm-sm-g2-coupling`, {prf:ref}`thm-sm-g3-coupling`,
{prf:ref}`thm-u1-coupling-constant`, and {prf:ref}`thm-effective-planck-constant`. In particular,

$$
\epsilon_c = \sqrt{\frac{2\hbar_{\text{eff}}C_2(2)}{C_2(d)\,g_2^2}},
\qquad
\rho = g_2\sqrt{\frac{2\hbar_{\text{eff}}}{m^2}},
\qquad
\tau = \frac{m\,\epsilon_c^2}{2\hbar_{\text{eff}}}.
$$

These equations define a restricted candidate set when all their relations are imposed. They do not fix omitted dimensionless parameters or the QSD statistics produced by a new run. Calling that set a manifold additionally requires the usual regularity and rank conditions for its defining equations.
:::

:::{prf:proof}
The first formula follows by solving
$g_2^2=2\hbar_{\mathrm{eff}}C_2(2)/(\epsilon_c^2C_2(d))$ for its positive
range. The displayed $\rho$ relation is equivalent to imposing
$g_2^2=m^2\rho^2/(2\hbar_{\mathrm{eff}})$, and the $\tau$ relation is
equivalent to imposing $\hbar_{\mathrm{eff}}=m\epsilon_c^2/(2\tau)$.
Thus they are compatible algebraic substitutions when these relations are
part of the selected model. The other coupling assignments constrain their
own parameters with their normalization factors held fixed; they supply no
additional equation for a parameter absent from those assignments.
:::


(sec-qft-calibration-code)=
## Theory-to-code map

The QFT modules mirror the notation of Volume 2. The main parameter hooks are:

- Companion selection kernel ({prf:ref}`def-fractal-set-companion-kernel`):
  run parameters are stored in `RunHistory.params` and consumed in the active analysis path by
  `src/fragile/physics/electroweak/electroweak_channels.py`.
  - Distance companion temperature $\epsilon_d$ is resolved from recorded run parameters rather than
    freely retuned inside the active electroweak channel path.
  - Clone companion temperature $\epsilon_c$ is likewise resolved from the recorded run parameters.
  - The active electroweak channel path keeps $\lambda_{\text{alg}} = 0$.
- Two-channel fitness ({prf:ref}`def-fractal-set-two-channel-fitness`):
  `src/fragile/physics/fractal_gas/fitness.py`.
- Cloning score ({prf:ref}`def-fractal-set-cloning-score`): `CloneOperator` parameters
  in `src/fragile/physics/fractal_gas/cloning.py`.
- Viscous force and color coupling ({prf:ref}`def-fractal-set-viscous-force`,
  {prf:ref}`thm-sm-su3-emergence`): `KineticOperator` parameters in
  `src/fragile/physics/fractal_gas/kinetic_operator.py`.
- Anisotropic diffusion ({prf:ref}`def-fractal-set-anisotropic-diffusion`):
  `src/fragile/physics/fractal_gas/kinetic_operator.py`.

The electroweak correlators are computed in:
- `src/fragile/physics/electroweak/chirality.py` (walker-role chirality partition and
  autocorrelation observables).
- `src/fragile/physics/electroweak/electroweak_spinors.py` (Dirac-spinor currents, Yukawa
  bilinears, and parity diagnostics).
- `src/fragile/physics/electroweak/electroweak_channels.py` (legacy proxy channels plus chirality
  channels, merged into the shared correlator pipeline).
- `src/fragile/physics/app/electroweak_correlators.py` (dashboard channel selection and operator
  family wiring).
- `src/fragile/physics/app/electroweak_mass_tab.py` (Bayesian mass extraction for the selected
  electroweak channels).

The broader correlator and mass-extraction machinery lives in:
- `src/fragile/physics/new_channels/correlator_channels.py`
- `src/fragile/physics/mass_extraction/`

Analysis window choices (fit start/stop, plateau detection, covariance model, priors) change
measurement quality, not the underlying physics.

(sec-qft-calibration-workflow)=
## Calibration workflow (parameter tuning loop)

:::{div} feynman-prose
1. Fix the observable, recorded frame convention, masks, and analysis settings.
   Verify its exact algebra, including any zero-channel identity.
2. Choose target ratios and a reference unit, recording which are input anchors.
   Use the report and notebook for the documented comparison protocol.
3. Generate histories with controlled time step, recording stride, and burn-in.
   Use the applicable QSD convergence result and observed stationarity diagnostics
   to assess the measurement window.
4. Measure the nonzero correlators and check fit-window stability, uncertainty,
   competing decay terms, and the effect of connected versus unconnected data.
5. Sweep a generating parameter or an analysis parameter separately, recording
   which changed. Verify the suggested direction from the measured response.
6. Apply scale, ratio, or gap constraints only within the model and uncertainty
   regime that justifies them. Repeat on independent runs before interpreting
   a fitted scale as a reproducible channel feature.
:::
