# Independent audit of the default full-dead response formula

(sec-dlba-source)=
## 1. Reviewed source, primitives and exact endpoint

:::{prf:definition} Frozen response source
:label: def-dlba-source

This review concerns `55_default_delayed_law_block.md` at SHA-256
`77cb63b52347e1e957ebf10a3313bebf56bd1be4238b6f7c3ce455440b316836`.
Its register {prf:ref}`def-dlb-register` keeps the actual harmonic
count population map at $h=.04$, $\nu=.3$, $V=2$, $V_c=4$,
$d=3$, $L=2$, positive source jitter and both full Gaussian
kinetic innovations. The raw reward remains $-|x|^2/2$.
Original-slot component velocities, current measured normalizers,
eligible alive donors, mandatory dead revival, native cap and
terminal marks are retained.

The positive exponent restriction is explicitly the weak interval
of {prf:ref}`def-dmc-register`, so $\theta\le1$ and the alive
accepted column is at most $1/8$. Unit configured powers are
not asserted to lie in that interval. The dead fraction is only
bounded by $\bar e=1-m_0$, with
$m_0=a_{\rm ret}/2>0$ from the actual default safe-return floor.
This may be close to one and is not treated as a small quantity.

All variation norms use full variation. This audit checks a complete
absolute provider bound, a signed Duhamel identity and its quantitative
current-alive readout. It checks no nonlinear contraction margin.
:::

(sec-dlba-closure)=
## 2. Source moments and the closed output class

:::{prf:remark} Gaussian and moment hypotheses are discharged
:label: rem-dlba-output-class

For every consistent entering law with positive alive mass, each
persistent, copied or revived frozen position source lies in $D$.
The original component readout is bounded by $V_c$. Because
$t\nu=.006<1$, its first count average $U$ is a convex
velocity average and has the same bound. The exact landing position
is the displayed source plus bounded $bU$ and the original Gaussian
sum. Conditional on its source/component plan, the Gaussian sum
has covariance at most
$(a_x^2\sigma_J^2+t^2q^2+s^2)I_3$.

The source's moment proof does not presume $U$ is independent of
copied jitter. Minkowski separates a pointwise bounded $bU$ from
the Gaussian sum, which proves its $H_{\rm box}$ bound despite
that dependence. The last position Gaussian is independent of the
stored capped velocity and the pre-final position. Therefore the
Gaussian-mixture representation, eighth moment and cap hypotheses
in {prf:ref}`lem-dlb-output-class` hold exactly.

The actual safe-return theorem
{prf:ref}`thm-dsa-default-box-alive-floor` gives alive mass at
least $a_{\rm ret}>m_0$. Thus the declared output class is
preserved and has no hidden retained-dead position restriction.
Entry occurs after one update even when entering dead coordinates
have no moment bound.

The independent copied jitter also gives a conditional prepared
fourth-weight bound. The exact identity
$$
\mathbb E|S+J|^4=|S|^4+2(d+2)\sigma_J^2|S|^2
                         +d(d+2)\sigma_J^4
$$
and $2|S|^2\le1+|S|^4$ yield
$\mathbb E W_4(S+IJ)\le D_JW_4(S)\le D_JB_L$.
Here $B_L=1+d^2L^4=145$. This bound is conditional on every
source/component outcome and hence also covers a common root law
under another environment. The fifth prepared-position moment has
the same source-box Minkowski proof. No component/source
independence is assumed.
:::

(sec-dlba-dead)=
## 3. Raw alive normalization and full mandatory-dead feedback

:::{prf:remark} Preparation hazards with arbitrary finite dead intensity
:label: rem-dlba-preparation

Every conditional alive environment has eighth moment at most
$H_A=(\sqrt dL)^8$ and fourth weight at most
$B_A=1+\sqrt{H_A}=145$. Exact normalized alive subtraction
gives
$$
\|\alpha_\mu-\alpha_{\mu'}\|_{W_4}
\le\frac{2B_L}{m_0}\|\mu-\mu'\|_1.
$$
The raw logistic-tail theorem is therefore applied at this actual
conditional alive moment budget. Reward means, variances and their
positive standardizer floors change with the environment and are
included in its $L_J^Q$; no bounded reward is substituted.

The alive forest is subcritical because $c_*\le1/8$. Its
ordered query bound and the corresponding forced-edge bound are
independent of dead intensity. Mandatory dead vertices cannot be
donors and consume their outgoing edge when attached, so they are
leaves. Their intensity ceiling
$\bar e/(\kappa_Cm_0)$ is finite, may exceed one, and appears
in the proof and constants of
{prf:ref}`lem-dlb-full-dead-preparation`.

Directly subtracting the exact dead-child denominator yields the
stated hazard
$H_D\le\bar e h_A\delta_A+h_D\delta_D$.
The source-first common dead root is charged before its remaining
component. Summing the three displayed alive-edge, extra-dead-leaf
and common-dead-root contributions gives a coefficient no larger
than $C_A(\theta+\bar e)$ on $\delta_A$ and exactly
$C_D$ on $\delta_D$. In this summation
$\theta\le1$ controls the product $\bar e\theta L_J^Q$.
No inequality uses $\bar e\le1/4$ or discards a large hazard.

Finally the fixed-provider entering-law term is bounded by
$C_{\rm src}\|\mu-\mu'\|_1$ using the conditional fourth
readout, and the common-root term is bounded by $A_J$.
This verifies both complete preparation estimates, including
original dead velocities in the component readout.
:::

(sec-dlba-scores)=
## 4. Default first inverse and correlated Gaussian scores

:::{prf:remark} All score hypotheses hold at the declared viscosity
:label: rem-dlba-scores

The raw marked BV proof applies with full dead intensity. Root
measurement and accepted-edge derivatives use the actual alive
fitness gradient $\theta[Q_x(H_A)+J_s/\sigma_s]$.
The incoming mandatory-dead intensity contributes its displayed
$\bar e\ell_C/(\kappa_Cm_0)$ derivative. Copied and revived
roots use their own recipient jitter; a persistent alive root uses
the entering final Gaussian and its full box-face trace.
Taking absolute derivatives before component and velocity mixing
supplies the required velocity-fibre source bound
$B_5^{\rm src}$. Its fifth entering moment is bounded by
$H_{\rm box}^{5/8}$.

The actual first inverse condition in
{prf:ref}`lem-pvb-joint-scores` is admissible at $\nu=.3$.
Here $m=.9996$, $t=.02$ and
$L_0=4dV_c\ell_\rho<48$, whence
$$
\frac m{2t}>1,\qquad \frac m{2t^2L_0}>1,
\qquad \bar\nu=1,
$$
$$
m-t^2\nu L_0>.99384>0,
\qquad 1-t\nu=.994>0.
$$
The source fifth moment and fibrewise BV verify its other
hypotheses. Under first-provider interpolation the prepared source
law is fixed, while the provider velocities remain bounded by
$V_c$; the same inverse and score constants apply.

The second comparison uses the actual joint density
$\rho(y,w)=f(y-tw,w)$ and its velocity score
$\nabla_w\rho=\nabla_w f-t\nabla_{x_1}f$.
The shear contribution $tS_x$ is included. Neither the score
theorem nor {prf:ref}`thm-pvb-viscous-feedback` resamples
independent velocity and position marginals. Their weighted
products are integrated inside each velocity fibre before mixing.

Changing preparation first and then both actual kinetic providers
gives exactly the source's $L_{\rm fb}$ bound in
{prf:ref}`thm-dlb-default-feedback`. The native cap, final
Gaussian and terminal mark are common Markov pushforwards and
contract full variation. A discontinuous indicator is not
differentiated in that comparison. Thus no Gaussian-tail or
boundary event is omitted from this complete marked bound.
:::

(sec-dlba-history)=
## 5. Chronological Duhamel products and exact alive readout

:::{prf:remark} Correct signed identity and normalization
:label: rem-dlba-duhamel

The frozen root Doeblin theorem
{prf:ref}`thm-rcb-source-box-frozen-gap` applies to each
$P_{\mu_j}$: its source is in the actual box, prepared velocity
is bounded, the joint OU provider has the proved finite moment,
and its proper-degree and final Gaussian minorization retain the
mandatory revival branch. Its common floor applies to every
consistent capped root, including roots drawn from the other
history. Consequently every zero-mass signed measure contracts
under each remaining frozen-history kernel by $q_D$.

The one-step subtraction is
$\Delta_{j+1}=\Delta_jP_j+\mathcal E_j$.
Successive substitution gives the kernel order in
{prf:ref}`thm-dlb-delayed-response` exactly. The response at
time $n+j$ is followed by $B-1-j$ kernels. Each response has
zero mass, since both root kernels are probabilities. Applying
the frozen contraction to each product and then taking the
triangle therefore proves its weighted-history sum. The two
histories remain their actual nonlinear histories; their
providers have not been identified.

For a complete actual output, each own alive mass is at least
$a_{\rm ret}$. Normalized alive subtraction gives the displayed
full-variation factor $2/a_{\rm ret}$.
Alive positions have squared diameter $48$ and capped velocities
have squared diameter $16$. Maximal coupling therefore charges
at most $64$ times half the alive full variation, giving the
$64/a_{\rm ret}$ phase $W_2^2$ coefficient in
{prf:ref}`cor-dlb-current-alive-response`. Each law uses its
own current-alive denominator. This target is neither a finite
QSD nor a law conditioned on future survival.
:::

(sec-dlba-endpoint)=
## 6. Audit conclusion and the unclosed signed margin

:::{prf:remark} Passed intermediate response, without a convergence assertion
:label: rem-dlba-endpoint

The frozen source passes this complete independent review.
Its primitive full-dead preparation and Gaussian-score interfaces,
default first inverse, actual marked provider bound, chronological
signed response and current-alive readout are discharged.
The output class is closed under the actual map and verifies
the required hypotheses at every time in the response formula.

The absolute upper-bound register gives only its declared
$(q_D+L_{\rm fb})^B$ response estimate. Its conservative
$L_{\rm fb}\ge C_D\ge256$ follows from the displayed
constants and is not a lower bound on actual feedback. The
source expressly does not claim nonlinear attraction from that
register. A sharper signed or transport estimate of the delayed
provider responses is still needed for a default global rate.

No invariant marked law, finite survivor/QSD mixing or
population-uniform finite alive convergence follows from this
audit. No source correction is requested.
:::
