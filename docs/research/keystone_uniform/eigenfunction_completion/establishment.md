# Native lineage amplification and the dispersed joint continuation

The retained native record is def-kuns-native-record in
../20_native_seed_selection.md: Rastrigin force, unit reward and diversity
exponents, both standardization floors .1, comparison widths and feature
radii two, h=.04, viscosity .3, configured clone \(\sigma_J=.1\) and
position \(\sigma_x=.1\), actual final-position amplitude
\(s=\sigma_x\sqrt h=.02\), actual OU variance
\(q^2=(1-e^{-.08})/2\), and cap two.
Both viscous normalization tags and complete component-Haar collisions
remain present. The entering restrictions below do not change this kernel.

The accepted one-step source is 20_native_seed_selection.md, SHA256
2200247888963d7c70e9d672291c4a9761be57a7a1dc81b2efe19f80b3afe713.
The definitions of \(w_-,w_+,q_B,q_A,a_g,g_{ff},b,\eta,q,\tau,g\) are
retained from that source. Its standalone interval verifier is
../verify_native_seed.py.

A passive label \(\ell_i\in\{0,1\}\) follows frozen position ancestry:
acceptance/revival inherits the frozen donor's label and persistence retains
the recipient's label. It has no feedback. Put
\[
 L_C(S,\ell)=\sum_i\ell_i a_i{\bf1}_C(x_i),\qquad
 C=[-1/16,1/16]^3.
\]
All counts below include terminal marking. An unlabelled resident landing
in \(C\) is not counted as lineage growth.

## 1. A positive first amplification event, uniform in population size

:::{prf:proposition} Native single-seed amplification probability
:label: prop-kue-seed-positive-amplification

Start with \(N-1\) all-alive unlabelled rows at \(e_1\), one all-alive
labelled row at zero and every stored velocity zero. Then the actual
complete update satisfies
\[
 \Pr\{L_C(S_1,\ell_1)\ge2\}>.0944\qquad(N\ge2).
 \tag{KULN.1}
\]
On this event the target is reached before extinction.

*Proof.* Write \(w=e^{-1/18}\) and \(p=w/(N-2+w)\). For \(N\ge3\), a
resident has a near measurement with probability \(1-p\), proposes the
central source with probability \(p\), and on that event has gate exactly
one. The complete global-normalizer certificate proves this last claim
for every assignment of all other measurements.

Every actual collision velocity is zero for every shared Haar matrix,
and both first-kick viscous fields vanish. A copied central offspring has
its actual unrestricted landing probability \(s_J>.2\). Its event uses
only its own independent jitter, OU and final-position variables. The
seed never accepts, receives no jitter, and has its independent landing
probability \(s_0>.993\). Thus favourable resident events are independent
across rows and of the seed event. This independence uses the identically
clipped gate, rather than an independent replacement for sampled fitness.
Consequently
\[
 \Pr\{L_C(S_1,\ell_1)\ge2\}
 \ge s_0\left[1-\{1-(1-p)p s_J\}^{N-1}\right].
 \tag{KULN.2}
\]
The native copy certificate and \(1-z\le e^{-z}\) give
\[
 (N-1)(1-p)p\ge \frac{2w}{(1+w)^2}>.4996,
\]
\[
 \Pr\{L_C(S_1,\ell_1)\ge2\}
 >.993(1-e^{-.2(.4996)})>.0944245607>.0944.
 \tag{KULN.3}
\]
At \(N=2\), the resident necessarily measures far, and its actual gate is
also identically one. It copies the seed with probability one; both
completed central landings have probability \(s_0s_J>.1986\).
The actual B2 and cap retain their original values and do not change this
positional observable. Since \(C\Subset D\), every counted landing is
alive. \(\square\)
:::

## 2. A source-count growth tail retaining the complete measured array

Use def-kuns-band-class: \(N\ge8\), \(1\le K\le N/8\), \(K\) labelled
all-alive sources in \(B(0,\varepsilon)\), the other \(n_A=N-K\) sources
in \(B(e_1,\varepsilon)\), and all stored velocity norms at most
\(\varepsilon=.001\). Let \(B_n,B_f,A_n,A_f\) be the realized near/far
measurement counts, with \(B_n+B_f=K\) and \(A_n+A_f=n_A\).

The independent lower landing event must erase the entire possible
graph-dependent first-kick displacement. Set \(R_e=1/16-2b\varepsilon\).
The persistent lower probability is
\[
 s_{0,e}=\ell_{R_e,\tau}(g(\varepsilon))^3
 >.9933970>.993,
 \tag{KULN.4}
\]
and the existing own-jitter integral gives
\[
 s_{J,e}=
 \left[\int_{\mathbb R}\ell_{R_e,\tau}(g(\varepsilon+.1z))
                    \phi(z)\,dz\right]^3>.20599>.2.
 \tag{KULN.5}
\]
These events depend only on the row's own Gaussian variables and its
source choice, and imply actual landing for every shared graph/Haar draw:
\(|bU_i|\le2b\varepsilon\). Both sides of the target are eroded in
(KULN.4). A fixed-graph shifted interval probability would not itself
give independent lower trials.

:::{prf:theorem} Native count-growth probability
:label: thm-kue-native-count-growth-tail

Put
\[
 r_B=q_B+.03,\quad r_A=q_A+.03,\quad\kappa=w_+/w_-,
 \quad s_0=.993,\quad s_J=.2,
\]
\[
 h_f=s_0-(s_0-s_J)q_Bg_{ff},\qquad
 h_n=s_0(1-\kappa r_A)-(s_0-s_J)q_B,
\]
\[
 \rho_*=(1-r_B)\left[h_f+\frac78(1-r_A)s_Ja_gw_-\right]
                  +r_Bh_n>1.0327.
 \tag{KULN.6}
\]
Then every entering state in this class satisfies
\[
 \Pr\{L_C(S_1,\ell_1)\ge1.01K\}
 \ge1-e^{-.0018K}-e^{-.0126K}-e^{-.0002494K}.
 \tag{KULN.7}
\]
For \(K\ge10000\), the displayed lower bound exceeds .9174.

*Proof.* Condition on the complete measured array \(\mathbf d\), retaining
its actual shared mean and standard deviation. All fitness gaps in
thm-kuns-band-selection remain valid on this conditioning.
A far-measured central recipient cannot accept a resident and accepts
a central donor with probability at most \(q_Bg_{ff}\). Its lower-trial
probability is at least \(h_f\).

A near-measured central recipient can accept a resident only if that
donor measured far. Its proposal probability for those residents is at
most \(w_+A_f/(w_-n_A)\); its proposal probability for central donors is
at most \(q_B\). Its lower-trial probability is at least
\[
 s_0(1-\kappa A_f/n_A)-(s_0-s_J)q_B.
\]
Each near-measured resident selects each far-measured central donor with
probability at least \(w_-/(N-1)\), accepts with probability at least
\(a_g\), and has its own lower landing probability at least \(s_J\).
Their sum \(T\le L_C(S_1,\ell_1)\) therefore has conditional mean
\[
 \begin{split}
 \mathbb E(T\mid\mathbf d)/K\ge{}&
  (B_f/K)h_f+
  (B_n/K)[s_0(1-\kappa A_f/n_A)-(s_0-s_J)q_B]\\
 &+\frac{A_nB_f}{(N-1)K}s_Ja_gw_- .
 \end{split}
 \tag{KULN.8}
\]
Conditional on \(\mathbf d\), each lower trial depends only on its own
independent donor/gate and Gaussian variables. Uniform erosion removed
the graph and shared Haar variables from those lower events. The
actual completed rows are not being declared independent.

On the good-measurement event
\(\mathcal E=\{B_n/K\le r_B,\ A_f/n_A\le r_A\}\), use
\(n_A/(N-1)\ge7/8\). The right side decreases in both displayed bad-mark
fractions. Its coefficient difference for the \(B_n/K\) monotonicity is
\[
 h_f+\frac78(1-r_A)s_Ja_gw_- -h_n>.3684.
\]
It is consequently at least \(\rho_*\).
The actual individual measurement draws are independent, with
\(\Pr(B\text{ near})\le q_B\) and
\(\Pr(A\text{ far})\le q_A\). Their exponential bounds give
\[
 \Pr(\mathcal E^c)
 \le e^{-2(.03)^2K}+e^{-2(.03)^2n_A}
 \le e^{-.0018K}+e^{-.0126K},
 \tag{KULN.9}
\]
because \(n_A\ge7K\). On \(\mathcal E\), independent Bernoulli lower
trials have mean at least \(1.0327K\), giving
\[
 \Pr(T<1.01K\mid\mathbf d)
 \le\exp\left[-\frac{(1.0327-1.01)^2}{2(1.0327)}K\right]
 \le e^{-.0002494K}.
 \tag{KULN.10}
\]
Both exponential inequalities follow by applying the Bernoulli
moment-generating function and optimizing its parameter. Combining
them gives (KULN.7). \(\square\)
:::

The primitive formulas evaluated at 85 decimal digits of outward interval
precision give
\[
 1.03270977119816<\rho_*<1.03270977119818,\qquad
 .00024948678222<(.0227)^2/[2(1.0327)]<.00024948678223.
\]
The 401-term integrated Gaussian series from the supplied verifier gives
\(.99339703149914<s_{0,e}<.99339703149915\).
No output class condition or later iteration is asserted in this theorem.

## 3. The actual first-generation dispersed joint producer

Let \(s=.02\), \(t=.02\), \(c=e^{-.04}\), and retain \(g,b,\eta,q\).
Put
\[
 m_A=(1-2\eta)e_1,\quad r^2=t^2q^2,\quad
 \alpha=r^2/(1+r^2),\quad w_A=-2ct e_1.
 \tag{KULN.11}
\]
The first resident bulk just before B2 has Gaussian-affine reference
\[
 Y_A=m_A+tq\xi,\qquad W_A=w_A+q\xi,\qquad \xi\sim N(0,I_3).
 \tag{KULN.12}
\]
For the original Gaussian viscous kernel, completing its Gaussian square
gives the exact fields
\[
 A(y)=\mathbb E e^{-|y-Y_A|^2/2}
 =(1+r^2)^{-3/2}e^{-|y-m_A|^2/[2(1+r^2)]},
\]
\[
 \mathbb E[e^{-|y-Y_A|^2/2}W_A]/A(y)
 =w_A+\alpha(y-m_A)/t=:\overline w_A(y).
 \tag{KULN.13}
\]
Thus the two actual reference B2 forces are
\[
 G_{2,\mathrm{count}}(y,w)=\nu A(y)[\overline w_A(y)-w],\qquad
 G_{2,\mathrm{row}}(y,w)=\nu[\overline w_A(y)-w].
 \tag{KULN.14}
\]
These use the pre-final-noise law. Substituting the completed position
law in those weights would change the stage.

For source \(\mu\) and copy indicator \(I\), take independent Gaussian
vectors \(J,\xi,\zeta\) and define
\[
 X=\mu+I(.1)J,\quad W=ctF(X)+q\xi,\quad Y=g(X)+tq\xi,
\]
\[
 V_{\mathfrak n}^{\rm pre}=W+tF(Y)+tG_{2,\mathfrak n}(Y,W),\quad
 V_{\mathfrak n}^+=2V_{\mathfrak n}^{\rm pre}/(2+|V_{\mathfrak n}^{\rm pre}|),
\]
\[
 X^+=Y+s\zeta,\qquad a^+={\bf1}_D(X^+).
 \tag{KULN.15}
\]
Denote these marked joint pushforwards for
\((\mu,I)=(e_1,0),(0,0),(0,1)\) by
\(\nu_{A,\mathfrak n}^0,\nu_{B,\mathfrak n}^0,\nu_{B,\mathfrak n}^J\).
Their inputs are Gaussian; their completed laws retain the nonlinear
force, B2 weights and radial cap. Dead coordinates and capped dead
velocities are retained.

:::{prf:proposition} First-generation Gaussian-input bulk and lineage producer
:label: prop-kue-first-dispersed-joint-producer

For the single-seed entering sequence of Section 1, the actual first
resident bulk converges to \(\nu_{A,\mathfrak n}^0\).
The first-generation full labelled point measure converges to
\[
 \delta_{Z_0}+\mathcal P,\qquad
 Z_0\sim\nu_{B,\mathfrak n}^0,\quad
 \mathcal P\text{ a Poisson point measure of intensity }
 w\nu_{B,\mathfrak n}^J,
 \tag{KULN.16}
\]
with \(Z_0\) independent of \(\mathcal P\).
Projecting onto \(a=1\) retains only actual surviving lineage.

*Proof.* The resident far-measurement count \(k\) has expectation
\((N-1)p_D\le1\). Near residents can accept only the central donor or
one of the \(k\) far-measured residents. Far residents can accept only
the central donor; the seed accepts none. For \(N\ge3\), expected
accepted rows are therefore at most
\[
 \frac{N-1}{N-2+w}(\mathbb E k+w)+p_C\mathbb E k
 \le2+w/(1+w)<2.5.
 \tag{KULN.17}
\]
The \(N=2\) count is one. All initial velocities are zero, so every
actual graph and shared Haar draw gives zero collision velocity.
Conditional copying changes a uniformly bounded expected number of
rows, and their independent Gaussian jitters have uniform moments of
every fixed order. Their empirical moment contribution tends to zero
in \(L^1\). The unchanged resident rows give (KULN.12).
Gaussian-kernel averaging yields count B2 fields (KULN.14) on compact
sets; row denominators converge there to the positive \(A(y)\).
Uniform Gaussian moments localize roots outside those compact sets.
The force and cap are continuous and the terminal boundary has zero
Gaussian mass, giving the marked joint bulk.

Every near resident proposing the seed accepts with probability one.
The discrepancy from giving this rule to every resident central
proposal has probability at most \(p_C\mathbb E k\to0\).
Independent sparse central proposals thus give the Poisson count with
parameter \((N-1)p_C\to w\). Independent Gaussian marks use (KULN.15);
their B2 background has the just-proved deterministic limit.
The persistent seed has its own independent Gaussian marks and the
same limiting deterministic field. This proves (KULN.16). \(\square\)
:::

### The next reward normalizer is explicitly different

The completed resident position marginal is \(N(m_A,\tau^2I_3)\) for
both normalization tags. For one coordinate \(Z\sim N(m,v)\), put
\[
 E_2=m^2+v,\quad E_4=m^4+6m^2v+3v^2,\quad
 C_1=e^{-a^2v/2}\cos(am),\quad
 C_2=\{1+e^{-2a^2v}\cos(2am)\}/2,\quad a=2\pi,
\]
\[
 E_{2c}=e^{-a^2v/2}[(m^2+v-a^2v^2)\cos(am)-2amv\sin(am)].
\]
For \(u(Z)=Z^2+10(1-\cos(aZ))\),
\[
 \mathbb E u=E_2+10(1-C_1),\qquad
 \mathbb E[u^2]=E_4+20(E_2-E_{2c})+100(1-2C_1+C_2).
 \tag{KULN.18}
\]
Independence adds the three coordinate variances. Substituting
\((m,v)=((m_A)_j,\tau^2)\) gives outward enclosures
\[
 1.24356364743449<\mathbb E U<1.24356364743450,\qquad
 .04088023948485<\operatorname{Var}(U)<.04088023948487,
\]
\[
 .22556648573060<\sqrt{\operatorname{Var}(U)+.01}
                         <.22556648573063.
 \tag{KULN.19}
\]
The actual next live reward normalizers condition this marginal on
\(D\). Its correction is bounded, rather than set to zero:
\[
 (2-(m_A)_1)/\tau>49,\qquad
 \beta_A\le6Q(49)\le6\phi(49)/49<2.082\cdot10^{-523}<10^{-520}.
 \tag{KULN.20}
\]
Since \(U(x)\le|x|^2+60\),
\[
 \mathbb E U^4\le8[(|m_A|+\tau\,945^{1/8})^8+60^4]<1.1\cdot10^8.
 \tag{KULN.21}
\]
Cauchy--Schwarz on \(D^c\), followed by division by \(1-\beta_A\),
bounds changes in the first two moments and their variance by
\(10^{-250}\). These fit inside (KULN.19). The marked dead mass is
still retained for mandatory revival.

The next diversity normalizer is the separate exact integral
\[
 \bar d_\nu=\frac1{\nu(a=1)}\int_{a_z=1}\nu(dz)
           \int M_\nu(z,du)d(z,u),
\]
\[
 s_\nu^2=.01+\frac1{\nu(a=1)}\int_{a_z=1}\nu(dz)
           \int M_\nu(z,du)[d(z,u)-\bar d_\nu]^2,
 \tag{KULN.22}
\]
where \(M_\nu\) is the original weighted eligible-alive measurement
law, the original phase-space squashes have radius two, and
\[
 d(z,u)=\sqrt{|\phi_x(x)-\phi_x(x_u)|^2+
                  |\phi_v(v)-\phi_v(v_u)|^2+10^{-6}}.
\]
Here \(\nu=\nu_{A,\mathfrak n}^0\) is the full joint pushforward
(KULN.15), so cap and position--velocity dependence enter (KULN.22).
A positional Gaussian alone does not evaluate this integral.

## 4. A reached inner-basin joint source class

The first-generation output provides a broader source producer than
the original radius-.001 class. Set
\[
 B_{\rm in}=[-.25,.25]^3,\qquad
 \kappa_0=2+40\pi^2,\quad K_0=\kappa_0(1-t^2\kappa_0)>0,\qquad
 H_*(x,v)=\tfrac12K_0|x|^2+\tfrac12|v|^2.
\]
This is a joint comparison price; it does not replace the native
potential or force.

:::{prf:proposition} First-generation inner-basin and joint-energy producer
:label: prop-kue-inner-joint-source-producer

For the single-seed entering state of Section 1, every copied central
offspring, conditional on its actual frozen-source plan, satisfies
\[
 \Pr(x^+\in B_{\rm in})>.9887,\qquad
 \Pr(a^+=0)<10^{-78},\qquad |v^+|<2,
\]
\[
 \mathbb E H_*(x^+,v^+)<4.991.
 \tag{KULN.22a}
\]
The persistent seed has \(\Pr(x^+\in B_{\rm in})>.999\) and
\(\mathbb E H_*(x^+,v^+)<2.208\).
For every \(N\ge2\), the actual complete first update therefore has
\[
 \Pr\{L_{B_{\rm in}}(S_1,\ell_1)\ge2\}>.3867.
 \tag{KULN.22b}
\]
These finite-particle bounds hold for either normalization tag,
without replacing the actual completed velocity by an independent law.

*Proof.* On the first entering configuration, all component velocities
and the first viscous field are identically zero. Conditional on a
copied frozen central source, its actual position is
\(g(.1J)+tq\xi+s\zeta\). The finite Gaussian sum with
\(z_j=-8+j/20\), \(a_j=.1\max(|z_{j-1}|,|z_j|)\), gives
\[
 \Pr(x^+\in B_{\rm in})\ge
 \left[\sum_{j=1}^{320}(\Phi(z_j)-\Phi(z_{j-1}))
                       \ell_{.25,\tau}(g(a_j))\right]^3
 >.98870749967850>.9887.
 \tag{KULN.22c}
\]
At 85 digits of outward precision the same CDF-series/tail certificate
proves this sum. The persistent seed probability is
\(\ell_{.25,\tau}(0)^3>.999\).

Put \(A_0=1-2\eta\), \(A_1=20\pi\eta\), \(\sigma^2=.01\) and \(a=2\pi\).
The unrestricted Gaussian sine moments give
\[
 m_J=\mathbb E[g(.1J_r)^2]
 =A_0^2\sigma^2-2A_0A_1a\sigma^2e^{-a^2\sigma^2/2}
       +\tfrac12A_1^2(1-e^{-2a^2\sigma^2}).
\]
The actual cap, applied after the full B2 field, gives
\(\tfrac12|v^+|^2\le2\) pointwise, including every OU tail.
Consequently
\[
 \mathbb E H_*(x^+,v^+)\le\tfrac32K_0(m_J+\tau^2)+2
 =4.9900305643901\ldots<4.991,
\]
\[
 \mathbb E H_*(x_{\rm seed}^+,v_{\rm seed}^+)
 \le\tfrac32K_0\tau^2+2=2.2079848008968\ldots<2.208.
 \tag{KULN.22d}
\]
These use the actual joint law; the kinetic contribution is its
structural cap bound, not an assumed Gaussian velocity moment.

For the birth death probability, write its position as
\(A_0(.1)J+tq\xi+s\zeta-A_1\sin(.2\pi J)\).
The first three terms are Gaussian of variance
\(A_0^2(.01)+\tau^2\); the last has magnitude at most \(A_1\).
The primitive parameters give
\[
 \frac{2-A_1}{\sqrt{A_0^2(.01)+\tau^2}}>19,
 \qquad \Pr(a^+=0)\le6Q(19)<10^{-78}.
 \tag{KULN.22e}
\]
No tail is removed from the innovation law.
Finally repeat the independent favourable-trial proof of (KULN.2)
with the inner landing floors .999 and .98:
for \(N\ge3\) its probability is at least
\[
 .999(1-e^{-.98(.4996)})>.3867462757>.3867.
\]
At \(N=2\), the actual far gate is one and the two inner landings
have probability greater than \(.999(.98)>.979\).
Every inner landing is alive. \(\square\)
:::

The reached source information in (KULN.22a) includes a full Gaussian
position producer, a true completed cap velocity, an explicit joint
energy budget, and a nonzero retained dead outcome. It is a class of
joint output laws, rather than a compact-support assertion. Conditioning
one such birth on being alive changes its energy upper bound by the
factor \(1/(1-10^{-78})\), so its live conditional price is below 4.992.

## 5. A larger basin producer for arbitrary retained cap velocities

Let \(B_{\rm in}=[-.25,.25]^3\) and
\(B_{\rm basin}=[-.5,.5]^3\). The source hypothesis in this section
is only that a chosen frozen central source belongs to \(B_{\rm in}\).
Every retained entering slot velocity may have the configured norm two.
For every actual component/Haar draw, \(|v_i^{\rm col}|\le4\), and
both first-kick matrices are stochastic, so \(|U_i|\le4\).
Put
\[
 R_{\rm basin,e}=.5-4b=.3431368448678\ldots.
\]
Odd monotonicity of the actual native \(g\) gives
\[
 |g(x_r)|+4b\le g(.25)+4b
 =.3571909836661\ldots<.3572.
 \tag{KULN.23}
\]
For a persistent inner source, the own-noise event
\[
 |g(x_r)+tq\xi_r+s\zeta_r|\le R_{\rm basin,e}\quad(r=1,2,3)
\]
therefore ensures actual basin landing independently of the shared
component/graph variables. Its lower probability is
\[
 p_{\rm basin,0}
 =\ell_{R_{\rm basin,e},\tau}(g(.25))^3
 >1-4\cdot10^{-12}.
 \tag{KULN.24}
\]
For an accepted inner source, its independent own-jitter lower
probability is
\[
 p_{\rm basin,J}
 =\left[\int_{\mathbb R}\ell_{R_{\rm basin,e},\tau}
                    (g(.25+.1z))\phi(z)\,dz\right]^3.
 \tag{KULN.25}
\]
This exact unrestricted integral is approximately .71982507. With
\(z_j=-8+j/20\), \(a_j=\max(|.25+.1z_{j-1}|,|.25+.1z_j|)\),
the finite lower certificate is
\[
 p_{\rm basin,J}\ge
 \left[\sum_{j=1}^{320}(\Phi(z_j)-\Phi(z_{j-1}))
           \ell_{R_{\rm basin,e},\tau}(g(a_j))\right]^3
 >.70894479354078>.7089.
 \tag{KULN.25a}
\]
The same 85-digit outward CDF-series/tail evaluation used in the
incoming verifier certifies this strict inequality. The omitted
Gaussian tails are nonnegative; no innovation law is clipped.
The source may be a donor, including one changed in velocity by its
component's shared rotation. These bounds do not assign it a copied
velocity or discard full OU tails.

The output can lie in the larger basin without lying in
\(B_{\rm in}\). The quantitative inner-source return and its actual
joint velocity budget remain part of the continuation below.

## 6. The reached joint class and the exact continuation operator

Conditional on the complete preparation, OU array and B2 stage, every
raw nonextinct update has
\[
 x_i^+=m_i+s\zeta_i,\qquad
 v_i^+=C_2(v_i^{\rm pre}),\qquad a_i^+={\bf1}_D(x_i^+),
 \tag{KULN.26}
\]
with independent original final noises. Centers, capped velocities
and lineages are functions of the retained earlier stages. This
Gaussian output class covers dispersed sources, dead coordinates and
their actual velocities. For survivor laws, multiply this product
Gaussian density by the nonextinction indicator and normalize by the
actual integrated survival mass; completed conditioned rows are not
declared independent.

In particular the old narrow all-row class is not preserved. For
every nonextinct input, the probability that all completed positions
lie in its two radius-.001 balls is at most
\[
 \left[\frac{(8\pi/3)(.001)^3}{(2\pi s^2)^{3/2}}\right]^N
 <(6.650\cdot10^{-5})^N.
 \tag{KULN.27}
\]
Conditioning before the independent final noises bounds each row's
density by \((2\pi s^2)^{-3/2}\); integration over the two balls and
multiplication proves this assertion. Velocity restrictions only
decrease the probability. This fails that particular class-preserving
inference; it does not obstruct establishment through a larger class.

For a bounded joint source price \(\psi(x,v)\ge0\), put
\(\Psi_N=\sum_i\ell_i a_i\psi(x_i,v_i)\). Its exact complete producer is
\[
 \begin{split}
 \mathsf T_N\Psi_N(S,\ell)
 ={}&\sum_{\mathbf d}P_D(\mathbf d\mid S)
  \sum_P P_{\rm plan}(P\mid S,F(\mathbf d))
  \int H_P(dO)\int\Gamma(dJ,d\xi,d\zeta)\\
 &\quad \sum_i\ell_{L_i(P)}
  {\bf1}_D(X_i^+)\psi(X_i^+,V_i^+).
 \end{split}
 \tag{KULN.28}
\]
The source \(L_i(P)\) is the frozen donor on acceptance/revival and
the recipient on persistence. \(H_P\) uses the actual one shared Haar
matrix per connected component, and \(\Gamma\) the original Gaussian
product law. Both actual viscous fields, both force kicks, cap and
marking remain in \(X_i^+,V_i^+\). The complete measured vector
\(F(\mathbf d)\) retains all shared alive normalizers. Formula
(KULN.28) follows by conditioning in the original stage order.
It includes reverse loss to resident donors, within-source cloning
and mandatory revival; for extinct input it is zero.

The next local inequality is a strictly positive reproduction margin
for this producer on a quantitatively controlled dispersed joint
phase class, for example
\[
 \mathbb E_{\mathcal L_n}\mathsf T_N\Psi_N
 \ge(1+\chi)\mathbb E_{\mathcal L_n}\Psi_N-d_{N,n},
 \qquad\chi>0,
 \tag{KULN.29}
\]
with computed fluctuations and cumulative reverse, exit and price-loss
budgets. The class must cover (KULN.15), (KULN.26) and the allowed
subsequent histories. Existing Keystone source-pressure and regional
signed-flux identities supply the preparation account; the completed
joint response in (KULN.28) still needs evaluation.

The first-generation mean ancestry \(1+e^{-1/18}>1\) is a valid
one-generation producer. It is not a homogeneous branching theorem:
the next reward normalizer has changed in (KULN.19), the diversity
normalizer is (KULN.22), and the new accepted graph changes the actual
component velocity law. In particular the next recipient does not
inherit its donor's frozen velocity.

A proved (KULN.29) with its joint class/exit account is consumed by the
existing target-first and stopped-composition machinery through
\[
 \Pr\{\tau_{C,K_N}\le\tau_{C,1}+b_e(N),\
       \tau_{C,K_N}<\tau_\dagger\wedge\sigma
       \mid\mathcal H_{\tau_{C,1}}\}\ge p_e(N)>0.
 \tag{KULN.30}
\]
The positive producers established here are (KULN.1), (KULN.7), the
full first-generation joint law (KULN.16), next reward normalizers
(KULN.19), the inner joint-source producer (KULN.22a)--(KULN.22e),
and both larger-basin landing bounds (KULN.24)--(KULN.25a).
No narrow-source producer has been iterated through (KULN.27).

## 7. The next actual signed ancestry producer

Write \(z=(x,v,a)\) for a completed marked slot and retain the
normalization tag \(\mathfrak n\). In this section
\(\nu=\nu_{A,\mathfrak n}^0\), \(a_\nu=\nu(a=1)\), and
\(\nu^a=\nu(\cdot\mid a=1)\). All dead coordinates remain in
\(\nu\). Put
\[
 \varphi(z)=(\phi_x(x),\phi_v(v)),\qquad
 D(z,u)^2=|\varphi(z)-\varphi(u)|^2,\qquad
 w(z,u)=e^{-D(z,u)^2/8},
\]
\[
 Z_\nu(z)=\int_{a_u=1}w(z,u)\nu(du),\qquad
 M_\nu(z,du)=\frac{{\bf1}_{a_u=1}w(z,u)\nu(du)}{Z_\nu(z)}.
 \tag{KULN.31}
\]
The record has the same width two for measurement and cloning; hence
the two eligible-companion laws agree in this formula. Their draws
are independent. Let \(\bar R_\nu,S_{R,\nu}\) be the actual
alive reward normalizers from (KULN.19)--(KULN.21), and let
\(\bar d_\nu,s_\nu\) be (KULN.22). The actual limiting fitness mark is
\[
 f_\nu(z,u)=G\!\left(\frac{-U(x)-\bar R_\nu}{S_{R,\nu}}\right)
             G\!\left(\frac{d(z,u)-\bar d_\nu}{s_\nu}\right),
 \quad a_z=1,
 \tag{KULN.32}
\]
and the configured gate is
\(A(f,g)=\min\{1,(g-f)_+/(f+10^{-6})\}\).

An alive labelled source \(z\) first draws its own measurement
\(u\sim M_\nu(z,\cdot)\). That one mark is shared by every recipient
which proposes this source. Conditional on it, define
\[
\begin{split}
 r_\nu(z,u)={}&1
 -\int M_\nu(z,dy)\int M_\nu(y,dv)
        A(f_\nu(z,u),f_\nu(y,v))\\
 &+\int_{a_y=1}\frac{w(y,z)}{Z_\nu(y)}\nu(dy)
          \int M_\nu(y,dv)A(f_\nu(y,v),f_\nu(z,u))\\
 &+\int_{a_y=0}\frac{w(y,z)}{Z_\nu(y)}\nu(dy).
\end{split}
 \tag{KULN.33}
\]
The three nonconstant terms are respectively reverse loss to a
resident donor, accepted incoming alive recipients, and mandatory
revival of dead recipients. No gate is applied to the last term.
Set \(\bar r_\nu(z)=\int M_\nu(z,du)r_\nu(z,u)\) for alive \(z\),
and \(\bar r_\nu(z)=0\) for dead \(z\).

:::{prf:proposition} Actual second-preparation signed source limit
:label: prop-kuln-second-preparation-source-limit

For the single-seed entering sequence of Section 1, let
\(K_{N,2}^{\rm prep}\) be its labelled count after the second frozen
copy/revival plan, before the second collision and kinetics. Then
\[
 \lim_{N\to\infty}\mathbb E K_{N,2}^{\rm prep}
 =\int\bar r_\nu(z)\nu_{B,\mathfrak n}^0(dz)
       +w\int\bar r_\nu(z)\nu_{B,\mathfrak n}^J(dz).
 \tag{KULN.34}
\]
This is a preparation producer. It is not a statement about the
second completed surviving count or about iteration of its source law.

*Proof.* For a realized first output and its complete measured
fitness array, a labelled alive source persists unless its own row
accepts an unlabelled donor. Every alive unlabelled recipient adds
its source label with exactly its proposal probability times the
actual gate. Every dead recipient adds that label with its proposal
probability, through mandatory revival. This is the exact finite
signed source identity. Transfers between labelled rows cancel in
the total labelled count. No independence of source fitness across
its incoming edges is used.

The first labelled point measure and resident bulk converge as in
(KULN.16). Conditional independent measurement draws have bounded
diversity marks and bounded alive rewards, so their empirical mean
and variance registers converge to (KULN.19)--(KULN.22). The floors
\(.1\) make their standardization continuous. Original feature
squashes are continuous and bounded on the entire unbounded position
space. For every alive slot \(|\phi_x(x)|\le3-\sqrt3\), while
\(|\phi_v(v)|\le1\) because \(|v|\le2\). Consequently
\[
 \kappa_a=e^{-(13-6\sqrt3)/2}>.27148,
 \qquad \kappa_d=e^{-4+5\sqrt3/4}>.15962,
 \tag{KULN.35}
\]
are lower bounds on every alive-to-alive and every dead-to-alive
companion weight. The second bound uses
\(|\phi_x(x)|<2\) for an arbitrary retained dead position; it does
not restrict that position. Thus \(Z_\nu(y)\ge\kappa_a a_\nu\)
for alive \(y\), and \(Z_\nu(y)\ge\kappa_d a_\nu\) for dead
\(y\). The actual Gaussian marked law has
\(a_\nu\ge1-10^{-520}>0\). These bounds permit passage to every
companion law and incoming source coefficient. For each finite
collection of rare roots their own measurement marks converge
jointly with the deterministic bulk registers. Each own mark is
retained once, giving (KULN.33), including its dead-recipient term.

For completeness, expectations also converge. Let \(L_N\) count
first accepted recipients and let \(k\) be the first far-measured
resident count. Conditional on \(k\), every accepted recipient
must propose the seed or one of these \(k\) residents. Its independent
proposal probability is at most
\(\min\{1,(k+w)/(N-2+w)\}\). For \(N\ge3\),
\(\mathbb E k\le1\) and \((N-1)/(N-2+w)<2\). Binomial
exponential bounds therefore give, for \(\theta>0\),
\[
 \mathbb E e^{\theta L_N}
 \le \exp\{2(e^\theta-1)+e^{2(e^\theta-1)}-1\}.
 \tag{KULN.36}
\]
At \(\theta=1/4\) the right side is below 3.792. For \(N\ge4\), on
\(L_N\le N/4-1\), at least \(3N/4\) unchanged residents have
independent first completed Gaussian positions with death probability
\(\beta_A<10^{-520}\). Conditioning on the entire first plan
does not affect those kinetic noises. A binomial union bound gives
\[
 \Pr\{n_{a,1}<N/2\}
 \le3.792e^{1/4}e^{-N/16}+2^N\beta_A^{N/4}
 <4.87e^{-N/16}+e^{-298N}.
 \tag{KULN.37}
\]
Integer thresholds only improve the union bound.

The first labelled count is bounded by one plus the number of
resident proposals of the seed. Those proposals are independent
Bernoulli variables with total mean at most one. Thus its
exponential moments are uniformly bounded. On \(n_{a,1}\ge N/2\)
the conditional expected second labelled count is bounded by a
constant times this first count: each source receives at most
\(N/[\kappa_d(n_{a,1}-1)]\) incoming proposals when
\(n_{a,1}\ge2\). This coefficient is bounded for \(N\ge4\).
The finitely many smaller \(N\) do not affect the limit. On the
complement, the second labelled count is at most \(N\), and
(KULN.37) makes its expected contribution vanish. These bounds
give uniform integrability of the conditional expected signed
source sums, each bounded by a constant times the first labelled
count on the good-alive event. They justify taking expectations of
the limiting finite source measure. Its persistent atom and its
Poisson intensity yield exactly (KULN.34). \(\square\)
:::

The last term of (KULN.33) has the explicit bound
\[
 0\le\int_{a_y=0}\frac{w(y,z)}{Z_\nu(y)}\nu(dy)
 \le\frac{1-a_\nu}{\kappa_d a_\nu}
 <6.266\cdot10^{-520}.
 \tag{KULN.38}
\]
It is retained in the exact identity. Its sign is favorable, but
neither the marked law nor its normalizer is replaced by an
all-alive law.

## 8. Quantitative reductions for the changed diversity register

There is an exact simplification of the velocity feature in the
first joint source integral. The configured cap and feature squash
have the same radius two, so
\[
 \phi_v(C_2(w))=\frac{w}{1+|w|}.
 \tag{KULN.39}
\]
This retains the full nonlinear pre-cap velocity from (KULN.15).
It does not replace that velocity by a Gaussian marginal.

Let \(Z,U\) be independent with law \(\nu^a\), put
\(Y=d(Z,U)\), and denote \(M_j=\mathbb E Y^j\). Because
\(w(Z,U)=e^{-(Y^2-10^{-6})/8}\) decreases with \(Y\), the
conditional weighted first and second moments at every root are
no larger than the corresponding unweighted moments. Also
\(1-w\le Y^2/8\) and the conditional weight denominator is at
least \(\kappa_a\). Hence
\[
 [M_1-M_3/(8\kappa_a)]_+\le\bar d_\nu\le M_1,
\]
\[
 s_\nu^2\le .01+M_2-[M_1-M_3/(8\kappa_a)]_+^2.
 \tag{KULN.40}
\]
To prove the lower mean bound at a fixed root, subtract the
weighted mean from the unweighted mean:
\[
 \mathbb EY-\frac{\mathbb E(wY)}{\mathbb Ew}
 =\frac{\mathbb E[(1-w)Y]-\mathbb E(1-w)\mathbb EY}{\mathbb Ew}
 \le\frac{\mathbb EY^3}{8\kappa_a}.
\]
Average over alive roots. Monotonicity gives both upper bounds,
and subtracting the squared lower mean proves the variance bound.

These estimates can be computed from full single-slot joint feature
moments. Write \(Q=\varphi(Z)-\mathbb E\varphi(Z)\),
\(a_2=\mathbb E|Q|^2\), \(a_4=\mathbb E|Q|^4\), and
\(\Sigma=\mathbb E QQ^\top\). For \(\delta=10^{-3}\),
\[
 M_2=2a_2+\delta^2,
\]
\[
 M_4=2a_4+2a_2^2+4\operatorname{tr}(\Sigma^2)
                +4\delta^2a_2+\delta^4,
 \tag{KULN.41}
\]
\[
 M_1\ge\frac{M_2^{3/2}}{\sqrt{M_4}},\qquad
 M_3\le\sqrt{M_2M_4}.
 \tag{KULN.42}
\]
Expanding \(|Q_1-Q_2|^4\) proves (KULN.41), using the mean-zero
independence of the two copies; Cauchy--Schwarz proves the second
inequality in (KULN.42). Interpolation
\(\|Y\|_2\le\|Y\|_1^{1/3}\|Y\|_4^{2/3}\)
proves the first. All cross covariances of the position and velocity
features remain in \(\operatorname{tr}(\Sigma^2)\). Dropping them
would change this account. Formula (KULN.15), alive marking, and
(KULN.39) make every moment here an explicit original Gaussian-input
integral with bounded integrand and its full marked-dead complement.

The exact next preparation source is now (KULN.34), with (KULN.33)
and the changed registers above. A positive numerical lower bound
for that source integral, a quantitative finite-\(N\) error, and
the subsequent complete response (KULN.28) are separate inequalities.
Neither first-generation positive growth nor a diagnostic numerical
evaluation establishes these three inequalities by itself.

### The reference covariance reduction retains the OU correlations

For the resident reference, the exact native pre-cap velocity in
(KULN.15) can also be written coordinate by coordinate as
\[
 V_j^{\rm pre}(\xi)=(w_A)_j-2t(m_A)_j
       +q[1-2t^2-\Lambda_{\mathfrak n}(\xi)]\xi_j
       -20\pi t\sin\{2\pi[(m_A)_j+tq\xi_j]\},
\]
\[
 \Lambda_{\rm count}(\xi)
  =t\nu(1-\alpha)(1+r^2)^{-3/2}e^{-\alpha|\xi|^2/2},\qquad
 \Lambda_{\rm row}(\xi)=t\nu(1-\alpha).
 \tag{KULN.43}
\]
Indeed \(\overline w_A(Y_A)-W_A=-(1-\alpha)q\xi\);
substitution of the original force proves the identity. Thus all
unconditioned velocity-feature moments use just this three-dimensional
Gaussian integral followed by (KULN.39), rather than an independent
Gaussian velocity replacement. In particular the position and velocity
use the same \(\xi\).

Let \(p=\tau^2/(1-\beta_A)\),
\(v_j=\operatorname{Var}_{\nu^a}(\phi_{v,j})\), and
\(b_4=\mathbb E_{\nu^a}|\phi_v-\mathbb E_{\nu^a}\phi_v|^4\).
The joint marked law is invariant under simultaneous reflection of
position and velocity in coordinate 2, and separately in coordinate 3,
as well as their interchange. This follows directly from (KULN.43),
the shared OU/final-noise formula, the radial squashes and the symmetric
alive box. Hence \(v_2=v_3\), all three covariance blocks are diagonal,
and
\[
 v_1+2v_2\le a_2\le3p+v_1+2v_2,
\]
\[
 a_4\le
 \left\{\frac{\sqrt{60}\,\tau^2}{1-\beta_A}+\sqrt{b_4}\right\}^2,
 \qquad
 \operatorname{tr}(\Sigma^2)
       \le(p+v_1)^2+2(p+v_2)^2.
 \tag{KULN.44}
\]
For the positional variance estimate, the original radius-two squash
has Jacobian norm at most one. Expanding a smooth approximation of
each coordinate in Gaussian Hermite polynomials gives
\(\operatorname{Var}(f(X))\le\tau^2\mathbb E|\nabla f(X)|^2\)
for \(X\sim N(m_A,\tau^2I_3)\): every nonconstant Hermite term has
degree at least one, whereas the gradient norm multiplies its square
coefficient by its degree. Approximation in Gaussian Sobolev norm
extends this inequality to the squash. Each raw positional coordinate
variance is therefore at most \(\tau^2\). Conditioning on alive
and minimizing over the centering constant gives the bound \(p\).
For the fourth moment, conditional Jensen and the Lipschitz squash give
\[
 \mathbb E_{\nu^a}|\phi_x-\mathbb E_{\nu^a}\phi_x|^4
 \le \mathbb E_{\nu^a\otimes\nu^a}|X-X'|^4
 \le\frac{60\tau^4}{(1-\beta_A)^2}.
\]
The triangle inequality in \(L^2\) for the sum of the squared
position/velocity feature deviations gives the bound on \(a_4\).
Reflection gives diagonal covariance blocks. Cauchy--Schwarz bounds
the squared position--velocity covariance in each diagonal pair by
its two variances, yielding the stated trace bound. Those diagonal
cross covariances are bounded explicitly, rather than dropped.

The raw velocity feature satisfies \(|\phi_v|\le1\). For every
bounded raw velocity-feature integrand \(h\), its alive correction is
\[
 \left|\mathbb E_{\nu^a}h-\mathbb E_\nu h\right|
 \le\frac{2\|h\|_\infty\beta_A}{1-\beta_A}.
 \tag{KULN.45}
\]
This follows by writing the conditional expectation as the raw
expectation minus its actual dead integral, then dividing by
\(1-\beta_A\). Equations (KULN.43)--(KULN.45) reduce conservative
bounds for the changed diversity register to bounded
three-dimensional original-OU integrals and an explicit marked-dead
correction. A numerical receipt for those integrals has not been
substituted for their proof.

## 9. Analytic native source pressure on the reached joint law

The imports are {prf:ref}`prop-kusp-native-mean-fitness`, which proves
\(M^{-1}\sum_{i:a_i=1}F_i\le1.614\) for every complete actual
measurement array, and {prf:ref}`prop-kusp-signed-native-source-pressure`.
The first bound includes both sampled global normalization registers,
their patched variances and ties. No favorable event is imposed on
every other row's measurement.

### Actual changed register bounds

Use the explicit dead-mass upper bound \(\beta_0=10^{-520}\).
Every alive correction below uses that upper bound.

Set \(\lambda=t\nu(1-\alpha)(1+r^2)^{-3/2}\),
\(B=1-2t^2-\lambda\), \(H=40\pi^2t^2\),
\(a=2\pi tq\), and \(\theta=2\pi(m_A)_1\). The exact
(KULN.43) has diagonal comparison Jacobian
\(q[B-H\cos(2\pi(m_A)_j+a\xi_j)]\). Its mean squared
Frobenius norm is
\[
 J_0=q^2\left\{3B^2-2BH e^{-a^2/2}(\cos\theta+2)
 +\frac{H^2}{2}[3+e^{-2a^2}(\cos2\theta+2)]\right\}.
 \tag{KULN.46}
\]
The remaining count Jacobian is
\(q[(\lambda-\Lambda)I+\alpha\Lambda\xi\xi^\top]\).
Using \(0\le\lambda-\Lambda\le\lambda\alpha|\xi|^2/2\)
and \(\Lambda\le\lambda\), its \(L^2\) Frobenius norm is
at most \(J_e=q\lambda\alpha(1+\sqrt3/2)\sqrt{15}\).
The cap-feature composition (KULN.39) is 1-Lipschitz. Gaussian
Poincare, proved by the Hermite expansion in Section 8, therefore gives
\[
 \operatorname{tr}\operatorname{Cov}_{\nu}(\phi_v)
 \le(\sqrt{J_0}+J_e)^2<.080472807569456596.
 \tag{KULN.47}
\]
For row normalization \(\lambda_{\rm row}=t\nu(1-\alpha)\ge\lambda\),
the radial correction vanishes. Also
\(1-2t^2-\lambda_{\rm row}-H>0\). Every positive diagonal
derivative decreases, pointwise, so (KULN.47) also bounds that mode.

Combining this estimate with the actual positional variance and dead
correction, put
\(A_2=[(\sqrt{J_0}+J_e)^2+3\tau^2]/(1-\beta_0)\).
The decreasing-weight argument (KULN.40) gives
\[
 \bar d_\nu\le\sqrt{2A_2+10^{-6}}<.404275741942611773,
 \qquad
 s_\nu\le\sqrt{.01+2A_2+10^{-6}}<.416459932674500186.
 \tag{KULN.48}
\]
These are bounds for the new actual register.

For the resident velocity-feature mean, remove the first-coordinate
constant and phase shift in (KULN.43). The remaining vector is odd
under \(\xi\mapsto-\xi\), so its radial cap feature has mean zero.
With \(\delta=2\pi[1-(m_A)_1]\), the actual first coordinate
differs by
\[
 (w_A)_1-2t(m_A)_1+
       40\pi t\sin(\delta/2)\cos(a\xi_1-\delta/2).
\]
The Lipschitz composition and (KULN.45) give
\[
 |\mathbb E_{\nu^a}\phi_v|
 \le-(w_A)_1+2t(m_A)_1+40\pi t\sin(\delta/2)
                    +2\beta_0/(1-\beta_0)
 =:m_v<.090754200463804027.
 \tag{KULN.49}
\]
This argument holds for both modes, including the count radial
factor, which is even in \(\xi\). For the position squash
\(f(r)x=2x/(2+r)\), its Hessian directional norm is at most
\(3|f'(r)|+r|f''(r)|=(12+10r)/(2+r)^3\le3/2\).
Its Jacobian is Lipschitz across zero. Taylor expansion and the
actual centered positional Gaussian thus give
\[
 |\mathbb E_{\nu^a}\phi_x|
 \le\frac{2(m_A)_1}{2+(m_A)_1}+rac94\tau^2
                   +4\beta_0/(1-\beta_0)
 =:m_x<.666903730007655400.
 \tag{KULN.50}
\]
No position--velocity covariance has been dropped from the joint law.

### A favorable source in the reached joint law

Set
\[
 r_* =\sqrt{.87/(1+20\pi^2)},\quad z_*=2r_* /(2+r_*),\quad
 \mathcal G=\{|x|\le r_*,\ |\phi_v(v)|\le.45\}.
 \tag{KULN.51}
\]
Here \(z_*<.064098906634716637\), and
\(U(x)\le(1+20\pi^2)|x|^2\) makes \(U\le.87\) on
\(\mathcal G\). For its unweighted eligible-alive companion degree,
Jensen and the two actual means and variance traces prove
\[
 d_\nu(z)=\int w(z,u)\nu^a(du)
 \ge\exp\!\left[-\frac18\left\{
 (m_x+z_*)^2+\frac{3\tau^2}{1-\beta_0}
 +(m_v+.45)^2+
       \frac{(\sqrt{J_0}+J_e)^2}{1-\beta_0}\right\}\right]
 >.892648527120498059.
 \tag{KULN.52}
\]
The exact mean squared feature distance is the sum of its squared
distances to the two means and their two variance traces, accounting
for all four displayed terms.

If its own sampled measurement companion has \(|x_u|\ge.85\), then
\[
 d(z,u)\ge2(.85)/(2+.85)-z_*>.532392321435458802.
\]
Using (KULN.19), (KULN.48) and monotonicity of \(G\) gives
\[
 f_\nu(z,u)\ge f_*>2.22894126182232251.
 \tag{KULN.53}
\]
Here \(f_*\) is the exact product of the two lower \(G\) factors,
using the reward endpoints from (KULN.19), the exact right sides of
(KULN.48), and the lower measured distance above. Write \(d_*\)
for the exact exponential lower bound in (KULN.52). These constants
are defined before any displayed decimal rounding.
The standardized diversity numerator is positive; its upper standard
deviation therefore gives the correct lower bound. For the actual
weighted measurement law,
\[
 \Pr\{|x_u|<.85\mid z\}
 \le\frac{Q(7.28)}{\kappa_a(1-\beta_0)}
 \le\frac{\phi(7.28)}{7.28\kappa_a(1-\beta_0)}
 <6.261\cdot10^{-13}.
 \tag{KULN.54}
\]
Indeed \(((m_A)_1-.85)/\tau>7.2829\), and
\(|x_u|<.85\) implies \((x_u)_1<.85\). The eligible denominator
is at least \(\kappa_a(1-\beta_0)\). This accounts for one source's
own sampled mark; it does not condition the other residents on
favorable marks.

The persistent first source has \(X^+=tq\xi+s\zeta\). On
\(|\xi|\le4.9\), use \(1-y^2/6\le\sin(y)/y\le1\)
in its exact native force. Both B2 modes have \(A\le1\), and
the reference mean shift has magnitude
\(h_0=-(w_A)_1+\alpha(m_A)_1/t\). Consequently
\[
 |V^{\rm pre}|
 \le q(4.9)\{1-t^2(2+40\pi^2)+H[a(4.9)]^2/6\}+t\nu h_0
 <.808846675386233782.
 \tag{KULN.55}
\]
The scalar multiplying each \(q\xi_j\) is positive, since
\(1-t^2(2+40\pi^2)-t\nu>.8352\). The upper coefficient
therefore bounds its absolute contribution, and (KULN.39) gives
the actual velocity-feature norm below .45. All thermostat tails
remain in the law.

For a standard three-dimensional Gaussian, let
\(T_3(u)=2Q(u)+\sqrt{2/\pi}\,u e^{-u^2/2}\), its exact radial
tail. A union bound retaining the shared \(\xi\) gives
\[
 \nu_{B,\mathfrak n}^0(\mathcal G)
 \ge1-T_3(r_*/\tau)-T_3(4.9)>.985597926409916760.
 \tag{KULN.56}
\]
The reference mean-fitness and signed-source theorems give a source
excess larger than
\[
 \delta_*=(f_*d_*-1.614/d_*)/(f_*+10^{-6})
                  >.0814550807578887.
\]
Since \(r_\nu(z,u)\ge0\) everywhere and is at least
\(1+\delta_*\) on the good source/own-mark event,
\(\bar r_\nu(z)\ge(1-6.261\cdot10^{-13})(1+\delta_*)\)
for \(z\in\mathcal G\). Integrating the good source probability proves
\[
 \int\bar r_\nu(z)\nu_{B,\mathfrak n}^0(dz)>1.0658798850997769.
 \tag{KULN.57}
\]
The verifier uses the exact formulas before displayed rounding.
Mandatory dead revival remains in (KULN.33) and is favorable.
No favorable multiplier is assigned to the separate jittered source
law; its contribution to (KULN.34) is nonnegative.

## 10. A uniform finite-population complete source producer

The target cube belongs to the actual central potential basin.
For \(0<x\le.5\), \(2x+20\pi\sin(2\pi x)>0\), so the
native force is negative and its first positive unstable barrier
is larger than .5. Oddness gives the negative side. This identifies
the spatial target; every stochastic exit remains in the estimates.

:::{prf:theorem} Measured favorable-source basin offspring
:label: thm-kuln-measured-native-source-basin-producer

Fix any actual entering state and complete current measurement array
for the native record, with \(M\ge1000\) alive slots. Suppose an
alive source \(j\) satisfies
\[
 |x_j|\le r_*,\qquad F_j\ge2.2,\qquad
 \frac1{M-1}\sum_{i:a_i=1,\ i\ne j}w(i,j)\ge.89.
 \tag{KULN.58}
\]
Every stored slot velocity has its original cap two; dead positions
are arbitrary. Keep cloning proposals, acceptance uniforms, full
noises and component/Haar draws unconditioned. Let \(Y_j^+\) count
immediate frozen-source descendants of \(j\) that complete this
update in \([-.5,.5]^3\). For both native normalization modes,
\[
 \mathbb E[Y_j^+\mid S,\mathbf d]>1.023,
 \qquad\Pr\{Y_j^+\ge2\mid S,\mathbf d\}>.0016.
 \tag{KULN.59}
\]
These constants are independent of the total slot number and of \(M\).
The 320-bin Gaussian lower sum sharpens them to
\(\mathbb E Y_j^+>1.061\) and \(\Pr\{Y_j^+\ge2\}>.0017\).

*Proof.* The pathwise mean-fitness and signed-source imports give
the exact alive preparation excess
\[
 \Delta_j\ge
 \frac{(2.2)(.89)-1.614(1000/999)/(.89)}{2.2+10^{-6}}
 =:\delta_M>.06486431391989019.
 \tag{KULN.60}
\]
The source lower expression increases with source fitness and degree.
Its excluded-row fitness average is at most \(1.614M/(M-1)\),
whose largest bound here occurs at \(M=1000\). Mandatory dead
revivals contribute nonnegative incoming descendants.

The actual component velocity is
\(\bar v+(1/2)O(v_i-\bar v)\), so its norm is at most four.
The original first-kick viscous matrix is stochastic in both modes,
and hence \(|U_i|\le4\). This includes rotated donors. A source
descendant's exact position is
\[
 g(x_j+I_i(.1)J_i)+bU_i+tq\xi_i+s\zeta_i.
\]
Its own combined noise \(Z_i=(tq\xi_i+s\zeta_i)/\tau\) is a
standard Gaussian, independent of its own jitter and pre-kinetic
donor/gate choices. Its actual \(U_i\) may depend on its own
jitter and all shared component variables, but the bound holds
for all those outcomes. Monotonicity of \(g\) and the primitive
strict margin
\[
 .5-g(r_*+.25)-4b-3.4\tau>.00319237132867730
\]
make \(\|J_i\|_\infty\le2.5,\ \|Z_i\|_\infty\le3.4\)
sufficient for basin landing of accepted descendants. For persistent
descendants only the second event is needed. Thus
\[
 p_0=[2\Phi(3.4)-1]^3>.997979786355910282,
 \quad p_J=[2\Phi(2.5)-1]^3p_0>.961256936349641733.
 \tag{KULN.61}
\]
These are lower events in unrestricted original Gaussian laws;
their complements retain nonnegative contributions. If
\(q_{\rm out}\) is the source's actual outgoing acceptance
probability and \(\lambda_{\rm in}\) its alive incoming preparation
mean, then
\[
 \mathbb E Y_j^+\ge p_0(1-q_{\rm out})+p_J\lambda_{\rm in}
 \ge p_J(1+\Delta_j)>1.02360820802669679.
\]
Here \(\Delta_j=\lambda_{\rm in}-q_{\rm out}\); the entire
original measurement array stays frozen.

For the probability bound, use only alive incoming recipients. Each
own proposal of \(j\), own gate and own jitter/Gaussian event forms
an independent Bernoulli lower event. Shared component and first
viscosity changes have already been removed from its sufficient
event by the uniform shift bound. Their mean sum is at least
\(\lambda=p_J\delta_M\), since
\(\lambda_{\rm in}=\Delta_j+q_{\rm out}\ge\delta_M\).
Each probability is at most
\(p_{\max}=1/[\kappa_a(999)]<.003688\), because each alive
recipient denominator is at least \(\kappa_a(M-1)\).
For independent Bernoulli probabilities bounded by \(p_{\max}\),
with actual mean \(L\ge\lambda>p_{\max}\),
\[
 \Pr\{\text{sum}\le1\}
 =\prod_i(1-p_i)\left[1+\sum_i\frac{p_i}{1-p_i}\right]
 \le e^{-L}\left[1+\frac{L}{1-p_{\max}}\right].
\]
The complement bound increases for \(L>p_{\max}\). Evaluation
at \(\lambda\) gives more than .00164809826735124 for two incoming
basin births, hence for \(Y_j^+\ge2\).

For the stronger lower sum, put \(R_e=.5-4b\) and use the original
one-coordinate producer
\[
 p_{J,\mathrm{strong}}=
 \left[\int\ell_{R_e,\tau}(g(r_*+.1z))\phi(z)\,dz\right]^3.
\]
The 320-bin endpoint construction (KULN.25a) gives
\(p_{J,\mathrm{strong}}>.996523592882837796\).
It covers every source coordinate in \([-r_*,r_*]\):
\(y\mapsto\ell_{R_e,\tau}(g(y))\) is even and decreases with
\(|y|\), since \(g\) is odd and increasing. Its convolution
with the centered jitter Gaussian also decreases with the absolute
source offset. A layer-cake decomposition into symmetric intervals
proves that assertion: each such interval's shifted Gaussian mass
decreases with the offset. Thus its smallest source probability is
attained at \(r_*\), the offset used in the lower sum.
Persistent landing exceeds \(1-3.9\cdot10^{-15}\). Replacing
\(p_J\) by this lower sum in the two preceding arguments proves
the sharper constants. The original Gaussian tails remain
nonnegative. \(\square\)
:::

The reached first joint law and this full conditional kinetic bound
also imply the actual two-update producer
\[
 \liminf_{N\to\infty}\mathbb E[
  \text{first central lineage survivors in }[-.5,.5]^3
                       \text{ after update two}]>1.0621744526811760.
 \tag{KULN.62}
\]
Use only the persistent first atom. Equations (KULN.54)--(KULN.57)
give its good-event probability and limiting signed excess. The
strong complete own-event is uniform over all actual second graphs,
both first-viscosity fields and every Haar draw. Section 7's first
convergence and expectation argument therefore gives the lower
product
\(\nu_B^0(\mathcal G)(1-6.261\cdot10^{-13})
  p_{J,\mathrm{strong}}(1+\delta_*)\).
Other labelled sources contribute nonnegative completed survivors.
No independence of second completed rows is asserted.

The accompanying `../verify_native_source_pressure.py` uses
85-digit outward intervals, exact Gaussian trigonometric moments,
the certified CDF series and finite lower sums. Its conditional
finite producer is independent of \(N\). A quantitative finite-\(N\)
error for the first reached reference and a continuation class
preserving favorable source pressure are separate from the proved
measured-class producer (KULN.59).

## 11. Weighted joint source quality and its exact continuation test

The preceding basin producer gives an actual spatial descendant
count. Subsequent fitness still uses each descendant's actual
potential, its velocity feature, its own sampled diversity and the
whole new alive population's two normalizers. A weighted joint
source test retains that information without assigning the basin
count the same reproduction factor on the next update.

For \(c>0\), define the analytic prices
\[
 \psi_c^U(x)=e^{-cU(x)},\qquad
 \psi_c^H(x,v)=e^{-c[U(x)+|v|^2/2]}.
 \tag{KULN.63}
\]
They do not alter the configured fitness or the algorithm's
normalization. At a completed state, the alive lineage price is
\(\Psi_c=\sum_i\ell_i a_i\psi_c(x_i,v_i)\). The full marked
unweighted population, including dead coordinates and capped dead
velocities, must also be retained to evaluate its next selection law.
More generally write
\(\psi_{c,\omega}(x,v)=e^{-c[U(x)+\omega|v|^2/2]}\)
for \(0\le\omega\le1\). The formulas below for a joint price
replace the velocity exponent by \(\omega\) times that exponent.

### Exact native Gaussian heat response

For one coordinate, let
\(u(x)=x^2+10[1-\cos(2\pi x)]\), and define
\[
 L_{c,\sigma}^D(m)=\int_{-2}^2 e^{-cu(x)}
                 \frac{e^{-(x-m)^2/(2\sigma^2)}}{\sqrt{2\pi}\sigma}\,dx.
 \tag{KULN.64}
\]
This is the original noise integrated against the original potential
and killing mark. Its unrestricted counterpart has an explicit
uniformly convergent Fourier expansion. Put
\(A=1+2c\sigma^2\), \(z=10c\), and
\[
 I_k(z)=\sum_{j\ge0}\frac{(z/2)^{2j+k}}{j!(j+k)!}.
\]
Expanding the two exponentials in
\(e^{z\cos t}=e^{(z/2)e^{it}}e^{(z/2)e^{-it}}\)
and grouping their Fourier coefficients proves
\[
\begin{split}
 L_{c,\sigma}^{\mathbb R}(m)
 ={}&\frac{e^{-10c-cm^2/A}}{\sqrt A}\left\{
 I_0(z)+2\sum_{k\ge1}I_k(z)
      e^{-2\pi^2k^2\sigma^2/A}\cos(2\pi k m/A)\right\}.
\end{split}
 \tag{KULN.65}
\]
Completing the Gaussian square gives the tilted mean \(m/A\)
and variance \(\sigma^2/A\), proving the termwise Gaussian
integrals. Uniform convergence follows from the positive coefficient
sum \(e^z\). Since \((j+k)!\ge k!\),
\(I_k(z)\le e^{z^2/4}(z/2)^k/k!\). Truncation after \(K\),
when \(z/[2(K+2)]<1\), has uniform error at most
\[
 \frac{e^{-10c-cm^2/A}}{\sqrt A}
 \frac{2e^{z^2/4}(z/2)^{K+1}}
      {(K+1)![1-z/(2(K+2))]}.
 \tag{KULN.66}
\]
For \(c=1/4,K=40\), the full uniform error, including the
prefactor \(e^{-10c}/\sqrt A\), is less than
\(2.765\cdot10^{-45}\). The actual killed response satisfies
\[
 0\le L_{c,\sigma}^{\mathbb R}(m)-L_{c,\sigma}^D(m)
 \le Q((2-m)/\sigma)+Q((2+m)/\sigma).
 \tag{KULN.67}
\]
Indeed the integrand's price is at most one. The actual outside
outcome is therefore computed or bounded explicitly rather than
discarded. These formulas apply to arbitrary unbounded means.

### The complete joint producer on an arbitrary reached history

Fix the current full physical state and its complete measurement
array. Draw the original frozen-source plan \(P\), all original
jitters, and its actual component/Haar matrices. Let \(X_i\) be
its prepared source position and \(U_i\) the actual first viscous
collision velocity. At this point, before OU, both are fixed; they
include the dependence on every prepared row and its component.
The exact pre-final position and actual completed velocity are
\[
 Y_i=g(X_i)+bU_i+tq\xi_i,\qquad
 V_i^+=C_2[\text{actual second-kick velocity from all }\xi].
 \tag{KULN.68}
\]
The terminal position noise alone gives the following exact joint
weighted descendant measure for a current alive source \(j\):
\[
\begin{split}
 \mathcal Q_{c,N}^j(dx,dv)={}&
 \mathbb E_{P,H,J,\boldsymbol\xi}\sum_{i:L_i(P)=j}
  {\bf1}_D(x)e^{-c[U(x)+|V_i^+|^2/2]}\\
 &\quad\times\varphi_{sI_3}(x-Y_i)\,dx\,
                  \delta_{V_i^+}(dv).
\end{split}
 \tag{KULN.69}
\]
The expectation uses the actual plan probabilities from the complete
current fitness array. The second field uses every OU row; the shared
OU contribution to \(Y_i\) and \(V_i^+\) has not been factorized.
The original cap remains in \(V_i^+\). Its mass is exactly
\[
 \mathcal K_{c,N}^{H,j}(S,\mathbf d)=
 \mathbb E_{P,H,J,\boldsymbol\xi}\sum_{i:L_i(P)=j}
 e^{-c|V_i^+|^2/2}\prod_{r=1}^3L_{c,s}^D((Y_i)_r).
 \tag{KULN.70}
\]
For the potential price alone, its OU contribution can also be
integrated, since the second kick does not change the position:
\[
 \mathcal K_{c,N}^{U,j}(S,\mathbf d)=
 \mathbb E_{P,H,J}\sum_{i:L_i(P)=j}
    \prod_{r=1}^3L_{c,\tau}^D((g(X_i)+bU_i)_r).
 \tag{KULN.71}
\]
These identities hold for every reached history, both normalization
modes, all clone/persistence/revival outcomes and all original noise
tails. They follow by conditioning in the original stage order.
For extinct input the source sums are zero. A dead current source
has no eligible incoming source event, and its own revival adopts
its donor's label; its current price is zero. Dead *recipients*
remain in the sum through their actual mandatory donor choice.

The full next marked population law accompanies (KULN.69). The
weighted measure is an analytic source test, not a substitute for
that full law. In particular next sampled companion measures and
both next shared normalizers must be formed from the actual
unweighted next population, not from a normalization of
\(\mathcal Q_{c,N}^j\).

### Quantitative positive initial prices

These prices have a positive producer for the retained single-seed
initial condition. Let
\(A_0=1-2\eta,A_1=20\pi\eta,a_0=2\pi,v_J=.01\).
The exact original Gaussian clone computation is
\[
 G_2=A_0^2v_J-2A_0A_1a_0v_Je^{-a_0^2v_J/2}
       +\frac{A_1^2}{2}(1-e^{-2a_0^2v_J}).
\]
The persistent and copied source potential means satisfy
\[
 E_0=3[\tau^2+10(1-e^{-2\pi^2\tau^2})]
     <.246216715990245338,
\]
\[
 E_J\le3(1+20\pi^2)(G_2+\tau^2)
     <3.554117616381229228.
 \tag{KULN.72}
\]
The first equality uses the Gaussian cosine expectation. The second
uses the unchanged global bound
\(U(x)\le(1+20\pi^2)|x|^2\), so it includes every clone-jitter
and thermostat tail. Actual first central source death is at most
\(\beta_J=10^{-78}\), as proved in (KULN.22a)--(KULN.22e).
Jensen and subtraction of that actual dead mass give, at \(c=1/4\),
\[
 \mathbb E\Psi_c^{U,+}
 \ge(e^{-cE_0}-\beta_J)+.4996(e^{-cE_J}-\beta_J)
 >1.14576754854288369,
 \qquad N\ge2.
 \tag{KULN.73}
\]
The source starts with price one; the uniform clone gain .4996 is
the original complete-binomial result of Section 1. Conditional on
its first source plan, all collision and first-viscosity velocities
are zero, so the actual positional producers used here are exact
for every \(N\) and both modes.

There is also a joint-price producer for every \(N\) and both modes.
The actual stored velocity cap gives \(|v|^2/2\le2\) pointwise,
including on every unbounded OU outcome. Thus
\[
 \psi_{c,\omega}(x,v)\ge e^{-2c\omega}\psi_c^U(x),\qquad
 \mathbb E\Psi_{1/4,\,1/100}^+
 \ge e^{-.005}\mathbb E\Psi_{1/4}^{U,+}
 >1.140053009054176683,
 \quad N\ge2.
 \tag{KULN.73a}
\]
The seed's initial joint price is one. The velocity weight \(1/100\)
is an analytic test coefficient; the actual velocity law, cap and
next diversity feature remain the configured ones. This estimate
uses the pointwise cap, without a new bound on a random row-normalized
viscous velocity moment.

The first reference also has a positive *joint-energy* price. Put
\(P_0=q(1-2t^2),Q_0=20\pi t,a=2\pi tq\). Its central
pre-B2 velocity has mean squared norm
\[
 E_{m v,base}=3[P_0^2-2P_0Q_0a e^{-a^2/2}
                     +Q_0^2(1-e^{-2a^2})/2].
\]
The source reference field subtracts a nonnegative scalar multiple
of \(q\xi\) and adds a vector of norm at most \(t\nu h_0\).
The native scalar before subtraction is at least
\(1-t^2(2+40\pi^2)\), and remains positive after subtraction
in both modes. The cap can only decrease the norm. Thus
\[
 \mathbb E|V_0^+|^2/2
 \le\tfrac12(\sqrt{E_{m v,base}}+t\nu h_0)^2
 <.040883412025297396.
\]
Copied stored kinetic energy is at most two, with all tails retained.
Jensen for the full joint price therefore gives
\[
 \int\psi_c^H\,d(\nu_B^0+w\nu_B^J)\big|_{a=1}
 \ge e^{-c(E_0+.040883412025297396)}-\beta_J
     +w[e^{-c(E_J+2)}-\beta_J]>1.16670215148441925.
 \tag{KULN.74}
\]
This is a first-reference producer. Its velocity estimate is not
asserted for an arbitrary finite row-normalized field.

The potential price is globally Lipschitz with bound
\[
 \sup|\nabla e^{-cU}|
 \le\sqrt{2c/e}+20\pi\sqrt3\,c.
 \tag{KULN.75}
\]
Use \(U\ge|x|^2\) and
\(|\nabla U|\le2|x|+20\pi\sqrt3\).
At \(c=1/4\) this bound is below 27.636. For the joint price on
actual capped velocities, its velocity derivative norm is at most
\(2c\). The finite-reference account carries these prices with its
separate terminal-mark discrepancy charge. On common alive states,
the inverse position squash has Lipschitz constant at most
\((1+\sqrt3)^2\), and the inverse velocity squash at most four.
Consequently its feature-error transfer uses
\[
 |\psi_{c,\omega}(x,v)-\psi_{c,\omega}(x',v')|
 \le(1+\sqrt3)^2L_{\rm pot}|\phi_x(x)-\phi_x(x')|
            +4(2c\omega)|\phi_v(v)-\phi_v(v')|.
\]
A mismatched terminal mark adds at most one per source-price pair;
both dead prices are zero. This retains the actual mark account
without a new noise truncation.
The primitive interval receipt is `../verify_native_weighted_source.py`.

### Exact remaining quality replacement

For a continued lineage and either price, the actual one-update
quality difference is
\[
 \mathcal R_{c,N}(S,\mathbf d,\ell)
 =\sum_{j:a_j=1}\ell_j\mathcal K_{c,N}^{\cdot,j}(S,\mathbf d)
                    -\sum_j\ell_ja_j\psi_c(x_j,v_j).
 \tag{KULN.76}
\]
Before the measurement, average this expression against the entire
original measurement-array law. A continued positive weighted
producer requires a lower bound for (KULN.76), together with a
proved reached joint/phase class and its cumulative departure budget.
The reverse charge is already inside the actual source plan; its
loss and the velocity transported by shared components cannot be
replaced by their first-reference values. In particular the .081
reference signed preparation excess does not itself dominate the
quality cost of an arbitrary new jittered descendant.

Equations (KULN.69)--(KULN.71) are the exact all-history replacement
test, and (KULN.73)--(KULN.74) are certified positive initial prices.
A nonnegative all-history lower bound for (KULN.76) on a preserved
phase class has not been established here. These initial price
inequalities are not iterated as a homogeneous branching law.

## 12. Exact component-Haar potential and outcome charges

:::{prf:lemma} Exact native component-Haar price and outcome charges
:label: lem-kuln-exact-component-haar-price

For the unchanged native three-dimensional Rastrigin record, on
every entering nonextinct state with original stored velocity cap
two, condition on the entire actual measurement array, original
frozen-source plan and all original jitters. Equations
(KULN.77)--(KULN.84) hold in both normalization modes. In particular
they give the exact native potential mean, actual outside/extinction
probabilities and a complete weighted joint-price lower producer,
with constants independent of the slot number.

*Proof.*

The full weighted operator has an additional analytic reduction that
keeps each actual orbital and component contribution. Condition on
the current physical state, its entire actual measurement array,
the actual frozen-source plan \(P\), and all original clone jitters
\(J\). Denote its collision components by \(C\), with the actual
retained pre-collision means \(m_C\). The prepared positions \(X\)
are fixed at this point. The native first viscous matrix
\(A_1(X)\) depends on these positions and the configured Gaussian
kernel, and is stochastic in both normalization modes. It does not
depend on the subsequently drawn component rotations.

The original restitution is \(1/2\), hence its exact collision
velocity is \(m_C+(1/2)O_C(v_k-m_C)\) for \(k\in C\).
Therefore the first viscous velocity has the exact decomposition
\[
 U_i=M_i+\sum_C O_C B_{iC},\qquad
 M_i=\sum_C\left(\sum_{k\in C}(A_1)_{ik}\right)m_C,
\]
\[
 B_{iC}=\frac12\sum_{k\in C}(A_1)_{ik}(v_k-m_C).
 \tag{KULN.77}
\]
All means, deviations and weights in this identity are those of
the actual plan. Dead retained velocities enter their original
component means before revival/collision. The same \(O_C\) appears
in every row to which component \(C\) contributes; no independent
row rotations have been substituted.

The stored cap and stochasticity give, pointwise,
\[
 |M_i|\le2,\qquad \sum_C|B_{iC}|\le2,
 \qquad R_i^2:=\sum_C|B_{iC}|^2\le4,
\]
\[
 \mathbb E_H|U_i|^2=|M_i|^2+R_i^2\le8,
 \qquad |U_i|\le4.
 \tag{KULN.78}
\]
The last deterministic bound retains every Haar outcome. The middle
identity averages independent mean-zero rotations of different
components, while maintaining all within-component dependence.

Put \(\bar m_i=g(X_i)+bM_i\),
\(\operatorname{sinc}(z)=\sin(z)/z\), with value one at zero, and
\[
 S_i=\prod_C\operatorname{sinc}(2\pi b|B_{iC}|).
\]
In dimension three, a coordinate of \(O_CB_{iC}\) is uniform
on \([-|B_{iC}|,|B_{iC}|]\), so its characteristic function is
that sinc factor. Conditional independence of *different*
components, and the original independent OU/final Gaussian noises,
give the exact native potential response
\[
\begin{split}
 \mathcal E_i(P,J)
 :={}&\mathbb E_{H,\boldsymbol\xi,\boldsymbol\zeta}U(X_i^+)\\
 ={}&|\bar m_i|^2+b^2R_i^2+3\tau^2
       +10\sum_{r=1}^3[1-e^{-2\pi^2\tau^2}
                     \cos(2\pi(\bar m_i)_r)S_i].
\end{split}
 \tag{KULN.79}
\]
To prove this identity, first average the square using the zero
mean and radial second moment of each Haar vector. For each cosine,
average its complex exponential one component at a time; the
independent Gaussian contribution multiplies it by
\(e^{-2\pi^2\tau^2}\). Linearity then sums the three coordinate
potentials. This calculation does not assume that different
completed coordinates or rows are independent after Haar averaging.
Both actual first-viscosity modes enter through their own
\(A_1(X)\), not a frozen reference field.

The signed landscape charge relative to the same mean without its
rotated deviations is particularly explicit. Let
\[
 \mathcal U_\tau(m)=|m|^2+3\tau^2
              +10\sum_r[1-e^{-2\pi^2\tau^2}\cos(2\pi m_r)].
\]
Then
\[
 \mathcal E_i-\mathcal U_\tau(\bar m_i)
 =b^2R_i^2+10e^{-2\pi^2\tau^2}(1-S_i)
                           \sum_r\cos(2\pi(\bar m_i)_r).
 \tag{KULN.80}
\]
The cosine sum keeps the sign of the actual orbital location.
Rotated spread can raise or lower the potential contribution
depending on that location. The actual native force still appears
in \(g(X_i)\).

Every component argument is at most \(4\pi b<.492801<.5\).
Using \(\operatorname{sinc}(z)\ge1-z^2/6\) and multiplying
nonnegative factors gives
\[
 0\le1-S_i\le(2\pi b)^2R_i^2/6,
 \qquad S_i>.95952467.
\]
Consequently
\[
 |\mathcal E_i-\mathcal U_\tau(\bar m_i)|
 \le b^2[1+20\pi^2e^{-2\pi^2\tau^2}]R_i^2
 <1.210497.
 \tag{KULN.81}
\]
The first inequality uses the absolute bound on the cosine sum;
the exact signed charge (KULN.80) remains available for phase
estimates. These are uniform population-size bounds, with no
contraction asserted between distinct orbital centers.

### Killing and quality replacement with all outcomes

Conditional on \(P,J,H\), the actual completed positions are
independent Gaussians with variance \(\tau^2I_3\) and means
\(\mu_i(H)=\bar m_i+b\sum_C O_CB_{iC}\). Thus the exact
per-row outside probability and complete extinction probability are
\[
 d_i(P,J)=1-\mathbb E_H\prod_{r=1}^3
                \ell_{2,\tau}((\mu_i(H))_r),
\]
\[
 p_\dagger(S,\mathbf d)
 =\mathbb E_{P,J,H}\prod_i
      \left[1-\prod_r\ell_{2,\tau}((\mu_i(H))_r)\right].
 \tag{KULN.82}
\]
These products use conditional Gaussian independence. The common
component rotations stay inside the outer expectation; they are
not declared independent across the completed rows. Both death
and survival are exact retained outcomes. The actual survivor
normalization, when used, divides a raw alive price by
\(1-p_\dagger\).

A primitive upper outside bound also follows without a component
integration. Set \(R_i^{(1)}=\sum_C|B_{iC}|\). The Gaussian
outside probability is even and increases with the absolute mean,
so a coordinate union bound gives
\[
 d_i(P,J)\le\min\!\left\{1,\sum_r
  \left[Q\!\left(\frac{2-|\bar m_{ir}|-bR_i^{(1)}}\tau\right)
       +Q\!\left(\frac{2+|\bar m_{ir}|+bR_i^{(1)}}\tau\right)
  \right]\right\}=:\bar d_i(P,J).
 \tag{KULN.83}
\]
No bound on the unbounded prepared \(X_i\) is assumed. For a large
mean, the right side retains the corresponding large death charge.

Jensen applies to the scalar exponential of the exact potential:
\(\mathbb E e^{-cU(X_i^+)}\ge e^{-c\mathcal E_i}\).
Subtracting the actual outside mass and using the pointwise cap
therefore yields the complete all-history lower replacement test
\[
 \mathcal K_{c,\omega,N}^j(S,\mathbf d)
 \ge e^{-2c\omega}\mathbb E_{P,J}
       \sum_{i:L_i(P)=j}[e^{-c\mathcal E_i(P,J)}-d_i(P,J)]_+.
 \tag{KULN.84}
\]
Replacing \(d_i\) by the computed upper bound \(\bar d_i\)
is also valid. Each source's outgoing loss, accepted incoming rows,
within-lineage transfers and mandatory dead revivals are in its
actual plan law. Its weighted response depends on the true
component means/deviations and their first-viscosity transport;
their costs have not been replaced by a common offspring constant.
\(\square\)
:::

A sufficient explicit weighted continuation inequality is therefore
\[
\begin{split}
 \mathbb E_{P,J}\sum_i\ell_{L_i(P)}
      [e^{-c\mathcal E_i(P,J)}-\bar d_i(P,J)]_+
 \ge{}&e^{2c\omega}(1+\chi)
       \sum_j\ell_ja_j\psi_{c,\omega}(x_j,v_j),\quad\chi>0.
\end{split}
 \tag{KULN.85}
\]
Before measurements, its left side is averaged against the whole
actual sampled-measurement law. Establishing (KULN.85) on a
quantitatively preserved reached joint/phase class would close
the quality replacement in (KULN.76). The exact all-history
energy and all-outcome charges (KULN.77)--(KULN.84) are proved;
the positive sign of (KULN.85) is not inferred from a first-law
bound on \(M_i\), from the spatial basin producer, or from
separate independent row rotations.
