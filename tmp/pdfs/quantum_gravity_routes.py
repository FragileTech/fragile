"""Two source-grounded geometric routes and their gravitational assembly."""

QUANTUM_GRAVITY_ROUTES = r'''
\clearpage
\section{Quantum gravity: two routes from the same dynamics}
\label{sec:quantum-gravity}

The gravitational extension uses the same controlled population and recorded
history as the Yang--Mills program. Fitness determines an adaptive metric;
the Fractal Set records causal evolution and geometric measurements. Two
routes develop the continuum geometry: differentiation of the fitness metric,
and reconstruction through discrete operators and curvature statistics.
Their agreement supplies a comparison between independently evaluated
observables of the same dynamics. The quantum-gravity target retains the
fluctuating geometry together with its coupled matter and gauge observables.

\begin{center}
\begin{tikzpicture}[
 x=1cm,y=1cm,
 every node/.style={align=center,font=\small},
 qbox/.style={draw=ink!65,fill=ink!4,rounded corners=2pt,
 text width=6.1cm,minimum height=1.2cm,inner sep=5pt},
 qwide/.style={qbox,text width=13.2cm},
 qcontrol/.style={qbox,draw=teal,fill=teal!6},
 qtarget/.style={qwide,draw=orange!70!black,fill=orange!7,dashed},
 qflow/.style={-{Stealth[length=2mm]},draw=ink!80,line width=.8pt}]
\node[qwide] (root) at (0,0) {\textbf{ONE POPULATION AND HISTORY LAW}\\
Confinement and tails; mean field; normalized fitness derivatives;\\
joint LSI/Poincar\'e; temporal mixing; recorded algorithmic time};
\node[qcontrol] (metric) at (-3.55,-2) {\textbf{A. FITNESS-HESSIAN ROUTE}\\
Fitness derivatives $\longrightarrow g_t$\\
Connection and Hessian curvature};
\node[qcontrol] (graph) at (3.55,-2) {\textbf{B. FRACTAL SET ROUTE}\\
Geometry and density weights\\
Laplacian and curvature estimators};
\node[qbox] (direct) at (-3.55,-3.95) {Local derivative convergence\\
Curvature contractions and volume\\
Lorentzian time/mixed derivatives};
\node[qbox] (discrete) at (3.55,-3.95) {Bias, covariance and bandwidth\\
Graph reconstruction and quadrature\\
Discrete gravitational action};
\node[qwide] (join) at (0,-5.95) {\textbf{COMMON GEOMETRY AND ACTION LIMIT}\\
Compare both readouts on regular regions; control integrated singular regimes\\
Identify the gravitational action in the native history law};
\node[qtarget] (law) at (0,-7.95) {\textbf{GRAVITATIONAL DYNAMICS AND QUANTUM FLUCTUATIONS}\\
Native variation/response $\longrightarrow$ curvature--stress equation\\
Joint geometry--matter correlators $\longrightarrow$ physical reconstruction};
\draw[qflow] (root.south) -- (metric.north);
\draw[qflow] (root.south) -- (graph.north);
\draw[qflow] (metric.south) -- (direct.north);
\draw[qflow] (graph.south) -- (discrete.north);
\draw[qflow] (direct.south) -- (join.north -| direct.south);
\draw[qflow] (discrete.south) -- (join.north -| discrete.south);
\draw[qflow] (join.south) -- (law.north);
\end{tikzpicture}
\end{center}
{\small Solid boxes collect constructions and analytic tools already developed
in the source, with their stated hypotheses. The dashed box gives the
gravitational and quantum assembly to establish.}

\clearpage
\subsection{Route A: curvature from the fitness Hessian}
\label{sec:gravity-hessian}

\paragraph{Metric and population limit.}
Stage~\ref{stage:9} differentiates the complete normalized fitness, including
the companion-law derivatives when an expected field is used. The normalized
mean-field integral theorem and smooth empirical-field convergence theorem
give the derivative transfer
\[
V_{\mathrm{fit},N}\longrightarrow V_{\mathrm{fit}}[\mu]
\quad\hbox{in }C^r(K)
\]
on compact query regions under their denominator, domination and
equicontinuity hypotheses, for fixed reconstruction parameters
@@thm-cinf-mean-field-integrals@@, @@thm-cinf-empirical-field-convergence@@.
For varying bandwidths and regularizers, carry their constants along the
declared scale schedule. Conditional sampled fields retain their donor
context; comparing them with an expected field also requires the corresponding
conditional fluctuation estimate.

On a smooth positive Hessian branch, define
\[
\Phi_N=V_{\mathrm{fit},N}+\tfrac12\epsilon_g|x|^2,
\qquad g_{N,ab}=\partial_a\partial_b\Phi_N.
\]
On each regular region use a spectral margin $g_N\succeq m_KI$ and the
established matrix estimates to pass inverses, noise factors and volume
densities to their limits. The executed positive-part branch is
$g_N=\epsilon_gI+(H_N)_+$; its general curvature calculation uses the
derivatives of that selected metric. Smooth spectral regions and justified
weak reconstructions carry their own regularity conditions
@@prop-geometry-clipped-metric@@.

\paragraph{Curvature theorem and proof mechanism.}
Write $C_{abc}=\partial_a\partial_b\partial_c\Phi$. The Hessian-curvature
lemma proves
\[
\Gamma^a_{bc}=\tfrac12g^{ap}C_{pbc},\qquad
R_{abcd}=\tfrac14g^{pq}
 (C_{adp}C_{bcq}-C_{acp}C_{bdq})
\qquad @@lem-curvature-hessian-cancellation@@.
\]
Symmetry cancels the fourth derivatives in the antisymmetrized connection
derivative. The remaining terms are products of third derivatives and the
inverse metric. Under the lemma's classical regularity hypotheses,
$C^3(K)$ convergence of the potentials and a common spectral margin therefore
give uniform convergence of these spatial curvature components and their
Ricci, scalar and Einstein contractions. This consequence follows by
substitution in the displayed formula. A constant positive Hessian gives a
flat spatial metric; curvature is controlled by its spatial variation.

\paragraph{Algorithmic time and coupled evolution.}
The spatial metric becomes part of a Lorentzian spacetime reconstruction,
with recorded $t_n=nh$. In the product model $G=-c^2dt^2+g_t$, spacetime
curvature also depends on temporal and mixed derivatives. Control these
through the actual metric increments and the interpolation used in the limit.
The conditional metric law and coupled-field theorem already give those
increments jointly with the population, mechanical balances, noise and history
@@thm-algorithmic-conditional-metric-law@@,
@@thm-algorithmic-coupled-field-system@@.
Thus the gravitational reduction starts from a specified coupled evolution.

\paragraph{Route A output.}
The output is a controlled spatial curvature field, its volume density, and
a spacetime curvature field when the temporal reconstruction estimates hold.
For the spacetime metric use the distinct notation
$\mathsf E_{ab}[G]=\operatorname{Ric}_{ab}[G]-\tfrac12R[G]G_{ab}$.
The fitness metric supplies the geometric input to this Einstein tensor.

\clearpage
\subsection{Route B: curvature and action from the Fractal Set}
\label{sec:gravity-fractal-set}

\paragraph{Transfer the existing analytic control.}
The complete-record encoding transports the established Dirichlet form, LSI
and Poincar\'e constants to Fractal Set observables
@@prop-fractal-set-analytic-transfer@@. The joint law and the transported form
remain those of the recorded experiment. Temporal observations use their
specified history law and mixing estimates. This is the same foundation used
by Stages~\ref{stage:10}, \ref{stage:14} and \ref{stage:15}.

\paragraph{Laplacian convergence.}
For $M$ spatial observations and length bandwidth $\varepsilon$, the
unnormalized kernel theorem identifies
\[
L_{M,\varepsilon}\phi\longrightarrow
\frac{m_2}{2}\bigl(\rho\Delta_g\phi
 +2\langle\nabla\rho,\nabla\phi\rangle_g\bigr).
\]
Density correction and row normalization give the geometric operator
\[
L^{\mathrm{corr}}_{M,\varepsilon}\phi(x)
\xrightarrow{\mathbb P}\Delta_g\phi(x)
\qquad @@thm-laplacian-convergence@@,
@@prop-density-corrected-limit@@.
\]
The proof expands the weighted integrand in normal coordinates, uses radial
kernel moments, and controls sampling fluctuations. The joint Poincar\'e
route bounds variance by $C_*C_H/(M\varepsilon^{d+4})$. For an estimated
density the theorem retains its local relative-error requirement. Recorded
graph distances, omitted edges and weights enter an explicit reconstruction
comparison after the operator normalization.

The spacetime nonlocal-operator theorem similarly converges to $\Box_G$,
with a covariance route or a joint Poincar\'e route for fluctuations
@@thm-cst-fractal-dalembertian-consistency@@,
@@lem-cst-poincare-variance@@. These operator results and the curvature
result below have their own calibrated observables and error estimates.

\paragraph{Scalar-curvature theorem and proof mechanism.}
The curved neighborhood expansion fixes $M_0^{(R)}$ and $M_R\ne0$ through
\[
\varepsilon^{-D}\int\kappa^R_{\varepsilon,p}\,d\mathrm{vol}_G
 =M_0^{(R)}+M_RR[G](p)\varepsilon^2+O(\varepsilon^3).
\]
With the geometric sampling weights, define
\[
\widehat R_{\mathrm{FG}}(p)=\frac1{M_R\varepsilon^2}
\left[\frac1{M\varepsilon^D}\sum_{i=1}^M
 W_{\mathrm{geo}}(Y_i)\kappa^R_{\varepsilon,p}(Y_i)-M_0^{(R)}\right].
\]
The conditional scalar-curvature theorem proves
\[
\mathbb E|\widehat R_{\mathrm{FG}}(p)-R[G](p)|^2
\le C_1\varepsilon^2+\frac{C_2}{M\varepsilon^{D+4}}
\qquad @@thm-fractal-gas-ricci@@.
\]
The expectation uses the calibrated geometric volume expansion; the variance
uses the covariance estimate for this curvature observable. Choose
$\varepsilon\to0$ with $M\varepsilon^{D+4}\to\infty$ and controlled constants.
Approximate episode inputs retain their normalized reconstruction error.

\paragraph{Route B output.}
With uniform convergence on the support of a test function $\chi$ and the
weighted quadrature law, the same theorem gives
\[
S_{\mathrm{FG},\chi}=\frac1{2\kappa M}\sum_i
 W_{\mathrm{geo}}(Y_i)\chi(Y_i)\widehat R_{\mathrm{FG}}(Y_i)
\xrightarrow{\mathbb P}
\frac1{2\kappa}\int\chi R[G]\,d\mathrm{vol}_G.
\]
This supplies an Einstein--Hilbert bulk-action approximation. Volume terms,
boundary terms and source contributions are added with their own common-law
quadrature and variation estimates. Noncompact action integrals use the
confinement and uniform-integrability estimates of Stage~\ref{stage:3}.

\clearpage
\subsection{Compare both routes and derive the gravitational law}

\paragraph{Agreement on a common geometry.}
Evaluate both routes using the same metric reconstruction, spacetime clock,
sampling law and normalization. On a regular compact region $K$, Route A
produces $R_N^{A}$ and Route B produces $\widehat R_N^{B}$. Their established
convergence estimates give
\[
|R_N^{A}(p)-\widehat R_N^{B}(p)|
\le |R_N^{A}(p)-R[G](p)|
 +|\widehat R_N^{B}(p)-R[G](p)|.
\]
Uniform versions give the action comparison on $K$. Spatial scalar curvature
is compared with a spatial estimator; spacetime scalar curvature is compared
with its spacetime estimator. Small-loop holonomy supplies a further
tensor-curvature readout under its transport-comparison hypotheses
@@thm-riemann-scutoid@@. The scalar estimator and holonomy measurements thus
have distinct roles in recovering the full curvature tensor.

\paragraph{A causal-set action option.}
The Benincasa--Dowker construction offers a retarded layer-count operator
whose continuum expression is $\Box_G-\tfrac12R[G]$. Applying it to the
constant field extracts scalar curvature. A discrete action follows by
summing these contributions. This is an additional action construction on
the recorded order, requiring the actual interval-count sampling law,
scale-dependent fluctuation control and boundary calibration.
The source's localized two-sided operator and this retarded operator are
compared by their kernels, causal orientation and normalization
@@rem-cst-bd-comparison@@.
The relevant external results are
\href{https://arxiv.org/abs/1001.2725}{Benincasa--Dowker, \emph{The Scalar Curvature of a Causal Set}}
and
\href{https://arxiv.org/abs/2007.13192}{Machet--Wang, \emph{On the continuum limit of the Benincasa--Dowker--Glaser causal set action}}.
The latter calculates the mean continuum action for sprinkled causal diamonds,
including an Einstein--Hilbert bulk term and a codimension-two boundary term.
The Fractal Gas application supplies its own correlated-history estimates.

\paragraph{Gravitational assembly criterion.}
Stage~\ref{stage:12} constructs the native descriptor action and source
response from the executed history law. Identify its geometric sector, on the
chosen scales, with an effective action of the form
\[
S_{\mathrm{eff}}[G,\Psi]
 =\frac1{2\kappa}\int(R[G]-2\Lambda)\,d\mathrm{vol}_G
 +S_{\mathrm{matter}}[G,\Psi]+S_{\mathrm{corr}}[G,\Psi].
\]
Here $\Psi$ denotes the retained matter and gauge fields, and
$S_{\mathrm{corr}}$ retains the derived higher-order and eliminated-field
contributions. Control first variations along the declared admissible metric
family, boundary data and sources. In a regime where stationarity holds for
the required metric variations, define
\[
T_{ab}=-\frac{2}{\sqrt{|\det G|}}
 \frac{\delta S_{\mathrm{matter}}}{\delta G^{ab}},\qquad
\mathcal C_{ab}=-\frac{2\kappa}{\sqrt{|\det G|}}
 \frac{\delta S_{\mathrm{corr}}}{\delta G^{ab}}.
\]
The variational identity then reads
\[
\mathsf E_{ab}[G]+\Lambda G_{ab}=\kappa T_{ab}+\mathcal C_{ab}.
\]
This displayed equation is the completion target for the native-action
identification. Its classical Einstein regime is obtained when the correction
tensor vanishes or is controlled at the stated approximation order.
Restricted Hessian variations initially give the corresponding projected
identity; obtaining the full tensor identity requires variations that determine
all its components. An equivalent route derives the curvature--stress law
directly from the coupled metric and mechanical evolution with a controlled
reduction. The source terms and memory corrections come from that evolution.

\clearpage
\subsection{Singular limits, information geometry and quantum fluctuations}

\paragraph{Regular regions and singular regimes.}
On each compact regular region $K$, use local derivative, spectral and sampling
estimates, allowing their constants to deteriorate toward singular regions.
A fixed spectral floor protects the implemented inverse; changing it requires
tracking its noise and comparison costs. Confinement controls probability tails.
Global bounded curvature is a restriction for a regular regime.

Near a proposed singular set, use an exhaustion by regular regions and a
specified weak or integrated curvature/action formulation. Estimate the omitted
contributions at their actual scale; if a finite integrated limit is sought,
prove the necessary integrability or state the chosen renormalization.
These are explicit tasks in the singular-regime extension of both routes.
A black-hole horizon is identified through the causal escape structure of the
reconstructed spacetime. Metric degeneracy, a clipping transition and invariant
curvature blow-up are measured separately. Horizon regularity can be tested
using a suitable spacetime chart; see
\href{https://vo.ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll7.html}{Carroll, \emph{Lecture Notes on General Relativity}, Chapter 7}.

\paragraph{Fisher and Ruppeiner descriptions of Route A.}
For a regular exponential family
\[
p_\eta(y)=\exp\{\eta^aT_a(y)-A(\eta)\}\,h(y),\qquad
g^{\mathrm F}_{ab}=\partial_a\partial_bA
 =\operatorname{Cov}_{p_\eta}(T_a,T_b),
\]
the log-partition Hessian is the Fisher metric: differentiating normalization
gives $\partial_aA=\mathbb E_\eta T_a$, then the covariance identity.
Ruppeiner geometry uses the negative entropy Hessian in thermodynamic state
coordinates. Identifying fitness with the appropriate thermodynamic potential,
coordinate map and normalization realizes the Hessian route in information
geometry. Establish these identifications for the chosen population model,
then reuse its curvature and action estimates. See
\href{https://doi.org/10.1103/RevModPhys.67.605}{Ruppeiner, \emph{Riemannian geometry in thermodynamic fluctuation theory}}.

\paragraph{Retain the quantum geometry.}
The quantum-gravity extension retains fluctuations of geometry and matter in
the same equilibrium history. Route A constructs derivative-based geometric
observables; Route B constructs their discrete operator and action readouts.
Introduce jointly sourced correlations on the common recorded law,
\[
\mathcal Z_{N,h}(J,K)=\mathbb E_{P_{N,h}^{\mathrm{hist}}}
 \exp\!\left(i\sum_\alpha J_\alpha O^{\mathrm{geom}}_\alpha
             +i\sum_\beta K_\beta O^{\mathrm{matter}}_\beta\right).
\]
Use the established fluctuation, mixing and uniform-integrability estimates
to extract a joint hierarchy. Curvature probes are smeared or regularized on
the declared scales, with their observable-gradient or covariance costs
included. Connected geometry--matter correlations retain the fluctuations
discarded by a deterministic mean metric. Classical stationarity describes
the corresponding effective regime; quantum reconstruction uses the full
hierarchy and its physical positivity, causal and symmetry criteria.

\paragraph{Completion tasks for the combined extension.}
Carry the derivative bounds through the scale family; establish the recorded
geometric sampling and curvature calibration; compare both action readouts;
identify the native action and its variation; control singular regions; and
reconstruct the physical joint hierarchy. Shared confinement, mean-field,
LSI/Poincar\'e and history tools support both routes.
'''
