"""Source-grounded additions to the Volume 2 research prospectus."""

ADDITIONS = {
    3: (
        r"""\subsection*{Structural transport and mass--shape control}
The positional transport chapter derives the centered structural control from the cloning variance reset and carries the barycenter separately in its full \(W_2\) decomposition. The structural landscape chapter evaluates the signed Keystone, collision, and kinetic terms for the complete update. These results retain the inward selection flux, the donor-normalization costs, and the actual force increments in the same one-step ledger.

The mass, Hellinger, and transport chapter combines the alive mass with normalized shape. For finite measures \(m\rho\) and \(m_*\rho_*\), it proves the exact mass--shape identity
\[
H^2(m\rho,m_*\rho_*)
=(\sqrt m-\sqrt{m_*})^2+\sqrt{mm_*}\,H^2(\rho,\rho_*).
\]
It combines this with entropy-to-Hellinger and entropy-to-transport inequalities and also bounds the canonical Hellinger--Kantorovich distance. Safe-walker survival, concentration of alive mass, and convergence of the conditioned shape therefore control separate parts of the finite-measure error. This is the population normalization used in the convergence analysis.
""",
        [
            ('thm-full-w2-split', 'Carries the barycenter and centered structural terms in full transport control.'),
            ('thm-hk-convergence-main-assembly', 'Assembles convergence of alive mass and normalized shape.'),
            ('thm-exponential-survival', 'Controls finite-horizon survival through the safe-walker estimates.'),
        ],
    ),
    5: (
        r"""\subsection*{The finite algorithm already has field equations}
The field-equations chapter derives the hierarchy directly from the configured donor, cloning, innovation, and boundary maps. Its full marked state retains current and historical donor data, eligibility, provider state, and the inputs needed by the next update. On this state, its marked field and age transport are exact.

For any square-integrable vector observable \(A\), the complete kernel gives
\begin{align*}
b_A(S)&=P_{N,h}A(S)-A(S),\\
\Gamma_A(S)&=P_{N,h}(AA^\top)(S)-P_{N,h}A(S)P_{N,h}A(S)^\top,\\
A(S_{n+1})-A(S_n)&=b_A(S_n)+\eta_{n+1},
\end{align*}
with \(\mathbb E[\eta_{n+1}\mid S_n]=0\) and conditional covariance \(\Gamma_A(S_n)\). The same formula applies to metric probes and gauge readouts. Fourier tests and the marked characteristic functional give the exact moment and field-correlation hierarchy, retaining the configured innovation law and cross-walker dependence.

For a row transition \((x,v,a)\mapsto(y,w,b)\), deposit along the segment from \(x\) to \(y\). The fundamental theorem of calculus gives the distributional density and current balances. Summing over the actual stages telescopes to
\[
\rho_{n+1}-\rho_n+\operatorname{div}\mathcal J_\rho=\mathcal S_\rho,
\qquad
j_{n+1}-j_n+\operatorname{div}\mathcal J_j=\mathcal S_j.
\]
All source and transport terms are computed from the executed update. The coupled-field theorem combines these mechanical equations with the population, history, Hessian metric, and thermostat covariance. Configured algorithm variants retain their own stage ordering and coefficients in these identities.
""",
        [
            ('thm-algorithmic-explicit-transition', 'Explicit finite-step population law for the configured complete state.'),
            ('thm-algorithmic-field-characteristics', 'Exact characteristics, characteristic functional, and moment hierarchy.'),
            ('thm-algorithmic-observable-increment', 'Exact drift and conditional fluctuation covariance of derived observables.'),
            ('thm-algorithmic-spatial-field-equations', 'Exact finite-step spatial density and momentum balances.'),
            ('thm-algorithmic-coupled-field-system', 'The coupled population, mechanical, and metric field system.'),
        ],
    ),
    6: (
        r"""\subsection*{Exchangeability connects the empirical law to tagged fields}
Kernel permutation symmetry and uniqueness give exchangeability of the selected finite law. The exchangeability chapter derives the finite empirical-mixture representation and proves that a deterministic empirical limit gives fixed-row chaos. It also evaluates the finite-population covariance exactly:
\[
\operatorname{Cov}(g(Z_1),h(Z_2))
=\frac{N\operatorname{Cov}(L_Ng,L_Nh)
-\operatorname{Cov}(g(Z_1),h(Z_1))}{N-1}.
\]
The empirical variance bound therefore controls distinct-walker correlations with the finite-size diagonal term retained. This supplies the passage between population observables and fixed-row invariant channels used in field reconstruction.
""",
        [
            ('thm-qsd-exchangeability', 'Exchangeability of the unique selected QSD.'),
            ('thm-propagation-chaos-qsd', 'Deterministic empirical limits imply chaos.'),
            ('thm-correlation-decay', 'Exact covariance identity for a finite exchangeable population.'),
        ],
    ),
    9: (
        r"""\subsection*{Fitness regularity produces the adaptive geometry}
The emergent-geometry chapter identifies the implemented positive spectral branch
\[
g=\epsilon_g I+(H_{\rm sym})_+,
\qquad D=g^{-1},
\qquad B=\sqrt{2\gamma T}\,g^{-1/2}.
\]
The fitness derivative hierarchy controls the Hessian and its variation. Spectral lower and upper bounds then control metric inversion, adaptive diffusion, and the noise form. The geometric-gas chapter transfers moment drift with explicit perturbation costs and provides Harris, joint-LSI, concentration, and tail results for its identified geometric law families.

For reconstructed matrices \(g,\widehat g\), the Yang--Mills chapter uses the spectral margin to bound inverse and square-root errors:
\[
\|\widehat g-g\|_{\rm F}\le\delta,
\quad \|\widehat g^{-1}-g^{-1}\|_{\rm F}\le\epsilon_g^{-2}\delta,
\quad
\|\widehat g^{-1/2}-g^{-1/2}\|_{\rm F}
\le\frac{\delta}{2\epsilon_g^{3/2}}.
\]
The normalized geometric-weight comparison is evaluated on the same recorded history. These estimates transport the existing coefficient regularity into metric, volume-weight, and field-readout accuracy.
""",
        [
            ('prop-geometry-clipped-metric', 'Identifies the metric represented by the implemented Hessian branch.'),
            ('thm-uniform-ellipticity-latent', 'Ellipticity from the established metric bounds.'),
            ('thm-gg-ueph-construction', 'Geometric uniform ellipticity from the spectral margin.'),
            ('thm-gg-lsi-main', 'Joint population-uniform LSI for the identified geometric law.'),
        ],
    ),
    10: (
        r"""\subsection*{Three spatial coordinates and the recorded clock}
For this program, each walker has \(x_i(n),v_i(n)\in\mathbb R^3\), and the observation clock is \(t_n=n\Delta t\). The spacetime episode coordinate is \((t_n,x_i(n))\). CST temporal edges increase the integer iteration index; IG and IA data encode the recorded interactions. The causal-set theorem proves irreflexivity, transitivity, and local finiteness from that recorded ordering. Between two finite integer layers, the interval contains only finitely many episodes.

The scutoid chapter builds slabs between successive frames using their spatial cells and neighbor changes. Its temporal direction comes from the recorded clock. Proper time is defined by the reconstructed Lorentzian metric on this ordered geometry; its relation to the recorded clock is kept in the trajectory-clock comparison. The continuum geometry and local-operator estimates are applied on this specified construction.

The causal-set chapter makes the analytic dependency order explicit: fitness derivatives precede metric inversion; joint LSI precedes empirical gradient control; local bias and variance estimates precede operator consistency. Graph order and distance comparisons identify the recorded geometry to which those estimates apply. This preserves the \(3+1\) origin of the construction throughout the prospectus.
""",
        [
            ('thm-fractal-is-causal-set', 'The recorded temporal order is a causal set.'),
            ('prop-fractal-causal-order-equivalence', 'The order-to-geometric-causality comparison under its stated slab hypotheses.'),
            ('thm-fractal-faithful-embedding', 'Volume matching and trajectory clocks for the specified embedding.'),
            ('prop-scutoid-cst-compatibility', 'Slab consistency in the moving-cell reconstruction.'),
        ],
    ),
    11: (
        r"""\subsection*{Recorded loops and geometric curvature}
The Voronoi Wilson-loop chapter supplies a parallel construction directly from \texttt{RunHistory}: spatial Delaunay neighbor links on a frame, temporal adjacency between frames, recorded fitness or interaction phases, and minimal closed cycles. These are observable constructions on existing data.

The curvature chapter controls holonomy composition by telescoping the difference of ordered transport products. With bounded face complexity and the stated edgewise comparison, the accumulated error is \(O(h^3)\). For a shape-controlled plaquette with area \(A_\Pi\asymp h^2\), the small-loop theorem yields
\[
\frac{(\mathcal H[\Pi]-I)V}{A_\Pi}
\longrightarrow R(X,Y)V,
\]
with an explicit \(O(A_\Pi^{1/2})\) leading comparison error. This is the geometric transport tool used alongside the gauge face-action expansion; each connection retains its declared representation and correspondence.
""",
        [
            ('lem-regge-holonomy-approx', 'Telescoping error bound for comparison holonomy.'),
            ('thm-riemann-scutoid', 'Curvature recovery from consistent plaquette transport.'),
            ('lem-ym-recorded-face-action-remainder', 'Quantitative curvature remainder for the recorded gauge face action.'),
            ('lem-reference-transport-covariance', 'Gauge covariance and invariant norms for fixed measurement paths.'),
        ],
    ),
    12: (
        r"""\subsection*{Reduced fields retain the derived memory}
The field-equations and direct-observable chapters both construct the exact reduction of the complete state. Conditional expectation onto the field descriptor splits the transition into resolved and unresolved blocks. Eliminating the unresolved block gives the transient memory equation. Conditioning on the complete observed field history gives the explicit filtering equation and its prediction covariance.

The prediction-complete descriptor is formed by closing the original bounded observable algebra under the actual transition. Its finite conditional partitions give convergent transition matrices. Thus the retained gauge channels have an exact history action and prediction law, and their simulation approximation can be refined without postulating a closed differential equation for a few initial readouts.

The expansion and macroscopic-closure chapter supplies the complementary closure calculus. Information closure is equality of the conditional macro-future laws given the microscopic and macroscopic pasts. Its theorem relates this condition to predictive causal-state maps. For generator-based coarse variables, its approximate Markov-closure proposition gives
\[
\sup_x|L_N(\varphi\circ f)(x)-\overline L\varphi(f(x))|
\le\epsilon_N
\quad\Longrightarrow\quad
|\mathbb E[\text{Dynkin residual over }[0,t]]|\le t\epsilon_N.
\]
This is a quantitative tool for comparing a proposed macroscopic evolution with the complete recorded dynamics. The finite-step construction above continues to use its actual discrete kernel.

The algorithmic-thermodynamics chapter derives stationary response from the discrete Poisson equation for the actual kernel. With the stated weighted mixing and differentiable-kernel hypotheses, put
\[
u_A=\sum_{n\ge0}K^n[A-\pi(A)],\qquad (I-K)u_A=A-\pi(A).
\]
If \(D\) is the derivative of the configured kernel, the response is
\[
\left.\frac{d}{d\theta}\pi_\theta(A)\right|_{0}=\pi(Du_A).
\]
This sums the delayed effect of the perturbation through the same dynamics. The path-KL identities likewise retain the specified forward and reverse experiments and their record pushforwards. They complement the native field-source likelihood calculations.
""",
        [
            ('thm-algorithmic-transient-field-memory', 'Exact reduced-field propagation with eliminated-state memory.'),
            ('thm-algorithmic-field-filter', 'Conditional-history evolution and its primitive-computed noise covariance.'),
            ('thm-sm-prediction-complete-descriptors', 'Prediction-complete descriptors generated by the actual transition.'),
            ('thm-sm-predictive-partition-convergence', 'Finite transition matrices approximating that completed field theory.'),
            ('thm-closure-equivalence-cosmo', 'Predictive closure relations for deterministic coarse observables.'),
            ('prop-cosmo-generator-closure', 'Exact block closure and a quantified approximate-generator residual.'),
            ('thm-algorithmic-stationary-response', 'Exact stationary response from the discrete Poisson equation.'),
            ('prop-algorithmic-path-kl-identities', 'Path likelihood and KL identities for the specified dynamics and reverse experiment.'),
        ],
    ),
    13: (
        r"""\subsection*{Native regional dynamics are already computed}
The lattice-QFT chapter derives localized fermion evolution and the regional covariance of the native readouts from the complete recorded transition. Its locality-defect theorem supplies coefficients for the declared regional modes. The inherited-history covariance theorem transports the full original temporal correlations, including overlaps and shared innovations. These calculations support the recorded CAR local-net application and specify its even observable sector.
""",
        [
            ('thm-lqft-native-local-fermion-evolution', 'Native fermion evolution, localization, and record covariance.'),
            ('thm-lqft-inherited-history-covariance', 'Exact inherited-history covariance of native regional readouts.'),
            ('cor-lqft-fock-lsi-transfer', 'Transport of existing LSI and native contraction estimates into the recorded Fock setting.'),
        ],
    ),
    14: (
        r"""\subsection*{The fluctuation field retains algorithmic time}
For the stationary discrete chain, the fluctuation theorem uses the piecewise-constant record at the fixed observation spacing. Its phase-space variable is \(z=(x,v)\in\mathbb R^6\). Physical spatial tests are evaluated through the same recorded positions and may be chosen independent of \(v\); their temporal variable remains the algorithmic observation clock.

The native gauge-hierarchy theorem retains the full Gram coordinates, complex determinants, triangle products, masks, sources, and configured weights. It constructs their finite-word and reflected-product limits on a common subsequence. The direct color observable theorem also derives nontrivial doublet fluctuations from the actual companion draw, while the paired momentum result gives a quantitative nonzero fluctuation estimate for its identified stationary model. These supply concrete finite-sector and fluctuation information within their respective law families.
""",
        [
            ('thm-sm-native-doublet-fluctuations', 'Nontrivial doublet fluctuations from the actual sampled companion.'),
            ('cor-ym-paired-momentum-fluctuations', 'A nonzero momentum fluctuation bound for the identified paired stationary model.'),
            ('lem-ym-physical-gauge-word-uniform-integrability', 'Explicit full-color word and mark-reconstruction bounds.'),
        ],
    ),
    15: (
        r"""\subsection*{Density correction is part of the existing construction}
The continuum and emergent-geometry chapters distinguish the walker sampling law from the geometric volume measure. If the episode marginal is \(\pi(dy)=q(y)\,d\mathrm{vol}_g(y)\), then
\[
\mathbb E\left[\frac1M\sum_{i=1}^{M}\frac{F(Y_i)}{q(Y_i)}\right]
=\int F\,d\mathrm{vol}_g.
\]
This expectation identity uses the actual marginal and requires no independence. The chapter also gives the explicit QSD/Doob distinction and the normalized time-window density. It therefore computes the importance weight from the selected sampling experiment. Walker normalization, geometric integration, and centered field fluctuations are separate quantities with explicitly related formulas.

\subsection*{A second continuum estimate follows directly from joint LSI}
For the smooth normalized summand \(H_{\varepsilon,p}\), the causal-set chapter differentiates the actual empirical estimator:
\[
\nabla_i A_{M,\varepsilon}=M^{-1}\nabla H_{\varepsilon,p}(Y_i).
\]
The established joint Poincar\'e inequality gives
\begin{align*}
\Var(A_{M,\varepsilon})
&\le\frac{C_*}{M}\int|\nabla H_{\varepsilon,p}|^2\,d\pi
\le\frac{C_*C_\nabla}{M\varepsilon^{D+4}},\\
\mathbb E|A_{M,\varepsilon}-\Box_gf(p)|^2
&\le C_b^2\varepsilon^4+
\frac{C_*C_\nabla}{M\varepsilon^{D+4}}.
\end{align*}
Thus \(\varepsilon=M^{-1/(D+8)}\) gives \(O(M^{-4/(D+8)})\) for that observation law and Sobolev summand. In \(D=4\), this is \(O(M^{-1/3})\). The covariance route above gives \(O(M^{-2/5})\) in \(D=4\) under its covariance hypotheses. These are two proved sufficient routes. Single-time observations use their joint law; pooled temporal episodes use the actual episode law or the chapter's temporal covariance estimate.

The spatial lattice-QFT chapter supplies the corresponding density-corrected kernel limit and sampling control for slice operators. Operator reconstruction and action quadrature are then composed with the same smooth-field and sampling estimates.

The computational-proxies chapter also specifies the geometric cell weights used by recorded observables:
\[
V_i^g=V_i^{\rm Eucl}\sqrt{\det g(x_i)},\qquad
w_i^g=V_i^g\Big/\sum_jV_j^g.
\]
The coordinate identity \(d\mathrm{vol}_g=\sqrt{\det g}\,dx\) supplies their geometric meaning. These weights are used with the chosen cell approximation; inverse sampling-density weights supply the separate Monte Carlo integration identity above. The chapter's curvature, distance, and covariance probes provide observable diagnostics for the reconstructed geometry.
""",
        [
            ('lem-continuum-a3-qsd-sampling', 'The actual normalized sampling law, Doob law, and inverse-density weights.'),
            ('prop-monte-carlo-riemannian-latent', 'Riemannian importance integration and joint-LSI variance control.'),
            ('cor-cst-inherited-lsi-consistency', 'Direct continuum estimator error from the already established joint LSI.'),
            ('prop-density-corrected-limit', 'Spatial density-corrected kernel limit with sufficient sampling control.'),
            ('thm-volume-element-transformation', 'Coordinate volume transformation underlying geometric cell weights.'),
        ],
    ),
    16: (
        r"""\subsection*{Action response and geometry fibers retain the source law}
The native response-current theorem writes the source derivative as a distribution tested on the supplied recorded spacetime coordinates. The weak connection-variation identity compares that actual response with the Wilson/matter variation. The common source-action limit keeps the same likelihood and descriptor refinement throughout the passage. Geometry-fiber disintegration retains its conditional denominator, so recombining the fiber and geometry contributions recovers the original sourced correlations.

In the present \(3+1\) roadmap the field integration domain is \(\mathbb R_t\times\mathbb R_x^3\), with \(t\) supplied by the calibrated recorded clock. The Yang--Mills quadratic contraction is taken with the metric convention of the Euclidean or Lorentzian reconstruction under discussion. This specifies the geometry used by the small-face and quadrature statements.
""",
        [
            ('thm-ym-native-response-current', 'Native source-response current on the recorded spacetime.'),
            ('prop-ym-native-connection-variation-identity', 'Native response and the weak connection-variation identity.'),
            ('lem-ym-discrete-ward', 'Exact graph Ward identity.'),
        ],
    ),
    17: (
        r"""\subsection*{The equilibrium correlator route is already specified}
The twistor and calibration chapters define frame channel observables from the actual recorded state, including donor memory and validity data. For an invariant law of that same kernel \(K\), let \(\widetilde f_a=f_a-\pi f_a\). The exact algorithmic-lag correlator is
\[
C_{ab}(\ell)=\langle\widetilde f_a,K^\ell\widetilde f_b\rangle_{L^2(\pi)}.
\]
The complete record transition fixes this correlation. The time separation is \(\ell\Delta t\). When the identified transfer representation is positive and self-adjoint with \(K=e^{-\Delta tH}\), the book derives
\[
C_{aa}(\ell)=\sum_{E_j>0}
|\langle j|\widehat f_a|0\rangle|^2e^{-E_j\ell\Delta t}.
\]
Its leading supported energy is read from the large-lag decay. The local twistor operators are observable channels feeding this pipeline; the mass is a spectral property of their correlator.

\subsection*{Which reflection and symmetry enter this program}
The chosen temporal reflection acts on the stationary history clock:
\[
\Theta:(t,x)\mapsto(-t,x),\qquad x\in\mathbb R^3.
\]
The reflection and transfer identities are applied to the derived temporal observable law. The book's separate tests of a translated embedding coordinate have their stated coordinate action; they are not imported into this algorithmic-clock program without the corresponding identification. Likewise, symmetry is evaluated through the physical observable pushforward and its temporal implementers, while the complete anchored record remains available for decoding and prediction.

The existing source statements give regularity, bosonic or graded symmetry, native local multiplication identities, the CAR local-net construction, and exact stationary temporal correlations. The positive-transfer and physical field-axiom statements retain their explicit representation and covariance hypotheses. In the proof program these hypotheses are applied on the same derived equilibrium hierarchy and clock. This identifies the object on which the final physical reconstruction is carried out.
""",
        [
            ('thm-effective-twistor-spectral-meaning', 'Exact algorithmic evolution and stationary frame-correlator identity.'),
            ('cor-effective-twistor-positive-transfer', 'Spectral expansion for the same channel under its stated positive-transfer hypothesis.'),
            ('thm-ym-native-multiplication-locality', 'Exact locality identity in the native multiplication representation.'),
        ],
    ),
}

OVERRIDE_PARAGRAPHS = {
    4: r"""For the conservative process, the drift and common-mass estimates give convergence to its identified invariant law. For the killed process, the surviving-block results give the selected QSD and its survival eigenvalue. The finite-window and Doob constructions specify which conditioned history is observed.

The recorded observation time is \(t_n=n\Delta t\), where \(\Delta t\) includes any configured recording stride and fixed physical-unit calibration. Burn-in advances this same dynamics to the selected equilibrium ensemble. Subsequent windows retain their algorithmic time differences and all recorded transition data.

The Yang--Mills chapter computes the selected QSD window weights and its relaxation error. The equilibrium law supplies the starting ensemble; the complete transition supplies every temporal correlation used below.
""",
    17: r"""At equilibrium, retain the complete gauge and matter descriptor hierarchy generated by the three-dimensional gas and its recorded time, using the \hyperref[sec:equilibrium-thermodynamics]{population equilibrium} specified above. The exact field-law and transition isomorphisms give its multi-time expectations. The fluctuation and native-hierarchy theorems supply moment bounds, test-space control, and common-subsequence correlation limits.

The Euclidean-regularity theorem carries actual moment and reconstruction bounds to tempered correlations. Commuting bosonic insertions give their permutation symmetry, and the exterior/CAR construction supplies the graded rule for the recorded matter operators. The local-net results establish isotony, the stated locality identities, and covariance under the identified same-law implementers.

The final physical representation is built from this equilibrium temporal hierarchy. Its reflection is in the algorithmic observation clock, and its spatial fields remain on \(\mathbb R^3\). The source's positive-transfer and quantum-reconstruction statements specify the representation hypotheses used to construct the physical Hilbert space, vacuum, and self-adjoint Hamiltonian. Those hypotheses remain attached to the corresponding field-axiom conclusions.
""",
}

OVERRIDE_SKETCHES = {
    17: r"""First transport the actual stationary history through the lossless descriptor map. The source's operator-product formula computes each time-ordered correlation using the original kernel. Apply the existing test-space, moment, and uniform-integrability bounds to the same retained observable algebra. For temporal reconstruction, form the future algebra on \(t\ge0\) and its clock-reflected pairing. The source's positive-transfer criterion, when identified for this hierarchy, gives the required positive semigroup representation. Quotient null vectors, complete the resulting Hilbert space, and identify physical time translation and its generator. Apply the local-net and field-axiom statements with their covariance, locality, spectrum, and vacuum hypotheses. This composes the established native construction with the physical representation step on its actual equilibrium law.""",
}

CHAPTER_GROUPS = [
    ('Microscopic definition and configured variants', '1_the_algorithm/01_algorithm_intuition.md; 02_fractal_gas_latent.md; 03_parameter_constraints.md; 04_gas_variants.md; convergence_program/01_fragile_gas_framework.md; 02_euclidean_gas.md', 'Complete state, update order, parameter floors, metric/noise choices, and each variant\'s own transition.'),
    ('Cloning, transport, and recurrence', 'convergence_program/03_cloning.md; 04_single_particle.md; 04_wasserstein_contraction.md; 05_kinetic_contraction.md; 06_convergence.md; 06a_structural_landscape_convergence.md; 07_discrete_qsd.md', 'Keystone pressure, reset and signed drift, barycenter/shape decomposition, kinetic estimates, moment/tail budgets, Harris and QSD convergence, quantified population/time limits.'),
    ('Population law and exchangeability', 'convergence_program/08_mean_field.md; 09_propagation_chaos.md; 11_hk_convergence.md; 12_qsd_exchangeability_theory.md', 'Fixed-step rooted population map, exact balances, particle consistency, survival normalization, mass--shape convergence, empirical mixtures, fixed-row chaos.'),
    ('Entropy, quantitative errors, and regularity', 'convergence_program/10_kl_hypocoercive.md; 13_quantitative_error_bounds.md; 14_a_geometric_gas_c3_regularity.md; 14_b_geometric_gas_cinf_regularity_full.md; 15_kl_convergence.md; 17_geometric_gas.md', 'Poincare and LSI, full-update entropy identities, hypocoercivity, observable error, normalized derivative calculus, smooth empirical fields, identified adaptive-law bounds.'),
    ('Causal and geometric continuum', 'convergence_program/16_continuum_discharge.md; 2_fractal_set/02_causal_set_theory.md; 3_fitness_manifold/01_emergent_geometry.md; 02_scutoid_spacetime.md; 03_curvature_gravity.md', 'Recorded causal order, slabs and trajectory clocks, regular metric coefficients, sampling correction, two estimator-error routes, operator reconstruction, holonomy and curvature comparison.'),
    ('Exact derived finite fields', '3_fitness_manifold/04_field_equations.md', 'Marked characteristic hierarchy, exact density/current balances, conditional metric law, drift/covariance, field memory and filtering, coupled population/mechanical/metric system.'),
    ('Recorded gauge and matter theory', '2_fractal_set/01_fractal_set.md; 03_lattice_qft.md; 04_standard_model.md; 05_yang_mills_noether.md; 3_fitness_manifold/08_voronoi_wilson_loops.md', 'Lossless history, invariant coordinates, CAR dynamics, native action and sources, Ward identities, gauge hierarchies, Wilson consistency, spectral-gap passage.'),
    ('Spectral channels and empirical checks', '2_fractal_set/06_empirical_validation.md; 07_qft_calibration_report.md; 08_twistor_formulation.md; 09_qft_calibration.md; partvi_experiments.md', 'Implemented channel and correlator definitions, algorithmic-time lag, source-grounded calibration, positive-transfer spectral criterion, observable diagnostics.'),
    ('Response, measurement, and geometric extensions', '3_fitness_manifold/05_holography.md; 06_cosmology.md; 07_computational_proxies.md; 09_measurement.md', 'Actual-kernel path KL and stationary Poisson response, graph-cut and boundary observables, equilibrium centering and coarse closure, computational geometric probes, reference-transport covariance for measurement. These chapters develop additional observables and extensions of the same recorded system.'),
]

CONSTRUCTION_SECTION = r"""
\clearpage
\section{How the proved walker dynamics supply the field theory}
This section collects the identifications used throughout the chapters into one dependency chain. Its starting point is the complete history of the three-dimensional gas at the recorded times \(t_n=n\Delta t\).

\subsection{Probability law, invariant descriptors, and frame observables}
The empirical walker law is
\[
L_N=\frac1N\sum_{i=1}^N\delta_{(x_i,v_i,a_i)}.
\]
The direct field law instead retains the full specified observable descriptor \(\mathscr O\) of the complete history:
\[
\mu_{\rm dir}=\mathscr O_*\mathbb P_{\rm rec},\qquad
\langle f\rangle_{\rm dir}
=\mathbb E_{\mathbb P_{\rm rec}}f(\mathscr O(\mathcal F)).
\]
The direct three-color definition sets \(d=3\) and constructs its masked vector from the actual viscous force and velocity:
\[
\widetilde c_i^a=F_i^{{\rm visc},a}e^{i\kappa v_i^a},\qquad
c_i=m_i\frac{\widetilde c_i}{\max(\|\widetilde c_i\|,\delta_c)}.
\]
The Gram and complex determinant coordinates identify the common internal-frame orbit and preserve its invariant history correlations. Full coordinates and masks are retained before forming the particular frame averages. A full-rank internal anchor chart is one explicit inverse chart; the full orbit theorem also covers the rank-deficient strata.

At a recorded frame, the book distinguishes the valid-weight average and its fixed-normalization companion:
\[
\mathcal A_t(O)=\frac{\sum_Iw_Im_IO_I}{W_t},\qquad
\mathcal A_t^N(O)=\frac1N\sum_Iw_Im_IO_I
=\frac{W_t}{N}\mathcal A_t(O),\qquad W_t=\sum_Iw_Im_I,
\]
with the recorded zero-denominator convention. Each is an observable of the same full history and has its own specified correlator. The physical meaning is read from the chosen descriptor and its temporal law.

\subsection{The history correlations retain the algorithmic clock}
For the selected invariant law \(\pi\), the lossless record theorem supplies
\[
\widehat P_hU=UP_h,\qquad
\widehat L=ULU^{-1}
\]
on the stated domain. The finite-step history formula and frame-correlator identity then give all retained correlations under the actual cloning and kinetic update. Cloning remains part of the evolution. Projection to a few channels uses the derived memory or the prediction-complete descriptor. Burn-in changes the initial law toward equilibrium; field time continues to be the calibrated recorded observation time.

\subsection{Geometric integrals and fluctuations have derived normalizations}
For the selected sampling law \(q\,d\mathrm{vol}_g\), the inverse-density formula converts normalized samples into geometric integrals. The normalized \(C^n\) calculus controls their coefficient fields. Joint LSI gives the direct empirical-gradient variance estimate, and the covariance route gives the alternative temporal/sampling estimate.

The centered hierarchy uses \(\sqrt N\) scaling around the actual stationary expectation. Its distribution-valued compactness and moment bounds retain the collective fluctuation information. The native gauge hierarchy separately retains the complex color algebra and its sourced history law. These constructions supply the background, fluctuations, and gauge correlations as distinct derived objects.

\subsection{Native likelihood and conditional geometry}
Condition the complete path likelihood on the descriptor and take its negative logarithm. This yields the field action, predictive kernel, and generating functional. The native source identities differentiate the executed update, and geometry-fiber disintegration preserves the conditional denominator. Recombination therefore returns the same sourced correlations. The continuum comparison uses the derived spatial geometry and the recorded time slices.

\subsection{Equilibrium representation and the gap}
The source specifies quantum reconstruction through the equilibrium temporal hierarchy and its stated physical representation hypotheses. The resulting transfer Hamiltonian uses that clock and observable law. The mass-gap passage then applies the proved self-adjoint gap-survival theorem with its uniform bound, embeddings, vacuum limit, and strong semigroup convergence.

\begin{criterion}[End-to-end composition]
For the configured \(3+1\) recorded family, apply the existing moment, population, regularity, sampling, and native-hierarchy results under their stated hypotheses. Identify its limiting equilibrium hierarchy with the physical Yang--Mills representation using the source's field-axiom and positive-transfer hypotheses. If the resulting Hamiltonians satisfy the uniform gap and convergence hypotheses of the gap-survival theorem, then
\[
\Spec(H)\subset\{0\}\cup[\lambda_*,\infty),\qquad
\ker H=\mathbb C\Omega,\qquad\lambda_*>0.
\]
The calibrated energy gap is at least \(\hbar_{\rm eff}\lambda_*\).
\end{criterion}
The criterion composes the source results. Representation hypotheses remain in their statements, while the population dynamics, normalizations, field equations, and inherited correlations are the constructions already derived in the chapters.
"""
