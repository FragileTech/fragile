# Mathematical Review: 05_yang_mills_noether.md

## Metadata

- Reviewed file: `docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md`.
- Review date: 2026-10-02.
- Reviewer: independent physical/spectral-chain audit, including adversarial review of this reviewer's earlier finite gauge-sector insertions.
- Scope: linear review of the complete chapter; detailed checks of the new finite gauge certificate and its proposed native-chain role; trace of native likelihood, source, state/law, geometry, physical-clock, reflection, locality, covariance, and spectrum dependencies. Cross-chapter checks cover the relevant construction anchors, not every claim in each other chapter.
- Source snapshot: working-tree source on the review date; line references are supplemented by labels because neighboring chapters are being edited by other agents. This audit changed no source, core code, algorithm, or parameter configuration.
- Framework anchors (definitions/axioms/permits):
  - Chapter 01 `def-fractal-set-record-coverage`, `thm-fractal-set-lossless`, `prop-fractal-set-analytic-transfer`: covered same-record encoding preserves the actual law, operator, and energy form; it does not supply new dynamics.
  - Chapter 04 `def-sm-direct-observable-law`, line 294; `prop-sm-direct-law-symmetry`, line 1438; `thm-sm-instantiated-record-transition`, line 2737: actual three-component color, common-frame versus local comparisons, and the implemented full-step law.
  - Chapter 04 `thm-sm-path-descriptor-density`, line 3958; `thm-sm-effective-recorded-gauge-dynamics`, line 4021; `prop-sm-recorded-color-connection-curvature`, line 4231: native pushforward effective density, history-dependent predictive kernel, and descriptor-space color-line connection.
  - Chapter 03 `thm-lqft-edge-second-quantization`, `thm-lqft-edge-filled-ground-gap`, `thm-lqft-edge-parity-ground-gap`: the existing specified finite matrix and exact spectral algebra.
  - Chapter 03 `thm-lqft-record-car-channel` and `thm-lqft-product-obstruction`: the actual completely positive recorded evolution and the distinction between classical products, replica CAR products, and unitary dynamics.
  - This chapter's complete path-density construction, lines 322–560; same-record metric bounds, lines 724–1027; native source/response hierarchy, lines 1687–4046; proposed weak connection identity, lines 4158–4584.
  - Same-law and reconstruction restrictions: Doob law at lines 4770–4791; generator/clock and transfer-gap conditions at lines 5181–5322; joint-LSI restrictions at lines 5903–6137; selected QSD-window law at lines 6138–6405; pullback-spectrum obstruction at lines 6585–6646.
  - Physical/readout restrictions: four-position convention at lines 7320–7355; same-record color/embedding requirements at lines 7839–7859; actual reflection test at lines 8105–8384; scalar face evaluation at lines 8385–8483; translation and normalization obstructions at lines 8562–9005; regional CAR/HK calculation at lines 9406–9739.
  - Fitness-manifold Chapter 03 `def-curvature-conditional-fitness-field` and `assump-curvature-geometric-setting`: conditional Hessian metric and additional spacetime comparison are distinct constructions.
  - Fitness-manifold Chapter 04 `thm-algorithmic-explicit-transition`, `thm-algorithmic-field-characteristics`, `thm-algorithmic-coupled-field-system`, and `ass-thermal-equilibrium`: exact full update/conditional field characteristics do not themselves identify a physical Einstein or equilibrium quantum law.
  - Fitness-manifold Chapter 05 `thm-qsd-gibbs`, `prop-fluctuation-dissipation`, and `rem-algorithmic-three-geometries`: Gibbs identification needs the actual eigenmeasure; static exponential tilt is not general algorithmic response; fitness Hessian, control Fisher information, and physical geometry are distinct.

## Executive summary

- Critical: 0.
- Major: 4.
- Moderate: 3.
- Minor: 0.
- Notes: 1.
- Primary themes: the new finite certificate and its invariant-mode corollary are valid conditional algebra. They do not establish an actual native local gauge representation, native physical time/energy, or a nontrivial physical vacuum. The existing chapter already proves several obstructions to literal identifications; those must remain live constraints on the completion claim.

No false main theorem was found in the new finite gauge-sector proofs. A record-derived observable operator is a valid mathematical object even when it is not the executed Markov generator. Likewise, trivial gauge action on gauge-invariant observables is not a contradiction. The report distinguishes these valid constructions from missing native discharges and from terminological/unit overreach in the completion ledger. It does not reject the reconstruction programme in general, nor replace the already defined Hamiltonian.

The unchanged-target instruction matters: a conditional theorem is not discharged by newly assuming its desired connection identity, positivity, same-law symmetry, or transfer Hamiltonian. Choosing a different state, representation, observable normalization, or clock can also change the target even if no simulation step is edited.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | 7041–7061, 7173–7248; roadmap 56 | Moderate | Miswording; secondary: Scope restriction | A computable commutation certificate and the valid \(u_g=I\) specialization do not instantiate the native local gauge action. |
| E-002 | 6523–6646, 6925–7039, 9485–9739 | Major | Scope restriction | Literal same-law pullback/exterior translations cannot yield a nontrivial forward-cone representation with a unique invariant vacuum. A separate finite edge gap does not remove this obstruction. |
| E-003 | 7173–7199 | Moderate | Scope restriction | Character twisting fixes a filled ground but changes the invariant-vector sector; equality of observable conjugation is not original-sector identification. |
| E-004 | 1687–4046, 4158–4584; Chapter 04: 4231 | Major | Proof gap / omission | Native Gaussian-force response is not yet the weak local non-Abelian connection variation or the pure Yang–Mills action. |
| E-005 | 2377–2403, 7320–7355, 7839–7859 | Major | Definition mismatch; secondary: Dimensional mismatch | The four-position physical-coordinate convention is not supplied by the proven three-spatial-coordinate/color variant and uses a different clock identification. |
| E-006 | 8105–8384, 8385–8483, 8562–9005 | Major | Scope restriction | Actual reflection/translation/nontriviality tests remain unresolved or obstructed for specific fixed readouts; compactness and finite positive gap cannot replace them. |
| E-007 | 2377–3108, 3715–4046 | Moderate | Scope restriction | Retaining projected currents and source densities in an augmented limit proves a valid enlarged probabilistic limit, not their identification as functions of the originally targeted geometric field. |
| N-001 | 7041–7248 | Note | Scope restriction | Commuting-projector, spectral-trace, Haar-excitation, parity, and finite cyclicity calculations check correctly. |

## Detailed findings

### [E-001] A certificate is not its native instantiation

- Location: `thm-ym-edge-gauge-spectral-certificate`, lines 7041–7061; `cor-ym-edge-filled-gauge-closure`, lines 7173–7248; completion ledger at `docs/research/volume_2_remaining_proof_steps.md`, line 56.
- Severity: Moderate.
- Type: Miswording (secondary: Scope restriction).
- Target inference being tested: whether the exact finite commutation/excitation theorem discharges commutation for the intended native physical gauge sector.
- Why this does not discharge the target:
  1. The theorem declares a continuous unitary representation \(u_g\) on the same finite mode space and asks for the exact test \(u_gh=hu_g\) for every \(g\). Its conclusions follow once that certificate is verified. “Not a new assumption on the gas” correctly distinguishes a diagnostic from a modified simulation rule, but the conclusions still require the certificate to pass for the actual intended representation.
  2. No native local gauge action on the complete gas record, induced finite modes, or corresponding coefficient matrix is evaluated in the new theorem. The generic numerical matrix helper does not compute \(u_g\), commutators, or native gauge-sector traces.
  3. The corollary's \(u_g=I\) case is mathematically valid when every selected mode is already invariant under an actual same-law action. Trivial gauge action on physical observables is expected and is not a contradiction. But selecting only invariant scalar functions does not construct or verify the underlying local non-Abelian action, its record-law invariance, its Gauss/physical sector, or its physical dynamics.
  4. Chapter 04 `prop-sm-direct-law-symmetry` explicitly distinguishes common internal transformations from independent local frames. The untransported overlap transforms as \(c_i^\dagger A_i^\dagger A_jc_j\); local comparisons need the already identified links and a verified full-record action. Algebraic cancellation in a contraction alone does not prove kernel/initial/survival-law invariance.
- Impact on downstream results: the finite theorem may be counted as an exact available certificate, not as a completed native local gauge identification. The ledger phrase “actual same-operator invariant modes discharge commutation identically” needs a concrete instantiated mode/action reference or the corollary's conditional scope.
- Fix guidance (step-by-step):
  1. Preserve the finite theorem and \(u_g=I\) corollary.
  2. Describe them in the ledger as certificate and specialization, unless the actual record action and selected invariant modes are explicitly instantiated.
  3. Keep local gauge-law and generator/sector identification open; do not supply them by declaring a new representation or adding a positive transfer premise.
- Required new assumptions/permits: none. The necessary native action/commutation verification must be derived from the unchanged recorded law, not postulated.
- Framework-first proof sketch for the fix: use the original record action, if defined, to induce the mode representation, then compute its commutator with the actual coefficient recipe. For already invariant modes the commutator is zero by definition, but this must not be promoted into a theorem identifying local Yang–Mills dynamics.
- Validation plan: require an end-to-end record/action/mode/coefficient calculation; distinguish the Haar average over an operator representation from imposing a Haar law on field links. The former is valid in the theorem; the latter would be a new physical measure.

### [E-002] The literal recorded translation representation has a spectral obstruction

- Location: `lem-ym-pullback-translation-spectrum`, lines 6585–6646; positive-energy premises at lines 6925–7039; native HK/CAR instantiation at lines 9485–9739.
- Severity: Major.
- Type: Scope restriction.
- Target inference being tested: whether the actual same-law recorded pullback representation or its established even exterior lift is already a nontrivial physical positive-energy representation once the finite edge Hamiltonian is shifted to have a positive gap.
- Why this does not discharge the target:
  1. The existing lemma proves \(JU(a)J=U(a)\) for coordinate conjugation and pullback translations. Hence its joint spectral projections satisfy \(JE(B)J=E(-B)\).
  2. A spectral measure supported in the forward cone and symmetric under \(p\mapsto-p\) is supported at zero. All translations then act trivially. A unique translation-invariant vacuum makes the Hilbert space one-dimensional.
  3. The actual regional \(L^2_0\) modes are conjugation closed. Their even exterior lift inherits the same obstruction. Exact Fractal Set encoding transports rather than removes it.
  4. Shifting \(d\Gamma(i(K-K^{\mathsf T}))\), filling its negative modes, or computing a finite positive spectral gap does not change the already defined spacetime translation implementers. If a different implementer/state is used, its identification with native physical-time correlations remains to be proved.
- Impact on downstream results: the full positive-energy, covariance, uniqueness, and nontriviality endpoint cannot be completed in these literal representations. This is an existing proved restriction, not a general no-go theorem for all possible reconstruction. The completion ledger must not use the edge gap as if it bypassed it.
- Fix guidance (step-by-step):
  1. Retain and cite this obstruction whenever the recorded pullback or even exterior implementation is proposed as the physical representation.
  2. Keep native physical reconstruction open; do not add positive energy, reflection positivity, or a new transfer Hamiltonian as an assumption and call it discharged.
  3. If an already specified reconstruction route has different physical implementers, verify its actual native correlations, operator identification, state, and time before applying the gap theorem.
- Required new assumptions/permits: none. The obstruction is already proved internally. A new representation would be a further construction needing justification, not a silent repair.
- Framework-first proof sketch for the fix: intersect the two spectral supports in the existing lemma, then test the same implementers used downstream. A finite nonzero gap of another operator is irrelevant to that intersection without an intertwining.
- Validation plan: list in one place the operators implementing physical translations, physical time, the vacuum state, and the native regression correlations. All covariance/spectrum/gap claims must use that same identified representation.

### [E-003] Phasing the implementation changes the fixed-vector sector

- Location: `cor-ym-edge-filled-gauge-closure`, lines 7173–7199.
- Severity: Moderate.
- Type: Scope restriction.
- Target inference being tested: whether the character twist is a closure of invariance in the originally chosen gauge sector.
- Why this does not discharge the target: the theorem correctly computes
  \[
  \Gamma_g\xi_-=\chi_-(g)\xi_-,\qquad
  V_g=\overline{\chi_-(g)}\Gamma_g.
  \]
  Scalar phases cancel in conjugation of observables, but the fixed-vector condition changes from \(\Gamma_g\eta=\eta\) to \(\Gamma_g\eta=\chi_-(g)\eta\). The projectors \(\int\Gamma_g\,dg\) and \(\int\overline{\chi_-}\Gamma_g\,dg\) need not agree. The corollary explicitly acknowledges this; any completion claim must preserve the acknowledgement. The original physical charge/Gauss sector is not determined solely by the conjugation action.
- Impact on downstream results: an invariant filled ground in the phased representation does not prove that the original untwisted physical sector contains a ground, an excitation, or a unique vacuum.
- Fix guidance (step-by-step):
  1. Keep the character twist as a correctly labeled representation option.
  2. Do not use that option as an automatic native-sector replacement.
  3. Compute the determinant character in the actual intended action and use its original projector for the native claim.
- Required new assumptions/permits: none. Choosing a new character sector would be additional target data, not a consequence of observable conjugation equality.
- Framework-first proof sketch for the fix: compare the two Haar projectors on the filled ground. If \(\chi_-\ne1\), the untwisted average of that vector is zero while the phased average fixes it.
- Validation plan: state the original charge sector before calculating its ground/excitation traces. Apply the full \(I-P_0\) test, not merely a variance around one vector when zero modes make the ground degenerate.

### [E-004] Source response has not been identified with connection variation

- Location: native source hierarchy, lines 1687–4046; `Native response and the weak connection-variation identity`, lines 4158–4291; source tangent and metric response, lines 4292–4584; Chapter 04 `prop-sm-recorded-color-connection-curvature`, line 4231.
- Severity: Major.
- Type: Proof gap / omission.
- Target inference being tested: whether the proved adaptive Gaussian-force likelihood identities provide the missing local Yang–Mills connection and field equation for the unchanged gas.
- Why this does not discharge the target:
  1. The source is an addressed mean shift of the native O-stage innovation, propagated through the original maps. At zero source it is the original law. Its exponential likelihood, projected score, entropy/Fisher identities, and selected-law normalization are real native response calculations.
  2. They do not yet supply a connection-space vector field \(X_f\) such that
     \[
     \int X_f O\,d\mu_g=\int O\,s_f\,d\mu_g.
     \]
     The chapter states this as the missing weak variation identity rather than proving existence of \(X_f\).
  3. A nonzero force source moves the positions after O by the displayed \(h^2\)-scale endpoint shift. The native geometry is recomputed from those positions and inputs. This is not a fixed-geometry connection variation merely because the source was placed in the Gaussian thermostat.
  4. On an available smooth coordinate chart, the density calculation includes the reference divergence term \(s_f=X_f S-\operatorname{div}_\lambda X_f\). Dropping it or identifying the effective native action with the Wilson functional is a further missing step.
  5. The original color-phase connection in Chapter 04 is the \(U(1)\) connection \(-ic^\dagger dc\) on the descriptor parameter space \((F,v)\), with a genuine recorded state-surface realization. It is explicitly not a new independent \(SU(3)\) spacetime link. Its validity does not discharge the local non-Abelian spacetime connection/action identification.
- Impact on downstream results: pure Yang–Mills EOM, its local Ward/constraint interpretation, physical field identification, and a physical gauge-sector mass cannot be declared complete from the source curves or a Wilson Taylor formula alone.
- Fix guidance (step-by-step):
  1. Retain the exact source likelihood/response results as proven.
  2. Keep the weak tangent, geometry variation, reference divergence, and native action identification as explicit remaining steps.
  3. Do not alter the original force, link readout, probability law, or introduce a new local connection to manufacture this identity.
- Required new assumptions/permits: none. Smooth chart/tangent/action hypotheses already displayed in conditional statements are not newly discharged by the finite spectral theorem.
- Framework-first proof sketch for the fix: compute an actual source tangent of the existing descriptor map, including geometry/masks/boundaries, then verify the weak identity against the native conditional density. Only after this calculation could the original effective action's first variation be compared to an existing identified local gauge action.
- Validation plan: compare the full native score with the proposed connection-space derivative for a separating family of observables; retain geometry motion and reference-measure terms. A source-deformed diagnostic family does not change the base algorithm, but is not itself proof that the base dynamics is Yang–Mills.

### [E-005] The two four-coordinate constructions are not the same native spacetime

- Location: source-current definition, lines 2377–2403; `prop-ym-recorded-physical-transformations`, lines 7320–7355; same-record color application, lines 7839–7859.
- Severity: Major.
- Type: Definition mismatch (secondary: Dimensional mismatch).
- Target inference being tested: whether the proved three-spatial-coordinate color/geometry variant already instantiates the physical reconstruction section's four-position convention.
- Why this does not discharge the target:
  1. The source current for \(d=3\) lives on test coordinates \((kh,x)\in\mathbb R^{1+3}\): recorded sampling time plus three position coordinates.
  2. The physical readout section instead assumes a record with \(x\in\mathbb R^4\) and a separate recorded \(\tau\), with physical projection \(p_{\rm phys}(\tau,x)=x\). It explicitly says \(x^0\ne\tau\) as stored coordinates. That is four position coordinates plus a separate sample clock, not the first convention.
  3. The native direct color formula is three-component on the established three-coordinate variant. Chapter 04 requires a separately specified \(\mathbb C^3\) readout for another positional dimension; the Chapter 05 application correctly propagates that requirement.
  4. The current discharge proves the three-spatial-coordinate reference variant, not a new four-position execution or a proved identification of its time coordinate with the sampling clock. Relabeling \(kh\) as \(x^0\), or projecting four-dimensional force into three color components, is not the unchanged already-proved readout without a derivation.
- Impact on downstream results: physical reflection, Poincaré/translation tests, local color hierarchy, and source/connection limits cannot be combined across these coordinate conventions as if they referred to one established native object.
- Fix guidance (step-by-step):
  1. Keep each theorem's coordinate convention visible in the dependency ledger.
  2. Mark the four-position/three-color same-record application uninstantiated for the presently discharged variant.
  3. Do not change dimension, add a new color map, or choose a new clock to close the target under the present instruction.
- Required new assumptions/permits: none. The needed same-record/time identification remains open rather than being added as a hypothesis.
- Framework-first proof sketch for the fix: compare the two embeddings and their field/source test domains before transferring any law or generator. Encoding can recover stored coordinates; it cannot supply a missing fourth stored position or prove its physical-clock role.
- Validation plan: one native parameter/record register must determine dimension, color map, geometry, sample clock, physical time, field support, and reflection plane for every downstream theorem.

### [E-006] Existing positivity, translation, and nontriviality tests cannot be bypassed

- Location: reflected-product test at lines 8105–8384; native scalar face evaluation at lines 8385–8483; same-law/background translation and normalized-field tests at lines 8562–9005.
- Severity: Major.
- Type: Scope restriction.
- Target inference being tested: whether bounded physical readout compactness and a finite edge spectral gap complete the physical field/reconstruction chain for the existing observables.
- Why this does not discharge the target:
  1. Actual reflection positivity is a law-and-product inequality. The chapter's exact decomposition requires the appropriate Hermitian reflected Gram matrix and its positive inequality; positive face weights, ordinary Gram positivity, and law reflection symmetry alone do not prove that inequality.
  2. For the raw scalar phase connection, the outer plaquette holonomy is identically one, so its Wilson-defect field is identically zero. Triangle observables can differ. Substituting triangles or matrix/color marks changes the specific readout unless the target already selected them.
  3. For the existing globally normalized nonnegative gauge readout, every limiting field is a finite nonnegative measure of mass at most two. The chapter proves that an all-translation-invariant physical law of that field forces the field to vanish almost surely. A nonzero, translation-invariant physical hierarchy cannot be obtained for that literal normalized object.
  4. A probability law retaining an absolute-position anchor cannot be invariant under all translations. Transporting a confining background and its QSD yields covariance of a family, not same-law translation invariance at one background. The chapter also supplies a negative reflection test for a translated finite-QSD family. This is a restriction of that family, not proof that every possible continuum field fails reflection positivity.
  5. A centered fluctuation field, relative-coordinate regulator, different normalization, or differently reconstructed quantum algebra might have different properties. None can silently be substituted under an unchanged-target completion claim.
- Impact on downstream results: nontriviality, physical translation covariance, reflection positivity, and hence physical quantum reconstruction remain genuine steps. They are not solved by adding a finite positive-energy matrix or by passing bounded coordinates to a subsequence.
- Fix guidance (step-by-step):
  1. Identify the exact original field/readout and law before stating the desired physical endpoint.
  2. Propagate each existing obstruction to that same field's downstream claims.
  3. Keep unproved positivity/nontriviality routes open rather than inserting a new reference measure, normalization, cutoff family, or observable to evade the tests.
- Required new assumptions/permits: none. A new positive transfer premise or a different target field is specifically not an allowed repair.
- Framework-first proof sketch for the fix: use the already proved field evaluations, finite-mass translation argument, anchor test, and reflected-product decomposition. Check the actual target against each test instead of assuming its conclusion.
- Validation plan: require a nonzero positive reflected class of the specified same-record observable and actual law, plus the exact target covariance. Keep scalar versus color/matrix marks, triangles versus outer plaquettes, and normalized versus fluctuation fields distinct.

### [E-007] Augmented compact limits are not yet the target field dictionary

- Location: current and source-density limit construction, lines 2377–3108; geometry-fiber/source augmentation, lines 3715–4046.
- Severity: Moderate.
- Type: Scope restriction.
- Target inference being tested: whether the proved joint subsequential source hierarchy is already a physical local connection/geometry hierarchy of the original descriptor \(B\).
- Why this does not discharge the target: the proof retains projected current coordinates \(J\), source-density coordinates \(r\), and geometry-fiber density \(q\) inside the augmented limit, for example \(\widehat Y=(B,J,r,B_G,q)\). This is a legitimate way to pass conditional identities and likelihood curves to a joint limit. It does not prove \(J,r,q\) are determined by \(B\) alone, nor that \(B\) is a geometrically regular local connection with the required topology and field products. Conditioning on an augmented variable that retains the conditional projection is not the same identification as conditioning on only the intended physical connection field.
- Impact on downstream results: probabilistic compactness and a valid source response hierarchy are established; the native local field/action dictionary and loss of extraneous history information are not.
- Fix guidance (step-by-step):
  1. Keep the augmented descriptor named whenever its limit identities are used.
  2. Do not replace it by the original physical descriptor in later statements without a proved sufficiency/dictionary map.
  3. Preserve the original record law and target; do not declare the retained auxiliary coordinates to be a new physical field merely to close the chain.
- Required new assumptions/permits: none. Measurability, locality, regularity, and sufficient physical field coordinates must be derived for the already identified target.
- Framework-first proof sketch for the fix: use the existing predictive/sufficient-descriptor machinery to test whether the selected original field determines the retained projections and source curves; then prove the relevant geometric/operator convergence. The present joint limit alone does not imply this.
- Validation plan: explicitly state the conditioning sigma algebra for every limiting score/current/action identity and track the original target's products and time evolution through the dictionary map.

### [N-001] The finite gauge certificate algebra checks correctly

- Location: lines 7041–7248.
- Severity: Note.
- Type: Scope restriction.
- Verified conclusions: when the declared representation commutes with the specified matrix, its exterior lift commutes with the Fock Hamiltonian and parity; Haar averaging is the invariant projection; traces of commuting energy/gauge/parity projections count retained multiplicities; the complete-ground positive gap is inherited only if that ground is retained; a sector whose minimum is positive needs its own next-level difference; full even invariant-algebra cyclicity follows from invariant rank-one operators; the averaged excitation norm subtracts the full ground projection; and the determinant character/twist and invariant-mode parity formulas are correct.
- Important qualifications preserved by the theorem: no uniform random-record gap; no excitation is not a physical massive particle; a smaller regional algebra has its own cyclicity test; degenerate ground directions must not be counted as excitations by subtracting only a scalar expectation.

## Scope restrictions and clarifications

### Linear coverage register

| Source region | Checked content | Review outcome |
|---|---|---|
| 1–723 | Scope, gauge/frame definitions, actual path density, connections and native effective action | Native density is not assumed Wilson/Haar; passive covariance and active symmetry remain distinct. |
| 724–1686 | Same-record metric/noise bounds, Itô/balance comparisons, temporal gauge, Wilson readout | Branch/mask/success restrictions remain explicit; fitness balance is not automatic Noether conservation. |
| 1687–2376 | Gaussian/addressed adaptive sources and likelihood/entropy bounds | Exact diagnostic source response, not yet connection variation. |
| 2377–3714 | Current distributions, augmented source limits, occupation/Fisher identities, orthogonal sources, projection and entropy | Valid same-law projected hierarchy; physical local field identification not supplied by augmentation. |
| 3715–4584 | Geometry-fiber conditioning, Wilson response, weak connection identity, source tangents, chart divergence | The structural connection/action step remains explicitly conditional/open. |
| 4585–5344 | Ward comparison, native partition function, law/Doob distinctions, Wilson continuum quadrature, units, gap-transfer inputs | Conditional results are not instantiated by finite QSD control or a finite edge spectrum alone. |
| 5345–6522 | Classical fields, LSI fluctuation limits, full generator/finite-step characteristics, cloning/Gibbs obstruction, selected QSD histories, momentum tests | Laws and time scales are distinguished. Joint stationary LSI remains an actual-law input, not implied by marginal tails or kinetic-stage estimates. |
| 6523–7040 | Actual replica generator, covariance, pullback-spectrum obstruction, operational causality, positivity and generic gauge gap | Literal representation restrictions persist. Capped kinetic substeps do not establish all-step relativistic causality. |
| 7041–7248 | New finite gauge certificate and character/invariant-mode corollary | Algebra passes; native action/operator/state instantiation remains separate. |
| 7249–8104 | OS form, physical embedding, native readouts and bounded color/geometry hierarchy | Four-position versus sampling-clock distinction and same-record color availability are explicit. |
| 8105–9005 | Actual reflected matrices, scalar face evaluation, global dependence, translation/RP/normalization/anchor tests | No automatic RP or nontrivial translation-invariant fixed-normalization field. |
| 9006–9739 | Local algebra conditions, symmetry/vacuum, signed covariance, exact regional even-CAR locality, HK instantiation | Full quantum locality and spectrum require the actual net/representation; geometric separation alone is not sufficient. |
| 9740–9896 | Synthesis, glossary, remaining conditions | Conditional scope and obstructions are largely accurately retained; ledger terminology must match them. |

### Hypothesis and law provenance

| Object/input | Actual provenance | What it does not authorize |
|---|---|---|
| Finite full-step population law | Complete configured cloning/kinetic/boundary transition | Replacing it by a reversible Gibbs sampler or a finite-rate continuous jump law. |
| Killed kernel \(Q\), QSD \(\nu\), survival weights | Actual killed process and its eigenproblem | Treating a terminal-survival path as stationary conservative dynamics. |
| Interior QSD-window marginals | \(\nu\) weighted by remaining-horizon survival function | Dropping future-survival normalization or treating all window positions as \(\nu\). |
| Doob kernel and invariant law | \(P^\eta=e^\alpha Q\eta/\eta\), stationary \(\eta\nu\) under its stated normalization | Identifying \(\eta\nu\) with \(\nu\), or importing an LSI from another law. |
| Conservative recorded CAR process | Explicit stationary \(\pi\) and native conditional expectation | Claiming its hypotheses follow merely from a finite killed-chain QSD. |
| Same-law gauge action | Must preserve the full record law, initial law/kernel, and selected event where applicable | Using only common-frame contraction invariance as a proof of local dynamical symmetry. |
| Native likelihood/effective action | Full Gaussian/donor/gate/jitter law pushed to descriptor histories | Assuming it is the Wilson action relative to an independent Haar link reference. |
| Force source hierarchy | Original innovation/source response at zero deformation | Assuming the source is a fixed-geometry local connection tangent. |
| Conditional Hessian metric | Existing query-field/full-record reconstruction on successful branches | Identifying it with control Fisher information or a spacetime Einstein metric without derivation. |
| Finite \(H=d\Gamma(i(K-K^{\mathsf T}))\) | Existing specified edge readout plus chosen finite modes | Treating it as actual \(d\Gamma(L)\), reconstructed physical time, or an already calibrated energy rate. |
| Compact group projector | Declared representation and a verified commutation certificate | A new Haar field measure or a native Gauss-sector identification by declaration. |
| Uniform physical gap/limit theorem | Same identified self-adjoint family, vacuum, Hilbert-space maps, strong semigroup convergence | A pointwise random-matrix gap, non-self-adjoint sampling rate, or graph bias estimate as a substitute. |

## Proposed edits (optional)

No reviewed source was edited. The minimal fixed-target corrections concern the completion wording: keep the finite algebra and existing Hamiltonian; call the gauge result an exact certificate with a valid invariant-mode specialization; state the actual native local action/physical sector instantiation as open. Preserve the chapter's existing pullback-spectrum, reflected-product, normalization, and clock obstructions in every dependent summary. Do not add assumptions or substitute a different gauge law, measure, state, or observable to mark the chain complete.

## Open questions

1. What actual local action on the complete native record induces the intended physical gauge sector and commutes with the intended existing Hamiltonian, rather than merely acting trivially on selected scalar observables?
2. Which native physical-time reconstruction has the required positivity and correlations while respecting the proved literal pullback/exterior obstruction?
3. Can the weak source-to-connection variation be derived for the existing readout, including moving geometry, branch/mask effects, and reference divergence, without a new connection or analytic premise?
4. Which already declared spacetime convention and field normalization is the physical target? The three-space-plus-clock source current and four-position-plus-separate-clock readout cannot be combined by relabeling.
5. For that exact target, which existing reflected class is nonzero and positive, and how are covariance/locality established under the actual law rather than a substituted reference?
