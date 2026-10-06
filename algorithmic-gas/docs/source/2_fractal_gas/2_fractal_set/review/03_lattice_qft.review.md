# Mathematical Review: 03_lattice_qft.md

## Metadata

- Reviewed file: `docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md`.
- Review date: 2026-10-02.
- Reviewer: independent physical/spectral-chain audit, including adversarial review of this reviewer's earlier finite spectral insertions.
- Scope: linear review of the complete chapter, with particular scrutiny of the new finite spectral definitions and theorems, their implementation, and their use as a purported discharge of the unchanged native physical chain. Cross-chapter anchors were checked where they supply a claimed identification; this is not a complete review of those other chapters.
- Source snapshot: line references refer to the working-tree source on the review date. No source, algorithm, or configuration was changed during this audit.
- Framework anchors (definitions/axioms/permits):
  - `def-lqft-scope`, line 28: a finite recorded Fractal Set does not itself supply the separate continuum geometric comparison.
  - `def-fermionic-kernel-lqft`, lines 236–258: the antisymmetrized score is explicitly an auxiliary fermionic coupling, not an identified propagator or particle statistics.
  - `def-lqft-record-fock-space`, line 303, and `thm-lqft-replica-isomorphism`, line 545: the actual one-particle modes are centered functions of a complete swarm under a specified conservative stationary law; exterior products correspond to independent whole-swarm replicas.
  - `thm-lqft-edge-second-quantization`, lines 793–861: the finite lift of a specified matrix is exact, but equality with the recorded transition requires one-particle equality.
  - `thm-lqft-instantiated-word-evolution`, line 1125, and `thm-lqft-record-car-channel`, lines 1265–1428: the recorded dynamics is the contraction lift of the actual conditional-expectation operator and its completely positive CAR map.
  - `thm-lqft-product-obstruction`, line 1940: scalar same-record multiplication, CAR multiplication, and antisymmetrized independent-replica moments are different objects.
  - `def-fractal-set-record-coverage`, `thm-fractal-set-lossless`, and `prop-fractal-set-analytic-transfer` in Chapter 01, lines 1471, 1639, and 1956: covered encoding transports the same law and operator; it does not identify an arbitrary new operator with that operator.
  - `thm-sm-instantiated-record-transition` and `rem-sm-actual-step-and-clock` in Chapter 04, lines 2737 and 3203: the implemented transition retains every configured stage, and its per-step gates and phase clock cannot silently be replaced by a continuous-time rate model.
  - `lem-ym-pullback-translation-spectrum` in Chapter 05, lines 6585–6646: the literal same-law pullback representation, including its even exterior lift, has symmetric energy-momentum spectrum, obstructing a nontrivial forward-cone representation with a unique invariant vacuum.

## Executive summary

- Critical: 0.
- Major: 3.
- Moderate: 1.
- Minor: 0.
- Notes: 1.
- Primary themes: the new finite spectrum/parity algebra is correct for a declared finite matrix; it does not identify that matrix, its clock, or its filled-ground state with the unchanged recorded gas's physical generator and vacuum. A direct calculation of the chapter's actual unmasked score matrix also forces a large zero-mode space.

The findings below concern the **native-chain discharge audit**, including the associated completion ledger, not a claim that the finite occupation diagonalization is false. The chapter and roadmap explicitly leave physical identification and a uniform physical gap open. That is correct. Where the ledger calls the finite readout “native,” that term must mean record-derived, not executed or already physically identified. The major findings record missing target discharges and concrete restrictions, rather than an invalid main theorem. No new cutoff is needed for the finite calculation, but absence of a cutoff does not supply the missing operator identification.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Chapter 03: 793–891, 1265–1335; roadmap: 55, 489–498, 566, 600–605 | Major | Proof gap / omission; secondary: Scope restriction | The valid unitary edge readout is not identified with the native conditional-expectation generator; that target discharge remains open. |
| E-002 | Chapter 03: 303–329, 864–891, 953–956; diagnostic helper: 127–141 | Moderate | Definition mismatch; secondary: Dimensional mismatch | Mode space, matrix recipe, inner product, and time units remain declared readout data; Stage 8 drops the physical-units conditional. |
| E-003 | Chapter 03: 199–258, 930–950 | Major | Scope restriction | The actual unmasked score kernel has rank at most two after antisymmetrization: generic finite-matrix uniqueness does not apply to it for three or more modes. |
| E-004 | Chapter 03: 303–389, 944–956, 1002–1055; Chapter 05: 7173–7248 | Major | Scope restriction | Filling the negative modes changes the state used by the established recorded CAR correspondence; that correctly stated finite result does not discharge native vacuum identification. |
| N-001 | Chapter 03: 895–1108 | Note | Scope restriction | Finite occupation, zero-mode, parity, and full-even-algebra cyclicity formulas check correctly with their stated finite scope. |

## Detailed findings

### [E-001] The finite Hamiltonian is not identified with native evolution

- Location: `thm-lqft-edge-second-quantization`, lines 793–861; `def-lqft-edge-spectral-parameters`, lines 864–891; `thm-lqft-record-car-channel`, lines 1265–1335. The corresponding overclaim appears in `docs/research/volume_2_remaining_proof_steps.md`, the discharged row at line 55 and Stage 8 at lines 489–498, with “existing native Hamiltonian” and “finite native spectrum” at lines 566 and 600–605.
- Severity: Major.
- Type: Proof gap / omission (secondary: Scope restriction).
- Target inference being tested: whether the exact spectrum of the existing record-derived edge readout discharges the unchanged native generator/physical-spectrum identification. The theorem does not assert this equality and the roadmap leaves physical identification open; the test does not pass merely from the finite spectral result.
- Why this is an error in the framework:
  1. The native one-particle operator already identified in the chapter is \(C_t=P_t|_{L^2_0(\pi)}\), or the corresponding finite-step conditional-expectation operator. Its generator is the complete recorded \(L\), retaining cloning, companions, kinetics, and their interactions.
  2. The new operator is \(h=i(K-K^{\mathsf T})\), on a separately declared finite mode space. Its one-particle dynamics is \(e^{-ith}\), hence is unitary. The already proved theorem explicitly says equality with the recorded evolution requires equality of the one-particle operators; it does not prove that equality.
  3. The exact native CAR evolution has the multiplicative defect
     \[
     \mathcal Q_t(a(f)a^\dagger(g))-
       \mathcal Q_t(a(f))\mathcal Q_t(a^\dagger(g))
       =\langle f,(I-C_t^*C_t)g\rangle I.
     \]
     Whenever this defect is nonzero, it cannot be the observable conjugation evolution of any Hamiltonian on that same CAR algebra. Hamiltonian conjugation is multiplicative. The native construction supplies a completely positive channel, not automatically a closed-system physical energy generator.
  4. Exact encoding preserves this distinction. A unitary change of record coordinates transports \(C_t\) and the defect; it cannot turn a strict contraction into the separately selected unitary \(e^{-ith}\).
  5. In code, `fermionic_spectral_diagnostics` accepts an arbitrary real matrix supplied by its caller. It does not construct \(K\) from `RunHistory` or `FractalSet`, identify a native mode space, or verify an operator intertwining. Its numerical tests therefore verify the finite algebra, not the missing native identity.
- Impact on downstream results: neither a positive physical transfer Hamiltonian, its physical gauge sector, nor a physical mass gap follows. The gap-survival theorem cannot be fed these matrices simply because they are record-derived. An observable readout functional is not thereby the physical time generator.
- Fix guidance (step-by-step):
  1. Retain the finite result explicitly as a matrix/readout spectral calculation.
  2. Remove its classification as a completed native-generator or native-physical-gap discharge in the ledger and final handoff.
  3. Keep the missing same-operator/physical-time identification as an unresolved structural step. Before any later closure claim, compare the actual one-particle transition/operator on the declared native modes, not only its spectrum or an antisymmetric score.
- Required new assumptions/permits: none are permitted by the current request. Assuming \(C_t=e^{-ith}\), postulating a new transfer law, or selecting a different physical representation would not be a proof for the unchanged target.
- Framework-first proof sketch for the fix: use the existing one-particle restriction identity and the already calculated CAR defect. A nonzero defect rules out equality with Hamiltonian conjugation on that algebra; if a concrete proposed subspace has zero defect, its invariance, unitary action, and actual coefficient identity still must be calculated from the implemented kernel.
- Validation plan: check every “native Hamiltonian,” “physical gap,” and “discharged” occurrence against an explicit same-operator identity. A call to the generic matrix helper alone must not satisfy that check.

### [E-002] The declared spectral data do not complete the native parameter identification

- Location: `def-lqft-record-fock-space`, lines 303–329; `def-lqft-edge-spectral-parameters`, lines 864–891; physical units conditional at lines 953–956; `src/fragile/fractalai/theory/coupled_gas_diagnostics.py`, lines 127–141.
- Severity: Moderate.
- Type: Definition mismatch (secondary: Dimensional mismatch).
- Target inference being tested: whether naming all choices as functions of \((\theta,Y)\) identifies the operator with that native algorithm and supplies its physical energy unit. This is not asserted by the conditional finite theorem, but the roadmap's unconditional physical-energy sentence needs correction.
- Why this is an error in the framework:
  1. The complete algorithm register fixes an executed kernel. It does not fix an arbitrary finite \(E_\theta(Y)\), orthonormal basis, or \(K_\theta(Y)\). Naming these choices readout data makes the finite theorem parameterized, but does not derive them.
  2. The native Fock mode space consists of centered random functions of the complete swarm under an actual stationary law. The walker/edge indices of a realized finite score matrix are not, without a proved map, orthonormal modes of that \(L^2_0\) space. A single realized history does not establish their \(L^2\) Gram matrix under that law.
  3. Orthonormalizing dependent recorded functions may change both their basis and regional support. The chapter already notes that orthogonalization can mix regions. It is not legitimate to use a raw edge matrix unchanged as the coordinate matrix of an operator in a different orthonormal basis without proving the transformation law.
  4. The cloning score \((V_j-V_i)/(V_i+\varepsilon)\) is dimensionless. The diagnostic helper declares its matrix input to have units of inverse algorithmic time. The finite theorem says physical energies are \(\hbar_{\rm eff}|\lambda|\) **if** \(h\) has inverse-physical-time units. No native score-to-rate and algorithmic-to-physical-clock derivation is supplied by that “if.” Stage 8's unconditional physical-energy sentence drops it.
- Impact on downstream results: the advertised gap is an exact function of a selected finite readout, not yet an exact physical energy of the existing gas. Free mode/basis/clock choices may alter the spectrum and field interpretation while leaving the simulated algorithm untouched.
- Fix guidance (step-by-step):
  1. State which input choices are original algorithm parameters and which are merely spectral readout choices.
  2. Keep physical units conditional until an existing native clock and score-to-generator conversion has been proved.
  3. Require an actual mode map, inner-product identification, and basis-covariant coefficient formula before using the matrix as the restriction of the native operator.
- Required new assumptions/permits: none. A user-chosen mode basis, a new rate conversion, or a physical-time calibration may define a diagnostic, but cannot replace the requested derivation.
- Framework-first proof sketch for the fix: start from the complete recorded conditional-expectation operator and its actual \(L^2\) inner product. Compute matrix elements in concretely defined native modes; prove the subspace is preserved if a closed finite restriction is claimed. Keep all dimensional conversion factors in those matrix elements.
- Validation plan: inspect an end-to-end construction from a covered record and the configured law. Verify that changing only a coordinate basis conjugates the operator rather than changes the recipe, and that time-unit conversion scales all eigenvalues consistently.

### [E-003] The explicit score kernel has forced zero modes

- Location: `thm-cloning-antisymmetry-lqft` and `def-fermionic-kernel-lqft`, lines 199–258; generic uniqueness/gap discussion in `thm-lqft-edge-filled-ground-gap`, lines 930–950.
- Severity: Major.
- Type: Scope restriction.
- Claim (paraphrase): the general finite matrix ground/gap result can serve as a unique-ground or uniformly positive-gap closure for the score operator already defined by cloning.
- Why this is an error in the framework: this score matrix has much more structure than a general real matrix. For the chapter's unmasked all-pairs score, set
  \[
  u_i=V_i+\varepsilon>0,\qquad a_i=u_i^{-1},\qquad
  K_{ij}=u_j/u_i-1.
  \]
  Then exactly
  \[
  K-K^{\mathsf T}=a u^{\mathsf T}-u a^{\mathsf T}.
  \]
  Its range is contained in \(\operatorname{span}\{a,u\}\), so its rank is at most two. If the positive \(u_i\) are nonconstant, \(a\) and \(u\) are linearly independent and its rank is two. The two nonzero eigenvalues of \(i(K-K^{\mathsf T})\) are
  \[
  \pm\sigma,\qquad
  \sigma^2=\left(\sum_i u_i^2\right)
             \left(\sum_i u_i^{-2}\right)-m^2.
  \]
  Indeed \(a\cdot u=m\) and the skew matrix has
  \(\operatorname{Tr}[-(K-K^{\mathsf T})^2]
       =2(\|a\|^2\|u\|^2-m^2)\).
  For \(m\ge3\), there are exactly \(m-2\) zero modes whenever this matrix is nonzero. Its Fock ground has dimension \(2^{m-2}\), with dimension \(2^{m-3}\) in either parity. For equal positive fitnesses the entire matrix is zero, every Fock vector is a ground vector, and there is no excitation. Near equal fitness, \(\sigma\to0\).
- Impact on downstream results: uniqueness is impossible for this literal nonconstant unmasked score operator with \(m\ge3\); positivity uniform over all admissible fitness records is also impossible. Generic invertible-matrix examples and tests do not discharge these facts. A masked or weighted graph recipe can have different rank, but it is not this explicit all-pairs formula and must be identified separately.
- Fix guidance (step-by-step):
  1. Record this exact specialization next to any attempted application of the generic theorem to the cloning score.
  2. Do not count generic invertibility, a unique vacuum, or a uniform random-record gap as proved.
  3. If the intended coefficient recipe includes original edge masks or weights, identify those exact recorded factors; do not silently replace the score kernel to remove the zero modes.
- Required new assumptions/permits: none. Excluding equal/near-equal fitness, dropping zero modes, or adding coefficients would change the admitted target or introduce a new premise.
- Framework-first proof sketch for the fix: the displayed rank-two factorization, trace identity, and occupation theorem already give the complete specialization. They do not use a new continuum theorem.
- Validation plan: symbolic substitution of the score formula and finite numerical checks. A read-only diagnostic on this review date checked \(u=(1,2,3,4,5)\): rank 2, eigenvalues approximately \((-7.449739,0,0,0,7.449739)\), ground dimension 8. For \(u=(1,1,1,1,1)\), rank 0 and ground dimension 32. For \(u=(1,1+10^{-5},1-10^{-5})\), the nonzero magnitude is approximately \(4.8989795\times10^{-5}\). These checks support the explicit algebra; floating-point zero thresholds are not the proof.

### [E-004] The filled ground is a new state, not the established recorded vacuum

- Location: `def-lqft-record-fock-space` and the CAR covariance correspondence, lines 303–389; ground/shift statements at lines 944–956; parity statements at lines 1002–1055; Chapter 05 `cor-ym-edge-filled-gauge-closure`, lines 7173–7248.
- Severity: Major.
- Type: Scope restriction.
- Target inference being tested: whether shifting the edge energy and filling its negative modes completes the native vacuum/cyclicity requirement. The finite theorem itself correctly distinguishes the states and parity sectors; the target identification remains open.
- Why this is an error in the framework:
  1. The established correspondence uses the empty exterior vacuum \(\Omega\). For example, the native two-point identity is \(\omega_\Omega(a(f)\mathcal Q_t(a^\dagger(g)))=\langle f,C_tg\rangle\).
  2. For a nonzero real skew edge matrix, the finite theorem correctly proves a negative energy and a filled negative-mode ground. That vector is not \(\Omega\).
  3. Let \(n_j=a_j^\dagger a_j\) for a negative eigenmode. Then \(\langle\Omega,n_j\Omega\rangle=0\), while every filled ground has \(\langle\xi_-,n_j\xi_-\rangle=1\). The states are observably different.
  4. Subtracting \(E_{\rm sea}I\) preserves unitary conjugation on observables, but does not change \(\Omega\) into \(\xi_-\) and does not preserve the original vacuum expectation by that fact alone.
  5. If the number of negative modes is odd and there are no zero modes, the filled ground lies in the odd parity sector. The existing empty-vacuum even sector then does not contain it. Choosing the filled-ground parity changes the vacuum sector rather than proving the original sector's vacuum property.
- Impact on downstream results: the exact full-even-algebra cyclicity of a filled vector is not a proof of physical vacuum cyclicity for the already defined regional net and recorded state. Correlation identities, parity sector, gauge character, and physical positivity must all refer to one identified state.
- Fix guidance (step-by-step):
  1. Keep both states named and distinct wherever these theorems are invoked.
  2. Do not claim that an energy counterterm alone discharges the native vacuum or recorded-correlation requirement.
  3. Leave the same-state correlation and physical-sector identification open; do not replace the original state or even sector to force closure.
- Required new assumptions/permits: none. A new ground-state readout is legitimate as a diagnostic but not as a proof that its correlations are those of the native process.
- Framework-first proof sketch for the fix: evaluate the bounded occupation observable \(n_j\) in the two states. This single exact comparison disproves an unqualified state identification. Use the parity theorem to check whether the ground even belongs to the original sector.
- Validation plan: compare the state on a spanning set of finite CAR words and verify the regional generated algebra separately. Equality only of observable conjugation dynamics does not pass the state check.

### [N-001] Finite spectral and parity calculations are correct with finite scope

- Location: lines 895–1108.
- Severity: Note.
- Type: Scope restriction.
- Verified conclusions: occupation subset spectrum; filling all negative and no positive modes; arbitrary zero-mode occupations; ground multiplicity \(2^z\); exact gap \(\min_{\lambda\ne0}|\lambda|\) above the complete ground; parity compensation by zero modes; two-smallest-cost gap in the filled-ground parity when \(z=0\); opposite-parity gap as the smaller of a larger single cost minus \(d_1\) and \(d_2+d_3\); explicit no-excitation convention; paired real-edge spectrum; and full even CAR algebra \(\mathcal B(\mathcal F_0)\oplus\mathcal B(\mathcal F_1)\).
- Qualification: cyclicity is proved for the full finite parity block. It is not automatically cyclicity for a smaller prescribed local algebra. Numerical diagnostics correctly expose a zero tolerance, which must not be mistaken for an exact rank certificate.

## Scope restrictions and clarifications

### Linear coverage register

| Source region | Checked content | Review outcome |
|---|---|---|
| 1–184 | Scope, graph/geometry comparison, direct field narrative | Separate continuum comparison remains explicit. |
| 185–792 | Weighted score identities, exterior symbols, actual record Fock construction, replicas, faithful insertions | Algebraic constructions keep independent replicas and native probability kernels distinct. |
| 793–1108 | Directed-edge lift and all new spectral/parity results | Finite algebra passes; native operator/state identification does not follow. |
| 1109–1479 | Instantiated transition words, native CAR channel, LSI transfer | Conditional-expectation evolution is retained; strict contraction obstructs unitary identification. LSI is law-specific. |
| 1480–2032 | Regional modes, covariance, survival/history martingales, leakage, product obstruction, Clifford identities | Exact locality tests are tests, not geometry-only proofs; the scalar/CAR product obstruction is explicit. |
| 2033–2548 | Sampling bias/variance, energy quadrature, weak dependence, density-corrected graph operator, summary | Smooth geometry and joint sampling hypotheses remain explicit. Density correction is a postprocessed operator, not the automatically executed transition. No graph consistency statement closes physical time or a quantum gap. |

### Hypothesis and construction provenance

| Input | Proven or defined provenance | Status for the unchanged native physical chain |
|---|---|---|
| Complete record codec | Chapter 01 coverage and lossless results | Transports covered objects, not unrecorded matrices or a new physical generator. |
| Actual \(P_h\), and \(L\) where justified | Implemented full update, Chapter 04 | Native. A continuous-time family must retain its own justified scaling. |
| Conservative stationary \(\pi\) used by the CAR theorem | Explicit theorem input | Not interchangeable with a killed-chain QSD or a terminal-survival law. |
| Exterior/CAR construction | Defined on actual \(L^2_0(\pi)\), replica theorem | Exact mathematical construction; not sampled-walker exchange statistics. |
| Finite \(E_\theta(Y)\), basis, \(K_\theta(Y)\) | New declared readout data | Not derived as a closed restriction of the native generator. |
| Filled ground and ground shift | Exact finite spectral theorem | Different from the native empty-vacuum state unless separately identified. |
| Clock/energy conversion | Conditional spectral unit statement and calibrated readout parameters | No automatic conversion of dimensionless cloning scores to physical inverse time. |
| Regional orthogonality | Conditional covariance/Gram criterion | Not obtained solely from spatial separation or a re-orthogonalized basis. |
| Uniform positive spectral gap and vacuum uniqueness | Not asserted by the finite theorem | Not discharged; literal unmasked scores have the additional zero-mode obstruction. |

## Proposed edits (optional)

No edits were made to the reviewed source. Under the user's fixed-target/no-new-assumptions instruction, the minimal editorial correction is to narrow the completion ledger: “finite spectrum of a specified edge readout matrix” is proved; “native physical generator, vacuum, gauge-sector gap” is not. Preserve the proved obstruction statements and do not repair the target by a new operator, cutoff, projection, clock, or state.

## Open questions

1. Which already defined native modes and complete-kernel matrix elements, if any, yield the intended Hamiltonian? A new free declaration of \(K_\theta(Y)\) does not answer this.
2. What is the actual same-record physical-time/transfer identification compatible with the proved nonmultiplicative CAR channel and pullback-spectrum obstruction?
3. Is the intended edge kernel the displayed all-pairs cloning score, the actually sampled masked graph, or another already defined record functional? Its exact formula and unit register are needed before interpreting its gap.
4. How can physical vacuum correlations be obtained without silently replacing the native recorded state by a filled-sea state or changing its parity sector?
