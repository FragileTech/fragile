# Chapter 15 analytic recovery

Edited only `docs/source/2_fractal_gas/convergence_program/15_kl_convergence.md`.
Old chapter: 8769 lines, 41227 words. Consolidated chapter: 2178 lines, 8943 words.

## Recovered proofs and provenance

| Source route | Consolidated targets | Analytic correction / scope |
|---|---|---|
| Old 15 Gibbs/Bakry–Émery, tensorization and perturbation routes; 10 kinetic reference; 4_ymmg `prop:gibbs-lsi` | `thm-bakry-emery`, `thm-tensorization`, `thm-lsi-perturbation`, `thm-kinetic-lsi`, `cor-n-particle-kinetic-lsi` | Full-gradient LSI. Bounded density tilt, not arbitrary bounded drift. Full tensorization proof keeps N-independent constant. |
| Old 15 nonconvex confinement/HWI route; Cattiaux–Guillin–Wu Section 3.3 | `thm-hwi-inequality`, `lem-kl-lyapunov-weighted-energy`, `thm-unconditional-lsi`, `thm-nonconvex-main` | Genuine transport geodesic. Explicit radial-confinement Lyapunov function, weighted moments, local Poincaré, defective entropy, Rothaus centering. No added compactness assumption. |
| Old 15 uniform curvature/reference routes | `cor-n-uniform-lsi`, `prop-kl-joint-interaction-curvature` | Four proved structural criteria: product; bounded joint density tilt; uniform full Hessian; contractive additive-noise invariant flow. Mean-field interaction curvature counted by quadratic form, without hidden N factor. |
| 10 and old 15 hypocoercive Fisher identities; Villani analytic method; 4_ymmg backbone | `lem-kinetic-evolution-bounds`, `lem-hypocoercive-dissipation`, `thm-villani-hypocoercivity`, `cor-n-particle-hypocoercive` | Correct mixed derivative has velocity-Hessian-velocity term. Keep mixed second derivatives in positive matrix form. Explicit positive metric a=c=2η,b=η fixes old ac<b². Full proof and dimension-independent rate. Initial prefactor uses H+modified Fisher. |
| Old 15 full-generator/cloning routes | `def-kl-full-generator`, `thm-cloning-entropy-contraction`, `lem-cloning-gamma2-bound`, `thm-main-kl-convergence` | Actual common-target entropy/data-processing proof; precise Fisher amplification bound; complete kinetic-plus-jump closure when that bound holds. |
| QSD distinction required throughout old 15/10/17 | `def-kl-finite-particle-laws`, `prop-kl-conditioned-entropy`, `prop-kl-boundary-entropy`, `lem-kl-functional-first-variation`, `thm-kl-convergence-euclidean` | Full normalized killed evolution derived explicitly. Includes covariance normalization and outgoing entropy flux. QSD is not silently invariant. Exact finite-state numerical checks support algebra. |
| QSD transform method | `prop-kl-doob-transform`, `lem-kl-bounded-reweighting`, `thm-main-kl-final` | Complete Doob conjugacy, invariant transformed law, KL reweighting proof; survival eigenfunction ratios retained. |
| Old 15 final Gamma2/acoustic route | `thm-hypo-curvature-bound`, `cor-acoustic-limit-explicit`, `cor-n-uniform-curvature` | Exact denominator λ−μ². Acoustic min requires both inequalities. Auxiliary full form distinguished from kinetic velocity-only carré du champ. |
| Old 15 heat smoothing and mean-field selection proof routes | `lem-cloning-fisher-info`, `thm-entropy-bound-debruijn`, `lem-meanfield-cloning-dissipation-hybrid`, `lem-softmax-lipschitz-status` | Relative-score reference term retained; fixed-reference heat entropy identity corrected; exact capped pair-energy dissipation proved; fixed candidate softmax TV formula proved. No false universal Var(energy)≥c KL. |
| Old 15 discrete hybrid/composition routes | `thm-lsi-implies-kl-convergence`, `thm-main-lsi-composition`, `thm-entropy-transport-contraction`, `lem-discrete-lsi-from-curvature` | Actual discrete entropy defect. All additive errors retained as floors; no W2-to-KL implication. |
| 4_ymmg `lem:mf-map-contraction`, `thm:mf-lsi` | `thm-kl-contractive-diffusion-lsi`, `cor-kl-frozen-alignment-lsi` | Reuse correct synchronous coupling algebra, repair exact friction threshold. Derive full LSI directly from flow gradients and invariant entropy interpolation; no reliance on paper's invalid repeated Fisher coefficient choice. Applies conservative frozen alignment field, including an identified stationary MF fixed point satisfying stated bounds. |
| 4_ymmg `lem:weighted-sym`, `prop:gaussian`; archive frozen-velocity routes | `prop-kl-frozen-ou-lsi`, `rem-kl-frozen-velocity-law` | Complete degree-weighted similarity proof and uniform Gaussian covariance/LSI bound. Frozen velocity law is not identified with moving-swarm conditional QSD. |
| Old 15 empirical and mean-field consequences | `cor-quantitative-lsi-final`, `cor-kl-lsi-mean-field-limit`, `cor-adaptive-lsi` | Full joint LSI→Poincaré→C L²/N empirical variance. Marginal and weak-limit proofs. Actual-law/full-form geometric comparison. |

## Sources searched

Current 15, full 10, relevant 17 LSI proofs; archival 15 and 11 routes; baseline 15; `docs/source/4_ymmg/01_convergence.tex` backbone, frozen Gaussian and stationary mean-field sections. Archival 15 and baseline 15 are the same mathematical text; current pre-rewrite 15 differed only in headings.

Primary external analytic sources checked: Villani, https://arxiv.org/abs/math/0609050 ; Cattiaux–Guillin–Wu, https://perso.math.univ-toulouse.fr/cattiaux/files/2013/11/cgw-submit-ptrf.pdf . Core proofs are written in the chapter, with their hypotheses and constants.

## Specific remaining full-algorithm identification

These are not gaps in the proved reference, structural, or frozen-field results. To instantiate `thm-kl-convergence-euclidean` for the actual killed, moving, cloning joint law, one must identify that law with a proved joint-LSI criterion (or prove its LSI by another valid route), control the actual cloning contribution to the complete modified-Fisher derivative, and bound the killing/normalization/boundary contribution. The searched purported automatic closures used false steps: QSD=Gibbs; velocity-only LSI for spatial tests; a status Dobrushin metric as a spatial derivative bound; W2 contraction as KL contraction; or deleting a forcing floor. None supplies the missing actual-law estimate. The chapter states the exact conditional implication while preserving all complete analytic subresults.

## Checks

- All 30 local proof/section references and all 7 document references resolve.
- All directive fences balanced; 338 display-math fences paired.
- `git diff --check` passed.
- No removed-volume/framework, appendix proof-home, or review-scaffolding terminology.
- No published incoming references to the removed/consolidated old labels.
- 91 original labels mapped in `kl_label_map.json`; 46 retained, 45 consolidated or removed with reasons.
- 250 randomized corrected kinetic dissipation matrix checks: largest residual eigenvalue −4.680188530424865e−09.
- 250 nonreversible finite killed-chain normalized entropy checks: maximum absolute identity error 5.329070518200751e−15.
