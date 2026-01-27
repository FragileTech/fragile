# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/01_emergent_geometry.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/01_emergent_geometry.md
- Review date: 2026-01-27
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-adaptive-diffusion-tensor-latent: $\Sigma_{\mathrm{reg}}(z,S) = (H(z,S)+\epsilon_\Sigma I)^{-1/2}$ and $D_{\mathrm{reg}} = (H+\epsilon_\Sigma I)^{-1}$.
  - assump-spectral-floor-latent: spectral bounds $-\Lambda_- \leq \lambda_{\min}(H) \leq \lambda_{\max}(H) \leq \Lambda_+$ with $\epsilon_\Sigma > \Lambda_-$.
  - def-riemannian-volume-element-latent: $dV_g(z) = \sqrt{\det g(z,S)}\,dz$ with $g=H+\epsilon_\Sigma I$.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 0
- Notes: 5
- Primary themes: equivalence-principle corrected to generator-level reinterpretation; volume/QSD regime explicit; geometric drift and Ito/Stratonovich issues resolved; mean-field fitness field aligned with code.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Sec. "Geometric Drift Term", Lemma "Geometric Drift for Riemannian Measure" (equation for $b_{\mathrm{geo}}$) | Note | Resolved | $b_{\mathrm{geo}}$ updated to covariant divergence with the missing term restored. |
| E-002 | Sec. "Riemannian Volume Elements and Integration" (definition + QSD discussion) | Note | Resolved | Regime is now explicit: mean-field $g(z;\mu)$ in analysis, frozen-$S$ within a step for finite $N$, with proof routes linked. |
| E-003 | Sec. "Equivalence Principle: Flat vs Curved" (observation + equivalence theorem) | Note | Resolved | Reframed as generator-level equivalence; removed global diffeomorphism/TV statement and added explicit remark. |
| E-004 | Sec. "Adaptive Diffusion Tensor" (mean-field Hessian formula) | Note | Resolved | Mean-field field definition and per-walker Hessian wording updated; no longer inconsistent with code. |
| E-005 | Sec. "Geometric Drift Term" (lemma SDE notation) | Note | Resolved | Lemma restored Stratonovich notation to match the section. |

## Detailed findings

### [E-001] Geometric drift formula misses required divergence terms (RESOLVED)
- Location: Sec. "Geometric Drift Term", Lemma "Geometric Drift for Riemannian Measure" (equation for $b_{\mathrm{geo}}$)
- Severity: Note
- Type: Resolved
- Resolution: The lemma now uses the covariant divergence formula
  $b_{\mathrm{geo}}^k=\frac{T}{2}\frac{1}{\sqrt{\det g}}\partial_{z_l}(\sqrt{\det g}\,g^{kl})$ and explicitly includes the
  additional $D_{\mathrm{reg}}\nabla\log\sqrt{\det g}$ term, consistent with the Riemannian invariant measure.
- Validation plan: Confirm the updated bound and invariant-density sentence match the corrected drift.

### [E-002] Riemannian volume and QSD weighting ignore $S$-dependence of the metric (RESOLVED)
- Location: Sec. "Riemannian Volume Elements and Integration" (definition and QSD-weighted discussion)
- Severity: Note
- Type: Resolved
- Resolution: Added {prf:ref}`rem-volume-element-regime`, which fixes the regime as mean-field $g(z;\mu)$ for analytic statements and frozen-$S$ within a step for finite $N$. Included a dropdown with two proof routes: propagation of chaos ({prf:ref}`thm-propagation-chaos-qsd`) and the frozen-potential step in {ref}`sec-eg-stage2`.
- Validation plan: Ensure links render and the mean-field/frozen wording matches the QSD sampling discussion.

### [E-003] Equivalence principle lacks a justified coordinate transformation (RESOLVED)
- Location: Sec. "Equivalence Principle: Flat vs Curved" (observation and "Equivalence Theorem")
- Severity: Note
- Type: Resolved
- Resolution: Replaced the diffeomorphism/TV claim with a generator-level equivalence statement and added
  {prf:ref}`rem-equivalence-reinterpretation` clarifying that no global coordinate transform is assumed; any coordinate
  change is local and requires $g=\Psi^*G$.
- Validation plan: Check that the new remark renders and the geometric drift lemma is referenced for generator equality.

### [E-004] Mean-field Hessian formula omits second-argument contributions (RESOLVED)
- Location: Sec. "Adaptive Diffusion Tensor" (mean-field formula for $H(z,S)$)
- Severity: Note
- Type: Resolved
- Claim (paraphrase): For $V_{\mathrm{fit}}(S)=\frac{1}{N}\sum_{i,j}\phi(z_i,z_j)$, a walker at $z$ experiences $H(z,S)=\frac{1}{N}\sum_j \nabla_z^2\phi(z,z_j)$.
- Resolution: The document now defines the per-walker Hessian $H(z,S)=\nabla_z^2 V_{\mathrm{fit}}^{(i)}(z;S)$ and introduces the mean-field fitness field $V_{\mathrm{fit}}(z;\mu)$ (Definition {prf:ref}`def-mean-field-fitness-field`), aligning with the code and removing the ambiguous global-energy statement.
- Validation plan: Confirm the per-walker wording remains consistent with `src/fragile/fractalai/core/fitness.py` and the adaptive diffusion tensor in the kinetic operator.

### [E-005] Ito vs Stratonovich notation is inconsistent in the drift lemma (RESOLVED)
- Location: Sec. "Geometric Drift Term", Lemma "Geometric Drift for Riemannian Measure"
- Severity: Note
- Type: Resolved
- Resolution: The lemma now uses Stratonovich notation ($\circ dW$), consistent with the section.
- Validation plan: Ensure subsequent references to the drift term are consistent with Stratonovich interpretation.

## Scope restrictions and clarifications
- Convergence claims (TLDR and summary) should explicitly point to the confinement/killing/minorization hypotheses in the convergence appendices, since uniform ellipticity and Lipschitz continuity alone do not guarantee QSD convergence.
- The "Equivalence Principle" is safest when stated as a generator-level reinterpretation rather than a global coordinate transformation.

## Proposed edits (optional)
- Replace the $b_{\mathrm{geo}}$ formula with a covariant divergence expression and update the equilibrium-density sentence to reference the corrected generator.
- Add a short paragraph in the volume/QSD subsection clarifying whether $g$ is frozen (conditional on $S$), mean-field via $g(z;\mu)$, or defined on configuration space.

## Open questions
- What is the definitive regime for the volume/QSD claims in this section: frozen geometry at fixed $S$, mean-field geometry $g(z;\mu)$, or configuration-space measure on $\mathcal{Z}^N$?
- Is the intended geometry on $\mathcal{Z}$ with $S$ frozen/mean-field, or on full configuration space $\mathcal{Z}^N$?
