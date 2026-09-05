# Mathematical Review: docs/source/1_agent/03_architecture/00_architecture_at_a_glance.md

## Metadata
- Reviewed file: docs/source/1_agent/03_architecture/00_architecture_at_a_glance.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (79 lines: TLDR, pipeline overview diagram, module map, pointers)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/01_foundations/01_definitions.md:8` (three-channel internal state $Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})$)
  - `docs/source/1_agent/03_architecture/02_disentangled_vae.md:44-64` (typed latents, $K_t=(K_{\mathrm{chart}},K_{\mathrm{code}})$, $z_{\mathrm{geo}}=c_{\mathrm{bar}}+z_{q,\mathrm{st}}+z_n$), `:125-142` (encoder diagram), `:250-261` (`def-total-disentangled-loss`)
  - `docs/source/1_agent/10_appendices/07_architecture.md:82-83, 275-300, 304, 408-422` (Appendix G encoder outputs, VQ block, decoder router, jump/supervised/classifier wiring)
  - `docs/source/1_agent/03_architecture/01_compute_tiers.md:1508-1513, 1781, 2649-2652, 2774-2782` (reference implementation: `Z_t` packing, decoder routing, `compute_jump_consistency_loss` signature and call)

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 3
- Notes: 1
- Primary themes: The overview is a faithful condensation of the TopoEncoder chapter and Appendix G. The encoder pipeline, router, per-chart codebooks, decoder projectors and renderer, texture-residual path, and optional attachments all agree with the detailed sources, and every cross-reference resolves. The discrepancies are all in how the summary abbreviates its sources: the macro state is reduced to the chart id and `K_code` disappears from text and diagram; the derived decoder input $z_{\mathrm{geo}}$ is presented as a fourth typed latent with its $z_n$ and $c_{\mathrm{bar}}$ dependencies hidden; the jump operator is wired to the blended nuisance vector instead of the per-chart tensor and router weights; and the training summary omits the tiered regularizers. None affects a result; all are one-line fixes to the TLDR or the pipeline diagram.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | TLDR bullet 1 (6-7); Pipeline Overview (27-36) | Minor | Definition mismatch | Framework | this chapter | $K$ called "chart id"; `K_code` absent from text and diagram although $K=(K_{\mathrm{chart}},K_{\mathrm{code}})$ upstream |
| E-002 | TLDR bullet 1 (6-7); Pipeline Overview (34-36) | Minor | Definition mismatch | Framework | this chapter | $z_{\mathrm{geo}}$ listed as a fourth typed latent and drawn as a raw VQ output, hiding $z_{\mathrm{geo}}=c_{\mathrm{bar}}+z_{q,\mathrm{st}}+z_n$ |
| E-003 | Pipeline Overview (48) | Minor | Algorithm mismatch | Framework | this chapter | Jump operator fed the blended `z_n [B,D]` instead of `z_n_all_charts [B,N_c,D]` plus router weights |
| E-004 | TLDR bullet 5 (13-14) | Note | Miswording | Framework | this chapter | Training summary omits the non-optional tiered regularizer sum |

## Detailed findings

### [E-001] Macro state reduced to the chart id; `K_code` missing (was F-001)
- Location: TLDR bullet 1, lines 6-7; Pipeline Overview diagram, lines 27-36
- Severity: Minor
- Type: Definition mismatch (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 6-7): "The representation stack is the TopoEncoder (Attentive Atlas) with typed latents: chart id $K$, nuisance $z_n$, texture $z_{\mathrm{tex}}$, and geometry $z_{\mathrm{geo}}$." In the diagram the router emits `Kchart["K_chart"]` (line 28) and the "per-chart VQ" node emits only `z_geo`, `z_tex`, `z_n` (lines 34-36).
- Upstream anchor: `02_disentangled_vae.md:52-53`: "$K_t = (K_{\mathrm{chart}}, K_{\mathrm{code}})$ is the discrete macro state. $K_{\mathrm{chart}}$ selects an atlas chart, and $K_{\mathrm{code}}$ selects a local code within that chart." `07_architecture.md:82-83`: "Output (ordered): `K_chart`, `K_code`, `z_n`, `z_tex`, `router_weights`, `z_geo`, ...". `07_architecture.md:281-282`: `Indices -- "indices [B, N_c]" --> Kcode["K_code (from K_chart)"]` / `Kchart -- "K_chart [B]" --> Kcode`. `01_compute_tiers.md:1511`: `Pack["Z_t = (K_chart, K_code, z_n, z_tex)"]`.
- Why this is an error: In the framework the discrete macro symbol $K$ is the pair (chart, code); the code index is produced by the per-chart VQ and is half of the macro state that `def-causal-enclosure` (`02_disentangled_vae.md:68`) is stated about. The overview calls $K$ the chart id and its diagram produces no discrete output from the VQ node, so a reader would conclude the quantizer contributes no macro information, contradicting the definition being summarized.
- Impact on downstream results: None on results; the detailed chapters are self-consistent. Misleads readers about the object the sieve and closure diagnostics act on.
- Fix guidance:
  1. Line 6: replace "chart id $K$" with "macro state $K=(K_{\mathrm{chart}},K_{\mathrm{code}})$".
  2. In the pipeline diagram add a node `Kcode["K_code"]` with edges `VQ --> Kcode` and `Kchart --> Kcode`, mirroring `07_architecture.md:281-282`.
- Required new assumptions/permits: none.
- Validation plan: Confirm the encoder output list in `07_architecture.md:82-83` and the `Z_t` packing in `01_compute_tiers.md:1511` both appear in the overview's diagram; render the mermaid block.

### [E-002] $z_{\mathrm{geo}}$ presented as an independent typed latent emitted by the VQ block (was F-002)
- Location: TLDR bullet 1, lines 6-7; Pipeline Overview diagram, lines 34-36
- Severity: Minor
- Type: Definition mismatch (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 6-7 list "typed latents: chart id $K$, nuisance $z_n$, texture $z_{\mathrm{tex}}$, and geometry $z_{\mathrm{geo}}$"; lines 34-36 draw `VQ --> Zgeo["z_geo"]`, `VQ --> Ztex["z_tex"]`, `VQ --> Zn["z_n"]` with no edge from `Cbar` or `Zn` into `Zgeo`.
- Upstream anchor: `01_definitions.md:8`: "$Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})$". `02_disentangled_vae.md:58-64`: "The geometry latent used by the decoder is $z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n$ where $c_{\mathrm{bar}}$ is the chart center mixture and $z_{q,\mathrm{st}}$ is the straight-through quantized code." `02_disentangled_vae.md:140-142`: `ZqSt --> Zgeo["z_geo = c_bar + z_q_st + z_n"]` / `Zn --> Zgeo` / `Cbar --> Zgeo`.
- Why this is an error: The framework's internal-state split has exactly three channels; $z_{\mathrm{geo}}$ is not a fourth channel but a deterministic function of the others that already contains $z_n$ and $c_{\mathrm{bar}}$. Listing it as a peer typed latent misstates the definition, and drawing it as a direct VQ output with `Zn` a sibling node (rather than an input) hides that $z_n$ reaches the decoder through $z_{\mathrm{geo}}$, which is precisely why the decoder signature takes only `(z_geo, z_tex)` (`07_architecture.md:132`). The coarse "VQ" node could legitimately absorb the $z_{q,\mathrm{st}}$ detail, but both `Zn` and `Cbar` exist as separate nodes in the overview, so the missing edges are omissions rather than abstraction.
- Impact on downstream results: None on results; can confuse readers about why $z_n$ is not a decoder input.
- Fix guidance:
  1. Line 6-7: write "typed latents $K$, $z_n$, $z_{\mathrm{tex}}$, and the derived decoder input $z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n$" (or reuse the phrasing of `02_disentangled_vae.md:8-10`).
  2. In the diagram add `Cbar --> Zgeo` and `Zn --> Zgeo`, as in `02_disentangled_vae.md:141-142`.
- Required new assumptions/permits: none.
- Validation plan: Check the overview diagram's in-edges of `Zgeo` match the set {`VQ`/`ZqSt`, `Zn`, `Cbar`} in `02_disentangled_vae.md:140-142` and `07_architecture.md:295-297`.

### [E-003] Jump operator wired to the blended $z_n$ instead of per-chart `z_n_all_charts` and router weights (was F-003)
- Location: Pipeline Overview diagram, line 48
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 48): `Zn --> Jump["Jump operator (optional)"]`, where `Zn` is the node "z_n" emitted by the VQ block, i.e. the router-weighted blend `z_n = sum(w_enc * z_n_all_charts) [B, D]` (`02_disentangled_vae.md:135`).
- Upstream anchor: `07_architecture.md:299`: `ZnAll -- "z_n_all_charts [B, N_c, D]" --> Jump["FactorizedJumpOperator (optional)"]`. `07_architecture.md:413-414`: `Enc -- "z_n_all_charts [B, N_c, D]" --> Jump` / `Enc -- "enc_w [B, N_c]" --> Jump`. `01_compute_tiers.md:2649-2652`: `def compute_jump_consistency_loss(z_n_by_chart, router_weights, jump_operator, ...)`; `:2779-2782`: `compute_jump_consistency_loss(z_n_all_charts, router_weights, self.jump_op)`.
- Why this is an error: The jump operator learns chart-to-chart transition maps and its consistency loss compares per-chart nuisance coordinates for points in chart overlaps, weighted by the router. It therefore requires the `[B, N_c, D]` tensor and the `[B, N_c]` weights. The blended `z_n` is a single `[B, D]` vector in which the per-chart information has already been summed away, so it cannot serve as the operator's input. The overview draws the wrong tensor and omits the router-weight input.
- Impact on downstream results: Local; Appendix G and the compute-tiers implementation agree with each other.
- Fix guidance:
  1. Add a node `ZnAll["z_n_all_charts"]` inside the encoder subgraph with `VQ --> ZnAll` and `ZnAll --> Zn`.
  2. Replace line 48 with `ZnAll --> Jump["Jump operator (optional)"]` and add `Wenc --> Jump`.
- Required new assumptions/permits: none.
- Validation plan: Compare the in-edges of `Jump` in the overview against `07_architecture.md:413-414` and the call at `01_compute_tiers.md:2779-2782`.

### [E-004] Training summary omits the tiered regularizers (was F-004)
- Location: TLDR bullet 5, lines 13-14
- Severity: Note
- Type: Miswording
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 13-14): "Training combines reconstruction + VQ + routing/consistency terms, with optional jump and supervised topology losses."
- Upstream anchor: `02_disentangled_vae.md:254-261` (`def-total-disentangled-loss`): $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \lambda_{\text{ent}}\mathcal{L}_{\text{entropy}} + \lambda_{\text{cons}}\mathcal{L}_{\text{consistency}} + \sum_{i \in \text{tiers}} \lambda_i \mathcal{L}_i + \lambda_{\text{jump}}\mathcal{L}_{\text{jump}} + \lambda_{\text{sup}}\mathcal{L}_{\text{sup}}$; `02_disentangled_vae.md:14-15`: "Training uses reconstruction + VQ + routing/consistency, plus tiered regularizers and optional jump and supervised topology losses."
- Why this is an error: The sentence reads as an exhaustive list in which only jump and supervised losses are optional, but the total loss contains a non-optional tiered regularizer sum. Not a mathematical error; the parallel TLDR in the detailed chapter includes the missing phrase.
- Impact on downstream results: None.
- Fix guidance:
  1. Insert "plus tiered regularizers" after "routing/consistency terms", matching `02_disentangled_vae.md:14`.
- Required new assumptions/permits: none.
- Validation plan: Compare the bullet with the term list of `def-total-disentangled-loss`.

## Scope restrictions and clarifications
- This chapter is a summary with no derivations, theorems, or numeric claims; all findings concern fidelity of the summary to `02_disentangled_vae.md` and Appendix G.
- The remaining diagram claims were checked and are consistent with Appendix G: decoder router "CovariantChartRouter or latent_router" (`07_architecture.md:304`, `01_compute_tiers.md:1781`); supervised topology loss fed by `w_enc` and `z_geo` (`07_architecture.md:408-409`); invariant classifier fed by `w_enc` and `z_geo` (`07_architecture.md:421-422`); module map attachments (`07_architecture.md:196`).
- All three `{ref}` targets resolve (`sec-topoencoder-architecture`, `sec-covariant-cross-attention-architecture`, `sec-computational-considerations`); the mechanical cross-reference pass reports no dangling references or duplicate labels for this file.

## Proposed edits (optional)
- Apply the four TLDR/diagram edits above in one pass; they touch only lines 6-7, 13-14, 34-36, and 48 of the overview.

## Open questions
- Whether the overview diagram should show `K_code` and `z_n_all_charts` at all is a presentation choice; if the author prefers to keep the diagram minimal, the TLDR text alone should at least state the pair form of $K$ and the derived nature of $z_{\mathrm{geo}}$.

## Rejected candidate findings
- None. All four stage-1 findings were confirmed at their original severity.
