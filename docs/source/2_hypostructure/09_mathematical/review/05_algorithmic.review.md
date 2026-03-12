# Mathematical Review: docs/source/2_hypostructure/09_mathematical/05_algorithmic.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/09_mathematical/05_algorithmic.md
- Review date: 2026-01-27 (initial), updated through 2026-03-12
- Reviewer: Codex (initial), Claude Opus 4.6 (Rounds 3-8)
- Scope: Entire document — P ≠ NP proof via five-modality obstruction

## Review History

| Round | Date | Reviewer | FATAL | SERIOUS | Status |
|-------|------|----------|-------|---------|--------|
| 1-2 | 2026-01-27 | Codex | 0 | 0 | Initial pass, no issues found |
| 3 | 2026-03-10 | Claude Opus 4.6 | 4 | 14 | All FATAL closed |
| 4 | 2026-03-10 | Claude Opus 4.6 | 0 | 5 | Tightening pass |
| 5 | 2026-03-11 | Claude Opus 4.6 (5 agents) | 4 | 9 | All FATAL/SERIOUS closed |
| 6 | 2026-03-11 | Claude Opus 4.6 (6 agents) | 9 | 18 | All 9 FATAL closed; 7 SERIOUS + 12 NT remaining |
| 7 | 2026-03-12 | Claude Opus 4.6 (5 agents) | 7 | 12 | External soundness review; F7.1 + F7.2 resolved |
| **8** | **2026-03-12** | **Claude Opus 4.6 (5 agents)** | **4** | **16** | **Complete from-scratch re-review** |

## Current Status (Post-Round 8)

Round 8 is a complete from-scratch hostile review against the current document state.
Two prior FATAL findings (F7.1: Foundation Assumption, F7.2: Schreiber theorem target)
are confirmed RESOLVED. Round 8 identifies **4 FATAL**, **16 SERIOUS**, and **18 NEEDS
TIGHTENING** issues.

### Key findings

1. **The coend formula in `thm-schreiber-structure` does not follow from the proof.**
   The proof establishes fracture-square pullbacks; the theorem claims a coend decomposition
   of hom-spaces. These are categorically dual.

2. **The ∗ and ∂ witness definitions lack modal restrictions that their blockage proofs
   require.** The ∗-witness merge map has no modal restriction; the ∂-witness permits
   "arbitrary polynomial-time computation on interface data." (Carried from R7.)

3. **Workspace separation is a mathematical artifact, not a property of algorithms.**
   The non-amplification principle is proved for a fictitious product state space.
   (Carried from R7.)

### What IS sound

- The ♯-channel pigeonhole argument (given constant D in the definition)
- The ∫-channel backbone-triple transfer lemma (rigorous combinatorial proof)
- The ♭-channel type-independent cardinality argument (for the search formulation)
- The bridge P_FM = P_DTM
- The overall conditional structure in `thm-conditional-nature`
- The primitive audit table at the instruction level
- The internal consistency fixes from Rounds 3-6

### Full details

See the individual round review files:
- `05_algorithmic_round5.review.md` — Rounds 3-5 (31 fixes applied)
- `05_algorithmic_round6.review.md` — Round 6 (9 FATAL closed, structural architecture)
- `05_algorithmic_round7.review.md` — Round 7 (external soundness; F7.1/F7.2 resolved)
- `05_algorithmic_round8.review.md` — **Round 8 (current, complete from-scratch review)**
- `06_complexity_bridge.review.md` — Complexity bridge review

## Error log (summary across all rounds)

| Round | FATAL found | FATAL closed | SERIOUS found | SERIOUS closed | Remaining |
|-------|-------------|-------------|---------------|----------------|-----------|
| 3 | 4 | 4 | 14 | 14 | 0 |
| 4 | 0 | 0 | 5 | 5 | 0 |
| 5 | 4 | 4 | 9 | 9 | 3 NT |
| 6 | 9 | 9 | 18 | 11 | 7 S + 12 NT |
| 7 | 7 | 2 (F7.1, F7.2) | 12 | 0 | 5 F + 12 S + 14 NT |
| 8 | 4 | 0 | 16 | 0 | 4 F + 16 S + 18 NT |
| **Current** | — | — | — | — | **4 FATAL + 16 SERIOUS + 18 NT** |

## Open questions

1. Can the coend formula in `thm-schreiber-structure` be proved, or should it be replaced
   with the pullback decomposition that IS proved?
2. Can the ∗ and ∂ witness definitions be tightened with explicit modal restrictions without
   making them too narrow to capture known algorithms?
3. Can workspace separation be proved for single-tape execution?
4. Can the purity-violation argument pattern be replaced with extensional arguments
   (like the ♯-pigeonhole or ∫-backbone)?
5. Can the universality theorems' (1)⇒(2) directions be proved?
