# Claude Review: 00_faq.md

## 1. Broken Cross-References

### Confirmed from Codex (9 issues with `/source` prefix and relative paths)
Lines 25, 26, 116, 267, 288, 333, 746, 766, 857 - all need path corrections.

### Additional finding
- **Line 10 (Dependencies):** Lists `{doc}`../2_fractal_set/02_causal_set_theory`` and `{doc}`../2_fractal_set/03_lattice_qft``, but later uses bare names like `{doc}`02_causal_set_theory`` (line 26), `{doc}`03_lattice_qft`` (lines 25, 267). **Inconsistent reference style.**

## 2. LaTeX Errors

**None found** - LaTeX is well-formed throughout.

## 3. Mathematical Inconsistencies

### CRITICAL: Line 237 (Section N.2.3) - Momentum Conservation Claim
**Gemini correctly identified this.** The text claims:
> "The inelastic collision $v_{\text{new}} = \alpha_{\text{rest}} v_j + (1 - \alpha_{\text{rest}}) v_i$ blends velocities, conserving total momentum $\sum_i m_i v_i$."

This is **mathematically incorrect**. This formula does NOT conserve momentum unless there's a compensating change elsewhere. The change in momentum is:
```
ΔP = m_i(v_new - v_i) = m_i α_rest(v_j - v_i) ≠ 0
```

### Lines 167-169 (Section N.1.4) - Population vs Mass Non-Conservation
The document contains a logical tension:
- Section N.1.4 states population $N$ is **fixed** ("not a limitation")
- Section N.3.3 (line 288) states "cloning creates mass (new walkers) and killing destroys mass"

These are reconciled by "revival" - dead walkers remain counted in $N$ - but this should be stated more clearly. The phrase "creates mass" is misleading when $N$ is fixed.

### Line 344 (Section N.4.1) - Dimensional Analysis Issue
**Gemini correctly flagged.** The acoustic stability constraint:
$$\gamma > \mathbb{E}[p_i] M^2 / (2dh)$$
has unclear units. Need to define what $M$ represents (Lipschitz constant? mass scale?).

### Lines 581/603 - $SU(N)$ vs $SU(d)$ Confusion
**Gemini correctly identified.** The document uses $N$ (population) and $d$ (dimension) interchangeably when discussing gauge groups. Should consistently use $SU(d)$ for color gauge group.

### NEW: Lines 721-723 - "Beliefs" Language
The statement "The walker population represents beliefs" in Section N.9.1 is imprecise. Walkers are point estimates, not probability distributions. This anthropomorphizes the algorithm unnecessarily.

## 4. Typos and Style Issues

### Line 242 - Undefined acronym "OS"
**Gemini caught this.** "OS reconstruction" should be "Osterwalder-Schrader reconstruction".

### Line 604 - Imprecise "modulo" usage
**Gemini caught this.** "$SU(d)$ (modulo the overall phase)" should be "$SU(d) \cong U(d)/U(1)$" or similar precise formulation.

### NEW: Inconsistent capitalization
- "Fractal Gas" vs "fractal gas" - should pick one and be consistent
- "Walker" vs "walker" - both used

### NEW: Section numbering
The FAQ claims to address "forty rigorous objections" (line 12) but the numbering system (N.1.1, N.2.1, etc.) makes it hard to verify there are exactly 40.

## 5. Structural Issues

### Redundant explanations
Some explanations are repeated nearly verbatim:
- WFR metric is explained in N.3.3 and referenced multiple times
- Doeblin minorization explained in N.1.2, N.4.1, and N.4.2

### Missing forward references
When discussing "Expansion Adjunction" early (e.g., line ~235), should reference which appendix contains the proof.

## Summary

**Severity ranking:**
1. **HIGH**: Momentum conservation claim (line 237) is mathematically wrong
2. **HIGH**: 9 broken cross-references need path fixes
3. **MEDIUM**: $SU(N)$ vs $SU(d)$ confusion
4. **MEDIUM**: Acoustic stability dimensional analysis unclear
5. **LOW**: Undefined acronyms, inconsistent style
