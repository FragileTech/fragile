# Literature Review Checklist: Unconditional LSI via Hypocoercivity

**Purpose**: Identify the precise mathematical conditions required to prove a Logarithmic Sobolev Inequality (LSI) for the Euclidean Gas without assuming log-concavity of the quasi-stationary distribution.

**Goal**: Prove that the full operator Ψ_total = Ψ_kin ∘ Ψ_clone satisfies an LSI with respect to π_QSD, using only:
1. Confining potential U(x) (NOT necessarily convex)
2. Positive friction γ > 0
3. Positive noise σ_v² > 0
4. Properties of the cloning operator

---

## Part 1: What We Need to Prove

### Target Theorem (Unconditional LSI)

We want to prove:

**Theorem (Target)**: Under the foundational axioms (confining potential, positive friction, positive noise), the full Euclidean Gas operator satisfies a Logarithmic Sobolev Inequality:

For all smooth functions f: S_N → ℝ with ∫ f² dπ_QSD = 1:

$$
\text{Ent}_{\pi_{\text{QSD}}}(f^2) \leq C_{\text{LSI}} \cdot \mathbb{E}_{\pi_{\text{QSD}}}[|\nabla f|^2]
$$

where C_LSI > 0 depends on:
- Physical parameters: γ, σ_v², α_U (confinement), d (dimension)
- Algorithmic parameters: τ (time step), N (number of walkers), fitness bounds
- **NOT on**: Convexity of U or log-concavity of π_QSD

### Why Standard Bakry-Émery Doesn't Apply

Classical Bakry-Émery criterion requires:
```
Γ₂(f, f) ≥ ρ Γ(f, f)
```
where Γ₂ is the "iterated carré du champ operator" and ρ > 0 is a curvature bound.

For a diffusion with generator L = Δ - ∇V · ∇, this becomes:
```
∇²V ≥ ρ I    (Hessian of potential bounded below)
```

**Problem for us**: Our potential U(x) + fitness terms is NON-CONVEX (multi-modal fitness landscapes). So ∇²V can have negative eigenvalues, and classical Bakry-Émery fails.

---

## Part 2: Key Papers to Read

### Priority 1: Hypocoercive Extensions of Bakry-Émery

#### Paper 1: Dolbeault-Mouhot-Schmeiser (2017)
**Title**: "Bakry-Émery meet Villani"
**Journal**: Journal of Functional Analysis
**DOI**: https://doi.org/10.1016/j.jfa.2017.08.003

**What to extract**:
- [ ] Precise statement of "hypocoercive Bakry-Émery criterion"
- [ ] Modified Γ₂ operator for kinetic equations
- [ ] Conditions under which hypocoercivity implies LSI
- [ ] Example applications to kinetic Fokker-Planck with non-convex potentials
- [ ] Constants: How does C_LSI depend on γ, α_U, σ_v²?

**Key questions for our application**:
1. Does the criterion apply to DISCRETE-TIME operators (our Ψ_total)?
2. Can it handle the CLONING operator (not just Langevin dynamics)?
3. What regularity is required (C², C^∞, or just confining)?

#### Paper 2: Mischler-Mouhot (2016)
**Title**: "Exponential stability of slowly decaying solutions to the kinetic Fokker-Planck equation"
**arXiv**: https://arxiv.org/abs/1412.7487

**What to extract**:
- [ ] Quantitative hypocoercivity estimates
- [ ] Treatment of non-convex confining potentials
- [ ] Relationship between spectral gap and LSI constant
- [ ] Weighted Poincaré inequalities

#### Paper 3: Grothaus-Stilgenbauer (2018)
**Title**: "φ-Entropies: convexity, coercivity and hypocoercivity for Fokker-Planck and kinetic Fokker-Planck equations"
**Journal**: Math. Models Methods Appl. Sci.
**DOI**: https://doi.org/10.1142/S0218202518500574

**What to extract**:
- [ ] Generalized entropy functionals (not just KL)
- [ ] Entropy-dissipation inequalities for kinetic equations
- [ ] Modified coercivity conditions for hypocoercive systems
- [ ] Comparison with Bakry-Émery approach

### Priority 2: Discrete-Time LSI

#### Paper 4: Caputo-Dai Pra-Posta (2009)
**Title**: "Convex entropy decay via the Bochner-Bakry-Emery approach"
**Journal**: Ann. Inst. Henri Poincaré Probab. Stat.

**What to extract**:
- [ ] Discrete-time version of Bakry-Émery
- [ ] LSI for Markov chains (not continuous-time)
- [ ] Conditions for entropy contraction per step

#### Paper 5: Caputo (2015)
**Title**: "Uniform Poincaré inequalities for unbounded conservative spin systems"
**Journal**: Ann. Probab.

**What to extract**:
- [ ] Techniques for proving LSI for particle systems
- [ ] Handling interactions between particles (relevant for cloning)

### Priority 3: Optimal Transport Approaches

#### Paper 6: Guillin-Le Bris-Monmarché (2021)
**Title**: "An optimal transport approach for hypocoercivity"
**arXiv**: https://arxiv.org/abs/2102.10667

**What to extract**:
- [ ] Wasserstein metrics adapted to kinetic equations
- [ ] Entropy-transport inequalities without log-concavity
- [ ] Explicit convergence rates

---

## Part 3: Mathematical Conditions Checklist

For each paper, extract the PRECISE conditions required for LSI. Use this checklist:

### Checklist A: Kinetic Operator Requirements

Does the paper require:

- [ ] **A.1 Confinement**: ⟨∇U(x), x⟩ ≥ α_U |x|² for |x| large
  - **Our status**: ✅ We have this (Axiom ax-confining-complete)

- [ ] **A.2 Convexity**: ∇²U ≥ κ I for some κ > 0
  - **Our status**: ❌ We do NOT have this (non-convex fitness)

- [ ] **A.3 Smoothness**: U ∈ C²(ℝ^d) with bounded Hessian
  - **Our status**: ⚠️ Partial (piecewise smooth, not globally C²)

- [ ] **A.4 Growth control**: |∇²U| ≤ C(1 + |x|^α) for some α < 2
  - **Our status**: ✅ Likely (need to verify for our U)

- [ ] **A.5 Friction**: γ > 0
  - **Our status**: ✅ We have this

- [ ] **A.6 Noise**: Uniformly elliptic diffusion σ_v² I with σ_v² > 0
  - **Our status**: ✅ We have this

- [ ] **A.7 Hypoellipticity**: Transport term v·∇_x couples position and velocity
  - **Our status**: ✅ This is the key to hypocoercivity!

### Checklist B: Operator Structure

Does the paper handle:

- [ ] **B.1 Continuous-time**: Generator L = ... with continuous Markov semigroup e^{tL}
  - **Our status**: ❌ We have discrete-time Ψ_total
  - **Gap**: Need discrete-time version or approximation argument

- [ ] **B.2 Discrete-time**: Markov kernel P with P^n → π
  - **Our status**: ✅ This is what we have (Ψ_total)

- [ ] **B.3 Composition**: Operator = Diffusion ∘ Jumps (like our Ψ_kin ∘ Ψ_clone)
  - **Our status**: ✅ This is exactly our structure!

- [ ] **B.4 Splitting**: Can decompose into "simple" + "complex" parts
  - **Our status**: ✅ Ψ_kin (hypocoercive) + Ψ_clone (contractive)

### Checklist C: Interacting Particles

Does the paper handle:

- [ ] **C.1 Single particle**: Analysis for one walker in potential
  - **Our status**: ✅ We can prove LSI for Ψ_kin alone (Part 2 of 10_T_non_convex_extensions.md)

- [ ] **C.2 N independent particles**: Product measure π = π₁ ⊗ ... ⊗ π_N
  - **Our status**: ⚠️ True for Ψ_kin, but NOT for Ψ_clone (interacting)

- [ ] **C.3 Mean-field interaction**: Particles interact through empirical measure
  - **Our status**: ✅ This is what cloning does! g(x, v, S) depends on all walkers

- [ ] **C.4 Propagation of chaos**: LSI holds in N → ∞ limit
  - **Our status**: ⚠️ Need to check if paper requires this or proves uniform-in-N

### Checklist D: LSI Statement

What form of LSI does the paper prove:

- [ ] **D.1 Standard LSI**: Ent(f²) ≤ C·E[|∇f|²]
  - **What we want**: ✅ This is the goal

- [ ] **D.2 Weighted LSI**: Ent_μ(f²) ≤ C·∫|∇f|² w(x) dμ(x) for some weight w
  - **What we want**: ⚠️ Could be sufficient if w = |∇V|² or similar

- [ ] **D.3 Modified LSI**: Ent(f²) ≤ C·[E[|∇f|²] + E[|∇_v f|²]] with coupling
  - **What we want**: ✅ This is hypocoercive LSI! Acceptable.

- [ ] **D.4 φ-entropy**: Ent_φ(f²) ≤ C·E[|∇f|²] for generalized entropy
  - **What we want**: ⚠️ Need to understand relationship to standard LSI

### Checklist E: Constants and Rates

What does the paper say about the LSI constant C_LSI:

- [ ] **E.1 Explicit formula**: C_LSI = f(γ, α_U, σ_v², d, ...)
  - **What we need**: ✅ Must track parameter dependence for Yang-Mills

- [ ] **E.2 N-dependence**: Does C_LSI depend on N (number of particles)?
  - **What we need**: ❌ Must be independent of N (N-uniformity)

- [ ] **E.3 Small parameter**: C_LSI → 0 as γ → 0 or α_U → 0?
  - **What we need**: ✅ Want C_LSI bounded even for moderate γ, α_U

- [ ] **E.4 Discretization**: How does discrete-time Δt = τ affect constant?
  - **What we need**: ✅ Track dependence on τ

---

## Part 4: Conditions Summary Template

For each paper, fill out:

### Paper: [Title]

**Applies to our problem?** [YES / NO / PARTIAL]

**Conditions satisfied:**
- [ ] Condition 1: ...
- [ ] Condition 2: ...
- [ ] ...

**Conditions NOT satisfied:**
- [ ] Condition X: ... (Why we don't have this)

**Key theorem:**
[State the main LSI theorem from the paper]

**Adaptation needed:**
[What modifications are required to apply this to Euclidean Gas?]

**Conclusion:**
[Can we use this paper? What additional work is needed?]

---

## Part 5: Implementation Checklist

Once literature review is complete:

### Step 5.1: Identify Best Approach

- [ ] Which paper provides the most promising framework?
- [ ] What is the minimum set of conditions we need to verify?
- [ ] Are there dealbreakers (conditions we provably DON'T satisfy)?

### Step 5.2: Verify Conditions for Euclidean Gas

For the chosen approach, rigorously verify:

- [ ] All operator structure conditions hold
- [ ] Modified Γ₂ operator can be computed
- [ ] Curvature/coercivity bounds can be established
- [ ] N-uniformity is preserved
- [ ] Discrete-time adaption works

### Step 5.3: Write the Proof

Create `10_U_unconditional_lsi_via_hypocoercivity.md` with:

- [ ] Section 1: Statement of unconditional LSI theorem
- [ ] Section 2: Review of hypocoercive Bakry-Émery framework (cite paper)
- [ ] Section 3: Verification that Euclidean Gas satisfies conditions
- [ ] Section 4: Computation of LSI constant
- [ ] Section 5: Main proof
- [ ] Section 6: Comparison with conditional proof

### Step 5.4: Update All Documents

- [ ] Update `10_kl_convergence_unification.md` to reference unconditional proof
- [ ] Update `hydrodynamics.md` Yang-Mills section
- [ ] Update `15_millennium_problem_completion.md`
- [ ] Remove all "conditional on log-concavity" warnings
- [ ] Add citation to hypocoercive LSI paper(s)

---

## Part 6: Fallback Plan

If no paper provides applicable framework:

### Option A: Prove Weaker Result

Prove **modified LSI** or **local LSI**:
- Local LSI within basins of attraction
- Global mixing via Foster-Lyapunov
- Combined two-scale argument

### Option B: Strengthen Assumptions

Accept conditional result but strengthen justification:
- Numerical verification of log-concavity for test cases
- Prove log-concavity in limiting regimes (γ → ∞, etc.)
- Physical arguments for plausibility

### Option C: State Clearly as Open Problem

- Document the gap honestly
- State Yang-Mills claim as conditional
- Propose unconditional LSI as major open problem
- Publish anyway (conditional results are still valuable!)

---

## Expected Timeline

- **Literature review** (Papers 1-6): 1 week
- **Condition verification**: 3-5 days
- **Proof writing** (if conditions hold): 1-2 weeks
- **Document updates**: 2-3 days

**Total**: 3-4 weeks if everything works out, longer if complications arise.

---

## Notes and Updates

[Space for recording discoveries during literature review]

### Paper 1 Notes:
[To be filled]

### Paper 2 Notes:
[To be filled]

...
