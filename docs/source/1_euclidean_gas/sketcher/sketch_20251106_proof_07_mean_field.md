# Proof Sketch for thm-mean-field-equation

**Document**: docs/source/1_euclidean_gas/07_mean_field.md
**Theorem**: thm-mean-field-equation
**Generated**: 2025-11-06
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} The Mean-Field Equations for the Euclidean Gas
:label: thm-mean-field-equation

The evolution of the Euclidean Gas in the mean-field limit is governed by a coupled system of equations for the alive density $f(t,z)$ and the dead mass $m_d(t)$:

**Equation for the Alive Density:**

$$
\boxed{
\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]
}
$$

**Equation for the Dead Mass:**

$$
\boxed{
\frac{\mathrm{d}}{\mathrm{d}t} m_d(t) = \int_{\Omega} c(z)f(t,z)\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)
}
$$

subject to initial conditions $f(0, \cdot) = f_0$ and $m_d(0) = 1 - \int_\Omega f_0$, where $m_a(0) + m_d(0) = 1$.

In explicit form, the equation for $f$ is:

$$
\partial_t f(t,z) = -\nabla\cdot(A(z) f(t,z)) + \nabla\cdot(\mathsf{D}\nabla f(t,z)) - c(z)f(t,z) + \lambda_{\text{revive}} m_d(t) \frac{f(t,z)}{m_a(t)} + S[f](t,z)
$$

where:
*   $A(z)$ is the drift field and $\mathsf{D}$ is the diffusion tensor from the kinetic transport (with reflecting boundaries)
*   $c(z)$ is the interior killing rate (zero in interior, positive near boundary)
*   $\lambda_{\text{revive}} > 0$ is the revival rate (free parameter, typical values 0.1-5)
*   $B[f, m_d] = \lambda_{\text{revive}} m_d(t) f/m_a$ is the revival operator
*   $S[f]$ is the mass-neutral internal cloning operator

The total alive mass is $m_a(t) = \int_\Omega f(t,z)\,\mathrm{d}z$, and the system conserves the total population: $m_a(t) + m_d(t) = 1$ for all $t$.
:::

**Informal Restatement**: The theorem states that the continuous-time evolution of the Euclidean Gas algorithm in the mean-field limit (as the number of walkers N→∞) is described by a coupled PDE-ODE system. The PDE governs the spatial-velocity density of "alive" walkers, accounting for kinetic transport (drift-diffusion from Langevin dynamics), death near boundaries, revival from a dead reservoir, and internal cloning. The ODE tracks the total dead mass, balancing the flow of walkers dying and being revived. The key claim is that these two equations, when coupled, exactly capture the algorithm's macroscopic behavior and conserve total population mass.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Method**: Direct Proof / Derivation from Physical Principles

**Key Steps**:
1. Derive PDE for alive density using continuity equation: $\partial_t f = -\nabla \cdot \mathbf{J} + Q_{net}$
2. Derive ODE for dead mass by integrating PDE over $\Omega$ and using total mass conservation
3. Verify total mass conservation: $\frac{\mathrm{d}}{\mathrm{d}t}(m_a + m_d) = 0$
4. Verify consistency of initial conditions

**Strengths**:
- Clear physical motivation: treats this as an assembly problem rather than a deep theorem
- Straightforward operator-by-operator construction
- Explicit verification of mass conservation as a sanity check
- Recognizes this is a derivation/assembly task, not an existence proof

**Weaknesses**:
- Does not emphasize weak formulation rigor
- Less detail on boundary term handling in weak sense
- Does not explicitly address functional-analytic regularity requirements
- Treats Leibniz integral rule somewhat casually

**Framework Dependencies**:
- def-kinetic-generator (Fokker-Planck operator)
- lem-mass-conservation-transport (∫L†f = 0)
- def-killing-operator, def-revival-operator, def-cloning-generator
- Total mass conservation axiom

---

### Strategy B: GPT-5's Approach

**Method**: Direct Proof via Weak Formulation and Operator Assembly

**Key Steps**:
1. Set up weak formulation for kinetic transport with test functions $\phi \in C_c^\infty(\Omega)$
2. Add interior killing as sink: $-c(z)f$
3. Add revival as source: $B[f, m_d]$
4. Add internal cloning: $S[f]$ with mass-neutral property
5. Derive PDE in explicit conservative form
6. Derive ODE for dead mass from integrated mass flows
7. Prove total mass conservation and initial consistency

**Strengths**:
- More rigorous weak formulation approach (pairing with test functions)
- Explicit boundary term treatment via reflecting conditions
- Clear identification of regularity requirements (f ∈ L¹∩L∞)
- Systematic derivation of ODE from PDE via integration
- More thorough handling of functional-analytic details

**Weaknesses**:
- More technically demanding presentation
- Requires more background in PDE weak formulation theory
- Some steps could be viewed as overly formal for an assembly theorem

**Framework Dependencies**:
- Same as Gemini, plus explicit weak form machinery
- More careful about regularity assumptions and trace theory for boundaries

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Hybrid Direct Derivation with Weak Formulation Rigor

**Rationale**:
Both strategists correctly identify that this theorem is an **assembly/derivation statement** rather than a deep existence theorem. The goal is to show that combining the four pre-defined operators (transport, killing, revival, cloning) produces the claimed coupled PDE-ODE system.

**Evidence-based synthesis**:
1. **Primary approach**: Follow Gemini's physical/intuitive structure (continuity equation + operator assembly)
2. **Rigor enhancements**: Adopt GPT-5's weak formulation machinery where needed (boundary terms, regularity)
3. **Key insight**: The proof has two logically distinct parts:
   - **Part A (Steps 1-4)**: Assembly of the PDE from operator definitions (straightforward)
   - **Part B (Steps 5-6)**: Derivation of the ODE by integrating the PDE (requires care with regularity)

**Integration**:
- Steps 1-3: Use Gemini's direct operator assembly (cleaner exposition)
- Step 4: Use GPT-5's weak formulation for boundary term justification (more rigorous)
- Step 5: Use GPT-5's systematic integration approach (clearer logic)
- Step 6: Use Gemini's algebraic verification (good sanity check)

**Verification Status**:
- ✅ All framework dependencies verified (operators defined in same document)
- ✅ No circular reasoning (operators → equations, not vice versa)
- ⚠ Requires assumption: $m_a(t) > 0$ for $B[f,m_d]$ well-definedness (both strategists note this)
- ⚠ Existence/uniqueness is NOT part of this theorem (both correctly identify this as separate)

---

## III. Framework Dependencies

### Verified Dependencies

**Definitions** (from same document):

| Label | Line | Statement | Used in Step | Verified |
|-------|------|-----------|--------------|----------|
| def-mean-field-phase-space | 39 | Phase space $\Omega = X_{\text{valid}} \times V_{\text{alg}}$ | All steps | ✅ |
| def-phase-space-density | 61 | Alive density $f(t,z)$ with $m_a(t) = \int_\Omega f\,\mathrm{d}z$ | All steps | ✅ |
| def-kinetic-generator | 311 | Langevin SDE generator, reflecting boundaries | Step 1 | ✅ |
| def-transport-operator | 554 | Transport operator $L^\dagger f = -\nabla \cdot J[f]$ | Step 1 | ✅ |
| def-killing-operator | 360 | Interior killing rate $c(z)$, $k_{\text{killed}}[f] = \int c(z)f\,\mathrm{d}z$ | Steps 1,5 | ✅ |
| def-revival-operator | 378 | Revival $B[f,m_d] = \lambda_{\text{rev}} m_d f/m_a$ | Steps 1,5 | ✅ |
| def-cloning-generator | 497 | Cloning $S[f] = S_{\text{src}} - S_{\text{sink}}$, mass-neutral | Steps 1,5 | ✅ |

**Lemmas** (proven in same document):

| Label | Line | Statement | Used in Step | Verified |
|-------|------|-----------|--------------|----------|
| lem-mass-conservation-transport | 572 | $\int_\Omega L^\dagger f\,\mathrm{d}z = 0$ (reflecting boundaries) | Step 5 | ✅ |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $\lambda_{\text{rev}}$ (or $\lambda_{\text{revive}}$) | Revival rate | $> 0$, typical 0.1-5 | Free parameter, N-independent |
| $c(z)$ | Killing rate function | $c \in C^\infty(\Omega)$, $\geq 0$ | Zero in interior, positive near boundary |
| $A(z)$, $\mathsf{D}$ | Drift field, diffusion tensor | From Langevin SDE | Bounded by framework axioms |

### Missing/Uncertain Dependencies

**Requires Additional Assumptions** (not lemmas to prove):

- **Assumption A**: Positivity of alive mass
  - **Statement**: For the revival operator $B[f,m_d] = \lambda_{\text{rev}} m_d f/m_a$ to be well-defined, we require $m_a(t) > 0$ for all $t \geq 0$.
  - **Why needed**: Division by zero otherwise
  - **How to handle**:
    1. Assume $m_a(0) > 0$ initially
    2. Separately prove positivity preservation (requires Axiom of Guaranteed Revival and $\lambda_{\text{rev}} > 0$)
    3. Alternatively, define $B \equiv 0$ when $m_a = 0$ as limiting case
  - **Status**: Standard assumption for this model; positivity preservation is a separate well-posedness question

- **Assumption B**: Regularity for Leibniz integral rule
  - **Statement**: To justify $\frac{\mathrm{d}}{\mathrm{d}t} \int_\Omega f = \int_\Omega \partial_t f$, we need sufficient regularity of $f$
  - **Why needed**: Exchange derivative and integral
  - **How to handle**:
    1. Assume $f \in C([0,\infty); L^1(\Omega))$ (stated in def-phase-space-density)
    2. Or work in weak/distributional sense with test functions
  - **Status**: Regularity assumption already made in framework definition (line 80)

**Notation Clarification**:
- The theorem uses both $\lambda_{\text{rev}}$ and $\lambda_{\text{revive}}$ for the same parameter. These should be unified (recommend $\lambda_{\text{rev}}$ for brevity).

---

## IV. Detailed Proof Sketch

### Overview

This proof establishes the mean-field equations by systematic assembly of four independently-defined physical operators: kinetic transport, interior killing, revival from dead population, and internal cloning. The strategy has two main components:

1. **PDE Assembly (Steps 1-4)**: Construct the evolution equation for $f$ by invoking the continuity equation for probability density and identifying each term with a physical process. The kinetic transport gives the flux divergence, while the reaction operators (killing, revival, cloning) provide local sources and sinks.

2. **ODE Derivation (Steps 5-6)**: Derive the dead mass equation by integrating the PDE over the entire phase space and exploiting the mass-neutral properties of transport and cloning. The killing and revival rates naturally couple the two equations.

The key mathematical insight is that the coupled system preserves **total population mass** $m_a(t) + m_d(t) = 1$ by construction, with mass flowing between the alive and dead populations at precisely balanced rates.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Operator Assembly for PDE**: Combine transport, killing, revival, and cloning into the claimed PDE form
2. **Weak Formulation Justification**: Verify the assembly is rigorous using test functions and boundary conditions
3. **Explicit Conservative Form**: Expand the transport operator into drift-diffusion representation
4. **Integration over Phase Space**: Compute $\frac{\mathrm{d}}{\mathrm{d}t}m_a(t)$ by integrating the PDE
5. **ODE Derivation**: Use total mass conservation to obtain the dead mass equation
6. **Verification**: Confirm mass conservation and initial condition consistency

---

### Detailed Step-by-Step Sketch

#### Step 1: Assemble the PDE from the Continuity Equation

**Goal**: Derive $\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]$ from physical principles.

**Substep 1.1**: Invoke the general continuity equation for probability density
- **Action**: For any conserved probability density, the time evolution satisfies:
  $$
  \partial_t f = -\nabla \cdot \mathbf{J} + Q_{\text{net}}
  $$
  where $\mathbf{J}$ is the probability flux and $Q_{\text{net}}$ is the net local source/sink rate.
- **Justification**: Standard result from statistical physics and continuum mechanics. This is the fundamental balance equation for any conserved quantity.
- **Why valid**: Follows from conservation of mass in infinitesimal volume elements.
- **Expected result**: A template equation with two components to fill in.

**Substep 1.2**: Identify the flux term with the kinetic transport operator
- **Action**: The probability flux from continuous stochastic motion (Langevin dynamics) is exactly the flux represented by the transport operator $L^\dagger f = -\nabla \cdot J[f]$ (def-transport-operator, line 554).
- **Justification**: $L^\dagger$ is defined as the formal $L^2$-adjoint of the backward kinetic generator, which is the Fokker-Planck operator for the Langevin SDE (def-kinetic-generator, line 311).
- **Why valid**: This is the defining relationship between the backward generator $L$ and forward operator $L^\dagger$ for Markov processes. The flux form $J[f] = (J_x, J_v)$ with $J_x = vf - D_x\nabla_x f$ and $J_v = A_v f - D_v \nabla_v f$ is derived from the SDE coefficients.
- **Expected result**: $\partial_t f = L^\dagger f + Q_{\text{net}}$

**Substep 1.3**: Identify the source/sink terms
- **Action**: The net local rate of change from reactions is the sum of three operators:
  1. **Killing**: $-c(z)f$ removes mass at rate $c(z)$ (def-killing-operator, line 360)
  2. **Revival**: $+B[f, m_d] = +\lambda_{\text{rev}} m_d(t) f/m_a(t)$ adds mass from dead reservoir (def-revival-operator, line 378)
  3. **Cloning**: $+S[f]$ redistributes alive mass neutrally (def-cloning-generator, line 497)
- **Justification**: In the mean-field limit, operators act independently at the infinitesimal level, so their contributions add linearly. This is a core assumption of the mean-field approximation: the probability of two events (e.g., killing AND cloning) occurring simultaneously in time $\mathrm{d}t$ is $o(\mathrm{d}t)$ and vanishes in the limit.
- **Why valid**: Each operator is defined separately in Section 2.3. The linear superposition follows from the independence of physical processes in the continuous-time limit.
- **Expected result**: $Q_{\text{net}} = -c(z)f + B[f, m_d] + S[f]$

**Substep 1.4**: Combine to obtain the boxed PDE
- **Action**: Substitute the identifications from Substeps 1.2 and 1.3 into the continuity equation:
  $$
  \boxed{\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]}
  $$
- **Conclusion**: This is the first claimed equation of the theorem.
- **Form**: PDE on $\Omega$ for the density function $f(t,z)$.

**Dependencies**:
- Uses: def-transport-operator, def-killing-operator, def-revival-operator, def-cloning-generator
- Requires: Mean-field independence assumption (operators act additively)

**Potential Issues**:
- ⚠ Well-definedness of $B[f, m_d] = \lambda_{\text{rev}} m_d f/m_a$ when $m_a \to 0$
- **Resolution**: Assume $m_a(t) > 0$ throughout (initial condition with $m_a(0) > 0$ and prove positivity preservation separately, using Axiom of Guaranteed Revival). Alternatively, adopt the convention $B \equiv 0$ when $m_a = 0$, which is consistent with the limiting behavior (no alive walkers means no spatial profile for revival).

---

#### Step 2: Justify Operator Assembly via Weak Formulation

**Goal**: Verify that the operator assembly in Step 1 is mathematically rigorous, particularly for the kinetic transport with boundary conditions.

**Substep 2.1**: Write the weak form with test functions
- **Action**: For any test function $\phi \in C_c^\infty(\Omega)$ (smooth, compactly supported), the weak form of the PDE is:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}\langle \phi, f \rangle = \langle \phi, L^\dagger f \rangle + \langle \phi, -c(z)f + B[f,m_d] + S[f] \rangle
  $$
  where $\langle \phi, f \rangle := \int_\Omega \phi(z) f(t,z)\,\mathrm{d}z$ is the $L^2$ pairing.
- **Justification**: This is the definition of weak solution. It makes sense for $f \in L^1(\Omega)$ even if $f$ is not smooth.
- **Why valid**: Test function framework is standard in PDE theory. The pairing is well-defined since $\phi$ is bounded and compactly supported, and $f \in L^1$.
- **Expected result**: Weak form equation for all test functions $\phi$.

**Substep 2.2**: Verify the kinetic transport term via integration by parts
- **Action**: The kinetic transport term $\langle \phi, L^\dagger f \rangle$ can be written as:
  $$
  \langle \phi, L^\dagger f \rangle = \langle L\phi, f \rangle + \text{Boundary Terms}
  $$
  where $L$ is the backward generator. Integration by parts on the flux form $L^\dagger f = -\nabla \cdot J[f]$ gives:
  $$
  \langle \phi, -\nabla \cdot J[f] \rangle = \langle \nabla\phi, J[f] \rangle - \int_{\partial\Omega} \phi (J[f] \cdot n)\,\mathrm{d}S
  $$
- **Justification**: Divergence theorem (Gauss's theorem).
- **Why valid**: $\Omega$ is a bounded domain with smooth boundary (from def-mean-field-phase-space, line 39). The boundary integral exists in the trace sense.
- **Expected result**: Explicit boundary term that must vanish.

**Substep 2.3**: Apply reflecting boundary conditions to cancel boundary term
- **Action**: The reflecting boundary conditions on $\partial X_{\text{valid}}$ and $\partial V_{\text{alg}}$ ensure that the normal component of the flux vanishes:
  $$
  J[f] \cdot n = 0 \quad \text{on } \partial\Omega
  $$
  This is proven in lem-mass-conservation-transport (line 572-597) using the specific reflection conditions:
  - On $\partial V_{\text{alg}}$: $J_v \cdot n_v = 0$ (velocity reflection)
  - On $\partial X_{\text{valid}}$: $J_x \cdot n_x = 0$ (position reflection)
- **Justification**: lem-mass-conservation-transport establishes this via divergence theorem.
- **Why valid**: The lemma is already proven in the document.
- **Expected result**: Boundary integral vanishes, so $\langle \phi, L^\dagger f \rangle = \langle \nabla\phi, J[f] \rangle$ (no boundary contribution).

**Substep 2.4**: Verify reaction terms are well-defined in weak sense
- **Action**: The killing, revival, and cloning terms are all pointwise multiplication or integral operators acting on $f$, so their pairings with $\phi$ are well-defined:
  - Killing: $\langle \phi, -c(z)f \rangle = -\int_\Omega c(z)\phi(z) f(t,z)\,\mathrm{d}z$ (exists since $c \in C^\infty$ and $f \in L^1$)
  - Revival: $\langle \phi, B[f,m_d] \rangle = \lambda_{\text{rev}} m_d(t) \int_\Omega \phi(z) \frac{f(t,z)}{m_a(t)}\,\mathrm{d}z$ (exists if $m_a(t) > 0$)
  - Cloning: $\langle \phi, S[f] \rangle$ is an integral of $\phi$ against the source-sink distribution (well-defined for bounded $\phi$ and $f \in L^1$)
- **Justification**: Standard weak formulation machinery.
- **Why valid**: All operators are constructed from integrals and pointwise products with smooth or bounded kernels.
- **Expected result**: Weak form is mathematically rigorous for $f \in C([0,\infty); L^1(\Omega))$.

**Dependencies**:
- Uses: lem-mass-conservation-transport (boundary flux vanishes)
- Requires: $f \in L^1(\Omega)$, $m_a(t) > 0$ for revival term

**Potential Issues**:
- ⚠ Trace theory for boundary terms in kinetic Fokker-Planck (hypoelliptic structure)
- **Resolution**: The reflecting boundary conditions are part of the definition of $L^\dagger$ (def-kinetic-generator, line 311). The vanishing of boundary flux is established by lem-mass-conservation-transport, which already handles the necessary trace theory implicitly via the divergence theorem.

---

#### Step 3: Expand Transport Operator into Explicit Conservative Form

**Goal**: Express the PDE in the explicit drift-diffusion form stated in the theorem.

**Substep 3.1**: Expand $L^\dagger f$ into divergence form
- **Action**: The transport operator has the explicit flux representation (def-transport-operator, line 554-567):
  $$
  L^\dagger f = -\nabla_x \cdot J_x[f] - \nabla_v \cdot J_v[f]
  $$
  where:
  - Positional flux: $J_x[f] = v f - D_x \nabla_x f$
  - Velocity flux: $J_v[f] = A_v f - D_v \nabla_v f$
- **Justification**: This is the definition of $L^\dagger$ as a Fokker-Planck operator.
- **Why valid**: Derived from the Langevin SDE coefficients in def-kinetic-generator (line 311-335).
- **Expected result**: $L^\dagger f = -\nabla_x \cdot (v f) + \nabla_x \cdot (D_x \nabla_x f) - \nabla_v \cdot (A_v f) + \nabla_v \cdot (D_v \nabla_v f)$

**Substep 3.2**: Combine spatial and velocity contributions
- **Action**: Write the full operator using combined notation. Let $A(z) = (v, A_v(z))$ be the full phase-space drift and $\mathsf{D}$ be the diffusion tensor (diagonal, with $D_x$ for position and $D_v$ for velocity). Then:
  $$
  L^\dagger f = -\nabla \cdot (A(z) f(t,z)) + \nabla \cdot (\mathsf{D} \nabla f(t,z))
  $$
- **Justification**: Compact notation for the drift-diffusion structure.
- **Why valid**: This is just a notational rewriting of Substep 3.1.
- **Expected result**: Conservative form suitable for PDE analysis.

**Substep 3.3**: Substitute into the PDE
- **Action**: Replace $L^\dagger f$ in the boxed equation from Step 1:
  $$
  \partial_t f(t,z) = -\nabla\cdot(A(z) f(t,z)) + \nabla\cdot(\mathsf{D}\nabla f(t,z)) - c(z)f(t,z) + \lambda_{\text{rev}} m_d(t) \frac{f(t,z)}{m_a(t)} + S[f](t,z)
  $$
- **Conclusion**: This is the explicit form stated in the theorem (following the boxed alive density equation).
- **Form**: Nonlinear, non-local PDE with drift-diffusion structure plus reaction terms.

**Dependencies**:
- Uses: def-kinetic-generator, def-transport-operator
- Requires: Nothing beyond what's in Step 1

**Potential Issues**: None (this is purely notational expansion).

---

#### Step 4: Integration Setup - Leibniz Rule and Regularity

**Goal**: Prepare to integrate the PDE over $\Omega$ to derive the ODE for $m_d(t)$.

**Substep 4.1**: Define the alive mass functional
- **Action**: The total alive mass is:
  $$
  m_a(t) := \int_\Omega f(t,z)\,\mathrm{d}z
  $$
  By the total mass conservation principle (def-phase-space-density, line 72-78), we have:
  $$
  m_d(t) = 1 - m_a(t)
  $$
- **Justification**: Definition from the framework.
- **Why valid**: This is part of the model setup (alive + dead = 1).
- **Expected result**: $m_d(t)$ is determined by $m_a(t)$.

**Substep 4.2**: Differentiate $m_a(t)$ with respect to time
- **Action**: We want to compute:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = \frac{\mathrm{d}}{\mathrm{d}t} \int_\Omega f(t,z)\,\mathrm{d}z
  $$
  Exchange derivative and integral using Leibniz's integral rule:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t} \int_\Omega f(t,z)\,\mathrm{d}z = \int_\Omega \frac{\partial f}{\partial t}(t,z)\,\mathrm{d}z
  $$
- **Justification**: Leibniz's rule for parameter-dependent integrals. The domain $\Omega$ is fixed (not time-dependent), so no boundary terms arise from the domain itself.
- **Why valid**: Requires $f \in C([0,\infty); L^1(\Omega))$, which is assumed in def-phase-space-density (line 80). Alternatively, this can be justified in the weak/distributional sense using test functions constant in space.
- **Expected result**: $\frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = \int_\Omega \partial_t f(t,z)\,\mathrm{d}z$

**Substep 4.3**: State the strategy for Step 5
- **Action**: Substitute the PDE for $\partial_t f$ (from Step 1) into the integral, then evaluate each term using the mass properties of the operators.
- **Justification**: Direct calculation.
- **Expected result**: An ODE for $m_a(t)$ involving known quantities.

**Dependencies**:
- Uses: def-phase-space-density (regularity assumption)
- Requires: $f \in C([0,\infty); L^1(\Omega))$

**Potential Issues**:
- ⚠ Justifying Leibniz rule rigorously for PDE solutions
- **Resolution**: The regularity assumption $f \in C([0,\infty); L^1(\Omega))$ is explicitly stated in def-phase-space-density (line 80). For full rigor in a weak sense, one can use test functions and argue in distributions, as GPT-5 suggests. This gives a "weak ODE" that can be made classical if $m_a(t)$ is absolutely continuous, which follows from the boundedness of the right-hand side.

---

#### Step 5: Derive the ODE for Dead Mass by Integration

**Goal**: Integrate the PDE over $\Omega$ and use operator mass properties to obtain the dead mass ODE.

**Substep 5.1**: Integrate the PDE
- **Action**: Substitute the PDE from Step 1 into the result of Step 4:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = \int_\Omega (L^\dagger f - c(z)f + B[f, m_d] + S[f])(t,z)\,\mathrm{d}z
  $$
  This is a sum of four integrals. Evaluate each separately.
- **Justification**: Linearity of integration.
- **Why valid**: All terms are in $L^1(\Omega)$ (integrable).
- **Expected result**: Four separate integrals to evaluate.

**Substep 5.2**: Evaluate the transport integral using mass conservation
- **Action**: Compute:
  $$
  \int_\Omega L^\dagger f(t,z)\,\mathrm{d}z = 0
  $$
- **Justification**: lem-mass-conservation-transport (line 572-597). The transport operator with reflecting boundary conditions is mass-neutral.
- **Why valid**: This is a proven lemma in the document. The reflecting boundaries ensure no flux escapes through $\partial\Omega$.
- **Expected result**: Transport contributes zero to $\frac{\mathrm{d}}{\mathrm{d}t}m_a$.

**Substep 5.3**: Evaluate the killing integral
- **Action**: Compute:
  $$
  \int_\Omega (-c(z)f(t,z))\,\mathrm{d}z = -\int_\Omega c(z)f(t,z)\,\mathrm{d}z = -k_{\text{killed}}[f](t)
  $$
  where $k_{\text{killed}}[f]$ is the total killed mass rate defined in def-killing-operator (line 360-376).
- **Justification**: Definition of $k_{\text{killed}}[f]$.
- **Why valid**: Direct substitution.
- **Expected result**: Killing removes mass at rate $k_{\text{killed}}[f]$.

**Substep 5.4**: Evaluate the revival integral
- **Action**: Compute:
  $$
  \int_\Omega B[f, m_d](t,z)\,\mathrm{d}z = \int_\Omega \lambda_{\text{rev}} m_d(t) \frac{f(t,z)}{m_a(t)}\,\mathrm{d}z
  $$
  Factor out constants:
  $$
  = \lambda_{\text{rev}} m_d(t) \cdot \frac{1}{m_a(t)} \int_\Omega f(t,z)\,\mathrm{d}z = \lambda_{\text{rev}} m_d(t) \cdot \frac{m_a(t)}{m_a(t)} = \lambda_{\text{rev}} m_d(t)
  $$
- **Justification**: def-revival-operator (line 378-403) states that the total revival rate is $\lambda_{\text{rev}} m_d(t)$.
- **Why valid**: The integral of the normalized density $f/m_a$ over $\Omega$ is 1 by definition (it's a probability distribution over the alive population).
- **Expected result**: Revival adds mass at rate $\lambda_{\text{rev}} m_d(t)$.

**Substep 5.5**: Evaluate the cloning integral
- **Action**: Compute:
  $$
  \int_\Omega S[f](t,z)\,\mathrm{d}z = 0
  $$
- **Justification**: def-cloning-generator (line 497-542) proves that the cloning operator is mass-neutral: $\int_\Omega S[f]\,\mathrm{d}z = 0$.
- **Why valid**: This is a proven property of $S[f]$ (see the explicit calculation at line 526-539 showing $S_{\text{src}}$ and $S_{\text{sink}}$ cancel when integrated).
- **Expected result**: Cloning contributes zero to total mass change (it only redistributes alive mass).

**Substep 5.6**: Combine results to get $\frac{\mathrm{d}}{\mathrm{d}t}m_a(t)$
- **Action**: Sum the four integrals:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = 0 - k_{\text{killed}}[f](t) + \lambda_{\text{rev}} m_d(t) + 0
  $$
  Simplify:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = -\int_\Omega c(z)f(t,z)\,\mathrm{d}z + \lambda_{\text{rev}} m_d(t)
  $$
- **Conclusion**: This is the ODE for the alive mass.
- **Form**: Explicit ODE showing how killing decreases and revival increases $m_a(t)$.

**Substep 5.7**: Derive the dead mass ODE
- **Action**: Use the constraint $m_a(t) + m_d(t) = 1$ to get:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_d(t) = -\frac{\mathrm{d}}{\mathrm{d}t}m_a(t)
  $$
  Substitute the result from Substep 5.6:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_d(t) = -\left(-\int_\Omega c(z)f(t,z)\,\mathrm{d}z + \lambda_{\text{rev}} m_d(t)\right)
  $$
  Simplify:
  $$
  \boxed{\frac{\mathrm{d}}{\mathrm{d}t} m_d(t) = \int_{\Omega} c(z)f(t,z)\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)}
  $$
- **Conclusion**: This is the second claimed equation of the theorem (the boxed dead mass ODE).
- **Form**: Linear ODE for $m_d(t)$ with source from killing and sink from revival.

**Dependencies**:
- Uses: lem-mass-conservation-transport, def-killing-operator, def-revival-operator, def-cloning-generator
- Requires: All mass properties of operators

**Potential Issues**: None at this stage (all calculations are straightforward once the operator properties are established).

---

#### Step 6: Verify Total Mass Conservation

**Goal**: Confirm that the coupled PDE-ODE system conserves total mass, as a consistency check.

**Substep 6.1**: Add the time derivatives of alive and dead mass
- **Action**: From Substep 5.6 and 5.7, we have:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = -\int_\Omega c(z)f\,\mathrm{d}z + \lambda_{\text{rev}} m_d(t)
  $$
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_d(t) = \int_{\Omega} c(z)f\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)
  $$
  Add them:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}(m_a(t) + m_d(t)) = \left[-\int_\Omega c(z)f\,\mathrm{d}z + \lambda_{\text{rev}} m_d(t)\right] + \left[\int_{\Omega} c(z)f\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)\right]
  $$
- **Justification**: Direct algebraic sum.
- **Why valid**: Both terms are well-defined ODEs.
- **Expected result**: Cancellation of all terms.

**Substep 6.2**: Observe perfect cancellation
- **Action**: The killing and revival terms cancel:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}(m_a(t) + m_d(t)) = 0
  $$
- **Conclusion**: The total mass $m_a(t) + m_d(t)$ is constant in time.
- **Form**: Conservation law confirmed.

**Substep 6.3**: Verify initial condition consistency
- **Action**: At $t = 0$, the initial conditions are:
  $$
  f(0, z) = f_0(z), \quad m_a(0) = \int_\Omega f_0\,\mathrm{d}z, \quad m_d(0) = 1 - \int_\Omega f_0
  $$
  Check:
  $$
  m_a(0) + m_d(0) = \int_\Omega f_0 + \left(1 - \int_\Omega f_0\right) = 1
  $$
- **Justification**: Direct substitution of initial conditions.
- **Why valid**: Initial conditions are stated in the theorem.
- **Expected result**: Total mass is 1 at $t = 0$.

**Substep 6.4**: Conclude total mass is conserved for all time
- **Action**: Since $\frac{\mathrm{d}}{\mathrm{d}t}(m_a + m_d) = 0$ and $m_a(0) + m_d(0) = 1$, we have:
  $$
  m_a(t) + m_d(t) = 1 \quad \forall t \geq 0
  $$
- **Conclusion**: The coupled system conserves total population mass, as claimed in the theorem.
- **Form**: Global conservation law.

**Dependencies**:
- Uses: Results from Step 5
- Requires: Nothing new (this is verification)

**Potential Issues**: None (this is a sanity check that confirms the derivation is correct).

**Q.E.D.** (for the derivation/assembly part of the theorem) ∎

---

## V. Technical Deep Dives

### Challenge 1: Well-Definedness of Revival Operator at $m_a = 0$

**Why Difficult**: The revival operator $B[f, m_d] = \lambda_{\text{rev}} m_d(t) \frac{f(t,z)}{m_a(t)}$ contains the term $m_a(t) = \int_\Omega f(t,z)\,\mathrm{d}z$ in the denominator. If the alive population is entirely killed ($m_a(t) = 0$), the normalization $f/m_a$ becomes undefined (indeterminate form $0/0$).

**Mathematical Obstacle**: This is a genuine singularity in the PDE. The system is not automatically well-posed on the closure of the state space $\{(f, m_d) : m_a = \int f \geq 0, m_d \geq 0, m_a + m_d = 1\}$ without additional regularization or assumptions.

**Proposed Solution**:

**Option A: Positivity Preservation** (preferred for physical fidelity)
- **Strategy**: Prove that if $m_a(0) > 0$ initially, then $m_a(t) > 0$ for all $t > 0$.
- **Mechanism**: The revival term $+\lambda_{\text{rev}} m_d(t)$ in the ODE for $m_a$ (Substep 5.6) acts as a source. If $m_a(t) \to 0$, then $m_d(t) \to 1$, so the revival term approaches its maximum rate $\lambda_{\text{rev}}$. This should prevent $m_a$ from reaching exactly zero in finite time.
- **Rigorous approach**:
  1. Consider the ODE $\frac{\mathrm{d}}{\mathrm{d}t}m_a = -k_{\text{killed}}[f] + \lambda_{\text{rev}}(1 - m_a)$ (using $m_d = 1 - m_a$).
  2. Show that $k_{\text{killed}}[f] \leq C m_a$ for some constant $C$ (using boundedness of $c(z)$ and integrability of $f$).
  3. If $\lambda_{\text{rev}} > C$, then $\frac{\mathrm{d}}{\mathrm{d}t}m_a > 0$ when $m_a$ is small, creating a barrier that prevents $m_a \to 0$.
  4. This requires the **Axiom of Guaranteed Revival** to ensure $\lambda_{\text{rev}}$ is large enough relative to the maximum killing rate.
- **Framework support**: The document mentions (remark-cemetery-state, line 144-172) that the behavior at $m_a = 0$ requires regularization assumptions. The standard choice is to assume $\lambda_{\text{rev}} > 0$ prevents the cemetery state from being reached (it's transient).
- **Status**: This is a **separate well-posedness question** beyond the scope of the assembly theorem. For the theorem as stated, we assume $m_a(t) > 0$ throughout.

**Option B: Regularization at $m_a = 0$** (mathematical convenience)
- **Strategy**: Define $B[f, m_d] \equiv 0$ when $m_a = 0$ as a limiting convention.
- **Justification**: If there are no alive walkers, there is no spatial distribution to clone from, so revival has no well-defined spatial profile. Setting $B = 0$ makes the PDE well-defined at the boundary of the state space.
- **Trade-off**: This creates an absorbing "cemetery state" where $m_a = 0, m_d = 1$ is a fixed point. The system cannot escape without an external perturbation. This contradicts the Axiom of Guaranteed Revival if $\lambda_{\text{rev}} > 0$.
- **Status**: Mentioned in remark-cemetery-state (line 144-172) as a modeling choice, but not the preferred physical interpretation.

**Alternative if Main Approach Fails**:
If positivity preservation cannot be proven with the current framework, the model may need modification:
- Change the revival mechanism to not depend on $f/m_a$ (e.g., revive uniformly over $\Omega$ instead of cloning from alive walkers).
- Add a small "safety population" that never dies, ensuring $m_a(t) \geq \epsilon > 0$ always.

**References**:
- Similar issue in McKean-Vlasov equations with conditional distributions: handled via positivity arguments or regularization (see Sznitman, "Topics in propagation of chaos").
- For Fragile framework: remark-cemetery-state (line 144-172) discusses this directly.

---

### Challenge 2: Existence and Uniqueness of Solutions to the Coupled PDE-ODE System

**Why Difficult**: The coupled system is:
1. **Nonlinear**: The revival operator $B[f, m_d] = \lambda_{\text{rev}} m_d f/m_a$ is nonlinear in $f$ through the normalization $m_a = \int f$.
2. **Non-local**: The cloning operator $S[f]$ involves integrals over $\Omega$ (fitness potential depends on global moments), and the revival operator couples to the global quantity $m_d(t)$.
3. **Degenerate**: The kinetic Fokker-Planck operator $L^\dagger$ is hypoelliptic (diffusion only in velocity, not position), which complicates regularity theory.
4. **Coupled**: The PDE for $f$ and ODE for $m_d$ are bidirectionally coupled ($f$ appears in the $m_d$ equation, $m_d$ appears in the $f$ equation).

These features place the system outside the scope of standard PDE existence theorems (e.g., classical parabolic theory).

**Mathematical Obstacle**: Proving that:
1. A solution $(f(t,z), m_d(t))$ exists for all $t \geq 0$ given initial data $(f_0, m_{d,0})$.
2. The solution is unique.
3. The solution has sufficient regularity (e.g., $f \in C([0,\infty); L^1(\Omega))$ at minimum).

**Proposed Technique**: Semigroup and Fixed-Point Approach

**Step 1: Split the problem into linear and nonlinear parts**
- Decompose: $\partial_t f = L^\dagger f + R[f, m_d]$, where $R[f, m_d] := -c(z)f + B[f, m_d] + S[f]$ is the reaction term.
- The transport operator $L^\dagger$ generates a $C_0$-semigroup on $L^1(\Omega)$ (with reflecting boundaries). This is known from kinetic Fokker-Planck theory (see Pazy, "Semigroups of Linear Operators").

**Step 2: Treat reaction term as perturbation**
- Show that $R[f, m_d]$ is locally Lipschitz in $f$ in an appropriate norm (e.g., $L^1$ or Wasserstein metric).
- Key challenge: The revival term $f/m_a$ and cloning term $S[f]$ (which involves squared integrals of $f$) require careful bounds.
- Use mass conservation ($\int f = m_a \leq 1$) and boundedness of domain to control higher moments.

**Step 3: Apply fixed-point theorem on short time intervals**
- Use Duhamel's formula (variation of constants):
  $$
  f(t) = e^{t L^\dagger} f_0 + \int_0^t e^{(t-s)L^\dagger} R[f(s), m_d(s)]\,\mathrm{d}s
  $$
- On a short time interval $[0, T]$ with $T$ small, the integral operator is a contraction in $C([0,T]; L^1(\Omega))$.
- This gives local existence and uniqueness.

**Step 4: Extend to global time via a priori estimates**
- Use mass conservation: $m_a(t) + m_d(t) = 1$ and $m_a, m_d \geq 0$ provide uniform bounds.
- Use entropy or Lyapunov functional (if available) to prevent blow-up.
- Iterate the fixed-point argument to extend the solution to $[0, \infty)$.

**Step 5: Handle the coupled ODE**
- The ODE for $m_d$ is:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}m_d = \int_\Omega c(z)f(t,z)\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)
  $$
- This is a linear ODE in $m_d$ with time-dependent coefficient $k_{\text{killed}}[f](t)$.
- If $f(t)$ is known (from the PDE), this ODE has a unique solution via standard ODE theory.
- Conversely, if $m_d(t)$ is known, the PDE for $f$ is well-posed (assuming $m_a > 0$).
- Treat as a fixed-point problem: Given $m_d^{(n)}(t)$, solve PDE for $f^{(n+1)}$, then solve ODE for $m_d^{(n+1)}$. Show the iteration converges.

**Alternative if Main Approach Fails**: Time-Splitting (Lie-Trotter)
- Approximate the solution by alternately solving:
  1. Pure transport: $\partial_t f = L^\dagger f$ for time $\Delta t$ (semigroup action).
  2. Pure reactions: $\partial_t f = R[f, m_d]$ for time $\Delta t$ (treat as ODE in function space).
- Prove convergence as $\Delta t \to 0$ using Trotter's theorem.
- This mirrors the algorithmic structure (BAOAB integrator for kinetic step, separate cloning step) and may be easier to analyze.

**References**:
- Kinetic Fokker-Planck semigroups: Pazy (1983), "Semigroups of Linear Operators"
- McKean-Vlasov well-posedness: Sznitman (1991), "Topics in propagation of chaos"
- Hypoelliptic PDEs: Hörmander (1967) or Hairer & Mattingly (2011) for kinetic systems
- Framework hint: The document states (line 80) that $f \in C([0,\infty); L^1(\Omega))$ is assumed, suggesting existence is taken as given for the purpose of this theorem.

**Status for This Proof**:
- The theorem as stated is an **assembly/derivation result**, not an existence theorem.
- Existence and uniqueness are **separate questions** that would follow in a subsequent theorem (e.g., "Theorem: Well-Posedness of the Mean-Field Equations").
- For this proof, we assume solutions exist with the stated regularity and verify that they satisfy the claimed equations.

---

### Challenge 3: Justifying the Mean-Field Limit from the N-Particle System

**Why Difficult**: This theorem presents the mean-field equations as the "limit" of the discrete Euclidean Gas algorithm. However, the proof strategy we've used is an assembly/derivation from operator definitions, not a rigorous limit $N \to \infty$ from the N-particle system.

**Mathematical Obstacle**: To rigorously derive the PDE from the algorithm, we would need to:
1. Write the master equation for the N-particle probability distribution $P_N(w_1, \ldots, w_N, k; t)$ where $w_i = (x_i, v_i, s_i)$ are walker states and $k$ is the number alive.
2. Derive the BBGKY hierarchy (marginal equations).
3. Prove **propagation of chaos**: as $N \to \infty$, the one-particle marginal $f_N^{(1)}(t,z)$ converges to $f(t,z)$ satisfying the mean-field PDE.
4. Control error terms: show $|f_N^{(1)} - f| \to 0$ in an appropriate metric.

This is a major research program in its own right, requiring:
- Stochastic analysis (for the Langevin SDE)
- Jump process theory (for cloning and death)
- Coupling methods (to compare N-particle and mean-field dynamics)
- Quantitative chaos estimates (for convergence rate)

**Proposed Technique**: Heuristic Justification (for this theorem)

**Option A: Accept operator definitions as the mean-field model** (current approach)
- The operators $L^\dagger$, $c(z)$, $B[f,m_d]$, $S[f]$ are **defined** in Section 2 as the mean-field analogues of the discrete operations.
- The theorem then states: "Given these definitions, the evolution equations are [PDE + ODE]."
- This is a **definitional/assembly theorem**, not a limit theorem.
- The burden of proving the definitions are the correct mean-field limits is placed on Section 2, where each operator was derived/motivated.

**Option B: Sketch propagation of chaos argument** (for future rigorous proof)
- In a subsequent theorem or chapter, prove that the N-particle empirical measure:
  $$
  f_N(t, z) = \frac{1}{N} \sum_{i=1}^{k(t)} \delta_{(x_i(t), v_i(t))}(z)
  $$
  converges to $f(t,z)$ solving the mean-field PDE as $N \to \infty$.
- Use techniques from:
  - Sznitman (1991) for McKean-Vlasov limits
  - Mischler & Mouhot (2013) for Kac-type collision models
  - Recent work on mean-field games with common noise (for the coupled $m_d(t)$ dynamics)

**Option C: Numerical validation** (not rigorous, but supportive)
- Simulate the N-particle system for increasing $N$ and compare to the PDE solution.
- Show empirical convergence of $f_N \to f$ and $m_{d,N} \to m_d$.
- This doesn't constitute proof but provides evidence the model is correct.

**Status for This Proof**:
- The theorem's title "The Mean-Field Equations for the Euclidean Gas" suggests this IS the mean-field limit, but the proof strategy treats it as operator assembly.
- This is **consistent with the document's structure**: Section 1 builds the mean-field vocabulary (defining how discrete sums → continuous integrals), Section 2 defines the operators, Section 3 assembles them.
- The implicit claim is: "If you accept the operator definitions from Section 2 as the correct mean-field limits (which were derived heuristically there), then these are the equations they satisfy."
- A fully rigorous treatment would require a separate theorem proving propagation of chaos, which is beyond the scope of this document.

**Alternative if Rigor is Required**:
- Rename the theorem to: "Forward Equations for the Mean-Field Model" (emphasizing this is definitional, not a limit).
- Add a separate theorem: "Propagation of Chaos for the Euclidean Gas" proving $f_N \to f$ with convergence rate.

**References**:
- Sznitman (1991), "Topics in propagation of chaos" (McKean-Vlasov limits)
- Jabin & Wang (2016), "Quantitative estimates of propagation of chaos" (with explicit rates)
- Mischler & Mouhot (2013), "Kac's program in kinetic theory" (for jump processes)

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps or stated definitions
  - Step 1: Operator assembly from continuity equation
  - Step 2: Weak formulation justification
  - Step 3: Explicit expansion of transport operator
  - Step 4: Leibniz rule setup
  - Step 5: Integration and ODE derivation
  - Step 6: Mass conservation verification

- [x] **Hypothesis Usage**: All operator definitions are used
  - $L^\dagger$ from def-kinetic-generator and def-transport-operator
  - $c(z)$ from def-killing-operator
  - $B[f, m_d]$ from def-revival-operator
  - $S[f]$ from def-cloning-generator

- [x] **Conclusion Derivation**: Both claimed equations are fully derived
  - Boxed PDE: $\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]$ (Step 1)
  - Boxed ODE: $\frac{\mathrm{d}}{\mathrm{d}t}m_d = \int c(z)f\,\mathrm{d}z - \lambda_{\text{rev}} m_d$ (Step 5)
  - Explicit form with drift-diffusion (Step 3)

- [x] **Constant Tracking**: All constants defined and bounded
  - $\lambda_{\text{rev}}$ (revival rate): free parameter, $> 0$, typical values 0.1-5
  - $c(z)$ (killing rate): $c \in C^\infty(\Omega)$, bounded
  - $A(z)$, $\mathsf{D}$ (drift, diffusion): from Langevin SDE, bounded by framework axioms

- [x] **No Circular Reasoning**: Operators are defined independently, equations are derived from them
  - The proof flows: Operator definitions → PDE/ODE system
  - Not: PDE/ODE system → operator definitions

- [x] **Framework Consistency**: All dependencies verified
  - lem-mass-conservation-transport used correctly (Step 5.2)
  - Mass-neutral properties of $S[f]$ used correctly (Step 5.5)
  - Reflecting boundaries justify vanishing boundary terms (Step 2.3)

- [⚠] **Regularity Requirements**: Addressed but require separate well-posedness theorem
  - $f \in C([0,\infty); L^1(\Omega))$ assumed (stated in def-phase-space-density)
  - $m_a(t) > 0$ required for revival operator (noted as Challenge 1)
  - Leibniz rule justified via stated regularity (Step 4.2)

- [⚠] **Edge Cases**: Boundary cases identified but not fully resolved
  - $m_a \to 0$ singularity in revival operator (Challenge 1): requires positivity preservation or regularization
  - Initial condition $m_a(0) = 0$ is technically allowed but makes revival undefined (should exclude this case)

- [x] **Mass Conservation**: Rigorously verified
  - $\frac{\mathrm{d}}{\mathrm{d}t}(m_a + m_d) = 0$ proven algebraically (Step 6)
  - Initial condition consistency checked (Step 6.3)

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Microscopic Derivation via Propagation of Chaos

**Approach**: Start with the full stochastic N-particle system. Write the generator for the N-particle Markov process (Langevin dynamics + cloning/death jumps). Derive the forward Kolmogorov equation for the N-particle distribution. Take the limit $N \to \infty$ under the "propagation of chaos" assumption (particles become independent) to obtain the mean-field PDE for the one-particle density.

**Pros**:
- Most fundamental justification: proves the PDE is the correct limit of the algorithm
- Provides quantitative error estimates: $\|f_N - f\| = O(N^{-1/2})$ or better
- Directly connects discrete algorithm to continuous model
- Standard approach in mathematical physics and mean-field game theory

**Cons**:
- Extremely technically demanding: requires advanced stochastic calculus, coupling methods, Wasserstein metrics
- The cloning jump process is non-standard (depends on fitness potential, which is a non-local functional of the empirical measure)
- Proving chaos for jump processes with mean-field interaction is still an active research area
- Would require an entire chapter or separate paper (not suitable for a single theorem)
- The framework has already defined the operators heuristically in Section 2, so this would be redundant

**When to Consider**:
- If the goal is a fully rigorous foundation for the mean-field model from first principles
- If quantitative error bounds are needed (e.g., for numerical validation)
- As a separate theorem after this one: "Theorem: The N-particle Euclidean Gas converges to the mean-field equations with rate O(N^{-1/2})"

---

### Alternative 2: Variational Formulation as Gradient Flow

**Approach**: Attempt to formulate the evolution as a gradient flow on the space of probability measures $\mathcal{P}(\Omega)$ equipped with the Wasserstein metric. The transport operator $L^\dagger f$ is known to be the gradient flow of the free energy functional (for Langevin dynamics). Try to incorporate the killing, revival, and cloning terms as perturbations or additional potential gradients.

**Pros**:
- Provides deep geometric insight into the solution space (Wasserstein geometry)
- Gradient flow structure implies strong stability properties (contraction, entropy dissipation)
- Can leverage powerful tools: Otto calculus, JKO scheme, displacement convexity
- Natural for proving long-term convergence properties (e.g., exponential approach to equilibrium)

**Cons**:
- The killing and revival terms break the pure gradient flow structure (mass is exchanged with the external reservoir $m_d$)
- The cloning operator $S[f]$ is highly non-local and nonlinear, unclear if it's a potential gradient
- The coupled PDE-ODE system doesn't fit the standard gradient flow framework (which typically operates on a single measure space)
- May not be possible to cast the full system in this form

**Partial Success**:
- The kinetic transport $L^\dagger f$ IS a gradient flow (of the kinetic Fokker-Planck entropy)
- This could be exploited in a **hybrid approach**: Treat $L^\dagger$ as gradient flow, treat $R[f,m_d]$ as perturbation, use stability of gradient flows to control the full system

**When to Consider**:
- When studying long-term behavior and convergence to stationary states
- When proving entropy dissipation or Lyapunov function properties
- As a tool for the well-posedness proof (gradient flow structure often gives compactness)
- For proving uniqueness of the stationary distribution

---

### Alternative 3: Operator Splitting via Lie-Trotter Formula

**Approach**: Instead of deriving the PDE directly, split the evolution into a composition of semigroups:
1. $\Phi_{\text{transport}}(t)$: semigroup generated by $L^\dagger$ (kinetic Fokker-Planck)
2. $\Phi_{\text{killing}}(t)$: semigroup generated by $-c(z) \cdot$ (exponential decay)
3. $\Phi_{\text{revival}}(t)$: semigroup generated by $B[f, m_d]$ (coupled with ODE for $m_d$)
4. $\Phi_{\text{cloning}}(t)$: semigroup generated by $S[f]$ (nonlinear redistribution)

Compose them via Lie-Trotter: $\Phi(t) = \lim_{n \to \infty} \left[\Phi_{\text{transport}}(t/n) \circ \Phi_{\text{killing}}(t/n) \circ \Phi_{\text{revival}}(t/n) \circ \Phi_{\text{cloning}}(t/n)\right]^n$. Show the generator of $\Phi(t)$ is $L^\dagger - c(z) + B[f,m_d] + S[f]$.

**Pros**:
- Directly mirrors the algorithmic structure (BAOAB split for kinetic, separate cloning step)
- Each substep may be easier to analyze individually
- Provides a constructive numerical scheme (operator splitting is a standard PDE solver)
- Trotter's theorem rigorously justifies that the split composition converges to the full PDE

**Cons**:
- Requires proving each operator generates a strongly continuous semigroup (or semiflow for nonlinear $S[f]$)
- The cloning operator is nonlinear and non-local, so it's not a standard semigroup generator
- Operator splitting introduces splitting error, need to control it
- More machinery needed than the direct derivation approach

**When to Consider**:
- When implementing numerical solvers (splitting is often more stable than directly discretizing the full PDE)
- When the full system is too complex but each piece is tractable
- As an alternative proof strategy if direct assembly fails

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Well-Posedness of the Coupled System**
   - **Gap**: Existence and uniqueness of solutions for all $t \geq 0$ is not proven in this theorem
   - **How critical**: HIGH - without this, the equations are formal and may not have solutions
   - **Suggested approach**: Use semigroup theory + fixed-point methods as outlined in Challenge 2. Prove local existence, then extend globally via a priori estimates and mass conservation.
   - **Estimated difficulty**: Medium to High (standard PDE techniques but applied to non-standard coupled system)

2. **Positivity Preservation: $m_a(t) > 0$**
   - **Gap**: The revival operator requires $m_a(t) > 0$, but this is not proven from $m_a(0) > 0$
   - **How critical**: HIGH - the PDE is singular at $m_a = 0$
   - **Suggested approach**: Prove a comparison lemma showing revival dominates killing when $m_a$ is small. Use the Axiom of Guaranteed Revival to bound $\lambda_{\text{rev}} / \sup c(z)$ appropriately.
   - **Estimated difficulty**: Medium (requires careful ODE analysis and use of framework axioms)

3. **Propagation of Chaos: $f_N \to f$ as $N \to \infty$**
   - **Gap**: The mean-field PDE is presented as "the limit" of the N-particle system, but this limit is not rigorously proven
   - **How critical**: MEDIUM - important for justifying the model, but operators are already defined heuristically
   - **Suggested approach**: Use coupling methods and Wasserstein metric. Prove $\mathbb{E}[W_1(f_N, f)] \leq C/\sqrt{N}$ with explicit constants.
   - **Estimated difficulty**: High (active research area, especially for jump processes with mean-field interaction)

### Conjectures

1. **Exponential Positivity Preservation**
   - **Conjecture**: If $m_a(0) > 0$ and $\lambda_{\text{rev}} > C_{\max} := \sup_z c(z)$, then $m_a(t) \geq m_a(0) e^{-(\lambda_{\text{rev}} - C_{\max})t}$ for all $t \geq 0$.
   - **Why plausible**: The ODE $\frac{\mathrm{d}}{\mathrm{d}t}m_a \geq -C_{\max} m_a + \lambda_{\text{rev}}(1 - m_a)$ provides a lower bound. If $\lambda_{\text{rev}} > C_{\max}$, the revival dominates and $m_a$ is bounded away from zero.
   - **How to test**: Prove $k_{\text{killed}}[f] \leq C_{\max} m_a$ using $c(z) \leq C_{\max}$ and $\int f = m_a$. Then apply Grönwall's inequality to the ODE for $m_a$.

2. **Unique Stationary State**
   - **Conjecture**: The coupled PDE-ODE system has a unique stationary state $(f_\infty, m_{d,\infty})$ satisfying $\partial_t f = 0$ and $\frac{\mathrm{d}}{\mathrm{d}t}m_d = 0$.
   - **Why plausible**: At stationarity, the killing and revival rates balance: $\int c(z)f_\infty\,\mathrm{d}z = \lambda_{\text{rev}} m_{d,\infty}$. The kinetic Fokker-Planck $L^\dagger f_\infty$ should balance the reaction terms. If the kinetic operator has a unique invariant measure (Gibbs distribution), the full system's stationary state may be unique.
   - **How to test**: Look for $f_\infty$ of the form $f_\infty(z) \propto e^{-H(z)}$ where $H$ is an effective potential. Check if the balance equation uniquely determines $H$ and $m_{d,\infty}$.

### Extensions

1. **Adaptive Gas Mean-Field Equations**
   - **Extension**: Derive the mean-field PDE for the Adaptive Gas (Chapter 2 of framework), which includes additional terms: adaptive force from mean-field fitness, viscous coupling, and regularized Hessian diffusion.
   - **Challenge**: These mechanisms introduce additional nonlinearities and non-localities (Hessian of fitness potential, pairwise interactions).
   - **Expected result**: A more complex McKean-Vlasov PDE with third-order terms and additional coupling to global fitness moments.

2. **Convergence Rate with Explicit Constants**
   - **Extension**: Not just prove $f_N \to f$, but bound $\|f_N - f\|_{W_1} \leq C(T) N^{-\alpha}$ with explicit $C(T)$ and $\alpha$.
   - **Motivation**: Needed to validate simulations and choose appropriate $N$ for numerical experiments.
   - **Technique**: Combine Wasserstein stability of the mean-field PDE with careful tracking of constants in the chaos argument.

3. **Boundary Layer Analysis for Killing Rate $c(z)$**
   - **Extension**: Study the structure of $f$ near the boundary $\partial X_{\text{valid}}$ where $c(z) > 0$. Is there a boundary layer where $f$ decays exponentially?
   - **Motivation**: Understanding the spatial profile of $f$ near death regions provides insight into how the algorithm balances exploration vs. risk.
   - **Technique**: Singular perturbation analysis, treating the killing zone as a thin layer of width $\epsilon$. Match inner (boundary layer) and outer (bulk) solutions.

---

## IX. Expansion Roadmap

This proof sketch provides a complete high-level strategy for proving the theorem. To expand to a full Annals of Mathematics-level proof, the following phases are recommended:

### Phase 1: Formalize Operator Assembly (Estimated: 1-2 weeks)

**Tasks**:
1. **Continuity equation derivation**: Write out the full derivation of $\partial_t f = -\nabla \cdot J + Q_{\text{net}}$ from first principles (conservation of probability in infinitesimal volumes). Include precise regularity assumptions.
   - **Difficulty**: Easy (standard PDE theory)
   - **Output**: 2-3 pages, rigorous continuum mechanics argument

2. **Mean-field independence justification**: Formalize the claim that operators act additively in the mean-field limit (linearity of $Q_{\text{net}}$). Cite standard references on mean-field approximations or prove via $o(\mathrm{d}t)$ argument for simultaneous events.
   - **Difficulty**: Easy to Medium (depends on desired rigor)
   - **Output**: 1-2 pages, probabilistic argument or citation

3. **Weak formulation with test functions**: Expand Step 2 into a full subsection. Define test function spaces precisely ($C_c^\infty(\Omega)$ or $H^1(\Omega)$). Write out integration by parts in detail. State trace theorems used for boundary terms.
   - **Difficulty**: Medium (functional analysis)
   - **Output**: 3-4 pages, rigorous weak formulation

### Phase 2: Detailed Integration and ODE Derivation (Estimated: 1 week)

**Tasks**:
1. **Leibniz rule justification**: Prove rigorously that $\frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega f = \int_\Omega \partial_t f$ for solutions in $C([0,\infty); L^1(\Omega))$. Or, use test functions to define a weak ODE for $m_a(t)$.
   - **Difficulty**: Easy to Medium (depends on whether weak or classical)
   - **Output**: 1-2 pages, analysis of regularity

2. **Integral calculations for each operator**: Expand Substeps 5.2-5.5 with full details. For the revival integral (Substep 5.4), explicitly write out each step of the normalization calculation. For the cloning integral (Substep 5.5), reference the mass-neutrality proof in def-cloning-generator and restate it if needed.
   - **Difficulty**: Easy (mostly algebra)
   - **Output**: 2-3 pages, detailed calculations

3. **Coupling between PDE and ODE**: Discuss the structure of the coupled system. Explain why $m_d$ appears in the PDE (revival term) and $f$ appears in the ODE (killing term). Comment on the feedback loop and why it preserves total mass.
   - **Difficulty**: Easy (conceptual discussion)
   - **Output**: 1 page, expository text

### Phase 3: Rigor for Boundary Conditions and Well-Posedness (Estimated: 2-3 weeks)

**Tasks**:
1. **Reflecting boundary conditions for kinetic Fokker-Planck**: Provide full mathematical definition of reflecting boundaries on $\partial\Omega$. State the precise form of the boundary condition (e.g., $J \cdot n = 0$ in the trace sense). Cite literature on kinetic PDEs with reflection (e.g., Desvillettes & Villani for Boltzmann, or Villani's Hypocoercivity notes).
   - **Difficulty**: Medium to High (requires kinetic PDE theory)
   - **Output**: 4-5 pages, boundary condition formulation

2. **Positivity preservation proof (Challenge 1)**: Prove that $m_a(t) > 0$ for all $t$ given $m_a(0) > 0$. Use the ODE for $m_a$ and bound $k_{\text{killed}}[f]$ by $C_{\max} m_a$. Apply comparison theorem or Grönwall's inequality. Invoke Axiom of Guaranteed Revival to ensure $\lambda_{\text{rev}}$ is large enough.
   - **Difficulty**: Medium (ODE analysis + framework axioms)
   - **Output**: 3-4 pages, lemma with proof

3. **Existence and uniqueness theorem (Challenge 2 - optional for this theorem)**: This could be a separate theorem, but if included here: Use semigroup theory for $L^\dagger$ (cite Pazy). Prove $R[f, m_d]$ is locally Lipschitz. Apply fixed-point theorem (Banach or Schauder). Extend to global time via mass conservation bounds.
   - **Difficulty**: High (advanced PDE theory)
   - **Output**: 8-10 pages, major theorem with proof
   - **Recommendation**: Make this a separate theorem after this one

### Phase 4: Verification and Consistency Checks (Estimated: 1 week)

**Tasks**:
1. **Mass conservation verification (Step 6)**: Expand into a formal lemma. State clearly: "Lemma: If $(f, m_d)$ solves the coupled system, then $m_a(t) + m_d(t) = \text{constant}$." Prove by the algebraic cancellation shown in Step 6. Include initial condition check.
   - **Difficulty**: Easy (already done in sketch)
   - **Output**: 2 pages, lemma with proof

2. **Initial condition consistency**: Verify that the initial conditions $(f_0, m_{d,0})$ with $m_{d,0} = 1 - \int f_0$ are well-posed. Check that $f_0 \in L^1(\Omega)$ with $\int f_0 \leq 1$ is sufficient. Discuss the case $\int f_0 = 0$ (all dead initially) and whether it's allowed (probably should be excluded if revival requires $m_a > 0$).
   - **Difficulty**: Easy (mostly discussion)
   - **Output**: 1 page, remark or lemma

3. **Framework dependency cross-check**: Create a table listing every definition, lemma, and axiom used in the proof. For each, state where it's defined, how it's used, and verify all preconditions are met. This ensures no circular reasoning or missing dependencies.
   - **Difficulty**: Easy (bookkeeping)
   - **Output**: 1-2 pages, dependency table

### Phase 5: Polish and Review (Estimated: 1 week)

**Tasks**:
1. **Write introduction and motivation**: Add a preamble explaining why the mean-field equations are important (connection to algorithm, analytical tractability, foundation for convergence proofs).
   - **Output**: 1-2 pages

2. **Add pedagogical remarks**: Throughout the proof, include admonitions explaining physical interpretation (e.g., "The killing term $-c(z)f$ represents walkers being removed near the boundary..."). Help the reader build intuition.
   - **Output**: Scattered throughout, ~2-3 pages total

3. **Dual review with Gemini and GPT-5**: Submit the expanded proof for review (following CLAUDE.md protocol). Address feedback iteratively.
   - **Output**: Revised proof based on reviewer suggestions

4. **Final formatting**: Ensure all LaTeX is correct, cross-references work, and the document compiles cleanly in Jupyter Book.
   - **Output**: Publication-ready document

### Total Estimated Expansion Time: 6-9 weeks

**Breakdown**:
- Phase 1 (Operator Assembly): 1-2 weeks
- Phase 2 (Integration/ODE): 1 week
- Phase 3 (Boundary/Well-Posedness): 2-3 weeks
  - Note: If existence/uniqueness is deferred to a separate theorem, reduce to 1-2 weeks
- Phase 4 (Verification): 1 week
- Phase 5 (Polish/Review): 1 week

**Critical Path**: Phase 3 (boundary conditions and positivity preservation) is the most technically demanding. If well-posedness is included, add 2-3 weeks. Otherwise, the proof can be completed in ~4-6 weeks of focused work.

**Parallelization**: Phases 1 and 2 can be done by one person, while another works on Phase 3 (boundary conditions) simultaneously, reducing total time to ~4-5 weeks with two researchers.

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`lem-mass-conservation-transport` (line 572, same document)

**Definitions Used**:
- {prf:ref}`def-mean-field-phase-space` (line 39, same document)
- {prf:ref}`def-phase-space-density` (line 61, same document)
- {prf:ref}`def-kinetic-generator` (line 311, same document)
- {prf:ref}`def-transport-operator` (line 554, same document)
- {prf:ref}`def-killing-operator` (line 360, same document)
- {prf:ref}`def-revival-operator` (line 378, same document)
- {prf:ref}`def-cloning-generator` (line 497, same document)

**Axioms Used**:
- Axiom of Guaranteed Revival (from earlier chapters): Ensures $\lambda_{\text{rev}} > 0$ prevents permanent extinction
- Axiom of Bounded Displacement (from earlier chapters): Ensures Lipschitz continuity of kinetic dynamics
- Total mass conservation principle: $m_a(t) + m_d(t) = 1$ (fundamental framework assumption)

**Related Proofs** (for comparison):
- Proof of lem-mass-conservation-transport (line 582-597): Uses divergence theorem with reflecting boundaries
- Derivation of cloning operator mass-neutrality (line 526-539): Explicit calculation showing source and sink cancel
- BAOAB integrator definition (line 255-291): Discrete-time integrator that motivates the kinetic transport operator

**Related Theorems** (in same document):
- {prf:ref}`thm-mass-conservation` (line 655): States the consequences of the coupled system (alive mass dynamics, equilibrium conditions)

---

**Proof Sketch Completed**: 2025-11-06
**Ready for Expansion**: Yes - all steps are actionable and framework dependencies are verified
**Confidence Level**: High

**Justification for High Confidence**:
1. ✅ **Both strategists agree** on the core approach (operator assembly via continuity equation)
2. ✅ **All framework dependencies verified** in the same document (no forward references)
3. ✅ **No contradictions between Gemini and GPT-5** strategies (complementary, not conflicting)
4. ✅ **Critical challenges identified and addressed** (positivity at $m_a=0$, well-posedness as separate question)
5. ✅ **Mass conservation verified algebraically** (serves as consistency check on derivation)
6. ✅ **Clear scope**: This is an assembly/derivation theorem, not an existence theorem (both strategists recognize this)

**Caveats**:
- ⚠ Positivity preservation ($m_a > 0$) is assumed, not proven in this theorem
- ⚠ Existence and uniqueness of solutions is not part of this theorem (separate well-posedness question)
- ⚠ Rigorous mean-field limit ($N \to \infty$) is not proven (operators are defined heuristically in Section 2)

These caveats are **not flaws** in the proof strategy but rather clear delineations of scope. The theorem accomplishes what it sets out to do: assemble the mean-field equations from pre-defined operators and verify mass conservation. The additional questions (positivity, well-posedness, propagation of chaos) are important but belong in subsequent theorems.

---

**Next Steps for User**:
1. Review this proof sketch
2. If satisfied with the strategy, proceed to Phase 1 of the expansion roadmap
3. If changes needed, specify which steps require modification
4. Consider whether positivity preservation should be proven as a lemma before this theorem, or as a separate theorem after it
5. Decide whether existence/uniqueness should be included here or deferred to a dedicated well-posedness theorem
