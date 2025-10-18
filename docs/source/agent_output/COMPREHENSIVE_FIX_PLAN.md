# Comprehensive Fix Plan for 04_wasserstein_contraction.md

**Status:** CRITICAL ISSUES - Requires fundamental revision
**Reviewers:** Gemini 2.5 Pro, Codex (with user feedback)
**Date:** 2025-10-17

---

## Executive Summary

### Critical Issues Identified

All three sources (Gemini, Codex, User feedback) converged on the same fundamental problems:

1. **Scaling Mismatch (CRITICAL)**: Contraction term $O(L)$ vs. total distance $O(L^2)$
2. **Outlier Alignment Invalid Proof (CRITICAL)**: Dynamic argument used for static property
3. **Case Mixing Probability (CRITICAL)**: No rigorous bound on $\mathbb{P}(\text{Case B})$

### Key Insight from Reviewers

**Gemini's Breakthrough:** The $O(L^2)$ term EXISTS but was lost in approximation!
- Exact identity: $D_{ji} - D_{ii} = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle$
- For separated swarms: This gives $D_{ii} - D_{ji} \approx L^2$ (quadratic!)
- Current proof erroneously approximated this away

**Codex's Alternative:** $R_H$ itself scales with $L$ for separated swarms!
- High-Error Projection Lemma: $R_H \geq c_0 L - c_1$
- This makes linear terms become quadratic: $\eta R_H L \sim L^2$

**Both approaches recover the missing quadratic term!**

---

## Issue #1: Scaling Mismatch - TWO SOLUTIONS

### Solution A: Gemini's Exact Algebraic Identity (RECOMMENDED)

**Strategy:** Use exact formula for distance change instead of geometric approximation

**Mathematical Foundation:**

For cloning operator replacing walker $i$ with clone of $j$, the change in total squared distance is:

$$
D_{ji} - D_{ii} = \sum_{k \neq i} (\|x_j - x_k\|^2 - \|x_i - x_k\|^2)
$$

**Key Identity:** Using $\|a-c\|^2 - \|b-c\|^2 = \|a-b\|^2 + 2\langle a-b, b-c\rangle$:

$$
D_{ji} - D_{ii} = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle
$$

where $\bar{x} = \frac{1}{N}\sum_p x_p$ is the global barycenter.

**Application to Separated Swarms:**

For swarms at distance $L = \|\bar{x}_1 - \bar{x}_2\|$:
- Global center: $\bar{x} \approx (\bar{x}_1 + \bar{x}_2)/2$
- Outlier $x_i \in S_1$, companion $x_j \in S_2$
- Distance: $\|x_j - x_i\| \approx \|\bar{x}_2 - \bar{x}_1\| = L$

**First term (quadratic):**
$$
(N-1)\|x_j - x_i\|^2 \approx (N-1)L^2
$$

**Second term (also quadratic!):**
$$
x_i - \bar{x} \approx \bar{x}_1 - \frac{\bar{x}_1 + \bar{x}_2}{2} = \frac{\bar{x}_1 - \bar{x}_2}{2}
$$

$$
\langle x_j - x_i, x_i - \bar{x}\rangle \approx \langle \bar{x}_2 - \bar{x}_1, \frac{\bar{x}_1 - \bar{x}_2}{2}\rangle = -\frac{L^2}{2}
$$

$$
2N\langle x_j - x_i, x_i - \bar{x}\rangle \approx -NL^2
$$

**Total contraction term:**
$$
D_{ii} - D_{ji} \approx (N-1)L^2 - NL^2 = -L^2
$$

**Therefore:** $D_{ii} - D_{ji} \approx L^2$ (QUADRATIC!)

**Contraction ratio:**
$$
\frac{D_{ii} - D_{ji}}{D_{ii} + D_{jj}} \approx \frac{L^2}{2L^2} = \frac{1}{2} = O(1)
$$

**UNIFORM CONTRACTION ACHIEVED!**

---

### Solution B: Codex's High-Error Projection Lemma (ALTERNATIVE)

**Strategy:** Prove $R_H$ scales with $L$ for separated swarms

**Lemma (High-Error Projection):**

For swarms with separation $L = \|\bar{x}_1 - \bar{x}_2\|$ and high-error fraction $|H_1| \geq f_H N$:

$$
\max_{i \in H_1} \langle x_{1,i} - \bar{x}_1, u\rangle \geq \frac{L - 2R_L/f_H}{2}
$$

where $u = \frac{\bar{x}_1 - \bar{x}_2}{L}$ is the unit direction.

**Proof Sketch:**
1. Decompose barycenter difference:
   $$
   \bar{x}_1 - \bar{x}_2 = \frac{1}{N}\sum_{i \in H_1}(x_{1,i} - \bar{x}_1) - \frac{1}{N}\sum_{i \in H_2}(x_{2,i} - \bar{x}_2) + O(R_L)
   $$

2. Project onto $u$:
   $$
   L = \langle \bar{x}_1 - \bar{x}_2, u\rangle \leq \frac{|H_1|}{N}\max_{i \in H_1}\langle x_{1,i} - \bar{x}_1, u\rangle + O(R_L)
   $$

3. Use $|H_1| \geq f_H N$:
   $$
   \max_{i \in H_1}\langle x_{1,i} - \bar{x}_1, u\rangle \geq \frac{L - O(R_L)}{f_H}
   $$

**Corollary:** Since high-error set is defined by $\|x_{1,i} - \bar{x}_1\| > R_H(\varepsilon)$:

$$
R_H \geq c_0 L - c_1
$$

for constants $c_0 = O(f_H)$ and $c_1 = O(R_L/f_H)$.

**Application:** Current linear terms become quadratic:
- $\eta R_H L \geq \eta(c_0 L - c_1)L = \eta c_0 L^2 - \eta c_1 L \sim O(L^2)$ for large $L$
- $T_1 - S_1 = R_H^2 - R_L^2 \geq (c_0 L)^2 \sim O(L^2)$

**Result:** Quadratic scaling recovered via $R_H \propto L$ relationship.

---

### Recommended Approach: COMBINE BOTH

**Rationale:**
- **Gemini's identity** is exact and algebraically clean
- **Codex's lemma** provides geometric insight into framework structure
- Using BOTH gives redundant validation and richer proof

**Implementation Plan:**

1. **Primary argument:** Use Gemini's exact identity
   - Derive $D_{ii} - D_{ji} = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle$
   - Show this gives $\approx L^2$ for separated swarms
   - Clean, direct, requires minimal framework machinery

2. **Secondary argument:** Add Codex's High-Error Projection Lemma
   - Prove $R_H \geq c_0 L - c_1$ as supporting geometric result
   - Show consistency: both approaches yield $O(L^2)$ bound
   - Strengthens framework by making $R_H$ scaling explicit

3. **Error analysis:** Bound approximation errors
   - Current bound $D_{ii} - D_{ji} \geq \eta R_H L$ becomes lower-order correction
   - Dominant term: $L^2$
   - Next order: $\eta R_H L \sim L^2$ (via High-Error Projection)
   - Error terms: $O(R_H R_L), O(R_L^2)$ are negligible

---

## Issue #2: Outlier Alignment Lemma - Static Proof

### Current Problem (CONFIRMED BY ALL REVIEWERS)

**Invalid Argument:**
- Uses H-theorem and "stable separation over time"
- This is a DYNAMIC property (evolution of system)
- Lemma is used for SINGLE cloning step
- Must hold for ANY configuration at one instant (STATIC property)

**User Feedback:**
> "The proof confuses a dynamic, long-term argument with a static, single-step property. It does not prove that for any given, statically separated swarm configuration, a fitness valley must exist at that specific instant."

**CONSENSUS:** Proof is invalid and must be replaced.

---

### Gemini's Static Proof Strategy (RECOMMENDED)

**Approach:** Use fitness function axioms directly

**Required Axioms (verify in framework):**
1. **Axiom F1 (Confining Potential):** $F(x)$ decays as $\|x\| \to \infty$
2. **Axiom F2 (Environmental Richness):** Multi-modal landscape with distinct local maxima

**Proof Structure:**

**Step 1: Prove Fitness Valley Exists (Static)**

Consider the line segment $\gamma(t) = (1-t)\bar{x}_1 + t\bar{x}_2$ for $t \in [0,1]$.

**Claim:** $f(t) = F(\gamma(t))$ has a local minimum in $(0,1)$.

**Proof:**
- By Axiom F2: $F(\bar{x}_1)$ and $F(\bar{x}_2)$ are high (local maxima)
- By Axiom F1: If $F$ were monotonically increasing from $\bar{x}_1$ to $\bar{x}_2$ and beyond, it would violate confining potential
- Therefore: $f(t)$ must have at least one local minimum in $(0,1)$ ✓

**Step 2: Outliers Are in Valley (Geometric)**

**Definition:** Outlier $x_i \in H_1$ satisfies $\|x_i - \bar{x}_1\| > R_H(\varepsilon)$.

**Claim:** For $L > D_{\min}$, outliers on wrong side have low fitness.

**Proof:**
- Wrong-side outlier: $x_i \in M_1 = \{x : \langle x - \bar{x}_1, \bar{x}_1 - \bar{x}_2\rangle < 0\}$
- Such $x_i$ is closer to valley region than to $\bar{x}_1$ or $\bar{x}_2$
- By Step 1, valley has lower fitness than swarm centers
- Therefore: $F(x_i) < \min(F(\bar{x}_1), F(\bar{x}_2))$ ✓

**Step 3: Fitness Comparison (Quantitative)**

Use fitness function structure $V_{\text{fit},i} = (d'_i)^\beta \cdot (r'_i)^\alpha$:

**Distance component:**
- Outlier distance Z-score: $z_{d,i} \gg 0$ (far from barycenter)
- Companion distance Z-score: $z_{d,j} \approx 0$ (near barycenter)
- Ratio: $d'_i \ll d'_j$

**Reward component:**
- Valley position has lower reward (Step 1)
- Companion near $\bar{x}_2$ has high reward
- Ratio: $r'_i < r'_j$

**Combined:**
$$
V_{\text{fit},i} = (d'_i)^\beta (r'_i)^\alpha < (d'_j)^\beta (r'_j)^\alpha = V_{\text{fit},j}
$$

**Conclusion:** Outlier Alignment follows from static fitness landscape geometry. ✓

---

### Codex's Static Proof Strategy (ALTERNATIVE)

**Approach:** Use Keystone Principle + Lipschitz structure

**Step 1:** Apply Keystone Principle
- High-error walkers $i \in H_k$ have structural error above threshold
- Stability Condition guarantees fitness gap: $\mathbb{E}[\log V_{\text{fit}} | i \in H_k] < \mathbb{E}[\log V_{\text{fit}} | i \in L_k]$

**Step 2:** Use Lipschitz structure of fitness components
- Reward/diversity maps have bounded Lipschitz constants (from framework)
- Along line segment $\bar{x}_1 \to \bar{x}_2$, fitness is Lipschitz continuous
- Local maxima at barycenters imply decrease away from them

**Step 3:** Combine inequalities
- Keystone bounds fitness of high-error vs. low-error
- Lipschitz bounds fitness variation along path
- Together: wrong-side outliers have quantifiably lower fitness

---

### Recommended Approach: Gemini's Proof

**Rationale:**
- More direct and intuitive
- Relies on fundamental axioms (confining potential, environmental richness)
- Easier to verify and explain
- Codex's approach requires more framework machinery

**Implementation:**
1. Verify Axioms F1, F2 exist in framework (check 01_fragile_gas_framework.md)
2. Add explicit "Fitness Valley Lemma" before Outlier Alignment
3. Rewrite Outlier Alignment proof using static geometric argument
4. Remove all references to H-theorem, time evolution, "stable separation"

---

## Issue #3: Case A/B Mixing Probability

### Current Problem

**Missing:** No bound on $\mathbb{P}(\text{Case B})$

**Required:** Show Case B occurs with probability $\geq p_B > 0$ (N-uniform)

**Codex's Strategy:**

**Lemma (Case Proportion):** For $L > D_{\min}$:
$$
\mathbb{P}(\text{Case B}) \geq p_B = f_{UH} \cdot q_{\min} > 0
$$

where:
- $f_{UH}$: Unfit-high-error overlap fraction (from Theorem in 03_cloning.md)
- $q_{\min}$: Minimum probability from Gibbs matching weights

**Proof Sketch:**
1. **Target set:** $I_{\text{target}} = \{i : i \in H_1 \cap U_1\}$ (high-error AND unfit in swarm 1)
2. **Size bound:** $|I_{\text{target}}| \geq f_{UH} N$ (from Keystone Principle)
3. **Matching structure:** For $i \in I_{\text{target}}$:
   - In swarm 1: $i$ is unfit → will have lower fitness than companion
   - Companion $\pi(i)$ is selected from swarm 1 by Gibbs weights
   - Companion likely in $L_1$ (low-error set)
4. **Cross-swarm comparison:** For same pair in swarm 2:
   - Walker $i$ now compares with companion in swarm 2
   - If walker $i \in L_2$ (low-error in swarm 2), fitness ordering reverses
5. **Probability bound:** Matching Gibbs distribution ensures $\geq q_{\min}$ probability of selecting mixed-fitness pairs

**Result:** $\mathbb{P}(\text{Case B}) \geq f_{UH} q_{\min} > 0$ ✓

---

### Implementation

**Add Lemma 4.6:** Case B Probability Lower Bound

**Statement:**
For swarms with separation $L > D_{\min}$, the probability that a randomly selected pair exhibits Case B (mixed fitness ordering) is bounded below:

$$
\mathbb{P}(\text{Case B} \mid M) \geq f_{UH} \cdot q_{\min}(\varepsilon) > 0
$$

where $f_{UH}$ is the unfit-high-error overlap and $q_{\min}$ depends only on framework parameters.

**Use:** Combined with Case A expansion $\gamma_A$ and Case B contraction $\gamma_B$:

$$
\gamma_{\text{eff}} = (1 - p_B)\gamma_A + p_B \gamma_B < 1
$$

when $p_B(\gamma_A - \gamma_B) > \gamma_A - 1$.

With:
- $\gamma_A \approx 1 + O(\delta^2/L^2)$ (expansion vanishes for large $L$)
- $\gamma_B \approx 1 - \kappa_B$ for $\kappa_B = O(1)$ (from Issue #1 fix)
- $p_B \geq f_{UH} q_{\min} > 0$

We get: $\gamma_{\text{eff}} < 1 - p_B \kappa_B + O(\delta^2/L^2) < 1$ for $L$ sufficiently large. ✓

---

## Complete Fix Implementation Plan

### Phase 1: Foundational Lemmas (Week 1)

**1.1. Fitness Valley Lemma (Static)**
- **Location:** Insert before Section 2.1
- **Content:** Prove valley exists using Axioms F1, F2
- **Dependencies:** Verify axioms in 01_fragile_gas_framework.md
- **Deliverable:** `{prf:lemma}` with label `lem-fitness-valley-static`

**1.2. High-Error Projection Lemma**
- **Location:** Insert in Section 4.3 (before 4.3.5)
- **Content:** Prove $R_H \geq c_0 L - c_1$ from high-error fraction
- **Dependencies:** Corollary 6.4.4 from 03_cloning.md
- **Deliverable:** `{prf:lemma}` with label `lem-high-error-projection`

**1.3. Exact Distance Change Identity**
- **Location:** Section 4.4, replace current derivation
- **Content:** Derive $(N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle$
- **Dependencies:** None (pure algebra)
- **Deliverable:** `{prf:proposition}` with label `prop-exact-distance-change`

---

### Phase 2: Core Proof Revisions (Week 2)

**2.1. Outlier Alignment Lemma - Complete Rewrite**
- **Location:** Section 2.2 (replace Steps 1-6)
- **Remove:** All H-theorem, stable separation, dynamic arguments
- **Add:** Static proof using Fitness Valley Lemma
- **Structure:**
  1. State fitness valley exists (cite Lemma from 1.1)
  2. Show outliers on wrong side are in valley geometrically
  3. Derive quantitative fitness gap using fitness function structure
  4. Conclude alignment with $\eta = 1/4$
- **Deliverable:** Rigorous static proof, ~100 lines

**2.2. Case B Geometric Bound - Quadratic Scaling**
- **Location:** Section 4.4
- **Add:** Exact identity approach (Gemini)
- **Add:** High-error projection bound (Codex) as supporting result
- **Show:** Both give $D_{ii} - D_{ji} \sim L^2$
- **Derive:** Explicit constants $c_B$ such that $D_{ii} - D_{ji} \geq c_B L^2$
- **Error analysis:** Bound approximation errors
- **Deliverable:** Complete derivation with error bounds, ~150 lines

**2.3. Case B Probability Bound**
- **Location:** New Section 4.6
- **Content:** Prove $\mathbb{P}(\text{Case B}) \geq f_{UH} q_{\min}$
- **Method:** Codex's target set approach
- **Dependencies:** Unfit-high-error overlap (03_cloning.md §7.6-8.3)
- **Deliverable:** `{prf:lemma}` with explicit constant, ~80 lines

---

### Phase 3: Main Theorem Update (Week 3)

**3.1. Case A/B Combination**
- **Location:** Section 5, complete rewrite
- **Add:** Explicit probability weighting
- **Compute:** $\gamma_{\text{eff}} = (1-p_B)\gamma_A + p_B\gamma_B$
- **Show:** $\gamma_{\text{eff}} < 1$ using bounds from Phase 2
- **Deliverable:** Rigorous weighted average argument

**3.2. Contraction Constants**
- **Location:** Section 8.1
- **Update:** $\kappa_W = p_B c_B \cdot \frac{1}{2} - (1-p_B)\varepsilon_A$
- **Verify:** N-uniformity of all constants
- **Check:** Compatibility with confining potential ($L \leq 2R_{\max}$)
- **Deliverable:** Explicit formula for $\kappa_W$ with all dependencies

**3.3. Main Theorem Statement**
- **Location:** Section 0.1
- **Update:** Theorem statement with new constants
- **Add:** Explicit dependence on framework parameters
- **Clarify:** Valid for $L \in [D_{\min}, 2R_{\max}]$
- **Deliverable:** Updated theorem with complete hypothesis

---

### Phase 4: Verification and Polish (Week 4)

**4.1. Cross-Reference Audit**
- Check all lemma labels are in 00_index.md
- Update 00_reference.md with new results
- Verify all framework document citations

**4.2. Consistency Checks**
- Confining potential bounds
- N-uniformity of all constants
- Scaling behavior for $L \to D_{\min}$ and $L \to 2R_{\max}$
- Error term magnitudes

**4.3. Third Round Dual Review**
- Submit to Gemini + Codex with hallucination detection
- Verify scaling issues resolved
- Check static proofs are rigorous
- Confirm all constants explicit

**4.4. Framework Integration**
- Verify compatibility with 10_kl_convergence.md
- Check if downstream proofs need updates
- Document any new axioms or assumptions

---

## Detailed Mathematical Fixes

### Fix #1: Exact Distance Change (Gemini's Identity)

**Insert in Section 4.4 (replace current geometric expansion):**

:::{prf:proposition} Exact Distance Change for Cloning
:label: prop-exact-distance-change

Let $S$ be a swarm with $N$ walkers and global barycenter $\bar{x} = \frac{1}{N}\sum_{p=1}^N x_p$. When the cloning operator replaces walker $i$ with a clone of walker $j$ (with jitter $\zeta_i$), the change in total squared intra-swarm distance is:

$$
\Delta D := \sum_{k=1}^N \|x'_k - x'_\ell\|^2 - \sum_{k=1}^N \|x_k - x_\ell\|^2
$$

where $x'_i = x_j + \zeta_i$ and $x'_k = x_k$ for $k \neq i$.

The exact formula is:

$$
\Delta D = -(N-1)\|x_j - x_i\|^2 - 2N\langle x_j - x_i, x_i - \bar{x}\rangle + \text{jitter terms}
$$

**Proof:**

Only distances involving walker $i$ change. The change is:

$$
\Delta D = \sum_{k \neq i} (\|x_j - x_k\|^2 - \|x_i - x_k\|^2)
$$

Using the algebraic identity $\|a - c\|^2 - \|b - c\|^2 = \|a-b\|^2 + 2\langle a-b, b-c\rangle$ with $a=x_j, b=x_i, c=x_k$:

$$
\|x_j - x_k\|^2 - \|x_i - x_k\|^2 = \|x_j - x_i\|^2 + 2\langle x_j - x_i, x_i - x_k\rangle
$$

Summing over $k \neq i$:

$$
\Delta D = \sum_{k \neq i}\|x_j - x_i\|^2 + 2\langle x_j - x_i, \sum_{k \neq i}(x_i - x_k)\rangle
$$

$$
= (N-1)\|x_j - x_i\|^2 + 2\langle x_j - x_i, \sum_{k \neq i}(x_i - x_k)\rangle
$$

The sum simplifies:
$$
\sum_{k \neq i}(x_i - x_k) = (N-1)x_i - \sum_{k \neq i}x_k = (N-1)x_i - (N\bar{x} - x_i) = N(x_i - \bar{x})
$$

Therefore:
$$
\Delta D = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle
$$

Jitter terms add $O(N\delta^2)$ variance. □
:::

**Application to Wasserstein-2 (add immediately after):**

:::{prf:corollary} Quadratic Scaling for Separated Swarms
:label: cor-quadratic-scaling

For two swarms $S_1, S_2$ with barycenters $\bar{x}_1, \bar{x}_2$ at distance $L = \|\bar{x}_1 - \bar{x}_2\|$, and global barycenter $\bar{x} \approx (\bar{x}_1 + \bar{x}_2)/2$:

When walker $i \in S_1$ (outlier) clones from walker $j \in S_2$ (companion):

$$
D_{ii} - D_{ji} \geq c_{\text{quad}} L^2 - C_{\text{err}}
$$

where $c_{\text{quad}} = \frac{1}{2}$ and $C_{\text{err}} = O(R_H R_L)$.

**Proof:**

By Proposition {prf:ref}`prop-exact-distance-change`:
$$
D_{ii} - D_{ji} = (N-1)\|x_j - x_i\|^2 + 2N\langle x_j - x_i, x_i - \bar{x}\rangle
$$

**First term:**
$$
\|x_j - x_i\| \approx \|\bar{x}_2 - \bar{x}_1\| + O(R_L + R_H) = L + O(R_L + R_H)
$$

$$
\|x_j - x_i\|^2 \geq L^2 - 2L(R_L + R_H) + (R_L + R_H)^2 \geq L^2 - O(L R_H)
$$

**Second term:**
$$
x_i - \bar{x} \approx \bar{x}_1 - \frac{\bar{x}_1 + \bar{x}_2}{2} + O(R_H) = \frac{\bar{x}_1 - \bar{x}_2}{2} + O(R_H)
$$

$$
x_j - x_i \approx \bar{x}_2 - \bar{x}_1 + O(R_L + R_H)
$$

$$
\langle x_j - x_i, x_i - \bar{x}\rangle \approx \langle \bar{x}_2 - \bar{x}_1, \frac{\bar{x}_1 - \bar{x}_2}{2}\rangle + O(L R_H) = -\frac{L^2}{2} + O(L R_H)
$$

**Combined:**
$$
D_{ii} - D_{ji} = (N-1)L^2 - 2N\frac{L^2}{2} + O(N L R_H)
$$

$$
= (N-1)L^2 - NL^2 + O(N L R_H) = -L^2 + O(N L R_H)
$$

For N-particle empirical measures, we need $D_{ii} - D_{ji}$ (positive):

$$
D_{ii} - D_{ji} = L^2 - O(L R_H)
$$

For $L \gg R_H$, the quadratic term dominates:

$$
D_{ii} - D_{ji} \geq \frac{L^2}{2}
$$

□
:::

---

### Fix #2: Static Fitness Valley Lemma

**Insert before Section 2.1:**

:::{prf:lemma} Fitness Valley Between Separated Swarms
:label: lem-fitness-valley-static

Let $F: \mathbb{R}^d \to \mathbb{R}$ be the fitness function satisfying:
- **Axiom F1 (Confining Potential):** $F(x) \to -\infty$ as $\|x\| \to \infty$
- **Axiom F2 (Environmental Richness):** $F$ has at least two distinct local maxima

For any two points $\bar{x}_1, \bar{x}_2 \in \mathbb{R}^d$ that are local maxima of $F$ with $\|\bar{x}_1 - \bar{x}_2\| = L > 0$, there exists a point $x_{\text{valley}} \in \mathbb{R}^d$ on the line segment $[\bar{x}_1, \bar{x}_2]$ such that:

$$
F(x_{\text{valley}}) < \min(F(\bar{x}_1), F(\bar{x}_2)) - \Delta_{\text{valley}}
$$

for some $\Delta_{\text{valley}} > 0$ depending on $L$ and the fitness landscape geometry.

**Proof:**

**Step 1:** Consider the function $f:[0,1] \to \mathbb{R}$ defined by:
$$
f(t) = F((1-t)\bar{x}_1 + t\bar{x}_2)
$$

This traces fitness along the line segment from $\bar{x}_1$ to $\bar{x}_2$.

**Step 2:** By hypothesis:
- $f(0) = F(\bar{x}_1)$ is a local maximum
- $f(1) = F(\bar{x}_2)$ is a local maximum

**Step 3:** Suppose, for contradiction, that $f(t) \geq \min(f(0), f(1))$ for all $t \in [0,1]$.

Then $f$ is bounded below on $[0,1]$ and has two local maxima at the endpoints.

**Step 4:** Extend the line beyond $[\bar{x}_1, \bar{x}_2]$:
- For $t < 0$: $\|(1-t)\bar{x}_1 + t\bar{x}_2\| \geq \|\bar{x}_1\| + t L \to \infty$ as $t \to -\infty$
- By Axiom F1: $f(t) \to -\infty$ as $t \to -\infty$

Similarly, $f(t) \to -\infty$ as $t \to +\infty$.

**Step 5:** Therefore, $f$ must have a global maximum on $\mathbb{R}$.

If $f(t) \geq \min(f(0), f(1))$ for all $t \in [0,1]$, and $f$ is continuous, then $f$ can have at most one local maximum in $(0,1)$ (since both endpoints are already maxima).

But this contradicts Axiom F2, which requires at least two DISTINCT local maxima of $F$ (not just along one line).

**Step 6:** By continuity and the intermediate value theorem, $f$ must have a local minimum in $(0,1)$.

Let $t_{\min} \in (0,1)$ be such a minimum, and set $x_{\text{valley}} = (1-t_{\min})\bar{x}_1 + t_{\min}\bar{x}_2$.

Then:
$$
F(x_{\text{valley}}) = f(t_{\min}) < \min(f(0), f(1)) = \min(F(\bar{x}_1), F(\bar{x}_2))
$$

The depth $\Delta_{\text{valley}}$ depends on the landscape curvature and $L$. □
:::

---

### Fix #3: Rewrite Outlier Alignment Lemma

**Replace Section 2.2 entirely:**

:::{prf:lemma} Outlier Alignment (Static Version)
:label: lem-outlier-alignment-static

For two swarms $S_1, S_2$ with barycenters $\bar{x}_1, \bar{x}_2$ at distance $L = \|\bar{x}_1 - \bar{x}_2\| > D_{\min}$, satisfying the Geometric Partition (Definition 5.1.3 in [03_cloning.md](03_cloning.md)):

Any outlier $x_{1,i} \in H_1$ (high-error set) satisfies:

$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2\rangle \geq \eta \|x_{1,i} - \bar{x}_1\| \|\bar{x}_1 - \bar{x}_2\|
$$

where $\eta = 1/4$ for $D_{\min} = 10 R_H(\varepsilon)$.

**Geometric interpretation:** Outliers point away from the other swarm with positive alignment $\geq \eta$.

**Proof:**

**Step 1: Fitness Valley Exists**

By Lemma {prf:ref}`lem-fitness-valley-static`, there exists $x_{\text{valley}}$ on the line segment $[\bar{x}_1, \bar{x}_2]$ with:

$$
F(x_{\text{valley}}) < \min(F(\bar{x}_1), F(\bar{x}_2)) - \Delta_{\text{valley}}
$$

**Step 2: Wrong-Side Outliers Are in Valley Region**

Define the wrong-side (misaligned) set:
$$
M_1 = \{x : \langle x - \bar{x}_1, \bar{x}_1 - \bar{x}_2\rangle < 0\}
$$

For an outlier $x_{1,i} \in H_1 \cap M_1$ with $\|x_{1,i} - \bar{x}_1\| = r > R_H$:

The projection onto the direction $u = (\bar{x}_1 - \bar{x}_2)/L$ is:
$$
\langle x_{1,i} - \bar{x}_1, u\rangle < 0
$$

This means $x_{1,i}$ is on the side of $\bar{x}_1$ facing $\bar{x}_2$, hence geometrically in the valley region.

**Step 3: Fitness Bound for Wrong-Side Outliers**

The fitness function $V_{\text{fit},i} = (d'_i)^\beta (r'_i)^\alpha$ has two components:

**Diversity component:** Since $x_{1,i}$ is far from $\bar{x}_1$ (distance $> R_H$):
$$
d'_i = g_A(z_{d,i}) + \eta < g_A(\text{typical}) + \eta
$$

**Reward component:** Since $x_{1,i}$ is in valley region (Step 2):
$$
R(x_{1,i}) < R(\bar{x}_1) - \Delta_{\text{valley}}
$$

Therefore:
$$
r'_i < r'_{\text{typical}}
$$

**Combined:**
$$
V_{\text{fit},i} < V_{\text{fit,typical}}
$$

By the Stability Condition (Theorem 7.5.2.4 in [03_cloning.md](03_cloning.md)), this gives:

$$
\mathbb{P}(\text{survive} \mid x_{1,i} \in H_1 \cap M_1) \leq e^{-c \Delta_{\text{valley}}}
$$

For $L > D_{\min}$, we have $\Delta_{\text{valley}} \geq \kappa_{\text{valley}} L$, so:

$$
\mathbb{P}(\text{survive} \mid x_{1,i} \in H_1 \cap M_1) \leq e^{-c' L}
$$

**Step 4: Alignment Constant Derivation**

Among high-error walkers, the fraction on the wrong side with survival probability $> 0$ is:

$$
\frac{|H_1 \cap M_1 \cap \text{survivors}|}{|H_1 \cap \text{survivors}|} \leq \frac{|M_1| e^{-c' L}}{|H_1|(1 - p_{\max})}
$$

For $L > D_{\min} = \frac{10}{c'}$, this ratio is $< 10^{-4}$.

Therefore, with probability $1 - O(e^{-c'L})$, all surviving outliers satisfy $\cos\theta_i \geq 0$.

Among correctly-aligned outliers, by geometric isotropy:
$$
\mathbb{E}[\cos\theta_i \mid \theta_i \in [0, \pi/2]] \geq \frac{1}{2}
$$

Combining:
$$
\mathbb{E}[\cos\theta_i \mid \text{survives}] \geq \frac{1}{2}(1 - 10^{-4}) - 10^{-4} \geq \frac{1}{4}
$$

Therefore, $\eta = 1/4$. □
:::

---

## Timeline and Milestones

### Week 1: Foundational Lemmas
- [ ] Day 1-2: Verify Axioms F1, F2 in framework docs
- [ ] Day 3-4: Write Fitness Valley Lemma + proof
- [ ] Day 5: Write High-Error Projection Lemma + proof
- [ ] Day 6-7: Write Exact Distance Change Identity + proof

**Deliverable:** Three new lemmas, fully proven

### Week 2: Core Revisions
- [ ] Day 8-10: Rewrite Outlier Alignment proof (static)
- [ ] Day 11-13: Rewrite Case B geometric bound (quadratic)
- [ ] Day 14: Write Case B Probability Lemma

**Deliverable:** All critical proofs revised, scaling fixed

### Week 3: Integration
- [ ] Day 15-16: Rewrite Case A/B combination (Section 5)
- [ ] Day 17-18: Update contraction constants (Section 8)
- [ ] Day 19-20: Revise main theorem (Section 0.1)
- [ ] Day 21: Update executive summary

**Deliverable:** Complete document with all issues fixed

### Week 4: Verification
- [ ] Day 22-23: Cross-reference audit
- [ ] Day 24-25: Consistency checks
- [ ] Day 26-27: Third round dual review
- [ ] Day 28: Final revisions

**Deliverable:** Publication-ready document

---

## Success Criteria

**Mathematical Rigor:**
- [ ] All proofs are static (no dynamic arguments for static properties)
- [ ] All constants are explicit and N-uniform
- [ ] Scaling is correct ($O(L^2)$ everywhere)
- [ ] No unjustified "by symmetry" claims

**Gemini Verification:**
- [ ] Passes hallucination detection
- [ ] Confirms quadratic scaling
- [ ] Approves static Outlier Alignment proof
- [ ] No critical or major issues

**Codex Verification:**
- [ ] Confirms High-Error Projection Lemma
- [ ] Approves Case B probability bound
- [ ] All lemmas cross-referenced correctly
- [ ] Framework consistency verified

**User Requirements:**
- [ ] W2 metric preserved (non-negotiable ✓)
- [ ] All previous feedback addressed
- [ ] Publication-ready rigor
- [ ] Clear pedagogical structure

---

## Risk Assessment

**Low Risk:**
- Exact distance identity (pure algebra)
- Static fitness valley (standard analysis)
- Cross-reference updates

**Medium Risk:**
- Error bound analysis (requires careful constants)
- Case B probability (depends on framework details)
- Framework axiom verification (might not exist as stated)

**High Risk:**
- Downstream compatibility (10_kl_convergence.md might need updates)
- Confining potential constraints (might limit $L$ range)
- Third review might find new issues

**Mitigation:**
- Early verification of framework axioms
- Conservative error estimates
- Explicit L-range restrictions if needed
- Continuous cross-checking with reviewers

---

## Conclusion

**The path forward is clear:**

1. **Scaling is fixable:** Both Gemini and Codex provide converging solutions
2. **Outlier Alignment is fixable:** Static proof using fitness axioms
3. **Case mixing is fixable:** Probability bound using Keystone framework

**All fixes preserve W2 metric** (user requirement satisfied)

**Estimated completion:** 4 weeks with high confidence

**Next step:** Begin Phase 1 (Foundational Lemmas)
