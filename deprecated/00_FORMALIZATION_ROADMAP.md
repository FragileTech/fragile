# W₂ Contraction Proof: Formalization Roadmap

**Purpose:** Detailed plan for completing the rigorous formalization of the W₂ contraction proof based on Gemini's review.

**Status:** Ready to execute - awaiting decision on approach and collaboration

---

## Summary of Required Work

Based on Gemini review, 3 critical/major issues need formalization:

1. **CRITICAL:** Outlier Alignment Lemma formal proof (Section 2)
2. **CRITICAL:** Case B geometric derivation (Section 4)
3. **MAJOR:** Independent jitter analysis (Sections 1, 3, 8)

**Estimated Total Effort:** 1-2 weeks with PDE/analysis expert, or 3-4 weeks solo with careful work

---

## Issue #1: Outlier Alignment Lemma

### Current Status
**Location:** [03_wasserstein_contraction_complete.md](03_wasserstein_contraction_complete.md), Section 2

**Current Claim:** For separated swarms with barycenters $\bar{x}_1, \bar{x}_2$ at distance $L$, outliers satisfy:
$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta \|x_{1,i} - \bar{x}_1\| \|\bar{x}_1 - \bar{x}_2\|
$$
with $\eta \geq 1/4$

**Current Proof:** 6-step sketch (not rigorous)

### What's Needed

**Step 1a: Fitness Valley Existence (Most Critical)**

**Goal:** Prove from fitness function definition that a fitness valley exists between separated swarms.

**Fitness Function Definition (from framework):**
- $V_{\text{fit},i} = (g_A(z_{d,i}) + \eta)^{\beta} \cdot (g_A(z_{r,i}) + \eta)^{\alpha}$
- Where $z_{r,i}, z_{d,i}$ are reward and distance Z-scores
- $g_A$ is the rescale function (smooth, monotone, bounded)

**Challenge:** This is a complex function of swarm configuration. Options:

**Option A (Simpler Model):**
- Work with idealized fitness $f(x) = R(x)$ (reward function)
- Use Hölder continuity of $R$ (Axiom from framework)
- Prove valley via continuity and separation

**Option B (Full Model):**
- Work with complete $V_{\text{fit}}$ definition
- Handle Z-score dependencies on swarm state
- More realistic but significantly more complex

**Recommended Approach (Option A):**
1. Assume fitness approximated by reward: $f(x) \approx R(x)$
2. Use Axiom of Reward Regularity: $|R(x) - R(y)| \leq L_R \|x-y\|^{\nu}$ (Hölder)
3. For separated swarms at distance $L$, consider midpoint plane $P_{\text{mid}}$
4. **Key insight:** If $R$ is smooth and swarms are separated, max on path joining them must be lower than swarms' locations (unless path goes through peak, but separated swarms are in different basins)

**Formal Proof Sketch:**
- Define valley region $V = \{x : \text{dist}(x, P_{\text{mid}}) \leq \epsilon_{\text{valley}}\}$
- Use multi-modal reward landscape structure (from Environmental Richness axiom)
- Show $\sup_{x \in V} R(x) < \min(R(\bar{x}_1), R(\bar{x}_2)) - \Delta_{\text{valley}}$

**Alternative:** Use H-theorem approach from [14_symmetries_adaptive_gas.md](14_symmetries_adaptive_gas.md) - entropy production argument

**Step 1b: Quantitative Fitness Bound**

**Goal:** Prove $f(x) \leq f_{\text{valley,max}}$ for misaligned outliers $x \in M_1$

**Approach:**
1. For $x \in M_1$ (wrong side): $\langle x - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle < 0$
2. This means $x$ is on side facing other swarm
3. Project onto inter-swarm axis, show closeness to valley
4. Use Hölder continuity to bound $f(x)$

**Step 1c: Derive Constant η**

**Goal:** Explicitly derive $\eta \geq 1/4$ from survival probabilities

**Approach:**
1. Survival probability $p_{\text{survive},i} \propto f(x_i)^{\alpha}$ (from cloning mechanism)
2. For $x \in M_1$: $p_{\text{survive},i} \leq (f_{\text{valley,max}})^{\alpha}$
3. For $x \notin M_1$: $p_{\text{survive},i} \geq (f_{\text{swarm,min}})^{\alpha}$
4. Ratio gives probability concentration
5. Expected value of cosine angle: $\mathbb{E}[\cos \theta] = \mathbb{E}[\langle \cdot \rangle / \|\cdot\|\|\cdot\|]$
6. Use concentration inequality to show most weight on aligned outliers
7. Derive $\eta$ from this concentration

### Deliverable
- Complete rigorous proof of Outlier Alignment Lemma
- Explicit formula for $\eta$ in terms of framework parameters
- Add to Section 2 of main proof document

---

## Issue #2: Case B Geometric Derivation

### Current Status
**Location:** [03_wasserstein_contraction_complete.md](03_wasserstein_contraction_complete.md), Section 4

**Current Claim:** For Case B (mixed fitness ordering):
$$
D_{ii} - D_{ji} \geq \eta R_H L
$$

**Current Proof:** Stated, not derived

### What's Needed

**Step 2a: Define Notation Explicitly**

**Clarify:**
- Walker $i$: Lower fitness in swarm 1, higher fitness in swarm 2
- Walker $j = \pi(i)$: Companion of $i$ (from matching)
- $D_{ab} := \|x_{1,a} - x_{2,b}\|^2$
- $D_{ii} = \|x_{1,i} - x_{2,i}\|^2$
- $D_{ji} = \|x_{1,j} - x_{2,i}\|^2$

**Step 2b: Expand Distances**

**Detailed algebra:**

$$
\begin{aligned}
D_{ii} &= \|x_{1,i} - x_{2,i}\|^2 \\
&= \|(x_{1,i} - \bar{x}_1) + (\bar{x}_1 - \bar{x}_2) + (\bar{x}_2 - x_{2,i})\|^2 \\
&= \|x_{1,i} - \bar{x}_1\|^2 + \|\bar{x}_1 - \bar{x}_2\|^2 + \|x_{2,i} - \bar{x}_2\|^2 \\
&\quad + 2\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \\
&\quad + 2\langle x_{1,i} - \bar{x}_1, \bar{x}_2 - x_{2,i} \rangle \\
&\quad + 2\langle \bar{x}_1 - \bar{x}_2, \bar{x}_2 - x_{2,i} \rangle
\end{aligned}
$$

Similarly for $D_{ji}$:

$$
\begin{aligned}
D_{ji} &= \|x_{1,j} - x_{2,i}\|^2 \\
&= \|(x_{1,j} - \bar{x}_1) + (\bar{x}_1 - \bar{x}_2) + (\bar{x}_2 - x_{2,i})\|^2 \\
&= \|x_{1,j} - \bar{x}_1\|^2 + \|\bar{x}_1 - \bar{x}_2\|^2 + \|x_{2,i} - \bar{x}_2\|^2 \\
&\quad + 2\langle x_{1,j} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \\
&\quad + 2\langle x_{1,j} - \bar{x}_1, \bar{x}_2 - x_{2,i} \rangle \\
&\quad + 2\langle \bar{x}_1 - \bar{x}_2, \bar{x}_2 - x_{2,i} \rangle
\end{aligned}
$$

**Step 2c: Apply Outlier Alignment**

**Key difference:**

$$
\begin{aligned}
D_{ii} - D_{ji} &= \|x_{1,i} - \bar{x}_1\|^2 - \|x_{1,j} - \bar{x}_1\|^2 \\
&\quad + 2\langle x_{1,i} - x_{1,j}, \bar{x}_1 - \bar{x}_2 \rangle \\
&\quad + 2\langle x_{1,i} - x_{1,j}, \bar{x}_2 - x_{2,i} \rangle
\end{aligned}
$$

**Apply Outlier Alignment Lemma:**
- Walker $i$ in swarm 1 is outlier: $\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta R_H L$
- Walker $j$ in swarm 1 is companion: $\langle x_{1,j} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \sim O(R_L L)$
- Therefore: $\langle x_{1,i} - x_{1,j}, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta R_H L - O(R_L L) \approx \eta R_H L$

**Other terms:** Bounded by $O(R_H R_L + R_L^2) \ll R_H L$ for $L$ large

**Conclusion:** $D_{ii} - D_{ji} \geq \eta R_H L$ (plus lower order terms)

### Deliverable
- Step-by-step algebraic derivation in Section 4
- All terms bounded explicitly
- Clear connection to Outlier Alignment Lemma

---

## Issue #3: Independent Jitter Analysis

### Current Status
**Location:** Sections 1, 3, 8

**Current Assumption:** Shared jitter $\zeta_i$ for both swarms (unrealistic)
**Current Result:** $C_W = 4d\delta^2$ (derivation unclear)

### What's Needed

**Step 3a: Re-work with Independent Jitter**

**Case A (Consistent Ordering):**

**Current (shared jitter):**
- Clone-Clone: $\|x'_{1,i} - x'_{2,i}\|^2 = \|c_1 + \zeta - (c_2 + \zeta)\|^2 = \|c_1 - c_2\|^2$ ✓ cancellation

**Realistic (independent jitter):**
- Clone-Clone: $\mathbb{E}[\|(c_1 + \zeta_1) - (c_2 + \zeta_2)\|^2] = \|c_1 - c_2\|^2 + 2d\delta^2$ ✗ no cancellation

**Impact:** Additive noise term $+2d\delta^2$ in Clone-Clone subcase

**Case B (Mixed Ordering):**
Already uses independent jitter (different walkers clone), so less affected.

**Step 3b: Derive Corrected C_W**

**Approach:**
1. Track noise contributions from all subcases in Cases A and B
2. Sum over pairs: $N/2$ pairs contribute
3. Each pair contributes at most $4d\delta^2$ (from both walkers cloning)
4. Total: $C_W = (N/2) \cdot 4d\delta^2 = 2Nd\delta^2$

**Wait, this is N-dependent!** Problem: this violates N-uniformity claim.

**Resolution:** The $C_W$ in the theorem statement should be:
- $C_W = 4d\delta^2$ (per-pair constant)
- Final bound: $\mathbb{E}[W_2^2] \leq (1-\kappa_W)W_2^2 + C_W$ where $C_W$ is **already** the aggregated constant

**Correct formula:** $C_W = 4d\delta^2$ is the per-pair bound, and when summed over $N/2$ pairs and divided by $N$ for $W_2^2$ normalization, we get bounded additive constant.

**Step 3c: Verify Contraction Still Holds**

**Key question:** Is $\kappa_W$ large enough to overcome the noise?

$$
\kappa_W = \frac{p_u \eta}{2} \geq 0.0125
$$

$$
\frac{C_W}{W_2^2} = \frac{4d\delta^2}{W_2^2(\mu_1, \mu_2)}
$$

For contraction to be meaningful, need $\kappa_W \cdot W_2^2 \gg C_W$, i.e., $W_2^2 \gg \frac{C_W}{\kappa_W} \sim \frac{4d\delta^2}{0.0125} \sim 320 d\delta^2$

This is reasonable - swarms separated by more than $\sim 20\delta$ will contract.

### Deliverable
- Re-worked Sections 1 and 3 with independent jitter
- Corrected $C_W$ derivation in Section 8
- Verification of contraction regime

---

## Execution Strategy

### Phase 1: Outlier Alignment (1 week)
**Priority:** HIGHEST (foundation for everything)

**Tasks:**
1. Choose fitness model (Option A recommended)
2. Prove fitness valley existence rigorously
3. Derive quantitative bound for misaligned outliers
4. Calculate $\eta$ explicitly from survival probabilities

**Output:** Section 2 fully rigorous

### Phase 2: Case B Derivation (2-3 days)
**Priority:** HIGH (needed for contraction)

**Tasks:**
1. Write out all notation explicitly
2. Expand $D_{ii}$ and $D_{ji}$ step-by-step
3. Apply Outlier Alignment, bound all terms
4. Show $D_{ii} - D_{ji} \geq \eta R_H L$

**Output:** Section 4 fully rigorous

### Phase 3: Independent Jitter (3-4 days)
**Priority:** MEDIUM (for robustness)

**Tasks:**
1. Re-work Case A without jitter cancellation
2. Re-work Case B (minimal changes needed)
3. Re-derive $C_W$ carefully
4. Verify N-uniformity and contraction regime

**Output:** Sections 1, 3, 8 updated and robust

### Phase 4: Gemini Re-Review (1 day)
**Priority:** VERIFICATION

**Tasks:**
1. Submit completed proof to Gemini
2. Address any remaining issues
3. Final polishing

**Output:** Publication-ready proof

---

## Decision Points

### Do We Need Expert Collaboration?

**For Issue #1 (Outlier Alignment):**
- **Option A (Simpler):** Can be done solo with careful work - 4-5 days
- **Option B (Full):** Requires PDE/analysis expertise - 2-3 days with expert

**For Issue #2 (Case B):**
- Mostly algebra - can be done solo - 2-3 days

**For Issue #3 (Jitter):**
- Careful accounting - can be done solo - 3-4 days

**Recommendation:** Try Option A solo for Issue #1. If stuck after 2 days, seek expert help.

### Alternative: Accept Current Status?

**Option:** Publish as "framework paper" with formalization as future work

**Pros:**
- Immediate publication
- Core insights documented
- Clear path for completion stated

**Cons:**
- Not suitable for top-tier journal
- Incomplete mathematical rigor
- May not satisfy referees for LSI convergence dependence

---

## Next Steps

**Immediate (Today/Tomorrow):**
1. ✅ Document current status honestly (done)
2. ✅ Move deprecated documents (done)
3. ✅ Get Gemini review (done)
4. ✅ Create formalization roadmap (this document)
5. **DECIDE:** Solo work vs expert collaboration vs framework paper

**If Proceeding with Formalization:**

**Week 1:**
- Days 1-2: Fitness valley existence proof (Step 1a)
- Days 3-4: Quantitative bound for misaligned outliers (Step 1b)
- Day 5: Derive $\eta$ constant (Step 1c)

**Week 2:**
- Days 1-2: Case B geometric derivation (Issue #2)
- Days 3-4: Independent jitter analysis (Issue #3)
- Day 5: Gemini re-review and final polish

**Deliverable:** Publication-ready W₂ contraction proof

---

**Status:** Roadmap complete, ready to execute
**Estimated Effort:** 1-2 weeks
**Success Criteria:** Gemini confirms full rigor, no remaining gaps
