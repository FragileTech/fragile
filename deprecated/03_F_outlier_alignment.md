# Outlier Alignment Lemma: Emergent Directional Structure from Cloning Dynamics

**Status:** Proof to be completed with Gemini

**Purpose:** Establishes that outliers in separated swarms are preferentially located on the "far side" away from the other swarm. This property is **emergent** from cloning dynamics, not an additional axiom.

---

## Statement

:::{prf:lemma} Asymptotic Outlier Alignment
:label: lem-outlier-alignment

Let $S_1$ and $S_2$ be two swarms satisfying the Geometric Partition (Definition 5.1.3 in [03_cloning.md](03_cloning.md)), with barycenters $\bar{x}_1$ and $\bar{x}_2$ separated by $\|\bar{x}_1 - \bar{x}_2\| = L > D_{\min}$ for some threshold $D_{\min}$ sufficiently large.

Let the fitness function $f(x)$ support these two separated swarms (i.e., both swarms are stable under the cloning operator).

Then for any outlier $x_{1,i} \in H_1$ (high-error set of swarm 1), the following **Outlier Alignment** property holds:

$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta \|x_{1,i} - \bar{x}_1\| \|\bar{x}_1 - \bar{x}_2\|
$$

where $\eta > 0$ is a uniform constant depending only on the framework parameters (not on $N$, $L$, or the specific swarm configurations).

**Geometric interpretation:** The vector from barycenter to outlier $(x_{1,i} - \bar{x}_1)$ has a positive projection onto the vector pointing away from the other swarm $(\bar{x}_1 - \bar{x}_2)$. Outliers are on the "far side" of their swarm.
:::

---

## Proof (To Be Completed with Gemini)

**Strategy (from Gemini's sketch):**

The proof establishes that outliers cannot survive on the "wrong side" of their swarm (the side facing the other swarm) because:
1. Two stable separated swarms imply a fitness valley between them
2. The cloning operator systematically removes low-fitness walkers
3. Walkers in the valley have low survival probability
4. Therefore surviving outliers must be on the far side

**Required steps:**

### Step 1: Establish the Fitness Valley

**Claim:** If two swarms $S_1$ and $S_2$ are stably separated, there exists a region of lower fitness between them.

**Formalization needed:**
- Define midpoint plane: $P = \{z \mid \langle z - \frac{\bar{x}_1 + \bar{x}_2}{2}, \bar{x}_1 - \bar{x}_2 \rangle = 0\}$
- Show: $\sup_{z \in P} f(z) < \min(f(\bar{x}_1), f(\bar{x}_2))$
- Use stability argument: if valley didn't exist, cloning would cause one swarm to collapse into the other

### Step 2: Quantify Survival Probability

From the cloning operator definition (Chapter 9, [03_cloning.md](03_cloning.md)):
- Walker $i$ survives with probability proportional to fitness $f(x_i)$
- Cloning score: $S_i = (V_{\text{fit},c_i} - V_{\text{fit},i})/(V_{\text{fit},i} + \varepsilon_{\text{clone}})$
- Survival probability: $p_{\text{survive},i} \propto V_{\text{fit},i}^{\alpha}$

### Step 3: Define the "Wrong Side"

For swarm $S_1$, define the **misaligned set**:

$$
M_1 = \{x \mid \langle x - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle < 0\}
$$

This is the half-space on the side of $S_1$ that faces $S_2$.

An outlier $x_{1,i} \in H_1 \cap M_1$ is "on the wrong side."

### Step 4: Show Low Fitness on the Wrong Side

**Key observation:** For an outlier on the wrong side:
- It is far from $\bar{x}_1$: $\|x_{1,i} - \bar{x}_1\| \geq R_H$ (by definition of high-error set)
- It is in the direction of $S_2$: $\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle < 0$
- Therefore it is in or near the fitness valley

**Requires:** Bounding $f(x_{1,i})$ from above using distance to valley plane $P$

### Step 5: Bound the Probability

Combine Steps 2 and 4 to show:

$$
\mathbb{P}(\text{outlier survives} \mid x_{1,i} \in H_1 \cap M_1) \leq \varepsilon(L)
$$

where $\varepsilon(L) \to 0$ as $L \to \infty$.

As the separation $L = \|\bar{x}_1 - \bar{x}_2\|$ increases, the probability of finding a surviving outlier in the misaligned set $M_1$ tends to zero.

### Step 6: Conclude Alignment

For sufficiently separated systems ($L > D_{\min}$), almost all surviving outliers satisfy:

$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle > 0
$$

The quantitative bound with constant $\eta$ follows from the concentration of the outlier distribution away from the valley.

---

## Connection to Keystone Principles

This lemma connects to existing framework results:

**From [03_cloning.md](03_cloning.md):**
- **Definition 5.1.3**: Geometric Partition (high-error vs low-error sets)
- **Corollary 6.4.4**: Large variance guarantees non-vanishing $f_H > 0$
- **Lemma 6.5.1**: Geometric separation $R_H \gg R_L$
- **Theorem 7.5.2.4**: Stability Condition (fitness correctly identifies outliers)
- **Lemma 8.3.2**: Unfit walkers have cloning probability $p_i \geq p_u(\varepsilon)$

**This lemma adds:** The directional/angular structure of outliers (not just their radial distance from barycenter).

---

## Usage in W₂ Contraction Proof

This lemma is used in the Case B (Mixed Ordering) contraction proof to bound cross-terms:

**In Case B:**
- $D_{ii} = \|x_{1,i} - x_{2,i}\|^2$ where $x_{1,i}$ is outlier, $x_{2,i}$ is central
- $D_{ji} = \|x_{1,j} - x_{2,i}\|^2$ where both are central

**The key inequality:** $D_{ii} - D_{ji} \geq \alpha_B \|x_{1,i} - x_{1,j}\|^2$

**Uses Outlier Alignment via:**
$$
D_{ii} - D_{ji} = 2\langle x_{1,i} - x_{1,j}, \frac{x_{1,i} + x_{1,j}}{2} - x_{2,i}\rangle
$$

The Outlier Alignment property ensures the inner product is positive when $x_{1,i}$ is an outlier aligned away from swarm 2.

---

## Next Steps

1. **Complete formal proof** with Gemini (fill in Steps 1-6)
2. **Derive explicit bound** for constant $\eta$
3. **Verify N-uniformity** of all constants
4. **Use in Case B proof** to establish uniform contraction factor

**Status:** Framework established, proof skeleton ready for completion.
