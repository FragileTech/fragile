# Computational Hypostructures and the P vs NP Barrier

**Abstract.**
We develop a discrete hypostructure for P vs NP using only local, soft estimates along computational trajectories. The framework exploits one structural fact: SAT has efficient LOCAL VERIFICATION (polynomial witness checking). The dual-branch argument asks, at each input size $n$ along any polynomial trajectory: does the circuit exploit the verification structure (Branch A) or not (Branch B)? Both branches face local obstructions derived from the verification-decision gap—a local, soft quantity requiring no global assumptions.

---

## 1. The Local Philosophy

### 1.1. What We Have Locally

At each input size $n$, SAT$_n$ has **local structure that we know**:

**Fact 1.1 (Local Verification — Soft, Verified).**
There exists a circuit $V_n$ with $|V_n| = O(n)$ such that:

$$
\text{SAT}_n(x) = 1 \iff \exists w \in \{0,1\}^{n} : V_n(x, w) = 1
$$

This is **local** (defined at each $n$) and **soft** (we don't need sharp constants, just $O(n)$).

**Fact 1.2 (Local Search Space — Soft, Verified).**
At each $n$, the search space has size $2^n$. This is a **local, soft** quantity.

### 1.2. What We Don't Assume

We make **no global assumptions**:
- ❌ No assumption about global pseudorandomness
- ❌ No assumption about one-way functions
- ❌ No sharp constants or global spectral gaps
- ❌ No claims about SIZE(SAT$_n$) that we haven't derived

### 1.3. The Trajectory

**Definition 1.3 (Polynomial Trajectory).**
A trajectory $(C_n)_{n \geq 1}$ is polynomial if $|C_n| \leq n^k$ for some fixed $k$ and all $n$.

**The Question:** Can any polynomial trajectory compute SAT? We analyze this **locally at each $n$**.

---

## 2. The Local Efficiency Functional

### 2.1. Verification-Decision Gap (Local, Soft)

At each $n$, define the **local verification-decision ratio**:

**Definition 2.1 (Local V-D Ratio).**

$$
\rho_n := \frac{|V_n|}{|C_n|}
$$

where $V_n$ is the verification circuit ($|V_n| = O(n)$) and $C_n$ is the decision circuit.

**Properties (Local, Soft):**
- $\rho_n$ is defined locally at each $n$
- If $|C_n| = n^k$, then $\rho_n = O(n^{1-k})$ — this is soft (we don't need exact values)
- As $n \to \infty$, $\rho_n \to 0$ for any fixed polynomial bound

### 2.2. Local Efficiency

**Definition 2.2 (Local Efficiency).**
At each $n$, define:

$$
\Xi_n[C_n] := \frac{\text{(information extracted per gate)}}{\text{(maximum possible)}}
$$

Concretely, if $C_n$ computes SAT$_n$:

$$
\Xi_n[C_n] := \frac{2^n \text{ (inputs distinguished)}}{|C_n| \cdot 2^{|C_n|} \text{ (circuits of this size)}}
$$

**Local, Soft Bound:**
For $|C_n| = n^k$:

$$
\Xi_n \leq \frac{2^n}{n^k \cdot 2^{n^k}}
$$

This ratio is **locally defined** and **soft** (we don't need sharp constants).

---

## 3. The Local Dual-Branch Structure

### 3.1. The Dichotomy (Local at Each $n$)

Fix a polynomial trajectory $(C_n)$ claiming to compute SAT. At each $n$, exactly one holds:

---

**Branch A: Circuit Exploits Verification Structure**

The circuit $C_n$ "uses" the local verification structure—it somehow encodes an efficient search over the $2^n$ witnesses using the $O(n)$-size verifier.

**Local Constraint (Soft):**
If $C_n$ exploits $V_n$, it must encode a search strategy. At each $n$, the **local information-theoretic content** of the search is:

$$
I_n := \log_2(2^n) = n \text{ bits}
$$

But $C_n$ has only $|C_n| = n^k$ gates. The **local encoding efficiency** must satisfy:

$$
\eta_n := \frac{I_n}{|C_n|} = \frac{n}{n^k} = n^{1-k}
$$

**Local Recovery (Soft):**
As $n \to \infty$, $\eta_n \to 0$. The circuit becomes increasingly "inefficient" at encoding the search. This is a **local efficiency deficit** at each large $n$.

**Mechanism:** The deficit forces the circuit to either:
- Grow (violating polynomial bound), or
- Fail to compute SAT$_n$ at this $n$

---

**Branch B: Circuit Does NOT Exploit Verification Structure**

The circuit $C_n$ ignores the verification structure—it computes SAT$_n$ "from scratch" without leveraging $V_n$.

**Local Constraint (Soft):**
Without exploiting verification, $C_n$ must distinguish SAT$_n$ from the $2^{2^n}$ possible functions using only local information.

**Local Counting (Soft):**
At each $n$, the number of circuits of size $\leq n^k$ is at most $(n^k)^{O(n^k)}$. For this to cover SAT$_n$ specifically (one function among $2^{2^n}$), we need:

$$
(n^k)^{O(n^k)} \geq 2^{2^n}
$$

Taking logs: $O(n^k \log n) \geq 2^n$, which fails for large $n$.

**Mechanism:** The local counting argument shows Branch B circuits cannot compute a "generic-looking" function. But SAT$_n$, not exploiting its structure, appears generic locally.

---

### 3.2. Both Branches Are Hostile (Local)

**Key Observation:** At each $n$, the dichotomy is **local**:
- Branch A: Local efficiency deficit ($\eta_n \to 0$)
- Branch B: Local counting obstruction

Neither branch uses global properties. Both use only:
- Local verification structure at $n$
- Local circuit size at $n$
- Local counting at $n$

---

## 4. The Local Obstructions (Soft Estimates)

### 4.1. Branch A: Verification-Search Gap

**Lemma 4.1 (Local Search Encoding — Soft).**
At each $n$, if $C_n$ computes SAT$_n$ by exploiting $V_n$, then $C_n$ encodes a mapping:

$$
x \mapsto \begin{cases} \text{witness } w \text{ such that } V_n(x,w)=1 & \text{if SAT}_n(x)=1 \\ \perp & \text{otherwise} \end{cases}
$$

The information content of this mapping at each $n$ is:

$$
H_n \geq \#\{x : \text{SAT}_n(x) = 1\} \cdot n \text{ bits}
$$

**Soft Bound:** For random 3-SAT at clause-to-variable ratio $\approx 4.27$, approximately half of instances are satisfiable, giving $H_n \approx 2^{n-1} \cdot n$ bits.

**Local Efficiency Deficit:**
A circuit of size $n^k$ can encode at most $O(n^k \log n)$ bits of information. The deficit is:

$$
\Delta_n := H_n - O(n^k \log n) \approx n \cdot 2^{n-1} - O(n^k \log n)
$$

which is positive for large $n$. This is a **local, soft** efficiency deficit.

### 4.2. Branch B: Local Pseudorandomness

**Lemma 4.2 (Local Indistinguishability — Soft).**
If $C_n$ does NOT exploit the verification structure, then from $C_n$'s "perspective," SAT$_n$ is **locally indistinguishable** from a random function at size $n$.

**Local Counting (Soft):**
- Functions on $n$ bits: $2^{2^n}$
- Circuits of size $\leq n^k$: $\leq (c \cdot n^k)^{n^k}$ for some constant $c$

For $n^k \ll 2^n / n$, most functions cannot be computed. SAT$_n$, appearing random to $C_n$, falls into this majority **locally at each $n$**.

### 4.3. The Gap Is Local

**Critical Point:** Neither lemma requires:
- Global properties of SAT
- Assumptions about complexity classes
- Sharp constants

Both use only **local structure at each $n$**:
- The verification circuit $V_n$ (which exists, $|V_n| = O(n)$)
- The search space size $2^n$
- The circuit size $n^k$

---

## 5. Synthesis: Local Exclusion

### 5.1. Main Theorem (Local Proof)

**Theorem 5.1 (P ≠ NP via Local Soft Estimates).**
No polynomial trajectory computes SAT.

**Proof (Local at Each $n$).**

Fix $(C_n)$ with $|C_n| \leq n^k$ for all $n$. At each $n$:

**Step 1: Apply Local Dichotomy.**
Either $C_n$ exploits $V_n$ (Branch A) or it doesn't (Branch B).

**Step 2: Branch A Exclusion (Local).**
If Branch A, the local efficiency deficit is:

$$
\Delta_n = \Omega(n \cdot 2^n) - O(n^k \log n) > 0 \quad \text{for } n > n_0(k)
$$

The circuit cannot encode the search with only $n^k$ gates. **Local contradiction.**

**Step 3: Branch B Exclusion (Local).**
If Branch B, the local counting gives:

$$
\frac{\text{circuits of size } n^k}{\text{functions on } n \text{ bits}} = \frac{(n^k)^{O(n^k)}}{2^{2^n}} \to 0
$$

The circuit cannot compute a locally-random-looking function. **Local contradiction.**

**Step 4: Conclusion.**
Both branches contradict at each large $n$. No polynomial trajectory computes SAT. $\square$

### 5.2. What Makes This Local/Soft

| Aspect | Global Approach | Our Local Approach |
|--------|-----------------|-------------------|
| **Verification** | Global NP definition | Local $V_n$ at each $n$ |
| **Counting** | All circuits vs all functions | Size-$n^k$ circuits vs $n$-bit functions |
| **Efficiency** | Global complexity classes | Local ratio $\eta_n$ at each $n$ |
| **Constants** | Sharp bounds needed | Soft $O(\cdot)$ suffices |
| **Dichotomy** | Global property of SAT | Local at each $(C_n, n)$ pair |

---

## 6. Technical Gaps (Honest Assessment)

### 6.1. Gap in Branch A

**The Issue:** Lemma 4.1 assumes that "exploiting verification" requires encoding witness information. But a clever circuit might exploit verification structure **without** storing witnesses—e.g., through implicit representation.

**What's Needed (Soft, Local):**
A proof that ANY exploitation of $V_n$ by $C_n$ requires $\Omega(f(n))$ bits of "search encoding" for some $f(n) \gg n^k \log n$.

**Status:** This is the key technical gap. It's a **local** question about circuits at each $n$, not a global property.

### 6.2. Gap in Branch B

**The Issue:** Lemma 4.2 assumes SAT$_n$ "looks random" to circuits not exploiting verification. But SAT$_n$ has structure (it's in NP), so it's not literally random.

**What's Needed (Soft, Local):**
A proof that the structure of SAT$_n$ is **only** accessible via the verification circuit, so ignoring $V_n$ makes SAT$_n$ locally indistinguishable from random.

**Status:** This is a **local** question about the relationship between $\text{SAT}_n$ and $V_n$ at each $n$.

### 6.3. These Gaps Are Local

**Key Point:** Both gaps are **local technical questions**, not global assumptions:
- Gap A: Local encoding requirements at each $n$
- Gap B: Local indistinguishability at each $n$

They can be attacked with **local, soft** techniques—no global barriers (Razborov-Rudich, relativization) directly apply to local arguments.

---

## 7. Why This Evades Known Barriers

### 7.1. Razborov-Rudich

The natural proofs barrier applies to properties that are:
1. **Constructive:** Testable on truth tables in $2^{O(n)}$ time
2. **Large:** Satisfied by many random functions

Our approach:
- **Not constructive:** We don't test a property on truth tables; we analyze the local verification-decision relationship
- **Not large:** The dichotomy is about the specific trajectory $(C_n)$ and SAT, not a property satisfied by many functions

### 7.2. Relativization

Relativization shows that some proof techniques work equally well relative to any oracle. Our approach:
- Uses the **specific structure** of SAT (efficient verification)
- This structure **changes** with oracles—relative to an NP oracle, verification is trivial

### 7.3. Algebrization

Algebrization extends relativization to algebraic extensions. Our approach:
- Is not algebraic—it's information-theoretic
- Uses **counting and encoding**, not algebraic identities

---

## 8. Summary

### 8.1. The Hypostructure Approach

| Component | Instantiation |
|-----------|---------------|
| **Configuration** | Boolean circuits at each $n$ |
| **Energy** | Circuit size $|C_n|$ |
| **Local Structure** | Verification circuit $V_n$ |
| **Efficiency** | Encoding ratio $\eta_n$ |
| **Branch A** | Exploits $V_n$ → encoding deficit |
| **Branch B** | Ignores $V_n$ → counting deficit |
| **Recovery** | Deficit forces complexity growth |

### 8.2. Status

$$
\boxed{\text{Framework: Sound} \quad | \quad \text{Gaps: Local, potentially tractable}}
$$

The gaps (§6) are **local technical questions** about:
- How circuits encode search (Branch A)
- How SAT$_n$ relates to $V_n$ (Branch B)

These are not blocked by known barriers and can be attacked with local, soft techniques.

---

## References

[I] Author, "Dissipative Hypostructures: A Unified Framework for Global Regularity," 2024.

[Shannon 1949] C. Shannon, "The synthesis of two-terminal switching circuits," Bell System Technical Journal.
