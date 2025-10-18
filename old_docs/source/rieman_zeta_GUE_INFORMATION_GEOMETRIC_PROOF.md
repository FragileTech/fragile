# GUE Universality via Information Geometry - Rigorous Proof

**Strategy**: Bypass overlapping walker problem using HWI + Reverse Talagrand
**Status**: Draft for Gemini validation
**Key Innovation**: Bound cumulants via Fisher information metric, not direct walker correlations

---

## Executive Summary

This document provides an **information-geometric proof** of the Wigner semicircle law for the Information Graph, circumventing the "overlapping walker" obstacle that invalidated the previous graph-theoretic approach.

**Previous Problem**: For edge weights $w_{12}, w_{13}$ sharing walker 1, the limiting cumulant $\text{Cum}_{\rho_0}(w_{12}, w_{13}) \neq 0$ because they're correlated even when walkers are independent.

**New Solution**: Represent cumulants via **Fisher information metric** on probability space, then bound metric eigenvalues using:
1. **HWI inequality** (Otto-Villani): Relates KL-divergence to Wasserstein distance and Fisher information
2. **Reverse Talagrand**: Bounds Wasserstein distance by KL-divergence and Hessian curvature
3. **LSI**: Provides exponential KL-convergence

All three theorems are **already proven** in our framework.

---

## Part 1: Setup - Moment-Generating Functional

:::{prf:definition} Tilted Measure and Moment-Generating Functional
:label: def-tilted-measure-information-graph

For the Information Graph adjacency matrix $A = (A_{ij})$ and source parameters $t = (t_{ij}) \in \mathbb{R}^{N \times N}$, define:

**Tilted measure**:
$$
\mu_t(X) := \frac{1}{Z(t)} \mu_0(X) \exp\left(\sum_{i < j} t_{ij} A_{ij}(X)\right)
$$

where $\mu_0 = \nu_N^{\text{QSD}}$ is the QSD measure on walker configurations $X = (w_1, \ldots, w_N)$.

**Partition function**:
$$
Z(t) := \int \mu_0(X) \exp\left(\sum_{i < j} t_{ij} A_{ij}(X)\right) dX
$$

**Moment-generating functional**:
$$
\Psi(t) := \log Z(t)
$$
:::

:::{prf:lemma} Cumulants as Hessian of Generating Functional
:label: lem-cumulant-hessian-identity

The cumulants of matrix entries are given by derivatives of $\Psi$:

$$
\mathbb{E}[A_{ij}] = \frac{\partial \Psi}{\partial t_{ij}}\Big|_{t=0}
$$

$$
\text{Cov}(A_{ij}, A_{kl}) = \frac{\partial^2 \Psi}{\partial t_{ij} \partial t_{kl}}\Big|_{t=0}
$$

More generally:

$$
\text{Cum}(A_{i_1 j_1}, \ldots, A_{i_m j_m}) = \frac{\partial^m \Psi}{\partial t_{i_1 j_1} \cdots \partial t_{i_m j_m}}\Big|_{t=0}
$$
:::

:::{prf:proof}
Standard result from probability theory (see Anderson-Guionnet-Zeitouni, Theorem 2.1.2).

**First derivative**:
$$
\frac{\partial \Psi}{\partial t_{ij}} = \frac{1}{Z(t)} \int A_{ij}(X) \mu_0(X) e^{\sum t A} dX = \mathbb{E}_{\mu_t}[A_{ij}]
$$

At $t=0$: $\mu_t = \mu_0$, so $\partial \Psi / \partial t_{ij}|_{t=0} = \mathbb{E}[A_{ij}]$.

**Second derivative**:
$$
\frac{\partial^2 \Psi}{\partial t_{ij} \partial t_{kl}} = \frac{\partial}{\partial t_{kl}} \mathbb{E}_{\mu_t}[A_{ij}]
$$

By Stein's lemma and integration by parts:
$$
= \mathbb{E}_{\mu_t}[A_{ij} A_{kl}] - \mathbb{E}_{\mu_t}[A_{ij}]\mathbb{E}_{\mu_t}[A_{kl}] = \text{Cov}_{\mu_t}(A_{ij}, A_{kl})
$$

Higher derivatives give higher cumulants by definition.

$\square$
:::

**Key Insight**: The Hessian matrix $\nabla^2 \Psi$ is the **covariance matrix** of edge weights, which is precisely what we need to bound for the method of moments!

---

## Part 2: Fisher Information Metric = Covariance Matrix

:::{prf:theorem} Fisher Metric Equals Hessian of Generating Functional
:label: thm-fisher-metric-cumulant-identity

The Fisher information metric on the space of tilted measures $\{\mu_t\}$ coincides with the Hessian of $\Psi$:

$$
g_{ij,kl}^{\text{Fisher}} = \int \frac{\partial \log \mu_t}{\partial t_{ij}} \frac{\partial \log \mu_t}{\partial t_{kl}} d\mu_t = \frac{\partial^2 \Psi}{\partial t_{ij} \partial t_{kl}}
$$

At $t=0$:

$$
g_{ij,kl}^{\text{Fisher}}\Big|_{t=0} = \text{Cov}_{\mu_0}(A_{ij}, A_{kl})
$$
:::

:::{prf:proof}

**Step 1**: Compute score function (gradient of log-density):
$$
\frac{\partial \log \mu_t}{\partial t_{ij}} = \frac{\partial}{\partial t_{ij}} \left[\log \mu_0 + \sum t A - \log Z(t)\right]
$$

$$
= A_{ij} - \frac{\partial \Psi}{\partial t_{ij}} = A_{ij} - \mathbb{E}_{\mu_t}[A_{ij}]
$$

**Step 2**: Fisher metric is covariance of score:
$$
g_{ij,kl}^{\text{Fisher}} = \int \left(A_{ij} - \mathbb{E}_{\mu_t}[A_{ij}]\right)\left(A_{kl} - \mathbb{E}_{\mu_t}[A_{kl}]\right) d\mu_t
$$

$$
= \text{Cov}_{\mu_t}(A_{ij}, A_{kl})
$$

**Step 3**: But by Lemma lem-cumulant-hessian-identity:
$$
\text{Cov}_{\mu_t}(A_{ij}, A_{kl}) = \frac{\partial^2 \Psi}{\partial t_{ij} \partial t_{kl}}
$$

Therefore: $g^{\text{Fisher}} = \nabla^2 \Psi$.

$\square$
:::

**Crucial Observation**: The Fisher metric provides a **Riemannian geometry** on probability space. Eigenvalues of $g^{\text{Fisher}}$ bound how fast probabilities change under parameter perturbations. We can bound these eigenvalues using information-theoretic inequalities!

---

## Part 3: Bound Fisher Metric via HWI + Reverse Talagrand

:::{prf:theorem} Eigenvalue Bound on Fisher Metric
:label: thm-fisher-eigenvalue-bound-information-graph

For the Information Graph at QSD equilibrium, the Fisher information metric satisfies:

$$
\lambda_{\max}(g^{\text{Fisher}}) \leq \frac{C_{\text{LSI}}^2}{N} \cdot \frac{1}{\kappa_{\text{conf}}}
$$

where:
- $C_{\text{LSI}}$: Log-Sobolev constant (from framework Theorem thm-qsd-lsi-rigorous)
- $\kappa_{\text{conf}}$: Conformal confinement parameter (from framework Theorem thm-axiom-conformal-confinement)
- $N$: Number of walkers
:::

:::{prf:proof}

**Step 1: HWI Inequality (Otto-Villani 2000)**

From framework document `information_theory.md`, Theorem thm-hwi-inequality:

$$
D_{\text{KL}}(\mu_t || \mu_0) \leq W_2(\mu_t, \mu_0) \sqrt{I(\mu_t | \mu_0)}
$$

where:
- $D_{\text{KL}}$: Kullback-Leibler divergence
- $W_2$: 2-Wasserstein distance
- $I$: Relative Fisher information

**Step 2: Reverse Talagrand Inequality**

From framework document `10_kl_convergence/10_kl_convergence.md`:

For log-concave measure $\mu_0$ with Hessian bound $\lambda_{\min}(\text{Hess}(-\log \mu_0)) \geq \kappa_{\text{conf}}$:

$$
W_2^2(\mu_t, \mu_0) \leq \frac{2}{\kappa_{\text{conf}}} D_{\text{KL}}(\mu_t || \mu_0)
$$

**Step 3: LSI for QSD**

From framework Theorem thm-qsd-lsi-rigorous (`15_geometric_gas_lsi_proof.md`):

$$
D_{\text{KL}}(\mu || \mu_0) \leq C_{\text{LSI}} \cdot I(\mu | \mu_0)
$$

for all measures $\mu$ on walker space.

**Step 4: Combine Inequalities**

From HWI:
$$
D_{\text{KL}} \leq W_2 \sqrt{I}
$$

Square both sides:
$$
D_{\text{KL}}^2 \leq W_2^2 \cdot I
$$

From Reverse Talagrand:
$$
W_2^2 \leq \frac{2}{\kappa_{\text{conf}}} D_{\text{KL}}
$$

Substitute:
$$
D_{\text{KL}}^2 \leq \frac{2}{\kappa_{\text{conf}}} D_{\text{KL}} \cdot I
$$

Divide by $D_{\text{KL}}$ (assuming $D_{\text{KL}} > 0$):
$$
D_{\text{KL}} \leq \frac{2}{\kappa_{\text{conf}}} I
$$

From LSI:
$$
I \geq \frac{1}{C_{\text{LSI}}} D_{\text{KL}}
$$

Therefore:
$$
D_{\text{KL}} \leq \frac{2}{\kappa_{\text{conf}}} I \leq \frac{2}{\kappa_{\text{conf}}} \cdot C_{\text{LSI}} D_{\text{KL}}
$$

Wait, this gives a circular bound. Let me reconsider...

**Correct Step 4: Bound Fisher Information Directly**

The Fisher information for tilted measure is:

$$
I(\mu_t | \mu_0) = \int \|\nabla \log \frac{\mu_t}{\mu_0}\|^2 d\mu_t
$$

$$
= \int \left\|\sum_{ij} t_{ij} \left(A_{ij} - \mathbb{E}_{\mu_t}[A_{ij}]\right)\right\|^2 d\mu_t
$$

For small $|t|$, expand $\mathbb{E}_{\mu_t}[A] \approx \mathbb{E}_{\mu_0}[A]$:

$$
I(\mu_t | \mu_0) \approx \int \left\|\sum_{ij} t_{ij} (A_{ij} - \mathbb{E}[A_{ij}])\right\|^2 d\mu_0
$$

$$
= \sum_{ij,kl} t_{ij} t_{kl} \int (A_{ij} - \mathbb{E}[A_{ij}])(A_{kl} - \mathbb{E}[A_{kl}]) d\mu_0
$$

$$
= \sum_{ij,kl} t_{ij} t_{kl} \cdot \text{Cov}(A_{ij}, A_{kl})
$$

$$
= t^T \cdot g^{\text{Fisher}} \cdot t
$$

Therefore:
$$
I(\mu_t | \mu_0) = \|t\|_{g^{\text{Fisher}}}^2
$$

**Step 5: Use LSI to Bound Metric**

From LSI: $D_{\text{KL}}(\mu_t || \mu_0) \leq C_{\text{LSI}} \cdot I(\mu_t | \mu_0) = C_{\text{LSI}} \|t\|_{g}^2$

On the other hand, for small $t$:
$$
D_{\text{KL}}(\mu_t || \mu_0) = \Psi(t) - \Psi(0) - \sum t_{ij} \frac{\partial \Psi}{\partial t_{ij}}\Big|_0
$$

$$
= \frac{1}{2} t^T \nabla^2 \Psi \cdot t + O(\|t\|^3) = \frac{1}{2} \|t\|_{g^{\text{Fisher}}}^2 + O(\|t\|^3)
$$

Therefore:
$$
\frac{1}{2} \|t\|_{g}^2 \leq C_{\text{LSI}} \|t\|_{g}^2
$$

This implies:
$$
\frac{1}{2} \leq C_{\text{LSI}}
$$

which is always true (LSI constant $\geq 1/2$ for any measure).

**Step 6: Use Normalization Constraint**

The key is that matrix entries are **normalized**:
$$
\frac{1}{N^2} \sum_{ij} A_{ij}^2 = O(1)
$$

This normalization constraint implies:
$$
\sum_{ij} \text{Var}(A_{ij}) = O(N^2)
$$

Since there are $O(N^2)$ entries, average variance per entry:
$$
\text{Var}(A_{ij}) = O(1)
$$

But from normalization $A_{ij} = (w_{ij} - \mathbb{E}[w_{ij}]) / \sqrt{N\sigma_w^2}$:
$$
\text{Var}(A_{ij}) = \frac{\text{Var}(w_{ij})}{N\sigma_w^2} = \frac{O(1)}{N \cdot O(1)} = O(1/N)
$$

Therefore, diagonal entries of Fisher metric:
$$
g_{ij,ij}^{\text{Fisher}} = \text{Var}(A_{ij}) = O(1/N)
$$

**Step 7: Off-Diagonal Bound via Cauchy-Schwarz**

For off-diagonal:
$$
|g_{ij,kl}^{\text{Fisher}}| = |\text{Cov}(A_{ij}, A_{kl})| \leq \sqrt{\text{Var}(A_{ij})} \sqrt{\text{Var}(A_{kl})} = O(1/N)
$$

**Step 8: Eigenvalue Bound**

Fisher metric is $N^2 \times N^2$ matrix (one index for each pair $(i,j)$).

All entries $O(1/N)$ → Gershgorin circle theorem:
$$
\lambda_{\max}(g^{\text{Fisher}}) \leq \max_i \left(g_{ii} + \sum_{j \neq i} |g_{ij}|\right) \leq O(1/N) + N^2 \cdot O(1/N) = O(N)
$$

Wait, this gives $O(N)$, not $O(1/N)$!

**Critical Insight**: Need to use **trace normalization** properly.

The trace of Fisher metric is:
$$
\text{Tr}(g^{\text{Fisher}}) = \sum_{ij} \text{Var}(A_{ij}) = O(N^2 / N) = O(N)
$$

If matrix is approximately diagonal (weak correlations), then:
$$
\lambda_{\max} \approx \frac{\text{Tr}}{N^2} = O(1/N)
$$

Rigorously: By Poincaré inequality (framework Theorem thm-qsd-poincare-rigorous):
$$
|\text{Cov}(A_{ij}, A_{kl})| \leq C_P \sqrt{\int |\nabla A_{ij}|^2} \sqrt{\int |\nabla A_{kl}|^2}
$$

Gradient localization (edge weight depends only on 2 walkers):
$$
\int |\nabla A_{ij}|^2 = O(1/N)
$$

Therefore:
$$
|g_{ij,kl}| \leq C_P \cdot (1/\sqrt{N}) \cdot (1/\sqrt{N}) = O(1/N)
$$

Maximum eigenvalue (sum over row):
$$
\lambda_{\max} \leq O(1/N) + N^2 \cdot O(1/N) = O(N)
$$

Hmm, still getting $O(N)$. Need to think more carefully about the structure...

**Revised Step 8**: Use that most off-diagonals are SMALL

The Poincaré bound $O(1/N)$ is for **adjacent** edges (sharing a walker). For **non-adjacent** edges (disjoint walkers):
$$
|\text{Cov}(A_{ij}, A_{kl})| \leq \exp(-c \cdot d(i,k)) \cdot O(1/N)
$$

where $d(i,k)$ is distance between walkers.

By exchangeability, typical distance $\sim N^{1/d}$, giving exponential suppression.

Number of "close" pairs: $O(N)$ (neighbors within distance $O(1)$)

Therefore, effective row sum:
$$
\lambda_{\max} \leq O(1/N) + N \cdot O(1/N) = O(1)
$$

After accounting for normalization factor $1/\sqrt{N}$ in matrix entries:
$$
\lambda_{\max}(g^{\text{Fisher}}) = O(1/N)
$$

$\square$
:::

**Status**: This proof is incomplete. The eigenvalue bound needs more careful analysis. The key issue is properly accounting for:
1. Sparse structure of correlations (most pairs don't overlap)
2. Exponential decay with distance for non-overlapping pairs
3. Normalization constraints

Let me flag this for Gemini review before continuing.

---

## Status and Next Steps

**What Works**:
- ✅ Cumulant-Fisher metric identity (rigorously proven)
- ✅ Framework theorems (HWI, Talagrand, LSI) all valid
- ✅ Conceptual strategy (avoid walkers, use information geometry)

**What Needs Work**:
- ⚠️ Eigenvalue bound on Fisher metric (incomplete)
- ⚠️ Need to properly use sparse correlation structure
- ⚠️ May need to combine with antichain decomposition (Strategy 2)

**Recommendation**: Submit to Gemini for guidance on completing the eigenvalue bound.
