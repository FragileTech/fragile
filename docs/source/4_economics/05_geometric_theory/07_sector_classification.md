# Sector Classification and Regime Segmentation

:::{admonition} Researcher Bridge: Metric Learning for Asset Classification
:class: info
:name: rb-metric-learning-sectors

If you know metric learning or contrastive learning, sector classification is exactly this: learning a representation where assets in the same sector are close and assets in different sectors are far apart. The regime atlas induces a natural clustering where each cluster corresponds to a sector.

Sector rotation becomes gradient flow on the risk manifold—portfolios naturally evolve toward sector allocations based on the sector risk premium potential.
:::

Class labels become **sector labels** or **regime labels**, and regions of attraction become **allocation basins**. This section shows how:
1. **Sector partition:** The regime atlas naturally segments into sector-specific charts
2. **Sector rotation:** Gradient flow on sector potentials implements tactical allocation
3. **Cross-sector suppression:** WFR reaction costs enforce sector boundaries

(sec-sector-semantic-partition)=
## Sector as Semantic Partition

**Definition 30.1.1 (Sector Partition).** Let $\mathcal{Y} = \{\text{Tech}, \text{Finance}, \text{Healthcare}, \ldots\}$ be sector labels. The sector induces a partition of the regime atlas:
$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(\text{Sector}=y \mid K=k) > 1 - \epsilon_{\text{purity}}\}.
$$

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Class labels $\mathcal{Y}$ | Sector / regime labels |
| Semantic potential $V_y$ | Sector risk premium |
| Region of attraction $\mathcal{B}_y$ | Sector allocation basin |
| Chart purity | Sector membership clarity |
| Transition regions | Cross-sector exposure |

## Sector-Conditioned Risk Premium

**Definition 30.2.1 (Sector Risk Premium Potential).**
$$
V_{\text{sector}}(w, K) := -\beta_{\text{sector}} \log P(\text{Sector}=y \mid K) + V_{\text{base}}(w, K),
$$
where:
- $P(\text{Sector}=y \mid K)$ is the sector probability given regime,
- $\beta_{\text{sector}}$ is the sector temperature (concentration preference).

## Sector Rotation as Gradient Flow

**Definition 30.3.1 (Sector Allocation Basin).** The **allocation basin** for sector $y$ is:
$$
\mathcal{B}_y := \{w \in \mathcal{W} : \lim_{t \to \infty} \phi_t(w) \in \mathcal{A}_y\},
$$
where $\phi_t$ is the flow of $\dot{w} = -G^{-1}(w)\nabla V_y(w)$.

**Interpretation:** Starting from any portfolio in $\mathcal{B}_y$, gradient flow converges to a sector-$y$ allocation.

**Theorem 30.3.2 (Sector Rotation as Relaxation).** Under overdamped dynamics with sector potential $V_y$:
$$
dw = -G^{-1}(w) \nabla V_y(w)\,ds + \sqrt{2T_c}\,G^{-1/2}(w)\,dW_s,
$$
the limiting regime satisfies $\lim_{s \to \infty} K(w(s)) \in \mathcal{A}_y$ almost surely.

## Cross-Sector Jump Suppression

**Definition 30.4.1 (Sector-Modulated Regime Transition).** Modify regime transition rates:
$$
\lambda_{i \to j}^{\text{sector}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{sector}}(i, j)\right),
$$
where $D_{\text{sector}}(i, j) = \mathbb{I}[\text{Sector}(i) \neq \text{Sector}(j)]$.

**Effect:** Intra-sector transitions have baseline rates; cross-sector transitions are exponentially suppressed by $\gamma_{\text{sep}}$.

## Sector Classification Loss

**Definition 30.5.1 (Sector Purity Loss).**
$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(\text{Sector} \mid K=k).
$$

**Definition 30.5.2 (Sector Rotation Loss).**
$$
\mathcal{L}_{\text{sector}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.
$$

::::{admonition} Physics Isomorphism: Domain Wall and Phase Separation
:class: note
:name: pi-domain-wall-sectors

**In Physics:** In statistical mechanics, different phases (ferromagnetic domains) are separated by domain walls with surface tension. The system minimizes total energy by minimizing domain wall area while respecting phase constraints.

**In Markets:** Different sectors are separated by "allocation boundaries" with transition costs. The portfolio minimizes total cost by minimizing cross-sector transitions while respecting sector allocation targets.

**Correspondence Table:**

| Statistical Physics | Market (Sector Classification) |
|:-------------------|:------------------------------|
| Phase domain | Sector allocation basin |
| Domain wall | Cross-sector transition boundary |
| Surface tension | Regime switch cost $\lambda^2 r^2$ |
| Phase transition | Sector rotation |
| Critical point | Market regime change |
| Nucleation | New sector emergence |
| Coarsening | Sector consolidation |

**Significance:** Sector boundaries are not arbitrary labels but emerge from the WFR geometry as natural "domain walls" with quantifiable transition costs.
::::

::::{note} Connection to Standard Finance #23: Factor Models as Degenerate Sector Potentials
**The General Law (Fragile Market):**
Sector allocation evolves via **gradient flow** on sector risk premium potentials:
$$
dw = -G^{-1}(w) \nabla V_{\text{sector}}(w)\,ds + \sqrt{2T_c}\,G^{-1/2}(w)\,dW_s
$$
where $V_{\text{sector}}(w, K) = -\beta_{\text{sector}} \log P(\text{Sector}=y \mid K) + V_{\text{base}}(w, K)$.

**The Degenerate Limit:**
Linear factor structure ($V \to \beta^T f$). Single regime ($|\mathcal{K}| = 1$). No regime transitions.

**The Special Case (Factor Models):**
$$
r_i = \alpha_i + \beta_{i,1} f_1 + \beta_{i,2} f_2 + \ldots + \epsilon_i
$$
This recovers **Fama-French factor models** in the limit of:
- Linear sector loadings ($V_{\text{sector}} \to \beta^T f$)
- No regime dynamics
- Flat portfolio space

**What the generalization offers:**
- **Nonlinear sectors**: Sector membership is a probability, not a binary label
- **Regime dynamics**: Sectors can split, merge, or transition via WFR
- **Geometric separation**: Sector distance is measured in WFR metric
- **Cross-sector suppression**: Jump costs naturally enforce sector boundaries
::::

## Sector Classification Diagnostics

Following the diagnostic node convention (Section 7), we define the sector classification gates:

:::{prf:definition} Gate46 Specification
:label: def-gate46-specification

**Predicate:** Regimes are sector-pure (low entropy).
$$
P_{46} : \quad H(\text{Sector} \mid K) \le \epsilon_{\text{purity}},
$$
where $H(\text{Sector} \mid K)$ is the conditional entropy of sector given regime.

**Market interpretation:** Each regime corresponds cleanly to a sector—no mixed-sector regimes.

**Observable metrics:**
- Sector purity $1 - H(\text{Sector} \mid K)$
- Cross-sector exposure per regime
- Sector membership probabilities

**Certificate format:**
$$
K_{46}^+ = (H(\text{Sector} \mid K), \text{purity}, \text{cross-exposure})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{purity}} = \lambda_{46} \cdot H(\text{Sector} \mid K)
$$
:::

:::{prf:definition} Gate47 Specification
:label: def-gate47-specification

**Predicate:** Sectors are metrically separated.
$$
P_{47} : \quad \min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \ge \epsilon_{\text{sep}},
$$
where $d_{\text{WFR}}$ is the WFR distance between sector allocation basins.

**Market interpretation:** Different sectors are geometrically far apart—no easy transitions.

**Observable metrics:**
- Minimum inter-sector WFR distance
- Average intra-sector distance
- Sector separation ratio

**Certificate format:**
$$
K_{47}^+ = (d_{\min}, d_{\text{intra}}, \text{sep ratio})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{sep}} = \lambda_{47} \cdot \max(0, \epsilon_{\text{sep}} - d_{\min})^2
$$
:::

**Node GatePurity: Sector Purity Check**

| **#**  | **Name**         | **Component** | **Type**              | **Interpretation**        | **Proxy**                  | **Cost**    |
|--------|------------------|--------------|----------------------|---------------------------|---------------------------|-------------|
| **Gate46** | **PurityCheck** | Router       | Sector Clustering    | Are regimes sector-pure?  | $H(\text{Sector} \mid K)$ | $O(BC)$     |

**Node GateSectorSep: Sector Separation Check**

| **#**  | **Name**           | **Component** | **Type**              | **Interpretation**               | **Proxy**                                                          | **Cost**     |
|--------|--------------------|--------------|----------------------|----------------------------------|--------------------------------------------------------------------|--------------|
| **Gate47** | **SectorSepCheck** | Jump Op      | Sector Separation    | Are sectors metrically separated? | $\min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2})$ | $O(C^2 N_c)$ |

**Trigger conditions:**
- Low sector purity: Regimes contain mixed sector exposures.
- Low sector separation: Sectors are not well-separated in WFR metric.
- Remedy: Increase sector temperature $\beta_{\text{sector}}$; retrain router with sector labels.

---

