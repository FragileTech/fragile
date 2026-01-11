# Introduction: The Market as a Bounded-Rationality Controller

We treat the market as an **open control system** operating under partial observability, finite information capacity, and institutional constraints. Agents are controllers; the market as a whole is a **coupled dynamical system** that must remain **self-consistent** with its own pricing rules.

## Definitions: Interaction Under Partial Observability

**Definition 1.1.1 (Market Controller).** The market has internal state
$$
Z_t := (K_t, Z_{n,t}, Z_{\mathrm{tex},t}) \in \mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}},
$$
where:
- $K_t$ is a **discrete macro state** (regimes, liquidity state, risk-on/off),
- $Z_{n,t}$ is **structured nuisance** (microstructure, seasonal effects, inventory),
- $Z_{\mathrm{tex},t}$ is **texture residual** (high-frequency noise, idiosyncratic features).

**Definition 1.1.2 (Boundary / Market Interface).** The boundary variables at time $t$ are:
$$
B_t := (x_t, y_t, p_t, d_t, f_t, m_t, a_t),
$$
where:
- $x_t$ is public information (macro data, news),
- $y_t$ is microstructure data (order flow, quotes, depth),
- $p_t$ are observed prices,
- $d_t$ are observed cash flows (dividends, coupons, funding),
- $f_t$ are funding and collateral rates,
- $m_t$ are margin and constraint signals,
- $a_t$ is the aggregate action (net demand / rebalancing flow).

**Definition 1.1.3 (Market as Input-Output Law).** The market environment is a conditional law of future boundary signals given boundary history and actions:
$$
P_{\partial}(B_{t+1} \mid B_{\le t}, a_{\le t}).
$$
In the Markov case this reduces to $P_{\partial}(B_{t+1} \mid B_t, a_t)$, but the interpretation is the same: **pricing and stability depend only on observable boundary signals**.

## Symmetries and Gauge Freedoms

Pricing is invariant under certain transformations; these are **gauge freedoms** in the market description.

**Definition 1.2.1 (Market symmetry group).** A minimal symmetry group is
$$
\mathcal{G}_{\mathbb{M}} := G_{\text{numeraire}} \times S_{|\mathcal{A}|} \times G_{\text{measure}} \times G_{\text{unit}},
$$
where:
- $G_{\text{numeraire}}$ is positive scaling of the unit of account,
- $S_{|\mathcal{A}|}$ permutes asset labels,
- $G_{\text{measure}}$ is change of measure equivalent under the SDF,
- $G_{\text{unit}}$ rescales data units (volatility, notional).

**Principle of covariance.** Market diagnostics and permits should be invariant under $\mathcal{G}_{\mathbb{M}}$, so that pricing conclusions do not depend on arbitrary units or relabeling.

## Market Category Theory: The Ambient Topos

We embed market dynamics in a **cohesive $(\infty,1)$-topos** $\mathcal{E}$ following the categorical foundations of `hypopermits_jb.md`. This provides the mathematical universe where pricing objects live.

**Definition 1.3.1 (Cohesive Market Topos).** The market topos $\mathcal{E}_{\text{mkt}}$ is a cohesive $(\infty,1)$-topos equipped with the adjoint quadruple:
$$
\Pi \dashv \flat \dashv \sharp \dashv \text{coDisc} : \mathcal{E}_{\text{mkt}} \to \infty\text{-Grpd},
$$
where:
- **$\Pi$ (Shape):** extracts the homotopy type of market configurations (e.g., connected components of trading networks, fundamental group of arbitrage cycles),
- **$\flat$ (Flat/Discrete):** embeds constant sheaves; distinguishes pointwise (spot) pricing from derived structures,
- **$\sharp$ (Sharp/Codiscrete):** contractible path spaces; enables continuous deformation of pricing strategies.

**Interpretation.** The cohesive structure allows us to distinguish:
1. **Discrete aspects:** individual trades, settlement events, default triggers.
2. **Continuous aspects:** smooth price evolution, gradual regime shifts.
3. **Homotopical aspects:** topologically distinct market states (e.g., normal vs. crisis regimes connected by paths vs. separated by barriers).

**Definition 1.3.2 (Market Object in Topos).** A market configuration is an object $\mathcal{M} \in \mathcal{E}_{\text{mkt}}$ such that:
$$
\pi_0(\mathcal{M}) = \text{market regimes (discrete states)}, \quad \pi_1(\mathcal{M}) = \text{arbitrage cycles (gauge symmetries)},
$$
$$
\pi_n(\mathcal{M}) = \text{higher anomalies and obstructions for } n \ge 2.
$$

**Remark 1.3.3 (Why Category Theory?).** The categorical framing provides:
1. **Universality:** pricing theorems become natural transformations, not ad-hoc formulas.
2. **Compositionality:** complex instruments are built from simpler ones via colimits.
3. **Invariance:** gauge-independent statements are morphisms in the topos.

## Cohomological Height: Wealth as Derived Functor

Market wealth is not a number but a **derived functor** measuring the "cohomological height" of a position.

**Definition 1.4.1 (Wealth Functor).** Define the wealth functional as a derived functor:
$$
\Phi_{\bullet} : \mathcal{E}_{\text{mkt}} \to \text{Ch}(\mathbb{R}),
$$
where $\text{Ch}(\mathbb{R})$ is the derived category of real-valued chain complexes. The degree-$n$ component $\Phi_n$ measures:
- **$\Phi_0$:** Mark-to-market value (0th homology = direct valuation).
- **$\Phi_1$:** Contingent claims and options (1st homology = linear exposure).
- **$\Phi_2$:** Convexity and gamma exposure (2nd homology = curvature risk).
- **$\Phi_n$:** Higher-order Greeks and exotic path dependencies.

**Definition 1.4.2 (Euler Characteristic of a Portfolio).** The total economic value is the alternating sum:
$$
\chi(\Phi_{\bullet}) := \sum_{n=0}^{\infty} (-1)^n \text{rank}(\Phi_n) = \text{Net Present Value} - \text{Optionality} + \text{Convexity} - \cdots
$$

**Theorem 1.4.3 (Cohomological Pricing).** Under no-arbitrage, the Euler characteristic is preserved under gauge-equivalent portfolio transformations:
$$
\chi(\Phi_{\bullet}(\mathcal{P})) = \chi(\Phi_{\bullet}(g \cdot \mathcal{P})) \quad \forall g \in \mathcal{G}_{\mathbb{M}}.
$$

*Proof Sketch.* Gauge transformations act as quasi-isomorphisms on the chain complex; Euler characteristic is a homotopy invariant. $\square$

## Modalities: Shape, Flat, and Sharp for Markets

The three modalities $\Pi, \flat, \sharp$ give distinct "views" of market data.

**Definition 1.5.1 (Shape Modality $\Pi$: Topological Market Structure).**
$$
\Pi(\mathcal{M}) = \text{homotopy type of market configuration space}.
$$
- **Application:** Detects whether two market states are "topologically equivalent" (connected by continuous deformation) or "topologically distinct" (separated by phase transition).
- **Observable:** Number of connected components = number of distinct regimes.

**Definition 1.5.2 (Flat Modality $\flat$: Spot Pricing).**
$$
\flat(\mathcal{M}) = \text{discrete/pointwise evaluation of prices}.
$$
- **Application:** Spot prices, mark-to-market, instantaneous valuation.
- **Contrast with $\sharp$:** $\flat$ ignores path dependence; $\sharp$ includes it.

**Definition 1.5.3 (Sharp Modality $\sharp$: Path-Dependent Pricing).**
$$
\sharp(\mathcal{M}) = \text{contractible deformation space of price paths}.
$$
- **Application:** Path-dependent options (Asian, barrier, lookback), accumulated dividends, accrued interest.
- **Mathematical structure:** $\sharp(\mathcal{M})$ has trivial homotopy groups—all paths are equivalent up to endpoints.

**Proposition 1.5.4 (Modal Decomposition of Pricing).** Any pricing functional $P$ decomposes as:
$$
P = P_{\flat} + P_{\sharp - \flat} + P_{\Pi},
$$
where:
- $P_{\flat}$ is the spot/intrinsic value,
- $P_{\sharp - \flat}$ is the path-dependent premium (time value, optionality),
- $P_{\Pi}$ is the topological risk premium (regime/crisis premium).

## The Trinity of Market Manifolds

We distinguish three geometric objects:

| Manifold | Symbol | Coordinates | Metric | Role |
|----------|--------|-------------|--------|------|
| **Price/Data Space** | $\mathcal{P}$ | $(p^1, \ldots, p^n)$ | Euclidean | Raw observed prices |
| **State/Risk Space** | $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$ | $(K, z_n)$ | Ruppeiner $G_{ij}(z)$ | Control-relevant states |
| **Parameter Space** | $\Theta$ | $(\theta^1, \ldots, \theta^m)$ | Fisher-Rao $\mathcal{F}(\theta)$ | Model parameters |

**Warning 1.6.1 (Category Error).** The Fisher-Rao metric on parameter space $\Theta$ is **not** the same as the Ruppeiner metric on state space $\mathcal{Z}$. Confusing these leads to:
- Incorrect risk attribution,
- Spurious hedging recommendations,
- Violation of coordinate invariance.

**Definition 1.6.2 (Ruppeiner Risk Metric).** The state-space metric is:
$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{\partial^2 F}{\partial z^i \partial z^j} \cdot \frac{1}{T},
$$
where $S$ is entropy, $F$ is free energy, and $T$ is risk temperature. This measures the **thermodynamic distance** between market states.

## Agent Types and Market Roles

Markets contain heterogeneous agents with distinct control objectives.

**Definition 1.7.1 (Agent Taxonomy).**

| Agent Type | Objective | Time Horizon | Key Constraint |
|------------|-----------|--------------|----------------|
| **Market Maker** | Minimize inventory risk | Intraday | Spread ≥ cost |
| **Arbitrageur** | Exploit mispricings | Seconds–days | Capital limits |
| **Hedger** | Minimize variance | Weeks–years | Basis risk |
| **Speculator** | Maximize expected return | Days–months | Drawdown limits |
| **Index Fund** | Track benchmark | Continuous | Tracking error |
| **Central Bank** | Stability | Permanent | Political mandate |

**Definition 1.7.2 (Aggregate Market Dynamics).** The market evolution $S_t$ is the composition of agent-level dynamics:
$$
S_t = \bigcirc_{j \in \mathcal{J}} S_t^{(j)},
$$
where $\mathcal{J}$ indexes active agents and $\bigcirc$ denotes composition under market clearing.

---

