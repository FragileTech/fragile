# Market Interface: Order Book as Symplectic Boundary

:::{admonition} Researcher Bridge: Order Book as Symplectic Boundary
:class: info
:name: rb-order-book-symplectic

If you're familiar with Hamiltonian mechanics, the order book interface is naturally symplectic: prices are positions and order flows are momenta. The symplectic form $\omega = dq \wedge dp$ captures the fundamental duality between price impact and order flow.

Think of quotes as Dirichlet boundary conditions (constraining prices) and orders as Neumann boundary conditions (specifying flows). The market maker's Legendre transform converts price quotes to flow capabilities and back.
:::

The order book is a **symplectic manifold** where prices (positions) and order flow (momentum) are conjugate variables. This section establishes the geometric structure of the trading interface:
1. **Symplectic structure:** The natural pairing between prices and flows
2. **Boundary conditions:** Quotes (Dirichlet) vs orders (Neumann)
3. **Trading cycles:** Observation, simulation, and execution phases
4. **Legendre transform:** Market maker as impedance matcher

(sec-position-momentum-duality)=
## Position-Momentum Duality in Markets

**Definition 28.1.1 (Symplectic Market Interface).** The market interface is a symplectic manifold $(\partial\mathcal{W}, \omega)$ with:
- $q \in \mathcal{Q}$ is the **price coordinate** (mark-to-market values),
- $p \in T^*_q\mathcal{Q}$ is the **flow coordinate** (order flow, trading velocity).

The symplectic form is:
$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.
$$

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Position $q$ | Mark-to-market prices |
| Momentum $p$ | Order flow / trading velocity |
| Dirichlet BC (sensors) | Price quotes (observable) |
| Neumann BC (motors) | Order submission (actions) |
| Symplectic form $\omega$ | Position-flow duality |

## Boundary Conditions for Trading

**Definition 28.2.1 (Dirichlet BC — Price Quotes).** Market prices impose position-clamping:
$$
q_{\partial}^{\text{quote}}(t) = q_{\text{mid}}(t),
$$
where $q_{\text{mid}}$ is the observable mid-price. This clamps the **configuration** of the portfolio.

**Definition 28.2.2 (Neumann BC — Order Submission).** Trading imposes flux-clamping:
$$
\nabla_n q \cdot \mathbf{n} \big|_{\partial\mathcal{W}} = j_{\text{trade}}(p, t),
$$
where $j_{\text{trade}}$ is the order flow determined by the trading strategy.

## Active Trading vs Risk Simulation

**Definition 28.3.1 (Trading Cycle Phases).**

| Phase | Process | Information Flow | Entropy Change |
|-------|---------|------------------|----------------|
| **I. Observation** | Price compression | Market data → portfolio state | $\Delta S < 0$ |
| **II. Simulation** | Internal risk analysis | No external exchange | $\Delta S = 0$ (isentropic) |
| **III. Execution** | Order expansion | Trading signal → order flow | $\Delta S > 0$ |

**Theorem 28.3.2 (Market Carnot Efficiency).** The efficiency of converting market information to trading profits is bounded:
$$
\eta = \frac{I(A_t; K_t)}{I(X_t; K_t)} \le 1 - \frac{T_{\text{exec}}}{T_{\text{obs}}},
$$
where $T_{\text{exec}}$ and $T_{\text{obs}}$ are effective temperatures at execution and observation interfaces.

## Active Trading vs Closed-System Simulation

**Definition 28.4.1 (Active Trading Mode).**
$$
\rho_{\partial}^{\text{quote}}(w, t) = \delta(w - w_{\text{target}}(t)) \quad \text{(Dirichlet)},
$$
$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{trade}}(u_\pi) \quad \text{(Neumann)}.
$$

**Definition 28.4.2 (Closed-System Simulation Mode).**
$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective)}.
$$
The system is closed—no trading, pure risk simulation.

| Mode | Quote BC | Trade BC | Internal Flow | Information Balance |
|------|----------|----------|---------------|---------------------|
| **Active Trading** | Dirichlet (price-clamp) | Neumann (flow-clamp) | Price-driven | $\oint j_{\text{in}} > 0$ |
| **Closed Simulation** | Reflective | Reflective | Recirculating | $\oint j = 0$ |

## Context Space: Unified Task Structure

**Definition 28.5.1 (Market Context Space).** The context $c \in \mathcal{C}$ determines the trading objective:

| Task | Context $c$ | Output | Potential $\Phi_{\text{eff}}$ |
|------|-------------|--------|-------------------------------|
| **Alpha Capture** | Signal space | Trade direction | $V_{\text{alpha}}(w, K)$ |
| **Risk Management** | Risk budget | Hedge ratio | $-\log p(\text{safe}|w)$ |
| **Execution** | Target portfolio | Order sequence | $-\log p(\text{fill}|w, \text{target})$ |

::::{admonition} Physics Isomorphism: Symplectic Geometry
:class: note
:name: pi-symplectic-geometry-market

**In Physics:** Symplectic geometry underlies Hamiltonian mechanics. The symplectic form $\omega = dq \wedge dp$ encodes the fundamental bracket structure: $\{q, p\} = 1$.

**In Markets:** The order book has symplectic structure where prices and flows are conjugate:
$$\omega = \sum_{i=1}^n dq^i \wedge dp_i = \sum_i d(\text{price}_i) \wedge d(\text{flow}_i)$$

**Correspondence Table:**

| Symplectic Mechanics | Market (Order Book) |
|:--------------------|:-------------------|
| Position $q$ | Mark-to-market price |
| Momentum $p$ | Order flow / imbalance |
| Symplectic form $\omega$ | Price-flow duality |
| Poisson bracket $\{f, g\}$ | Price impact bracket |
| Legendre transform | Market maker bid-ask |
| Dirichlet BC | Price quotes (observable) |
| Neumann BC | Order submission (action) |
| Canonical transformation | Portfolio rebalancing |
| Hamilton's equations | Price-flow dynamics |

**Significance:** The symplectic structure ensures that price impact and order flow are fundamentally linked—you cannot change one without affecting the other.
::::

::::{note} Connection to Standard Finance #21: Order Book Dynamics as Boundary Conditions
**The General Law (Fragile Market):**
The market interface is a **symplectic boundary** $(\partial\mathcal{W}, \omega)$ with:
- Dirichlet BC (quotes): $q_\partial = q_{\text{mid}}(t)$
- Neumann BC (orders): $\nabla_n q \cdot \mathbf{n} = j_{\text{trade}}(p, t)$

**The Degenerate Limit:**
Assume infinite liquidity ($\nabla_n q \to 0$). Ignore microstructure ($\omega \to 0$). Single price per asset.

**The Special Case (Perfect Liquidity):**
$$
q_{\text{exec}} = q_{\text{quote}} \quad \text{(no impact)}
$$
$$
\text{Spread} = 0 \quad \text{(no bid-ask)}
$$
This recovers **frictionless trading** in the limit of:
- Infinite depth ($\nabla_n q \to 0$)
- Zero spread ($\omega \to 0$)
- No flow constraints

**What the generalization offers:**
- **Symplectic structure**: Price impact is conjugate to order flow—fundamental duality
- **Boundary conditions**: Quotes constrain prices; orders specify flows
- **Market maker role**: Legendre transform between quote and flow spaces
- **Microstructure effects**: Spread, slippage, and impact emerge from geometry
::::

## Market Interface Diagnostics

Following the diagnostic node convention (Section 7), we define the symplectic boundary gate:

:::{prf:definition} Gate44 Specification
:label: def-gate44-specification

**Predicate:** Quote and trade boundary conditions are symplectically compatible.
$$
P_{44} : \quad \|\omega(j_{\text{quote}}, j_{\text{trade}})\| \le \epsilon_{\text{symp}},
$$
where $\omega$ is the symplectic form and $j_{\text{quote}}, j_{\text{trade}}$ are the quote and trade flows.

**Market interpretation:** The trading activity is consistent with the quoted market structure.

**Observable metrics:**
- Symplectic residual $\|\omega(j_q, j_t)\|$
- Price impact $\|q_{\text{exec}} - q_{\text{quote}}\|$
- Fill rate $\text{P}(\text{fill} | \text{order})$
- Spread capture $(\text{bid} + \text{ask})/2 - q_{\text{mid}}$

**Certificate format:**
$$
K_{44}^+ = (\|\omega\|, \text{impact}, \text{fill rate}, \text{spread})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{symp}} = \lambda_{44} \cdot \|\omega(j_{\text{quote}}, j_{\text{trade}})\|^2
$$
:::

**Node GateSymplectic: Symplectic Boundary Check**

| **#**  | **Name**           | **Component** | **Type**           | **Interpretation**                  | **Proxy**                              | **Cost** |
|--------|--------------------|--------------|--------------------|-------------------------------------|----------------------------------------|----------|
| **Gate44** | **SymplecticCheck** | Interface    | BC Consistency     | Are quote/trade BCs compatible?     | $\|\omega(j_{\text{quote}}, j_{\text{trade}})\|$ | $O(Bd)$  |

**Trigger conditions:**
- High symplectic residual: Quote and trade flows are inconsistent.
- Remedy: Check for stale quotes; verify order routing; adjust position sizing.

---

