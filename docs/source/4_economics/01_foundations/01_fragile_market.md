# The Fragile Market: Thermoeconomic Asset Pricing with Hypostructure Permits

## Positioning: Connections, Differences, Advantages

This document is a **full economic theory of asset pricing** built as a **bounded-rationality control system** with **thermoeconomic potentials** and the **hypostructure permit machinery**. It is an explicit synthesis of:
- standard mathematical finance (no-arbitrage, SDF, martingale pricing),
- thermoeconomics (entropy, dissipation, free energy),
- and the **Gate + Barrier Sieve** language of `docs/source/hypopermits_jb.md`.

The contribution is to make dependencies **operational and auditable**: every modeling step is typed as a **permit** with a clear observable predicate and certificate. Prices are therefore not just equilibrium objects, but **verified fixed points** of the market dynamics under information, liquidity, and solvency constraints.

:::{admonition} Researcher Bridge: How this relates to `1_agent/`
:class: info
This volume reuses the **Fragile Agent** machinery (boundary/interface laws, gauge symmetries,
capacity-constrained geometry, WFR transport, and Sieve permits) and reads it as a market theory.
Use `docs/source/4_economics/reference.md` as the dictionary back to `docs/source/1_agent/reference.md`.
:::

### Main Advantages (Why This Framing Is Useful)

1. **Online auditability.** Pricing assumptions become **checkable constraints** (no-arbitrage, solvency, liquidity, information coupling) rather than hidden modeling lore.
2. **Explicit market macro state.** A discrete macro register $K_t$ makes regime changes and risk-state shifts **measurable**, supporting robust pricing and stress testing.
3. **Thermoeconomic clarity.** Risk premia and discounting emerge from a **free-energy principle** that unifies expected utility, entropy, and information costs.
4. **Sieve integration.** Hypostructure permits provide a typed protocol for when prices are valid, when they are merely indicative, and when they are **structurally invalid**.
5. **Asset-type unification.** Equities, rates, credit, commodities, FX, and derivatives are treated as instances of a single pricing kernel with domain-specific constraints.

### Contributions and Foundations

**Core contributions of this framework:**
1. **Market Hypostructure object.** Asset pricing encoded as a hypostructure with explicit boundary coupling and permit-verified transitions.
2. **Thermoeconomic SDF.** The stochastic discount factor linked to entropy production and free-energy minimization.
3. **Sieve-based market validity.** Pricing accepted only if a finite set of **gate + barrier permits** hold (solvency, liquidity, information grounding, etc).
4. **Representation constraints.** The macro register $K_t$ treated as a bounded-rate statistic with closure and capacity checks.
5. **WFR portfolio transport.** The Wasserstein-Fisher-Rao metric unifies continuous rebalancing with discrete regime transitions.
6. **Pricing kernel as Helmholtz solver.** DCF/Bellman equation as screened Poisson equation with discount rate as screening mass.
7. **Symplectic order book interface.** Order book as symplectic manifold with price/flow as conjugate variables, boundary conditions as Dirichlet (quotes) vs Neumann (orders).
8. **Ruppeiner geometry for risk.** Full Ruppeiner metric tensor formalism applied to financial risk metrics.

**Foundational literature:**
- **No-arbitrage + SDF:** Fundamental theorem of asset pricing {cite}`harrison1979martingales,harrison1981martingales,delbaen1994ftap`.
- **Equilibrium asset pricing:** Euler equations, representative agent, factor models {cite}`lucas1978asset,breeden1979intertemporal,cochrane2005asset`.
- **Term structure and derivatives:** Risk-neutral valuation and replication methods {cite}`vasicek1977equilibrium,cox1985theory,heath1992bond,black1973pricing,merton1973theory,hull2018options`.
- **Thermoeconomics:** Entropy, dissipation, and free-energy objectives {cite}`jaynes1957information,cover2006elements`.
- **Optimal transport:** Wasserstein metrics and distributionally robust optimization {cite}`esfahani2018dro,mohajerin2018dro,chizat2018wfr`.
- **Information geometry:** Natural gradients and Riemannian optimization {cite}`amari1998natural,martens2020ngd`.
- **Symplectic economics:** Symplectic geometry as the natural geometry of maximizing behavior {cite}`russell2011symplectic`.
- **PDE methods in finance:** Hamilton-Jacobi-Bellman equations {cite}`forsyth2007numerical,pham2009continuous`.
- **Market microstructure:** Order book dynamics, bid-ask spreads, market impact {cite}`gueant2016microstructure,cartea2015algorithmic`.
- **Thermodynamic geometry:** Ruppeiner geometry for fluctuation theory {cite}`ruppeiner1979thermodynamics`.

### Comparison Snapshot

| Area | Typical baseline | Fragile Market difference |
|---|---|---|
| **Asset pricing** | equilibrium + no-arbitrage | explicit permits + Sieve validation |
| **Risk premia** | statistical factor models | thermoeconomic free-energy decomposition |
| **Market stability** | ad-hoc stress tests | gate/barrier constraints with certificates |
| **Regime modeling** | latent continuous factors | explicit discrete $K_t$ with capacity checks |
| **Microstructure** | separate from macro pricing | boundary interface with explicit coupling |
| **Portfolio rebalancing** | discrete reoptimization | WFR geodesic transport (Sec. 25) |
| **Risk metric** | covariance matrix | Ruppeiner tensor with curvature (Sec. 4, 27) |
| **Valuation PDE** | Black-Scholes / HJB | screened Poisson / Helmholtz (Sec. 29) |
| **Order book** | statistical models | symplectic manifold with BCs (Sec. 28) |
| **Sector allocation** | discrete optimization | gradient flow to attraction basins (Sec. 30) |
| **Capacity constraints** | ad-hoc position limits | information-theoretic area law (Sec. 24) |
| **Price discovery** | efficient markets hypothesis | entropic drift on Poincaré disk (Sec. 26) |

### Axiomatic Foundation

The Fragile Market theory rests on **seven foundational axioms**:

:::{prf:axiom} A1: Bounded Rationality
:label: axiom-bounded-rationality

All market agents operate under finite information capacity and computational constraints:
$$
I(a_t; Z_t) \le C_{\text{agent}} < \infty,
$$
where $a_t$ is agent action, $Z_t$ is market state, and $C_{\text{agent}}$ is agent channel capacity.
:::

:::{prf:axiom} A2: Thermodynamic Consistency
:label: axiom-thermo-consistency

Market dynamics obey the laws of thermodynamics:
1. **First law (Conservation):** Capital is conserved modulo external flows and dissipation.
2. **Second law (Entropy):** Entropy production is non-negative; $\Delta S \ge 0$ for isolated systems.
3. **Third law (Irreversibility):** Finite-time transactions have irreducible friction cost.
:::

:::{prf:axiom} A3: No-Arbitrage
:label: axiom-no-arbitrage

In the absence of barrier breaches, there exists no self-financing strategy yielding positive return with zero risk:
$$
\nexists \theta : V_0(\theta) = 0, \; V_T(\theta) \ge 0 \; \text{a.s.}, \; \mathbb{P}(V_T(\theta) > 0) > 0.
$$
:::

:::{prf:axiom} A4: Positive SDF
:label: axiom-positive-sdf

There exists a strictly positive stochastic discount factor $M_t > 0$ such that:
$$
p_t = \mathbb{E}_t[M_{t+1} \cdot \text{Payoff}_{t+1}].
$$
:::

:::{prf:axiom} A5: Information Grounding
:label: axiom-info-grounding

Prices must be coupled to observable boundary data:
$$
I(p_t; B_t) > 0,
$$
where $B_t$ is the boundary signal (quotes, flows, news).
:::

:::{prf:axiom} A6: Finite Complexity
:label: axiom-finite-complexity

The market state has bounded Kolmogorov complexity:
$$
K(Z_t) \le K_{\max} < \infty.
$$
:::

:::{prf:axiom} A7: Permit Completeness
:label: axiom-permit-completeness

Every market failure mode is detectable by at least one gate or barrier:
$$
\forall \text{ failure } F, \; \exists \text{ permit } P : P(F) = \text{FAIL}.
$$
:::

**Derived Principles.** From these axioms, we derive:
- **MKT-Consistency** (Theorem 16.1) follows from A3, A4.
- **MKT-Exclusion** (Theorem 16.2) follows from A3.
- **MKT-Trichotomy** (Theorem 16.3) follows from A6, A7.
- **MKT-Equivariance** (Theorem 16.4) follows from A4 plus gauge symmetry.
- **MKT-HorizonLimit** (Theorem 16.5) follows from A1, A6.

### Notation Glossary

| Symbol | Meaning | Domain | Reference |
|--------|---------|--------|-----------|
| $Z_t$ | Full market state | $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}}$ | Def. 1.1.1 |
| $K_t$ | Discrete macro state (regime) | $\mathcal{K}$ (finite set) | Def. 1.1.1 |
| $B_t$ | Boundary data | $(x_t, y_t, p_t, d_t, f_t, m_t, a_t)$ | Def. 1.1.2 |
| $M_t$ | Stochastic discount factor | $\mathbb{R}_{>0}$ | Def. 4.4.1 |
| $F_t$ | Free energy | $\mathbb{R}$ (CU) | Def. 4.1.1 |
| $S_t$ | Entropy | $\mathbb{R}_{\ge 0}$ (nats) | Def. 4.1.1 |
| $T_t$ | Risk temperature | $\mathbb{R}_{>0}$ | Def. 4.1.1 |
| $G_{ij}$ | Ruppeiner metric tensor | Positive definite matrix | Def. 4.5.1 |
| $\Phi$ | Risk potential | $\mathbb{R}_{\ge 0}$ | Def. 3.1.1 |
| $K_i^+$ | Gate $i$ certificate (PASS) | Boolean | Sec. 7.1 |
| $K_i^-$ | Gate $i$ certificate (FAIL) | Boolean | Sec. 7.1 |
| $K^{\text{blk}}$ | Barrier blocked | Status | Sec. 7.3 |
| $K^{\text{br}}$ | Barrier breached | Status | Sec. 7.3 |
| $\alpha, \beta, \gamma, \delta$ | Scaling exponents | $\mathbb{R}$ | Def. 4.7.1 |
| $\Psi$ | Phase order parameter | $[0, 1]$ | Def. 4.6.2 |
| $W_2$ | Wasserstein-2 distance | $\mathbb{R}_{\ge 0}$ | Def. 4.12.1 |
| $\mathcal{L}_{\text{Sieve}}$ | Sieve loss function | $\mathbb{R}_{\ge 0}$ | Sec. 18.5 |

**Subscript conventions:**
- $_t$ : time index
- $_i, _j$ : asset or coordinate indices
- $_K$ : regime-conditioned quantity
- $^{\mathbb{Q}}$ : risk-neutral measure
- $^{\mathbb{P}}$ : physical measure

### Document Structure

**Document Structure (30 Sections):**

| Part | Sections | Content |
|------|----------|---------|
| **Foundations** | 0–4 | Positioning, introduction, units, market hypostructure, thermoeconomic foundations |
| **Core Pricing** | 5–10 | Representation constraints, asset pricing core, market sieve, dynamics, risk measures, asset classes |
| **Implementation** | 11–18 | Market understanding, implementation, summary, failure modes, surgery contracts, metatheorems, algorithmic pricing, full implementation |
| **Applications** | 19–23 | Worked examples, summary/cross-refs, calibration, risk attribution, backtesting |
| **Geometric Theory** | 24–30 | Capacity constraints, WFR transport, price discovery, equations of motion, market interface, pricing kernel, sector classification |

**Geometric Theory Sections (24–30):**

| Section | Content |
|---------|---------|
| **24. Capital Capacity** | Information-theoretic position limits via area law; capacity saturation diagnostic |
| **25. WFR Transport** | Unbalanced optimal transport unifies continuous rebalancing with discrete regime switches |
| **26. Price Discovery** | Entropic drift models spread compression; market maker as symmetry-breaking control |
| **27. Equations of Motion** | Portfolio geodesic SDE with Christoffel corrections; BAOAB integrator for risk-aware trading |
| **28. Market Interface** | Order book as symplectic manifold; Dirichlet (quotes) vs Neumann (orders) boundary conditions |
| **29. Pricing Kernel** | DCF as screened Poisson equation; discount rate = screening mass; Green's function valuation |
| **30. Sector Classification** | Sector rotation as gradient flow; allocation basins as regions of attraction |

**Diagnostic Nodes (Gates 40–47):**

| Node | Section | Monitors |
|------|---------|----------|
| Gate40 | §24 | Capacity saturation ratio |
| Gate41 | §25 | WFR continuity consistency |
| Gate42 | §26 | Price discovery convergence |
| Gate43 | §27 | Geodesic trajectory consistency |
| Gate44 | §28 | Symplectic boundary compatibility |
| Gate45 | §29 | Helmholtz/Bellman residual |
| Gate46 | §30 | Sector purity |
| Gate47 | §30 | Cross-sector separation |

---
