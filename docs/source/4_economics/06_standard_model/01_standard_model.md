# A Standard Model of Finance (Fragile Mechanics Form)

This chapter compresses the full `4_economics/` volume into a **standard model**: a small set of
primitive objects, symmetries, and equations that reproduce mainstream finance as limiting cases,
while remaining compatible with **bounded rationality**, **frictions**, and **systemic failures**.

## Primitives

The standard model is specified by the tuple:

$$
\\mathfrak{M}
:=
\\Big(\\mathcal{Z},\\ B_t,\\ P_{\\partial},\\ \\mathcal{G},\\ \\Phi,\\ \\mathfrak{D},\\ G,\\ \\mathcal{S}\\Big),
$$

where:

1. **State (bulk):** a market state manifold/space $\\mathcal{Z}$, typically decomposed as
   $Z_t=(K_t,Z_{n,t},Z_{\\mathrm{tex},t})$ (regime / structured frictions / residuals).
2. **Interface (boundary):** observable boundary signals $B_t$ (prices, order flow, cashflows,
   funding, constraints).
3. **Dynamics (kernel):** a boundary law $P_{\\partial}(B_{t+1}\\mid B_{\\le t},a_{\\le t})$ (“price
   formation law”, with feedback).
4. **Symmetry (gauge):** invariances $\\mathcal{G}$ (numeraire/unit changes, equivalent measure
   changes, relabelings).
5. **Potential:** a risk/value functional $\\Phi$ (what the system “pays” to be in a state).
6. **Dissipation:** a friction functional $\\mathfrak{D}$ (transaction costs, impact, funding frictions).
7. **Metric:** a geometry $G$ encoding local conditioning (risk curvature / liquidity geometry).
8. **Sieve:** a permit system $\\mathcal{S}$ (gates + barriers) defining when the model is valid.

This is the minimum set needed to treat markets as **physical systems**: state, boundary, symmetry,
energy, dissipation, geometry, and admissibility.

## Core equations

### No-arbitrage as flatness

In the idealized frictionless limit (no barriers breached), no-arbitrage is a **flatness**
constraint: pricing is path-independent under admissible transformations (replication/numeraire),
and arbitrage cycles are curvature defects.

### Prices via a discount kernel

The pricing postulate is an SDF / pricing kernel representation:

$$
P_t
=
\\mathbb{E}_t\\big[M_{t,t+1}\\,X_{t+1}\\big],
$$

with $M_{t,t+1}>0$ and $X_{t+1}$ the payoff/cashflow. The economics interpretation is:
- $M$ carries **time preference + risk + funding + constraints**,
- “risk-neutral pricing” is a change of measure induced by $M$,
- deviations from idealized martingale structure are handled as **permit failures**.

### Value as a screened potential

Valuation is a boundary value problem: discounted control/DCF corresponds to a screened potential
equation (HJB–Helmholtz), linking pricing to geometry and boundary cashflows.

See `docs/source/4_economics/05_geometric_theory/06_pricing_kernel.md`.

### Dynamics as transport with dissipation

Rebalancing and market evolution are transport problems on $\\mathcal{Z}$ with non-conservation
(issuance/default/deleveraging). WFR transport provides the canonical geometry for this setting.

See `docs/source/4_economics/05_geometric_theory/02_wfr_transport.md`.

## Classical finance as limits (sanity checks)

The standard model is designed to reduce to familiar theory by dialing frictions/capacity to
special limits:

- **Frictionless complete markets:** $\\mathfrak{D}\\to 0$, barriers inactive $\\Rightarrow$ classical
  replication + risk-neutral pricing.
- **Black–Scholes:** lognormal diffusion, constant coefficients, complete market $\\Rightarrow$ BS PDE.
- **Consumption CAPM / Euler equation:** SDF $M \\propto \\beta\\,u'(C_{t+1})/u'(C_t)$ under a
  representative-agent closure.
- **Factor models:** low-dimensional $K_t$ with approximately closing dynamics yields linear pricing
  relations with macro factors.

## The scientific method: permits as falsification

The Sieve is the “physics lab protocol”:
- **Gates** encode observable prerequisites (liquidity, solvency, grounding, closure).
- **Barriers** encode hard failure modes (default cascades, funding freezes, blow-ups).
- **Surgeries** encode controlled interventions (circuit breakers, margin changes, liquidity facilities).

This turns “assumptions” into **checkable claims**, and model risk into a monitored state variable.

