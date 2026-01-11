# Market reference (no proofs)

This file is a **crosswalk**: it imports the **Fragile Agent mathematical machinery**
(`docs/source/1_agent/`) and re-reads it as a **market/economy theory**.

## Dictionary (Agent → Market)

| Fragile Agent object | Finance interpretation | Where in `4_economics/` |
|---|---|---|
| Internal state $Z_t=(K_t,Z_{n,t},Z_{\mathrm{tex},t})$ | Market state: regimes $K_t$, structured frictions/microstructure $Z_{n,t}$, high-rate residuals $Z_{\mathrm{tex},t}$ | `docs/source/4_economics/01_foundations/02_market_controller.md` |
| Boundary tuple $B_t$ | Observable market interface: prices, order flow, cashflows, funding, constraints | `docs/source/4_economics/01_foundations/02_market_controller.md` |
| Environment kernel $P_{\\partial}$ | The market as an input-output law over boundary signals (stylized “price formation law”) | `docs/source/4_economics/01_foundations/02_market_controller.md` |
| Value $V$ / reward field | Value field / pricing kernel driver; “what the market is optimizing/penalizing” | `docs/source/4_economics/05_geometric_theory/06_pricing_kernel.md` |
| Capacity-constrained metric $G$ | Risk geometry (local conditioning, leverage, liquidity curvature) | `docs/source/4_economics/02_core_pricing/05_risk_measures.md` |
| WFR / unbalanced transport | Portfolio rebalancing with creation/destruction of mass (default, issuance, deleveraging) | `docs/source/4_economics/05_geometric_theory/02_wfr_transport.md` |
| Sieve (gates + barriers) | Model validity & systemic risk supervision: solvency/liquidity/info constraints as checkable permits | `docs/source/4_economics/02_core_pricing/03_market_sieve.md` |
| Gauge symmetry | Numeraire changes, equivalent measure changes, accounting/unit conventions | `docs/source/4_economics/01_foundations/02_market_controller.md` |
| Multi-agent gauge theory | No-arbitrage as flatness; arbitrage loops as curvature/defects; market-making as symmetry breaking | `docs/source/4_economics/05_geometric_theory/05_market_interface.md` |

## Imported machinery (minimal statements)

The economics volume reuses the *agent-side* objects and reads them as market objects:

({prf:ref}`def-bounded-rationality-controller`) **Bounded-Rationality Controller**

Finance reading: “the market” (and each participant) is a bounded controller; the relevant state is
the *representation the system can actually sustain*.

({prf:ref}`def-boundary-markov-blanket`) **Boundary / Markov Blanket**

Finance reading: the only legitimate primitives are observable interface signals (prices, flows,
funding, constraints); “latent fundamentals” must be justified by coupling to boundary data.

({prf:ref}`def-environment-as-generative-process`) **Environment as Generative Process**

Finance reading: the market is not a set of equilibrium equations; it is a **law over boundary
streams** conditioned on actions/flows (including feedback effects).

({prf:ref}`def-agent-symmetry-group-operational`) **Operational symmetry group**

Finance reading: invariances under (i) units/numeraire, (ii) relabelings, (iii) admissible canonical
reparameterizations. Diagnostics should be covariant to these symmetries.

({prf:ref}`def-closure-defect`) **Closure defect**

Finance reading: regime models ($K_t$) are only legitimate if the macro dynamics closes (no hidden
state is “leaking in”); closure defects are a quantitative “model risk” monitor.

({prf:ref}`def-entropy-regularized-objective-functional`) **Entropy-regularized objective**

Finance reading: risk/return trade-offs are free-energy trade-offs; “temperature” encodes the
degree of randomization / liquidity / exploration.

({prf:ref}`thm-the-hjb-helmholtz-correspondence`) **HJB–Helmholtz correspondence**

Finance reading: valuation is a **screened potential problem** (discounting = screening mass); a
value function (or pricing potential) is the solution of a boundary value problem driven by
cashflows and constraints.

## The “standard model” target

The point of importing this machinery is not metaphor: it is a **standard model** that can be
checked and calibrated like physics.

See:
- `docs/source/4_economics/06_standard_model/01_standard_model.md`
- `docs/source/4_economics/06_standard_model/02_economy_as_physical_system.md`

