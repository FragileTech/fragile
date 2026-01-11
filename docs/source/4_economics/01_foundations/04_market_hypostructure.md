# The Market Hypostructure

We now instantiate the hypostructure formalism for markets.

## Market Hypostructure Object

**Definition 3.1.1 (Market Hypostructure).** A market hypostructure is a tuple
$$
\mathbb{H}_{\text{mkt}} = (\mathcal{X}, \nabla, \Phi_{\bullet}, \tau, \partial_{\bullet}),
$$
where:
1. **State stack $\mathcal{X}$:** the configuration stack of balance sheets, contracts, positions, and market microstructure.
2. **Connection $\nabla$:** time evolution under trading, settlement, and policy constraints.
3. **Potential $\Phi_{\bullet}$:** a thermoeconomic potential encoding total utility, risk, and costs.
4. **Truncation structure $\tau$:** market constraints (capital, leverage, liquidity, information capacity, topology of the trading network).
5. **Boundary morphism $\partial_{\bullet}$:** restriction to the market interface $B_t$.

**Interpretation.** $\mathbb{H}_{\text{mkt}}$ is the **object on which pricing lives**. Prices are not intrinsic; they are sections of boundary data consistent with $\nabla$ and $\Phi_{\bullet}$ under $\tau$.

## Self-Consistency Principle

**Definition 3.2.1 (Self-consistent market).** A market trajectory is self-consistent if the evolution $S_t$ preserves all permits and converges to a state where pricing is internally and externally consistent.

**Principle (Market fixed point).** Under strict dissipation and permit satisfaction, persistent market states are fixed points (or invariant sets) of $S_t$.

This is the market analogue of the hypostructure fixed-point principle: **prices that persist must be compatible with their own dynamics and constraints**.

## Thin Inputs and Permit Mapping

The Sieve uses a **thin interface** representation of the market:
- $\mathcal{X}^{\text{thin}} = (X, d, \mu)$: market state space, distance, and observed measure.
- $\Phi^{\text{thin}} = (\Phi, \nabla, \alpha)$: potential, evolution, and curvature.
- $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$: boundary interface, trace map, boundary flux, and risk signal.

These thin inputs are the minimal objects needed to evaluate permits in the market Sieve.

## Thin Market Kernel: Minimal Specification

Following the Thin Kernel Objects of `hypopermits_jb.md` Section 4, we define the minimal market data required for Sieve operation.

**Definition 3.4.1 (Thin Market Kernel).** A thin market kernel is a quintuple:
$$
\mathcal{T}_{\text{mkt}} = (\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}}),
$$
where each component has explicit market interpretation:

**Component 1: Arena $\mathcal{X}^{\text{thin}} = (X, d, \mathfrak{m})$**
- **$X$:** Polish metric space of market configurations (positions, prices, balances).
- **$d$:** Distance function; typically $d(x, x') = \|p - p'\|_2 + \lambda \|\theta - \theta'\|_{\text{param}}$.
- **$\mathfrak{m}$:** Reference measure; empirical distribution of historical market states.

**Component 2: Potential $\Phi^{\text{thin}} = (\Phi, \alpha_{\Phi})$**
- **$\Phi : X \to \mathbb{R}_{\ge 0}$:** Total market risk functional (e.g., aggregate VaR, expected shortfall).
- **$\alpha_{\Phi}$:** Scaling dimension; for equity markets, typically $\alpha_{\Phi} \approx 2$ (quadratic risk).

**Component 3: Dissipation $\mathfrak{D}^{\text{thin}} = (\mathfrak{D}, \beta_{\mathfrak{D}})$**
- **$\mathfrak{D} : X \times X \to \mathbb{R}_{\ge 0}$:** Transaction cost and friction functional.
- **$\beta_{\mathfrak{D}}$:** Scaling dimension; typically $\beta_{\mathfrak{D}} = 1$ (linear in volume) or $3/2$ (with impact).

**Component 4: Symmetry $G^{\text{thin}}$**
- Symmetry group acting on $\mathcal{X}$; at minimum, $G^{\text{thin}} = G_{\text{numeraire}} \times S_{|\mathcal{A}|}$.

**Component 5: Boundary $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$**
- **$\mathcal{B}$:** Boundary data space (observed prices, flows, news).
- **$\mathrm{Tr} : X \to \mathcal{B}$:** Trace map projecting bulk state to boundary observables.
- **$\mathcal{J}$:** Boundary flux (order flow, capital flow).
- **$\mathcal{R}$:** Risk signal (VIX, credit spreads, funding stress).

**Theorem 3.4.2 (Thin Kernel Sufficiency).** Given a thin market kernel $\mathcal{T}_{\text{mkt}}$, the Sieve constructor $F_{\text{Sieve}}$ produces a full market hypostructure:
$$
F_{\text{Sieve}}(\mathcal{T}_{\text{mkt}}) = \mathbb{H}_{\text{mkt}}.
$$

*Proof.* By the Expansion Adjunction (`hypopermits_jb.md` Theorem 5.3), thin kernels promote to full hypostructures via Postnikov tower construction. $\square$

## Market RCD Condition (Curvature-Dimension Bound)

Markets satisfy a **Riemannian Curvature-Dimension** condition that bounds complexity.

**Definition 3.5.1 (Market RCD Condition).** The market state space $(\mathcal{X}, d, \mathfrak{m})$ satisfies $\mathrm{RCD}(K, N)$ if:
1. **Ricci curvature bounded below:** $\mathrm{Ric} \ge K$ (market has limited "negative curvature" / instability).
2. **Dimension bounded above:** $\dim \le N$ (finite degrees of freedom).

**Interpretation.**
- **$K > 0$:** Market has intrinsic stability; perturbations decay exponentially.
- **$K = 0$:** Flat market; perturbations persist (random walk).
- **$K < 0$:** Hyperbolic market; small perturbations amplify (crisis-prone).

**Theorem 3.5.2 (RCD Convergence for Markets).** If the market satisfies $\mathrm{RCD}(K, N)$ with $K > 0$, then:
$$
W_2(\mu_t, \mu_{\infty}) \le e^{-Kt} W_2(\mu_0, \mu_{\infty}),
$$
where $W_2$ is Wasserstein-2 distance and $\mu_{\infty}$ is the equilibrium distribution.

*Market implication:* Prices converge to equilibrium at rate $K$. Higher curvature = faster price discovery.

## Cheeger Energy and Market Liquidity

The **Cheeger energy** connects metric structure to measure structure.

**Definition 3.6.1 (Market Cheeger Energy).**
$$
\mathrm{Ch}(f | \mathfrak{m}) := \frac{1}{2} \inf \left\{ \liminf_{n \to \infty} \int_X |\nabla f_n|^2 \, d\mathfrak{m} : f_n \to f \text{ in } L^2 \right\}.
$$

**Market interpretation:** Cheeger energy measures the **liquidity cost** of moving probability mass (capital) across market states. High Cheeger energy = illiquid transitions.

**Proposition 3.6.2 (Liquidity as Cheeger Constant).** The market liquidity index is:
$$
\mathcal{L}_{\text{mkt}} := \inf_{A : 0 < \mathfrak{m}(A) < 1} \frac{\text{Per}(A)}{\min(\mathfrak{m}(A), 1 - \mathfrak{m}(A))},
$$
where $\text{Per}(A)$ is the perimeter of set $A$ in the metric-measure space.

Low $\mathcal{L}_{\text{mkt}}$ indicates "bottlenecks" where capital cannot flow freely—liquidity traps.

---

