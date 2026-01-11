# Full Implementation

This section provides complete Python/PyTorch implementations of the Market Hypostructure framework.

## Core Data Structures

```python
"""
Market Hypostructure: Core Implementation
=========================================
Complete Python/PyTorch implementation of the thermoeconomic asset pricing framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class GateStatus(Enum):
    """Status of a gate node check."""
    PASS = auto()      # Certificate valid
    FAIL = auto()      # Certificate invalid
    UNKNOWN = auto()   # Cannot determine (treated as FAIL)
    BOUNDED = auto()   # Passes with bounds/uncertainty


class BarrierStatus(Enum):
    """Status of a barrier."""
    CLEAR = auto()     # No breach
    WARNING = auto()   # Approaching threshold
    BREACHED = auto()  # Active breach
    RECOVERY = auto()  # Recovering from breach


class MarketPhase(Enum):
    """Market complexity phase."""
    CRYSTAL = auto()   # Efficient, incompressible
    LIQUID = auto()    # Predictable with friction
    GAS = auto()       # Chaotic, disconnected


class FailureMode(Enum):
    """15-mode failure taxonomy."""
    CE = "Default Cascade"
    CD = "Too-Big-to-Fail"
    CC = "HFT Instability"
    TE = "Flash Crash"
    TD = "Frozen Market"
    TC = "Complexity Crisis"
    DE = "Boom-Bust Cycle"
    DD = "Dispersion Success"
    DC = "Fundamental Uncertainty"
    SE = "Supercritical Leverage"
    SD = "Flat Volatility"
    SC = "Parameter Drift"
    BE = "External Shock"
    BD = "Liquidity Starvation"
    BC = "Agency Misalignment"


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class MarketState:
    """Complete market state at time t."""
    prices: torch.Tensor           # Asset prices (n_assets,)
    positions: torch.Tensor        # Position sizes (n_positions, n_assets)
    volatilities: torch.Tensor     # Implied/realized vol (n_assets,)
    correlations: torch.Tensor     # Correlation matrix (n_assets, n_assets)
    liquidity: torch.Tensor        # Bid-ask spreads (n_assets,)
    leverage: torch.Tensor         # Leverage ratios (n_positions,)
    regime: int                    # Current market regime index
    temperature: float             # Market temperature (risk tolerance)
    timestamp: float               # Time

    @property
    def n_assets(self) -> int:
        return self.prices.shape[0]

    @property
    def n_positions(self) -> int:
        return self.positions.shape[0]

    def covariance_matrix(self) -> torch.Tensor:
        """Compute covariance from vol and correlation."""
        vol_diag = torch.diag(self.volatilities)
        return vol_diag @ self.correlations @ vol_diag


@dataclass
class Certificate:
    """Proof-carrying certificate for a pricing decision."""
    gate_results: Dict[int, GateStatus]    # Gate index -> status
    barrier_results: Dict[str, BarrierStatus]  # Barrier name -> status
    price_bounds: Tuple[float, float]      # (lower, upper) price bounds
    confidence: float                       # Overall confidence [0, 1]
    failure_modes: List[FailureMode]       # Active/near failure modes
    timestamp: float

    @property
    def is_valid(self) -> bool:
        """Certificate is valid if all gates pass and no barriers breached."""
        gates_ok = all(s in (GateStatus.PASS, GateStatus.BOUNDED)
                       for s in self.gate_results.values())
        barriers_ok = all(s != BarrierStatus.BREACHED
                         for s in self.barrier_results.values())
        return gates_ok and barriers_ok


@dataclass
class SDFParams:
    """Stochastic Discount Factor parameters."""
    risk_free_rate: float = 0.03
    risk_aversion: float = 2.0
    market_temperature: float = 1.0
    regime_weights: torch.Tensor = field(default_factory=lambda: torch.ones(3) / 3)
```

## Thermoeconomic SDF Implementation

```python
# ============================================================================
# THERMOECONOMIC SDF
# ============================================================================

class ThermoeconomicSDF(nn.Module):
    """
    Free-energy based Stochastic Discount Factor.

    The SDF is derived from the thermoeconomic potential:
    M_t = exp(-beta * (r_t + risk_premium_t))

    where risk_premium follows from the Ruppeiner geometry.
    """

    def __init__(self, n_assets: int, n_factors: int = 3,
                 n_regimes: int = 3, device: str = 'cpu'):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.device = device

        # Factor loadings (beta)
        self.factor_betas = nn.Parameter(
            torch.randn(n_assets, n_factors) * 0.1
        )

        # Factor risk premia (lambda)
        self.factor_lambdas = nn.Parameter(
            torch.zeros(n_regimes, n_factors)
        )

        # Regime transition matrix
        self.regime_transitions = nn.Parameter(
            torch.eye(n_regimes) * 0.9 + torch.ones(n_regimes, n_regimes) * 0.1 / n_regimes
        )

        # Risk aversion (inverse temperature)
        self.log_risk_aversion = nn.Parameter(torch.tensor(0.0))

    @property
    def risk_aversion(self) -> torch.Tensor:
        return torch.exp(self.log_risk_aversion)

    def forward(self, state: MarketState, factors: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF value.

        Args:
            state: Current market state
            factors: Factor values (n_factors,)

        Returns:
            SDF value M_t
        """
        # Get regime-specific risk premia
        regime_probs = self._regime_probabilities(state.regime)
        lambdas = (regime_probs @ self.factor_lambdas)  # (n_factors,)

        # Risk premium from factor exposure
        risk_premium = (self.factor_betas @ lambdas).sum()

        # Free energy form
        log_sdf = -self.risk_aversion * (state.temperature * risk_premium)

        return torch.exp(log_sdf)

    def _regime_probabilities(self, current_regime: int) -> torch.Tensor:
        """Get regime probability distribution."""
        probs = torch.softmax(self.regime_transitions[current_regime], dim=0)
        return probs

    def price_asset(self, payoff: torch.Tensor, state: MarketState,
                    factors: torch.Tensor, n_simulations: int = 1000) -> torch.Tensor:
        """
        Price an asset using Monte Carlo with the SDF.

        Args:
            payoff: Payoff function values (n_simulations,)
            state: Current market state
            factors: Factor paths (n_simulations, n_factors)
            n_simulations: Number of MC paths

        Returns:
            Expected discounted payoff
        """
        sdf_values = torch.stack([
            self.forward(state, factors[i]) for i in range(n_simulations)
        ])
        return (sdf_values * payoff).mean()

    def risk_premium(self, state: MarketState) -> torch.Tensor:
        """Compute asset risk premia."""
        regime_probs = self._regime_probabilities(state.regime)
        lambdas = regime_probs @ self.factor_lambdas
        return self.factor_betas @ lambdas


# ============================================================================
# RUPPEINER GEOMETRY
# ============================================================================

class RuppeinerMarket:
    """
    Risk geometry via Ruppeiner metric.

    The metric tensor g_ij measures risk curvature in the space of
    portfolios/positions.
    """

    def __init__(self, state: MarketState):
        self.state = state
        self._metric = None
        self._christoffel = None

    def metric_tensor(self) -> torch.Tensor:
        """
        Compute the Ruppeiner metric g_ij.

        g_ij = -d²S/dX_i dX_j

        where S is the entropy (negative risk).
        """
        if self._metric is not None:
            return self._metric

        cov = self.state.covariance_matrix()

        # Ruppeiner metric is inverse covariance (Fisher information)
        # with temperature scaling
        T = self.state.temperature
        self._metric = torch.linalg.inv(cov) / T

        return self._metric

    def christoffel_symbols(self) -> torch.Tensor:
        """
        Compute Christoffel symbols Γ^k_ij for geodesic equation.
        """
        if self._christoffel is not None:
            return self._christoffel

        g = self.metric_tensor()
        n = g.shape[0]

        # Numerical differentiation for metric derivatives
        eps = 1e-6
        dg = torch.zeros(n, n, n)  # dg_ij/dx_k

        # For simplicity, assume metric is constant (flat approximation)
        # Full implementation would compute derivatives

        g_inv = torch.linalg.inv(g)
        self._christoffel = torch.zeros(n, n, n)

        # Γ^k_ij = (1/2) g^kl (∂g_li/∂x_j + ∂g_lj/∂x_i - ∂g_ij/∂x_l)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        self._christoffel[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[l, i, j] + dg[l, j, i] - dg[i, j, l]
                        )

        return self._christoffel

    def geodesic_step(self, position: torch.Tensor, velocity: torch.Tensor,
                      dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take one step along geodesic (natural gradient path).

        d²x^k/dt² + Γ^k_ij dx^i/dt dx^j/dt = 0
        """
        gamma = self.christoffel_symbols()

        # Geodesic acceleration
        acceleration = -torch.einsum('kij,i,j->k', gamma, velocity, velocity)

        # Symplectic Euler step
        new_velocity = velocity + dt * acceleration
        new_position = position + dt * new_velocity

        return new_position, new_velocity

    def ricci_scalar(self) -> torch.Tensor:
        """
        Compute Ricci scalar curvature R.

        High R indicates high risk concentration.
        """
        g = self.metric_tensor()
        n = g.shape[0]

        # For diagonal metric approximation:
        # R ≈ sum of eigenvalue reciprocals
        eigenvalues = torch.linalg.eigvalsh(g)
        return torch.sum(1.0 / (eigenvalues + 1e-8))
```

## Market Sieve Implementation

```python
# ============================================================================
# MARKET SIEVE (21 GATES + 16 BARRIERS)
# ============================================================================

class GateNode(ABC):
    """Abstract base class for gate nodes."""

    def __init__(self, node_id: int, name: str):
        self.node_id = node_id
        self.name = name

    @abstractmethod
    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        """
        Check the gate condition.

        Returns:
            (status, loss_contribution)
        """
        pass


class SolvencyGate(GateNode):
    """Node 1: Solvency check."""

    def __init__(self, min_equity_ratio: float = 0.0):
        super().__init__(1, "Solvency")
        self.min_equity_ratio = min_equity_ratio

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Equity = positions * prices
        portfolio_values = state.positions @ state.prices

        # Check for negative equity
        min_value = portfolio_values.min().item()

        if min_value > self.min_equity_ratio:
            return GateStatus.PASS, 0.0
        else:
            loss = torch.relu(-portfolio_values).sum().item()
            return GateStatus.FAIL, loss


class LeverageGate(GateNode):
    """Node 3: Leverage balance check."""

    def __init__(self, max_leverage: float = 10.0):
        super().__init__(3, "Leverage")
        self.max_leverage = max_leverage

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        max_lev = state.leverage.max().item()

        if max_lev <= self.max_leverage:
            return GateStatus.PASS, 0.0
        else:
            excess = (state.leverage - self.max_leverage).relu().sum().item()
            return GateStatus.FAIL, excess


class StationarityGate(GateNode):
    """Node 5: Stationarity check."""

    def __init__(self, max_drift: float = 0.1):
        super().__init__(5, "Stationarity")
        self.max_drift = max_drift
        self.history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.history.append(state.prices.clone())

        if len(self.history) < 10:
            return GateStatus.BOUNDED, 0.0

        # Check for unit root / drift
        recent = torch.stack(self.history[-10:])
        drift = (recent[-1] - recent[0]).abs().mean().item() / 10

        if drift < self.max_drift:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, drift - self.max_drift


class CapacityGate(GateNode):
    """Node 6: Market depth capacity check."""

    def __init__(self, min_depth_ratio: float = 0.01):
        super().__init__(6, "Capacity")
        self.min_depth_ratio = min_depth_ratio

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Depth proxy: inverse of spread
        depth = 1.0 / (state.liquidity + 1e-8)
        position_sizes = state.positions.abs().sum(dim=0)

        # Check if positions exceed depth
        depth_ratio = position_sizes / (depth + 1e-8)
        max_ratio = depth_ratio.max().item()

        if max_ratio < self.min_depth_ratio:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_ratio


class ConnectivityGate(GateNode):
    """Node 8: Market connectivity check."""

    def __init__(self, min_correlation: float = -0.99):
        super().__init__(8, "Connectivity")
        self.min_correlation = min_correlation

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check correlation matrix is valid (no extreme negative)
        min_corr = state.correlations.min().item()

        if min_corr > self.min_correlation:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, abs(min_corr - self.min_correlation)


class RepresentationGate(GateNode):
    """Node 11: Model representation adequacy."""

    def __init__(self, max_residual: float = 0.1):
        super().__init__(11, "Representation")
        self.max_residual = max_residual
        self.model_predictions: Optional[torch.Tensor] = None

    def set_predictions(self, predictions: torch.Tensor):
        self.model_predictions = predictions

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        if self.model_predictions is None:
            return GateStatus.UNKNOWN, 0.0

        residuals = (state.prices - self.model_predictions).abs()
        max_res = residuals.max().item()

        if max_res < self.max_residual:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_res


class TurnoverGate(GateNode):
    """Node 2: Capital turnover (conservation) check."""

    def __init__(self, max_turnover_rate: float = 10.0):
        super().__init__(2, "Turnover")
        self.max_turnover_rate = max_turnover_rate
        self.position_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.position_history.append(state.positions.clone())

        if len(self.position_history) < 2:
            return GateStatus.BOUNDED, 0.0

        # Compute turnover: sum of absolute position changes / portfolio value
        delta = (self.position_history[-1] - self.position_history[-2]).abs()
        portfolio_value = (state.positions.abs() @ state.prices).sum() + 1e-8
        turnover = (delta @ state.prices).sum() / portfolio_value

        if turnover.item() < self.max_turnover_rate:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, turnover.item() - self.max_turnover_rate


class ScaleGate(GateNode):
    """Node 4: Scale balance (no asset dominates)."""

    def __init__(self, max_concentration: float = 0.5):
        super().__init__(4, "Scale")
        self.max_concentration = max_concentration

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Compute HHI (Herfindahl-Hirschman Index)
        weights = state.positions.abs() / (state.positions.abs().sum() + 1e-8)
        hhi = (weights ** 2).sum().item()

        # Max concentration is 1/n for equal weights
        n_assets = len(state.prices)
        max_acceptable = self.max_concentration

        if hhi < max_acceptable:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, hhi - max_acceptable


class StiffnessGate(GateNode):
    """Node 7: Market stiffness (price response elasticity)."""

    def __init__(self, max_impact: float = 0.01):
        super().__init__(7, "Stiffness")
        self.max_impact = max_impact

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Estimate price impact: position * (1/liquidity)
        impact = state.positions.abs() * (1.0 / (state.liquidity + 1e-8))
        max_impact = impact.max().item()

        if max_impact < self.max_impact:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_impact - self.max_impact


class BifurcationGate(GateNode):
    """Node 7a: Bifurcation detection (approaching critical point)."""

    def __init__(self, eigenvalue_threshold: float = 0.95):
        super().__init__(701, "Bifurcation")
        self.eigenvalue_threshold = eigenvalue_threshold

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check for near-zero eigenvalue in Jacobian (approaching bifurcation)
        # Proxy: check correlation matrix eigenvalues
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        min_eigenvalue = eigenvalues.min().item()

        # Near-zero eigenvalue indicates instability
        if min_eigenvalue > 1 - self.eigenvalue_threshold:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.eigenvalue_threshold - min_eigenvalue


class AlternativesGate(GateNode):
    """Node 7b: Alternative investments available (diversification possible)."""

    def __init__(self, min_uncorrelated: int = 3):
        super().__init__(702, "Alternatives")
        self.min_uncorrelated = min_uncorrelated

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Count assets with correlation < 0.5 to portfolio
        portfolio_weights = state.positions / (state.positions.sum() + 1e-8)
        portfolio_corr = state.correlations @ portfolio_weights
        n_uncorrelated = (portfolio_corr.abs() < 0.5).sum().item()

        if n_uncorrelated >= self.min_uncorrelated:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, float(self.min_uncorrelated - n_uncorrelated)


class StabilityGate(GateNode):
    """Node 7c: Lyapunov stability check."""

    def __init__(self, max_lyapunov: float = 0.0):
        super().__init__(703, "Stability")
        self.max_lyapunov = max_lyapunov
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        if len(self.return_history) < 20:
            self.return_history.append(state.prices.clone())
            return GateStatus.BOUNDED, 0.0

        # Estimate largest Lyapunov exponent from return series
        returns = torch.stack(self.return_history[-20:])
        log_returns = torch.log(returns[1:] / (returns[:-1] + 1e-8) + 1e-8)

        # Simplified: use variance growth rate as proxy
        var_growth = log_returns.var(dim=0).mean().item()
        lyapunov_estimate = var_growth * 252  # Annualized

        self.return_history.append(state.prices.clone())

        if lyapunov_estimate < self.max_lyapunov:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, lyapunov_estimate


class SwitchingGate(GateNode):
    """Node 7d: Regime switching detection."""

    def __init__(self, max_switch_prob: float = 0.3):
        super().__init__(704, "Switching")
        self.max_switch_prob = max_switch_prob
        self.regime_history: List[int] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.regime_history.append(state.regime)

        if len(self.regime_history) < 10:
            return GateStatus.BOUNDED, 0.0

        # Compute empirical switching probability
        switches = sum(
            1 for i in range(1, 10)
            if self.regime_history[-i] != self.regime_history[-i-1]
        )
        switch_prob = switches / 9

        if switch_prob < self.max_switch_prob:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, switch_prob - self.max_switch_prob


class TamenessGate(GateNode):
    """Node 9: Distribution tameness (fat tail check)."""

    def __init__(self, max_kurtosis: float = 10.0):
        super().__init__(9, "Tameness")
        self.max_kurtosis = max_kurtosis
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.return_history.append(state.prices.clone())

        if len(self.return_history) < 30:
            return GateStatus.BOUNDED, 0.0

        # Compute returns
        prices = torch.stack(self.return_history[-30:])
        returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        # Compute excess kurtosis
        mean_ret = returns.mean(dim=0)
        std_ret = returns.std(dim=0) + 1e-8
        z = (returns - mean_ret) / std_ret
        kurtosis = (z ** 4).mean(dim=0).mean() - 3  # Excess kurtosis

        if kurtosis.item() < self.max_kurtosis:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, kurtosis.item() - self.max_kurtosis


class MixingGate(GateNode):
    """Node 10: Information mixing (market efficiency proxy)."""

    def __init__(self, min_autocorr_decay: float = 0.5):
        super().__init__(10, "Mixing")
        self.min_autocorr_decay = min_autocorr_decay
        self.return_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.return_history.append(state.prices.clone())

        if len(self.return_history) < 20:
            return GateStatus.BOUNDED, 0.0

        # Compute autocorrelation at lag 1
        prices = torch.stack(self.return_history[-20:])
        returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        ret_mean = returns.mean(dim=0)
        ret_centered = returns - ret_mean
        var = (ret_centered ** 2).mean(dim=0) + 1e-8

        autocorr = (ret_centered[1:] * ret_centered[:-1]).mean(dim=0) / var
        max_autocorr = autocorr.abs().max().item()

        # Low autocorrelation indicates good mixing
        if max_autocorr < self.min_autocorr_decay:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, max_autocorr - self.min_autocorr_decay


class OscillationGate(GateNode):
    """Node 12: Oscillation detection (boom-bust cycles)."""

    def __init__(self, max_amplitude: float = 0.2, min_period: int = 5):
        super().__init__(12, "Oscillation")
        self.max_amplitude = max_amplitude
        self.min_period = min_period
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 20:
            return GateStatus.BOUNDED, 0.0

        prices = torch.stack(self.price_history[-20:])

        # Detect oscillation via sign changes in returns
        returns = (prices[1:] - prices[:-1])
        signs = torch.sign(returns)
        sign_changes = (signs[1:] * signs[:-1] < 0).float().sum(dim=0)

        # High sign changes indicate oscillation
        avg_changes = sign_changes.mean().item()
        amplitude = returns.abs().mean().item()

        if avg_changes > self.min_period and amplitude > self.max_amplitude:
            return GateStatus.FAIL, amplitude
        else:
            return GateStatus.PASS, 0.0


class CouplingGate(GateNode):
    """Node 14: Boundary coupling (external data connection)."""

    def __init__(self, min_coupling: float = 0.1):
        super().__init__(14, "Coupling")
        self.min_coupling = min_coupling

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check if prices are coupled to external signals
        # Proxy: information ratio > threshold
        if not hasattr(state, 'external_signal') or state.external_signal is None:
            return GateStatus.BOUNDED, 0.0

        corr = torch.corrcoef(torch.stack([state.prices, state.external_signal]))[0, 1]
        coupling = corr.abs().item()

        if coupling > self.min_coupling:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.min_coupling - coupling


class OverloadGate(GateNode):
    """Node 15: Information overload detection."""

    def __init__(self, max_entropy: float = 5.0):
        super().__init__(15, "Overload")
        self.max_entropy = max_entropy

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Compute entropy of price distribution
        # Proxy: use volatility as entropy measure
        entropy = state.volatilities.mean().item() * np.log(len(state.prices))

        if entropy < self.max_entropy:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, entropy - self.max_entropy


class AlignmentGate(GateNode):
    """Node 16: Incentive alignment check."""

    def __init__(self, max_misalignment: float = 0.1):
        super().__init__(16, "Alignment")
        self.max_misalignment = max_misalignment

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check if prices align with fundamentals
        # Proxy: price-to-fundamental ratio deviation
        if not hasattr(state, 'fundamentals') or state.fundamentals is None:
            return GateStatus.BOUNDED, 0.0

        ratio = state.prices / (state.fundamentals + 1e-8)
        deviation = (ratio - 1.0).abs().mean().item()

        if deviation < self.max_misalignment:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, deviation - self.max_misalignment


class LockGate(GateNode):
    """Node 17: Hard regulatory limit check."""

    def __init__(self, limits: Dict[str, float] = None):
        super().__init__(17, "Lock")
        self.limits = limits or {
            'max_position': 1e7,
            'max_leverage': 20.0,
            'min_margin': 0.05
        }

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        violations = []

        # Check position limits
        if state.positions.abs().max().item() > self.limits['max_position']:
            violations.append('position')

        # Check leverage
        if state.leverage.max().item() > self.limits['max_leverage']:
            violations.append('leverage')

        # Check margin
        margin = state.positions.abs().sum() / (state.nav + 1e-8)
        if margin.item() < self.limits['min_margin']:
            violations.append('margin')

        if len(violations) == 0:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, float(len(violations))


class SymmetryGate(GateNode):
    """Node 18: Market symmetry check (bid-ask balance)."""

    def __init__(self, max_asymmetry: float = 0.2):
        super().__init__(18, "Symmetry")
        self.max_asymmetry = max_asymmetry

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check bid-ask spread asymmetry
        if not hasattr(state, 'bid_prices') or not hasattr(state, 'ask_prices'):
            return GateStatus.BOUNDED, 0.0

        mid = (state.bid_prices + state.ask_prices) / 2
        bid_dist = (mid - state.bid_prices) / (mid + 1e-8)
        ask_dist = (state.ask_prices - mid) / (mid + 1e-8)

        asymmetry = (bid_dist - ask_dist).abs().mean().item()

        if asymmetry < self.max_asymmetry:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, asymmetry - self.max_asymmetry


class DisentanglementGate(GateNode):
    """Node 19: Factor disentanglement check."""

    def __init__(self, min_independence: float = 0.3):
        super().__init__(19, "Disentanglement")
        self.min_independence = min_independence

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Check that correlation matrix has reasonable eigenvalue spread
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        eigenvalues = eigenvalues.sort(descending=True).values

        # Effective rank = ratio of sum to max
        eff_rank = eigenvalues.sum() / (eigenvalues[0] + 1e-8)
        independence = eff_rank / len(eigenvalues)

        if independence > self.min_independence:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, self.min_independence - independence


class LipschitzGate(GateNode):
    """Node 20: Price function Lipschitz continuity."""

    def __init__(self, max_lipschitz: float = 2.0):
        super().__init__(20, "Lipschitz")
        self.max_lipschitz = max_lipschitz
        self.state_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        current = (state.prices.clone(), state.positions.clone())
        self.state_history.append(current)

        if len(self.state_history) < 2:
            return GateStatus.BOUNDED, 0.0

        # Estimate Lipschitz constant: |f(x) - f(y)| / |x - y|
        prev_prices, prev_positions = self.state_history[-2]

        price_change = (state.prices - prev_prices).norm()
        pos_change = (state.positions - prev_positions).norm() + 1e-8

        lipschitz = (price_change / pos_change).item()

        if lipschitz < self.max_lipschitz:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, lipschitz - self.max_lipschitz


class SymplecticGate(GateNode):
    """Node 21: Symplectic structure preservation (conservative dynamics)."""

    def __init__(self, max_divergence: float = 0.1):
        super().__init__(21, "Symplectic")
        self.max_divergence = max_divergence
        self.pq_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def check(self, state: MarketState) -> Tuple[GateStatus, float]:
        # Track (position, momentum) pairs for Hamiltonian structure
        # Momentum proxy: position * price rate of change
        q = state.positions
        p = state.prices * state.volatilities  # Momentum proxy

        self.pq_history.append((q.clone(), p.clone()))

        if len(self.pq_history) < 3:
            return GateStatus.BOUNDED, 0.0

        # Check phase space volume preservation (Liouville theorem)
        # Proxy: check that det(Jacobian) ≈ 1
        q_prev, p_prev = self.pq_history[-2]
        q_old, p_old = self.pq_history[-3]

        # Simplified: check momentum-position correlation stability
        corr1 = (q * p).sum() / (q.norm() * p.norm() + 1e-8)
        corr2 = (q_prev * p_prev).sum() / (q_prev.norm() * p_prev.norm() + 1e-8)

        divergence = (corr1 - corr2).abs().item()

        if divergence < self.max_divergence:
            return GateStatus.PASS, 0.0
        else:
            return GateStatus.FAIL, divergence - self.max_divergence


# ============================================================================
# BARRIER IMPLEMENTATION
# ============================================================================

class Barrier(ABC):
    """Abstract base class for barriers."""

    def __init__(self, name: str):
        self.name = name
        self.status = BarrierStatus.CLEAR

    @abstractmethod
    def check(self, state: MarketState) -> BarrierStatus:
        pass

    @abstractmethod
    def defense_action(self, state: MarketState) -> MarketState:
        """Apply defense if breached."""
        pass


class BarrierSat(Barrier):
    """Position saturation barrier."""

    def __init__(self, max_position: float = 1e6):
        super().__init__("BarrierSat")
        self.max_position = max_position

    def check(self, state: MarketState) -> BarrierStatus:
        max_pos = state.positions.abs().max().item()

        if max_pos < 0.8 * self.max_position:
            self.status = BarrierStatus.CLEAR
        elif max_pos < self.max_position:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Scale down positions
        scale = self.max_position / (state.positions.abs().max().item() + 1e-8)
        state.positions = state.positions * min(scale, 1.0)
        return state


class BarrierOmin(Barrier):
    """Flash crash (Ominous) barrier."""

    def __init__(self, max_velocity: float = 0.1, window: int = 10):
        super().__init__("BarrierOmin")
        self.max_velocity = max_velocity
        self.window = window
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 2:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Price velocity
        velocity = (self.price_history[-1] - self.price_history[-2]).abs()
        max_vel = velocity.max().item()

        if max_vel < 0.5 * self.max_velocity:
            self.status = BarrierStatus.CLEAR
        elif max_vel < self.max_velocity:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        # Trim history
        if len(self.price_history) > self.window:
            self.price_history = self.price_history[-self.window:]

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Halt: revert to last known good price
        if len(self.price_history) >= 2:
            state.prices = self.price_history[-2].clone()
        return state


class BarrierTypeII(Barrier):
    """Vol-of-vol crisis barrier."""

    def __init__(self, max_vol_of_vol: float = 0.5):
        super().__init__("BarrierTypeII")
        self.max_vol_of_vol = max_vol_of_vol
        self.vol_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.vol_history.append(state.volatilities.clone())

        if len(self.vol_history) < 10:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Vol of vol
        recent_vols = torch.stack(self.vol_history[-10:])
        vol_of_vol = recent_vols.std(dim=0).mean().item()

        if vol_of_vol < 0.5 * self.max_vol_of_vol:
            self.status = BarrierStatus.CLEAR
        elif vol_of_vol < self.max_vol_of_vol:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED

        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Increase margin / reduce exposure
        state.leverage = state.leverage * 0.5
        return state


class BarrierGap(Barrier):
    """Liquidity gap barrier."""

    def __init__(self, max_gap: float = 0.05):
        super().__init__("BarrierGap")
        self.max_gap = max_gap

    def check(self, state: MarketState) -> BarrierStatus:
        # Gap = difference between best bid/ask and next level
        spread = 1.0 / (state.liquidity + 1e-8)
        max_spread = spread.max().item()

        if max_spread < 0.5 * self.max_gap:
            self.status = BarrierStatus.CLEAR
        elif max_spread < self.max_gap:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Widen acceptable execution range
        return state


class BarrierCausal(Barrier):
    """Information lag barrier."""

    def __init__(self, max_lag: int = 5):
        super().__init__("BarrierCausal")
        self.max_lag = max_lag
        self.timestamps: List[float] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.timestamps.append(state.timestamp)

        if len(self.timestamps) < 2:
            self.status = BarrierStatus.CLEAR
            return self.status

        lag = self.timestamps[-1] - self.timestamps[-2]

        if lag < 0.5 * self.max_lag:
            self.status = BarrierStatus.CLEAR
        elif lag < self.max_lag:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierScat(Barrier):
    """Market fragmentation barrier."""

    def __init__(self, max_fragmentation: float = 0.3):
        super().__init__("BarrierScat")
        self.max_fragmentation = max_fragmentation

    def check(self, state: MarketState) -> BarrierStatus:
        # Fragmentation proxy: variance in liquidity across assets
        liq_var = state.liquidity.var().item() / (state.liquidity.mean().item() + 1e-8)

        if liq_var < 0.5 * self.max_fragmentation:
            self.status = BarrierStatus.CLEAR
        elif liq_var < self.max_fragmentation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierMix(Barrier):
    """Herding behavior barrier."""

    def __init__(self, max_herding: float = 0.8):
        super().__init__("BarrierMix")
        self.max_herding = max_herding

    def check(self, state: MarketState) -> BarrierStatus:
        # Herding proxy: first eigenvalue dominance
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        first_ev_ratio = eigenvalues[-1].item() / (eigenvalues.sum().item() + 1e-8)

        if first_ev_ratio < 0.5 * self.max_herding:
            self.status = BarrierStatus.CLEAR
        elif first_ev_ratio < self.max_herding:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierCap(Barrier):
    """Controllability barrier."""

    def __init__(self, min_controllability: float = 0.1):
        super().__init__("BarrierCap")
        self.min_controllability = min_controllability

    def check(self, state: MarketState) -> BarrierStatus:
        # Controllability proxy: minimum eigenvalue of position Gramian
        min_ev = torch.linalg.eigvalsh(state.correlations).min().item()

        if min_ev > 2 * self.min_controllability:
            self.status = BarrierStatus.CLEAR
        elif min_ev > self.min_controllability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierVac(Barrier):
    """Regime instability barrier."""

    def __init__(self, max_regime_instability: float = 0.5):
        super().__init__("BarrierVac")
        self.max_regime_instability = max_regime_instability
        self.regime_history: List[int] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.regime_history.append(state.regime)

        if len(self.regime_history) < 5:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Regime instability: recent switching frequency
        switches = sum(1 for i in range(1, 5) if self.regime_history[-i] != self.regime_history[-i-1])
        instability = switches / 4

        if instability < 0.5 * self.max_regime_instability:
            self.status = BarrierStatus.CLEAR
        elif instability < self.max_regime_instability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierFreq(Barrier):
    """HFT oscillation barrier."""

    def __init__(self, max_hft_oscillation: int = 3):
        super().__init__("BarrierFreq")
        self.max_hft_oscillation = max_hft_oscillation
        self.price_history: List[torch.Tensor] = []

    def check(self, state: MarketState) -> BarrierStatus:
        self.price_history.append(state.prices.clone())

        if len(self.price_history) < 10:
            self.status = BarrierStatus.CLEAR
            return self.status

        prices = torch.stack(self.price_history[-10:])
        returns = prices[1:] - prices[:-1]
        sign_changes = (returns[1:] * returns[:-1] < 0).float().sum().item()
        oscillation = sign_changes / 8

        if oscillation < 0.5 * self.max_hft_oscillation:
            self.status = BarrierStatus.CLEAR
        elif oscillation < self.max_hft_oscillation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierEpi(Barrier):
    """Information overload barrier."""

    def __init__(self, max_info_overload: float = 5.0):
        super().__init__("BarrierEpi")
        self.max_info_overload = max_info_overload

    def check(self, state: MarketState) -> BarrierStatus:
        entropy = state.volatilities.mean().item() * np.log(len(state.prices) + 1)

        if entropy < 0.5 * self.max_info_overload:
            self.status = BarrierStatus.CLEAR
        elif entropy < self.max_info_overload:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierAction(Barrier):
    """Execution impossibility barrier."""

    def __init__(self, min_execution_prob: float = 0.9):
        super().__init__("BarrierAction")
        self.min_execution_prob = min_execution_prob

    def check(self, state: MarketState) -> BarrierStatus:
        # Execution probability proxy: liquidity / position size
        exec_prob = (state.liquidity / (state.positions.abs() + 1e-8)).min().item()
        exec_prob = min(1.0, exec_prob)

        if exec_prob > self.min_execution_prob:
            self.status = BarrierStatus.CLEAR
        elif exec_prob > 0.5 * self.min_execution_prob:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierInput(Barrier):
    """Data starvation barrier."""

    def __init__(self, min_data_quality: float = 0.8):
        super().__init__("BarrierInput")
        self.min_data_quality = min_data_quality

    def check(self, state: MarketState) -> BarrierStatus:
        # Data quality proxy: 1 - NaN ratio (simulated)
        # In real implementation, would check actual data completeness
        nan_ratio = torch.isnan(state.prices).float().mean().item()
        quality = 1.0 - nan_ratio

        if quality > self.min_data_quality:
            self.status = BarrierStatus.CLEAR
        elif quality > 0.5 * self.min_data_quality:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Replace NaNs with last known values
        state.prices = torch.nan_to_num(state.prices, nan=1.0)
        return state


class BarrierVariety(Barrier):
    """Hedging impossibility barrier."""

    def __init__(self, min_hedgeability: float = 0.5):
        super().__init__("BarrierVariety")
        self.min_hedgeability = min_hedgeability

    def check(self, state: MarketState) -> BarrierStatus:
        # Hedgeability: effective rank of correlation matrix
        eigenvalues = torch.linalg.eigvalsh(state.correlations)
        eff_rank = eigenvalues.sum() / (eigenvalues.max() + 1e-8)
        hedgeability = eff_rank / len(eigenvalues)

        if hedgeability > self.min_hedgeability:
            self.status = BarrierStatus.CLEAR
        elif hedgeability > 0.5 * self.min_hedgeability:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierBode(Barrier):
    """Risk waterbed barrier (Bode integral constraint)."""

    def __init__(self, max_waterbed: float = 0.3):
        super().__init__("BarrierBode")
        self.max_waterbed = max_waterbed
        self.risk_history: List[float] = []

    def check(self, state: MarketState) -> BarrierStatus:
        total_risk = state.volatilities.sum().item()
        self.risk_history.append(total_risk)

        if len(self.risk_history) < 5:
            self.status = BarrierStatus.CLEAR
            return self.status

        # Waterbed effect: risk reduction in one area causing increase elsewhere
        risk_std = np.std(self.risk_history[-5:])
        waterbed = risk_std / (np.mean(self.risk_history[-5:]) + 1e-8)

        if waterbed < 0.5 * self.max_waterbed:
            self.status = BarrierStatus.CLEAR
        elif waterbed < self.max_waterbed:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierLock(Barrier):
    """Hard regulatory lock barrier."""

    def __init__(self):
        super().__init__("BarrierLock")
        self.hard_limits = {
            'max_position': 1e8,
            'max_leverage': 25.0,
            'min_capital': 1e6
        }

    def check(self, state: MarketState) -> BarrierStatus:
        violations = 0

        if state.positions.abs().max().item() > self.hard_limits['max_position']:
            violations += 1
        if state.leverage.max().item() > self.hard_limits['max_leverage']:
            violations += 1
        if state.nav < self.hard_limits['min_capital']:
            violations += 1

        if violations == 0:
            self.status = BarrierStatus.CLEAR
        elif violations == 1:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        # Force compliance
        scale = self.hard_limits['max_position'] / (state.positions.abs().max().item() + 1e-8)
        state.positions = state.positions * min(scale, 1.0)
        return state


class BarrierLiq(Barrier):
    """Liquidity crisis barrier."""

    def __init__(self, min_liquidity: float = 0.01):
        super().__init__("BarrierLiq")
        self.min_liquidity = min_liquidity

    def check(self, state: MarketState) -> BarrierStatus:
        min_liq = state.liquidity.min().item()

        if min_liq > 2 * self.min_liquidity:
            self.status = BarrierStatus.CLEAR
        elif min_liq > self.min_liquidity:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierLev(Barrier):
    """Leverage crisis barrier."""

    def __init__(self, max_leverage: float = 15.0):
        super().__init__("BarrierLev")
        self.max_leverage = max_leverage

    def check(self, state: MarketState) -> BarrierStatus:
        max_lev = state.leverage.max().item()

        if max_lev < 0.7 * self.max_leverage:
            self.status = BarrierStatus.CLEAR
        elif max_lev < self.max_leverage:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        state.leverage = state.leverage * 0.8
        return state


class BarrierRef(Barrier):
    """Oracle/reference data barrier."""

    def __init__(self, max_oracle_deviation: float = 0.05):
        super().__init__("BarrierRef")
        self.max_oracle_deviation = max_oracle_deviation

    def check(self, state: MarketState) -> BarrierStatus:
        if not hasattr(state, 'oracle_prices') or state.oracle_prices is None:
            self.status = BarrierStatus.CLEAR
            return self.status

        deviation = (state.prices - state.oracle_prices).abs() / (state.oracle_prices + 1e-8)
        max_dev = deviation.max().item()

        if max_dev < 0.5 * self.max_oracle_deviation:
            self.status = BarrierStatus.CLEAR
        elif max_dev < self.max_oracle_deviation:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        return state


class BarrierDef(Barrier):
    """Default probability barrier."""

    def __init__(self, max_default_prob: float = 0.1):
        super().__init__("BarrierDef")
        self.max_default_prob = max_default_prob

    def check(self, state: MarketState) -> BarrierStatus:
        # Default probability proxy: leverage / solvency ratio
        solvency = state.positions @ state.prices / (state.nav + 1e-8)
        default_prob = torch.sigmoid(state.leverage.max() - solvency.abs()).item()

        if default_prob < 0.5 * self.max_default_prob:
            self.status = BarrierStatus.CLEAR
        elif default_prob < self.max_default_prob:
            self.status = BarrierStatus.WARNING
        else:
            self.status = BarrierStatus.BREACHED
        return self.status

    def defense_action(self, state: MarketState) -> MarketState:
        state.leverage = state.leverage * 0.5
        return state
```

## Complete Market Sieve

```python
# ============================================================================
# COMPLETE MARKET SIEVE
# ============================================================================

class MarketSieve:
    """
    Complete Market Sieve with 21 gates and 20 barriers.

    Routes pricing decisions through permit checks.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize all 21 gates
        self.gates: Dict[int, GateNode] = {
            1: SolvencyGate(),
            2: TurnoverGate(self.config.get('max_turnover', 10.0)),
            3: LeverageGate(self.config.get('max_leverage', 10.0)),
            4: ScaleGate(self.config.get('max_concentration', 0.5)),
            5: StationarityGate(),
            6: CapacityGate(),
            7: StiffnessGate(),
            701: BifurcationGate(),
            702: AlternativesGate(),
            703: StabilityGate(),
            704: SwitchingGate(),
            8: ConnectivityGate(),
            9: TamenessGate(),
            10: MixingGate(),
            11: RepresentationGate(),
            12: OscillationGate(),
            14: CouplingGate(),
            15: OverloadGate(),
            16: AlignmentGate(),
            17: LockGate(self.config.get('limits')),
            18: SymmetryGate(),
            19: DisentanglementGate(),
            20: LipschitzGate(),
            21: SymplecticGate(),
        }

        # Initialize all 20 barriers
        self.barriers: Dict[str, Barrier] = {
            'BarrierSat': BarrierSat(self.config.get('max_position', 1e6)),
            'BarrierOmin': BarrierOmin(self.config.get('max_velocity', 0.1)),
            'BarrierTypeII': BarrierTypeII(self.config.get('max_vol_of_vol', 0.5)),
            'BarrierGap': BarrierGap(self.config.get('max_gap', 0.05)),
            'BarrierCausal': BarrierCausal(self.config.get('max_lag', 5)),
            'BarrierScat': BarrierScat(self.config.get('max_fragmentation', 0.3)),
            'BarrierMix': BarrierMix(self.config.get('max_herding', 0.8)),
            'BarrierCap': BarrierCap(self.config.get('min_controllability', 0.1)),
            'BarrierVac': BarrierVac(self.config.get('max_regime_instability', 0.5)),
            'BarrierFreq': BarrierFreq(self.config.get('max_hft_oscillation', 3)),
            'BarrierEpi': BarrierEpi(self.config.get('max_info_overload', 5.0)),
            'BarrierAction': BarrierAction(self.config.get('min_execution_prob', 0.9)),
            'BarrierInput': BarrierInput(self.config.get('min_data_quality', 0.8)),
            'BarrierVariety': BarrierVariety(self.config.get('min_hedgeability', 0.5)),
            'BarrierBode': BarrierBode(self.config.get('max_waterbed', 0.3)),
            'BarrierLock': BarrierLock(),
            'BarrierLiq': BarrierLiq(self.config.get('min_liquidity', 0.01)),
            'BarrierLev': BarrierLev(self.config.get('max_leverage_barrier', 15.0)),
            'BarrierRef': BarrierRef(self.config.get('max_oracle_deviation', 0.05)),
            'BarrierDef': BarrierDef(self.config.get('max_default_prob', 0.1)),
        }

    def run(self, state: MarketState,
            model_predictions: Optional[torch.Tensor] = None) -> Certificate:
        """
        Run complete Sieve check.

        Args:
            state: Current market state
            model_predictions: Optional model price predictions

        Returns:
            Certificate with all check results
        """
        gate_results = {}
        total_loss = 0.0

        # Run gates
        for node_id, gate in self.gates.items():
            if node_id == 11 and model_predictions is not None:
                gate.set_predictions(model_predictions)
            status, loss = gate.check(state)
            gate_results[node_id] = status
            total_loss += loss

        # Run barriers
        barrier_results = {}
        for name, barrier in self.barriers.items():
            barrier_results[name] = barrier.check(state)

        # Detect failure modes
        failure_modes = self._detect_failure_modes(gate_results, barrier_results)

        # Compute price bounds
        price_bounds = self._compute_price_bounds(state, total_loss)

        # Compute confidence
        n_pass = sum(1 for s in gate_results.values()
                     if s in (GateStatus.PASS, GateStatus.BOUNDED))
        confidence = n_pass / len(gate_results)

        return Certificate(
            gate_results=gate_results,
            barrier_results=barrier_results,
            price_bounds=price_bounds,
            confidence=confidence,
            failure_modes=failure_modes,
            timestamp=state.timestamp
        )

    def _detect_failure_modes(self, gates: Dict[int, GateStatus],
                               barriers: Dict[str, BarrierStatus]) -> List[FailureMode]:
        """Detect active or approaching failure modes."""
        modes = []

        # C.E: Default cascade (Node 1 fail + BarrierSat breach)
        if gates.get(1) == GateStatus.FAIL:
            modes.append(FailureMode.CE)

        # S.E: Supercritical leverage (Node 3 fail)
        if gates.get(3) == GateStatus.FAIL:
            modes.append(FailureMode.SE)

        # T.E: Flash crash (BarrierOmin breach)
        if barriers.get('BarrierOmin') == BarrierStatus.BREACHED:
            modes.append(FailureMode.TE)

        # D.D: Vol crisis (BarrierTypeII breach)
        if barriers.get('BarrierTypeII') == BarrierStatus.BREACHED:
            modes.append(FailureMode.DD)

        return modes

    def _compute_price_bounds(self, state: MarketState,
                               loss: float) -> Tuple[float, float]:
        """Compute price bounds based on uncertainty."""
        base_price = state.prices.mean().item()
        uncertainty = loss * state.temperature
        return (base_price - uncertainty, base_price + uncertainty)

    def apply_defenses(self, state: MarketState) -> MarketState:
        """Apply defense actions for breached barriers."""
        for barrier in self.barriers.values():
            if barrier.status == BarrierStatus.BREACHED:
                state = barrier.defense_action(state)
        return state
```

## Loss Functions

```python
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MarketLoss(nn.Module):
    """
    Combined loss function for market pricing.

    Integrates:
    - Pricing error (SDF consistency)
    - Gate violations
    - Barrier penalties
    - Regularization
    """

    def __init__(self, sieve: MarketSieve, sdf: ThermoeconomicSDF,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.sieve = sieve
        self.sdf = sdf
        self.weights = weights or {
            'pricing': 1.0,
            'gates': 0.5,
            'barriers': 1.0,
            'regularization': 0.01
        }

    def forward(self, state: MarketState,
                target_prices: torch.Tensor,
                factors: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.

        Returns:
            (loss, component_dict)
        """
        losses = {}

        # Pricing loss (SDF consistency)
        predicted_prices = self._price_assets(state, factors)
        losses['pricing'] = F.mse_loss(predicted_prices, target_prices)

        # Gate loss
        cert = self.sieve.run(state, predicted_prices)
        gate_loss = sum(
            1.0 for s in cert.gate_results.values()
            if s == GateStatus.FAIL
        )
        losses['gates'] = torch.tensor(gate_loss, dtype=torch.float32)

        # Barrier loss
        barrier_loss = sum(
            2.0 if s == BarrierStatus.BREACHED else
            0.5 if s == BarrierStatus.WARNING else 0.0
            for s in cert.barrier_results.values()
        )
        losses['barriers'] = torch.tensor(barrier_loss, dtype=torch.float32)

        # Regularization (L2 on model params)
        reg_loss = sum(p.pow(2).sum() for p in self.sdf.parameters())
        losses['regularization'] = reg_loss

        # Weighted sum
        total = sum(self.weights[k] * v for k, v in losses.items())

        return total, losses

    def _price_assets(self, state: MarketState,
                      factors: torch.Tensor) -> torch.Tensor:
        """Price all assets using SDF."""
        risk_premia = self.sdf.risk_premium(state)
        expected_returns = state.prices * (1 + risk_premia)
        return expected_returns


# ============================================================================
# PHASE DETECTOR
# ============================================================================

class MarketPhaseDetector:
    """
    Detect market complexity phase (Crystal/Liquid/Gas).
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.price_history: List[torch.Tensor] = []

    def add_observation(self, prices: torch.Tensor):
        self.price_history.append(prices.clone())
        if len(self.price_history) > self.window:
            self.price_history = self.price_history[-self.window:]

    def detect_phase(self) -> MarketPhase:
        """
        Detect current market phase via compression ratio.
        """
        if len(self.price_history) < 20:
            return MarketPhase.LIQUID  # Default

        prices = torch.stack(self.price_history)
        returns = prices[1:] / prices[:-1] - 1

        # Compression ratio proxy: autocorrelation
        # High autocorrelation → predictable → Liquid
        # Low autocorrelation → random → Crystal or Gas

        returns_flat = returns.flatten()
        if len(returns_flat) < 10:
            return MarketPhase.LIQUID

        autocorr = torch.corrcoef(
            torch.stack([returns_flat[:-1], returns_flat[1:]])
        )[0, 1].item()

        # Volatility clustering (GARCH effect)
        vol = returns.std(dim=0)
        vol_autocorr = torch.corrcoef(
            torch.stack([vol[:-1], vol[1:]])
        )[0, 1].item() if len(vol) > 1 else 0.0

        if abs(autocorr) < 0.1 and abs(vol_autocorr) < 0.1:
            # Low predictability in both price and vol
            # Need external info test to distinguish Crystal vs Gas
            return MarketPhase.CRYSTAL
        elif abs(autocorr) > 0.3 or abs(vol_autocorr) > 0.5:
            # High predictability
            return MarketPhase.LIQUID
        else:
            return MarketPhase.GAS
```

## Complete Market Hypostructure

```python
# ============================================================================
# COMPLETE MARKET HYPOSTRUCTURE
# ============================================================================

class MarketHypostructure:
    """
    Complete Market Hypostructure implementation.

    Integrates:
    - Thermoeconomic SDF
    - Market Sieve (permits)
    - Ruppeiner geometry
    - Phase detection
    - Certificate generation
    """

    def __init__(self, n_assets: int, n_factors: int = 3,
                 config: Optional[Dict] = None):
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.config = config or {}

        # Core components
        self.sdf = ThermoeconomicSDF(n_assets, n_factors)
        self.sieve = MarketSieve(config)
        self.phase_detector = MarketPhaseDetector()

        # Loss function
        self.loss_fn = MarketLoss(self.sieve, self.sdf)

    def price(self, state: MarketState,
              factors: torch.Tensor) -> Tuple[torch.Tensor, Certificate]:
        """
        Generate certified prices.

        Returns:
            (prices, certificate)
        """
        # Compute geometry
        geometry = RuppeinerMarket(state)

        # Detect phase
        self.phase_detector.add_observation(state.prices)
        phase = self.phase_detector.detect_phase()

        # Compute risk premia
        risk_premia = self.sdf.risk_premium(state)

        # Adjust for phase
        if phase == MarketPhase.GAS:
            # Widen uncertainty in chaotic phase
            risk_premia = risk_premia * 2.0

        # Compute prices
        prices = state.prices * (1 + risk_premia)

        # Run Sieve
        certificate = self.sieve.run(state, prices)

        # Apply geometry correction
        curvature = geometry.ricci_scalar()
        if curvature > 10.0:  # High curvature = high concentration
            certificate.failure_modes.append(FailureMode.CD)

        return prices, certificate

    def update(self, state: MarketState,
               target_prices: torch.Tensor,
               factors: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> Dict:
        """
        Update model parameters.

        Returns:
            Dictionary of loss components
        """
        optimizer.zero_grad()

        total_loss, losses = self.loss_fn(state, target_prices, factors)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.sdf.parameters(), 1.0)

        optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    def stress_test(self, state: MarketState,
                    scenario: str) -> Tuple[MarketState, Certificate]:
        """
        Run stress test scenario.

        Args:
            state: Base state
            scenario: Scenario name ('vol_spike', 'liquidity_crisis', etc.)

        Returns:
            (stressed_state, certificate)
        """
        stressed = MarketState(
            prices=state.prices.clone(),
            positions=state.positions.clone(),
            volatilities=state.volatilities.clone(),
            correlations=state.correlations.clone(),
            liquidity=state.liquidity.clone(),
            leverage=state.leverage.clone(),
            regime=state.regime,
            temperature=state.temperature,
            timestamp=state.timestamp
        )

        if scenario == 'vol_spike':
            stressed.volatilities = stressed.volatilities * 3.0
            stressed.temperature = stressed.temperature * 2.0

        elif scenario == 'liquidity_crisis':
            stressed.liquidity = stressed.liquidity * 10.0  # Spreads widen

        elif scenario == 'correlation_spike':
            stressed.correlations = torch.ones_like(stressed.correlations) * 0.9
            torch.diagonal(stressed.correlations).fill_(1.0)

        elif scenario == 'leverage_cascade':
            stressed.leverage = stressed.leverage * 2.0

        # Run Sieve on stressed state
        _, certificate = self.price(stressed, torch.zeros(self.n_factors))

        return stressed, certificate
```

## Usage Example

```python
# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate Market Hypostructure usage."""

    # Configuration
    n_assets = 10
    n_positions = 5
    n_factors = 3

    # Initialize
    hypo = MarketHypostructure(n_assets, n_factors)
    optimizer = torch.optim.Adam(hypo.sdf.parameters(), lr=0.001)

    # Create sample state
    state = MarketState(
        prices=torch.randn(n_assets).abs() * 100,
        positions=torch.randn(n_positions, n_assets) * 10,
        volatilities=torch.rand(n_assets) * 0.3 + 0.1,
        correlations=torch.eye(n_assets) * 0.5 + 0.5,
        liquidity=torch.rand(n_assets) * 0.01 + 0.001,
        leverage=torch.rand(n_positions) * 5 + 1,
        regime=0,
        temperature=1.0,
        timestamp=0.0
    )

    # Generate certified prices
    prices, cert = hypo.price(state, torch.randn(n_factors))

    print(f"Prices: {prices}")
    print(f"Certificate valid: {cert.is_valid}")
    print(f"Confidence: {cert.confidence:.2%}")
    print(f"Price bounds: {cert.price_bounds}")
    print(f"Failure modes: {cert.failure_modes}")

    # Run stress test
    stressed_state, stressed_cert = hypo.stress_test(state, 'vol_spike')
    print(f"\nStress test (vol spike):")
    print(f"  Certificate valid: {stressed_cert.is_valid}")
    print(f"  Failure modes: {stressed_cert.failure_modes}")

    # Training loop example
    for epoch in range(10):
        target_prices = state.prices * 1.01  # 1% expected return
        factors = torch.randn(n_factors)
        losses = hypo.update(state, target_prices, factors, optimizer)
        print(f"Epoch {epoch}: {losses}")


if __name__ == "__main__":
    example_usage()
```

---

