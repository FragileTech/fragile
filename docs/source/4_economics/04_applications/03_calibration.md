# Calibration Guidance

This section provides practical guidance for **calibrating thresholds** in the gate nodes and barriers. Proper calibration is essential for avoiding false positives (unnecessary trading halts) and false negatives (missing genuine risks).

## General Calibration Principles

:::{prf:remark} Calibration Philosophy
:label: rem-calibration-philosophy

Thresholds should be set to achieve:
1. **High recall for catastrophic events** (never miss a crisis)
2. **Acceptable precision for warnings** (tolerate some false alarms)
3. **Regime-dependent adjustment** (tighter in stress, looser in calm)
4. **Asset-class specificity** (equities differ from bonds)
:::

**The Precision-Recall Tradeoff:**

| Setting | False Positives | False Negatives | Use Case |
|---------|----------------|-----------------|----------|
| Conservative | High | Low | Critical infrastructure, pension funds |
| Balanced | Medium | Medium | Standard institutional trading |
| Aggressive | Low | High | Prop trading, market makers |

## Gate Node Threshold Calibration

### Node 1: Solvency Threshold

**Recommended thresholds:**
```
Conservative:  NAV_threshold = 0.20 (fail if leverage > 5×)
Balanced:      NAV_threshold = 0.10 (fail if leverage > 10×)
Aggressive:    NAV_threshold = 0.05 (fail if leverage > 20×)
```

**Calibration procedure:**
1. Compute historical NAV/Notional ratios
2. Identify the 1st percentile of the ratio distribution
3. Set threshold at 2× the 1st percentile (buffer for measurement error)

**Python calibration:**
```python
def calibrate_solvency(nav_history, notional_history, percentile=1):
    """Calibrate solvency threshold from historical data."""
    ratios = np.array(nav_history) / np.array(notional_history)
    threshold = 2 * np.percentile(ratios, percentile)
    return max(threshold, 0.05)  # Floor at 5%
```

### Node 3: Leverage Ratio Threshold

**Recommended thresholds by asset class:**

| Asset Class | Conservative | Balanced | Aggressive |
|-------------|-------------|----------|------------|
| Equities | 2.0 | 3.0 | 5.0 |
| Government Bonds | 10.0 | 15.0 | 25.0 |
| Corporate Bonds | 5.0 | 8.0 | 12.0 |
| FX | 20.0 | 50.0 | 100.0 |
| Commodities | 3.0 | 5.0 | 10.0 |
| Derivatives | 1.5 | 2.5 | 4.0 |

**Calibration procedure:**
1. Collect leverage ratios at historical stress points
2. Identify maximum leverage that survived stress without default
3. Apply 0.8× safety factor

```python
def calibrate_leverage(leverage_history, stress_events, survived):
    """Calibrate leverage from stress survival data."""
    stress_leverages = [leverage_history[t] for t in stress_events if survived[t]]
    max_safe = max(stress_leverages) if stress_leverages else 2.0
    return 0.8 * max_safe
```

### Node 5: Stationarity Threshold

**Recommended thresholds for regime change detection:**
```
p-value threshold (ADF test): 0.05 (standard), 0.10 (sensitive)
Break detection window: 20-60 days
Minimum observations: 252 (1 year)
```

**Calibration via rolling ADF:**
```python
def calibrate_stationarity(returns, window=252):
    """Calibrate stationarity threshold via rolling ADF."""
    from statsmodels.tsa.stattools import adfuller

    p_values = []
    for i in range(window, len(returns)):
        result = adfuller(returns[i-window:i])
        p_values.append(result[1])

    # Set threshold at 95th percentile of calm period p-values
    return np.percentile(p_values, 95)
```

### Node 6: Capacity Utilization Threshold

**Recommended thresholds:**
```
Warning level:   70% of estimated market capacity
Critical level:  90% of estimated market capacity
Halt level:      95% of estimated market capacity
```

**Capacity estimation:**
```python
def estimate_capacity(volume_history, price_impact_history):
    """Estimate market capacity from impact regression."""
    # Market capacity ≈ volume at which impact exceeds 1%
    X = np.log(volume_history).reshape(-1, 1)
    y = np.abs(price_impact_history)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)

    # Solve for volume where impact = 0.01
    capacity = np.exp((0.01 - reg.intercept_) / reg.coef_[0])
    return capacity
```

### Node 11: Representation Accuracy Threshold

**Recommended thresholds:**
```
Maximum prediction error (RMSE): 2 × historical volatility
Maximum model complexity: Effective parameters < n/10
Maximum regime uncertainty: H(K) < log(|K|) - 0.5
```

**Calibration for model accuracy:**
```python
def calibrate_representation(predictions, actuals, vol_estimate):
    """Calibrate representation threshold."""
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    threshold = 2 * vol_estimate
    return rmse < threshold, threshold
```

## Barrier Threshold Calibration

### BarrierSat: Position Limits

**Recommended limits:**

| Position Type | Conservative | Balanced | Aggressive |
|---------------|-------------|----------|------------|
| Single name (% of NAV) | 5% | 10% | 20% |
| Sector (% of NAV) | 20% | 30% | 50% |
| Single name (% of ADV) | 5% | 10% | 25% |
| Gross exposure | 100% | 150% | 200% |

**Dynamic adjustment:**
```python
def adjust_position_limit(base_limit, vol_regime, liquidity_regime):
    """Adjust position limits based on market regime."""
    vol_factor = 1.0 / (1 + vol_regime)  # Reduce in high vol
    liq_factor = liquidity_regime  # Reduce in low liquidity
    return base_limit * vol_factor * liq_factor
```

### BarrierTypeII: Volatility-of-Volatility Crisis

**Recommended thresholds:**
```
Vol-of-vol warning:  VVIX/VIX > 5.0 (historical median ~4.5)
Vol-of-vol critical: VVIX/VIX > 7.0 (99th percentile)
Vol regime change:   VIX > 2 × 20-day MA
```

**Calibration from historical vol dynamics:**
```python
def calibrate_vol_of_vol(vix_history, vvix_history):
    """Calibrate vol-of-vol thresholds."""
    ratio = np.array(vvix_history) / np.array(vix_history)
    return {
        'warning': np.percentile(ratio, 75),
        'critical': np.percentile(ratio, 99),
        'median': np.median(ratio)
    }
```

### BarrierOmin: Flash Crash Detection

**Recommended thresholds:**
```
Price move threshold: -5% in 5 minutes (equities)
Volume spike: > 10× average 5-minute volume
Quote withdrawal: > 50% reduction in depth
Recovery time: < 30 minutes for temporary classification
```

**Real-time detection:**
```python
def detect_flash_crash(prices, volumes, depths, window_minutes=5):
    """Detect flash crash conditions."""
    price_change = (prices[-1] - prices[-window_minutes]) / prices[-window_minutes]
    volume_ratio = volumes[-window_minutes:].sum() / volumes[-60:-window_minutes].mean()
    depth_change = depths[-1] / depths[-window_minutes]

    flash_crash = (
        price_change < -0.05 and  # 5% drop
        volume_ratio > 10 and      # 10× volume spike
        depth_change < 0.5         # 50% depth reduction
    )
    return flash_crash, {'price': price_change, 'volume': volume_ratio, 'depth': depth_change}
```

### BarrierFreq: HFT Oscillation Detection

**Recommended thresholds:**
```
Price oscillation: > 3 reversals per minute
Quote flickering: > 100 updates per second
Layering detection: > 5 levels with < 100ms lifetime
Momentum ignition: Correlation(flow, returns) > 0.9 with reversal
```

**Detection algorithm:**
```python
def detect_hft_oscillation(prices, timestamps):
    """Detect HFT-induced oscillation."""
    # Count sign changes in 1-minute windows
    returns = np.diff(prices)
    sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
    duration_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60

    reversals_per_minute = sign_changes / duration_minutes
    return reversals_per_minute > 3, reversals_per_minute
```

## Regime-Dependent Calibration

Thresholds should adjust based on the macro regime $K_t$:

:::{prf:definition} Regime-Adjusted Threshold
:label: def-regime-threshold

For base threshold $\tau_0$ and regime $K$, the adjusted threshold is:
$$
\tau_K = \tau_0 \cdot \phi_K,
$$
where $\phi_K$ is the regime adjustment factor:

| Regime | $\phi_K$ | Interpretation |
|--------|----------|----------------|
| Risk-On | 1.2 | Looser thresholds |
| Neutral | 1.0 | Base thresholds |
| Risk-Off | 0.8 | Tighter thresholds |
| Crisis | 0.5 | Much tighter |
| Recovery | 0.9 | Slightly tight |
:::

**Regime detection and adjustment:**
```python
class RegimeAdjustedThresholds:
    """Adjust thresholds based on market regime."""

    def __init__(self, base_thresholds):
        self.base = base_thresholds
        self.phi = {
            'risk_on': 1.2,
            'neutral': 1.0,
            'risk_off': 0.8,
            'crisis': 0.5,
            'recovery': 0.9
        }

    def detect_regime(self, vix, credit_spread, momentum):
        """Simple regime detection."""
        if vix > 30 and credit_spread > 500:
            return 'crisis'
        elif vix > 25 or credit_spread > 300:
            return 'risk_off'
        elif vix < 15 and momentum > 0:
            return 'risk_on'
        elif vix < 20 and credit_spread < 200:
            return 'recovery' if self.prev_regime == 'crisis' else 'neutral'
        return 'neutral'

    def get_threshold(self, name, regime):
        """Get regime-adjusted threshold."""
        return self.base[name] * self.phi[regime]
```

## Cross-Validation and Backtesting

**Calibration validation procedure:**

1. **In-sample calibration:** Fit thresholds to 70% of historical data
2. **Out-of-sample validation:** Test on remaining 30%
3. **Stress period validation:** Ensure thresholds trigger appropriately during known crises
4. **False positive rate:** Target < 5% false alarms in calm periods
5. **True positive rate:** Target > 95% detection of known stress events

```python
def validate_calibration(thresholds, test_data, known_crises):
    """Validate calibration against test data."""
    predictions = []
    actuals = []

    for t in range(len(test_data)):
        # Check if any gate/barrier triggers
        triggered = any(
            test_data[t][gate] > thresholds[gate]
            for gate in thresholds.keys()
        )
        predictions.append(triggered)
        actuals.append(t in known_crises)

    # Compute metrics
    tp = sum(p and a for p, a in zip(predictions, actuals))
    fp = sum(p and not a for p, a in zip(predictions, actuals))
    fn = sum(not p and a for p, a in zip(predictions, actuals))
    tn = sum(not p and not a for p, a in zip(predictions, actuals))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'recall': recall,           # Target > 0.95
        'precision': precision,      # Target > 0.50
        'false_positive_rate': fpr   # Target < 0.05
    }
```

## Threshold Summary Table

```{list-table} Complete Threshold Summary
:header-rows: 1
:name: threshold-summary

* - Component
  - Parameter
  - Conservative
  - Balanced
  - Aggressive
* - Node 1 (Solvency)
  - NAV/Notional minimum
  - 0.20
  - 0.10
  - 0.05
* - Node 3 (Leverage)
  - Max equity leverage
  - 2.0
  - 3.0
  - 5.0
* - Node 5 (Stationarity)
  - ADF p-value
  - 0.10
  - 0.05
  - 0.01
* - Node 6 (Capacity)
  - Utilization warning
  - 0.60
  - 0.70
  - 0.80
* - Node 11 (Representation)
  - Max RMSE / vol
  - 1.5
  - 2.0
  - 3.0
* - BarrierSat
  - Single name % NAV
  - 0.05
  - 0.10
  - 0.20
* - BarrierTypeII
  - VVIX/VIX critical
  - 6.0
  - 7.0
  - 8.0
* - BarrierOmin
  - 5-min price drop
  - -0.03
  - -0.05
  - -0.10
* - BarrierFreq
  - Reversals/minute
  - 2
  - 3
  - 5
```

---

