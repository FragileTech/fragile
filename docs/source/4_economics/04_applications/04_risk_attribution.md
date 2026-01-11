# Risk Attribution Framework

This section provides a framework for **attributing risk and losses** to specific gate failures, barrier breaches, and failure modes.

## Hierarchical Risk Attribution

:::{prf:definition} Risk Attribution Decomposition
:label: def-risk-attribution

Total portfolio risk $\sigma^2$ decomposes hierarchically:
$$
\sigma^2 = \underbrace{\sigma^2_{\text{sys}}}_{\text{Systematic}} + \underbrace{\sigma^2_{\text{idio}}}_{\text{Idiosyncratic}} + \underbrace{\sigma^2_{\text{regime}}}_{\text{Regime}} + \underbrace{\sigma^2_{\text{barrier}}}_{\text{Barrier}}.
$$
:::

**Component definitions:**
- $\sigma^2_{\text{sys}}$: Risk from factor exposures (market, sector, style)
- $\sigma^2_{\text{idio}}$: Asset-specific residual risk
- $\sigma^2_{\text{regime}}$: Risk from potential regime changes
- $\sigma^2_{\text{barrier}}$: Risk from potential barrier breaches

## Gate-Based Risk Attribution

Each gate failure contributes to total risk. We attribute risk to gates based on **proximity to threshold**:

:::{prf:definition} Gate Risk Contribution
:label: def-gate-risk

For gate $i$ with current value $v_i$ and threshold $\tau_i$, the gate risk contribution is:
$$
R_i = w_i \cdot \max\left(0, \frac{v_i - \tau_i^{\text{warn}}}{\tau_i^{\text{crit}} - \tau_i^{\text{warn}}}\right)^2,
$$
where $w_i$ is the weight representing potential loss if gate $i$ fails.
:::

**Implementation:**
```python
class GateRiskAttributor:
    """Attribute risk to individual gates."""

    def __init__(self, gate_weights, warn_thresholds, crit_thresholds):
        self.weights = gate_weights  # Potential loss per gate
        self.warn = warn_thresholds
        self.crit = crit_thresholds

    def attribute(self, gate_values):
        """Compute risk attribution to each gate."""
        attributions = {}

        for gate_name, value in gate_values.items():
            warn = self.warn[gate_name]
            crit = self.crit[gate_name]
            weight = self.weights[gate_name]

            if value < warn:
                attributions[gate_name] = 0.0
            else:
                proximity = (value - warn) / (crit - warn)
                attributions[gate_name] = weight * min(proximity, 1.0)**2

        return attributions

    def total_gate_risk(self, gate_values):
        """Compute total risk from gate proximity."""
        return sum(self.attribute(gate_values).values())
```

## Failure Mode Risk Attribution

Risk attributed to each failure mode based on proximity and conditional severity:

:::{prf:definition} Failure Mode Risk
:label: def-fm-risk

For failure mode $F$ with probability $p_F$ and severity $s_F$:
$$
R_F = p_F \cdot s_F \cdot \mathbb{E}[\text{Loss} \mid F],
$$
where $p_F$ is estimated from gate/barrier states.
:::

**Failure mode severity table:**

| Mode | Base Severity | Typical Loss | Recovery Time |
|------|--------------|--------------|---------------|
| C.E (Blow-up) | 10 | 50-100% | Permanent |
| C.D (Concentration) | 7 | 20-50% | 1-6 months |
| C.C (Zeno) | 5 | 5-20% | Days-weeks |
| T.E (Flash crash) | 6 | 10-30% | Hours-days |
| T.D (Frozen) | 8 | 20-40% | Weeks-months |
| T.C (Complexity) | 6 | 10-30% | Weeks |
| D.E (Oscillation) | 7 | 15-40% | Months |
| D.D (Dispersion) | 3 | Gains | N/A |
| D.C (Undecidable) | 8 | Unknown | Unknown |
| S.E (Supercritical) | 9 | 30-70% | Months |
| S.D (Flat vol) | 2 | Opportunity cost | N/A |
| S.C (Drift) | 5 | 10-25% | Ongoing |
| B.E (External) | 8 | 20-50% | Variable |
| B.D (Starvation) | 6 | 15-35% | Weeks |
| B.C (Misalignment) | 4 | 5-20% | Variable |

**Implementation:**
```python
class FailureModeAttributor:
    """Attribute risk to failure modes."""

    SEVERITIES = {
        'C.E': 10, 'C.D': 7, 'C.C': 5,
        'T.E': 6, 'T.D': 8, 'T.C': 6,
        'D.E': 7, 'D.D': 3, 'D.C': 8,
        'S.E': 9, 'S.D': 2, 'S.C': 5,
        'B.E': 8, 'B.D': 6, 'B.C': 4
    }

    EXPECTED_LOSS = {
        'C.E': 0.75, 'C.D': 0.35, 'C.C': 0.12,
        'T.E': 0.20, 'T.D': 0.30, 'T.C': 0.20,
        'D.E': 0.27, 'D.D': -0.10, 'D.C': 0.50,
        'S.E': 0.50, 'S.D': 0.05, 'S.C': 0.17,
        'B.E': 0.35, 'B.D': 0.25, 'B.C': 0.12
    }

    def estimate_probability(self, mode, gate_states, barrier_states):
        """Estimate failure mode probability from current states."""
        # Map failure modes to relevant gates/barriers
        mode_gates = {
            'C.E': ['solvency', 'turnover'],
            'C.D': ['solvency', 'capacity'],
            'T.E': ['connectivity', 'stiffness'],
            'D.E': ['oscillation', 'stability'],
            'S.E': ['leverage', 'stationarity'],
            'B.E': ['coupling', 'input'],
            # ... etc
        }

        relevant_gates = mode_gates.get(mode, [])
        if not relevant_gates:
            return 0.01  # Base rate

        # Probability increases with gate proximity to failure
        gate_risks = [gate_states.get(g, 0) for g in relevant_gates]
        return min(1.0, np.mean(gate_risks) * 2)

    def attribute(self, gate_states, barrier_states):
        """Compute risk attribution to each failure mode."""
        attributions = {}

        for mode in self.SEVERITIES:
            p = self.estimate_probability(mode, gate_states, barrier_states)
            s = self.SEVERITIES[mode] / 10  # Normalize to [0, 1]
            loss = self.EXPECTED_LOSS[mode]

            if loss > 0:  # Only attribute negative outcomes
                attributions[mode] = p * s * loss

        return attributions
```

## Loss Attribution Post-Event

After a loss event, attribute the loss to specific causes:

```python
class LossAttributor:
    """Attribute realized losses to causes."""

    def attribute_loss(self, loss, pre_event_state, post_event_state, timeline):
        """
        Attribute a realized loss to gates, barriers, and failure modes.

        Args:
            loss: Total loss amount
            pre_event_state: State before event
            post_event_state: State after event
            timeline: List of (time, event) during the event
        """
        attribution = {
            'gates': {},
            'barriers': {},
            'failure_modes': {},
            'unexplained': 0.0
        }

        # Identify which gates failed
        for gate in pre_event_state['gates']:
            if pre_event_state['gates'][gate] == 'PASS' and \
               post_event_state['gates'][gate] == 'FAIL':
                attribution['gates'][gate] = self._estimate_gate_contribution(
                    gate, loss, pre_event_state, post_event_state
                )

        # Identify which barriers breached
        for barrier in pre_event_state['barriers']:
            if pre_event_state['barriers'][barrier] == 'CLEAR' and \
               post_event_state['barriers'][barrier] == 'BREACHED':
                attribution['barriers'][barrier] = self._estimate_barrier_contribution(
                    barrier, loss, pre_event_state, post_event_state
                )

        # Map to failure modes
        attribution['failure_modes'] = self._identify_failure_modes(
            attribution['gates'], attribution['barriers']
        )

        # Unexplained residual
        explained = (sum(attribution['gates'].values()) +
                    sum(attribution['barriers'].values()))
        attribution['unexplained'] = max(0, loss - explained)

        return attribution
```

## Risk Attribution Dashboard

**Key metrics for risk monitoring:**

```python
class RiskDashboard:
    """Real-time risk attribution dashboard."""

    def compute_metrics(self, portfolio, market_state):
        """Compute dashboard metrics."""
        return {
            # Level 1: Summary
            'total_var_95': self.compute_var(portfolio, 0.95),
            'total_es_95': self.compute_es(portfolio, 0.95),

            # Level 2: Category attribution
            'systematic_risk': self.systematic_attribution(portfolio),
            'idiosyncratic_risk': self.idiosyncratic_attribution(portfolio),
            'regime_risk': self.regime_attribution(portfolio, market_state),
            'barrier_risk': self.barrier_attribution(portfolio, market_state),

            # Level 3: Gate proximity
            'gate_risk_scores': self.gate_attributor.attribute(market_state['gates']),
            'nearest_gate_to_fail': self.find_nearest_gate(market_state['gates']),

            # Level 4: Failure mode probabilities
            'failure_mode_risks': self.fm_attributor.attribute(
                market_state['gates'], market_state['barriers']
            ),
            'dominant_failure_mode': self.find_dominant_mode(market_state),

            # Level 5: Action recommendations
            'recommended_hedges': self.recommend_hedges(portfolio, market_state),
            'recommended_reductions': self.recommend_reductions(portfolio, market_state)
        }
```

---

