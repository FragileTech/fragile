# Backtesting Framework

This section provides a rigorous backtesting framework for validating the Market Sieve on historical data.

## Backtesting Objectives

:::{prf:remark} Backtesting Goals
:label: rem-backtest-goals

The backtesting framework aims to:
1. **Validate gate thresholds** against historical crises
2. **Measure false positive/negative rates** for barriers
3. **Test surgery effectiveness** on historical interventions
4. **Calibrate regime detection** accuracy
5. **Estimate economic value** of the Sieve
:::

## Historical Event Database

**Required data structure:**

```python
@dataclass
class HistoricalEvent:
    """Documented market stress event."""
    name: str                    # e.g., "2008 Financial Crisis"
    start_date: datetime
    end_date: datetime
    peak_date: datetime          # Maximum stress
    asset_classes: List[str]     # Affected asset classes
    failure_modes: List[str]     # Failure modes observed
    interventions: List[str]     # Surgeries applied
    max_drawdown: float          # Peak-to-trough loss
    recovery_time: int           # Days to recovery
    gates_that_failed: List[str] # Gates that should have triggered
    barriers_breached: List[str] # Barriers that should have triggered

HISTORICAL_EVENTS = [
    HistoricalEvent(
        name="1987 Black Monday",
        start_date=datetime(1987, 10, 14),
        end_date=datetime(1987, 10, 26),
        peak_date=datetime(1987, 10, 19),
        asset_classes=["equities", "options"],
        failure_modes=["T.E", "D.E"],
        interventions=["SurgTE"],
        max_drawdown=0.226,
        recovery_time=452,
        gates_that_failed=["stiffness", "oscillation", "connectivity"],
        barriers_breached=["BarrierOmin", "BarrierTypeII"]
    ),
    HistoricalEvent(
        name="1998 LTCM",
        start_date=datetime(1998, 8, 1),
        end_date=datetime(1998, 10, 15),
        peak_date=datetime(1998, 9, 23),
        asset_classes=["credit", "fx", "equity_vol"],
        failure_modes=["C.E", "S.E", "T.D"],
        interventions=["SurgCE", "SurgSE"],
        max_drawdown=0.44,
        recovery_time=90,
        gates_that_failed=["solvency", "leverage", "mixing"],
        barriers_breached=["BarrierTypeII", "BarrierGap", "BarrierLev"]
    ),
    HistoricalEvent(
        name="2008 Financial Crisis",
        start_date=datetime(2008, 9, 1),
        end_date=datetime(2009, 3, 9),
        peak_date=datetime(2008, 10, 10),
        asset_classes=["credit", "equities", "real_estate"],
        failure_modes=["C.E", "C.D", "T.D", "S.E"],
        interventions=["SurgCE", "SurgCD", "SurgTD", "SurgSE"],
        max_drawdown=0.569,
        recovery_time=1403,
        gates_that_failed=["solvency", "leverage", "connectivity", "mixing", "stationarity"],
        barriers_breached=["BarrierSat", "BarrierTypeII", "BarrierGap", "BarrierLev", "BarrierDef"]
    ),
    HistoricalEvent(
        name="2010 Flash Crash",
        start_date=datetime(2010, 5, 6),
        end_date=datetime(2010, 5, 6),
        peak_date=datetime(2010, 5, 6),
        asset_classes=["equities", "etfs"],
        failure_modes=["T.E", "C.C"],
        interventions=["SurgTE", "SurgCC"],
        max_drawdown=0.099,
        recovery_time=1,
        gates_that_failed=["stiffness", "connectivity"],
        barriers_breached=["BarrierOmin", "BarrierFreq"]
    ),
    HistoricalEvent(
        name="2015 CNH Devaluation",
        start_date=datetime(2015, 8, 11),
        end_date=datetime(2015, 8, 25),
        peak_date=datetime(2015, 8, 24),
        asset_classes=["fx", "em_equities"],
        failure_modes=["B.E", "D.E"],
        interventions=["SurgBE"],
        max_drawdown=0.12,
        recovery_time=60,
        gates_that_failed=["stationarity", "coupling"],
        barriers_breached=["BarrierCausal", "BarrierInput"]
    ),
    HistoricalEvent(
        name="2018 Volmageddon",
        start_date=datetime(2018, 2, 2),
        end_date=datetime(2018, 2, 9),
        peak_date=datetime(2018, 2, 5),
        asset_classes=["equity_vol", "etfs"],
        failure_modes=["S.E", "D.E", "C.E"],
        interventions=["SurgSE"],
        max_drawdown=0.12,
        recovery_time=14,
        gates_that_failed=["bifurcation", "stability", "leverage"],
        barriers_breached=["BarrierTypeII", "BarrierOmin"]
    ),
    HistoricalEvent(
        name="2020 COVID Crash",
        start_date=datetime(2020, 2, 20),
        end_date=datetime(2020, 3, 23),
        peak_date=datetime(2020, 3, 16),
        asset_classes=["all"],
        failure_modes=["B.E", "T.D", "S.E", "C.E"],
        interventions=["SurgBE", "SurgTD", "SurgSE", "SurgCE"],
        max_drawdown=0.339,
        recovery_time=148,
        gates_that_failed=["connectivity", "stationarity", "coupling", "leverage"],
        barriers_breached=["BarrierOmin", "BarrierTypeII", "BarrierGap", "BarrierInput"]
    ),
    HistoricalEvent(
        name="2022 LDI Crisis",
        start_date=datetime(2022, 9, 23),
        end_date=datetime(2022, 10, 14),
        peak_date=datetime(2022, 9, 28),
        asset_classes=["gilts", "gbp"],
        failure_modes=["C.E", "S.E", "T.D"],
        interventions=["SurgCE", "SurgSE", "SurgTD"],
        max_drawdown=0.25,
        recovery_time=21,
        gates_that_failed=["solvency", "leverage", "stiffness"],
        barriers_breached=["BarrierLev", "BarrierGap", "BarrierTypeII"]
    ),
]
```

## Backtesting Metrics

:::{prf:definition} Sieve Performance Metrics
:label: def-backtest-metrics

For a set of $N$ historical events, the Sieve is evaluated on:

1. **Detection rate:** $DR = \frac{\text{Events where any gate/barrier triggered}}{\text{Total events}}$

2. **Early warning rate:** $EW = \frac{\text{Events with trigger} \ge 5 \text{ days before peak}}{\text{Total events}}$

3. **False positive rate:** $FPR = \frac{\text{Triggers in non-event periods}}{\text{Total non-event days}}$

4. **Failure mode accuracy:** $FMA = \frac{\text{Correctly identified failure modes}}{\text{Actual failure modes}}$

5. **Economic value:** $EV = \sum_{\text{events}} (\text{Loss avoided by early exit}) - (\text{Opportunity cost of false exits})$
:::

## Backtesting Implementation

```python
class SieveBacktester:
    """Backtest the Market Sieve on historical data."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events
        self.results = {}

    def run_backtest(self):
        """Run full backtest across all events."""
        for event in self.events:
            self.results[event.name] = self._backtest_event(event)

        self._compute_aggregate_metrics()
        return self.results

    def _backtest_event(self, event: HistoricalEvent):
        """Backtest single event."""
        result = {
            'detected': False,
            'first_trigger_date': None,
            'days_before_peak': None,
            'gates_triggered': [],
            'barriers_triggered': [],
            'failure_modes_detected': [],
            'correct_gates': [],
            'missed_gates': [],
            'false_gates': [],
            'loss_at_trigger': None,
            'loss_at_peak': None,
            'loss_avoided': None
        }

        # Run sieve on each day from 30 days before start to event end
        start_window = event.start_date - timedelta(days=30)

        for date in self._date_range(start_window, event.end_date):
            state = self.data.get_state(date)

            # Run sieve
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                if not result['detected']:
                    result['detected'] = True
                    result['first_trigger_date'] = date
                    result['days_before_peak'] = (event.peak_date - date).days
                    result['loss_at_trigger'] = self._compute_loss(
                        event.start_date, date, event.asset_classes
                    )

                # Record triggers
                result['gates_triggered'].extend(certificate.get('failed_gates', []))
                result['barriers_triggered'].extend(certificate.get('breached_barriers', []))

        # Compute accuracy metrics
        result['gates_triggered'] = list(set(result['gates_triggered']))
        result['barriers_triggered'] = list(set(result['barriers_triggered']))

        result['correct_gates'] = [
            g for g in result['gates_triggered']
            if g in event.gates_that_failed
        ]
        result['missed_gates'] = [
            g for g in event.gates_that_failed
            if g not in result['gates_triggered']
        ]
        result['false_gates'] = [
            g for g in result['gates_triggered']
            if g not in event.gates_that_failed
        ]

        # Loss computation
        result['loss_at_peak'] = event.max_drawdown
        if result['loss_at_trigger'] is not None:
            result['loss_avoided'] = event.max_drawdown - result['loss_at_trigger']

        # Failure mode detection
        result['failure_modes_detected'] = self._identify_failure_modes_from_triggers(
            result['gates_triggered'], result['barriers_triggered']
        )

        return result

    def _compute_aggregate_metrics(self):
        """Compute aggregate backtest metrics."""
        n_events = len(self.events)

        # Detection rate
        detected = sum(1 for r in self.results.values() if r['detected'])
        self.results['_aggregate'] = {
            'detection_rate': detected / n_events,

            # Early warning rate (>= 5 days before peak)
            'early_warning_rate': sum(
                1 for r in self.results.values()
                if r['days_before_peak'] is not None and r['days_before_peak'] >= 5
            ) / n_events,

            # Average days of warning
            'avg_warning_days': np.mean([
                r['days_before_peak'] for r in self.results.values()
                if r['days_before_peak'] is not None
            ]),

            # Gate accuracy
            'gate_precision': self._compute_gate_precision(),
            'gate_recall': self._compute_gate_recall(),

            # Total loss avoided
            'total_loss_avoided': sum(
                r['loss_avoided'] for r in self.results.values()
                if r['loss_avoided'] is not None
            ),

            # Average loss avoided
            'avg_loss_avoided': np.mean([
                r['loss_avoided'] for r in self.results.values()
                if r['loss_avoided'] is not None
            ])
        }

    def _compute_gate_precision(self):
        """Compute precision of gate triggers."""
        correct = sum(len(r['correct_gates']) for r in self.results.values() if isinstance(r, dict))
        total = sum(len(r['gates_triggered']) for r in self.results.values() if isinstance(r, dict))
        return correct / total if total > 0 else 0

    def _compute_gate_recall(self):
        """Compute recall of gate triggers."""
        correct = sum(len(r['correct_gates']) for r in self.results.values() if isinstance(r, dict))
        expected = sum(len(e.gates_that_failed) for e in self.events)
        return correct / expected if expected > 0 else 0

    def generate_report(self):
        """Generate backtest report."""
        agg = self.results.get('_aggregate', {})

        report = f"""
# Market Sieve Backtest Report

## Summary Metrics
- Detection Rate: {agg.get('detection_rate', 0):.1%}
- Early Warning Rate (≥5 days): {agg.get('early_warning_rate', 0):.1%}
- Average Warning Days: {agg.get('avg_warning_days', 0):.1f}
- Gate Precision: {agg.get('gate_precision', 0):.1%}
- Gate Recall: {agg.get('gate_recall', 0):.1%}
- Total Loss Avoided: {agg.get('total_loss_avoided', 0):.1%}

## Event-by-Event Results
"""

        for event in self.events:
            r = self.results[event.name]
            report += f"""
### {event.name}
- Detected: {'✓' if r['detected'] else '✗'}
- Warning Days: {r['days_before_peak']}
- Gates Triggered: {', '.join(r['gates_triggered']) or 'None'}
- Barriers Triggered: {', '.join(r['barriers_triggered']) or 'None'}
- Loss at Trigger: {r['loss_at_trigger']:.1%} if r['loss_at_trigger'] else 'N/A'
- Loss Avoided: {r['loss_avoided']:.1%} if r['loss_avoided'] else 'N/A'
"""

        return report
```

## False Positive Analysis

**Measuring false positives in non-crisis periods:**

```python
class FalsePositiveAnalyzer:
    """Analyze false positive rates."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events

    def compute_false_positive_rate(self, start_date, end_date):
        """Compute FPR over a date range excluding known events."""
        # Create mask of event periods
        event_dates = set()
        for event in self.events:
            for date in self._date_range(event.start_date, event.end_date):
                event_dates.add(date)

        non_event_days = 0
        false_triggers = 0

        for date in self._date_range(start_date, end_date):
            if date in event_dates:
                continue

            non_event_days += 1
            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                false_triggers += 1

        fpr = false_triggers / non_event_days if non_event_days > 0 else 0

        return {
            'false_positive_rate': fpr,
            'false_triggers': false_triggers,
            'non_event_days': non_event_days,
            'annualized_false_triggers': fpr * 252  # Trading days per year
        }

    def analyze_false_positives(self, start_date, end_date):
        """Detailed analysis of false positives."""
        event_dates = set()
        for event in self.events:
            for date in self._date_range(event.start_date, event.end_date):
                event_dates.add(date)

        false_positives = []

        for date in self._date_range(start_date, end_date):
            if date in event_dates:
                continue

            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)

            if certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                # Analyze what triggered and why
                false_positives.append({
                    'date': date,
                    'gates': certificate.get('failed_gates', []),
                    'barriers': certificate.get('breached_barriers', []),
                    'subsequent_5d_return': self._compute_forward_return(date, 5),
                    'subsequent_20d_return': self._compute_forward_return(date, 20),
                    'was_near_event': self._is_near_event(date)
                })

        return false_positives
```

## Economic Value Computation

**Computing the economic value of the Sieve:**

```python
class EconomicValueComputer:
    """Compute economic value of the Sieve."""

    def __init__(self, sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.sieve = sieve
        self.data = data
        self.events = events

    def compute_value(self, initial_capital=100, exit_strategy='immediate'):
        """
        Compute economic value of using the Sieve.

        exit_strategy options:
        - 'immediate': Exit fully on trigger
        - 'gradual': Reduce 50% on warning, 100% on critical
        - 'hedge': Buy protection instead of exiting
        """

        # Baseline: Buy and hold through all events
        baseline_path = self._compute_baseline_path(initial_capital)

        # Sieve-protected path
        protected_path = self._compute_protected_path(initial_capital, exit_strategy)

        # Compute metrics
        baseline_final = baseline_path[-1]
        protected_final = protected_path[-1]

        baseline_dd = self._max_drawdown(baseline_path)
        protected_dd = self._max_drawdown(protected_path)

        baseline_sharpe = self._sharpe_ratio(baseline_path)
        protected_sharpe = self._sharpe_ratio(protected_path)

        return {
            'baseline_final_value': baseline_final,
            'protected_final_value': protected_final,
            'value_added': protected_final - baseline_final,
            'value_added_pct': (protected_final - baseline_final) / baseline_final,
            'baseline_max_drawdown': baseline_dd,
            'protected_max_drawdown': protected_dd,
            'drawdown_reduction': baseline_dd - protected_dd,
            'baseline_sharpe': baseline_sharpe,
            'protected_sharpe': protected_sharpe,
            'sharpe_improvement': protected_sharpe - baseline_sharpe
        }

    def _compute_protected_path(self, initial_capital, exit_strategy):
        """Compute capital path with Sieve protection."""
        capital = initial_capital
        path = [capital]
        in_market = True
        exit_date = None
        reentry_date = None

        for date in self._all_dates():
            state = self.data.get_state(date)
            certificate = self.sieve.check_full_sieve(state)
            daily_return = self.data.get_return(date)

            # Exit logic
            if in_market and certificate['status'] in ['FAIL', 'BLOCKED', 'BREACHED']:
                if exit_strategy == 'immediate':
                    in_market = False
                    exit_date = date
                elif exit_strategy == 'gradual':
                    # Reduce exposure
                    capital *= (1 + 0.5 * daily_return)  # 50% exposure
                    if certificate['severity'] == 'CRITICAL':
                        in_market = False
                        exit_date = date

            # Re-entry logic: wait for all-clear for 5 consecutive days
            if not in_market:
                if certificate['status'] == 'VALID':
                    if reentry_date is None:
                        reentry_date = date
                    elif (date - reentry_date).days >= 5:
                        in_market = True
                        reentry_date = None
                else:
                    reentry_date = None

            # Apply return
            if in_market:
                capital *= (1 + daily_return)
            else:
                capital *= (1 + self.data.get_risk_free_rate(date) / 252)  # Cash return

            path.append(capital)

        return path
```

## Sensitivity Analysis

**Testing sensitivity to threshold choices:**

```python
class ThresholdSensitivityAnalyzer:
    """Analyze sensitivity to threshold choices."""

    def __init__(self, base_sieve: MarketSieve, data: MarketData, events: List[HistoricalEvent]):
        self.base_sieve = base_sieve
        self.data = data
        self.events = events

    def sweep_threshold(self, gate_name, threshold_range):
        """Sweep a single threshold and measure impact."""
        results = []

        for threshold in threshold_range:
            # Create modified sieve
            modified_sieve = self._create_modified_sieve(gate_name, threshold)

            # Run backtest
            backtester = SieveBacktester(modified_sieve, self.data, self.events)
            backtest_results = backtester.run_backtest()
            agg = backtest_results['_aggregate']

            # Run FPR analysis
            fpr_analyzer = FalsePositiveAnalyzer(modified_sieve, self.data, self.events)
            fpr_results = fpr_analyzer.compute_false_positive_rate(
                self.data.start_date, self.data.end_date
            )

            results.append({
                'threshold': threshold,
                'detection_rate': agg['detection_rate'],
                'early_warning_rate': agg['early_warning_rate'],
                'false_positive_rate': fpr_results['false_positive_rate'],
                'f1_score': self._f1_score(agg['detection_rate'], fpr_results['false_positive_rate'])
            })

        return results

    def find_optimal_threshold(self, gate_name, threshold_range, objective='f1'):
        """Find optimal threshold for a gate."""
        results = self.sweep_threshold(gate_name, threshold_range)

        if objective == 'f1':
            best = max(results, key=lambda x: x['f1_score'])
        elif objective == 'detection':
            # Maximize detection subject to FPR < 5%
            valid = [r for r in results if r['false_positive_rate'] < 0.05]
            best = max(valid, key=lambda x: x['detection_rate']) if valid else results[0]
        elif objective == 'economic':
            # Maximize economic value
            best = max(results, key=lambda x: x.get('economic_value', 0))

        return best['threshold']
```

---

