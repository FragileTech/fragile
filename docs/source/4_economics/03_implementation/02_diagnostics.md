# Implementation and Diagnostics

A practical deployment uses the Sieve as a runtime auditor:
- **Data feeds:** prices, order flow, balance sheet aggregates, funding rates.
- **Checks:** gate and barrier permits evaluated each step.
- **Output:** price estimates with certificate states (valid, bounded, or suspended).

A model is acceptable only if it produces **certificate-backed prices** in the intended operating regime.

---

