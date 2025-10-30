# Gemini Batch Processing Instructions

## Generated Prompts

51 prompts generated in: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/data/gemini_prompts`

## Processing Method

Use the following approach to process all prompts:

### Option 1: Sequential Processing (Recommended)
```python
import json
from pathlib import Path

prompts_dir = Path('/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/data/gemini_prompts')
results_dir = prompts_dir.parent / 'gemini_results'
results_dir.mkdir(exist_ok=True)

for prompt_file in sorted(prompts_dir.glob('*.txt')):
    label = prompt_file.stem
    print(f"Processing: {label}")

    # Read prompt
    prompt = prompt_file.read_text()

    # Use Gemini via MCP
    # result = mcp__gemini-cli__ask-gemini(prompt=prompt, model="gemini-2.5-pro")

    # Save result
    # result_file = results_dir / f"{label}.json"
    # result_file.write_text(result)
```

### Option 2: Parallel Processing
Process 5-10 prompts in parallel using multiple MCP calls

## Theorem List

Total: 51 theorems

- lem-polishness-and-w2: Polishness of the quotient state space and $W_2$
- thm-revival-guarantee: Almost‑sure revival under the global constraint
- thm-mean-square-standardization-error: Asymptotic Behavior of the Mean-Square Standardization Error
- thm-forced-activity: Theorem of Forced Activity
- lem-boundary-uniform-ball: Uniform‑ball death probability is Lipschitz under finite perimeter
- lem-boundary-heat-kernel: Heat‑kernel death probability is Lipschitz with constant $\lesssim 1/\sigma$
- lem-empirical-aggregator-properties: Axiomatic Properties of the Empirical Measure Aggregator
- lem-set-difference-bound: Bound on the Error from Companion Set Change
- lem-normalization-difference-bound: Bound on the Error from Normalization Change
- thm-total-error-status-bound: Total Error Bound in Terms of Status Changes
- lem-cubic-patch-uniqueness: Existence and Uniqueness of the Smooth Rescale Patch
- lem-cubic-patch-coefficients: Explicit Coefficients of the Smooth Rescale Patch
- lem-cubic-patch-derivative: Explicit Form of the Polynomial Patch Derivative
- lem-polynomial-patch-monotonicity: Monotonicity of the Polynomial Patch
- lem-cubic-patch-derivative-bounds: Bounds on the Polynomial Patch Derivative
- lem-rescale-monotonicity: Monotonicity of the Smooth Rescale Function
- thm-rescale-function-lipschitz: Global Lipschitz Continuity of the Smooth Rescale Function
- lem-sigma-patch-derivative-bound: Derivative bound for \sigma\'_{\text{reg}}
- thm-canonical-logistic-validity: The Canonical Logistic Function is a Valid Rescale Function
- lem-single-walker-positional-error: Bound on Single-Walker Error from Positional Change
- lem-single-walker-structural-error: Bound on Single-Walker Error from Structural Change
- lem-single-walker-own-status-error: Bound on Single-Walker Error from Own Status Change
- lem-total-squared-error-unstable: Bound on the Total Squared Error for Unstable Walkers
- lem-total-squared-error-stable: Bound on the Total Squared Error for Stable Walkers
- thm-expected-raw-distance-bound: Bound on the Expected Raw Distance Vector Change
- thm-expected-raw-distance-k1: Deterministic Behavior of the Expected Raw Distance Vector at $k=1$
- thm-distance-operator-satisfies-bounded-variance-axiom: The Distance Operator Satisfies the Bounded Variance Axiom
- thm-distance-operator-mean-square-continuity: Mean-Square Continuity of the Distance Operator
- lem-stats-value-continuity: Value Continuity of Statistical Properties
- lem-stats-structural-continuity: Structural Continuity of Statistical Properties
- thm-z-score-norm-bound: General Bound on the Norm of the Standardized Vector
- thm-asymptotic-std-dev-structural-continuity: Asymptotic Behavior of the Structural Continuity for the Regularized Standard Deviation
- thm-standardization-value-error-mean-square: Bounding the Expected Squared Value Error
- thm-standardization-structural-error-mean-square: Bounding the Expected Squared Structural Error
- thm-deterministic-error-decomposition: Decomposition of the Total Standardization Error
- sub-lem-lipschitz-value-error-decomposition: Algebraic Decomposition of the Value Error
- thm-lipschitz-value-error-bound: Bounding the Squared Value Error
- thm-lipschitz-structural-error-bound: Bounding the Squared Structural Error
- thm-global-continuity-patched-standardization: Global Continuity of the Patched Standardization Operator
- sub-lem-perturbation-positional-bound-reproof: Bounding the Output Positional Displacement
- thm-mcdiarmids-inequality: McDiarmid's Inequality (Bounded Differences Inequality) (Boucheron–Lugosi–Massart)
- sub-lem-probabilistic-bound-perturbation-displacement-reproof: Probabilistic Bound on Total Perturbation-Induced Displacement
- thm-perturbation-operator-continuity-reproof: Probabilistic Continuity of the Perturbation Operator
- thm-post-perturbation-status-update-continuity: Probabilistic Continuity of the Post-Perturbation Status Update
- lem-cloning-probability-lipschitz: Lipschitz Continuity of the Conditional Cloning Probability Function (case split)
- thm-expected-cloning-action-continuity: Continuity of the Conditional Expected Cloning Action
- thm-total-expected-cloning-action-continuity: Continuity of the Total Expected Cloning Action
- thm-k1-revival-state: Theorem of Guaranteed Revival from a Single Survivor
- prop-w2-bound-no-offset: W2 continuity bound without offset (for $k\ge 2$)
- prop-psi-markov-kernel: The Swarm Update defines a Markov kernel
- prop-coefficient-regularity: Boundedness and continuity of composite coefficients
