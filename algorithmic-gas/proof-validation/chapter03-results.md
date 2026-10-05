# Chapter 3 validation results

The Rust source-expression gate and native proposal experiment matrix both pass.
This run closes the 37 remaining source-expression gaps and adds independent
measured-versus-predicted comparisons for the cloning estimates.

| Validation | Result |
|---|---:|
| Required local Chapter 3 source expressions with executed evidence | 1,378 / 1,378 |
| Executed formula evidence records | 7,700 |
| Numerical formula comparisons | 1,416,406 |
| Failed numerical comparisons / hypotheses / unmatched evidence | 0 / 0 / 0 |
| Native proposal cases | 102 |
| Independent repetitions per case | 1,024 |
| Native proposals | 208,896 |
| Measured-versus-predicted case comparisons | 1,062 |
| Rejected statistical comparisons | 0 |

The [source-expression summary](../outputs/convergence/chapter03-complete/summary.md)
links the full ledger of quoted formulas, inputs, hypotheses, measured values and
slack. The [empirical comparison table](../outputs/convergence/chapter03-empirical/summary.md)
contains every case, observed mean, theoretical prediction, residual and standard
error. Numerical checks of a formula on recorded inputs do not prove its universal
validity.

The native matrix uses N=4/16/64, d=1/2, quadratic, sphere, Rastrigin and constant
landscapes, and four profiles: canonical, changed selection, half alive, and one
survivor. Entering dead coordinates have magnitude 1e9. The algorithm revives them
from current live donors before the measured proposal stage. All scalar moments
are normalized empirical quantities, and transport minimizes over couplings;
storage order supplies no intrinsic walker identity.

| Chapter 3 quantity | Native measurement and prediction | Comparisons | Rejected | One-sided support* |
|---|---|---:|---:|---:|
| Exact positional variance | Proposal variance versus retained row-law conditional mean | 96 | 0 | — |
| Positional reset B_x | Proposal variance versus diameter/jitter upper bound | 96 | 0 | 87 |
| Barycenter covariance | Squared displacement from conditional center versus exact covariance | 96 | 0 | — |
| Barycenter concentration bound | Same displacement versus (diameter² + d·jitter²)/N | 96 | 0 | 79 |
| Component velocity dissipation | Velocity variance versus initial full-slot variance minus (1−α²) times actual component energy | 96 | 0 | — |
| Alive velocity drift C_v | Proposal minus entering alive-normalized variance versus 0 or 8 V_max² | 96 | 0 | 96 |
| Declared quadratic boundary integral | Mean squared position norm versus exact conditional Gaussian/source moment | 96 | 0 | — |
| Conditional boundary drift | Boundary moment versus retained-pressure affine bound | 96 | 0 | 96 |
| Weighted signed internal drift | Weighted positional, velocity and boundary increments versus conditional balance | 96 | 0 | — |
| Inter-swarm transport C_W | Optimal physical hypocoercive transport between independent proposals versus 4 M_h | 96 | 0 | 96 |
| Complete two-swarm weighted drift | Transport plus both internal variance and boundary increments versus assembled bound | 96 | 0 | 96 |
| Keystone feedback χ | Actual error-weighted acceptance activity versus χ V_struct | 6 | 0 | 6 |

*One-sided support means the six-standard-error residual interval lies on the
predicted side of the bound, allowing numerical roundoff. A comparison can be
compatible with a bound without establishing one-sided support. These intervals
are sampling diagnostics, not simultaneous confidence certificates. Standard
errors use independent engine repetitions, never interacting walker rows.

The Keystone coefficient is fixed before simulation at
χ=(4/9) A_0(0.5)=0.302519153. The reference clouds have radii 1 and 0.5,
zero velocity, and canonical quadratic rewards.

| N | d | Measured Q/V_struct | Standard error | Theoretical lower bound |
|---:|---:|---:|---:|---:|
| 4 | 1 | 0.474121 | 0.00752 | 0.302519 |
| 4 | 2 | 0.454834 | 0.00730 | 0.302519 |
| 16 | 1 | 0.498474 | 0.00397 | 0.302519 |
| 16 | 2 | 0.494019 | 0.00396 | 0.302519 |
| 64 | 1 | 0.494522 | 0.00204 | 0.302519 |
| 64 | 2 | 0.497772 | 0.00196 | 0.302519 |

![Keystone feedback against population](../outputs/convergence/chapter03-empirical/keystone-feedback.png)

![Positional reset against population](../outputs/convergence/chapter03-empirical/positional-reset.png)

The source-expression suite also checks the affine boundary recurrence and its
geometric tail, signed fitness flux, target-error capture, Gaussian moments,
two-walker fitness ties, singleton donors and all-dead termination. Rate assembly
and decay envelopes are checked under their recorded component hypotheses. The
native proposal matrix retains zero-pressure realizations; it does not infer a
positive uniform boundary rate from those cases. χ measures feedback, rather than
a fitted stationary-law convergence rate. Some expansion bounds have substantial
slack, and cloning proposal drift need not be negative.

This completes finite-input coverage of Chapter 3's local source-expression
catalog and the listed native proposal comparisons. The stationary-law and
survival-conditioned QSD results referenced from later chapters require their
own hypotheses and law-level experiments; this gate does not certify them.
Trajectory decay remains measured by the separate `gas-decay` suite described in
the [validation guide](README.md).

The frozen Chapter 3 source SHA-256 is
`3db9c94b054593ed85bf4119fa5e6cb0216ab8b13d75c4ae982ec571768b81e1`.
Both strict commands, summary generation and plots are reproducible using the
[Chapter 3 commands](README.md#complete-chapter-3-validation). Focused validation
completed with 34 Rust tests, seven Python tests, all-target Clippy with warnings
denied, Ruff, rustfmt and whitespace checks passing.
