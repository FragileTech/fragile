# Uniform-margin obstruction: completed proof record

The complete statements and proofs now appear in the published chapter
[Population-uniform Keystone estimates for the coupled gas](../../source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md):

- `prop-ku-terminal-mark-quadratic-obstruction` proves a linear terminal-mark
  response against a quadratic input at exact fitness ties, for every
  coupling of both raw and survivor-conditioned output laws. A uniformly
  vanishing affine error floor is impossible.
- `prop-ku-singleton-revival-uniform-obstruction` proves that mandatory
  revival broadcasts one perturbed donor to every slot. It excludes a
  population-independent fixed-slot quadratic coefficient, even with no
  mark cost and under optimal coupling, for raw and survivor-conditioned
  kernels.
- `thm-ku-allalive-source-coupling-obstruction` proves a linear physical
  contribution under the prescribed maximal measurement/source coupling,
  with all entering walkers alive and positive measured cloning activity.
  Both kinetic kicks, unbounded noises, cap and terminal marking are kept.
- `rem-ku-allalive-source-coupling-obstruction-scope` distinguishes that
  prescribed-coupling result from an optimal Wasserstein lower bound.

The all-alive calculation and its proof were independently audited before
integration. Its original local equation tags `KU.O1`--`KU.O7` are
`KU.O14`--`KU.O20` in the published chapter. The singleton construction
uses identical dead positions outside the terminal box so its input marks
are consistent. The survivor extension uses exponentially small extinction
and an explicit Gaussian moment bound.

These results refute the respective full-array or prescribed-coupling
quadratic margins. They do not refute optimal alive empirical transport
or transport of the alive-sampled survivor marginal. A lower bound for
one selected coupling does not lower-bound the transport infimum, and
alive sampling removes dead coordinates and changes the normalization.
`prop-ku-prescribed-coupling-versus-optimal-transport` and
`prop-w2-prescribed-coupling-scope` record those distinctions.

`lem-ku-gaussian-mixture-weight-smoothing` gives a complete elementary
example where a selected source coupling costs linearly in a mixture-weight
change, while an alternative Gaussian-smoothed transport costs quadratically.
Its independent-product extension is conditional on row independence; it
is not a proof of the globally standardized gas's contraction.

The separate `prop-ku-rastrigin-alive-transport-expansion` concerns actual
optimal positional transport. Consensus starts near the edge of the same
deterministic well expand in one update, with death disabled and with
nonextinction-conditioned alive sampling. The complete proof integrates
the original Gaussian noise, computes the actual conditional alive law,
and establishes a uniform derivative bound for its truncated Gaussian mean.
This establishes failure of global one-step monotonicity, rather than
failure of later mixing in law or of a block convergence estimate.

The finite-population QSD theorem remains valid. A population-independent
alive-law rate requires its actual landscape/coercivity and coupling
conditions, or a separately proved global mixing comparison. A shared
phase name alone and an assumed favorable remainder do not supply them.
