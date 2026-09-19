# QFT operator parity fixtures

Reference values of the Python QFT pipeline (`src/fragile/physics`) for the Rust spectroscopy
parity tests. They cover only what the QFT audit judged correct; everything the audit lists as a
defect is excluded (see [Parity exclusions](#parity-exclusions)) because Rust pins the corrected
behaviour there.

Regenerate from the repository root (the output is byte-identical between runs):

```
uv run python algorithmic-gas/tools/export_qft_fixtures.py
```

The exporter builds `PreparedChannelData` directly from seeded synthetic float64 tensors, not from
a `RunHistory`, and calls the real Python operator functions. It asserts, for every float32 series
it writes, that a float64 masked mean of the exported per-pair / per-triplet values reproduces it
to 1e-6, so the per-element tables and the frame series of one file are mutually consistent.

## Cases

| File | N | Companion maps | What it exercises |
|---|---|---|---|
| `non_involutive.json` | 10 | independent uniform `c(i) != i` for both roles, never an involution | exchange-odd channels are non-trivial; every colour valid, every walker alive |
| `mutual_pairing_odd.json` | 9 | `random_pairing_fisher_yates(9)` per frame and role: `c(c(i)) = i`, exactly one self companion per frame and role | audit D1 cancellation; self-companion masking; `lambda_alg = 0.5` with velocities |
| `masked.json` | 10 | uniform, then edited | dead walkers, zero-force colours (frame 5 entirely), indices `-1`, `N`, `N+3`, self companions, 4 coincident position pairs |

All cases: `T = 12` frames, `d = 3`. In `masked.json` a dead walker always has an invalid colour,
an alive walker is never paired with a dead one (the algorithm never does that, and Python's
electroweak operators do not check `alive[j]`), and some valid colours are paired with invalid
ones.

## Conventions

- Arrays are nested JSON lists in row-major order with the shape given below; `T` = frame,
  `N` = walker, `P` = pair slot (`0` = distance companion, `1` = cloning companion).
- Companion indices are ABSOLUTE walker indices into the same frame (`0..N-1`), not pool indices.
  Any value outside `[0, N)` is an out-of-range companion and masks the element.
- Masks are `0`/`1` integers. Complex arrays are split into `*_re` / `*_im`.
- `null` means "undefined" (masked frame, absent or invalid fit window). JSON has no NaN or inf.
- Inputs are rounded to 5 decimals and are exact as written (colours have norm `1 ± 1e-5`; do not
  assume exactly unit norm). float64 outputs carry the full shortest round-trip repr.
- `dtype: "f32"` values are the shortest decimal that round-trips the float32 Python produced.
  Python computes these series in float32 (audit D11i; Rust is f64 everywhere).
- Masked per-element values are written as `0.0` by Python; always read the mask.

## Schema (`schema_version = 1`)

Top level of every file:

| Field | Type / shape | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `name`, `description`, `generator`, `seed` | string / int | case identity; numpy `default_rng(seed)`, torch seed for the pairing |
| `shape` | `{frames, walkers, dimension}` | `T`, `N`, `d` |
| `involutive` | bool | both companion maps are in-range involutions |
| `parameters` | object | every scalar used, see below |
| `inputs` | object | the `PreparedChannelData` tensors |
| `pairs`, `triplets` | object | per-element values with validity masks (float64, tolerance 1e-12) |
| `series` | object | per-frame operator series |
| `electroweak` | object | U(1) / SU(2) phase series |
| `statistics` | object | origin statistics and correlators of a fixed synthetic series |
| `window_scan` | object | AIC window-scan table of a fixed synthetic correlator |
| `exchange_cancellation` | object | only when `involutive`: measured D1 magnitudes |
| `provenance_root` | string | `src/fragile/physics`; provenance paths are relative to it |
| `provenance` | `{key: "file:line function[; note]"}` | table referenced by every `provenance` list |

Every block carries a `provenance` entry: either a list of keys into the top-level table (records)
or `{output field: [keys]}` (blocks). Line numbers are resolved at export time from the Python
source, so they are those of the exporting checkout.

### `parameters`

`eps` (1e-12: validity threshold `|.| > eps` on `q`, `b` and the three links of `Pi`; Python
`PreparedChannelData.eps`), `norm_floor` (`max(eps, 1e-20)`, unit-displacement floor),
`pair_selection` (`"both"`), `h_eff` (1.0; also used for the SU(2) action scale, i.e.
`h_s = None`), `epsilon_d` (0.9, U(1) Gaussian range), `epsilon_clone` (0.01, used ONLY as the
SU(2) phase regulariser here), `lambda_alg` (velocity weight in `D^2`; 0 except in
`mutual_pairing_odd`), `flux_exp_alpha`, `su2_operator_mode` (`"standard"`).

### `inputs`

| Field | Shape | Notes |
|---|---|---|
| `color_re`, `color_im` | `[T][N][3]` | colour state `c_i`; exactly 0 where `color_valid = 0` |
| `color_valid` | `[T][N]` | |
| `companions_distance`, `companions_clone` | `[T][N]` int | absolute walker index; may be `-1`, `N`, `N+3` or `i` |
| `scores` | `[T][N]` | cloning scores (no ties) |
| `positions` | `[T][N][3]` | used as `positions` and `positions_full` |
| `velocities` | `[T][N][3]` or `null` | only `mutual_pairing_odd` (enters `u1_dressed*` through `lambda_alg`) |
| `fitness` | `[T][N]` | in `[0.5, 1.5]` |
| `alive`, `will_clone` | `[T][N]` | |

### `pairs` — `[T][N][P]`

`j = companions_distance[t][i]` for `P = 0`, `companions_clone[t][i]` for `P = 1`.

- `q_re`, `q_im`: `q_ij = c_i^dagger c_j = sum_a conj(c_i[a]) c_j[a]`.
- `valid`: `j in [0,N)` and `j != i` and `color_valid[i]` and `color_valid[j]` and `q` finite and
  `|q| > eps`.
- `valid_unit_displacement`: `valid` and `|x_j - x_i| > norm_floor`. The key is ABSENT when it
  equals `valid` (only `masked.json` has coincident positions).

### `triplets` — `[T][N]`

Triplet `(i, j, k)` with `j = companions_distance[t][i]`, `k = companions_clone[t][i]`.

- `b_re`, `b_im`: `b_ijk = det[c_i, c_j, c_k]` (columns), expanded as
  `a0(b1 c2 - b2 c1) - a1(b0 c2 - b2 c0) + a2(b0 c1 - b1 c0)` with `a = c_i, b = c_j, c = c_k`.
- `pi_re`, `pi_im`: `Pi_ijk = q_ij q_jk q_ki`.
- Structural validity: `j, k in [0,N)`, `j != i`, `k != i`, `j != k`; plus the three colours
  valid and the value finite. `b_valid` adds `|b| > eps`; `pi_valid` adds `|q_ij|, |q_jk|, |q_ki| > eps`.

### `series`

`counts`: `{pairs, pairs_unit_displacement, triplets_b, triplets_pi, triplets_b_and_pi}`, each
`[T]` int — the number of valid elements per frame (denominator of the masked mean).
`float32_roundoff`: largest `|f64 masked mean - f32 Python series|` the exporter measured.
`records[name] = {values, count, dtype, status, components?, provenance}`:

- `values`: `[T]`, or `[T][C]` when `components` is present (`["x","y","z"]`).
- `count`: key into `counts`. Every series is the valid-count masked mean
  `sum(valid * value) / count` (chapter 04 normalisation).
- A frame with `count = 0` is `0.0` in an `f32` record (Python's placeholder, audit D11b) and
  `null` in an `f64` record. Rust must treat it as weight 0; `masked.json` frame 5 is such a frame.
- `status`: `audited` (audit section A verdict "fine"; parity required), `non_book` (Python default
  that is not a book mode), `extra` (Python behaviour the audit neither confirmed nor rejected:
  parity optional, a mismatch needs investigation rather than a Rust change).

| Record | Per-element value | count | status |
|---|---|---|---|
| `scalar`, `pseudoscalar` | `Re q`, `Im q` | pairs | audited |
| `*_score_directed` (meson) | `q -> conj(q)` when `s_j - s_i < 0`, then Re / Im | pairs | extra |
| `*_score_weighted` (meson) | `q * abs(s_j - s_i)`, then Re / Im | pairs | extra |
| `vector`, `axial` | `Re q * r`, `Im q * r`, `r = x_j - x_i`, per component | pairs | audited |
| `vector_unit`, `axial_unit` | same with `r / max(abs(r), norm_floor)` | pairs_unit_displacement | audited |
| `vector_score_directed`, `axial_score_directed` | score-directed `q`, raw `r`, projection `full` | pairs | extra |
| `baryon_re_f64`, `baryon_im_f64`, `baryon_abs2_f64` | `Re b`, `Im b`, `abs(b)^2` (book modes; float64 masked mean of `triplets.b` computed by the exporter, Python has no such output) | triplets_b | audited |
| `baryon_det_abs` | `abs(b)` (Python default, not a book mode) | triplets_b | non_book |
| `baryon_score_signed` | `Re det` of the three colours ordered by ascending score | triplets_b | extra |
| `baryon_flux_action` | `abs(b) * (1 - cos arg Pi)` | triplets_b_and_pi | extra |
| `glueball_re_plaquette` | `Re Pi` | triplets_pi | audited |
| `glueball_action_re_plaquette` | `1 - Re Pi` | triplets_pi | audited |
| `glueball_phase_action` | `1 - cos arg Pi` | triplets_pi | audited |
| `glueball_phase_sin2` | `sin^2 arg Pi` | triplets_pi | audited |

Python casts the displacement `r` to float32 before the product, and the scores to float32 for the
baryon ordering. In `mutual_pairing_odd.json` the exchange-odd records are absent (see below).

### `electroweak`

`counts`: `{u1, su2, alive, u1_python_unmasked, su2_python_unmasked}`, each `[T]`.
`records[name]` as in `series`; complex records have `components: ["re","im"]` and `values [T][2]`.

- Source validity: `alive[i]` and companion in `[0,N)` and companion `!= i`. Colour validity plays
  no role. Python does NOT mask self companions (audit D11d); `self_companions_masked: true` means
  the exporter removed those sources through the `alive` mask it handed to the Python operator,
  which is the behaviour Rust pins. `*_python_unmasked` are the counts Python would have used; a
  self companion contributes the phasor `1 + 0i` with amplitude 1 to Python's unmasked mean.
- `u1_phase`: mean of `exp(i theta)`, `theta = -(F_j - F_i) / h_eff`, `j` = distance companion.
  `u1_phase_q2`: `exp(2 i theta)`.
- `u1_dressed`, `u1_dressed_q2`: the same times `exp(-D^2 / (4 epsilon_d^2))`,
  `D^2 = |x_j - x_i|^2 + lambda_alg |v_j - v_i|^2` (velocity term only when `lambda_alg > 0`).
- `su2_phase`: mean of `exp(i theta)`, `theta = (F_k - F_i) / ((abs(F_i) + epsilon_clone) h_eff)`,
  `k` = cloning companion. `su2_phase_directed`: `exp(i abs(theta))`.
- `fitness_phase` = mean over alive of `-F_i`; `clone_indicator` = mean over alive of
  `will_clone` (`status: extra`, count `alive`).
- Python evaluates the phasors in complex64: compare with an ABSOLUTE tolerance of 1e-6
  (`float32_roundoff` records the measured deviation, about 1e-7).
- In `mutual_pairing_odd.json` the U(1) records have `components: ["re"]` and `values [T][1]`: the
  imaginary part is exchange-odd and excluded.

### `statistics`

Inputs: `scalar_series [16]`, `vector_series [16][3]`, `max_lag = 6`. Blocks `connected` and `raw`:

- `scalar_sums [16][7]`, `scalar_counts [16][7]` (`connected` only): origin statistics.
  `sums[t][lag] = w[t] * w[t+lag]` for `t + lag < T`, else 0 with count 0; count is 1 otherwise.
- `scalar_mean [7]`: `sum_t sums / max(sum_t counts, 1)`, i.e. normalisation `1 / (T - lag)`.
- `scalar_fft [7]`: the FFT estimator on the same series (agrees with `scalar_mean` to 1e-12).
- `vector_contracted_mean`, `vector_contracted_fft [7]`: `C(lag) = sum_c C_cc(lag)`.
- `connected`: `w = x - mean_t(x)` with the FULL-series mean per component subtracted before any
  product (not a lag-dependent mean); `raw`: `w = x`.

Excluded: `resample_statistics` / jackknife errors (audit D6).

### `window_scan`

Inputs: `correlator [13]` (noiseless `a1 e^{-m1 t} + a2 e^{-m2 t}`, `model` gives rates and
amplitudes), `error [13]` (supplied diagonal error), `dt = 1`, `window_widths = [3, 5, 8]`,
`max_log_error = 0.5`, `min_mass = 0`.

- Log-space fit: `y = ln C`, `sigma_y = error / abs(C)`. `point_valid[t]` = `C > 0` finite and
  `sigma_y` finite, `> 0` and `<= max_log_error` (the last two lags fail it in every file).
- Tables `[width index][t0]`: window `W = window_widths[width index]` covers lags `t0 .. t0+W-1`;
  `null` where `t0 > 13 - W` or a point of the window is invalid.
- `window_mass` = minus the slope of the weighted least-squares line (weights `1 / sigma_y^2`);
  `window_mass_variance = S_w / (S_w S_tt - S_t^2)`; `window_chi2` = weighted residual sum;
  `window_aic = chi2 + 2*2 + 2*(n_valid_points - W)` with `n_valid_points = sum(point_valid) = 11`.
- Average over the `n_valid_windows` windows: weights `p ~ exp(-(AIC - AIC_min) / 2)`;
  `mass = sum p m`, `window_spread = sqrt(sum p (m - mass)^2)`,
  `statistical_error = sqrt(sum p variance)`, `mass_error = sqrt(stat^2 + spread^2)`;
  `best_window = {width, t_start, mass, mass_error, aic}` is the minimum-AIC window; its
  `mass_error` is `sqrt(window_mass_variance)` of that single window, not the averaged error.
- The Python `mass > 0` filter and `min_mass` (audit D11j) are inert on these inputs (asserted).
- Tolerance 1e-9 relative on mass, variance, AIC and the averages. `window_chi2` is evaluated by
  Python through the normal-equation identity and loses digits when small: measured against
  40-digit arithmetic it is exact to 2e-11 absolute but only 8e-8 relative; compare it with an
  absolute tolerance of 1e-9.

## Tolerances (audit C2)

| Quantity | Tolerance |
|---|---|
| per-pair `q`, per-triplet `b`, `Pi`, and their masks (exact) | 1e-12 |
| `f64` series | 1e-12 |
| `f32` series | `abs(a - e) <= 1e-6 * max(1, abs(e))` |
| statistics sums, means, correlators | 1e-10 |
| window-scan table and averages | 1e-9 (`window_chi2`: absolute) |

The `f32` bound is absolute for the O(1)-bounded series of these fixtures. A purely relative 1e-6
does not hold: float32 summation error is about 1e-7 absolute, which is up to 1.7e-5 relative on
the small frame means here. The files were re-derived twice from this README alone by independent
numpy scripts (no `fragile` or `torch` import): every mask and count exact, `q`, `b`, `Pi` within
4e-16, and every `series` record (including `baryon_score_signed`), every `electroweak` record,
the statistics blocks and the window-scan tables and averages within the tolerances above.

## Parity exclusions

Not exported, or exported only as a magnitude; Rust pins the corrected behaviour.

| Audit | Excluded | Where Python does it |
|---|---|---|
| D1 | frame means of exchange-odd operators on an involutive pairing: `pseudoscalar`, `pseudoscalar_score_weighted`, `vector`, `vector_unit`, `vector_score_directed`, `axial_score_directed`, `Im` of every U(1) channel — absent from `mutual_pairing_odd.json`, magnitudes in `exchange_cancellation` | `operators/meson_operators.py:222`, `operators/vector_operators.py:280` |
| D4 | SU(2) amplitudes: `su2_component`, `su2_doublet`, `su2_doublet_diff`, their `_directed` and walker-type variants, `ew_mixed` (`epsilon_clone` doubles as Gaussian range) | `operators/electroweak_operators.py:176-177, 317-319` |
| D8 | component-mean channels (vector / axial / tensor components averaged before correlating); the fixtures keep components and contract `C_kk` | `new_channels/dirac_spinors.py:473-483`, `operators/dirac_operators.py:157-163` |
| D3 | GEVP bootstrap | `new_channels/gevp_channels.py:722-735` |
| D6 | block size 10 and jackknife with the full-sample mean (`resample_statistics`, `sample_covariance`); only origin sums and central values are exported | `qft_utils/statistics.py`, `new_channels/correlator_channels.py:558` |
| D7 | `cloning_frames_only` | `electroweak/chirality.py:358-384` |
| D10 | flow `w0` | `new_channels/wilson_flow.py` |
| D11c | `gvar(0,0)` effective-mass placeholders | — |
| D2 | PDG-tuned priors | `app/mass_extraction_tab.py:298-306` |
| D11b | `0.0` for frames without valid elements: exported with `count = 0`, to be read as weight 0 | per-frame means |
| D11d | unmasked electroweak self companions: the fixtures mask them | `operators/electroweak_operators.py:99, 156` |
| D11e | `projection_mode` other than `full` | `operators/vector_operators.py:166` |
| D11g | momentum projections | `operators/glueball_operators.py:202-269` |
| D11j | `mass > 0` filter / `min_mass` in the AIC average: inert on the exported correlators | `new_channels/correlator_channels.py:278, 364` |
| — | walker-type split, `velocity_norm_*`, tensor, Dirac, twistor, chirality, multiscale channels: not part of this fixture set | — |

## Measured D1 cancellation (`mutual_pairing_odd.json`, N = 9)

On the Fisher–Yates mutual pairing `q_ji = conj(q_ij)` and `r_ji = -r_ij`, so both orientations of
every pair are averaged and exchange-odd frame means vanish identically. Measured maxima over
frames and components of Python's float32 series (`exchange_cancellation` holds the numbers):

| Cancelled (roundoff) | max | Surviving | max |
|---|---|---|---|
| `pseudoscalar` | 7.5e-9 | `scalar` | 0.202 |
| `pseudoscalar_score_weighted` | 1.1e-8 | `scalar_score_directed` | 0.202 |
| `vector` | 7.5e-9 | `scalar_score_weighted` | 0.310 |
| `vector_unit` | 3.7e-9 | `pseudoscalar_score_directed` | 0.206 |
| `vector_score_directed` | 7.5e-9 | `axial` | 0.308 |
| `axial_score_directed` | 7.5e-9 | `axial_unit` | 0.266 |
| `u1_phase_im` / `u1_phase_q2_im` | 7.5e-9 / 1.5e-8 | | |
| `u1_dressed_im` / `u1_dressed_q2_im` | 3.7e-9 / 3.3e-9 | | |

In float64 the frame sum of `Im q` over valid pairs is at most 3.3e-16 (identity test bound
`1e-14 * N`). This matches the audit (pseudoscalar 1.3e-8, vector 1.5e-8 against scalar 0.14,
axial 0.35 on a real run). Two observations beyond the audit text: score direction repairs only
the pseudoscalar — it conjugates `q` on downhill pairs, so `Re q * r` stays odd and `Im q * r`
BECOMES odd (`axial_score_directed` cancels although `axial` survives); score weighting keeps the
orientation of the pair, so `pseudoscalar_score_weighted` cancels too. The exporter asserts
cancelled `< 1e-6` and surviving `> 1e-3`.
