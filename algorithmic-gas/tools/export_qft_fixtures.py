"""Export reference values of the Python QFT operators for the Rust parity tests.

Builds ``PreparedChannelData`` directly from seeded synthetic float64 tensors
(no ``RunHistory``), runs the operator functions of ``fragile.physics`` on them
and writes one JSON document per case to
``crates/algorithmic-gas/tests/fixtures/qft``. Only the pieces the QFT audit
judged correct are exported; the parity exclusions and the file schema are in
the ``README.md`` of that directory. Every output names, through the
``provenance`` table of its file, the Python function and ``file:line`` that
produced it.

Inputs are rounded to five decimals so that their float64 repr stays short; they
are exact as written. float64 outputs carry the full repr. float32 operator
series are written as the shortest decimal that round-trips the float32 value.
Every float32 series is checked here against a float64 masked mean of the
exported per-pair and per-triplet values.

Usage: uv run python algorithmic-gas/tools/export_qft_fixtures.py
"""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from fragile.physics.fractal_gas.euclidean_gas import random_pairing_fisher_yates
from fragile.physics.new_channels.correlator_channels import (
    ConvolutionalAICExtractor,
    CorrelatorConfig,
    extract_mass_aic,
)
from fragile.physics.operators import (
    baryon_operators,
    electroweak_operators,
    glueball_operators,
    meson_operators,
    vector_operators,
)
from fragile.physics.operators.config import (
    BaryonOperatorConfig,
    ElectroweakOperatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    VectorOperatorConfig,
)
from fragile.physics.operators.correlators import compute_correlators_batched
from fragile.physics.operators.preparation import PreparedChannelData
from fragile.physics.qft_utils.companions import (
    build_companion_pair_indices,
    build_companion_triplets,
)
from fragile.physics.qft_utils.fft import _fft_correlator_batched
from fragile.physics.qft_utils.statistics import CorrelatorStatistics, series_statistics


REPO = Path(__file__).resolve().parents[2]
PROVENANCE_ROOT = "src/fragile/physics"
OUTPUT = Path(__file__).resolve().parents[1] / "crates/algorithmic-gas/tests/fixtures/qft"
SCHEMA_VERSION = 1
FRAMES = 12
DIMENSION = 3
DECIMALS = 5
MAX_BYTES = 60_000

EPS = 1e-12
NORM_FLOOR = max(EPS, 1e-20)
H_EFF = 1.0
EPSILON_D = 0.9
EPSILON_CLONE = 0.01
FLUX_EXP_ALPHA = 1.0
MAX_LAG = 6
STATISTICS_FRAMES = 16
WINDOW_WIDTHS = [3, 5, 8]
WINDOW_LAGS = 13
MAX_LOG_ERROR = 0.5

# Frame means that vanish identically when both companion maps are involutions (audit D1):
# q_ji = conj(q_ij) and r_ji = -r_ij. Score direction conjugates q on downhill pairs, which
# repairs Im q but makes the axial product odd; score weighting keeps the pair orientation.
EXCHANGE_ODD = {
    "pseudoscalar",
    "pseudoscalar_score_weighted",
    "vector",
    "vector_unit",
    "vector_score_directed",
    "axial_score_directed",
}
EXCHANGE_ODD_BOUND = 1e-6
EXCHANGE_EVEN_FLOOR = 1e-3

PROVENANCE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Provenance and JSON helpers
# ---------------------------------------------------------------------------


def ref(key: str, obj, needle: str | None = None, note: str = "") -> str:
    """Register ``file:line name`` of *obj* (or of its first line containing *needle*)."""
    lines, start = inspect.getsourcelines(obj)
    path = Path(inspect.getsourcefile(obj)).resolve().relative_to(REPO / PROVENANCE_ROOT)
    offset = 0
    if needle is not None:
        offset = next(k for k, line in enumerate(lines) if needle in line)
    entry = f"{path}:{start + offset} {obj.__qualname__}"
    PROVENANCE[key] = f"{entry}; {note}" if note else entry
    return key


def short32(value: float) -> float:
    """Shortest decimal that round-trips the float32 *value*, as a Python float."""
    return float(np.format_float_scientific(np.float32(value), unique=True))


def f32(tensor: torch.Tensor) -> list:
    assert tensor.dtype == torch.float32, tensor.dtype
    return np.vectorize(short32, otypes=[object])(tensor.numpy()).tolist()


def f64(array) -> list:
    """Nested list with ``None`` for non-finite entries (JSON has no NaN or inf)."""
    array = np.asarray(array, dtype=np.float64)
    out = array.astype(object)
    out[~np.isfinite(array)] = None
    return out.tolist()


def bits(mask) -> list:
    return np.asarray(mask).astype(np.int64).tolist()


def frame_mean(values: np.ndarray, valid: np.ndarray) -> list:
    """float64 masked frame mean; a frame without valid elements is ``None``, never 0."""
    frames = values.shape[0]
    mask = valid.reshape(frames, -1)
    flat = values.reshape(frames, mask.shape[1], -1)
    out: list = []
    for t in range(frames):
        count = int(mask[t].sum())
        if count == 0:
            out.append(None)
            continue
        mean = (flat[t] * mask[t][:, None]).sum(0) / count
        out.append(mean.tolist() if mean.size > 1 else float(mean[0]))
    return out


def check_f32(name: str, exact: list, single: list) -> float:
    """The float64 mean of the exported elements must reproduce Python's float32 series."""
    worst = 0.0
    for e, s in zip(exact, single):
        if e is None:
            assert not np.any(np.asarray(s)), f"{name}: zero-count frame is not 0.0"
            continue
        e, s = np.atleast_1d(e).astype(float), np.atleast_1d(s).astype(float)
        worst = max(worst, float(np.abs(e - s).max()))
        assert np.all(np.abs(e - s) <= 1e-6 * np.maximum(1.0, np.abs(e))), (name, e, s)
    return worst


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def rounded(array: np.ndarray) -> np.ndarray:
    return np.round(array, DECIMALS) + 0.0


def base_inputs(rng: np.random.Generator, n: int) -> dict:
    shape = (FRAMES, n, DIMENSION)
    color = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    color /= np.linalg.norm(color, axis=-1, keepdims=True)
    return {
        "color": rounded(color.real) + 1j * rounded(color.imag),
        "color_valid": np.ones((FRAMES, n), dtype=bool),
        "scores": rounded(rng.normal(size=(FRAMES, n))),
        "positions": rounded(rng.uniform(-1.0, 1.0, size=shape)),
        "velocities": None,
        "fitness": rounded(rng.uniform(0.5, 1.5, size=(FRAMES, n))),
        "alive": np.ones((FRAMES, n), dtype=bool),
        "will_clone": rng.uniform(size=(FRAMES, n)) < 0.3,
    }


def random_companions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform companion c(i) != i, independently per walker: generically not an involution."""
    draw = rng.integers(0, n - 1, size=(FRAMES, n))
    return draw + (draw >= np.arange(n)[None, :])


def is_involution(companions: np.ndarray) -> bool:
    n = companions.shape[1]
    if not ((companions >= 0) & (companions < n)).all():
        return False
    return bool((np.take_along_axis(companions, companions, axis=1) == np.arange(n)).all())


def non_involutive_inputs(rng: np.random.Generator) -> dict:
    inputs = base_inputs(rng, 10)
    for role in ("companions_distance", "companions_clone"):
        inputs[role] = random_companions(rng, 10)
        assert not is_involution(inputs[role])
    return inputs


def mutual_pairing_inputs(rng: np.random.Generator, seed: int) -> dict:
    n = 9
    inputs = base_inputs(rng, n)
    inputs["velocities"] = rounded(rng.normal(scale=0.5, size=(FRAMES, n, DIMENSION)))
    torch.manual_seed(seed)
    for role in ("companions_distance", "companions_clone"):
        maps = torch.stack([random_pairing_fisher_yates(n) for _ in range(FRAMES)]).numpy()
        assert is_involution(maps)
        assert ((maps == np.arange(n)).sum(1) == 1).all()
        inputs[role] = maps
    return inputs


def masked_inputs(rng: np.random.Generator) -> dict:
    n = 10
    inputs = base_inputs(rng, n)
    alive, valid = inputs["alive"], inputs["color_valid"]
    for t in range(FRAMES):
        if t % 3 != 0:
            alive[t, rng.choice(n, size=1 + t % 2, replace=False)] = False
    # A dead walker has no force; some alive walkers have zero force as well.
    valid &= alive
    valid &= rng.uniform(size=valid.shape) >= 0.15
    valid[5] = False  # a frame whose viscous force vanishes identically
    inputs["color"][~valid] = 0.0

    index = np.broadcast_to(np.arange(n), (FRAMES, n))
    for role in ("companions_distance", "companions_clone"):
        companions = random_companions(rng, n)
        for t in range(FRAMES):
            living = np.flatnonzero(alive[t])
            for i in living:
                # The algorithm never pairs an alive walker with a dead one.
                if not alive[t, companions[t, i]]:
                    companions[t, i] = rng.choice(living[living != i])
        edits = rng.uniform(size=companions.shape)
        companions = np.where(edits < 0.06, -1, companions)
        companions = np.where((edits >= 0.06) & (edits < 0.10), n, companions)
        companions = np.where((edits >= 0.10) & (edits < 0.13), n + 3, companions)
        companions = np.where((edits >= 0.13) & (edits < 0.19), index, companions)
        inputs[role] = companions

    # Coincident positions: the unit displacement is undefined on these pairs.
    companions = inputs["companions_distance"]
    inside = (companions >= 0) & (companions < n) & (companions != index) & valid
    frames, walkers = np.nonzero(inside)
    for k in rng.choice(len(frames), size=4, replace=False):
        t, i = frames[k], walkers[k]
        inputs["positions"][t, i] = inputs["positions"][t, companions[t, i]]

    for role in ("companions_distance", "companions_clone"):
        companions = inputs[role]
        linked = alive & (companions >= 0) & (companions < n)
        target = np.take_along_axis(alive, np.clip(companions, 0, n - 1), axis=1)
        assert target[linked].all(), "an alive walker is paired with a dead one"
        target = np.take_along_axis(valid, np.clip(companions, 0, n - 1), axis=1)
        assert (valid & linked & ~target).any(), "no valid colour paired with an invalid one"
        assert (companions[alive] == index[alive]).any()
        assert (companions[alive] < 0).any()
        assert (companions[alive] >= n).any()
    return inputs


def prepared(inputs: dict) -> PreparedChannelData:
    def real(key: str) -> torch.Tensor | None:
        return None if inputs[key] is None else torch.tensor(inputs[key], dtype=torch.float64)

    return PreparedChannelData(
        color=torch.tensor(inputs["color"], dtype=torch.complex128),
        color_valid=torch.tensor(inputs["color_valid"]),
        companions_distance=torch.tensor(inputs["companions_distance"], dtype=torch.long),
        companions_clone=torch.tensor(inputs["companions_clone"], dtype=torch.long),
        scores=real("scores"),
        positions=real("positions"),
        positions_axis=None,
        projection_length=None,
        frame_indices=list(range(FRAMES)),
        device=torch.device("cpu"),
        eps=EPS,
        fitness=real("fitness"),
        alive=torch.tensor(inputs["alive"]),
        velocities=real("velocities"),
        positions_full=real("positions"),
        will_clone=torch.tensor(inputs["will_clone"]),
    )


# ---------------------------------------------------------------------------
# Operator outputs
# ---------------------------------------------------------------------------


def pair_block(data: PreparedChannelData) -> tuple[dict, dict]:
    pair_indices, structural = build_companion_pair_indices(
        data.companions_distance, data.companions_clone, "both"
    )
    inner = meson_operators._compute_inner_products_for_pairs
    observables = vector_operators._compute_pair_observables
    q, valid = inner(data.color, data.color_valid, pair_indices, structural, data.eps)
    assert q.dtype == torch.complex128
    arrays = {"q": q.numpy(), "valid": valid.numpy()}
    for unit in (False, True):
        q_vector, _, valid_vector = observables(
            data.color,
            data.color_valid,
            data.positions,
            pair_indices,
            structural,
            data.eps,
            unit,
            operator_mode="standard",
            projection_mode="full",
        )
        assert torch.equal(q_vector, torch.where(valid_vector, q, torch.zeros_like(q)))
        arrays["valid_unit" if unit else "valid_raw"] = valid_vector.numpy()
    assert np.array_equal(arrays["valid_raw"], arrays["valid"])
    block = {
        "layout": "[T][N][P]; P=0 distance companion, P=1 cloning companion",
        "q_re": f64(q.real),
        "q_im": f64(q.imag),
        "valid": bits(valid),
        "valid_unit_displacement": bits(arrays["valid_unit"]),
        "provenance": {
            "q": [ref("pair_q", inner, "inner = (")],
            "valid": [
                ref("pair_valid", inner, "valid = structural"),
                ref("pair_structural", build_companion_pair_indices, "valid_j = "),
            ],
            "valid_unit_displacement": [
                ref("pair_valid_unit", observables, "valid = valid & (disp_norm")
            ],
        },
    }
    if np.array_equal(arrays["valid_unit"], arrays["valid"]):
        del block["valid_unit_displacement"]
        del block["provenance"]["valid_unit_displacement"]
        del PROVENANCE["pair_valid_unit"]
    return block, arrays


def triplet_block(data: PreparedChannelData) -> tuple[dict, dict]:
    companions = (data.companions_distance, data.companions_clone)
    determinant = baryon_operators._compute_determinants_for_indices
    plaquette = glueball_operators._compute_color_plaquette_for_triplets
    b, b_valid = determinant(data.color, data.color_valid, *companions, data.eps)
    pi, pi_valid = plaquette(data.color, data.color_valid, *companions, data.eps)
    pi_baryon, pi_baryon_valid = baryon_operators._compute_triplet_plaquette_for_indices(
        color=data.color,
        color_valid=data.color_valid,
        companion_j=companions[0],
        companion_k=companions[1],
        eps=data.eps,
    )
    assert torch.equal(pi, pi_baryon)
    assert torch.equal(pi_valid, pi_baryon_valid)
    assert b.dtype == torch.complex128
    assert pi.dtype == torch.complex128
    arrays = {
        "b": b.numpy(),
        "b_valid": b_valid.numpy(),
        "pi": pi.numpy(),
        "pi_valid": pi_valid.numpy(),
    }
    structural = ref("triplet_structural", build_companion_triplets, "structural_valid = ")
    block = {
        "layout": "[T][N]; triplet (i, j=companions_distance[t][i], k=companions_clone[t][i])",
        "b_re": f64(b.real),
        "b_im": f64(b.imag),
        "b_valid": bits(b_valid),
        "pi_re": f64(pi.real),
        "pi_im": f64(pi.imag),
        "pi_valid": bits(pi_valid),
        "provenance": {
            "b": [ref("triplet_b", determinant, "det = _det3")],
            "b_valid": [ref("triplet_b_valid", determinant, "valid = struct"), structural],
            "pi": [ref("triplet_pi", plaquette, "pi = z_ij")],
            "pi_valid": [ref("triplet_pi_valid", plaquette, "valid = struct"), structural],
        },
    }
    return block, arrays


def series_block(
    data: PreparedChannelData, inputs: dict, pairs: dict, triplets: dict, involutive: bool
) -> tuple[dict, dict]:
    """Frame series. Returns the block and, separately, the exchange-odd series."""
    records: dict[str, dict] = {}
    odd: dict[str, list] = {}
    worst = 0.0

    def add(name, values, count, dtype, status, provenance, components=None):
        if involutive and name in EXCHANGE_ODD:
            odd[name] = values
            return
        record = {"values": values, "count": count, "dtype": dtype, "status": status}
        if components:
            record["components"] = components
        record["provenance"] = provenance
        records[name] = record

    n = inputs["positions"].shape[1]
    q, valid = pairs["q"], pairs["valid"]
    frame = np.arange(FRAMES)[:, None]
    gathered = np.stack(
        [
            inputs["positions"][frame, np.clip(inputs[role], 0, n - 1)]
            for role in ("companions_distance", "companions_clone")
        ],
        axis=2,
    )
    displacement = gathered - inputs["positions"][:, :, None, :]
    norm = np.linalg.norm(displacement, axis=-1, keepdims=True)
    unit = displacement / np.maximum(norm, NORM_FLOOR)

    # Mesons.
    meson = meson_operators.compute_meson_operators
    pair_mean = ref("pair_mean", meson_operators._per_frame_series)
    for mode in ("standard", "score_directed", "score_weighted"):
        out = meson(data, MesonOperatorConfig(operator_mode=mode, pair_selection="both", eps=EPS))
        suffix = "" if mode == "standard" else f"_{mode}"
        provenance = {
            "scalar": [ref("scalar", meson, "scalar_obs = inner.real"), pair_mean],
            "pseudoscalar": [
                ref("pseudoscalar", meson, "pseudoscalar_obs = inner.imag"),
                pair_mean,
            ],
        }
        extra: list[str] = []
        if mode == "score_directed":
            orient = meson_operators._orient_inner_products_by_scores
            extra = [ref("meson_score_directed", orient, "inner_oriented = torch.where(ds")]
        elif mode == "score_weighted":
            weight = meson_operators._weight_inner_products_by_score_gap
            extra = [ref("meson_score_weighted", weight, "inner_weighted = inner * gap")]
        for channel in ("scalar", "pseudoscalar"):
            add(
                f"{channel}{suffix}",
                f32(out[channel]),
                "pairs",
                "f32",
                "audited" if mode == "standard" else "extra",
                provenance[channel] + extra,
            )
            if mode == "standard":
                part = q.real if channel == "scalar" else q.imag
                exact = frame_mean(part, valid)
                worst = max(worst, check_f32(channel, exact, f32(out[channel])))

    # Vector and axial vector, per component.
    axes = ["x", "y", "z"]
    vector = vector_operators.compute_vector_operators
    observables = vector_operators._compute_pair_observables
    vector_mean = ref("vector_mean", vector_operators._per_frame_vector_series)
    variants = ((False, "standard"), (True, "standard"), (False, "score_directed"))
    for unit_displacement, mode in variants:
        out = vector(
            data,
            VectorOperatorConfig(
                operator_mode=mode,
                projection_mode="full",
                use_unit_displacement=unit_displacement,
                pair_selection="both",
                eps=EPS,
            ),
        )
        suffix = "_unit" if unit_displacement else ""
        suffix += "" if mode == "standard" else f"_{mode}"
        mask = pairs["valid_unit"] if unit_displacement else valid
        arm = unit if unit_displacement else displacement
        for channel, key, needle, part in (
            ("vector", "vector", "vector_obs = inner.real", q.real),
            ("axial", "axial_vector", "axial_obs = inner.imag", q.imag),
        ):
            provenance = [
                ref(channel, vector, needle),
                ref("displacement", observables, "displacement = (pos_j - pos_i)"),
                vector_mean,
            ]
            if unit_displacement:
                provenance.append(
                    ref("unit_displacement", observables, "displacement = displacement / disp")
                )
            if mode == "score_directed":
                provenance.append(
                    ref("vector_score_directed", observables, "inner = torch.where(ds >= 0")
                )
            add(
                f"{channel}{suffix}",
                f32(out[key]),
                "pairs_unit_displacement" if unit_displacement else "pairs",
                "f32",
                "audited" if mode == "standard" else "extra",
                provenance,
                components=axes,
            )
            if mode == "standard":
                exact = frame_mean(part[..., None] * arm, mask)
                worst = max(worst, check_f32(channel + suffix, exact, f32(out[key])))

    # Baryons: book modes from the per-triplet determinant, Python modes from the operator.
    b, b_valid = triplets["b"], triplets["b_valid"]
    baryon = baryon_operators.compute_baryon_operators
    triplet_mean = ref(
        "triplet_mean",
        baryon,
        "operator_baryon_series[valid_t] = sums",
        "float32 there; the baryon_*_f64 records apply the same masked mean to triplets.b in "
        "float64 (frame_mean of algorithmic-gas/tools/export_qft_fixtures.py), null on a "
        "zero-count frame",
    )
    for name, values in (
        ("baryon_re_f64", b.real),
        ("baryon_im_f64", b.imag),
        ("baryon_abs2_f64", np.abs(b) ** 2),
    ):
        exact = frame_mean(values, b_valid)
        add(name, exact, "triplets_b", "f64", "audited", ["triplet_b", triplet_mean])
    score_ordered = baryon_operators._compute_score_ordered_determinants_for_indices
    flux_weight = baryon_operators._baryon_flux_weight_from_plaquette
    modes = {
        "det_abs": [ref("baryon_det_abs", baryon, "source_obs = source_det.abs()")],
        "score_signed": [ref("baryon_score_ordered", score_ordered, "order = torch.argsort")],
        "flux_action": [
            "baryon_det_abs",
            ref("baryon_flux_action", flux_weight, "return (1.0 - torch.cos(phase))"),
        ],
    }
    for mode, provenance in modes.items():
        config = BaryonOperatorConfig(operator_mode=mode, flux_exp_alpha=FLUX_EXP_ALPHA, eps=EPS)
        add(
            f"baryon_{mode}",
            f32(baryon(data, config)["nucleon"]),
            "triplets_b_and_pi" if mode == "flux_action" else "triplets_b",
            "f32",
            "non_book" if mode == "det_abs" else "extra",
            [*provenance, triplet_mean],
        )
    exact = frame_mean(np.abs(b), b_valid)
    worst = max(worst, check_f32("baryon_det_abs", exact, records["baryon_det_abs"]["values"]))

    # Glueballs.
    pi, pi_valid = triplets["pi"], triplets["pi_valid"]
    phase = np.angle(pi)
    glueball = glueball_operators.compute_glueball_operators
    observable = glueball_operators._glueball_observable_from_plaquette
    glueball_mean = ref("glueball_mean", glueball, "operator_series[valid_t] =")
    for mode, values, needle in (
        ("re_plaquette", pi.real, "return pi.real"),
        ("action_re_plaquette", 1.0 - pi.real, "return (1.0 - pi.real)"),
        ("phase_action", 1.0 - np.cos(phase), "return (1.0 - torch.cos(phase))"),
        ("phase_sin2", np.sin(phase) ** 2, "return torch.sin(phase).square()"),
    ):
        out = glueball(data, GlueballOperatorConfig(operator_mode=mode, eps=EPS))
        single = f32(out["glueball"])
        add(
            f"glueball_{mode}",
            single,
            "triplets_pi",
            "f32",
            "audited",
            [ref(f"glueball_{mode}", observable, needle), glueball_mean],
        )
        exact = frame_mean(values, pi_valid)
        worst = max(worst, check_f32(f"glueball_{mode}", exact, single))

    counts = {
        "pairs": valid.sum((1, 2)).tolist(),
        "pairs_unit_displacement": pairs["valid_unit"].sum((1, 2)).tolist(),
        "triplets_b": b_valid.sum(1).tolist(),
        "triplets_pi": pi_valid.sum(1).tolist(),
        "triplets_b_and_pi": (b_valid & pi_valid).sum(1).tolist(),
    }
    return {"counts": counts, "float32_roundoff": worst, "records": records}, odd


def electroweak_block(
    data: PreparedChannelData, inputs: dict, involutive: bool, lambda_alg: float
) -> tuple[dict, dict]:
    """U(1) and SU(2) phase series; SU(2) amplitudes and the mixed channel are excluded.

    Python does not mask a walker that is its own companion (audit D11d). The exported series
    remove those sources through the ``alive`` mask handed to the Python operator, which is the
    behaviour Rust pins; the unmasked Python counts are exported next to the masked ones.
    """
    config = ElectroweakOperatorConfig(
        h_eff=H_EFF,
        epsilon_d=EPSILON_D,
        epsilon_clone=EPSILON_CLONE,
        lambda_alg=lambda_alg,
        su2_operator_mode="standard",
        enable_walker_type_split=False,
        enable_directed_variants=True,
        enable_parity_velocity=False,
    )
    operators = electroweak_operators.compute_electroweak_operators
    n = inputs["fitness"].shape[1]
    alive, fitness = inputs["alive"], inputs["fitness"]
    roles = {"u1": inputs["companions_distance"], "su2": inputs["companions_clone"]}
    proper = {key: c != np.arange(n)[None, :] for key, c in roles.items()}
    unmasked = {key: alive & (c >= 0) & (c < n) for key, c in roles.items()}
    source = {key: unmasked[key] & proper[key] for key in roles}
    outputs = {
        key: operators(replace(data, alive=torch.tensor(alive & proper[key])), config)
        for key in roles
    }

    u1 = electroweak_operators._compute_u1_operators
    su2 = electroweak_operators._compute_su2_operators
    complex_mean = ref("complex_mean", electroweak_operators._average_complex)
    u1_theta = ref("u1_theta", u1, "theta = ")
    su2_theta = ref("su2_theta", su2, "theta = ")
    u1_amplitude = ref("u1_amplitude", u1, "amp = torch.exp")
    directed = ref("su2_directed", su2, "phase_directed = torch.where")
    channels = (
        ("u1_phase", "u1", [ref("u1_phase", u1, '"u1_phase":'), u1_theta]),
        ("u1_phase_q2", "u1", [ref("u1_phase_q2", u1, '"u1_phase_q2":'), u1_theta]),
        ("u1_dressed", "u1", [ref("u1_dressed", u1, '"u1_dressed":'), u1_theta, u1_amplitude]),
        (
            "u1_dressed_q2",
            "u1",
            [ref("u1_dressed_q2", u1, '"u1_dressed_q2":'), u1_theta, u1_amplitude],
        ),
        ("su2_phase", "su2", [ref("su2_phase", su2, "su2_phase_std = "), su2_theta]),
        (
            "su2_phase_directed",
            "su2",
            [ref("su2_phase_directed", su2, "su2_phase_dir = "), directed, su2_theta],
        ),
    )
    gather = {
        key: np.take_along_axis(fitness, np.clip(c, 0, n - 1), axis=1) for key, c in roles.items()
    }
    theta = {
        "u1_phase": -(gather["u1"] - fitness) / max(H_EFF, 1e-12),
        "su2_phase": (gather["su2"] - fitness)
        / (np.abs(fitness) + max(EPSILON_CLONE, 1e-12))
        / max(H_EFF, 1e-12),
    }
    theta["u1_phase_q2"] = 2.0 * theta["u1_phase"]
    theta["su2_phase_directed"] = np.abs(theta["su2_phase"])

    records: dict[str, dict] = {}
    odd: dict[str, list] = {}
    worst = 0.0
    for name, role, provenance in channels:
        values = np.asarray(f32(outputs[role][name]), dtype=object)
        if name in theta:
            phasor = np.exp(1j * theta[name])
            exact = frame_mean(np.stack([phasor.real, phasor.imag], -1), source[role])
            worst = max(worst, check_f32(name, exact, values.astype(float).tolist()))
        components = ["re", "im"]
        if involutive and role == "u1":
            # The exchange-odd imaginary part cancels on a mutual pairing (audit D1).
            odd[f"{name}_im"] = values[:, 1].tolist()
            components, values = ["re"], values[:, :1]
        records[name] = {
            "values": values.tolist(),
            "count": role,
            "dtype": "f32",
            "status": "audited",
            "components": components,
            "provenance": [*provenance, complex_mean],
        }
    symmetry = electroweak_operators._compute_symmetry_breaking
    scalar_mean = ref("scalar_mean", electroweak_operators._average_scalar)
    sites = operators(data, config)
    for name, needle in (
        ("fitness_phase", "fp = -fitness"),
        ("clone_indicator", "ci = will_clone"),
    ):
        records[name] = {
            "values": f32(sites[name]),
            "count": "alive",
            "dtype": "f32",
            "status": "extra",
            "provenance": [ref(name, symmetry, needle), scalar_mean],
        }
    block = {
        "counts": {
            "u1": source["u1"].sum(1).tolist(),
            "su2": source["su2"].sum(1).tolist(),
            "alive": alive.sum(1).tolist(),
            "u1_python_unmasked": unmasked["u1"].sum(1).tolist(),
            "su2_python_unmasked": unmasked["su2"].sum(1).tolist(),
        },
        "self_companions_masked": True,
        "float32_roundoff": worst,
        "records": records,
    }
    return block, odd


# ---------------------------------------------------------------------------
# Estimator outputs
# ---------------------------------------------------------------------------


def statistics_block(rng: np.random.Generator) -> dict:
    """Origin statistics and correlators of a fixed AR(1)-like series."""
    noise = rng.normal(size=(STATISTICS_FRAMES, 4))
    series = np.zeros_like(noise)
    series[0] = noise[0]
    for t in range(1, STATISTICS_FRAMES):
        series[t] = 0.7 * series[t - 1] + noise[t]
    series = rounded(series + np.array([0.4, -0.2, 0.1, 0.3]))
    scalar = torch.tensor(series[:, 0], dtype=torch.float64)
    vector = torch.tensor(series[:, 1:], dtype=torch.float64)

    block: dict = {
        "max_lag": MAX_LAG,
        "scalar_series": series[:, 0].tolist(),
        "vector_series": series[:, 1:].tolist(),
        "layout": "sums, counts: [origin t][lag]; correlators: [lag]",
    }
    for connected in (True, False):
        stats = series_statistics(scalar, MAX_LAG, connected)
        contracted = series_statistics(vector, MAX_LAG, connected)
        fft = compute_correlators_batched(
            {"scalar": scalar, "vector": vector}, MAX_LAG, use_connected=connected
        )
        for name, direct in (("scalar", stats), ("vector", contracted)):
            assert fft[name].dtype == torch.float64
            assert torch.allclose(fft[name], direct.mean(), rtol=0.0, atol=1e-12)
        values = {
            "scalar_mean": f64(stats.mean()),
            "scalar_fft": f64(fft["scalar"]),
            "vector_contracted_mean": f64(contracted.mean()),
            "vector_contracted_fft": f64(fft["vector"]),
        }
        if connected:
            values["scalar_sums"] = f64(stats.sums)
            values["scalar_counts"] = stats.counts.to(torch.int64).tolist()
        block["connected" if connected else "raw"] = values
    block["provenance"] = {
        "scalar_sums": [ref("origin_sums", series_statistics, "sums[: T - lag, lag]")],
        "scalar_counts": [ref("origin_counts", series_statistics, "counts[: T - lag, lag]")],
        "connected": [ref("connected", series_statistics, "work = work - work.mean")],
        "scalar_mean": [ref("origin_mean", CorrelatorStatistics.mean)],
        "scalar_fft": [ref("fft", _fft_correlator_batched, "result = corr")],
        "vector_contracted_mean": ["origin_sums", "origin_mean"],
        "vector_contracted_fft": [
            "fft",
            ref("fft_contraction", compute_correlators_batched, "contracted = corr"),
        ],
    }
    return block


def window_scan_block(rates: tuple[float, float], amplitudes: tuple[float, float]) -> dict:
    """AIC window scan of a noiseless two-exponential correlator with supplied errors."""
    t = np.arange(WINDOW_LAGS, dtype=np.float64)
    values = amplitudes[0] * np.exp(-rates[0] * t) + amplitudes[1] * np.exp(-rates[1] * t)
    # Relative error 0.004 at t = 0 growing to 0.6 at the last but one lag: the tail is cut.
    growth = np.log(0.6 / 0.004) / (WINDOW_LAGS - 2)
    error = rounded(values * 0.004 * np.exp(growth * t))
    correlator = torch.tensor(values, dtype=torch.float64)
    config = CorrelatorConfig(window_widths=list(WINDOW_WIDTHS), min_mass=0.0)
    result = extract_mass_aic(
        correlator, 1.0, config, correlator_err=torch.tensor(error, dtype=torch.float64)
    )
    assert result["uncertainty_method"] == "supplied_diagonal_errors"
    assert result["window_widths"] == WINDOW_WIDTHS

    log_corr = torch.tensor(np.log(values)).reshape(1, 1, -1)
    log_err = torch.tensor(error / np.abs(values)).reshape(1, 1, -1)
    extractor = ConvolutionalAICExtractor(
        window_widths=list(WINDOW_WIDTHS), max_log_error=MAX_LOG_ERROR
    )
    chi2 = np.full((len(WINDOW_WIDTHS), WINDOW_LAGS), np.nan)
    for row, width in enumerate(WINDOW_WIDTHS):
        out = extractor._fit_single_width_full(log_corr, log_err, width)
        span = out["chi2"].shape[-1]
        ok = out["valid"].reshape(-1).numpy()
        chi2[row, :span] = np.where(ok, out["chi2"].reshape(-1).numpy(), np.nan)
        python = result["window_masses"][row, :span].numpy()
        assert np.array_equal(out["mass"].reshape(-1).numpy()[ok], python[ok])
        assert np.isnan(python[~ok]).all()
    masses = result["window_masses"].numpy()
    point_valid = log_err.reshape(-1).numpy() <= MAX_LOG_ERROR
    assert (~point_valid).sum() == 2, point_valid
    # The positive-mass filter of the Python average (audit D11j) must be inert here.
    assert np.all(masses[np.isfinite(masses)] > 0)
    assert result["n_valid_windows"] == int(np.isfinite(masses).sum())
    best = result["best_window"]
    single = ConvolutionalAICExtractor._fit_single_width_full
    average = ConvolutionalAICExtractor.fit_all_widths
    return {
        "model": {"rates": list(rates), "amplitudes": list(amplitudes)},
        "dt": 1.0,
        "window_widths": WINDOW_WIDTHS,
        "max_log_error": MAX_LOG_ERROR,
        "min_mass": 0.0,
        "correlator": values.tolist(),
        "error": error.tolist(),
        "point_valid": bits(point_valid),
        "layout": "[width index][window start t0]; null = window absent or invalid",
        "window_mass": f64(masses),
        "window_aic": f64(result["window_aic"]),
        "window_chi2": f64(chi2),
        "window_mass_variance": f64(result["window_mass_variance"]),
        "n_valid_windows": result["n_valid_windows"],
        "mass": result["mass"],
        "statistical_error": result["statistical_error"],
        "window_spread": result["window_spread"],
        "mass_error": result["mass_error"],
        "best_window": {
            key: best[key] for key in ("width", "t_start", "mass", "mass_error", "aic")
        },
        "provenance": {
            "log_inputs": [ref("scan_log_inputs", extract_mass_aic, "log_err[mask] = ")],
            "point_valid": [ref("scan_point_valid", single, "point_valid &= err")],
            "window_mass": [ref("scan_mass", single, "mass = -slope")],
            "window_chi2": [ref("scan_chi2", single, "chi2 = (")],
            "window_aic": [ref("scan_aic", single, "aic = chi2 + 4.0")],
            "window_mass_variance": [ref("scan_variance", single, "slope_var = S_w")],
            "mass": [
                ref("scan_weights", average, "weights = torch.exp"),
                ref("scan_average", average, "mass_final = "),
            ],
            "window_spread": [ref("scan_spread", average, "window_spread = ")],
            "statistical_error": [ref("scan_statistical", average, "statistical_error = ")],
            "mass_error": [ref("scan_error", average, "mass_error = float")],
            "best_window": [ref("scan_best", average, "best_flat_idx = int")],
        },
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def exchange_cancellation(
    pairs: dict, series: dict, odd_series: dict, odd_electroweak: dict
) -> dict:
    """Measured size of the exchange-odd frame means on the mutual pairing (audit D1)."""

    def peak(values) -> float:
        rows = [row if isinstance(row, list) else [row] for row in values]
        return max(abs(v) for row in rows for v in row)

    cancelled = {name: peak(values) for name, values in (odd_series | odd_electroweak).items()}
    surviving = {
        name: peak(record["values"])
        for name, record in series["records"].items()
        if name.split("_")[0] in {"scalar", "pseudoscalar", "vector", "axial"}
    }
    assert set(odd_series) == EXCHANGE_ODD
    assert max(cancelled.values()) < EXCHANGE_ODD_BOUND, cancelled
    assert min(surviving.values()) > EXCHANGE_EVEN_FLOOR, surviving
    q, valid = pairs["q"], pairs["valid"]
    return {
        "note": "exchange-odd frame means are excluded from parity; max |series| over frames "
        "and components of the float32 Python output",
        "max_abs_frame_sum_im_q_f64": float(np.abs((q.imag * valid).sum((1, 2))).max()),
        "cancelled": cancelled,
        "surviving": surviving,
        "provenance": [
            ref("pairing", random_pairing_fisher_yates),
            ref("pseudoscalar", meson_operators.compute_meson_operators, "pseudoscalar_obs = "),
            ref("vector", vector_operators.compute_vector_operators, "vector_obs = inner.real"),
            "vector_score_directed",
            "meson_score_weighted",
            "complex_mean",
        ],
    }


def build_case(name: str, description: str, seed: int, window: tuple) -> dict:
    PROVENANCE.clear()
    rng = np.random.default_rng(seed)
    lambda_alg = 0.0
    if name == "non_involutive":
        inputs = non_involutive_inputs(rng)
    elif name == "mutual_pairing_odd":
        inputs = mutual_pairing_inputs(rng, seed)
        lambda_alg = 0.5
    else:
        inputs = masked_inputs(rng)
    involutive = is_involution(inputs["companions_distance"]) and is_involution(
        inputs["companions_clone"]
    )
    data = prepared(inputs)
    pairs, pair_arrays = pair_block(data)
    triplets, triplet_arrays = triplet_block(data)
    series, odd_series = series_block(data, inputs, pair_arrays, triplet_arrays, involutive)
    electroweak, odd_electroweak = electroweak_block(data, inputs, involutive, lambda_alg)
    velocities = inputs["velocities"]

    case = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "generator": "algorithmic-gas/tools/export_qft_fixtures.py",
        "seed": seed,
        "shape": {"frames": FRAMES, "walkers": inputs["alive"].shape[1], "dimension": DIMENSION},
        "involutive": involutive,
        "parameters": {
            "eps": EPS,
            "norm_floor": NORM_FLOOR,
            "pair_selection": "both",
            "h_eff": H_EFF,
            "epsilon_d": EPSILON_D,
            "epsilon_clone": EPSILON_CLONE,
            "lambda_alg": lambda_alg,
            "flux_exp_alpha": FLUX_EXP_ALPHA,
            "su2_operator_mode": "standard",
        },
        "inputs": {
            "color_re": inputs["color"].real.tolist(),
            "color_im": inputs["color"].imag.tolist(),
            "color_valid": bits(inputs["color_valid"]),
            "companions_distance": inputs["companions_distance"].tolist(),
            "companions_clone": inputs["companions_clone"].tolist(),
            "scores": inputs["scores"].tolist(),
            "positions": inputs["positions"].tolist(),
            "velocities": None if velocities is None else velocities.tolist(),
            "fitness": inputs["fitness"].tolist(),
            "alive": bits(inputs["alive"]),
            "will_clone": bits(inputs["will_clone"]),
        },
        "pairs": pairs,
        "triplets": triplets,
        "series": series,
        "electroweak": electroweak,
        "statistics": statistics_block(rng),
        "window_scan": window_scan_block(*window),
    }
    if involutive:
        case["exchange_cancellation"] = exchange_cancellation(
            pair_arrays, series, odd_series, odd_electroweak
        )
    else:
        assert not odd_series
        assert not odd_electroweak
    case["provenance_root"] = PROVENANCE_ROOT
    case["provenance"] = dict(PROVENANCE)
    return case


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        build_case(
            "non_involutive",
            "independent uniform companions c(i) != i, all colours valid, all walkers alive",
            20260919,
            ((0.35, 0.9), (1.0, 0.6)),
        ),
        build_case(
            "mutual_pairing_odd",
            "Fisher-Yates mutual pairing for both roles with N = 9: one self companion per role "
            "and frame",
            20260920,
            ((0.25, 0.7), (0.8, 0.5)),
        ),
        build_case(
            "masked",
            "dead walkers, zero-force colours (one frame entirely), out-of-range and self "
            "companion indices, coincident positions",
            20260921,
            ((0.5, 1.2), (1.5, 0.4)),
        ),
    ]
    for case in cases:
        text = json.dumps(case, allow_nan=False, separators=(",", ":")) + "\n"
        size = len(text.encode())
        (OUTPUT / f"{case['name']}.json").write_text(text)
        print(f"{case['name']}: {size} bytes")
        for block in ("series", "electroweak"):
            print(f"  {block} float32 roundoff {case[block]['float32_roundoff']:.3e}")
        if "exchange_cancellation" in case:
            print("  " + json.dumps(case["exchange_cancellation"], indent=1).replace("\n", "\n  "))
        assert size < MAX_BYTES, (case["name"], size)
    print(f"wrote {len(cases)} fixtures to {OUTPUT}")


if __name__ == "__main__":
    main()
