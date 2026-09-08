"""Cross-language objective and C ABI contracts for Optimization Lab."""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fragile.fractalai.core import benchmarks as bm


ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "fractal-gas-web/build-optimization-native/optimization/libfg_optimization.so"


@pytest.fixture(scope="module")
def native():
    if not LIB.exists():
        pytest.skip("Run make optimization-native to build the native library")
    lib = ctypes.CDLL(str(LIB))
    lib.fgo_create.argtypes = [ctypes.c_char_p]
    lib.fgo_create.restype = ctypes.c_uint32
    lib.fgo_error.restype = ctypes.c_char_p
    lib.fgo_catalog.restype = ctypes.c_char_p
    lib.fgo_config.argtypes = [ctypes.c_uint32]
    lib.fgo_config.restype = ctypes.c_char_p
    lib.fgo_destroy.argtypes = [ctypes.c_uint32]
    lib.fgo_step.argtypes = [ctypes.c_uint32]
    lib.fgo_snapshot_size.argtypes = [ctypes.c_uint32]
    lib.fgo_snapshot.argtypes = [ctypes.c_uint32]
    lib.fgo_snapshot.restype = ctypes.POINTER(ctypes.c_double)
    lib.fgo_sample.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    return lib


def create(lib, config):
    handle = lib.fgo_create(json.dumps(config).encode())
    assert handle, lib.fgo_error().decode()
    return handle


def sample(lib, handle, points):
    points = np.ascontiguousarray(points, dtype=np.float32)
    result = np.empty(points.shape[0], dtype=np.float64)
    assert (
        lib.fgo_sample(
            handle,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(points),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        == 0
    )
    return result


CASES = [
    ("sphere", bm.Sphere, {}),
    ("quadratic", bm.QuadraticWell, {}),
    ("mexican_hat", bm.MexicanHat, {"tilt": 0.2}),
    ("rastrigin", bm.Rastrigin, {}),
    ("eggholder", bm.EggHolder, {}),
    ("styblinski_tang", bm.StyblinskiTang, {}),
    ("rosenbrock", bm.Rosenbrock, {}),
    ("easom", bm.Easom, {}),
    ("holder_table", bm.HolderTable, {}),
    ("constant", bm.Constant, {}),
]


@pytest.mark.parametrize(("name", "constructor", "params"), CASES)
def test_benchmark_python_parity(native, name, constructor, params):
    config = {"benchmark": name, "dimensions": 2, "walkers": 4, **params}
    handle = create(native, config)
    try:
        points = np.random.default_rng(8).uniform(-3, 3, (32, 2)).astype(np.float32)
        expected = constructor(dims=2, **params)(torch.from_numpy(points)).numpy()
        np.testing.assert_allclose(sample(native, handle, points), expected, rtol=2e-5, atol=2e-5)
    finally:
        native.fgo_destroy(handle)


def test_mixture_realized_parameters_and_lennard_jones(native):
    handle = create(native, {"benchmark": "gaussian_mixture", "dimensions": 5, "walkers": 4})
    try:
        config = json.loads(native.fgo_config(handle))
        points = np.random.default_rng(9).uniform(-5, 5, (16, 5)).astype(np.float32)
        reference = bm.MixtureOfGaussians(
            dims=5,
            n_gaussians=config["n_gaussians"],
            centers=np.array(config["centers"]).reshape(-1, 5),
            stds=np.array(config["stds"]).reshape(-1, 5),
            weights=np.array(config["weights"]),
        )
        np.testing.assert_allclose(
            sample(native, handle, points), reference(torch.from_numpy(points)), rtol=3e-6
        )
    finally:
        native.fgo_destroy(handle)
    handle = create(native, {"benchmark": "lennard_jones", "n_atoms": 2, "walkers": 4})
    try:
        points = np.array([[0, 0, 0, 2 ** (1 / 6), 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
        values = sample(native, handle, points)
        assert values[0] == pytest.approx(-1, abs=1e-6)
        assert np.isinf(values[1])
    finally:
        native.fgo_destroy(handle)


def test_catalog_noise_and_invalid_handles(native):
    assert len(json.loads(native.fgo_catalog())["benchmarks"]) == 13
    handle = create(native, {"benchmark": "stochastic_gaussian", "dimensions": 3, "walkers": 4})
    try:
        initial = np.ctypeslib.as_array(
            native.fgo_snapshot(handle), shape=(native.fgo_snapshot_size(handle),)
        ).copy()
        np.testing.assert_array_equal(sample(native, handle, np.ones((100, 3))), 0)
        current = np.ctypeslib.as_array(native.fgo_snapshot(handle), shape=initial.shape).copy()
        np.testing.assert_array_equal(initial, current)
        assert not json.loads(native.fgo_config(handle))["potential_force"]
    finally:
        native.fgo_destroy(handle)
    assert native.fgo_step(handle) == -1
    assert b"Invalid optimization session" in native.fgo_error()
    assert not native.fgo_create(b'{"dimensions":-1}')


def test_plain_baoab_uses_half_duration_force_kicks(monkeypatch):
    from fragile.fractalai.core.euclidean_gas import SwarmState
    from fragile.fractalai.core.kinetic_operator import KineticOperator

    monkeypatch.setattr(torch, "randn", lambda *shape, **_kwargs: torch.zeros(shape))

    def potential(x):
        return -2 * x.sum(dim=1)  # constant force +2

    operator = KineticOperator(
        gamma=0.8,
        beta=1.2,
        delta_t=0.01,
        integrator="baoab",
        potential=potential,
        use_anisotropic_diffusion=False,
        use_viscous_coupling=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = operator.apply(SwarmState(torch.zeros(2, 2), torch.zeros(2, 2)))
    damp = np.exp(-0.8 * 0.01)
    torch.testing.assert_close(state.v, torch.full((2, 2), 0.01 * (1 + damp)))
    torch.testing.assert_close(state.x, torch.full((2, 2), 0.00005 * (1 + damp)))
