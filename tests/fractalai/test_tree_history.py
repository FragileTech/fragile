"""Tests for TreeHistory: graph-backed history recorder."""

from pathlib import Path
import tempfile

import pytest
import torch

from fragile.fractalai.core.tree_history import TreeHistory


# ---------------------------------------------------------------------------
# Mock SwarmState (minimal duck-type for the recording API)
# ---------------------------------------------------------------------------


class _MockState:
    """Minimal SwarmState stand-in."""

    def __init__(self, x: torch.Tensor, v: torch.Tensor):
        self.x = x
        self.v = v

    @property
    def N(self) -> int:
        return self.x.shape[0]

    @property
    def d(self) -> int:
        return self.x.shape[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, D = 4, 3


def _make_state(val: float = 0.0) -> _MockState:
    return _MockState(
        x=torch.full((N, D), val, dtype=torch.float32),
        v=torch.full((N, D), val + 0.5, dtype=torch.float32),
    )


def _make_info(step: int) -> dict:
    """Create a mock info dict matching VectorizedHistoryRecorder.record_step expectations."""
    return {
        "U_before": torch.full((N,), float(step)),
        "U_after_clone": torch.full((N,), float(step) + 0.1),
        "U_final": torch.full((N,), float(step) + 0.2),
        "fitness": torch.full((N,), float(step) * 10.0),
        "rewards": torch.full((N,), float(step) * 0.1),
        "cloning_scores": torch.full((N,), 0.5),
        "cloning_probs": torch.full((N,), 0.25),
        "will_clone": torch.tensor([step % 2 == 0] * N, dtype=torch.bool),
        "alive_mask": torch.ones(N, dtype=torch.bool),
        "companions_distance": torch.arange(N, dtype=torch.long),
        "companions_clone": torch.arange(N, dtype=torch.long),
        "clone_jitter": torch.zeros(N, D),
        "clone_delta_x": torch.zeros(N, D),
        "clone_delta_v": torch.zeros(N, D),
        "distances": torch.full((N,), 1.0),
        "z_rewards": torch.zeros(N),
        "z_distances": torch.zeros(N),
        "pos_squared_differences": torch.zeros(N),
        "vel_squared_differences": torch.zeros(N),
        "rescaled_rewards": torch.full((N,), float(step) * 0.1),
        "rescaled_distances": torch.full((N,), 1.0),
        "num_cloned": 0,
        "mu_rewards": torch.tensor(0.1),
        "sigma_rewards": torch.tensor(0.01),
        "mu_distances": torch.tensor(1.0),
        "sigma_distances": torch.tensor(0.1),
    }


def _record_mock(n_steps: int = 3) -> TreeHistory:
    """Create a TreeHistory with n_steps of mock data."""
    hist = TreeHistory(N=N, d=D)
    state0 = _make_state(0.0)
    hist.record_initial_state(state0, n_alive=N)
    for t in range(1, n_steps + 1):
        sb = _make_state(float(t))
        sc = _make_state(float(t) + 0.1)
        sf = _make_state(float(t) + 0.2)
        info = _make_info(t)
        hist.record_step(sb, sc, sf, info, step_time=0.01 * t)
    hist.build(
        n_steps=n_steps,
        record_every=1,
        terminated_early=False,
        final_step=n_steps,
        total_time=1.0,
        init_time=0.1,
        recorded_steps=list(range(n_steps + 1)),
    )
    return hist


# ---------------------------------------------------------------------------
# Tests: Construction and recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_initial_state_sets_n_recorded(self):
        hist = TreeHistory(N=N, d=D)
        state = _make_state()
        hist.record_initial_state(state, n_alive=N)
        assert hist.n_recorded == 1

    def test_record_step_increments_n_recorded(self):
        hist = TreeHistory(N=N, d=D)
        state = _make_state()
        hist.record_initial_state(state, n_alive=N)
        info = _make_info(1)
        hist.record_step(
            _make_state(1.0), _make_state(1.1), _make_state(1.2), info, step_time=0.01
        )
        assert hist.n_recorded == 2


# ---------------------------------------------------------------------------
# Tests: Dense tensor shapes and values
# ---------------------------------------------------------------------------


class TestDenseTensors:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.hist = _record_mock(3)

    def test_x_final_shape(self):
        # Full-axis: [n_recorded, N, d] = [4, 4, 3]
        assert self.hist.x_final.shape == (4, N, D)

    def test_v_final_shape(self):
        assert self.hist.v_final.shape == (4, N, D)

    def test_x_after_clone_shape(self):
        # Minus-one axis: [n_recorded-1, N, d] = [3, 4, 3]
        assert self.hist.x_after_clone.shape == (3, N, D)

    def test_fitness_shape(self):
        assert self.hist.fitness.shape == (3, N)

    def test_alive_mask_shape(self):
        assert self.hist.alive_mask.shape == (3, N)

    def test_will_clone_dtype(self):
        assert self.hist.will_clone.dtype == torch.bool

    def test_companions_distance_dtype(self):
        assert self.hist.companions_distance.dtype == torch.long

    def test_n_alive_shape(self):
        assert self.hist.n_alive.shape == (4,)

    def test_mu_rewards_shape(self):
        assert self.hist.mu_rewards.shape == (3,)

    def test_step_times_shape(self):
        assert self.hist.step_times.shape == (3,)

    def test_force_fields_shape(self):
        # No kinetic_info was passed → should be zeros
        assert self.hist.force_stable.shape == (3, N, D)

    def test_x_final_values(self):
        # At step 0, x_final should be 0.0 (initial state)
        assert torch.allclose(self.hist.x_final[0], torch.zeros(N, D))
        # At step 1, x_final should be 1.2 (t + 0.2 for final state)
        assert torch.allclose(self.hist.x_final[1], torch.full((N, D), 1.2))

    def test_fitness_values(self):
        # At step 1 (index 0 in minus-one), fitness = step * 10 = 10.0
        assert torch.allclose(self.hist.fitness[0], torch.full((N,), 10.0))

    def test_U_before_values(self):
        # Step 0: U_before = 0.0 (default)
        assert torch.allclose(self.hist.U_before[0], torch.zeros(N))


# ---------------------------------------------------------------------------
# Tests: Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_cache_invalidated_on_record_step(self):
        hist = TreeHistory(N=N, d=D)
        hist.record_initial_state(_make_state(), n_alive=N)
        info = _make_info(1)
        hist.record_step(
            _make_state(1.0), _make_state(1.1), _make_state(1.2), info, step_time=0.01
        )
        _ = hist.x_final  # populate cache
        assert len(hist._cache) > 0

        # Record another step → cache should be cleared
        info2 = _make_info(2)
        hist.record_step(
            _make_state(2.0), _make_state(2.1), _make_state(2.2), info2, step_time=0.02
        )
        assert len(hist._cache) == 0

    def test_cache_reused_on_repeated_access(self):
        hist = _record_mock(2)
        _ = hist.x_final
        cache_size_1 = len(hist._cache)
        _ = hist.x_final  # should reuse
        cache_size_2 = len(hist._cache)
        assert cache_size_1 == cache_size_2


# ---------------------------------------------------------------------------
# Tests: Query API
# ---------------------------------------------------------------------------


class TestQueryAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.hist = _record_mock(3)

    def test_get_walker_trajectory_final(self):
        traj = self.hist.get_walker_trajectory(0, stage="final")
        assert traj["x"].shape == (4, D)
        assert traj["v"].shape == (4, D)

    def test_get_walker_trajectory_before_clone(self):
        traj = self.hist.get_walker_trajectory(0, stage="before_clone")
        assert traj["x"].shape == (4, D)

    def test_get_walker_trajectory_after_clone(self):
        traj = self.hist.get_walker_trajectory(0, stage="after_clone")
        assert traj["x"].shape == (3, D)

    def test_get_walker_trajectory_invalid_stage(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            self.hist.get_walker_trajectory(0, stage="invalid")

    def test_get_clone_events(self):
        events = self.hist.get_clone_events()
        # will_clone is True at even steps: step 2 (t=2) has will_clone=True
        # Step 1 → will_clone=False (1%2=1), Step 2 → True (2%2=0), Step 3 → False
        assert isinstance(events, list)
        # At step 2 all N walkers clone
        clone_at_2 = [(s, w, c) for s, w, c in events if s == 2]
        assert len(clone_at_2) == N

    def test_get_alive_walkers_step0(self):
        walkers = self.hist.get_alive_walkers(0)
        assert walkers.shape == (N,)

    def test_get_alive_walkers_later_step(self):
        walkers = self.hist.get_alive_walkers(1)
        assert walkers.shape == (N,)  # all alive in mock

    def test_get_step_index(self):
        assert self.hist.get_step_index(0) == 0
        assert self.hist.get_step_index(3) == 3

    def test_get_step_index_invalid(self):
        with pytest.raises(ValueError, match="not recorded"):
            self.hist.get_step_index(999)

    def test_summary(self):
        s = self.hist.summary()
        assert "TreeHistory" in s
        assert "walkers" in s


# ---------------------------------------------------------------------------
# Tests: Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        hist = _record_mock(2)
        path = str(tmp_path / "tree_hist.pt")
        hist.save(path)
        loaded = TreeHistory.load(path)

        assert loaded.N == hist.N
        assert loaded.d == hist.d
        assert loaded.n_recorded == hist.n_recorded
        assert torch.allclose(loaded.x_final, hist.x_final)
        assert torch.allclose(loaded.fitness, hist.fitness)
        assert torch.allclose(loaded.n_alive, hist.n_alive)


# ---------------------------------------------------------------------------
# Tests: Conversion to RunHistory
# ---------------------------------------------------------------------------


class TestToRunHistory:
    def test_to_run_history_produces_valid_object(self):
        hist = _record_mock(3)
        rh = hist.to_run_history()
        from fragile.fractalai.core.history import RunHistory

        assert isinstance(rh, RunHistory)
        assert rh.N == N
        assert rh.d == D
        assert rh.n_recorded == 4
        assert rh.x_final.shape == (4, N, D)
        assert rh.fitness.shape == (3, N)

    def test_to_run_history_values_match(self):
        hist = _record_mock(2)
        rh = hist.to_run_history()
        assert torch.allclose(rh.x_final, hist.x_final)
        assert torch.allclose(rh.fitness, hist.fitness)
        assert torch.allclose(rh.alive_mask, hist.alive_mask)


# ---------------------------------------------------------------------------
# Tests: With kinetic info
# ---------------------------------------------------------------------------


class TestWithKineticInfo:
    def test_force_fields_recorded(self):
        hist = TreeHistory(N=N, d=D)
        hist.record_initial_state(_make_state(), n_alive=N)
        info = _make_info(1)
        kinetic = {
            "force_stable": torch.ones(N, D),
            "force_adapt": torch.zeros(N, D),
            "force_viscous": torch.zeros(N, D),
            "force_friction": torch.full((N, D), -0.1),
            "force_total": torch.ones(N, D) * 0.9,
            "noise": torch.randn(N, D),
        }
        hist.record_step(
            _make_state(1.0),
            _make_state(1.1),
            _make_state(1.2),
            info,
            step_time=0.01,
            kinetic_info=kinetic,
        )
        assert torch.allclose(hist.force_stable, torch.ones(1, N, D))
        assert hist.force_friction.shape == (1, N, D)

    def test_gradient_fields_recorded(self):
        hist = TreeHistory(N=N, d=D)
        hist.record_initial_state(_make_state(), n_alive=N)
        info = _make_info(1)
        grad = torch.randn(N, D)
        hist.record_step(
            _make_state(1.0),
            _make_state(1.1),
            _make_state(1.2),
            info,
            step_time=0.01,
            grad_fitness=grad,
        )
        assert hist.fitness_gradients is not None
        assert hist.fitness_gradients.shape == (1, N, D)
