"""Tests for RunHistory and VectorizedHistoryRecorder.

Tests cover:
- RunHistory construction, indexing, trajectory extraction, clone events,
  alive walkers, summary, serialization, and dict conversion.
- VectorizedHistoryRecorder allocation, initial state recording, step
  recording, build trimming, and optional field handling.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.fractal_gas.euclidean_gas import SwarmState
from fragile.physics.fractal_gas.history import RunHistory, VectorizedHistoryRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_history(N: int = 5, d: int = 2, n_steps: int = 3, n_recorded: int = 4) -> RunHistory:
    """Create a minimal valid RunHistory for testing."""
    T = n_recorded
    T1 = T - 1  # minus-one indexed fields
    return RunHistory(
        N=N,
        d=d,
        n_steps=n_steps,
        n_recorded=T,
        record_every=1,
        terminated_early=False,
        final_step=n_steps,
        recorded_steps=list(range(T)),
        delta_t=0.01,
        x_before_clone=torch.randn(T, N, d),
        v_before_clone=torch.randn(T, N, d),
        U_before=torch.randn(T, N),
        x_after_clone=torch.randn(T1, N, d),
        v_after_clone=torch.randn(T1, N, d),
        U_after_clone=torch.randn(T1, N),
        x_final=torch.randn(T, N, d),
        v_final=torch.randn(T, N, d),
        U_final=torch.randn(T, N),
        n_alive=torch.full((T,), N, dtype=torch.long),
        num_cloned=torch.zeros(T1, dtype=torch.long),
        step_times=torch.rand(T1),
        fitness=torch.randn(T1, N),
        rewards=torch.randn(T1, N),
        cloning_scores=torch.randn(T1, N),
        cloning_probs=torch.rand(T1, N),
        will_clone=torch.zeros(T1, N, dtype=torch.bool),
        alive_mask=torch.ones(T1, N, dtype=torch.bool),
        companions_distance=torch.zeros(T1, N, dtype=torch.long),
        companions_clone=torch.zeros(T1, N, dtype=torch.long),
        clone_jitter=torch.zeros(T1, N, d),
        clone_delta_x=torch.zeros(T1, N, d),
        clone_delta_v=torch.zeros(T1, N, d),
        distances=torch.rand(T1, N),
        z_rewards=torch.randn(T1, N),
        z_distances=torch.randn(T1, N),
        pos_squared_differences=torch.rand(T1, N),
        vel_squared_differences=torch.rand(T1, N),
        rescaled_rewards=torch.rand(T1, N),
        rescaled_distances=torch.rand(T1, N),
        mu_rewards=torch.randn(T1),
        sigma_rewards=torch.rand(T1) + 0.1,
        mu_distances=torch.randn(T1),
        sigma_distances=torch.rand(T1) + 0.1,
        force_stable=torch.zeros(T1, N, d),
        force_adapt=torch.zeros(T1, N, d),
        force_viscous=torch.zeros(T1, N, d),
        force_friction=torch.zeros(T1, N, d),
        force_total=torch.zeros(T1, N, d),
        noise=torch.randn(T1, N, d),
        total_time=1.0,
        init_time=0.1,
    )


def _make_info(N: int, d: int) -> dict:
    """Create a minimal info dict matching what record_step expects."""
    return {
        "U_before": torch.randn(N),
        "U_after_clone": torch.randn(N),
        "U_final": torch.randn(N),
        "alive_mask": torch.ones(N, dtype=torch.bool),
        "num_cloned": 0,
        "fitness": torch.randn(N),
        "rewards": torch.randn(N),
        "cloning_scores": torch.randn(N),
        "cloning_probs": torch.rand(N),
        "will_clone": torch.zeros(N, dtype=torch.bool),
        "companions_distance": torch.zeros(N, dtype=torch.long),
        "companions_clone": torch.zeros(N, dtype=torch.long),
        "clone_jitter": torch.zeros(N, d),
        "clone_delta_x": torch.zeros(N, d),
        "clone_delta_v": torch.zeros(N, d),
        "distances": torch.rand(N),
        "z_rewards": torch.randn(N),
        "z_distances": torch.randn(N),
        "pos_squared_differences": torch.rand(N),
        "vel_squared_differences": torch.rand(N),
        "rescaled_rewards": torch.rand(N),
        "rescaled_distances": torch.rand(N),
        "mu_rewards": torch.tensor(0.0),
        "sigma_rewards": torch.tensor(1.0),
        "mu_distances": torch.tensor(0.0),
        "sigma_distances": torch.tensor(1.0),
    }


def _make_kinetic_info(N: int, d: int) -> dict:
    """Create a minimal kinetic_info dict for record_step."""
    return {
        "force_stable": torch.randn(N, d),
        "force_adapt": torch.randn(N, d),
        "force_viscous": torch.randn(N, d),
        "force_friction": torch.randn(N, d),
        "force_total": torch.randn(N, d),
        "noise": torch.randn(N, d),
    }


# ===========================================================================
# TestRunHistory
# ===========================================================================


class TestRunHistory:
    """Tests for the RunHistory Pydantic model."""

    def test_construction_valid(self):
        """RunHistory can be constructed with all required fields."""
        h = _make_run_history()
        assert h.N == 5
        assert h.d == 2
        assert h.n_recorded == 4
        assert h.n_steps == 3

    def test_get_step_index_valid(self):
        """get_step_index returns correct index for a recorded step."""
        h = _make_run_history(n_recorded=4)
        # recorded_steps = [0, 1, 2, 3]
        assert h.get_step_index(0) == 0
        assert h.get_step_index(2) == 2
        assert h.get_step_index(3) == 3

    def test_get_step_index_raises_for_unrecorded(self):
        """get_step_index raises ValueError for a step that was not recorded."""
        h = _make_run_history(n_recorded=4)
        # recorded_steps = [0, 1, 2, 3], so step 10 is not recorded
        with pytest.raises(ValueError, match="Step 10 was not recorded"):
            h.get_step_index(10)

    def test_get_walker_trajectory_final(self):
        """get_walker_trajectory returns x and v with shape [n_recorded, d] for 'final' stage."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        traj = h.get_walker_trajectory(0, stage="final")
        assert "x" in traj
        assert "v" in traj
        assert traj["x"].shape == (4, 2)
        assert traj["v"].shape == (4, 2)
        # Verify it extracts the correct walker slice
        assert torch.allclose(traj["x"], h.x_final[:, 0, :])
        assert torch.allclose(traj["v"], h.v_final[:, 0, :])

    def test_get_walker_trajectory_before_clone(self):
        """get_walker_trajectory for 'before_clone' stage returns [n_recorded, d]."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        traj = h.get_walker_trajectory(2, stage="before_clone")
        assert traj["x"].shape == (4, 2)
        assert traj["v"].shape == (4, 2)
        assert torch.allclose(traj["x"], h.x_before_clone[:, 2, :])

    def test_get_walker_trajectory_after_clone(self):
        """get_walker_trajectory for 'after_clone' stage returns [n_recorded-1, d]."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        traj = h.get_walker_trajectory(1, stage="after_clone")
        assert traj["x"].shape == (3, 2)  # n_recorded - 1
        assert traj["v"].shape == (3, 2)
        assert torch.allclose(traj["x"], h.x_after_clone[:, 1, :])

    def test_get_walker_trajectory_unknown_stage_raises(self):
        """get_walker_trajectory raises ValueError for unknown stage."""
        h = _make_run_history()
        with pytest.raises(ValueError, match="Unknown stage: invalid"):
            h.get_walker_trajectory(0, stage="invalid")

    def test_get_clone_events(self):
        """get_clone_events returns correct (step, cloner_idx, companion_idx) tuples."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        # Set up: at time index 1 (step=2), walker 0 clones from walker 3
        h.will_clone[1, 0] = True
        h.companions_clone[1, 0] = 3
        events = h.get_clone_events()
        assert len(events) == 1
        step, cloner, companion = events[0]
        assert step == 2  # recorded_steps[1+1] = recorded_steps[2] = 2
        assert cloner == 0
        assert companion == 3

    def test_get_clone_events_empty(self):
        """get_clone_events returns empty list when no cloning occurred."""
        h = _make_run_history()
        # will_clone is all False by default
        assert h.get_clone_events() == []

    def test_get_alive_walkers_step_zero(self):
        """get_alive_walkers at step 0 returns all walker indices."""
        h = _make_run_history(N=5, n_recorded=4)
        alive = h.get_alive_walkers(0)
        assert torch.equal(alive, torch.arange(5))

    def test_get_alive_walkers_later_step(self):
        """get_alive_walkers uses alive_mask for steps after t=0."""
        h = _make_run_history(N=5, n_recorded=4)
        # step 2 => index 2, alive_mask uses index 2-1=1
        h.alive_mask[1] = torch.tensor([True, False, True, False, True])
        alive = h.get_alive_walkers(2)
        assert torch.equal(alive, torch.tensor([0, 2, 4]))

    def test_summary_contains_key_info(self):
        """summary() returns a string containing steps, walkers, dimension."""
        h = _make_run_history(N=5, d=2, n_steps=3)
        s = h.summary()
        assert "3 steps" in s
        assert "5 walkers" in s
        assert "2D" in s
        assert "terminated_early=False" in s

    def test_summary_with_gradients(self):
        """summary() mentions gradient recording when fitness_gradients is set."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        T1 = h.n_recorded - 1
        h.fitness_gradients = torch.zeros(T1, h.N, h.d)
        s = h.summary()
        assert "gradients recorded" in s

    def test_to_dict_excludes_none(self):
        """to_dict() excludes None-valued fields."""
        h = _make_run_history()
        d = h.to_dict()
        # Optional fields that are None should not appear
        assert "fitness_gradients" not in d
        assert "fitness_hessians_diag" not in d
        # Required fields should appear
        assert "N" in d
        assert "x_final" in d

    def test_save_load_roundtrip(self, tmp_path):
        """save() then load() produces equivalent RunHistory."""
        h = _make_run_history(N=5, d=2, n_recorded=4)
        path = str(tmp_path / "test_history.pt")
        h.save(path)
        h2 = RunHistory.load(path)
        assert h2.N == h.N
        assert h2.d == h.d
        assert h2.n_recorded == h.n_recorded
        assert h2.n_steps == h.n_steps
        assert h2.recorded_steps == h.recorded_steps
        assert torch.allclose(h2.x_final, h.x_final)
        assert torch.allclose(h2.v_final, h.v_final)
        assert torch.allclose(h2.rewards, h.rewards)


# ===========================================================================
# TestVectorizedHistoryRecorder
# ===========================================================================


class TestVectorizedHistoryRecorder:
    """Tests for the VectorizedHistoryRecorder builder class."""

    def test_init_allocates_correct_shapes(self):
        """__init__ pre-allocates tensors with expected shapes."""
        N, d, n_rec = 10, 3, 6
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        assert rec.x_before_clone.shape == (n_rec, N, d)
        assert rec.v_before_clone.shape == (n_rec, N, d)
        assert rec.x_after_clone.shape == (n_rec - 1, N, d)
        assert rec.fitness.shape == (n_rec - 1, N)
        assert rec.n_alive.shape == (n_rec,)
        assert rec.num_cloned.shape == (n_rec - 1,)
        assert rec.force_total.shape == (n_rec - 1, N, d)

    def test_record_initial_state(self):
        """record_initial_state stores state at index 0."""
        N, d, n_rec = 8, 2, 5
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        x = torch.randn(N, d, dtype=torch.float64)
        v = torch.randn(N, d, dtype=torch.float64)
        state = SwarmState(x, v)
        rec.record_initial_state(state, n_alive=N)

        assert torch.allclose(rec.x_before_clone[0], x)
        assert torch.allclose(rec.v_before_clone[0], v)
        assert torch.allclose(rec.x_final[0], x)
        assert torch.allclose(rec.v_final[0], v)
        assert rec.n_alive[0].item() == N

    def test_record_step_stores_at_correct_index(self):
        """record_step writes data at recorded_idx and increments it."""
        N, d, n_rec = 6, 2, 5
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        x0 = torch.randn(N, d, dtype=torch.float64)
        v0 = torch.randn(N, d, dtype=torch.float64)
        rec.record_initial_state(SwarmState(x0, v0), n_alive=N)
        assert rec.recorded_idx == 1

        # Record one step
        x_before = torch.randn(N, d, dtype=torch.float64)
        v_before = torch.randn(N, d, dtype=torch.float64)
        x_cloned = torch.randn(N, d, dtype=torch.float64)
        v_cloned = torch.randn(N, d, dtype=torch.float64)
        x_final = torch.randn(N, d, dtype=torch.float64)
        v_final = torch.randn(N, d, dtype=torch.float64)
        info = _make_info(N, d)
        kinetic_info = _make_kinetic_info(N, d)

        rec.record_step(
            state_before=SwarmState(x_before, v_before),
            state_cloned=SwarmState(x_cloned, v_cloned),
            state_final=SwarmState(x_final, v_final),
            info=info,
            step_time=0.05,
            kinetic_info=kinetic_info,
        )

        assert rec.recorded_idx == 2
        # Check the state was stored at index 1
        assert torch.allclose(rec.x_before_clone[1], x_before)
        assert torch.allclose(rec.x_final[1], x_final)
        # After-clone stored at idx_minus_1 = 0
        assert torch.allclose(rec.x_after_clone[0], x_cloned)

    def test_build_produces_run_history_with_correct_n_recorded(self):
        """build() produces RunHistory with n_recorded matching actual recordings."""
        N, d, n_rec = 6, 2, 5
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        x0 = torch.randn(N, d, dtype=torch.float64)
        v0 = torch.randn(N, d, dtype=torch.float64)
        rec.record_initial_state(SwarmState(x0, v0), n_alive=N)

        # Record 2 steps (so recorded_idx goes to 3)
        for _ in range(2):
            rec.record_step(
                state_before=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_cloned=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_final=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                info=_make_info(N, d),
                step_time=0.01,
                kinetic_info=_make_kinetic_info(N, d),
            )

        history = rec.build(
            record_every=1,
            terminated_early=False,
            final_step=2,
            total_time=0.5,
            init_time=0.05,
            recorded_steps=[0, 1, 2],
            delta_t=0.01,
        )

        assert isinstance(history, RunHistory)
        assert history.n_recorded == 3  # initial + 2 steps

    def test_build_trims_unused_preallocated_space(self):
        """build() trims tensors to actual_recorded, not the full pre-allocated size."""
        N, d, n_rec = 4, 2, 10  # pre-allocate 10, but only record 2
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        rec.record_initial_state(
            SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            n_alive=N,
        )
        rec.record_step(
            state_before=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_cloned=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_final=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            info=_make_info(N, d),
            step_time=0.01,
            kinetic_info=_make_kinetic_info(N, d),
        )

        history = rec.build(
            record_every=1,
            terminated_early=False,
            final_step=1,
            total_time=0.1,
            init_time=0.01,
            recorded_steps=[0, 1],
            delta_t=0.01,
        )

        # Should be trimmed to 2 (initial + 1 step), not 10
        assert history.n_recorded == 2
        assert history.x_final.shape[0] == 2
        assert history.x_after_clone.shape[0] == 1  # n_recorded - 1
        assert history.fitness.shape[0] == 1

    def test_optional_fields_none_when_not_recorded(self):
        """Optional fields are None in build output when flags are False."""
        N, d, n_rec = 4, 2, 3
        rec = VectorizedHistoryRecorder(
            N=N,
            d=d,
            n_recorded=n_rec,
            device=torch.device("cpu"),
            dtype=torch.float64,
            record_gradients=False,
            record_hessians_diag=False,
            record_hessians_full=False,
        )
        rec.record_initial_state(
            SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            n_alive=N,
        )
        rec.record_step(
            state_before=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_cloned=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_final=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            info=_make_info(N, d),
            step_time=0.01,
            kinetic_info=_make_kinetic_info(N, d),
        )

        history = rec.build(
            record_every=1,
            terminated_early=False,
            final_step=1,
            total_time=0.1,
            init_time=0.01,
            recorded_steps=[0, 1],
            delta_t=0.01,
        )

        assert history.fitness_gradients is None
        assert history.fitness_hessians_diag is None
        assert history.fitness_hessians_full is None

    def test_optional_fields_allocated_when_flags_true(self):
        """Optional fields are allocated when corresponding flags are True."""
        N, d, n_rec = 4, 2, 3
        rec = VectorizedHistoryRecorder(
            N=N,
            d=d,
            n_recorded=n_rec,
            device=torch.device("cpu"),
            dtype=torch.float64,
            record_gradients=True,
            record_hessians_diag=True,
            record_hessians_full=True,
        )
        assert rec.fitness_gradients is not None
        assert rec.fitness_gradients.shape == (n_rec - 1, N, d)
        assert rec.fitness_hessians_diag is not None
        assert rec.fitness_hessians_diag.shape == (n_rec - 1, N, d)
        assert rec.fitness_hessians_full is not None
        assert rec.fitness_hessians_full.shape == (n_rec - 1, N, d, d)

    def test_optional_gradients_in_build_output(self):
        """When record_gradients=True, build output contains fitness_gradients tensor."""
        N, d, n_rec = 4, 2, 3
        rec = VectorizedHistoryRecorder(
            N=N,
            d=d,
            n_recorded=n_rec,
            device=torch.device("cpu"),
            dtype=torch.float64,
            record_gradients=True,
        )
        rec.record_initial_state(
            SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            n_alive=N,
        )
        grad = torch.randn(N, d, dtype=torch.float64)
        rec.record_step(
            state_before=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_cloned=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            state_final=SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            info=_make_info(N, d),
            step_time=0.01,
            grad_fitness=grad,
            kinetic_info=_make_kinetic_info(N, d),
        )

        history = rec.build(
            record_every=1,
            terminated_early=False,
            final_step=1,
            total_time=0.1,
            init_time=0.01,
            recorded_steps=[0, 1],
            delta_t=0.01,
        )

        assert history.fitness_gradients is not None
        assert history.fitness_gradients.shape == (1, N, d)
        assert torch.allclose(history.fitness_gradients[0], grad)

    def test_multiple_steps_recorded_in_sequence(self):
        """Recording multiple steps produces correctly ordered data."""
        N, d, n_rec = 4, 2, 6
        rec = VectorizedHistoryRecorder(
            N=N, d=d, n_recorded=n_rec, device=torch.device("cpu"), dtype=torch.float64
        )
        rec.record_initial_state(
            SwarmState(
                torch.zeros(N, d, dtype=torch.float64), torch.zeros(N, d, dtype=torch.float64)
            ),
            n_alive=N,
        )

        x_finals = []
        for step_i in range(4):
            x_f = torch.full((N, d), float(step_i + 1), dtype=torch.float64)
            x_finals.append(x_f)
            rec.record_step(
                state_before=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_cloned=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_final=SwarmState(x_f.clone(), torch.randn(N, d, dtype=torch.float64)),
                info=_make_info(N, d),
                step_time=0.01,
                kinetic_info=_make_kinetic_info(N, d),
            )

        history = rec.build(
            record_every=1,
            terminated_early=False,
            final_step=4,
            total_time=0.5,
            init_time=0.05,
            recorded_steps=[0, 1, 2, 3, 4],
            delta_t=0.01,
        )

        assert history.n_recorded == 5  # initial + 4 steps
        # x_final[0] should be all zeros (initial)
        assert torch.allclose(history.x_final[0], torch.zeros(N, d, dtype=torch.float64))
        # x_final[1..4] should be 1, 2, 3, 4
        for i, x_f in enumerate(x_finals):
            assert torch.allclose(history.x_final[i + 1], x_f)

    def test_recorded_idx_tracks_position(self):
        """recorded_idx starts at 1 and increments correctly after each record_step."""
        N, d, n_rec = 4, 2, 10
        rec = VectorizedHistoryRecorder(
            N=N,
            d=d,
            n_recorded=n_rec,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        assert rec.recorded_idx == 1  # Before initial state recording, already at 1

        rec.record_initial_state(
            SwarmState(
                torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
            ),
            n_alive=N,
        )
        # record_initial_state does not increment recorded_idx
        assert rec.recorded_idx == 1

        for expected_idx in range(1, 5):
            rec.record_step(
                state_before=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_cloned=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                state_final=SwarmState(
                    torch.randn(N, d, dtype=torch.float64), torch.randn(N, d, dtype=torch.float64)
                ),
                info=_make_info(N, d),
                step_time=0.01,
                kinetic_info=_make_kinetic_info(N, d),
            )
            assert rec.recorded_idx == expected_idx + 1
