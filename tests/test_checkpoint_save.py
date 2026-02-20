"""Tests for checkpoint save/load roundtrip."""

import os

import torch
from torch import nn

from fragile.learning.checkpoints import (
    _atomic_save,
    load_checkpoint,
    save_checkpoint,
    save_model_checkpoint,
)
from fragile.learning.config import TopoEncoderConfig


class _TinyModel(nn.Module):
    """Minimal model for checkpoint tests."""

    def __init__(self, in_dim: int = 4, out_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def _make_config(**overrides) -> TopoEncoderConfig:
    defaults = {
        "dataset": "mnist",
        "epochs": 2,
        "num_charts": 2,
        "codes_per_chart": 4,
        "hidden_dim": 8,
        "n_samples": 100,
        "save_every": 1,
    }
    defaults.update(overrides)
    return TopoEncoderConfig(**defaults)


class TestAtomicSave:
    def test_produces_nonzero_file(self, tmp_path):
        path = str(tmp_path / "test.pt")
        _atomic_save({"key": torch.randn(3, 3)}, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_loadable_after_save(self, tmp_path):
        path = str(tmp_path / "test.pt")
        data = {"tensor": torch.randn(5), "string": "hello", "number": 42}
        _atomic_save(data, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert torch.equal(loaded["tensor"], data["tensor"])
        assert loaded["string"] == "hello"
        assert loaded["number"] == 42

    def test_no_file_on_failure(self, tmp_path):
        path = str(tmp_path / "bad.pt")

        class Unpicklable:
            def __reduce__(self):
                msg = "cannot pickle"
                raise RuntimeError(msg)

        try:
            _atomic_save({"bad": Unpicklable()}, path)
        except Exception:
            pass
        assert not os.path.exists(path)

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dirs" / "test.pt")
        _atomic_save({"x": 1}, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestSaveCheckpoint:
    def test_roundtrip_with_all_models(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        atlas = _TinyModel(4, 2)
        jump = _TinyModel(2, 2)
        std = _TinyModel(4, 2)
        ae = _TinyModel(4, 2)

        path = str(tmp_path / "topoencoder" / "epoch_00001.pt")
        save_checkpoint(
            path,
            config,
            model_atlas=atlas,
            jump_op=jump,
            metrics={"atlas_losses": [1.0, 0.5]},
            data_snapshot={},
            epoch=1,
            model_std=std,
            model_ae=ae,
        )

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        ckpt = load_checkpoint(path)
        assert ckpt["epoch"] == 1
        assert "atlas_losses" in ckpt["metrics"]

        # Verify all model states are present
        state = ckpt["state"]
        assert state["atlas"] is not None
        assert state["jump"] is not None
        assert state["std"] is not None
        assert state["ae"] is not None

        # Verify state dicts can be loaded back
        atlas2 = _TinyModel(4, 2)
        atlas2.load_state_dict(state["atlas"])
        # Weights should match
        for p1, p2 in zip(atlas.parameters(), atlas2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())

        std2 = _TinyModel(4, 2)
        std2.load_state_dict(state["std"])
        for p1, p2 in zip(std.parameters(), std2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())

        ae2 = _TinyModel(4, 2)
        ae2.load_state_dict(state["ae"])
        for p1, p2 in zip(ae.parameters(), ae2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())

    def test_none_models_saved_as_none(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        atlas = _TinyModel(4, 2)
        jump = _TinyModel(2, 2)

        path = str(tmp_path / "topoencoder" / "epoch_00000.pt")
        save_checkpoint(
            path,
            config,
            model_atlas=atlas,
            jump_op=jump,
            metrics={},
            data_snapshot={},
            epoch=0,
            model_std=None,
            model_ae=None,
        )

        ckpt = load_checkpoint(path)
        assert ckpt["state"]["std"] is None
        assert ckpt["state"]["ae"] is None

    def test_optimizer_state_saved(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        atlas = _TinyModel(4, 2)
        jump = _TinyModel(2, 2)
        opt = torch.optim.Adam(atlas.parameters(), lr=0.01)
        # Do a step so optimizer has state
        loss = atlas(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()

        path = str(tmp_path / "topoencoder" / "epoch_00001.pt")
        save_checkpoint(
            path,
            config,
            model_atlas=atlas,
            jump_op=jump,
            metrics={},
            data_snapshot={},
            epoch=1,
            optimizer_atlas=opt,
        )

        ckpt = load_checkpoint(path)
        assert ckpt["optim"]["atlas"] is not None
        assert "state" in ckpt["optim"]["atlas"]


class TestSaveModelCheckpoint:
    def test_vq_roundtrip(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        model = _TinyModel(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        path = str(tmp_path / "vq" / "epoch_00005.pt")
        save_model_checkpoint(
            path,
            model,
            opt,
            config,
            epoch=5,
            hidden_dim=8,
            model_type="standard_vq",
            extra_metrics={"losses": [1.0, 0.8, 0.6]},
        )

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 5
        assert loaded["model_type"] == "standard_vq"
        assert loaded["hidden_dim"] == 8
        assert loaded["metrics"]["losses"] == [1.0, 0.8, 0.6]

        # Load state dict back
        model2 = _TinyModel(4, 2)
        model2.load_state_dict(loaded["state_dict"])
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1.cpu(), p2.cpu())

    def test_ae_roundtrip(self, tmp_path):
        config = _make_config(output_dir=str(tmp_path))
        model = _TinyModel(4, 2)

        path = str(tmp_path / "ae" / "epoch_00010.pt")
        save_model_checkpoint(
            path,
            model,
            None,
            config,
            epoch=10,
            hidden_dim=16,
            model_type="vanilla_ae",
        )

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 10
        assert loaded["model_type"] == "vanilla_ae"
        assert loaded["hidden_dim"] == 16
        assert loaded["optimizer"] is None


class TestSubdirectoryStructure:
    def test_all_three_folders_saved(self, tmp_path):
        """Simulate saving to all 3 subdirectories like the training loop does."""
        config = _make_config(output_dir=str(tmp_path))
        atlas = _TinyModel(4, 2)
        jump = _TinyModel(2, 2)
        std = _TinyModel(4, 2)
        ae = _TinyModel(4, 2)

        # Save topoencoder checkpoint
        topo_path = str(tmp_path / "topoencoder" / "epoch_00010.pt")
        save_checkpoint(
            topo_path,
            config,
            atlas,
            jump,
            metrics={"atlas_losses": [1.0]},
            data_snapshot={},
            epoch=10,
            model_std=std,
            model_ae=ae,
        )

        # Save VQ checkpoint
        vq_path = str(tmp_path / "vq" / "epoch_00010.pt")
        save_model_checkpoint(
            vq_path,
            std,
            None,
            config,
            epoch=10,
            hidden_dim=8,
            model_type="standard_vq",
        )

        # Save AE checkpoint
        ae_path = str(tmp_path / "ae" / "epoch_00010.pt")
        save_model_checkpoint(
            ae_path,
            ae,
            None,
            config,
            epoch=10,
            hidden_dim=16,
            model_type="vanilla_ae",
        )

        # Verify all exist and are non-zero
        for p in [topo_path, vq_path, ae_path]:
            assert os.path.exists(p), f"Missing: {p}"
            assert os.path.getsize(p) > 0, f"Empty: {p}"

        # Verify each is independently loadable
        topo_ckpt = load_checkpoint(topo_path)
        assert topo_ckpt["state"]["std"] is not None
        assert topo_ckpt["state"]["ae"] is not None

        vq_ckpt = torch.load(vq_path, map_location="cpu", weights_only=False)
        assert vq_ckpt["model_type"] == "standard_vq"

        ae_ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        assert ae_ckpt["model_type"] == "vanilla_ae"
