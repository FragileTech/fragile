"""SmolVLA feature extraction, caching, and dataset for VLA experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import VLAConfig


# ---------------------------------------------------------------------------
# Feature hook
# ---------------------------------------------------------------------------


class SmolVLAFeatureHook:
    """Context manager that hooks into a SmolVLA policy to capture hidden states.

    SmolVLA's ``SmolVLMWithExpertModel`` decomposes transformer layer
    operations manually (calling sub-modules like ``mlp``, ``o_proj``, etc.)
    so standard ``DecoderLayer`` forward hooks never fire.  Instead we hook
    into the **final RMSNorm** of the expert LM (``lm_expert.norm``), which
    *is* called via a normal ``module(x)`` invocation and gives us the
    post-norm hidden states from the last layer.
    """

    def __init__(self, policy: torch.nn.Module) -> None:
        self._policy = policy
        self._hook_handle = None
        self.features: torch.Tensor | None = None

    @staticmethod
    def _find_hook_target(policy: torch.nn.Module) -> tuple[str, torch.nn.Module]:
        """Find a hookable module that captures the final hidden states.

        Priority order:
        1. ``model.vlm_with_expert.lm_expert.norm`` (final RMSNorm, always called)
        2. Last MLP in the last decoder layer (called directly by the decomposed forward)
        3. Any RMSNorm named ``norm`` inside the model
        """
        # Strategy 1: lm_expert.norm (final layer norm)
        for name, module in policy.named_modules():
            if name.endswith("lm_expert.norm"):
                return name, module

        # Strategy 2: last MLP
        mlp_candidates: list[tuple[str, torch.nn.Module]] = []
        for name, module in policy.named_modules():
            if name.endswith(".mlp") and "expert" in name:
                mlp_candidates.append((name, module))
        if mlp_candidates:
            return mlp_candidates[-1]

        # Strategy 3: any final norm
        for name, module in policy.named_modules():
            cls = type(module).__name__
            if "RMSNorm" in cls and name.endswith(".norm"):
                return name, module

        msg = (
            "Could not find a hookable module in the SmolVLA policy. "
            "Check that the model is loaded correctly."
        )
        raise RuntimeError(msg)

    def __enter__(self) -> SmolVLAFeatureHook:
        name, target = self._find_hook_target(self._policy)
        print(f"  Hooking into: {name} ({type(target).__name__})")

        def _hook(_module: torch.nn.Module, _input: object, output: object) -> None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.features = hidden.detach()

        self._hook_handle = target.register_forward_hook(_hook)
        return self

    def __exit__(self, *_: object) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _build_episode_index(dataset) -> dict[int, list[int]]:
    """Build a mapping from episode_id -> list of dataset indices.

    Uses the dataset's parquet metadata when available (lerobot v0.4+),
    falling back to sequential scan.
    """
    # lerobot v0.4 stores episode info in meta
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        episodes = dataset.meta.episodes
        if episodes is not None:
            ep_ids = sorted(episodes.keys()) if isinstance(episodes, dict) else list(range(len(episodes)))
        else:
            ep_ids = None
    else:
        ep_ids = None

    # Build index by scanning the hf_dataset (fast, uses arrow)
    hf = dataset.hf_dataset
    ep_col = hf["episode_index"]

    if ep_ids is None:
        ep_ids = sorted(set(int(e) for e in ep_col))

    index: dict[int, list[int]] = {eid: [] for eid in ep_ids}
    for i, eid in enumerate(ep_col):
        eid_int = int(eid)
        if eid_int in index:
            index[eid_int].append(i)

    return index


def extract_smolvla_features(config: VLAConfig) -> Path:
    """Extract and cache 960-dim features from a frozen SmolVLA backbone.

    Iterates every frame in the lerobot dataset, mean-pools the token
    hidden states from the last transformer layer, and saves per-episode
    ``.pt`` files plus a ``meta.json`` at the cache root.

    Returns:
        Path to *feature_cache_dir* (for convenience).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoTokenizer

    cache_dir = Path(config.feature_cache_dir)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        print(f"Feature cache already exists at {cache_dir}, skipping extraction.")
        return cache_dir

    device = torch.device(config.device)

    # Load frozen policy (float16 for memory efficiency on 8GB GPU)
    print(f"Loading SmolVLA from {config.smolvla_model_id} …")
    policy = SmolVLAPolicy.from_pretrained(config.smolvla_model_id)
    policy.eval()
    policy.to(device=device, dtype=torch.float16)
    for p in policy.parameters():
        p.requires_grad_(False)

    # Load tokenizer for language conditioning
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Load dataset
    print(f"Loading dataset {config.dataset_name} …")
    dataset = LeRobotDataset(config.dataset_name)
    print(f"  Total frames: {len(dataset)}")

    # Build mapping from dataset image keys to policy-expected keys.
    # The smolvla_base model expects camera1/camera2/camera3 but datasets
    # may use different names (e.g. top/wrist for svla_so100_pickplace).
    policy_img_keys = sorted(policy.config.image_features.keys())
    sample0 = dataset[0]
    dataset_img_keys = sorted(
        k for k in sample0.keys() if k.startswith("observation.images.")
    )
    img_key_map: dict[str, str] = {}
    if set(dataset_img_keys) != set(policy_img_keys):
        for ds_k, pol_k in zip(dataset_img_keys, policy_img_keys):
            img_key_map[ds_k] = pol_k
        print(f"  Image key remap: {img_key_map}")

    # Pre-tokenize unique task descriptions
    task_tokens_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _tokenize_task(task_str: str) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            task_str, return_tensors="pt", padding="max_length",
            max_length=64, truncation=True,
        )
        return {
            "observation.language.tokens": encoded.input_ids.to(device=device),
            "observation.language.attention_mask": encoded.attention_mask.bool().to(device=device),
        }

    for idx in range(min(len(dataset), 100)):
        s = dataset[idx]
        task_str = s.get("task", "Pick up the object.")
        if task_str not in task_tokens_cache:
            task_tokens_cache[task_str] = _tokenize_task(task_str)
    if not task_tokens_cache:
        task_tokens_cache["Pick up the object."] = _tokenize_task("Pick up the object.")
    print(f"  Unique tasks: {len(task_tokens_cache)}")

    # Build per-episode index
    ep_index = _build_episode_index(dataset)
    episode_ids = sorted(ep_index.keys())
    if config.max_episodes > 0:
        episode_ids = episode_ids[: config.max_episodes]
    print(f"  Episodes to process: {len(episode_ids)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0

    with SmolVLAFeatureHook(policy) as hook:
        for ep_id in episode_ids:
            ep_dir = cache_dir / f"episode_{ep_id}"
            if (ep_dir / "features.pt").exists():
                n_cached = torch.load(ep_dir / "features.pt", weights_only=True).shape[0]
                total_frames += n_cached
                print(f"  Episode {ep_id}: {n_cached} frames (cached)")
                continue
            ep_dir.mkdir(parents=True, exist_ok=True)

            indices = ep_index[ep_id]
            feat_list: list[torch.Tensor] = []
            act_list: list[torch.Tensor] = []
            state_list: list[torch.Tensor] = []
            task_idx_list: list[int] = []

            for idx in indices:
                sample = dataset[idx]

                # Build observation dict for the policy, remapping image keys
                # to match the policy's expected feature names.
                obs = {}
                for k, v in sample.items():
                    if not k.startswith("observation"):
                        continue
                    mapped_k = img_key_map.get(k, k)
                    obs[mapped_k] = v.unsqueeze(0).to(device=device, dtype=torch.float16)

                # Add tokenized language conditioning
                task_str = sample.get("task", "Pick up the object.")
                if task_str not in task_tokens_cache:
                    task_tokens_cache[task_str] = _tokenize_task(task_str)
                for tk, tv in task_tokens_cache[task_str].items():
                    obs[tk] = tv

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    policy.select_action(obs)

                hidden = hook.features  # [1, T_tokens, 960]
                if hidden is None:
                    msg = "Hook did not capture features – check model architecture."
                    raise RuntimeError(msg)

                if config.pooling == "mean":
                    feat = hidden.float().mean(dim=1)  # [1, 960]
                else:
                    feat = hidden.float().mean(dim=1)

                feat_list.append(feat.cpu().squeeze(0))

                if "action" in sample:
                    act_list.append(sample["action"].cpu())
                if "observation.state" in sample:
                    state_list.append(sample["observation.state"].cpu())
                if "task_index" in sample:
                    task_idx_list.append(int(sample["task_index"]))

            features_t = torch.stack(feat_list)
            torch.save(features_t, ep_dir / "features.pt")
            if act_list:
                torch.save(torch.stack(act_list), ep_dir / "actions.pt")
            if state_list:
                torch.save(torch.stack(state_list), ep_dir / "states.pt")
            if task_idx_list:
                torch.save(torch.tensor(task_idx_list, dtype=torch.long), ep_dir / "task_indices.pt")

            total_frames += len(feat_list)
            print(f"  Episode {ep_id}: {len(feat_list)} frames")

    # Write metadata
    meta = {
        "model_id": config.smolvla_model_id,
        "dataset": config.dataset_name,
        "feature_dim": config.feature_dim,
        "pooling": config.pooling,
        "num_episodes": len(episode_ids),
        "total_frames": total_frames,
        "episode_ids": episode_ids,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved {total_frames} frames across {len(episode_ids)} episodes → {cache_dir}")

    # Free GPU memory
    del policy
    torch.cuda.empty_cache()

    return cache_dir


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VLAFeatureDataset(Dataset):
    """Loads cached SmolVLA features with single-frame or sequence modes.

    Args:
        cache_dir: Path to the feature cache (output of ``extract_smolvla_features``).
        sequence_length: 1 for Phase 1 (single frames), >1 for Phase 2/3 (rollout
            windows respecting episode boundaries).
    """

    def __init__(self, cache_dir: str | Path, sequence_length: int = 1) -> None:
        self.cache_dir = Path(cache_dir)
        self.sequence_length = sequence_length

        meta_path = self.cache_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.episode_ids: list[int] = meta["episode_ids"]
        else:
            # Discover episodes from directory structure
            self.episode_ids = sorted(
                int(p.name.split("_")[1])
                for p in self.cache_dir.iterdir()
                if p.is_dir() and p.name.startswith("episode_")
            )

        # Pre-load all episodes into memory for fast indexing
        self._features: list[torch.Tensor] = []  # list of [T_i, D]
        self._actions: list[torch.Tensor] = []  # list of [T_i, A]
        self._ep_ids: list[int] = []
        self._offsets: list[int] = []  # cumulative start index per episode

        offset = 0
        for ep_id in self.episode_ids:
            ep_dir = self.cache_dir / f"episode_{ep_id}"
            feat = torch.load(ep_dir / "features.pt", weights_only=True)
            act_path = ep_dir / "actions.pt"
            if act_path.exists():
                act = torch.load(act_path, weights_only=True)
            else:
                act = torch.zeros(feat.shape[0], 1)

            self._features.append(feat)
            self._actions.append(act)
            self._ep_ids.append(ep_id)
            self._offsets.append(offset)

            if sequence_length <= 1:
                offset += feat.shape[0]
            else:
                # Sliding windows that fit within the episode
                valid = max(0, feat.shape[0] - sequence_length + 1)
                offset += valid

        self._total = offset

    def __len__(self) -> int:
        return self._total

    def _find_episode(self, idx: int) -> tuple[int, int]:
        """Return (episode_index_in_list, local_offset) for a global idx."""
        for i in range(len(self._offsets) - 1, -1, -1):
            if idx >= self._offsets[i]:
                return i, idx - self._offsets[i]
        return 0, idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, local = self._find_episode(idx)
        features = self._features[ep_i]
        actions = self._actions[ep_i]
        ep_id = self._ep_ids[ep_i]

        if self.sequence_length <= 1:
            return {
                "feature": features[local],
                "action": actions[local],
                "episode_id": torch.tensor(ep_id, dtype=torch.long),
                "timestep": torch.tensor(local, dtype=torch.long),
            }

        # Sequence mode: return a window [local : local + H]
        end = local + self.sequence_length
        return {
            "features": features[local:end],
            "actions": actions[local:end],
            "episode_id": torch.tensor(ep_id, dtype=torch.long),
        }
