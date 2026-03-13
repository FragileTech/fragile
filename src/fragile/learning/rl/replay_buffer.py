"""Simple sequence replay buffer for Dreamer-style training."""

from __future__ import annotations

import numpy as np
import torch


class SequenceReplayBuffer:
    """Stores full episodes and samples contiguous sub-sequences.

    Each episode is a dict of numpy arrays with a leading time axis.
    Sampling returns a batch of sub-sequences of length ``seq_len``.
    """

    def __init__(self, capacity: int, seq_len: int) -> None:
        self.capacity = capacity
        self.seq_len = seq_len
        self._episodes: list[dict[str, np.ndarray]] = []
        self._total_steps = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        """Add an episode dict with keys: obs, actions, rewards, dones."""
        ep_len = len(episode["obs"])
        self._episodes.append(episode)
        self._total_steps += ep_len
        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._total_steps -= len(removed["obs"])

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of sub-sequences.

        Returns dict with:
            obs: [B, T, obs_dim]
            actions: [B, T, action_dim]
            rewards: [B, T]
            dones: [B, T]
        """
        # Build index of valid (episode, start) pairs
        indices = []
        for ep_idx, ep in enumerate(self._episodes):
            ep_len = len(ep["obs"])
            if ep_len >= self.seq_len + 1:
                for t in range(ep_len - self.seq_len):
                    indices.append((ep_idx, t))
        if not indices:
            # Fall back: use whatever episodes we have, pad if needed
            for ep_idx, ep in enumerate(self._episodes):
                indices.append((ep_idx, 0))

        chosen = [indices[i] for i in np.random.randint(0, len(indices), size=batch_size)]

        obs_list, act_list, rew_list, done_list = [], [], [], []
        for ep_idx, start in chosen:
            ep = self._episodes[ep_idx]
            end = min(start + self.seq_len + 1, len(ep["obs"]))
            sl = slice(start, end)
            obs_list.append(ep["obs"][sl])
            act_list.append(ep["actions"][sl])
            rew_list.append(ep["rewards"][sl])
            done_list.append(ep["dones"][sl])

        def _pad_and_stack(arrays: list[np.ndarray]) -> torch.Tensor:
            max_len = max(a.shape[0] for a in arrays)
            padded = []
            for a in arrays:
                if a.shape[0] < max_len:
                    pad_shape = (max_len - a.shape[0], *a.shape[1:])
                    a = np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)])
                padded.append(a)
            return torch.from_numpy(np.stack(padded)).float().to(device)

        return {
            "obs": _pad_and_stack(obs_list),
            "actions": _pad_and_stack(act_list),
            "rewards": _pad_and_stack(rew_list),
            "dones": _pad_and_stack(done_list),
        }
