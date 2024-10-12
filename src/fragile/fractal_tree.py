import sys

import torch

from fragile.benchmarks import Rastrigin
from fragile.fractalai import calculate_clone, calculate_virtual_reward, clone_tensor


def get_is_cloned(compas_ix, will_clone):
    target = torch.zeros_like(will_clone)
    cloned_to = compas_ix[will_clone].unique()
    target[cloned_to] = True
    return target


def get_is_leaf(parents):
    is_leaf = torch.ones_like(parents, dtype=torch.bool)
    is_leaf[parents] = False
    return is_leaf


def step(x, actions, benchmark):
    """Step the environment."""
    new_x = x + actions.to(x.device) * 0.1
    rewards = benchmark(new_x)
    oobs = benchmark.bounds.contains(new_x).to(x.device)
    return new_x, rewards, oobs


class FractalTree:
    def __init__(self, n_walkers, env, policy, n_steps, device="cuda"):
        self.n_walkers = n_walkers
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.device = device

        self.parents = torch.zeros(n_walkers, dtype=torch.long, device=self.device)
        self.can_clone = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_cloned = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_compa_distance = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_compa_clone = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_dead = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)

        self.observ = torch.zeros(n_walkers, device=self.device)
        self.reward = torch.zeros(n_walkers, device=self.device)
        self.oobs = torch.zeros(n_walkers, device=self.device)
        self.action = torch.zeros(n_walkers, device=self.device)

        self.virtual_reward = torch.zeros(n_walkers, device=self.device)
        self.clone_prob = torch.zeros(n_walkers, device=self.device)
        self.clone_ix = torch.zeros(n_walkers, device=self.device)
        self.distance_ix = torch.zeros(n_walkers, device=self.device)
        self.wants_clone = torch.zeros(n_walkers, device=self.device)
        self.will_clone = torch.zeros(n_walkers, device=self.device)
        self.distance = torch.zeros(n_walkers, device=self.device)
        self.scaled_distance = torch.zeros(n_walkers, device=self.device)
        self.scaled_reward = torch.zeros(n_walkers, device=self.device)

    def causal_cone(self, observ, action, n_steps=None):
        n_steps = n_steps if n_steps is not None else self.n_steps
        self.action = action
        self.observ, self.reward, self.oobs = step(observ, self.action, self.env)
        for i in range(n_steps):
            self.virtual_reward, self.distance_ix, self.distance = calculate_virtual_reward(
                self.observ, -1 * self.reward, self.oobs, return_distance=True, return_compas=True
            )
            leaf_ix = get_is_leaf(self.parents)

            self.clone_ix, self.wants_clone, self.clone_prob = calculate_clone(
                self.virtual_reward, self.oobs
            )
            self.is_cloned = get_is_cloned(self.clone_ix, self.wants_clone)
            self.will_clone = self.wants_clone & ~self.is_cloned & leaf_ix
            best = self.reward.argmin()
            self.will_clone[best] = False
            self.observ = clone_tensor(self.observ, self.clone_ix, self.will_clone)
            self.reward = clone_tensor(self.reward, self.clone_ix, self.will_clone)
            self.parents[self.will_clone] = self.clone_ix[self.will_clone]

            observ = self.observ[self.will_clone]
            action = self.policy(observ)
            observ, reward, oobs = step(observ, action, self.env)

            self.observ[self.will_clone] = observ
            self.reward[self.will_clone] = reward
            self.oobs[self.will_clone] = oobs
            if i % 1000 == 0:
                print(self.will_clone.sum().item(), self.reward.min().item())


def run():
    dim = 5
    benchmark = Rastrigin(dim)
    n_walkers = 5000
    device = "cuda"
    fai = FractalTree(n_walkers, env=benchmark, policy=torch.randn_like, n_steps=200000)
    state = benchmark.sample(n_walkers).to(device)
    state[:] = state[0, :]
    fai.causal_cone(state, torch.randn(n_walkers, dim, device=device))
    return 0


if __name__ == "__main__":
    sys.exit(run())
