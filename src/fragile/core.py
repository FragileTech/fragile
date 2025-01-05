import torch

from fragile.benchmarks import Rastrigin
from fragile.fractalai import clone_tensor, fai_iteration


def compute(args):
    """Compute a placeholder for the compute function.

    Example:
        >>> compute(["1", "2", "3"])
        '1'

    """
    return max(args, key=len)


def sample_actions(x):
    """Sample actions from the environment."""
    return torch.randn_like(x)


def step(x, actions, benchmark):
    """Step the environment."""
    new_x = x + actions * 0.1
    rewards = benchmark(new_x)
    oobs = benchmark.bounds.contains(new_x)
    return new_x, rewards, oobs


def causal_cone(state, env, policy, n_steps, init_action):
    """Compute the causal cone of a state."""
    env.set_state(state)
    action = init_action
    for i in range(n_steps):
        data = env.step(state=state, action=action)
        state, observ, reward, end, _truncated, _info = data
        compas_ix, will_clone, *_rest_data = fai_iteration(observ, reward, end)
        observ = clone_tensor(observ, compas_ix, will_clone)
        state = clone_tensor(state, compas_ix, will_clone)
        action = policy(observ)


def run_swarm(n_walkers, benchmark, n_steps):
    x = benchmark.sample(n_walkers)
    x[:] = x[0, :]

    actions = sample_actions(x)
    x, rewards, oobs = step(x, actions, benchmark)

    for i in range(n_steps):
        print(rewards.numpy(force=True).min())
        compas_ix, will_clone, *_rest_data = fai_iteration(x, -rewards, oobs)
        x = clone_tensor(x, compas_ix, will_clone)
        # rewards = clone_tensor(rewards, compas_ix, will_clone)

        actions = sample_actions(x)
        x, rewards, oobs = step(x, actions, benchmark)


if __name__ == "__main__":
    benchmark = Rastrigin(5)
    run_swarm(500, benchmark=benchmark, n_steps=1000)
