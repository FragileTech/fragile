import einops
import numpy as np
import torch

from fragile.benchmarks import Rastrigin
from fragile.fractalai import (
    calculate_clone,
    calculate_virtual_reward,
    clone_tensor,
    fai_iteration,
)
from fragile.utils import numpy_dtype_to_torch_dtype


def get_is_cloned(compas_ix, will_clone):
    target = torch.zeros_like(will_clone)
    cloned_to = compas_ix[will_clone].unique()
    target[cloned_to] = True
    return target


def get_is_leaf(parents):
    is_leaf = torch.ones_like(parents, dtype=torch.bool)
    is_leaf[parents] = False
    return is_leaf


def step(state, action, env):
    """Step the environment."""
    dt = np.random.randint(1, 4, size=state.shape[0])  # noqa: NPY002
    data = env.step(state=state, action=action, dts=dt)
    new_state, observ, reward, end, _truncated, info = data
    return new_state, observ, reward, end, info


def aggregate_visits(array, block_size=5, upsample=True):
    """
    Aggregates the input array over blocks in the last two dimensions.

    Parameters:
    - array (numpy.ndarray): Input array with shape (batch_size, width, height).
    - block_size (int): Size of the block over which to aggregate.

    Returns:
    - numpy.ndarray: Aggregated array with reduced dimensions.
    """
    batch_size, width, height = array.shape
    new_width = width // block_size
    new_height = height // block_size

    # Ensure that width and height are divisible by block_size
    if width % block_size != 0 or height % block_size != 0:
        msg = (
            "Width and height must be divisible by block_size. "
            "Got width: {}, height: {}, block_size: {}"
        )
        raise ValueError(msg.format(width, height, block_size))

    reshaped_array = array.reshape(batch_size, new_width, block_size, new_height, block_size)
    aggregated_array = reshaped_array.sum(axis=(2, 4))
    if not upsample:
        return aggregated_array
    return np.repeat(np.repeat(aggregated_array, block_size, axis=1), block_size, axis=2)


class BaseFractalTree:
    def __init__(
        self,
        max_walkers,
        env,
        policy=None,
        device="cuda",
        start_walkers: int = 100,
        min_leafs: int = 100,
        erase_coef: float = 0.05,
    ):
        self.max_walkers = max_walkers
        self.n_walkers = start_walkers
        self.start_walkers = start_walkers
        self.min_leafs = min_leafs
        self.env = env
        self.policy = policy
        self.device = device

        self.total_steps = 0
        self.iteration = 0
        self.erase_coef = erase_coef

        self.parent = torch.zeros(start_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = torch.ones(start_walkers, device=self.device, dtype=torch.long)
        self.can_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_cloned = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_distance = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_clone = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_dead = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.oobs = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.virtual_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_prob = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.distance_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.wants_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.will_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)

    def add_walkers(self, new_walkers):
        self.n_walkers += new_walkers

        parent = torch.zeros(new_walkers, device=self.device, dtype=torch.long)
        self.parent = torch.cat((self.parent, parent), dim=0).contiguous()

        is_leaf = torch.ones(new_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = torch.cat((self.is_leaf, is_leaf), dim=0).contiguous()

        can_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.can_clone = torch.cat((self.can_clone, can_clone), dim=0).contiguous()

        is_cloned = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_cloned = torch.cat((self.is_cloned, is_cloned), dim=0).contiguous()

        is_compa_distance = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_distance = torch.cat(
            (self.is_compa_distance, is_compa_distance), dim=0
        ).contiguous()

        is_compa_clone = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_clone = torch.cat((self.is_compa_clone, is_compa_clone), dim=0).contiguous()

        is_dead = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.is_dead = torch.cat((self.is_dead, is_dead), dim=0).contiguous()

        reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.reward = torch.cat((self.reward, reward), dim=0).contiguous()

        cum_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.cat((self.cum_reward, cum_reward), dim=0).contiguous()

        oobs = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.oobs = torch.cat((self.oobs, oobs), dim=0).contiguous()

        virtual_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.virtual_reward = torch.cat((self.virtual_reward, virtual_reward), dim=0).contiguous()

        clone_prob = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.clone_prob = torch.cat((self.clone_prob, clone_prob), dim=0).contiguous()

        clone_ix = torch.zeros(new_walkers, device=self.device, dtype=torch.int64)
        self.clone_ix = torch.cat((self.clone_ix, clone_ix), dim=0).contiguous()

        distance_ix = torch.zeros(new_walkers, device=self.device, dtype=torch.int64)
        self.distance_ix = torch.cat((self.distance_ix, distance_ix), dim=0).contiguous()

        wants_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.wants_clone = torch.cat((self.wants_clone, wants_clone), dim=0).contiguous()

        will_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.will_clone = torch.cat((self.will_clone, will_clone), dim=0).contiguous()

        distance = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.distance = torch.cat((self.distance, distance), dim=0).contiguous()

        scaled_distance = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.scaled_distance = torch.cat(
            (self.scaled_distance, scaled_distance), dim=0
        ).contiguous()

        scaled_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.scaled_reward = torch.cat((self.scaled_reward, scaled_reward), dim=0).contiguous()


class FractalTree(BaseFractalTree):
    def __init__(
        self,
        max_walkers,
        env,
        policy=None,
        device="cuda",
        start_walkers=100,
        min_leafs=100,
        obs_shape: tuple[int, ...] | None = None,
        obs_dtype: torch.dtype | None = None,
        action_shape: tuple[int, ...] | None = None,
        action_dtype: torch.dtype | None = None,
        state_shape: tuple[int, ...] | None = (),
        state_dtype: torch.dtype | type = object,
    ):
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            policy=policy,
            device=device,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
        )

        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.observ = self.reset_observ()

        self.action_shape = action_shape
        self.action_type = action_dtype
        self.action = self.reset_action

        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.state = self.reset_state()

        self.state_step = None
        self.action_step = None
        self.dt = None

    def reset_observ(self):
        self.obs_shape = (
            self.obs_shape if self.obs_shape is not None else self.env.observation_space.shape
        )
        self.obs_dtype = (
            self.obs_dtype
            if self.obs_dtype is not None
            else numpy_dtype_to_torch_dtype(self.env.observation_space.dtype)
        )

        return torch.zeros(
            (self.start_walkers, *self.obs_shape), device=self.device, dtype=self.obs_dtype
        )

    def reset_action(self):
        self.action_shape = (
            self.action_shape if self.action_shape is not None else self.env.action_space.shape
        )
        self.action_type = (
            self.action_type
            if self.action_type is not None
            else numpy_dtype_to_torch_dtype(self.env.action_space.dtype)
        )
        return torch.zeros(
            (self.start_walkers, *self.action_shape), device=self.device, dtype=self.action_type
        )

    def reset_state(self):
        return np.empty((self.start_walkers, *self.state_shape), dtype=self.state_dtype)

    def add_walkers(self, new_walkers):
        super().add_walkers(new_walkers)
        observ = self.observ[:new_walkers].detach().clone()
        self.observ = torch.cat((self.observ, observ), dim=0).contiguous()

        action = self.action[:new_walkers].detach().clone()
        self.action = torch.cat((self.action, action), dim=0).contiguous()

        state = self.state[:new_walkers].copy()
        self.state = np.concatenate((self.state, state), axis=0)  # type: ignore

    def to_dict(self):
        observ = self.observ.cpu().numpy()
        return {
            "parent": self.parent.cpu().numpy(),
            "can_clone": self.can_clone.cpu().numpy(),
            "is_cloned": self.is_cloned.cpu().numpy(),
            "is_leaf": self.is_leaf.cpu().numpy(),
            "is_compa_distance": self.is_compa_distance.cpu().numpy(),
            "is_compa_clone": self.is_compa_clone.cpu().numpy(),
            "is_dead": self.is_dead.cpu().numpy(),
            "x": observ[:, 0],
            "y": observ[:, 1],
            "reward": self.reward.cpu().numpy(),
            "oobs": self.oobs.to(torch.float32).cpu().numpy(),
            "virtual_reward": self.virtual_reward.cpu().numpy(),
            "clone_prob": self.clone_prob.cpu().numpy(),
            "clone_ix": self.clone_ix.cpu().numpy(),
            "distance_ix": self.distance_ix.cpu().numpy(),
            "wants_clone": self.wants_clone.cpu().numpy(),
            "will_clone": self.will_clone.cpu().numpy(),
            "distance": self.distance.cpu().numpy(),
            "scaled_distance": self.scaled_distance.cpu().numpy(),
            "scaled_reward": self.scaled_reward.cpu().numpy(),
        }

    def summary(self):
        return {
            "iteration": self.iteration,
            "leaf_nodes": self.is_leaf.sum().cpu().item(),
            "oobs": self.oobs.sum().cpu().item(),
            "best_reward": self.cum_reward.max().cpu().item(),
            "best_ix": self.cum_reward.argmax().cpu().item(),
            "will_clone": self.will_clone.sum().cpu().item(),
            "total_steps": self.total_steps,
            "n_walkers": self.n_walkers,
        }

    def clone_data(self):
        best = self.cum_reward.argmax()
        self.will_clone[best] = False
        try:
            self.observ = clone_tensor(self.observ, self.clone_ix, self.will_clone)
            self.reward = clone_tensor(self.reward, self.clone_ix, self.will_clone)
            self.cum_reward = clone_tensor(self.cum_reward, self.clone_ix, self.will_clone)
            self.state = clone_tensor(self.state, self.clone_ix, self.will_clone)
            # This line is what updates the parents to create the tree structure
            # comment this and you get the swarm wave
            self.parent[self.will_clone] = self.clone_ix[self.will_clone]
        except Exception as e:
            print("CACA", self.observ.shape, self.will_clone.shape, self.clone_ix.shape)
            raise e

    def sample_dt(self, n_walkers: int | None = None):
        if n_walkers is None:
            n_walkers = self.n_walkers
        return np.ones(n_walkers)

    def sample_actions(self, max_walkers: int | None = None):
        if max_walkers is None:
            max_walkers = self.n_walkers
        acts = [self.env.sample_action() for _ in range(max_walkers)]
        return torch.tensor(acts, device=self.device, dtype=self.action_type)

    def step_env(self):
        self.dt = self.sample_dt(self.action_step.shape[0])
        action = einops.asnumpy(self.action_step)
        data = self.env.step_batch(states=self.state_step, actions=action, dt=self.dt)
        new_states, observ, reward, oobs, truncateds, infos = data
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        obs_tensor = torch.tensor(observ, dtype=torch.float32, device=self.device)
        oobs_tensor = torch.tensor(oobs, dtype=torch.bool, device=self.device)
        return new_states, obs_tensor, reward_tensor, oobs_tensor, truncateds, infos

    def step_walkers(self):
        wc_np = self.will_clone.cpu().numpy()
        self.state_step = self.state[wc_np]
        if len(self.state_step) == 0:
            return None, None, None, None, None, None
        self.action_step = self.sample_actions(int(wc_np.sum()))
        try:
            new_states, observ, reward, oobs, _truncateds, _infos = self.step_env()
        except Exception as e:
            print("CACA", self.state_step.shape, self.action_step, self.dt.shape, wc_np.sum())
            raise e
        self.observ[self.will_clone] = observ
        self.reward[self.will_clone] = reward
        self.cum_reward[self.will_clone] += reward
        self.oobs[self.will_clone] = oobs
        self.state[wc_np] = new_states
        self.action[self.will_clone] = self.action_step
        return new_states, observ, reward, oobs, _truncateds, _infos

    def calculate_other_reward(self):
        return 1.0

    def step_tree(self):
        visits_reward = self.calculate_other_reward()
        self.virtual_reward, self.distance_ix, self.distance = calculate_virtual_reward(
            self.observ,
            self.cum_reward,
            self.oobs,
            return_distance=True,
            return_compas=True,
            other_reward=visits_reward,
        )
        self.is_leaf = get_is_leaf(self.parent)

        self.clone_ix, self.wants_clone, self.clone_prob = calculate_clone(
            self.virtual_reward, self.oobs
        )
        self.is_cloned = get_is_cloned(self.clone_ix, self.wants_clone)
        self.wants_clone[self.oobs] = True
        self.will_clone = self.wants_clone & ~self.is_cloned & self.is_leaf
        if self.will_clone.sum().cpu().item() == 0:
            self.iteration += 1
            return

        self.clone_data()

        self.step_walkers()

        leafs = self.is_leaf.sum().cpu().item()
        new_walkers = self.min_leafs - leafs
        if new_walkers > 0:
            self.add_walkers(new_walkers)

        self.total_steps += self.will_clone.sum().cpu().item()
        self.iteration += 1

    def reset(self):
        self.total_steps = 0
        self.iteration = 0

        self.observ = self.reset_observ()
        self.reward = torch.zeros(self.start_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.zeros(self.start_walkers, device=self.device, dtype=torch.float32)
        self.oobs = torch.zeros(self.start_walkers, device=self.device, dtype=torch.bool)
        self.parent = torch.zeros(self.start_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = get_is_leaf(self.parent)

        self.action = self.sample_actions(self.start_walkers)

        state, _obs, _info = self.env.reset()
        self.state = np.array([state.copy() for _ in range(self.start_walkers)])

        self.state_step = self.state
        self.action_step = self.action

        new_states, observ, reward, oobs, _truncateds, _infos = self.step_env()
        # here we set the firs state to the one after reset. This state will act as the root
        # of the tree
        self.observ[0, :] = observ[0, :]
        # We fill up the remaining states with the ones returned  after stepping the environment.
        # All these states have the root as their parent.
        self.observ[1:, :] = observ[1:, :]
        self.reward[1:] = reward[1:]
        self.cum_reward[1:] = reward[1:]
        self.oobs[1 : self.n_walkers] = oobs[1 : self.n_walkers]
        self.state[1:] = new_states[1:]
        return (state, _obs, _info), new_states, observ, reward, oobs, _truncateds, _infos


def sample_actions(x):
    """Sample actions from the environment."""
    return torch.randn_like(x)


def _step(x, actions, benchmark):
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
    x, rewards, oobs = _step(x, actions, benchmark)

    for i in range(n_steps):
        print(rewards.numpy(force=True).min())
        compas_ix, will_clone, *_rest_data = fai_iteration(x, -rewards, oobs)
        x = clone_tensor(x, compas_ix, will_clone)
        # rewards = clone_tensor(rewards, compas_ix, will_clone)

        actions = sample_actions(x)
        x, rewards, oobs = _step(x, actions, benchmark)


if __name__ == "__main__":
    benchmark = Rastrigin(5)
    run_swarm(500, benchmark=benchmark, n_steps=1000)
