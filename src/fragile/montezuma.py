import numpy as np
import torch

from fragile.fractalai import asymmetric_rescale, calculate_clone, calculate_fitness, clone_tensor


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
    dt = np.random.randint(1, 4, size=state.shape[0])
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
        rgb_shape: tuple[int, ...] = (210, 160, 3),
        erase_coef=0.05,
        agg_block_size: int = 5,
    ):
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            policy=policy,
            device=device,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
            erase_coef=erase_coef,
        )
        self.agg_block_size = agg_block_size
        self.rgb_shape = rgb_shape
        self.obs_shape = self.env.observation_space.shape

        self.observ = torch.zeros(
            (self.start_walkers, *self.obs_shape), device=self.device, dtype=torch.float32
        )
        self.action = torch.zeros(self.start_walkers, device=self.device, dtype=torch.int64)
        self.state = np.empty(self.start_walkers, dtype=object)
        self.rgb = np.zeros((self.start_walkers, *self.rgb_shape), dtype=np.uint8)
        self.visits = np.zeros((24, 160, 160), dtype=np.int64)

    def add_walkers(self, new_walkers):
        super().add_walkers(new_walkers)
        observ = torch.zeros(
            (new_walkers, *self.obs_shape), device=self.device, dtype=torch.float32
        )
        self.observ = torch.cat((self.observ, observ), dim=0).contiguous()

        action = torch.zeros(new_walkers, device=self.device, dtype=torch.int64)
        self.action = torch.cat((self.action, action), dim=0).contiguous()

        state = self.state[:new_walkers].copy()
        self.state = np.concatenate((self.state, state), axis=0)
        rgb = np.zeros((new_walkers, *self.rgb_shape), dtype=np.uint8)
        self.rgb = np.concatenate((self.rgb, rgb), axis=0)

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
            self.rgb = clone_tensor(self.rgb, self.clone_ix, self.will_clone)
            self.parent[self.will_clone] = self.clone_ix[self.will_clone]
        except Exception as e:
            print("CACA", self.observ.shape, self.will_clone.shape, self.clone_ix.shape)
            raise e

    def step_walkers(self):
        wc_np = self.will_clone.cpu().numpy()
        states = self.state[wc_np]
        if len(states) == 0:
            return
        actions = self.sample_actions(int(wc_np.sum()))
        dt = np.random.randint(1, 5, size=len(actions))
        try:
            data = self.env.step_batch(states=states, actions=actions, dt=dt)
        except Exception as e:
            print("CACA", states.shape, actions, dt.shape, wc_np.sum())
            raise e
        new_states, observ, reward, oobs, _truncateds, infos = data
        self.update_visits(np.array(observ, dtype=np.int64))
        self.observ[self.will_clone] = torch.tensor(
            observ, dtype=torch.float32, device=self.device
        )
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.reward[self.will_clone] = reward_tensor
        self.cum_reward[self.will_clone] += reward_tensor
        self.oobs[self.will_clone] = torch.tensor(oobs, dtype=torch.bool, device=self.device)
        self.state[wc_np] = new_states
        self.rgb[wc_np] = np.array([info["rgb"] for info in infos])
        self.action[self.will_clone] = torch.tensor(actions, dtype=torch.int64, device=self.device)

    def step_tree(self):
        visits_reward = self.calculate_visits_reward()
        self.virtual_reward, self.distance_ix, self.distance = calculate_fitness(
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

        self.total_steps += self.will_clone.sum().cpu().item()
        self.iteration += 1
        leafs = self.is_leaf.sum().cpu().item()
        new_walkers = self.min_leafs - leafs
        if new_walkers > 0:
            self.add_walkers(new_walkers)

    def sample_actions(self, max_walkers: int):
        return [self.env.sample_action() for _ in range(max_walkers)]

    def reset(self):
        obs_shape = (self.start_walkers, *self.env.observation_space.shape)
        self.observ = torch.zeros(obs_shape, device=self.device, dtype=torch.float32)
        self.reward = torch.zeros(self.start_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.zeros(self.start_walkers, device=self.device, dtype=torch.float32)
        self.oobs = torch.zeros(self.start_walkers, device=self.device, dtype=torch.bool)
        self.parent = torch.zeros(self.start_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = get_is_leaf(self.parent)

        self.action = torch.tensor(
            self.sample_actions(self.start_walkers), device=self.device, dtype=torch.int64
        )
        self.rgb = np.zeros((self.start_walkers, 210, 160, 3), dtype=np.uint8)
        self.visits = np.zeros((24, 160, 160), dtype=np.float32)
        state, obs, _info = self.env.reset()
        self.rgb[0, :] = self.env.get_image()
        self.state = np.array([state.copy() for _ in range(self.start_walkers)])
        self.total_steps = 0
        self.iteration = 0
        data = self.env.step_batch(states=self.state, actions=self.action.numpy(force=True))
        new_states, observ, reward, oobs, _truncateds, infos = data
        self.update_visits(np.array(observ, dtype=np.int64))
        self.observ[0, :] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.observ[1:, :] = torch.tensor(
            np.vstack(observ[1:]), dtype=torch.float32, device=self.device
        )
        self.reward[1:] = torch.tensor(reward[1:], dtype=torch.float32, device=self.device)
        self.cum_reward[1:] = torch.tensor(reward[1:], dtype=torch.float32, device=self.device)
        self.oobs[1 : self.n_walkers] = torch.tensor(
            oobs[1 : self.n_walkers], dtype=torch.bool, device=self.device
        )
        self.state[1:] = new_states[1:]
        self.rgb[1:, :] = np.array([info["rgb"] for info in infos[1:]])

    def update_visits(self, observ):
        observ = observ.astype(np.float64)
        observ[:, 0] /= int(self.env.gym_env._x_repeat)
        observ = observ.astype(np.int64)

        self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] = np.where(
            np.isnan(self.visits[observ[:, 2], observ[:, 1], observ[:, 0]]),
            1,
            self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] + 1,
        )
        self.visits = np.clip(self.visits - self.erase_coef, 0, 1000)

    def calculate_visits_reward(self):
        visits = aggregate_visits(self.visits, block_size=self.agg_block_size, upsample=True)
        obs = self.observ.numpy(force=True).astype(np.int64)
        x, y, room_ix = obs[:, 0], obs[:, 1], obs[:, 2]
        visits_val = torch.tensor(visits[room_ix, y, x], device=self.device, dtype=torch.float32)
        return asymmetric_rescale(-visits_val)
