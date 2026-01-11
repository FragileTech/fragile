import torch

from fragile.fractalai import calculate_clone, calculate_fitness, clone_tensor


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
    return new_x, rewards, ~oobs


class FractalTree:
    def __init__(self, n_walkers, env, policy, device="cuda"):
        self.n_walkers = n_walkers
        self.env = env
        self.policy = policy
        self.device = device

        self.total_steps = 0
        self.iteration = 0

        self.parent = torch.zeros(n_walkers, dtype=torch.long, device=self.device)
        self.is_leaf = torch.ones(n_walkers, dtype=torch.long, device=self.device)
        self.can_clone = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_cloned = torch.zeros(n_walkers, dtype=torch.bool, device=self.device)
        self.is_compa_distance = torch.ones(n_walkers, dtype=torch.bool, device=self.device)
        self.is_compa_clone = torch.ones(n_walkers, dtype=torch.bool, device=self.device)
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
            "best_reward": self.reward.min().cpu().item(),
            "best_ix": self.reward.argmin().cpu().item(),
            "will_clone": self.will_clone.sum().cpu().item(),
            "total_steps": self.total_steps,
        }

    def step_tree(self):
        self.virtual_reward, self.distance_ix, self.distance = calculate_fitness(
            self.observ, -1 * self.reward, self.oobs, return_distance=True, return_compas=True
        )
        self.is_leaf = get_is_leaf(self.parent)

        self.clone_ix, self.wants_clone, self.clone_prob = calculate_clone(
            self.virtual_reward, self.oobs
        )
        self.is_cloned = get_is_cloned(self.clone_ix, self.wants_clone)
        self.wants_clone[self.oobs] = True
        self.will_clone = self.wants_clone & ~self.is_cloned & self.is_leaf

        best = self.reward.argmin()
        self.will_clone[best] = False
        self.observ = clone_tensor(self.observ, self.clone_ix, self.will_clone)
        self.reward = clone_tensor(self.reward, self.clone_ix, self.will_clone)
        self.parent[self.will_clone] = self.clone_ix[self.will_clone]

        observ = self.observ[self.will_clone]
        action = self.policy(observ)
        observ, reward, oobs = step(observ, action, self.env)

        self.observ[self.will_clone] = observ
        self.reward[self.will_clone] = reward
        self.oobs[self.will_clone] = oobs
        self.total_steps += self.will_clone.sum().cpu().item()
        self.iteration += 1

    def reset(self, observ, action):
        self.action = action
        self.total_steps = 0
        self.iteration = 0
        self.observ, self.reward, self.oobs = step(observ, self.action, self.env)
        self.observ[0, :] = observ[0, :]
        self.parent = torch.zeros(self.n_walkers, dtype=torch.long, device=self.device)
        self.is_leaf = get_is_leaf(self.parent)

    def causal_cone(self, observ, action, n_steps):
        self.reset(observ, action)
        for i in range(n_steps):
            self.step_tree()
            if i % 1000 == 0:
                print(self.will_clone.sum().item(), self.reward.min().item())
