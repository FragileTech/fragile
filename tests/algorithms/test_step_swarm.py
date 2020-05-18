from plangym import ClassicControl
import pytest

from fragile.core import DiscreteEnv, DiscreteUniform
from fragile.algorithms import FollowBestModel, StepToBest, StepSwarm
from fragile.distributed.env import ParallelEnv
from tests.core.test_swarm import TestSwarm


def get_cartpole_env():
    return ParallelEnv(lambda: DiscreteEnv(ClassicControl(name="CartPole-v0")))


def create_majority_step_swarm():
    swarm = StepSwarm(
        model=lambda x: DiscreteUniform(env=x),
        env=get_cartpole_env,
        reward_limit=10,
        n_walkers=10,  # 0,
        max_epochs=2,  # 0,
        step_epochs=2,  # 5,
    )
    return swarm


def create_follow_best_step_swarm():
    swarm = StepSwarm(
        root_model=FollowBestModel,
        model=lambda x: DiscreteUniform(env=x),
        env=get_cartpole_env,
        reward_limit=101,
        n_walkers=10,  # 0,
        max_epochs=2,  # 00,
        step_epochs=2,  # 5,
    )
    return swarm


def create_follow_best_step_swarm_after_impr():
    swarm = StepSwarm(
        root_model=FollowBestModel,
        model=lambda x: DiscreteUniform(env=x),
        env=get_cartpole_env,
        reward_limit=101,
        n_walkers=10,  # 0,
        max_epochs=2,  # 200,
        step_epochs=2,  # 5,
        step_after_improvement=True,
    )
    return swarm


def create_step_to_best():
    swarm = StepToBest(
        model=lambda x: DiscreteUniform(env=x),
        env=get_cartpole_env,
        reward_limit=51,
        n_walkers=10,  # 0,
        max_epochs=2,
        step_epochs=2,
    )
    return swarm


def create_step_to_best_after_impr():
    from plangym import AtariEnvironment
    from fragile.core import GaussianDt

    env = AtariEnvironment(name="MsPacman-ram-v0", clone_seeds=True, autoreset=True)
    dt = GaussianDt(min_dt=3, max_dt=100, loc_dt=5, scale_dt=2)
    # get_env = lambda: ParallelEnv(lambda: DiscreteEnv(env))
    swarm = StepToBest(
        model=lambda x: DiscreteUniform(env=x, critic=dt),
        env=lambda: DiscreteEnv(env),
        reward_limit=-100,
        n_walkers=67,
        max_epochs=2,
        step_epochs=5,
        step_after_improvement=True,
    )
    return swarm


swarm_dict = {
    "majority": create_majority_step_swarm,
    "follow_best": create_follow_best_step_swarm,
    "step_to_best": create_step_to_best,
    "follow_best_after_impr": create_follow_best_step_swarm_after_impr,
    "step_to_best_after_impr": create_step_to_best_after_impr,
}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "majority": -10,
    "follow_best": -100,
    "step_to_best": -50,
    "follow_best_after_impr": -100,
    "step_to_best_after_impr": -100,
}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    return swarm_dict.get(request.param)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    swarm = swarm_dict.get(request.param)()
    score = test_scores[request.param]
    return swarm, score
