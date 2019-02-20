import gym
from garage.baselines import LinearFeatureBaseline
from garage.theano.baselines import GaussianMLPBaseline
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.theano.algos.capg_nn import CAPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment
from garage.misc.ext import set_seed
import numpy as np


for i in range(5):
    seed = np.random.randint(1,10000)
    env_name = "SGD_nn_Reacher"
    hidden_sizes = (32, 32)
    env = TheanoEnv(normalize(gym.make("Reacher-v2")))
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=hidden_sizes)
    backup_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
    mix_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
    pos_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
    neg_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = CAPG(
        env=env,
        policy=policy,
        backup_policy=backup_policy,
        mix_policy=mix_policy,
        pos_eps_policy=pos_eps_policy,
        neg_eps_policy=neg_eps_policy,
        n_timestep=1e6,
        learning_rate=0.002,
        batch_size= 5000,
        minibatch_size=500,
        n_sub_itr = 0,
        center_adv=True,
        baseline=baseline,
        max_path_length=500,
        discount=0.99,
        log_dir='./sgd_nn/' + env_name + "seed" + str(seed) + '/',
    )
    algo.train()
