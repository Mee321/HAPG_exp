import gym
from garage.baselines import LinearFeatureBaseline
from garage.theano.baselines import GaussianMLPBaseline
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.theano.algos.capg import CAPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment
from garage.misc.ext import set_seed
import numpy as np

for learning_rate in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    for batch_size in [1000]:
        for _ in range(10):
            seed = np.random.randint(1,10000)
            env_name = "SGD_nn_CartPole"
            hidden_sizes = (8,)
            env = TheanoEnv(normalize(CartpoleEnv()))
            policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=hidden_sizes)
            backup_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
            mix_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
            pos_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
            neg_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)

            baseline = ZeroBaseline(env_spec=env.spec)

            algo = CAPG(
                env=env,
                policy=policy,
                backup_policy=backup_policy,
                mix_policy=mix_policy,
                pos_eps_policy=pos_eps_policy,
                neg_eps_policy=neg_eps_policy,
                n_timestep=5e5,
                learning_rate=learning_rate,
                batch_size= batch_size,
                minibatch_size=500,
                n_sub_itr = 0,
                center_adv=True,
                baseline=baseline,
                max_path_length=100,
                decay_learing_rate=True,
                discount=0.99,
                log_dir='./result/' + env_name + "seed" + str(seed) + '/',
            )
            algo.train()
