from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.tf.algos.catrpo import CATRPO
import gym
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

env = TfEnv(normalize(CartpoleEnv()))
hidden_size = (8,)

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=hidden_size)
backup_policy = GaussianMLPPolicy(
    name="backup_policy", env_spec=env.spec, hidden_sizes=hidden_size)
mix_policy = GaussianMLPPolicy(
    name="mix_policy", env_spec=env.spec, hidden_sizes=hidden_size)
pos_eps_policy = GaussianMLPPolicy(
    name="pos_eps_policy", env_spec=env.spec, hidden_sizes=hidden_size)
neg_eps_policy = GaussianMLPPolicy(
    name="neg_eps_policy", env_spec=env.spec, hidden_sizes=hidden_size)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = CATRPO(
    env=env,
    policy=policy,
    backup_policy=backup_policy,
    mix_policy=mix_policy,
    pos_eps_policy=pos_eps_policy,
    neg_eps_policy=neg_eps_policy,
    n_itr=100,
    n_sub_itr=10,
    baseline=baseline,
    batch_size=5000,
    max_path_length=500,
    discount=0.99,
    step_size=0.01,
    delta=0.01,
    plot=False)
algo.train()
