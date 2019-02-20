import gym
from garage.baselines import LinearFeatureBaseline
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.mujoco import SwimmerEnv
from garage.tf.algos.capg import CAPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment


env_name = "Swimmer"
hidden_sizes = (64, 64)
env = TfEnv(normalize(SwimmerEnv()))
policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=hidden_sizes)
backup_policy = GaussianMLPPolicy(
    name="backup_policy", env_spec=env.spec, hidden_sizes=hidden_sizes)
mix_policy = GaussianMLPPolicy(
    name="mix_policy", env_spec=env.spec, hidden_sizes=hidden_sizes)
pos_eps_policy = GaussianMLPPolicy(
    name="pos_eps_policy", env_spec=env.spec, hidden_sizes=hidden_sizes)
neg_eps_policy = GaussianMLPPolicy(
    name="neg_eps_policy", env_spec=env.spec, hidden_sizes=hidden_sizes)

baseline = ZeroBaseline(env_spec=env.spec)

algo = CAPG(
    env=env,
    policy=policy,
    backup_policy=backup_policy,
    mix_policy=mix_policy,
    pos_eps_policy=pos_eps_policy,
    neg_eps_policy=neg_eps_policy,
    n_timestep=1e6,
    learning_rate=0.01,
    batch_size=5000,
    minibatch_size=500,
    n_sub_itr = 10,
    baseline=baseline,
    max_path_length=500,
    discount=0.99,
    log_dir='./logs/' + env_name,
)
algo.train()

