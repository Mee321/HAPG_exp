import gym
import random
from garage.baselines import LinearFeatureBaseline
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.mujoco import HalfCheetahEnv
from garage.theano.algos.capg_re import CAPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment

def run_task(*_):
    env_name = "HumanoidStandup-v2"
    hidden_sizes = (100,50,25)
    env = TheanoEnv(normalize(gym.make(env_name)))
    print(env.spec.observation_space, env.spec.action_space)
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
        n_timestep=5e6,
        learning_rate=0.05,
        batch_size=5000,
        minibatch_size=500,
        n_sub_itr = 10,
        baseline=baseline,
        max_path_length=500,
        discount=0.99,
        decay_learing_rate=True,
        log_dir='./logs/' + env_name,
    )
    algo.train()

for _ in range(5):
    run_experiment(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=3,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random
        # seed will be used
        seed=random.randint(0,100000),
        # plot=True,
    )
