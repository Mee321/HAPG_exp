import numpy as np
import theano
import tensorflow as tf
import copy
from tensorboardX import SummaryWriter
import time
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OnPolicyVectorizedSampler
from garage.tf.misc import tensor_utils
from garage.misc import ext
from garage.misc import special
timestamp = time.time()
timestruct = time.localtime(timestamp)
exp_time = time.strftime('%Y-%m-%d %H:%M:%S', timestruct)


class CAPG(object):

    def __init__(self,
                 env,
                 policy,
                 backup_policy,
                 mix_policy,
                 pos_eps_policy,
                 neg_eps_policy,
                 baseline,
                 n_timestep=1e6,
                 learning_rate=0.01,
                 batch_size=50000,
                 minibatch_size = 5000,
                 n_sub_itr=10,
                 max_path_length=500,
                 discount=0.99,
                 sampler_cls=None,
                 sampler_args=None,
                 scope=None,
                 whole_paths=True,
                 gae_lamda=1,
                 decay_learing_rate=True,
                 center_adv=True,
                 positive_adv=False,
                 force_batch_sampler=False,
                 log_dir=None,):
        self.env = env
        self.policy = policy
        self.backup_policy = backup_policy
        self.mix_policy = mix_policy
        self.pos_eps_policy = pos_eps_policy
        self.neg_eps_policy = neg_eps_policy
        self.baseline = baseline
        self.n_timestep = n_timestep
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.n_sub_itr = n_sub_itr
        self.max_path_length = max_path_length
        self.discount = discount
        self.scope=scope
        self.whole_paths = whole_paths
        self.gae_lambda=gae_lamda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.decay_learning_rate = decay_learing_rate
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = OnPolicyVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        log_dir = log_dir + '/capg%s_batchsize_%d_minibatchsize_%d_subitr_%d_lr_%f' % (exp_time, batch_size, minibatch_size, self.n_sub_itr, learning_rate)
        self.writer = SummaryWriter(log_dir)

    def start_worker(self, sess):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def sample_paths(self, traj_num, sample_policy):
        paths = []
        # Sample Trajectories
        for _ in range(traj_num):
            observations = []
            actions = []
            rewards = []

            observation = self.env.reset()

            for _ in range(self.max_path_length):
                # policy.get_action() returns a pair of values. The second
                # one returns a dictionary, whose values contains
                # sufficient statistics for the action distribution. It
                # should at least contain entries that would be returned
                # by calling policy.dist_info(), which is the non-symbolic
                # analog of policy.dist_info_sym(). Storing these
                # statistics is useful, e.g., when forming importance
                # sampling ratios. In our case it is not needed.
                action, _ = sample_policy.get_action(observation)
                # Recall that the last entry of the tuple stores diagnostic
                # information about the environment. In our case it is not needed.
                next_observation, reward, terminal, _ = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observation = next_observation
                if terminal:
                    # Finish rollout if terminal state reached
                    break

            # We need to compute the empirical return for each time step along the
            # trajectory
            path = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
            )
            path_baseline = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baseline[1:] - \
                     path_baseline[:-1]
            advantages = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda
            )
            path["returns"] = special.discount_cumsum(
                path["rewards"], self.discount
            )
            if self.center_adv:
                advantages = (advantages - np.mean(advantages)) / (
                    np.std(advantages) + 1e-8)

            path["advantages"] = advantages

            paths.append(path)
        return paths
    @staticmethod
    def grad_norm(s_g):
        res = s_g[0].flatten()
        for i in range(1,len(s_g)):
            res = np.concatenate((res, s_g[i].flatten()))
        l2_norm = np.linalg.norm(res)
        return l2_norm
    @staticmethod
    def normalize_gradient(s_g):
        res = s_g[0].flatten()
        for i in range(1, len(s_g)):
            res = np.concatenate((res, s_g[i].flatten()))
        l2_norm = np.linalg.norm(res)
        return [x/l2_norm for x in s_g]
    @staticmethod
    def flatten_parameters(params):
        return np.concatenate([p.flatten() for p in params])

    def generate_mix_policy(self):
        a = np.random.uniform(0.0, 1.0)
        mix = a * self.policy.get_param_values() + (1 - a) * self.backup_policy.get_param_values()
        self.mix_policy.set_param_values(mix, trainable=True)

    def sgd_update(self, gradient, learning_rate):
        previous_params = self.policy.get_param_values()
        gradient = self.flatten_parameters(gradient)
        updated_params = previous_params + learning_rate * gradient
        self.policy.set_param_values(updated_params, trainable=True)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        self.init_opt()
        j = 0
        while(self.batch_size < self.n_timestep - j):
            paths = self.sampler.obtain_samples(j)
            sample_data = self.sampler.process_samples(j, paths)
            j += sum([len(path["rewards"]) for path in paths])
            avg_returns = np.mean([sum(p["rewards"]) for p in paths])
            print("timesteps: " + str(j) + " average return: " + str(avg_returns))
            self.writer.add_scalar("AverageReturn", avg_returns, j)
            self.writer.add_scalar("Average_outerloop_return", avg_returns, j)
            self.outer_optimize(j, sample_data)
            for _ in range(self.n_sub_itr):
                n_sub = 0
                sub_paths_all = []
                self.generate_mix_policy()
                sub_paths = self.sample_paths(1, self.mix_policy)
                sub_paths_all.append(sub_paths[0])
                n_sub += len(sub_paths[0]["rewards"])
                j += len(sub_paths[0]["rewards"])
                sub_observations = [p["observations"] for p in sub_paths]
                sub_actions = [p["actions"] for p in sub_paths]
                sub_advantages = [p["advantages"] for p in sub_paths]
                eps = 1e-6
                d_vector = self.policy.get_param_values() - self.backup_policy.get_param_values()
                pos_params = self.mix_policy.get_param_values() + d_vector * eps
                neg_params = self.mix_policy.get_param_values() - d_vector * eps
                self.pos_eps_policy.set_param_values(pos_params, trainable=True)
                self.neg_eps_policy.set_param_values(neg_params, trainable=True)

                # first component: dot(gradient of loglikelihood, theta_t - theta_t-1) * policy gradient
                g_mix = self._opt_fun["f_mix_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                g_lh = self._opt_fun["f_mix_lh"](sub_observations[0], sub_actions[0])
                g_lh = self.flatten_parameters(g_lh)
                self.writer.add_scalar("gradient of loglikelihood", self.grad_norm(g_lh), j)
                inner_product = np.dot(g_lh, d_vector)
                self.writer.add_scalar("g_mix norm", self.grad_norm(g_mix), j)
                self.writer.add_scalar("loglikelihood times d_vector", inner_product, j)
                fst = [inner_product * g for g in g_mix]

                # second component: dot(Hessian, theta_t - theta_t-1)
                g_pos = self._opt_fun["f_pos_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                g_neg = self._opt_fun["f_neg_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                hv = [(pos - neg) / (2 * eps) for pos, neg in zip(g_pos, g_neg)]

                while(n_sub < self.minibatch_size):
                    self.generate_mix_policy()
                    sub_paths = self.sample_paths(1, self.mix_policy)
                    n_sub += len(sub_paths[0]["rewards"])
                    j += len(sub_paths[0]["rewards"])
                    sub_paths_all.append(sub_paths[0])
                    sub_observations = [p["observations"] for p in sub_paths]
                    sub_actions = [p["actions"] for p in sub_paths]
                    sub_advantages = [p["advantages"] for p in sub_paths]

                    pos_params = self.mix_policy.get_param_values() + d_vector * eps
                    neg_params = self.mix_policy.get_param_values() - d_vector * eps
                    self.pos_eps_policy.set_param_values(pos_params, trainable=True)
                    self.neg_eps_policy.set_param_values(neg_params, trainable=True)

                    # first component: dot(likelihood, theta_t - theta_t-1) * policy gradient
                    g_mix = self._opt_fun["f_mix_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                    g_lh = self._opt_fun["f_mix_lh"](sub_observations[0], sub_actions[0])
                    g_lh = self.flatten_parameters(g_lh)
                    inner_product = np.dot(g_lh, d_vector)
                    fst_i = [inner_product * g for g in g_mix]
                    fst = [sum(x) for x in zip(fst, fst_i)]

                    # second component: dot(Hessian, theta_t - theta_t-1)
                    g_pos = self._opt_fun["f_pos_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                    g_neg = self._opt_fun["f_neg_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
                    hv_i = [(pos - neg) / (2 * eps) for pos, neg in zip(g_pos, g_neg)]
                    hv = [sum(x) for x in zip(hv, hv_i)]
                fst = [x / len(sub_paths_all) for x in fst]
                hv = [x / len(sub_paths_all) for x in hv]
                #fst = [x/10 for x in fst]
                # gradient as sum
                fst_norm = self.grad_norm(fst)
                hv_norm = self.grad_norm(hv)
                backup_gradient_norm = self.grad_norm(self.gradient_backup)
                self.writer.add_scalar("first_component_norm", fst_norm, j)
                self.writer.add_scalar("hv_norm", hv_norm, j)
                self.writer.add_scalar("back_gradient_norm", backup_gradient_norm, j)

                g_d = [sum(x) for x in zip(fst, hv, self.gradient_backup)]
                self.gradient_backup = copy.deepcopy(g_d)
                avg_returns = np.mean([sum(p["rewards"]) for p in sub_paths_all])
                self.writer.add_scalar("AverageReturn", avg_returns, j)
                self.writer.add_scalar("Gradient norm", self.grad_norm(g_d), j)
                print("timesteps: " + str(j) + " average return: " + str(avg_returns))

                g_d = self.normalize_gradient(g_d)
                self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
                self.sgd_update(g_d, self.learning_rate)

                self.baseline.fit(sub_paths_all)

                #Compute KL divergence after updated
                sub_observations = [p["observations"] for p in sub_paths]
                mean_kl, max_kl = self._opt_fun["f_kl"](sub_observations[0])
                self.writer.add_scalar("MeanKL", mean_kl, j)
                self.writer.add_scalar("MaxKL", max_kl, j)

        self.shutdown_worker(sess)

    def log_diagnostics(self, paths):
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        with tf.name_scope("inputs"):
            observations_var = self.env.observation_space.new_tensor_variable(
                'observations',
                extra_dims=1
            )
            actions_var = self.env.action_space.new_tensor_variable(
                'actions',
                extra_dims=1
            )
            advantages_var = tensor_utils.new_tensor(
                'advantage', ndim=1, dtype=tf.float32)
            dist = self.policy.distribution
            dist_info_vars = self.policy.dist_info_sym(observations_var)
            old_dist_info_vars = self.backup_policy.dist_info_sym(observations_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        pos_eps_dist_info_vars = self.pos_eps_policy.dist_info_sym(observations_var)
        neg_eps_dist_info_vars = self.neg_eps_policy.dist_info_sym(observations_var)
        mix_dist_info_vars = self.mix_policy.dist_info_sym(observations_var)

        surr = tf.reduce_mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)
        surr_pos_eps = tf.reduce_mean(dist.log_likelihood_sym(actions_var, pos_eps_dist_info_vars) * advantages_var)
        surr_neg_eps = tf.reduce_mean(dist.log_likelihood_sym(actions_var, neg_eps_dist_info_vars) * advantages_var)
        surr_mix = tf.reduce_mean(dist.log_likelihood_sym(actions_var, mix_dist_info_vars) * advantages_var)
        surr_loglikelihood = tf.reduce_sum(dist.log_likelihood_sym(actions_var, mix_dist_info_vars))

        params = self.policy.get_params(trainable=True)
        mix_params = self.mix_policy.get_params(trainable=True)
        pos_eps_params = self.pos_eps_policy.get_params(trainable=True)
        neg_eps_params = self.neg_eps_policy.get_params(trainable=True)

        grads = tf.gradients(surr, params)
        grad_pos_eps = tf.gradients(surr_pos_eps, pos_eps_params)
        grad_neg_eps = tf.gradients(surr_neg_eps, neg_eps_params)
        grad_mix = tf.gradients(surr_mix, mix_params)
        grad_mix_lh = tf.gradients(surr_loglikelihood, mix_params)

        self._opt_fun = ext.LazyDict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var, advantages_var],
                outputs=surr,
                log_name="f_loss",
            ),
            f_train=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var, advantages_var],
                outputs=grads,
                log_name="f_grad"
            ),
            f_mix_grad=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var, advantages_var],
                outputs=grad_mix,
                log_name="f_mix_grad"
            ),
            f_pos_grad=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var, advantages_var],
                outputs=grad_pos_eps
            ),
            f_neg_grad=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var, advantages_var],
                outputs=grad_neg_eps
            ),
            f_mix_lh=lambda: tensor_utils.compile_function(
                inputs=[observations_var, actions_var],
                outputs=grad_mix_lh
            ),
            f_kl=lambda: tensor_utils.compile_function(
                inputs=[observations_var],
                outputs=[mean_kl, max_kl],
            )

        )

    def outer_optimize(self, itr, samples_data):
        observations = ext.extract(samples_data, "observations")
        print(len(observations))
        actions = ext.extract(samples_data, "actions")
        advantages = ext.extract(samples_data, "advantages")
        num_traj = len(samples_data["paths"])
        print(num_traj)

        observations = observations[0].reshape(-1, self.env.spec.observation_space.shape[0])
        actions = actions[0].reshape(-1, self.env.spec.action_space.shape[0])
        advantages = advantages[0].reshape(-1)

        s_g = self._opt_fun["f_train"](observations, actions, advantages)
        #s_g = [x/num_traj for x in s_g]
        self.writer.add_scalar("Gradient norm", self.grad_norm(s_g), itr)
        self.writer.add_scalar("Outerloop_gradient_norm", self.grad_norm(s_g), itr)
        self.gradient_backup = copy.deepcopy(s_g)
        self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
        s_g = self.normalize_gradient(s_g)
        self.sgd_update(s_g, self.learning_rate)
        mean_kl, max_kl = self._opt_fun["f_kl"](observations)
        self.writer.add_scalar("MeanKL", mean_kl, itr)
        self.writer.add_scalar("MaxKL", max_kl, itr)

        paths = samples_data["paths"]
        if hasattr(self.baseline, "fit_with_samples"):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

