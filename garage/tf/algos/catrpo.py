import time
from garage.misc import logger
from garage.misc import ext
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.optimizers.cg_optimizer import CGOptimizer
from garage.tf.misc import tensor_utils
from garage.core.serializable import Serializable
import tensorflow as tf
import numpy as np
import copy


class CATRPO(BatchPolopt, Serializable):
    """
    Curvature-aided Trust Region Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            backup_policy,
            mix_policy,
            pos_eps_policy,
            neg_eps_policy,
            baseline,
            minibatch_size=500,
            n_sub_itr=10,
            optimizer=None,
            optimizer_args=None,
            delta=0.01,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.optimizer = optimizer
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            self.optimizer = CGOptimizer(**optimizer_args)

        self.opt_info = None
        self.backup_policy = backup_policy
        self.mix_policy = mix_policy
        self.pos_eps_policy = pos_eps_policy
        self.neg_eps_policy = neg_eps_policy
        self.minibatch_size = minibatch_size
        self.n_sub_itr = n_sub_itr
        self.delta = delta
        super(CATRPO, self).__init__(
            env=env, policy=policy, baseline=baseline, **kwargs)

    def generate_mix_policy(self):
        a = np.random.uniform(0.0, 1.0)
        mix = a * self.policy.get_param_values() + (1 - a) * self.backup_policy.get_param_values()
        self.mix_policy.set_param_values(mix, trainable=True)

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
            path_baseline = self.baseline.predict(path)
            advantages = []
            returns = []
            return_so_far = 0
            for t in range(len(rewards) - 1, -1, -1):
                return_so_far = rewards[t] + self.discount * return_so_far
                returns.append(return_so_far)
                advantage = return_so_far - path_baseline[t]
                advantages.append(advantage)
            # The advantages are stored backwards in time, so we need to revert it
            advantages = np.array(advantages[::-1])
            # And we need to do the same thing for the list of returns
            returns = np.array(returns[::-1])

            advantages = (advantages - np.mean(advantages)) / (
                np.std(advantages) + 1e-8)

            path["advantages"] = advantages
            path["returns"] = returns

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

    @overrides
    def init_opt(self):
        observations_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        actions_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantages_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = self.backup_policy.dist_info_sym(observations_var)
        dist_info_vars = self.policy.dist_info_sym(observations_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        pos_eps_dist_info_vars = self.pos_eps_policy.dist_info_sym(observations_var)
        neg_eps_dist_info_vars = self.neg_eps_policy.dist_info_sym(observations_var)
        mix_dist_info_vars = self.mix_policy.dist_info_sym(observations_var)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr = -tf.reduce_mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)
        surr_pos_eps = -tf.reduce_mean(dist.log_likelihood_sym(actions_var, pos_eps_dist_info_vars) * advantages_var)
        surr_neg_eps = -tf.reduce_mean(dist.log_likelihood_sym(actions_var, neg_eps_dist_info_vars) * advantages_var)
        surr_mix = -tf.reduce_mean(dist.log_likelihood_sym(actions_var, mix_dist_info_vars) * advantages_var)
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

        inputs_list = [observations_var, actions_var, advantages_var]

        self.optimizer.update_opt(loss=surr, target=self.policy,
                                  leq_constraint=(mean_kl, self.delta),
                                  inputs=inputs_list)

        self._opt_fun = ext.LazyDict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=surr,
                log_name="f_loss",
            ),
            f_train=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=grads,
                log_name="f_grad"
            ),
            f_mix_grad=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=grad_mix,
                log_name="f_mix_grad"
            ),
            f_pos_grad=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=grad_pos_eps
            ),
            f_neg_grad=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=grad_neg_eps
            ),
            f_mix_lh=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=grad_mix_lh
            ),
            f_kl=lambda: tensor_utils.compile_function(
                inputs=inputs_list,
                outputs=[mean_kl, max_kl],
            )

        )

    @overrides
    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker(sess)
            start_time = time.time()
            self.num_samples = 0
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining new samples...")
                    paths = self.obtain_samples(itr)
                    for path in paths:
                        self.num_samples += len(path["rewards"])
                    logger.log("total num samples..." + str(self.num_samples))
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.outer_optimize(samples_data)
                    for sub_itr in range(self.n_sub_itr):
                        logger.log("Minibatch Optimizing...")
                        self.inner_optimize(samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(
                        itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular(
                        'ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    #if self.plot:
                     #   self.update_plot()
                      #  if self.pause_for_plot:
                       #     input("Plotting evaluation run: Press Enter to "
                        #          "continue...")
        self.shutdown_worker()

    def outer_optimize(self, samples_data):
        logger.log("optimizing policy")
        observations = ext.extract(samples_data, "observations")
        actions = ext.extract(samples_data, "actions")
        advantages = ext.extract(samples_data, "advantages")

        num_traj = len(samples_data["paths"])

        observations = observations[0].reshape(-1, self.env.spec.observation_space.shape[0])
        actions = actions[0].reshape(-1,self.env.spec.action_space.shape[0])
        advantages = advantages[0].reshape(-1)
        inputs = tuple([observations, actions, advantages])

        s_g = self._opt_fun["f_train"](*(list(inputs)))
        #s_g = [x / num_traj for x in s_g]
        self.gradient_backup = copy.deepcopy(s_g)
        g_flat = self.flatten_parameters(s_g)

        loss_before = self._opt_fun["f_loss"](*(list(inputs)))
        self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
        self.optimizer.optimize(inputs, g_flat)
        loss_after = self._opt_fun["f_loss"](*(list(inputs)))
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self._opt_fun['f_kl'](*(list(inputs)))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    def inner_optimize(self, outer_sample):

        observations = ext.extract(outer_sample, "observations")
        actions = ext.extract(outer_sample, "actions")
        advantages = ext.extract(outer_sample, "advantages")

        outer_observations = observations[0].reshape(-1, self.env.spec.observation_space.shape[0])
        outer_actions = actions[0].reshape(-1,self.env.spec.action_space.shape[0])
        outer_advantages = advantages[0].reshape(-1)

        n_sub = 0
        sub_paths_all = []
        self.generate_mix_policy()
        sub_paths = self.sample_paths(1, self.mix_policy)
        sub_paths_all.append(sub_paths[0])
        n_sub += len(sub_paths[0]["rewards"])
        self.num_samples += len(sub_paths[0]["rewards"])

        sub_observations = [p["observations"] for p in sub_paths]
        sub_actions = [p["actions"] for p in sub_paths]
        sub_advantages = [p["advantages"] for p in sub_paths]
        eps = 1e-6
        d_vector = self.policy.get_param_values() - self.backup_policy.get_param_values()
        pos_params = self.mix_policy.get_param_values() + d_vector * eps
        neg_params = self.mix_policy.get_param_values() - d_vector * eps
        self.pos_eps_policy.set_param_values(pos_params, trainable=True)
        self.neg_eps_policy.set_param_values(neg_params, trainable=True)

        # first component: dot(likelihood, theta_t - theta_t-1) * policy gradient
        g_mix = self._opt_fun["f_mix_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
        g_lh = self._opt_fun["f_mix_lh"](sub_observations[0], sub_actions[0])
        g_lh = self.flatten_parameters(g_lh)
        inner_product = np.dot(g_lh, d_vector)
        fst = [inner_product * g for g in g_mix]

        # second component: dot(Hessian, theta_t - theta_t-1)
        g_pos = self._opt_fun["f_pos_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
        g_neg = self._opt_fun["f_neg_grad"](sub_observations[0], sub_actions[0], sub_advantages[0])
        hv = [(pos - neg) / (2 * eps) for pos, neg in zip(g_pos, g_neg)]

        while (n_sub < self.minibatch_size):
            self.generate_mix_policy()
            sub_paths = self.sample_paths(1, self.mix_policy)
            n_sub += len(sub_paths[0]["rewards"])
            self.num_samples += len(sub_paths[0]["rewards"])

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
        fst = [x/10 for x in fst]
        # gradient as sum
        fst_norm = self.grad_norm(fst)
        hv_norm = self.grad_norm(hv)
        backup_gradient_norm = self.grad_norm(self.gradient_backup)
        #self.writer.add_scalar("first_component_norm", fst_norm, j)
        #self.writer.add_scalar("hv_norm", hv_norm, j)
        #self.writer.add_scalar("back_gradient_norm", backup_gradient_norm, j)

        g_d = [sum(x) for x in zip(fst, hv, self.gradient_backup)]
        self.gradient_backup = copy.deepcopy(g_d)
        avg_returns = np.mean([sum(p["rewards"]) for p in sub_paths_all])
        #self.writer.add_scalar("AverageReturn", avg_returns, j)
        #self.writer.add_scalar("Gradient norm", self.grad_norm(g_d), j)
        print("timesteps: " + str(self.num_samples) + " average return: " + str(avg_returns))

        sub_observations = np.concatenate([p["observations"] for p in sub_paths_all])
        sub_actions = np.concatenate([p["actions"] for p in sub_paths_all])
        sub_advantages = np.concatenate([p["advantages"] for p in sub_paths_all])
        sub_observations = sub_observations.reshape(-1, self.env.spec.observation_space.shape[0])
        sub_actions = sub_actions.reshape(-1, self.env.spec.action_space.shape[0])
        sub_advantages = sub_advantages.reshape(-1)

        #sub_observations = np.concatenate((sub_observations, outer_observations))
        #sub_actions = np.concatenate((sub_actions, outer_actions))
        #sub_advantages = np.concatenate((sub_advantages, outer_advantages))

        print(sub_observations.shape)

        inputs = tuple([sub_observations, sub_actions, sub_advantages])
        self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
        flat_g_d = self.flatten_parameters(g_d)
        self.optimizer.optimize(inputs, flat_g_d)
        # Compute KL divergence after updated
        #sub_observations = [p["observations"] for p in sub_paths]
        #mean_kl, max_kl = self.f_kl(sub_observations[0])
        #self.writer.add_scalar("MeanKL", mean_kl, j)
        #self.writer.add_scalar("MaxKL", max_kl, j)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
