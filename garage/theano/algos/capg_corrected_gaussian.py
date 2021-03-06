from lasagne.updates import sgd
import numpy as np
import theano
import theano.tensor as TT
import copy
from tensorboardX import SummaryWriter
import time
from garage.sampler import BatchSampler
from garage.theano.misc import tensor_utils
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
                 decay_learing_rate=False,
                 center_adv=True,
                 positive_adv=False,
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
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        log_dir = log_dir + '/capg%s_batchsize_%d_minibatchsize_%d_subitr_%d_lr_%f' % (exp_time, batch_size, minibatch_size, self.n_sub_itr, learning_rate)
        self.writer = SummaryWriter(log_dir)

    def start_worker(self):
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
            # added for correction
            discount_array = self.discount ** np.arange(len(path["rewards"]))
            advantages = advantages * discount_array

            path["returns"] = special.discount_cumsum(
                path["rewards"], self.discount
            )

            '''
            returns = special.discount_cumsum(
                path["rewards"], self.discount)
            path['returns'] = returns * discount_array
            '''
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

    def train(self):
        self.start_worker()
        self.init_opt()
        j = 0
        while(self.batch_size < self.n_timestep - j):
            paths = self.sampler.obtain_samples(j)
            sample_data = self.sampler.process_samples_discount(j, paths)
            j += sum([len(path["rewards"]) for path in paths])
            avg_returns = np.mean([sum(p["rewards"]) for p in paths])
            print("timesteps: " + str(j) + " average return: " + str(avg_returns))
            self.writer.add_scalar("AverageReturn", avg_returns, j)
            self.writer.add_scalar("Average_outerloop_return", avg_returns, j)
            self.outer_optimize(j, sample_data)

            v = 1e-3
            for _ in range(self.n_sub_itr):
                n_sub = 0 # num of subsamples
                sub_paths_all = []
                self.generate_mix_policy()

                x_paramater = self.flatten_parameters(self.mix_policy.get_param_values())
                dim_theta = len(x_paramater)
                u = np.random.randn(dim_theta)
                y_parameter = self.flatten_parameters(self.mix_policy.get_param_values()) + u * v
                self.mix_policy.set_param_values(y_parameter, trainable = True)

                sub_paths = self.sample_paths(1, self.mix_policy)
                sub_paths_all.append(sub_paths[0])
                n_sub += len(sub_paths[0]["rewards"])
                j += len(sub_paths[0]["rewards"])
                sub_observations = [p["observations"] for p in sub_paths]
                sub_actions = [p["actions"] for p in sub_paths]
                sub_advantages = [p["advantages"] for p in sub_paths]
                d_vector = self.policy.get_param_values() - self.backup_policy.get_param_values()
                g_y = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                self.mix_policy.set_param_values(x_paramater, trainable=True)
                g_x = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                p = np.dot(self.flatten_parameters(g_y), d_vector) * u / v
                print("diff norm:", np.linalg.norm(self.flatten_parameters(g_y)))
                print("dot norm:", np.dot(self.flatten_parameters(g_y) - self.flatten_parameters(g_x), d_vector))
                print("u norm:", np.linalg.norm(u))

                #d_vector = -self.policy.get_param_values() + self.backup_policy.get_param_values()
                '''
                pos_params = self.mix_policy.get_param_values() + d_vector * eps
                neg_params = self.mix_policy.get_param_values() - d_vector * eps
                self.pos_eps_policy.set_param_values(pos_params, trainable=True)
                self.neg_eps_policy.set_param_values(neg_params, trainable=True)

                # first component: dot(likelihood, theta_t - theta_t-1) * policy gradient
                g_mix = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                g_lh = self.f_mix_lh(sub_observations[0], sub_actions[0])
                g_lh = self.flatten_parameters(g_lh)
                #g_mix = self.flatten_parameters(g_mix)
                inner_product = np.dot(g_lh, d_vector)
                #inner_product = np.dot(g_mix, d_vector)
                self.writer.add_scalar("g_mix norm", self.grad_norm(g_mix), j)
                self.writer.add_scalar("loglikelihood times d_vector", inner_product, j)
                fst = [inner_product * g for g in g_mix]
                #fst = [inner_product * g for g in g_lh]

                # second component: dot(Hessian, theta_t - theta_t-1)
                g_pos = self.f_pos_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                g_neg = self.f_neg_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                hv = [(pos - neg) / (2 * eps) for pos, neg in zip(g_pos, g_neg)]
                '''

                while(n_sub < self.minibatch_size):
                    self.generate_mix_policy()

                    x_paramater = self.flatten_parameters(self.mix_policy.get_param_values())
                    dim_theta = len(x_paramater)
                    u = np.random.rand(dim_theta)
                    y_parameter = self.flatten_parameters(self.mix_policy.get_param_values()) + u * v
                    self.mix_policy.set_param_values(y_parameter, trainable=True)

                    sub_paths = self.sample_paths(1, self.mix_policy)
                    sub_paths_all.append(sub_paths[0])
                    n_sub += len(sub_paths[0]["rewards"])
                    j += len(sub_paths[0]["rewards"])
                    sub_observations = [p["observations"] for p in sub_paths]
                    sub_actions = [p["actions"] for p in sub_paths]
                    sub_advantages = [p["advantages"] for p in sub_paths]
                    d_vector = self.policy.get_param_values() - self.backup_policy.get_param_values()
                    g_y = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                    self.mix_policy.set_param_values(x_paramater, trainable=True)
                    g_x = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                    p += np.dot(self.flatten_parameters(g_y) - self.flatten_parameters(g_x), d_vector) * u / v

                    '''
                    pos_params = self.mix_policy.get_param_values() + d_vector * eps
                    neg_params = self.mix_policy.get_param_values() - d_vector * eps
                    self.pos_eps_policy.set_param_values(pos_params, trainable=True)
                    self.neg_eps_policy.set_param_values(neg_params, trainable=True)

                    # first component: dot(likelihood, theta_t - theta_t-1) * policy gradient
                    g_mix = self.f_mix_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                    g_lh = self.f_mix_lh(sub_observations[0], sub_actions[0])
                    g_lh = self.flatten_parameters(g_lh)
                    #g_mix = self.flatten_parameters(g_mix)
                    inner_product = np.dot(g_lh, d_vector)
                    #inner_product = np.dot(g_mix, d_vector)
                    fst_i = [inner_product * g for g in g_mix]
                    #fst_i = [inner_product * g for g in g_lh]
                    fst = [sum(x) for x in zip(fst, fst_i)]

                    # second component: dot(Hessian, theta_t - theta_t-1)
                    g_pos = self.f_pos_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
                    g_neg = self.f_neg_grad(sub_observations[0], sub_actions[0], sub_advantages[0])
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
                '''
                
                #fst = [x/fst_norm*hv_norm for x in fst]
                p /= len(sub_paths_all)
                print(np.linalg.norm(p))
                print(np.linalg.norm(self.flatten_parameters(self.gradient_backup)))
                g_d = self.flatten_parameters(self.gradient_backup) + p
                self.gradient_backup = copy.deepcopy(g_d)
                avg_returns = np.mean([sum(p["rewards"]) for p in sub_paths_all])
                self.writer.add_scalar("AverageReturn", avg_returns, j)
                self.writer.add_scalar("Gradient norm", self.grad_norm(g_d), j)
                print("timesteps: " + str(j) + " average return: " + str(avg_returns))

                if j > 5e5:
                    # check the accuracy of estimator
                    gradient_sample_real = self.sample_paths(100, self.policy)
                    gradient_sample_1 = self.sample_paths(10, self.policy)
                    gradient_sample_2 = self.sample_paths(1,self.policy)

                    gradient_sample_real_observations = np.concatenate([p["observations"] for p in gradient_sample_real])
                    gradient_sample_real_actions = np.concatenate([p["actions"] for p in gradient_sample_real])
                    gradient_sample_real_advantages = np.concatenate([p["advantages"] for p in gradient_sample_real])
                    gradient_real = self.f_train(gradient_sample_real_observations, gradient_sample_real_actions,
                                              gradient_sample_real_advantages)
                    gradient_real = [g/100 for g in gradient_real]

                    gradient_sample_1_observations = np.concatenate([p["observations"] for p in gradient_sample_1])
                    gradient_sample_1_actions = np.concatenate([p["actions"] for p in gradient_sample_1])
                    gradient_sample_1_advantages = np.concatenate([p["advantages"] for p in gradient_sample_1])
                    gradient_1 = self.f_train(gradient_sample_1_observations, gradient_sample_1_actions, gradient_sample_1_advantages)
                    gradient_1 = [g/10 for g in gradient_1]

                    gradient_sample_2_observations = np.concatenate([p["observations"] for p in gradient_sample_2])
                    gradient_sample_2_actions = np.concatenate([p["actions"] for p in gradient_sample_2])
                    gradient_sample_2_advantages = np.concatenate([p["advantages"] for p in gradient_sample_2])
                    gradient_2 = self.f_train(gradient_sample_2_observations, gradient_sample_2_actions,
                                              gradient_sample_2_advantages)

                    diff = self.grad_norm([g1-g2 for g1,g2 in zip(gradient_real, gradient_1)])
                    diff1 = self.grad_norm([g1-g2 for g1,g2 in zip(gradient_real, gradient_2)])
                    print(g_d.shape)
                    diff2 = self.grad_norm([g1-g2 for g1,g2 in zip(gradient_real, g_d)])
                    print("diff real with 10:", diff)
                    print("diff real with 1:", diff1)
                    print("diff real with g_d:", diff2)
                    self.writer.add_scalar("diff_10", diff, j)
                    self.writer.add_scalar("diff_1", diff1, j)
                    self.writer.add_scalar("diff_gd", diff2, j
)
                print(g_d.shape, np.linalg.norm(g_d))
                g_d = g_d/np.linalg.norm(g_d)
                self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
                if self.decay_learning_rate:
                    cur_lr = self.learning_rate * (10 ** (-1 * j / self.n_timestep))
                    print(cur_lr)
                    self.sgd_update(g_d, cur_lr)
                else:
                    self.sgd_update(g_d, self.learning_rate)
                #Compute KL divergence after updated
                sub_observations = [p["observations"] for p in sub_paths]
                mean_kl, max_kl = self.f_kl(sub_observations[0])
                self.writer.add_scalar("MeanKL", mean_kl, j)
                self.writer.add_scalar("MaxKL", max_kl, j)

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        observations_var = self.env.observation_space.new_tensor_variable(
            'observations',
            extra_dims=1
        )
        actions_var = self.env.action_space.new_tensor_variable(
            'actions',
            extra_dims=1
        )
        advantages_var = tensor_utils.new_tensor(
            'advantage', ndim=1, dtype=theano.config.floatX)
        dist = self.policy.distribution
        dist_info_vars = self.policy.dist_info_sym(observations_var)
        old_dist_info_vars = self.backup_policy.dist_info_sym(observations_var)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        mean_kl = TT.mean(kl)
        max_kl = TT.max(kl)

        pos_eps_dist_info_vars = self.pos_eps_policy.dist_info_sym(observations_var)
        neg_eps_dist_info_vars = self.neg_eps_policy.dist_info_sym(observations_var)
        mix_dist_info_vars = self.mix_policy.dist_info_sym(observations_var)

        surr = TT.sum(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)
        surr_pos_eps = TT.sum(dist.log_likelihood_sym(actions_var, pos_eps_dist_info_vars) * advantages_var)
        surr_neg_eps = TT.sum(dist.log_likelihood_sym(actions_var, neg_eps_dist_info_vars) * advantages_var)
        surr_mix = TT.sum(dist.log_likelihood_sym(actions_var, mix_dist_info_vars) * advantages_var)
        surr_loglikelihood = TT.sum(dist.log_likelihood_sym(actions_var, mix_dist_info_vars))

        params = self.policy.get_params(trainable=True)
        mix_params = self.mix_policy.get_params(trainable=True)
        pos_eps_params = self.pos_eps_policy.get_params(trainable=True)
        neg_eps_params = self.neg_eps_policy.get_params(trainable=True)
        backup_params = self.backup_policy.get_params(trainable=True)

        grads = theano.grad(surr, params)
        grad_pos_eps = theano.grad(surr_pos_eps, pos_eps_params)
        grad_neg_eps = theano.grad(surr_neg_eps, neg_eps_params)
        grad_mix = theano.grad(surr_mix, mix_params)
        grad_mix_lh = theano.grad(surr_loglikelihood, mix_params)

        self.f_surr = theano.function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=surr
        )
        self.f_train = theano.function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=grads
        )
        self.f_pos_grad = theano.function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=grad_pos_eps
        )
        self.f_neg_grad = theano.function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=grad_neg_eps
        )
        self.f_mix_grad = theano.function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=grad_mix
        )
        self.f_mix_lh = theano.function(
            inputs=[observations_var, actions_var],
            outputs=grad_mix_lh
        )
        #self.f_update = theano.function(
         #   inputs=[eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7],
          #  outputs=None,
           # updates=sgd([eval_grad1, eval_grad2, eval_grad3, eval_grad4, eval_grad5, eval_grad6, eval_grad7], params,
            #            learning_rate=self.learning_rate)
        #)
        self.f_kl = tensor_utils.compile_function(
            inputs=[observations_var],
            outputs=[mean_kl, max_kl],
        )
        return dict()

    def outer_optimize(self, itr, samples_data):
        observations = ext.extract(samples_data, "observations")
        actions = ext.extract(samples_data, "actions")
        advantages = ext.extract(samples_data, "advantages")
        num_traj = len(samples_data["paths"])

        s_g = self.f_train(observations[0], actions[0], advantages[0])
        s_g = [x/num_traj for x in s_g]

        if itr > 1e5:
            # check the accuracy of estimator
            gradient_sample_real = self.sample_paths(10, self.policy)
            gradient_sample_real_observations = np.concatenate([p["observations"] for p in gradient_sample_real])
            gradient_sample_real_actions = np.concatenate([p["actions"] for p in gradient_sample_real])
            gradient_sample_real_advantages = np.concatenate([p["advantages"] for p in gradient_sample_real])
            gradient_real = self.f_train(gradient_sample_real_observations, gradient_sample_real_actions,
                                         gradient_sample_real_advantages)
            gradient_real = [g / 10 for g in gradient_real]
            print("real norm:", self.grad_norm(gradient_real))
            print("s_g norm:", self.grad_norm(s_g))
            diff = self.grad_norm([g1 - g2 for g1, g2 in zip(gradient_real, s_g)])
            print("out diff real with 10:", diff)

        self.writer.add_scalar("Gradient norm", self.grad_norm(s_g), itr)
        self.writer.add_scalar("Outerloop_gradient_norm", self.grad_norm(s_g), itr)
        self.gradient_backup = copy.deepcopy(s_g)
        self.backup_policy.set_param_values(self.policy.get_param_values(trainable=True), trainable=True)
        s_g = self.normalize_gradient(s_g)
        if self.decay_learning_rate:
            cur_lr = self.learning_rate * (10**(-1 * itr/self.n_timestep))
            print(cur_lr)
            self.sgd_update(s_g, cur_lr)
        else:
            self.sgd_update(s_g, self.learning_rate)
        mean_kl, max_kl = self.f_kl(observations[0])
        self.writer.add_scalar("MeanKL", mean_kl, itr)
        self.writer.add_scalar("MaxKL", max_kl, itr)
        return dict()


