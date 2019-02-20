import time

import pyprind
import tensorflow as tf
from garage.misc import ext

from garage.core import Serializable
from garage.misc import ext
from garage.misc import logger
from garage.optimizers import BatchDataset
from garage.tf.misc import tensor_utils
import numpy as np
from numpy import linalg as LA


class SVRGOptimizer(Serializable):
    """
    Performs stochastic variance reduction gradient in VPG.
    """

    def __init__(
            self,
            alpha=0.001,
            # learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-5,
            batch_size=32,
            max_batch=20,
            epsilon=1e-8,
            verbose=False,
            scale=1.0,
            name="SVRGOptimizer",
            **kwargs):
        """
        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will
         be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._alpha = alpha
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._max_batch = max_batch
        self._epsilon = epsilon
        self._verbose = verbose
        self._scale = scale
        self._input_vars = None
        self._name = name


    def update_opt(self, loss, loss_tilde, target, target_tilde,
                   leq_constraint, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should
         implement methods of the
        :class:`garage.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon),
         of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs

        self._target = target
        self._target_tilde = target_tilde

        constraint_term, constraint_value = leq_constraint
        self._max_constraint_val = constraint_value

        w = target.get_params(trainable=True)
        grads = tf.gradients(loss, xs=w)
        for idx, (g, param) in enumerate(zip(grads, w)):
            if g is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        w_tilde = target_tilde.get_params(trainable=True)
        grads_tilde = tf.gradients(loss_tilde, xs=w_tilde)
        for idx, (g_t, param_t) in enumerate(zip(grads_tilde, w_tilde)):
            if g_t is None:
                grads_tilde[idx] = tf.zeros_like(param_t)
        flat_grad_tilde = tensor_utils.flatten_tensor_variables(grads_tilde)

        self._opt_fun = ext.LazyDict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs = inputs + extra_inputs,
                outputs = loss,
            ),
            f_loss_tilde=lambda :tensor_utils.compile_function(
                inputs = inputs + extra_inputs,
                outputs = loss_tilde,
            ),
            f_grad=lambda :tensor_utils.compile_function(
                inputs = inputs + extra_inputs,
                outputs = flat_grad,
            ),
            f_grad_tilde=lambda :tensor_utils.compile_function(
                inputs = inputs + extra_inputs,
                outputs = flat_grad_tilde,
            ),
            f_loss_constraint=lambda :tensor_utils.compile_function(
                inputs = inputs + extra_inputs,
                outputs = [loss, constraint_term],
            ),
        )
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)


    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def loss_tilde(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)
        return self._opt_fun["f_loss_tilde"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None):

        if not inputs:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]
        f_grad = self._opt_fun["f_grad"]
        f_grad_tilde = self._opt_fun["f_grad_tilde"]

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        param = np.copy(self._target.get_param_values(trainable=True))
        logger.log("Start SVRPG optimization: #parameters: %d, #inputs %d" % (
            len(param), len(inputs[0])
        ))
        dataset = BatchDataset(
            inputs, self._batch_size, extra_inputs=extra_inputs
        )
        start_time = time.time()

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))
            grad_sum = np.zeros_like(param)
            g_mean_tilde = f_grad_tilde(inputs,extra_inputs)
            logger.record_tabular('g_mean_tilde', LA.norm(g_mean_tilde))
            print("-------------mini-batch-------------------")
            num_batch = 0
            while num_batch < self._max_batch:
                batch = dataset.random_batch()
                g = f_grad(*(batch)) - f_grad_tilde(*(batch)) + g_mean_tilde
                grad_sum += g
                prev_w = np.copy(self._target.get_param_values(trainable=True))
                step = self._alpha * g
                cur_w = prev_w + step
                self._target.set_param_value(cur_w, trainable=True)
                num_batch += 1
            print("max batch achieved {:}".format(num_batch))
            grad_sum /= 1.0 * num_batch
            logger.record_tabular('gdist', LA.norm(
                grad_sum - g_mean_tilde))
            cur_w = np.copy(self._target.get_param_values(trainable=True))
            w_tilde = self._target_tilde.get_params_values(trainable=True)
            self._target_tilde.set_param_values(cur_w, trainable=True)
            logger.record_tabular('wnorm', LA.norm(cur_w))
            logger.record_tabular('w_dist', LA.norm(
                cur_w - w_tilde) / LA.norm(cur_w))

            if self._verbose:
                if progbar.active:
                    progbar.stop()
            if abs(LA.norm(cur_w - w_tilde) /
                   LA.norm(cur_w)) < self._tolerance:
                break
