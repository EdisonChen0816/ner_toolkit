# encoding=utf-8
import re
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops


class AdamWeightDecayOptimizer(tf.train.Optimizer):

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, exclude_from_weight_decay=None, name='AdamWeightDecayOptimizer'):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for grad,param in grads_and_vars:
            if grad is None or param is None: continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name+'/adam_m',
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name+'/adam_v',
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            next_m = tf.multiply(self.beta_1, m)+tf.multiply(1.0-self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v)+tf.multiply(1.0-self.beta_2, tf.square(grad))
            update = next_m/(tf.sqrt(next_v)+self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate*param
            update_with_lr = self.learning_rate*update
            next_param = param-update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _get_variable_name(self, param_name):
        m = re.match('^(.*):\\d+$', param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _do_use_weight_decay(self, param_name):
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class AdamWeightDecayOptimizerMulti(optimizer.Optimizer):

    def __init__(self, learning_rate, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, exclude_from_weight_decay=None, name='AdamWeightDecayOptimizer'):
        super(AdamWeightDecayOptimizerMulti, self).__init__(False, name)
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare(self):
        self.learning_rate_t = ops.convert_to_tensor(
            self.learning_rate, name='learning_rate')
        self.weight_decay_rate_t = ops.convert_to_tensor(
            self.weight_decay_rate, name='weight_decay_rate')
        self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
        self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
        self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'm', self._name)
            self._zeros_slot(v, 'v', self._name)

    def _apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(self.weight_decay_rate_t, var.dtype.base_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        next_m = (tf.multiply(beta_1_t, m)+tf.multiply(1.0-beta_1_t, grad))
        next_v = (tf.multiply(beta_2_t, v)+tf.multiply(1.0-beta_2_t, tf.square(grad)))
        update = next_m/(tf.sqrt(next_v)+epsilon_t)
        if self._do_use_weight_decay(self._get_variable_name(var)):
            update += weight_decay_rate_t*var
        update_with_lr = learning_rate_t*update
        next_param = var-update_with_lr
        return control_flow_ops.group(*[var.assign(next_param), m.assign(next_m), v.assign(next_v)])

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(self.weight_decay_rate_t, var.dtype.base_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        next_m = (tf.multiply(beta_1_t, m)+tf.multiply(1.0-beta_1_t, grad))
        next_v = (tf.multiply(beta_2_t, v)+tf.multiply(1.0-beta_2_t, tf.square(grad)))
        update = next_m/(tf.sqrt(next_v)+epsilon_t)
        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t*var
        update_with_lr = learning_rate_t*update
        next_param = var-update_with_lr
        return control_flow_ops.group(*[var.assign(next_param), m.assign(next_m), v.assign(next_v)])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        learning_rate_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(self.weight_decay_rate_t, var.dtype.base_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        m_t = state_ops.assign(m, m*beta_1_t, use_locking=self._use_locking)
        m_scaled_g_values = grad*(1-beta_1_t)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        v_scaled_g_values = (grad*grad)*(1-beta_2_t)
        v_t = state_ops.assign(v, v*beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        update = m_t/(math_ops.sqrt(v_t)+epsilon_t)
        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t*var
        update_with_lr = learning_rate_t*update
        var_update = state_ops.assign_sub(var, update_with_lr, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        m = re.match('^(.*):\\d+$', param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, hvd=None, num_gpus=1, use_fp16=False):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(tf.constant(value=init_lr, shape=[], dtype=tf.float32),
                                              global_step, num_train_steps, end_learning_rate=0.0, power=1.0, cycle=False)
    if num_gpus > 1:
        learning_rate = learning_rate*num_gpus
    global_steps = tf.cast(tf.cast(global_step, tf.int32), tf.float32)
    warmup_steps = tf.cast(tf.constant(num_warmup_steps, dtype=tf.int32), tf.float32)
    is_warmup = tf.cast(global_steps < warmup_steps, tf.float32)
    learning_rate = (1.0-is_warmup)*learning_rate+is_warmup*init_lr*global_steps/warmup_steps
    if num_gpus > 1:
        optimizer = AdamWeightDecayOptimizerMulti(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']
        )
    else:
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']
        )
        if use_fp16:
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
                init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
        if hvd is not None:
            from horovod.tensorflow.compression import Compression
            optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True, compression=Compression.none)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    is_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads]) \
        if use_fp16 else tf.constant(True, dtype=tf.bool)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0,
                                      use_norm=tf.cond(is_finite, lambda: tf.global_norm(grads), lambda: tf.constant(1.0)))
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    n_global_step = tf.cond(is_finite, lambda: global_step+1, lambda: global_step)
    return tf.group(train_op, [global_step.assign(n_global_step)])