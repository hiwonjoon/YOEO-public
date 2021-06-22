import gin
import tensorflow as tf
import tensorflow_addons as tfa

class Optimizer():
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm,
        weight_decay,
    ):
        self.vars = vars
        self.decay_vars = decay_vars
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay

        self.reports = {
            'loss': tf.keras.metrics.Mean(name='loss'),
        }

        if weight_decay > 0:
            self.reports['W^2'] = tf.keras.metrics.Mean(name='W^2')

        if max_grad_norm > 0:
            self.reports['grad_norm'] = tf.keras.metrics.Mean(name='grad_norm')

        self.optimizer = None

    def minimize(self,tape,loss):
        self.reports['loss'](loss)

        if self.weight_decay > 0:
            with tape:
                W_sq = tf.add_n([0.5 * tf.reduce_sum(var**2) for var in self.decay_vars])
                self.reports['W^2'](W_sq)

                loss += self.weight_decay * W_sq

        gradients = tape.gradient(loss, self.vars)

        if self.max_grad_norm > 0:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.reports['grad_norm'](grad_norm)

        self.optimizer.apply_gradients(zip(gradients, self.vars))

@gin.configurable(module=__name__)
class AdamOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        weight_decay=0.,
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False
    ):
        super().__init__(vars,decay_vars,max_grad_norm,weight_decay)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon, amsgrad=amsgrad)

@gin.configurable(module=__name__)
class SGDOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        weight_decay=0.,
        lr=1e-3,
        momentum=0.9,
        nesterov=False
    ):
        super().__init__(vars,decay_vars,max_grad_norm,weight_decay)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=momentum,nesterov=nesterov)


@gin.configurable(module=__name__)
class RMSPropOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        lr=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-05,
        centered=True
    ):
        super().__init__(vars,decay_vars,max_grad_norm)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, epsilon=epsilon, centered=centered)

@gin.configurable(module=__name__)
class AdamWOptimizer(Optimizer):
    def __init__(
        self,
        vars,
        decay_vars,
        max_grad_norm=0.,
        lr=1e-3,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False
    ):
        super().__init__(vars,decay_vars,max_grad_norm,weight_decay)
        assert len(self.decay_vars) > 0

        self.optimizer = tfa.optimizers.AdamW(learning_rate=lr,weight_decay=weight_decay,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad)

    def minimize(self,tape,loss):
        self.reports['loss'](loss)
        gradients = tape.gradient(loss, self.vars)

        if self.max_grad_norm > 0:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.reports['grad_norm'](grad_norm)

        self.optimizer.apply_gradients(zip(gradients, self.vars),decay_var_list=self.decay_vars)

        W_sq= tf.add_n([tf.reduce_sum(var**2) for var in self.decay_vars])
        self.reports['W^2'](W_sq)

class ClipConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be in some range"""

    def __init__(self, min_value, max_value):
        self.min_value, self.max_value = min_value, max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}
