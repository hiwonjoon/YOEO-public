# Deterministic Policy
import gin
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
import tensorflow_probability as tfp
tfd = tfp.distributions

@gin.configurable(module=__name__)
class DeterministicPolicy(Model):
    def __init__(self,Net,scale=1.,act_noise=0.1):
        super().__init__()

        self.net = Net()

        self.scale = scale
        self.act_noise = act_noise

    def _action(self,ob,stochastic=False):
        a = self.scale * tf.nn.tanh(self.net(ob))

        if stochastic:
            noise = tf.random.get_global_generator().normal(tf.shape(a),stddev=self.act_noise)
            a += noise

            a = tf.clib_by_value(a,-self.scale,self.scale)
        
        return a

    @tf.function
    def action(self,ob,stochastic=True):
        a = self._action(ob,stochastic=stochastic)
        return a, None

    @tf.function
    def action_sample(self,ob,num_samples):
        a = self.scale * tf.nn.tanh(self.net(ob))

        noise = tf.random.get_global_generator().normal(tf.concat([[num_samples],tf.shape(a)],axis=0),stddev=self.act_noise)
        a = a[None] + noise

        a = tf.clip_by_value(a,-self.scale,self.scale)

        return a

    def __call__(self,ob,stochastic=True):
        if ob.ndim == 1:
            ob = ob[None]
            flatten = True
        else:
            flatten = False

        a, _ = self.action(ob,stochastic)

        if flatten:
            a = a[0].numpy()

        return a, None

    @gin.configurable(module=__name__)
    def prepare_update(
        self,
        Qs,
        learning_rate = 1e-4,
        use_target_Q_for_optimize = False,
        max_grad_norm = 0.,
        reduce='min',
        optimizer='adam',
    ):
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.95, momentum=0.0, epsilon=1e-5, centered=True)
        else:
            assert False

        reports = {
            'loss' : tf.keras.metrics.Mean(name='loss'),
            'grad_norm' : tf.keras.metrics.Mean(name='grad_norm'),
        }

        @tf.function
        def update_fn(ob):
            with tf.GradientTape() as tape:
                a = self._action(ob,stochastic=False)

                qs = tf.stack([
                    Q(ob,a,use_target=use_target_Q_for_optimize)
                    for Q in Qs],axis=-1)

                if reduce == 'min':
                    target = tf.reduce_min(qs,axis=-1)
                elif reduce == 'mean':
                    target = tf.reduce_mean(qs,axis=-1)
                elif reduce == 'max':
                    target = tf.reduce_max(qs,axis=-1)
                else:
                    assert False

                loss = -tf.reduce_mean(target,axis=0)

            gradients = tape.gradient(loss, self.net.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            if max_grad_norm > 0:
                gradients = gradients_clipped

            reports['loss'](loss)
            reports['grad_norm'](grad_norm)

            optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
            return loss

        return update_fn, reports

    @gin.configurable(module=__name__)
    def prepare_behavior_clone(
        self,
        learning_rate=1e-4,
        max_grad_norm=0,
    ):
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        reports = {
            'loss':tf.keras.metrics.Mean(),
            'grad_norm' : tf.keras.metrics.Mean(),
        }

        @tf.function
        def update_fn(ob,ac):
            with tf.GradientTape() as tape:
                a = self._action(ob,stochastic=False)

                loss = tf.reduce_mean(0.5 * (a-ac)**2)
                reports['loss'](loss)

            gradients = tape.gradient(loss, self.net.trainable_variables)

            gradients_clipped, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            reports['grad_norm'](grad_norm)

            if max_grad_norm > 0:
                gradients = gradients_clipped

            optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

            return loss

        return update_fn, reports
