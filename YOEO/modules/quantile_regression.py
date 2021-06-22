import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

@gin.configurable(module=__name__)
class CosineBasedPhi(Layer):
    # [0,1] --> R^d
    def __init__(
        self,
        n, # number of basis
        d, # output dimension
    ):
        super().__init__()

        self.n = n
        self.i = tf.constant(np.arange(self.n),tf.float32)
        self.l = Dense(d,activation='relu')

        self.l.build(input_shape=(n,))

    @tf.function
    def call(self,inputs,training=None):
        """
            inputs : taus, shape of [multi-dimensional batch dim]
            outputs: [multi-dimensional batch dim, d]
        """
        tau = inputs

        i = tf.broadcast_to(self.i,tf.concat([tf.shape(tau),[self.n]],axis=0))
        tau = tf.expand_dims(tau,axis=-1)

        inp = tf.math.cos(3.1415 * tau * i)
        return self.l(inp)

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        return [self.l.kernel]

class QuantileNetwork(object):
    """
    learns a distribution P_x(Y) in the form of quantile function y = Q_x(p)
    """
    def __init__(
        self,
        psi, # Feature Generator; R^(|x|) -> R^|d|
        phi, # Tau feature generator; [0,1] -> R^|d|
        f, # MLP generating final action-value prediction Z@tau; R^|d| -> R
    ):
        self.psi = psi
        self.phi = phi
        self.f = f

    def _get_B(self,x):
        if isinstance(x,tuple):
            return tf.shape(x[0])[:-1]
        else:
            return tf.shape(x)[:-1]

    @tf.function
    def call(self,x,tau):
        """
        inp:
           x: [batch dim] + [|x|]
           tau: [batch dim, N]
        return y = Q_x(tau); [batch_dim, N]
        """
        return tf.squeeze(
            self.f(tf.expand_dims(self.psi(x),axis=-2)*self.phi(tau)), #[B,1,feature_dim] * [B,N,feature_dim]
            axis=-1)

    @tf.function
    def quantile(self,x,p):
        # sugar for easy use.
        # return y s.t. P_x(Y <= y) = p
        tau = p * tf.ones(self._get_B(x),tf.float32)[...,None]
        return tf.squeeze(self.call(x,tau),axis=-1)

    @tf.function
    def sample(self,x,num_samples):
        # sample y ~ P_x(Y) from the distribution

        batch_dims = tf.concat([self._get_B(x),[num_samples]],axis=0)
        tau = tf.stop_gradient(tf.random.get_global_generator().uniform(batch_dims,maxval=1.))

        return tau, self.call(x,tau)

    @tf.function
    def CDF(self,x,y,num_samples):
        # return P_x(Y <= y)
        _, ys = self.sample(x,num_samples)
        return ys, tf.reduce_mean(tf.cast(tf.less(ys,y[:,None]),tf.float32),axis=-1)

    @tf.function
    def entropy(self,x,num_samples,step_size=0.05,minimum_diff=1.):
        # return (approximated, sample-based) differential entropy H(Y_x)
        # step_size & minimum_diff is a hyperparameter.

        """
        # version 1 -- follows the definition
        # does not work due to imperfection of quantile approximation
        with tf.GradientTape() as tape:
            #tape.watch(tau)
            tau, ys = self.sample(x,num_samples)

        grad = tape.gradient(ys,tau)

        entropy = tf.reduce_mean(grad,axis=-1)
        return entropy
        """

        # version 2 -- smooth gradient calculation with large step.
        batch_dims = tf.concat([self._get_B(x),[num_samples]],axis=0)
        tau = tf.random.get_global_generator().uniform(batch_dims,minval=step_size,maxval=1.-step_size)

        ys_low = self.call(x,tau-step_size)
        ys_high = self.call(x,tau+step_size)

        grad = tf.maximum(ys_high-ys_low,minimum_diff) / (2*step_size)
        entropy = tf.reduce_mean(tf.math.log(tf.maximum(grad,1e-8)),axis=-1)
        return entropy

    @tf.function
    def entropy_as_normal(self,x,num_samples):
        # return (approximated, sample-based) differential entropy H(Y_x), treating P_x(Y) is normal
        # distribution.
        _, ys = self.sample(x,num_samples)
        sigma = tf.math.reduce_std(ys,axis=-1)
        entropy = tf.math.log(sigma * (2*3.1415*2.718)**0.5)
        return entropy

    @staticmethod
    def distortion(distortion_type,eta):
        if distortion_type is None:
            distortion_fn = lambda tau: tau
        elif distortion_type == 'CPW':
            distortion_fn = lambda tau: tau**eta / (tau**eta + (1-tau)**eta)**(1/eta)
        elif distortion_type == 'Pow':
            distortion_fn = lambda tau: tau ** (1 / (1+eta)) if eta > 0 else 1 - (1 - tau)**(1/(1+eta))
        elif distortion_type == 'CVaR':
            distortion_fn = lambda tau: eta*tau
        elif distortion_type == 'Wang':
            assert False
        elif distortion_type == 'truncated':
            distortion_fn = lambda tau: (1-2*eta) * tau + eta
        elif distortion_type == 'fixed':
            distortion_fn = lambda tau: eta * tf.ones_like(tau)
        else:
            assert False
        return distortion_fn

    def build_sample_fn_with_distortion(self,distortion_fn):
        @tf.function
        def sample_fn(x,num_samples):
            batch_dims = tf.concat([self._get_B(x),[num_samples]],axis=0)
            tau = tf.stop_gradient(tf.random.get_global_generator().uniform(batch_dims,maxval=1.))

            return self.call(x,distortion_fn(tau))

        return sample_fn

    @tf.function
    def huber_quantile_loss(self,x,target_dist,N,kappa):
        # some weird side effect exists.
        # gradients gets weird when `self.sample(x,N)` is used. why?
        tau, quantiles = self.sample(x,N)

        delta = target_dist[:,None,:] - quantiles[:,:,None] # [B,1,N'] - [B,N,1] = [B,N,N']

        L_kappa = tf.where(
            tf.less_equal(tf.abs(delta),kappa),
            0.5 * delta**2,
            kappa * (tf.abs(delta) - 0.5 * kappa)
        ) #[B,N_i,N_j]

        L = tf.reduce_sum(tf.reduce_mean(
                tf.abs(tau[:,:,None] - tf.cast(tf.less_equal(delta,0),tf.float32)) * L_kappa / kappa,
            axis=2),axis=1)

        return L

class QuantileNetworkEnsemble(QuantileNetwork):
    def __init__(
        self,
        qns
    ):
        # Quantile / Entropy does not work
        # sole purpose of this wrapper is for sample / CDF.
        self.qns = qns

    def call(self,x,tau):
        assert False, 'cannot be directly called'

    @tf.function
    def sample(self,x,num_samples):
        return 0., tf.concat([qn.sample(x,num_samples//len(self.qns))[1] for qn in self.qns],axis=-1)

    @tf.function
    def quantile_sample_based(self,x,p,num_samples):
        # get a quantile based on samples
        # return y s.t. P_x(Y <= y) = p

        ys = tf.sort(self.sample(x,num_samples)[1],axis=-1)
        idx = tf.cast(p * tf.cast(num_samples,tf.float32),tf.int32)

        return ys[:,idx]

    def build_sample_fn_with_distortion(self,distortion_fn,stack=False):
        sample_fns = [
            qn.build_sample_fn_with_distortion(distortion_fn) for qn in self.qns
        ]

        @tf.function
        def sample_fn(x,num_samples):
            samples = [sample(x,num_samples//len(self.qns)) for sample in sample_fns]
            if stack:
                return tf.stack(samples,axis=-2)
            else:
                return tf.concat(samples,axis=-1)

        return sample_fn

class DeltaNetwork(QuantileNetwork):
    # adapter class for V and Q network, which makes a point estimate instead of distributional output
    def __init__(self,net):
        self.net = net

    def call(self,x,tau):
        N = tf.shape(tau)[-1]
        return tf.repeat(self.net(x),N,axis=-1)

    @tf.function
    def sample(self,x,num_samples):
        return None, tf.repeat(self.net(x),num_samples,axis=-1)

    @tf.function
    def entropy(self,x,num_samples,step_size=0.05,minimum_diff=1.):
        return tf.zeros(self._get_B(x),tf.float32)

    @tf.function
    def entropy_as_noraml(self,x,num_samples):
        return tf.zeros(self._get_B(x),tf.float32)
