import gin
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Wrapper
import tensorflow_addons as tfa

@gin.configurable(module=__name__)
class MLP(Layer):
    def __init__(self,num_layers,dim,out_dim,activation='relu',name=None,in_dim=None,spectral_norm=False):
        super().__init__()

        self.layers = []
        for l in range(num_layers):
            l = Dense(
                dim,
                activation = activation,
                name = None if name is None else f'{name}_{l}',
            )

            if spectral_norm:
                l = tfa.layers.SpectralNormalization(l)

            if in_dim is not None:
                l.build((in_dim,))
                in_dim = dim

            self.layers.append(l)

        l = Dense(
            out_dim,
            name = None if name is None else f'{name}_{num_layers}',
        )
        if in_dim is not None:
            l.build((in_dim,))

        self.layers.append(l)

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in self.layers
            ]

@gin.configurable
class Dropout(Wrapper):
    # apply dropout on its "input"
    # Always apply dropout since it is for MC
    def __init__(self, layer, drop_rate):
        super().__init__(layer)
        self.drop_rate = drop_rate

    @tf.function
    def call(self,inputs,training=None):
        x = tf.nn.dropout(inputs,self.drop_rate)
        return self.layer(x,training=training)

@gin.configurable
class MLPDropout(Layer):
    def __init__(self,Dropout,num_layers,dim,out_dim,activation='relu',name=None,in_dim=None,skip_first=False):
        super().__init__()

        self.layers = []
        for l in range(num_layers):
            l = Dense(
                dim,
                activation = activation,
                name = None if name is None else f'{name}_{l}',
            )
            if in_dim is not None:
                l.build((in_dim,))
                in_dim = dim

            if skip_first:
                self.layers.append(l)
            else:
                self.layers.append(Dropout(l))

        l = Dense(
            out_dim,
            name = None if name is None else f'{name}_{num_layers}',
        )
        if in_dim is not None:
            l.build((in_dim,))

        self.layers.append(Dropout(l))

    @tf.function
    def call(self,inputs,training=None):
        o = tf.concat(inputs,axis=-1)
        for l in self.layers:
            o = l(o,training=training)
        return o

    @property
    def decay_vars(self):
        # return only kernels without bias in the network
        return [
            l.layer.kernel if isinstance(l,tf.keras.layers.Wrapper) \
            else l.kernel \
                for l in self.layers
            ]
