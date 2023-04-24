import math

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
        layer_normalization=None,
        kernel_constraint=None,
        use_bias=True,
    ):
        super().__init__()
        self.layer_size = layer_sizes
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias

    @property
    def inputs(self):
        return self.x

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.y_

    @timing
    def build(self):
        print("Building feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        for i in range(len(self.layer_size) - 2):
            if self.batch_normalization is None and self.layer_normalization is None:
                y = self._dense(
                    y,
                    self.layer_size[i + 1],
                    activation=(
                        self.activation[i]
                        if isinstance(self.activation, list)
                        else self.activation
                    ),
                    use_bias=self.use_bias,
                )
            elif self.batch_normalization and self.layer_normalization:
                raise ValueError(
                    "Can not apply batch_normalization and layer_normalization at the "
                    "same time."
                )
            elif self.batch_normalization == "before":
                y = self._dense_batchnorm_v1(y, self.layer_size[i + 1])
            elif self.batch_normalization == "after":
                y = self._dense_batchnorm_v2(y, self.layer_size[i + 1])
            elif self.layer_normalization == "before":
                y = self._dense_layernorm_v1(y, self.layer_size[i + 1])
            elif self.layer_normalization == "after":
                y = self._dense_layernorm_v2(y, self.layer_size[i + 1])
            else:
                raise ValueError(
                    "batch_normalization: {}, layer_normalization: {}".format(
                        self.batch_normalization, self.layer_normalization
                    )
                    + "\n'before' and 'after' are the only acceptable values for"
                    + " batch_normalization and layer_normalization."
                )
            if self.dropout_rate > 0:
                y = tf.layers.dropout(y, rate=self.dropout_rate, training=self.training)
        self.y = self._dense(y, self.layer_size[-1], use_bias=self.use_bias)
        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True

    def _dense(self, inputs, units, activation=None, use_bias=True):
        # Cannot directly replace tf.layers.dense() with tf.keras.layers.Dense() due to
        # some differences. One difference is that tf.layers.dense() will add
        # regularizer loss to the collection REGULARIZATION_LOSSES, but
        # tf.keras.layers.Dense() will not. Hence, tf.losses.get_regularization_loss()
        # cannot be used for tf.keras.layers.Dense().
        # References:
        # - https://github.com/tensorflow/tensorflow/issues/21587
        # - https://www.tensorflow.org/guide/migrate
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.kernel_constraint,
        )

    @staticmethod
    def _dense_weightnorm(inputs, units, activation=None, use_bias=True):
        fan_in = inputs.shape[1]
        W = tf.Variable(tf.random_normal([fan_in, units], stddev=math.sqrt(2 / fan_in)))
        g = tf.Variable(tf.ones(units))
        W = tf.nn.l2_normalize(W, axis=0) * g
        y = tf.matmul(inputs, W)
        if use_bias:
            b = tf.Variable(tf.zeros(units))
            y += b
        if activation is not None:
            return activation(y)
        return y

    def _dense_batchnorm_v1(self, inputs, units):
        # FC - BN - activation
        y = self._dense(inputs, units, use_bias=False)
        y = tf.layers.batch_normalization(y, training=self.training)
        return self.activation(y)

    def _dense_batchnorm_v2(self, inputs, units):
        # FC - activation - BN
        y = self._dense(inputs, units, activation=self.activation)
        return tf.layers.batch_normalization(y, training=self.training)

    @staticmethod
    def _layer_normalization(inputs, elementwise_affine=True):
        """References:

        - https://tensorflow.google.cn/api_docs/python/tf/keras/layers/LayerNormalization?hl=en
        - https://github.com/taki0112/Group_Normalization-Tensorflow
        """

        with tf.variable_scope("layer_norm"):
            mean, var = tf.nn.moments(inputs, axes=[1], keepdims=True)

            if elementwise_affine:
                gamma = tf.Variable(
                    initial_value=tf.constant_initializer(1.0)(shape=[1, 1]),
                    trainable=True,
                    name="gamma",
                    dtype=config.real(tf),
                )
                beta = tf.Variable(
                    initial_value=tf.constant_initializer(0.0)(shape=[1, 1]),
                    trainable=True,
                    name="beta",
                    dtype=config.real(tf),
                )
            else:
                gamma, beta = None, None

            return tf.nn.batch_normalization(
                inputs, mean, var, offset=beta, scale=gamma, variance_epsilon=1e-3
            )

    def _dense_layernorm_v1(self, inputs, units):
        # FC - LN - activation
        y = self._dense(inputs, units, use_bias=False)
        y = self._layer_normalization(y)
        return self.activation(y)

    def _dense_layernorm_v2(self, inputs, units):
        # FC - activation - LN
        y = self._dense(inputs, units, activation=self.activation)
        return self._layer_normalization(y)


class PFNN(FNN):
    """Parallel fully-connected neural network that uses independent sub-networks for
    each network output.

    Args:
        layer_sizes: A nested list to define the architecture of the neural network (how
            the layers are connected). If `layer_sizes[i]` is int, it represent one
            layer shared by all the outputs; if `layer_sizes[i]` is list, it represent
            `len(layer_sizes[i])` sub-layers, each of which exclusively used by one
            output. Note that `len(layer_sizes[i])` should equal to the number of
            outputs. Every number specify the number of neurons of that layer.
    """

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
    ):
        super().__init__(
            layer_sizes,
            activation,
            kernel_initializer,
            regularization,
            dropout_rate,
            batch_normalization,
        )

    @timing
    def build(self):
        def layer_map(_y, layer_size, net):
            if net.batch_normalization is None:
                _y = net._dense(_y, layer_size, activation=net.activation)
            elif net.batch_normalization == "before":
                _y = net._dense_batchnorm_v1(_y, layer_size)
            elif net.batch_normalization == "after":
                _y = net._dense_batchnorm_v2(_y, layer_size)
            else:
                raise ValueError("batch_normalization")
            if net.dropout_rate > 0:
                _y = tf.layers.dropout(_y, rate=net.dropout_rate, training=net.training)
            return _y

        print("Building feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        # hidden layers
        for i_layer in range(len(self.layer_size) - 2):
            if isinstance(self.layer_size[i_layer + 1], (list, tuple)):
                if isinstance(y, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    if len(self.layer_size[i_layer + 1]) != len(
                        self.layer_size[i_layer]
                    ):
                        raise ValueError(
                            "Number of sub-layers should be the same when feed-forwarding"
                        )
                    y = [
                        layer_map(y[i_net], self.layer_size[i_layer + 1][i_net], self)
                        for i_net in range(len(self.layer_size[i_layer + 1]))
                    ]
                else:
                    # e.g. 64 -> [8, 8, 8]
                    y = [
                        layer_map(y, self.layer_size[i_layer + 1][i_net], self)
                        for i_net in range(len(self.layer_size[i_layer + 1]))
                    ]
            else:
                # e.g. 64 -> 64
                y = layer_map(y, self.layer_size[i_layer + 1], self)
        # output layers
        if isinstance(y, (list, tuple)):
            # e.g. [3, 3, 3] -> 3
            if len(self.layer_size[-2]) != self.layer_size[-1]:
                raise ValueError(
                    "Number of sub-layers should be the same as number of outputs"
                )
            y = [self._dense(y[i_net], 1) for i_net in range(len(y))]
            self.y = tf.concat(y, axis=1)
        else:
            self.y = self._dense(y, self.layer_size[-1])

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True
