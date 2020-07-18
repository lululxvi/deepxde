from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from . import activations
from . import initializers
from . import regularizers
from .map import Map
from .. import config
from ..backend import tf
from ..utils import timing


class PFNN(Map):
    """Parallel Feed-forward neural networks.
    """

    def __init__(
            self,
            layer_size,
            activation,
            kernel_initializer,
            regularization=None,
            dropout_rate=0,
            batch_normalization=None,
    ):
        super(PFNN, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

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

        def layer_map(_y, layer_size, net):
            if net.batch_normalization is None:
                _y = net.dense(_y, layer_size, activation=net.activation)
            elif net.batch_normalization == "before":
                _y = net.dense_batchnorm_v1(_y, layer_size)
            elif net.batch_normalization == "after":
                _y = net.dense_batchnorm_v2(_y, layer_size)
            else:
                raise ValueError("batch_normalization")
            if net.dropout_rate > 0:
                _y = tf.layers.dropout(_y, rate=net.dropout_rate, training=net.dropout)

            return _y

        print("Building feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        # hidden layers
        for i_layer in range(len(self.layer_size) - 2):
            if type(self.layer_size[i_layer + 1]) is list:
                if type(y) is list:
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    assert len(self.layer_size[i_layer + 1]) == len(self.layer_size[i_layer])
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
        if type(y) is list:
            # e.g. [3, 3, 3] -> [3]
            y = [self.dense(y[i_net], 1) for i_net in range(len(y))]
            self.y = tf.concat(y, axis=1)
        else:
            self.y = self.dense(y, self.layer_size[-1])

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True

    def dense(self, inputs, units, activation=None, use_bias=True):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
        )

    @staticmethod
    def dense_weightnorm(inputs, units, activation=None, use_bias=True):
        shape = inputs.get_shape().as_list()
        fan_in = shape[1]
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

    def dense_batchnorm_v1(self, inputs, units):
        # FC - BN - activation
        y = self.dense(inputs, units, use_bias=False)
        y = tf.layers.batch_normalization(y, training=self.training)
        return self.activation(y)

    def dense_batchnorm_v2(self, inputs, units):
        # FC - activation - BN
        y = self.dense(inputs, units, activation=self.activation)
        return tf.layers.batch_normalization(y, training=self.training)
