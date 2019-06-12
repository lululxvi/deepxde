from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from . import activations
from . import initializers
from . import regularizers
from .map import Map
from .. import config
from ..utils import timing


class FNN(Map):
    """feed-forward neural networks
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
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        super(FNN, self).__init__()

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
        print("\nBuilding feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        for i in range(len(self.layer_size) - 2):
            if self.batch_normalization is None:
                y = self.dense(y, self.layer_size[i + 1], activation=self.activation)
            elif self.batch_normalization == "before":
                y = self.dense_batchnorm_v1(y, self.layer_size[i + 1])
            elif self.batch_normalization == "after":
                y = self.dense_batchnorm_v2(y, self.layer_size[i + 1])
            else:
                raise ValueError("batch_normalization")
            if self.dropout_rate > 0:
                y = tf.layers.dropout(y, rate=self.dropout_rate, training=self.dropout)
        self.y = self.dense(y, self.layer_size[-1])

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])

    def outputs_modify(self, modify):
        self.y = modify(self.inputs, self.outputs)

    def dense(self, inputs, units, activation=None, use_bias=True):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
        )

    def dense_weightnorm(self, inputs, units, activation=None, use_bias=True):
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
