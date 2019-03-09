from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import activations
from . import initializers
from . import regularizers
from .. import config
from ..utils import timing


class MfNN(object):
    """Multifidelity neural networks
    """

    def __init__(
        self,
        layer_size_low_fidelity,
        layer_size_high_fidelity,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        self.layer_size_lo = layer_size_low_fidelity
        self.layer_size_hi = layer_size_high_fidelity
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self.training = None
        self.dropout = None
        self.data_id = None
        self.X = None
        self.y_lo = None
        self.y_hi = None
        self.target_lo = None
        self.target_hi = None

        self.build()

    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return [self.y_lo, self.y_hi]

    @property
    def targets(self):
        return [self.target_lo, self.target_hi]

    @timing
    def build(self):
        print("Building multifidelity neural network...")
        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.bool)
        self.data_id = tf.placeholder(tf.uint8)
        self.X = tf.placeholder(config.real(tf), [None, self.layer_size_lo[0]])

        # Low fidelity
        y = self.X
        for i in range(len(self.layer_size_lo) - 2):
            y = self.dense(
                y,
                self.layer_size_lo[i + 1],
                activation=self.activation,
                regularizer=self.regularizer,
            )
        self.y_lo = self.dense(y, self.layer_size_lo[-1], regularizer=self.regularizer)

        # High fidelity
        X_hi = tf.concat([self.X, self.y_lo], 1)
        # Linear
        y_hi_l = self.dense(X_hi, self.layer_size_hi[-1])
        # Nonlinear
        y = X_hi
        for i in range(len(self.layer_size_hi) - 1):
            y = self.dense(
                y,
                self.layer_size_hi[i],
                activation=self.activation,
                regularizer=self.regularizer,
            )
        y_hi_nl = self.dense(
            y, self.layer_size_hi[-1], use_bias=False, regularizer=self.regularizer
        )
        # Linear + nonlinear
        alpha = tf.Variable(0, dtype=config.real(tf))
        alpha = activations.get("tanh")(alpha)
        self.y_hi = y_hi_l + alpha * y_hi_nl

        self.target_lo = tf.placeholder(config.real(tf), [None, self.layer_size_lo[-1]])
        self.target_hi = tf.placeholder(config.real(tf), [None, self.layer_size_hi[-1]])

    def dense(self, inputs, units, activation=None, use_bias=True, regularizer=None):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
        )
