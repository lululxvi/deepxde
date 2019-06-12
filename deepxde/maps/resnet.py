from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import activations
from . import initializers
from . import regularizers
from .map import Map
from .. import config
from ..utils import timing


class ResNet(Map):
    """Residual neural network
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_neurons,
        num_blocks,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_blocks = num_blocks
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        super(ResNet, self).__init__()

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
        print("Building residual neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.input_size])

        y = self.dense(self.x, self.num_neurons, activation=self.activation)
        for _ in range(self.num_blocks):
            y = self.residual_block(y)
        self.y = self.dense(y, self.output_size)

        self.y_ = tf.placeholder(config.real(tf), [None, self.output_size])

    def dense(self, inputs, units, activation=None, use_bias=True):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
        )

    def residual_block(self, inputs):
        """A residual block in ResNet.
        """
        shape = inputs.get_shape().as_list()
        units = shape[1]

        x = self.dense(inputs, units, activation=self.activation)
        x = self.dense(x, units)

        x += inputs
        x = self.activation(x)

        return x
