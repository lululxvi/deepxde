from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import activations
from .. import initializers
from ...backend import tf


class FNN(tf.keras.Model):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super(FNN, self).__init__()

        self.denses = []
        activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        for units in layer_sizes[1:-1]:
            self.denses.append(
                tf.keras.layers.Dense(
                    units, activation=activation, kernel_initializer=initializer
                )
            )
        self.denses.append(
            tf.keras.layers.Dense(layer_sizes[-1], kernel_initializer=initializer)
        )

    def call(self, inputs, training=False):
        for f in self.denses:
            inputs = f(inputs)
        return inputs
