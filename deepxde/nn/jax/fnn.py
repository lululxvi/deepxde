from typing import Any

import jax
from flax import linen as nn

from .nn import NN


class FNN(NN):
    """Fully-connected neural network"""

    layer_sizes: Any = None
    activation: Any = None
    kernel_initializer: Any = None

    def setup(self):
        # TODO: implement get activation, get initializer
        self._activation = jax.nn.tanh
        kernel_initializer = jax.nn.initializers.glorot_normal()
        initializer = jax.nn.initializers.zeros

        self.denses = [
            nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )
            for unit in self.layer_sizes[1:]
        ]

    def __call__(self, inputs, training=True):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.denses[:-1]:
            x = self._activation(linear(x))
        x = self.denses[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
