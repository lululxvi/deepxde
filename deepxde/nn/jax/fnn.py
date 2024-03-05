from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers


class FNN(NN):
    """Fully-connected neural network."""

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        # TODO: implement get regularizer
        if isinstance(self.activation, list):
            if not (len(self.layer_sizes) - 1) == len(self.activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self._activation = list(map(activations.get, self.activation))
        else:
            self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        self.denses = [
            nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )
            for unit in self.layer_sizes[1:]
        ]

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.denses[:-1]):
            x = (
                self._activation[j](linear(x))
                if isinstance(self._activation, list)
                else self._activation(linear(x))
            )
        x = self.denses[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        if len(self.layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(self.layer_sizes[0], int):
            raise ValueError("input size must be integer")
        if not isinstance(self.layer_sizes[-1], int):
            raise ValueError("output size must be integer")

        n_output = self.layer_sizes[-1]

        self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        def make_dense(n_input, n_output):
            return nn.Dense(
                n_output,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )

        layers = [
            make_dense(self.layer_sizes[i - 1], self.layer_sizes[i])
            if isinstance(self.layer_sizes[i], int)
            else [
                make_dense(self.layer_sizes[i - 1], self.layer_sizes[i][j])
                for j in range(n_output)
            ]
            for i in range(1, len(self.layer_sizes) - 1)
        ]

        if isinstance(self.layer_sizes[-2], (list, tuple)):
            layers.append(
                [
                    make_dense(self.layer_sizes[-2][j], 1)
                    for j in range(n_output)
                ]
            )
        else:
            layers.append(make_dense(self.layer_sizes[-2], n_output))
        self.layers = layers #can't assign directly to self.layers because linen list attributes are converted to tuple, see https://github.com/google/flax/issues/524

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, tuple):
                if isinstance(x, list):
                    x = [
                        self._activation(layer[f](x_))
                        for f, x_ in zip(range(len(layer)), x)
                    ]
                else:
                    x = [self._activation(layer[f](x)) for f in range(len(layer))]
            else:
                x = self._activation[layer](x)

        # output layers
        if isinstance(x, list):
            if x[0].ndim == 1:
                x = jnp.concatenate([f(x_) for f, x_ in zip(self.layers[-1], x)], axis=0)
            else:
                x = jnp.concatenate([f(x_) for f, x_ in zip(self.layers[-1], x)], axis=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x