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
    """Parallel fully-connected network that can have multiple subnetworks.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers. If a list layer is followed by
            an int layer, the output of each sub-layer will be concatenated and fed into
            the int layer. Two consecutive list layers must have the same length.
            If the last layer is a list, it specifies the output size for each subnetwork
            before concatenation. If the last layer is an int and preceded by a list layer,
            the output size must be equal to the number of subnetworks (=len(layer_sizes[-2])).
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

        if not any(
            isinstance(layer_size, (list, tuple)) for layer_size in self.layer_sizes
        ):
            raise ValueError(
                "no list in layer_sizes, use FNN instead of PFNN for single subnetwork"
            )

        self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        def make_dense(unit):
            return nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )

        denses = []
        for i in range(1, len(self.layer_sizes) - 1):
            prev_layer_size = self.layer_sizes[i - 1]
            curr_layer_size = self.layer_sizes[i]
            if isinstance(curr_layer_size, int):
                denses.append(make_dense(curr_layer_size))
            else:
                if isinstance(prev_layer_size, (list, tuple)) and len(
                    prev_layer_size
                ) != len(curr_layer_size):
                    raise ValueError(
                        "number of sub-networks should be the same between two consecutive list layers"
                    )
                else:
                    denses.append([make_dense(unit) for unit in curr_layer_size])

        if isinstance(self.layer_sizes[-1], int):
            if isinstance(self.layer_sizes[-2], (list, tuple)):
                # if output layer size is an int and the previous layer size is a list,
                # the output size must be equal to the number of subnetworks (=len(layer_sizes[-2])),
                # then all subnetworks have an output size of 1 and are then concatenated
                if len(self.layer_sizes[-2]) != self.layer_sizes[-1]:
                    raise ValueError(
                        "if layer_sizes[-1] is an int and layer_sizes[-2] is a list, len(layer_sizes[-2]) must be equal to layer_sizes[-1]"
                    )
                else:
                    denses.append([make_dense(1) for _ in range(self.layer_sizes[-1])])
            else:
                denses.append(make_dense(self.layer_sizes[-1]))
        else:
            # if the output layer size is a list, it specifies the output size for each subnetwork before concatenation
            denses.append([make_dense(unit) for unit in self.layer_sizes[-1]])

        self.denses = denses  # can't assign directly to self.denses because linen list attributes are converted to tuple
        # see https://github.com/google/flax/issues/524

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.denses[:-1]:
            if isinstance(layer, (list, tuple)):
                if isinstance(x, list):
                    x = [self._activation(dense(x_)) for dense, x_ in zip(layer, x)]
                else:
                    x = [self._activation(dense(x)) for dense in layer]
            else:
                if isinstance(x, list):
                    x = jnp.concatenate(x, axis=0 if x[0].ndim == 1 else 1)
                x = self._activation(layer(x))

        # output layers
        if isinstance(x, list):
            if len(x) != len(self.denses[-1]):
                raise ValueError(
                    "number of sub-networks should be the same between two consecutive list layers"
                )
            x = jnp.concatenate(
                [f(x_) for f, x_ in zip(self.denses[-1], x)],
                axis=0 if x[0].ndim == 1 else 1,
            )
        else:
            if isinstance(self.denses[-1], (list, tuple)):
                x = jnp.concatenate(
                    [f(x) for f in self.denses[-1]], axis=0 if x.ndim == 1 else 1
                )
            else:
                x = self.denses[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
