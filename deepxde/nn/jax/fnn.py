from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers
from ...utils import list_handler


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
            used by one output. Every layer_sizes[i] list must have the same length
            (= number of subnetworks). If the last element of `layer_sizes` is an int
            preceded by a list, it must be equal to the number of subnetworks: all
            subnetworks have an output size of 1 and are then concatenated. If the last
            element is a list, it specifies the output size for each subnetwork before
            concatenation.
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

        list_layer = [
            layer_size
            for layer_size in self.layer_sizes
            if isinstance(layer_size, (list, tuple))
        ]
        if not list_layer:  # if there is only one subnetwork (=FNN)
            raise ValueError(
                "no list in layer_sizes, use FNN instead of PFNN for single subnetwork"
            )
        n_subnetworks = len(list_layer[0])
        if not all(len(sublist) == n_subnetworks for sublist in list_layer):
            raise ValueError(
                "all layer_size lists must have the same length(=number of subnetworks)"
            )
        if (
            isinstance(self.layer_sizes[-1], int)
            and n_subnetworks != self.layer_sizes[-1]
            and isinstance(self.layer_sizes[-2], (list, tuple))
        ):
            raise ValueError(
                "if the last element of layer_sizes is an int preceded by a list, it must be equal to the number of subnetworks"
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

        denses = [
            (
                make_dense(unit)
                if isinstance(unit, int)
                else [make_dense(unit[j]) for j in range(n_subnetworks)]
            )
            for unit in self.layer_sizes[1:-1]
        ]

        if isinstance(self.layer_sizes[-1], int):
            if isinstance(self.layer_sizes[-2], (list, tuple)):
                # if output layer size is an int and the previous layer size is a list,
                # the output size must be equal to the number of subnetworks:
                # all subnetworks have an output size of 1 and are then concatenated
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
            x = jnp.concatenate(
                [f(x_) for f, x_ in zip(self.denses[-1], x)],
                axis=0 if x[0].ndim == 1 else 1,
            )
        else:
            x = self.denses[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

@list_handler
class SPINN(NN):
    layer_sizes: Any
    activation: Any
    kernel_initializer: Any

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        self.r = self.layer_sizes[-2]  # rank of the approximated tensor
        self.out_dim = self.layer_sizes[-1]  # output dimension

    @nn.compact
    def __call__(self, inputs, training=False):
        if (
            inputs.ndim == 1
        ):  # jax compute grad pointwise (n_dim,), so we need to reshape to (1, n_dim)
            inputs = inputs.reshape(1, -1)
        x = inputs

        kernel_initializer = initializers.get(self.kernel_initializer)
        if self._input_transform is not None:
            x = self._input_transform(x)
        x, outputs, pred = [x[:, 0:1], x[:, 1:2]], [], []
        # if self.mlp == 'mlp':
        for x_ in x:
            for fs in self.layer_sizes[:-2]:
                x_ = nn.Dense(fs, kernel_init=kernel_initializer)(x_)
                x_ = nn.activation.tanh(x_)
            x_ = nn.Dense(self.r * self.out_dim, kernel_init=kernel_initializer)(x_)
            outputs += [x_]
        # else:
        #     for X in inputs:
        #         U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
        #         V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
        #         H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
        #         for fs in self.features[:-1]:
        #             Z = nn.Dense(fs, kernel_init=init)(H)
        #             Z = nn.activation.tanh(Z)
        #             H = (jnp.ones_like(Z)-Z)*U + Z*V
        #         H = nn.Dense(self.r, kernel_init=init)(H)
        #         outputs += [H]
        for i in range(self.out_dim):
            # pred += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [
                jnp.dot(
                    outputs[0][:, self.r * i : self.r * (i + 1)],
                    outputs[-1][:, self.r * i : self.r * (i + 1)].T,
                ).ravel()
            ]
            # pred += [jnp.sum(outputs[0][:,self.r*i:self.r*(i+1)] * outputs[1][:,self.r*i:self.r*(i+1)], axis=-1)]
        pred = jnp.stack(pred, axis=1)
        if self._output_transform is not None:
            pred = self._output_transform(inputs, pred)
        return (
            pred.squeeze()
        )  # JAX compute grad pointwise (n_dim,), so we need to squeeze it back to (n_dim,)
