# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Union, Callable, Sequence, Optional

import brainstate as bst
import brainunit as u

from deepxde.pinnx.utils import get_activation
from .base import NN


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Union[str, Callable, Sequence[str], Sequence[Callable]],
        kernel_initializer: bst.init.Initializer = bst.init.KaimingUniform(),
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__(input_transform=input_transform,
                         output_transform=output_transform)

        # activations
        if isinstance(activation, (list, tuple)):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError("Total number of activation functions do not match with "
                                 "sum of hidden layers and output layer!")
            self.activation = list(map(get_activation, activation))
        else:
            self.activation = get_activation(activation)

        # layers
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(bst.nn.Linear(layer_sizes[i - 1], layer_sizes[i], w_init=kernel_initializer))

        # output transform
        if output_transform is not None:
            self.apply_output_transform(output_transform)

    def update(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.layers[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.layers[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """
    Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Union[str, Callable, Sequence[str], Sequence[Callable]],
        kernel_initializer: bst.init.Initializer = bst.init.KaimingUniform(),
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__(
            input_transform=input_transform,
            output_transform=output_transform
        )
        self.activation = get_activation(activation)

        if len(layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("input size must be integer")
        if not isinstance(layer_sizes[-1], int):
            raise ValueError("output size must be integer")

        n_output = layer_sizes[-1]

        self.layers = []
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "number of sub-layers should equal number of network outputs"
                    )
                if isinstance(prev_layer_size, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    self.layers.append(
                        [
                            bst.nn.Linear(prev_layer_size[j], curr_layer_size[j], w_init=kernel_initializer)
                            for j in range(n_output)
                        ]
                    )
                else:  # e.g. 64 -> [8, 8, 8]
                    self.layers.append(
                        [
                            bst.nn.Linear(prev_layer_size, curr_layer_size[j], w_init=kernel_initializer)
                            for j in range(n_output)
                        ]
                    )
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError(
                        "cannot rejoin parallel subnetworks after splitting"
                    )
                self.layers.append(bst.nn.Linear(prev_layer_size, curr_layer_size, w_init=kernel_initializer))

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.layers.append(
                [bst.nn.Linear(layer_sizes[-2][j], 1, w_init=kernel_initializer) for j in range(n_output)]
            )
        else:
            self.layers.append(bst.nn.Linear(layer_sizes[-2], n_output, w_init=kernel_initializer))

    def update(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, list):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))

        # output layers
        if isinstance(x, list):
            x = u.math.concatenate([f(x_) for f, x_ in zip(self.layers[-1], x)], axis=-1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
