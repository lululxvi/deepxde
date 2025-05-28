import torch

from .nn import NN
from .. import activations
from .. import initializers
from ... import config


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=None
    ):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
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
            used by one output. Every list in `layer_sizes` must have the same length
            (= number of subnetworks). If the last element of `layer_sizes` is an int
            preceded by a list, it must be equal to the number of subnetworks: all
            subnetworks have an output size of 1 and are then concatenated. If the last
            element is a list, it specifies the output size for each subnetwork before
            concatenation.
        activation: Activation function.
        kernel_initializer: Initializer for the kernel weights.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        if len(layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("input size must be integer")

        # Determine the number of subnetworks from the first list layer
        list_layers = [
            layer for layer in layer_sizes if isinstance(layer, (list, tuple))
        ]
        if not list_layers:
            raise ValueError(
                "No list layers found; use FNN instead of PFNN for single subnetwork."
            )
        n_subnetworks = len(list_layers[0])
        for layer in list_layers:
            if len(layer) != n_subnetworks:
                raise ValueError(
                    "All list layers must have the same length as the first list layer."
                )

        # Validate output layer if preceded by a list layer
        if (
            isinstance(layer_sizes[-1], int)
            and isinstance(layer_sizes[-2], (list, tuple))
            and layer_sizes[-1] != n_subnetworks
        ):
            raise ValueError(
                "If last layer is an int and previous is a list, the int must equal the number of subnetworks."
            )

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear

        self.layers = torch.nn.ModuleList()

        # Process hidden layers (excluding the output layer)
        for i in range(1, len(layer_sizes) - 1):
            prev_layer = layer_sizes[i - 1]
            curr_layer = layer_sizes[i]

            if isinstance(curr_layer, (list, tuple)):
                # Parallel layer
                if isinstance(prev_layer, (list, tuple)):
                    # Previous is parallel: each subnetwork input is previous subnetwork output
                    sub_layers = [
                        make_linear(prev_layer[j], curr_layer[j])
                        for j in range(n_subnetworks)
                    ]
                else:
                    # Previous is shared: all subnetworks take the same input
                    sub_layers = [
                        make_linear(prev_layer, curr_layer[j])
                        for j in range(n_subnetworks)
                    ]
                self.layers.append(torch.nn.ModuleList(sub_layers))
            else:
                # Shared layer
                if isinstance(prev_layer, (list, tuple)):
                    # Previous is parallel: concatenate outputs
                    input_size = sum(prev_layer)
                else:
                    input_size = prev_layer
                self.layers.append(make_linear(input_size, curr_layer))

        # Process output layer
        prev_output_layer = layer_sizes[-2]
        output_layer = layer_sizes[-1]

        if isinstance(output_layer, (list, tuple)):
            if isinstance(prev_output_layer, (list, tuple)):
                # Each subnetwork input is corresponding previous output
                output_layers = [
                    make_linear(prev_output_layer[j], output_layer[j])
                    for j in range(n_subnetworks)
                ]
            else:
                # All subnetworks take the same shared input
                output_layers = [
                    make_linear(prev_output_layer, output_layer[j])
                    for j in range(n_subnetworks)
                ]
            self.layers.append(torch.nn.ModuleList(output_layers))
        else:
            if isinstance(prev_output_layer, (list, tuple)):
                # Each subnetwork outputs 1 and concatenates to output_layer size
                output_layers = [
                    make_linear(prev_output_layer[j], 1) for j in range(n_subnetworks)
                ]
                self.layers.append(torch.nn.ModuleList(output_layers))
            else:
                # Shared output layer
                self.layers.append(make_linear(prev_output_layer, output_layer))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                # Parallel layer processing
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                # Shared layer processing (concatenate if necessary)
                if isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = self.activation(layer(x))

        # Output layer processing
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
