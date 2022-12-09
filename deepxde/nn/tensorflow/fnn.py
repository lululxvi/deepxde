from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ...backend import tf


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self.denses = []
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            activation = list(map(activations.get, activation))
        else:
            activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        for j, units in enumerate(layer_sizes[1:-1]):
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(
                        activation[j]
                        if isinstance(activation, list)
                        else activation
                    ),
                    kernel_initializer=initializer,
                    kernel_regularizer=self.regularizer,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                kernel_initializer=initializer,
                kernel_regularizer=self.regularizer,
            )
        )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses:
            y = f(y, training=training)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


class PFNN(NN):
    """Parallel fully-connected neural network that uses independent sub-networks for
    each network output.

    Args:
        layer_sizes: A nested list to define the architecture of the neural network (how
            the layers are connected). If `layer_sizes[i]` is int, it represent one
            layer shared by all the outputs; if `layer_sizes[i]` is list, it represent
            `len(layer_sizes[i])` sub-layers, each of which exclusively used by one
            output. Note that `len(layer_sizes[i])` should equal to the number of
            outputs. Every number specify the number of neurons of that layer.
    """

    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=None
    ):
        super().__init__()
        activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        n_output = layer_sizes[-1]
        self.denses = []
        # hidden layers
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            # Non-Shared layers
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "number of sub-layers should equal number of network outputs"
                    )
                # e.g. [8, 8, 8] -> [16, 16, 16] or 64 -> [8, 8, 8]
                self.denses.append(
                    [
                        tf.keras.layers.Dense(
                            units,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=self.regularizer,
                        )
                        for units in curr_layer_size
                    ]
                )
            # Shared layers
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError(
                        "cannot rejoin parallel subnetworks after splitting"
                    )
                self.denses.append(
                    tf.keras.layers.Dense(
                        curr_layer_size,
                        activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=self.regularizer,
                    )
                )

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.denses.append(
                [
                    tf.keras.layers.Dense(
                        1,
                        kernel_initializer=initializer,
                        kernel_regularizer=self.regularizer,
                    )
                    for _ in range(n_output)
                ]
            )
        else:
            self.denses.append(
                tf.keras.layers.Dense(
                    n_output,
                    kernel_initializer=initializer,
                    kernel_regularizer=self.regularizer,
                )
            )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)

        # hidden layers
        for layer in self.denses[:-1]:
            if isinstance(layer, list):
                if isinstance(y, list):
                    y = [f(x, training=training) for f, x in zip(layer, y)]
                else:
                    y = [f(y, training=training) for f in layer]
            else:
                y = layer(y, training=training)

        # output layers
        if isinstance(y, list):
            y = [f(x, training=training) for f, x in zip(self.denses[-1], y)]
            y = tf.concat(y, 1)
        else:
            y = self.denses[-1](y, training=training)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
