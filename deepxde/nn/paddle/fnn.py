import paddle

from .nn import NN
from .. import activations
from .. import initializers


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
