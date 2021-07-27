import torch

from .. import activations
from .. import initializers


class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super(FNN, self).__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
