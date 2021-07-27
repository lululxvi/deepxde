import torch


class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super(FNN, self).__init__()
        activation = torch.nn.ReLU()

        layers = []
        for i in range(1, len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(activation)
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
