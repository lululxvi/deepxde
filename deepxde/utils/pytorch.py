"""Utilities of pytorch."""

import torch


class LLAAF(torch.nn.Module):
    """Pytorch implementation of layer-wise locally adaptive
    activation functions (L-LAAF).

    Args:
        activation: The activation function to use.
        n: The scaling factor.

    Examples:

    To define a L-LAAF ReLU with the scaling factor ``n = 10``:

    .. code-block:: python

        n = 10
        llaaf = LLAAF(torch.relu, n)

    References:
        `A. D. Jagtap, K. Kawaguchi, & G. E. Karniadakis. Locally adaptive activation
        functions with slope recovery for deep and physics-informed neural networks.
        Proceedings of the Royal Society A, 476(2239), 20200334, 2020
        <https://doi.org/10.1098/rspa.2020.0334>`_.
    """

    def __init__(self, activation, n):
        super().__init__()
        self.activation = activation
        self.n = n
        self.a = torch.nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return self.activation(self.n * self.a * x)
