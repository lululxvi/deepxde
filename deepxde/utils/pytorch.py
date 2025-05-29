"""Utilities of pytorch."""

import torch


class LLAAF(torch.nn.Module):
    def __init__(self, activation, n):
        super().__init__()
        self.activation = activation
        self.n = n
        self.a = torch.nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return self.activation(self.n * self.a * x)
