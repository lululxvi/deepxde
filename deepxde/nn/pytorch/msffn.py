import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import NN
from .. import activations
from .. import initializers


class MsFFN(NN):
    """Multi-scale fourier feature networks.

    Args:
        sigmas: List of standard deviation of the distribution of fourier feature
            embeddings.

    References:
        `S. Wang, H. Wang, & P. Perdikaris. On the eigenvector bias of Fourier feature
        networks: From regression to solving multi-scale PDEs with physics-informed
        neural networks. Computer Methods in Applied Mechanics and Engineering, 384,
        113938, 2021 <https://doi.org/10.1016/j.cma.2021.113938>`_.
    """

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        sigmas,
        dropout_rate=0,
    ):
        super().__init__()
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.sigmas = sigmas  # list or tuple
        self.fourier_feature_weights = None
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.b = nn.ParameterList()
        for sigma in self.sigmas:
            param = nn.Parameter(
                torch.empty(layer_sizes[0], layer_sizes[1] // 2)
            )
            nn.init.normal_(param, mean=0.0, std=sigma)
            param.requires_grad_(False)  #freeze parameters
            self.b.append(param)

        self.linears = nn.ModuleList()
        for i in range(2, len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            initializer(linear.weight)
            initializer_zero(linear.bias)
            self.linears.append(linear)

        self._dense = nn.Linear(layer_sizes[-2] * 2, layer_sizes[-1])
        initializer(self._dense.weight)
        initializer_zero(self._dense.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        # fourier feature layer
        yb = [
            self._fourier_feature_forward(x, self.b[i])
            for i in range(len(self.sigmas))
        ]
        y = [elem[0] for elem in yb]
        self.fourier_feature_weights = [elem[1] for elem in yb]

        # fully-connected layers
        y = [self._fully_connected_forward(_y) for _y in y]

        # concatenate all the fourier features
        y = torch.cat(y, dim=1)
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

    def _fourier_feature_forward(self, y, b):
        # y: [N, in_dim], b: [in_dim, layer_sizes[1] // 2]
        proj = torch.matmul(y, b)
        y_ff = torch.cat(
            [torch.cos(proj), torch.sin(proj)],
            dim=1,
        )
        return y_ff, b

    def _fully_connected_forward(self, y):
        for linear in self.linears:
            y = self.activation(linear(y))
            if self.dropout_rate > 0:
                y = F.dropout(y, p=self.dropout_rate, training=self.training)
        return y


class STMsFFN(MsFFN):
    """Spatio-temporal multi-scale fourier feature networks.

    References:
        `S. Wang, H. Wang, & P. Perdikaris. On the eigenvector bias of Fourier feature
        networks: From regression to solving multi-scale PDEs with physics-informed
        neural networks. Computer Methods in Applied Mechanics and Engineering, 384,
        113938, 2021 <https://doi.org/10.1016/j.cma.2021.113938>`_.
    """

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        sigmas_x,
        sigmas_t,
        dropout_rate=0,
    ):
        super().__init__(
            layer_sizes,
            activation,
            kernel_initializer,
            [],
            dropout_rate,
        )
        for sigma in sigmas_x:
            param = nn.Parameter(
                torch.empty(layer_sizes[0] - 1, layer_sizes[1] // 2)
            )
            nn.init.normal_(param, mean=0.0, std=sigma)
            param.requires_grad_(False) # freeze parameters
            self.b.append(param)

        for sigma in sigmas_t:
            param = nn.Parameter(
                torch.empty(1, layer_sizes[1] // 2)
            )
            nn.init.normal_(param, mean=0.0, std=sigma)
            param.requires_grad_(False) # freeze parameters
            self.b.append(param)

        self.sigmas_x = sigmas_x
        self.sigmas_t = sigmas_t

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            # The last column should be function of t.
            x = self._input_transform(x)

        # fourier feature layer for space (x)
        yb_x = [
            self._fourier_feature_forward(x[:, :-1], self.b[i])
            for i in range(len(self.sigmas_x))
        ]
        # fourier feature layer for time (t)
        offset = len(self.sigmas_x)
        yb_t = [
            self._fourier_feature_forward(x[:, -1:], self.b[offset + i])
            for i in range(len(self.sigmas_t))
        ]
        self.fourier_feature_weights = [elem[1] for elem in (yb_x + yb_t)]

        # fully-connected layers (reuse)
        y_x = [self._fully_connected_forward(_yb[0]) for _yb in yb_x]
        y_t = [self._fully_connected_forward(_yb[0]) for _yb in yb_t]

        # point-wise multiplication layer
        y = [torch.mul(_y_x, _y_t) for _y_x in y_x for _y_t in y_t]

        # concatenate all the fourier features
        y = torch.cat(y, dim=1)
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)

        return y
