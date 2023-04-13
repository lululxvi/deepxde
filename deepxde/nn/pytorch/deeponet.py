import torch

from .fnn import FNN
from .nn import NN
from .. import activations


class DeepONet(NN):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        # Add bias
        x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PODDeepONet(NN):
    """Deep operator network with proper orthogonal decomposition (POD) for dataset in
    the format of Cartesian product.

    Args:
        pod_basis: POD basis used in the trunk net.
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network. If ``None``, then only use POD basis as the trunk net.

    References:
        `L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A
        comprehensive and fair comparison of two neural operators (with practical
        extensions) based on FAIR data. arXiv preprint arXiv:2111.05512, 2021
        <https://arxiv.org/abs/2111.05512>`_.
    """

    def __init__(
        self,
        pod_basis,
        layer_sizes_branch,
        activation,
        kernel_initializer,
        layer_sizes_trunk=None,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        x_func = self.branch(x_func)
        if self.trunk is None:
            # POD only
            x = torch.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = torch.einsum("bi,ni->bn", x_func, torch.cat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
