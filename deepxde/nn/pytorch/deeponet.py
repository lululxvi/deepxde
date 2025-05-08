__all__ = ["DeepONet", "DeepONetCartesianProd", "PODDeepONet"]

import torch

from .fnn import FNN
from .nn import NN
from .. import activations
from ..deeponet_strategy import (
    SingleOutputStrategy,
    IndependentStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)


class DeepONet(NN):
    """Deep operator network.

    `Lu et al. Learning nonlinear operators via DeepONet based on the universal
    approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularization

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError(
                    "num_outputs is set to 1, but multi_output_strategy is not None."
                )
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                'Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,bi->b", x_func, x_loc)
        y = torch.unsqueeze(y, dim=1)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.concat(ys, dim=1)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularization

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError(
                    "num_outputs is set to 1, but multi_output_strategy is not None."
                )
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                'Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
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
        self.regularizer = regularization
        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

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

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self.trunk is None:
            # POD only
            x = torch.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = torch.einsum(
                "bi,ni->bn", x_func, torch.concat((self.pod_basis, x_loc), 1)
            )
            x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
