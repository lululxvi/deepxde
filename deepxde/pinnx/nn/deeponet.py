# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

from typing import Union, Callable, Sequence, Dict, Optional

import brainstate as bst
import brainunit as u

from deepxde.nn.deeponet_strategy import (
    DeepONetStrategy,
    SingleOutputStrategy,
    IndependentStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy
)
from deepxde.pinnx.utils import get_activation
from .base import NN
from .fnn import FNN

strategies = {
    None: SingleOutputStrategy,
    "independent": IndependentStrategy,
    "split_both": SplitBothStrategy,
    "split_branch": SplitBranchStrategy,
    "split_trunk": SplitTrunkStrategy,
}

__all__ = ["DeepONet", "DeepONetCartesianProd", "PODDeepONet"]


class DeepONet(NN):
    """
    Deep operator network.

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
        layer_sizes_branch: Sequence[int],
        layer_sizes_trunk: Sequence[int],
        activation: Union[str, Callable, Dict[str, str], Dict[str, Callable]],
        kernel_initializer: bst.init.Initializer = bst.init.KaimingUniform(),
        num_outputs: int = 1,
        multi_output_strategy=None,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__(input_transform=input_transform,
                         output_transform=output_transform)

        # activation function
        if isinstance(activation, dict):
            self.activation_branch = get_activation(activation["branch"])
            self.activation_trunk = get_activation(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = get_activation(activation)

        # initialize kernel
        self.kernel_initializer = kernel_initializer

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1, but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                  'Use "independent" as the multi_output_strategy.')
        self.multi_output_strategy: DeepONetStrategy = strategies[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(layer_sizes_branch,
                                                                   layer_sizes_trunk)
        self.b = bst.ParamState([0.0 for _ in range(self.num_outputs)])

    def build_branch_net(self, layer_sizes_branch) -> FNN:
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch,
                   self.activation_branch,
                   self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk) -> FNN:
        return FNN(layer_sizes_trunk,
                   self.activation_trunk,
                   self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = u.math.sum(x_func * x_loc, axis=-1, keepdims=True)
        y += self.b.value[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return u.math.concatenate(ys, axis=1)

    def update(self, inputs):
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
    """
    Deep operator network for dataset in the format of Cartesian product.

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
        layer_sizes_branch: Sequence[int],
        layer_sizes_trunk: Sequence[int],
        activation: Union[str, Callable, Dict[str, str], Dict[str, Callable]],
        kernel_initializer: bst.init.Initializer = bst.init.KaimingUniform(),
        num_outputs: int = 1,
        multi_output_strategy=None,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__(input_transform=input_transform,
                         output_transform=output_transform)
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = get_activation(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = get_activation(activation)
        self.kernel_initializer = kernel_initializer

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError("num_outputs is set to 1, but multi_output_strategy is not None.")
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                  'Use "independent" as the multi_output_strategy.')
        self.multi_output_strategy = strategies[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(layer_sizes_branch,
                                                                   layer_sizes_trunk)
        self.b = bst.ParamState([0.0 for _ in range(self.num_outputs)])

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = u.math.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b.value[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return u.math.stack(ys, axis=2)

    def update(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x if x.ndim == 3 else x[..., None]


class PODDeepONet(NN):
    """
    Deep operator network with proper orthogonal decomposition (POD) for dataset in
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
        layer_sizes_branch: Sequence[int],
        activation: Union[str, Callable, Dict[str, str], Dict[str, Callable]],
        kernel_initializer: bst.init.Initializer = bst.init.KaimingUniform(),
        layer_sizes_trunk: Sequence[int] = None,
        regularization=None,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__(input_transform=input_transform,
                         output_transform=output_transform)
        self.regularization = regularization  # TODO: currently unused
        self.pod_basis = pod_basis
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = get_activation(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = get_activation(activation)

        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)

        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
            self.b = bst.ParamState(0.0)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self.trunk is None:
            # POD only
            x = u.math.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = u.math.einsum("bi,ni->bn", x_func, u.math.concatenate((self.pod_basis, x_loc), axis=1))
            x += self.b.value

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
