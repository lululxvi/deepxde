__all__ = ["DeepONet", "DeepONetCartesianProd", "PODDeepONet"]

from abc import ABC, abstractmethod
from itertools import cycle

from .fnn import FNN
from .nn import NN
from .. import activations
from ... import config
from ...backend import tf


class DeepONetStrategy(ABC):
    """DeepONet building strategy.

    See the section 3.1.6. in
    L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. Karniadakis.
    A comprehensive and fair comparison of two neural operators
    (with practical extensions) based on FAIR data.
    Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
    """

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layer_sizes_branch, layer_sizes_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc, training=False):
        """Forward pass."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        branch = self.net.build_branch_net(layer_sizes_branch)
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = self.net.merge_branch_trunk(x_func, x_loc)
        return x


class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        single_output_strategy = SingleOutputStrategy(self.net)
        branch, trunk = [], []
        for _ in range(self.net.num_outputs):
            branch_, trunk_ = single_output_strategy.build(
                layer_sizes_branch, layer_sizes_trunk
            )
            branch.append(branch_)
            trunk.append(trunk_)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = self.net.branch[i](x_func)
            x_loc_ = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x = self.net.merge_branch_trunk(x_func_, x_loc_)
            xs.append(x)
        return self.net.concatenate_outputs(xs)


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func and x_loc into respective outputs
        shift = 0
        size = x_func.shape[1] // self.net.num_outputs
        xs = []
        for _ in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc_)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_branch[-1] / self.net.num_outputs != layer_sizes_trunk[-1]:
            raise AssertionError(
                f"Output size of the trunk net does not equal to {layer_sizes_branch[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func into respective outputs
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for _ in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_trunk[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the trunk net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_trunk[-1] / self.net.num_outputs != layer_sizes_branch[-1]:
            raise AssertionError(
                f"Output size of the branch net does not equal to {layer_sizes_trunk[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc, training=False):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_loc into respective outputs
        shift = 0
        size = x_func.shape[1]
        xs = []
        for _ in range(self.net.num_outputs):
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func, x_loc_)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)


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
        self.regularization = regularization

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

        self.b = cycle(
            [
                tf.Variable(tf.zeros(1, dtype=config.real(tf)))
                for _ in range(self.num_outputs)
            ]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(
            layer_sizes_branch,
            self.activation_branch,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def merge_branch_trunk(self, x_func, x_loc):
        y = tf.einsum("bi,bi->b", x_func, x_loc)
        y = tf.expand_dims(y, axis=1)
        y += next(self.b)
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return tf.concat(ys, axis=1)

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc, training)
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
        self.regularization = regularization

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

        self.b = cycle(
            [
                tf.Variable(tf.zeros(1, dtype=config.real(tf)))
                for _ in range(self.num_outputs)
            ]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(
            layer_sizes_branch,
            self.activation_branch,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
            regularization=self.regularization,
        )

    def merge_branch_trunk(self, x_func, x_loc):
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        y += next(self.b)
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return tf.stack(ys, axis=2)

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc, training)
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
        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float32)
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
            self.branch = FNN(
                layer_sizes_branch,
                activation_branch,
                kernel_initializer,
                regularization=regularization,
            )

        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk,
                self.activation_trunk,
                kernel_initializer,
                regularization=regularization,
            )
            self.b = tf.Variable(tf.zeros(1, dtype=config.real(tf)))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self.trunk is None:
            # POD only
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
