import paddle

from .fnn import FNN
from .nn import NN
from .. import activations
from .. import initializers
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
        layer_sizes_branch: A list of integers as the width of a fully connected
            network, or `(dim, f)` where `dim` is the input dimension and `f` is a
            network function. The width of the last layer in the branch and trunk net
            should be equal.
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
        use_bias=True,
    ):
        super().__init__()
        self.layer_sizes_func = layer_sizes_branch
        self.layer_sizes_loc = layer_sizes_trunk

        if isinstance(activation, dict):
            self.activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)

        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.use_bias = use_bias
        if use_bias:
            # register bias to parameter for updating in optimizer and storage
            self.b = self.create_parameter(
                shape=(1,), default_initializer=initializers.get("zeros")
            )

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
        x = paddle.sum(x_func * x_loc, axis=1, keepdim=True)
        # Add bias
        if self.use_bias:
            x += self.b
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
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer

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
            self.branch = paddle.nn.LayerList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = paddle.nn.LayerList(self.trunk)
        self.b = paddle.nn.ParameterList(
            [
                paddle.create_parameter(
                    shape=[1,],
                    dtype=paddle.get_default_dtype(),
                    default_initializer=paddle.nn.initializer.Constant(value=0),
                )
                for _ in range(self.num_outputs)
            ]
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
        y = x_func @ x_loc.T
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return paddle.stack(ys, axis=2)

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
