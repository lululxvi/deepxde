from abc import ABC
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

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        pass

    def call(self, x_func, x_loc, training=False):
        pass


class SingleOutputStrategy(DeepONetStrategy):
    """
    Single output build strategy is the standard build method. Example:

    net = dde.nn.DeepONetCartesianProd(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
        num_outputs = 1,
    )
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if any(isinstance(i, list) for i in layer_sizes_branch) or any(
            isinstance(i, list) for i in layer_sizes_trunk
        ):
            raise AssertionError(
                "Nested lists cannot be used with single output strategy."
            )
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )

        # Branch net to encode the input function
        branch = self.net.build_branch_net(layer_sizes_branch)
        # Trunk net to encode the domain of the output function
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        # Branch net to encode the input function
        x_func = self.net.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = self.net.merge_branch_trunk(x_func, x_loc)
        # Add bias
        x += self.net.b
        return x


class IndependentStrategy(DeepONetStrategy):
    """
    Directly use n independent DeepONets, and each DeepONet outputs only one
        function. For the same architectures, use the single output strategy
        format. For different architectures use a nested lists...

    net = dde.nn.DeepONetCartesianProd(
        [[m, 40, 40],[m, 80, 80]],
        [[dim_x, 40, 40],[dim_x, 80, 80]],
        "relu",
        "Glorot normal",
        num_outputs = 2,
    )
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        single_strategy = SingleOutputStrategy(self.net)
        branch = []
        trunk = []
        if any(isinstance(i, list) for i in layer_sizes_branch):
            if not any(isinstance(i, list) for i in layer_sizes_trunk):
                raise AssertionError(
                    "Trunk and branch must both be nested for different architectures."
                )
            for i in range(self.net.num_outputs):
                branch_tmp, trunk_tmp = single_strategy.build(
                    layer_sizes_branch[i], layer_sizes_trunk[i]
                )
                branch.append(branch_tmp)
                trunk.append(trunk_tmp)
            return branch, trunk
        if any(isinstance(i, list) for i in layer_sizes_trunk):
            raise AssertionError(
                "Trunk and branch must both be nested for different architectures."
            )
        for i in range(self.net.num_outputs):
            branch_tmp, trunk_tmp = single_strategy.build(
                layer_sizes_branch, layer_sizes_trunk
            )
            branch.append(branch_tmp)
            trunk.append(trunk_tmp)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        x = []
        for i in range(self.net.num_outputs):
            # Branch net to encode the input function
            x_func_i = self.net.branch[i](x_func)
            # Trunk net to encode the domain of the output function
            x_loc_i = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x_i = self.net.merge_branch_trunk(x_func_i, x_loc_i)
            # Add bias
            x_i += self.net.b[i]
            x.append(x_i)

        x = self.net.concatenate_outputs(x)

        return x


class SplitBothStrategy(DeepONetStrategy):
    """
    Split the outputs of both the branch net and the trunk net into n groups.
        Define desired widths in the last layer of the branch and trunk...

    net = dde.nn.DeepONetCartesianProd(
        [m, 40, [40, 40, 60]],
        [dim_x, 40, [40, 40, 60]],
        "relu",
        "Glorot normal",
        num_outputs = 3,
    )
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if not (
            isinstance(layer_sizes_branch[-1], list)
            and isinstance(layer_sizes_branch[-1], list)
        ):
            raise AssertionError(
                "For split both strategy, last layer must be a list of widths"
            )
        if len(layer_sizes_branch[-1]) != len(layer_sizes_trunk[-1]):
            raise AssertionError(
                "Number of outputs in trunk ({}) and branch ({}) do not match".format(
                    len(layer_sizes_branch[-1]), len(layer_sizes_trunk[-1])
                )
            )
        for i in range(len(layer_sizes_branch[-1])):
            if layer_sizes_branch[-1][i] != layer_sizes_trunk[-1][i]:
                raise AssertionError(
                    "Output width of branch ({}) and trunk ({}) does not match".format(
                        layer_sizes_branch[-1][i], layer_sizes_trunk[-1][i]
                    )
                )
        self.net.output_widths = layer_sizes_branch[-1]
        layer_sizes_branch[-1] = sum(layer_sizes_branch[-1])
        layer_sizes_trunk[-1] = layer_sizes_branch[-1]
        single_strategy = SingleOutputStrategy(self.net)
        return single_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc, training=False):
        # Branch net to encode the input function
        x_func = self.net.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))

        # Split x_func and x_loc into respective outputs
        widths = 0
        x = []
        for i, width in enumerate(self.net.output_widths):
            widths += width
            x_func_i = x_func[:, :widths][:, widths - width :]
            x_loc_i = x_loc[:, :widths][:, widths - width :]
            x_i = self.net.merge_branch_trunk(x_func_i, x_loc_i)
            # Add bias
            x_i += self.net.b[i]
            x.append(x_i)

        x = self.net.concatenate_outputs(x)

        return x


class SplitBranchStrategy(DeepONetStrategy):
    """
    Uses independent branch nets and shares the trunk net. Different branch net
        architectures can be used but must all have the same last layer width
        as the trunk net. For the same architectures, use the single output format.
        For different architectures use a nested lists...

    net = dde.nn.DeepONetCartesianProd(
        [[m, 40, 40],[m, 80, 40]],
        [dim_x, 40, 40],
        "relu",
        "Glorot normal",
        num_outputs = 2
    )

    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        branch = []
        if any(isinstance(i, list) for i in layer_sizes_trunk):
            raise AssertionError(
                "Trunk net cannot be nested for split_branch strategy."
            )

        for i in range(self.net.num_outputs):
            if layer_sizes_branch[i][-1] != layer_sizes_trunk[-1]:
                raise AssertionError(
                    "Output sizes of branch net and trunk net do not match."
                )
            # Branch net to encode the input function
            branch.append(self.net.build_branch_net(layer_sizes_branch[i]))
            # Trunk net to encode the domain of the output function
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, inputs, x_func, x_loc, training=False):
        # Trunk net to encode the domain of the output function
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        x = []
        for i in range(self.net.num_outputs):
            # Branch net to encode the input function
            x_func_i = self.net.branch[i](x_func)
            x_i = self.net.merge_branch_trunk(x_func_i, x_loc)
            # Add bias
            x_i += self.net.b[i]
            x.append(x_i)

        x = self.net.concatenate_outputs(x)

        return x


class SplitTrunkStrategy(DeepONetStrategy):
    """
    Uses independent trunk nets and shares the branch net. Different trunk net
        architectures can be used but must all have the same last layer width
        as the branch net. For the same architectures, use the single output
        format. For different architectures use a nested list...

    net = dde.nn.DeepONetCartesianProd(
        [m, 40, 40],
        [[dim_x, 40, 40], [dim_x, 80, 40]],
        "relu",
        "Glorot normal",
        num_outputs = 2
    )

    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        trunk = []
        if any(isinstance(i, list) for i in layer_sizes_branch):
            raise AssertionError(
                "Branch net cannot be nested for split_trunk strategy."
            )

        for i in range(self.net.num_outputs):
            if layer_sizes_branch[-1] != layer_sizes_trunk[i][-1]:
                raise AssertionError(
                    "Output sizes of branch net and trunk net do not match."
                )
            # Trunk net to encode the domain of the output function
            trunk.append(self.net.build_trunk_net(layer_sizes_trunk[i]))
        # Branch net to encode the input function
        branch = self.net.build_branch_net(layer_sizes_branch)
        return branch, trunk

    def call(self, x_func, x_loc, training=False):
        # Branch net to encode the input function
        x_func = self.net.branch(x_func)
        x = []
        for i in range(self.net.num_outputs):
            # Trunk net to encode the domain of the output function
            x_loc_i = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x_i = self.net.merge_branch_trunk(x_func, x_loc_i)
            # Add bias
            x_i += self.net.b[i]
            x.append(x_i)

        x = self.net.concatenate_outputs(x)

        return x


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
        num_outputs (integer): number of outputs.
        multi_output_strategy (str) or None: None, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet. Can not be used with num_outputs > 1.

            - independent
            Use num_outputs independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into num_outputs
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
            self.activation_branch = self.activation_trunk = activations.get(
                activation
            )
        self.kernel_initializer = kernel_initializer
        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                multi_output_strategy = None
                print("multi_output_strategy is forcibly changed to None.")
        elif multi_output_strategy == None:
            multi_output_strategy = "independent"
            print(
                'multi_output_strategy is forcibly changed to "independent".'
            )
        self.multi_output_strategy = {
            "independent": IndependentStrategy,
            "split": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
            None: SingleOutputStrategy,
        }.get(multi_output_strategy, IndependentStrategy)(self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )

        self.b = []
        for i in range(self.num_outputs):
            self.b.append(tf.Variable(tf.zeros(1, dtype=config.real(tf))))

    def build_branch_net(self, layer_sizes_branch):
        if callable(layer_sizes_branch[1]):
            # User-defined network
            branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            branch = FNN(
                layer_sizes_branch,
                self.activation_branch,
                self.kernel_initializer,
            )
        return branch

    def build_trunk_net(self, layer_sizes_trunk):
        trunk = FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
        )
        return trunk

    def merge_branch_trunk(self, x_func, x_loc):
        # Dot product
        y = tf.einsum("bi,bi->b", x_func, x_loc)
        y = tf.expand_dims(y, axis=1)
        return y

    @staticmethod
    def concatenate_outputs(x):
        return tf.concat(x, axis=1)

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
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): number of outputs.
        multi_output_strategy (str) or None: None, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet. Can not be used with num_outputs > 1.

            - independent
            Use num_outputs independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into num_outputs
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
            self.activation_branch = self.activation_trunk = activations.get(
                activation
            )
        self.kernel_initializer = kernel_initializer
        self.regularization = regularization
        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                multi_output_strategy = None
                print("multi_output_strategy is forcibly changed to None.")
        elif multi_output_strategy == None:
            multi_output_strategy = "independent"
            print(
                'multi_output_strategy is forcibly changed to "independent".'
            )
        self.multi_output_strategy = {
            "independent": IndependentStrategy,
            "split": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
            None: SingleOutputStrategy,
        }.get(multi_output_strategy, IndependentStrategy)(self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )

        self.b = []
        for i in range(self.num_outputs):
            self.b.append(tf.Variable(tf.zeros(1, dtype=config.real(tf))))

    def build_branch_net(self, layer_sizes_branch):
        if callable(layer_sizes_branch[1]):
            # User-defined network
            branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            branch = FNN(
                layer_sizes_branch,
                self.activation_branch,
                self.kernel_initializer,
                regularization=self.regularization,
            )
        return branch

    def build_trunk_net(self, layer_sizes_trunk):
        trunk = FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
            regularization=self.regularization,
        )
        return trunk

    def merge_branch_trunk(self, x_func, x_loc):
        # Dot product
        y = tf.einsum("bi,ni->bn", x_func, x_loc)
        return y

    @staticmethod
    def concatenate_outputs(x):
        return tf.stack(x, axis=2)

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
            activation_branch = self.activation_trunk = activations.get(
                activation
            )

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
            x = tf.einsum(
                "bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1)
            )
            x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
