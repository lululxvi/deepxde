__all__ = ["DeepONet", "DeepONetCartesianProd"]

from abc import ABC, abstractmethod

import numpy as np

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


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

    def _build_branch_and_trunk(self):
        # Branch net to encode the input function
        branch = self.net.build_branch_net()
        # Trunk net to encode the domain of the output function
        trunk = self.net.build_trunk_net()
        return branch, trunk

    @abstractmethod
    def build(self):
        """Return the output tensor."""


class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self):
        branch, trunk = self._build_branch_and_trunk()
        if branch.shape[-1] != trunk.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        y = self.net.merge_branch_trunk(branch, trunk)
        return y


class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self):
        single_output_strategy = SingleOutputStrategy(self.net)
        ys = [single_output_strategy.build() for _ in range(self.net.num_outputs)]
        return self.net.concatenate_outputs(ys)


class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self):
        branch, trunk = self._build_branch_and_trunk()
        if branch.shape[-1] != trunk.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if branch.shape[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        branch_groups = tf.split(
            branch, num_or_size_splits=self.net.num_outputs, axis=1
        )
        trunk_groups = tf.split(trunk, num_or_size_splits=self.net.num_outputs, axis=1)
        ys = []
        for i in range(self.net.num_outputs):
            y = self.net.merge_branch_trunk(branch_groups[i], trunk_groups[i])
            ys.append(y)
        return self.net.concatenate_outputs(ys)


class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self):
        branch, trunk = self._build_branch_and_trunk()
        if branch.shape[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        if branch.shape[-1] / self.net.num_outputs != trunk.shape[-1]:
            raise AssertionError(
                f"Output size of the trunk net does not equal to {branch.shape[-1] // self.net.num_outputs}."
            )
        branch_groups = tf.split(
            branch, num_or_size_splits=self.net.num_outputs, axis=1
        )
        ys = []
        for i in range(self.net.num_outputs):
            y = self.net.merge_branch_trunk(branch_groups[i], trunk)
            ys.append(y)
        return self.net.concatenate_outputs(ys)


class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self):
        branch, trunk = self._build_branch_and_trunk()
        if trunk.shape[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the trunk net is not evenly divisible by {self.net.num_outputs}."
            )
        if trunk.shape[-1] / self.net.num_outputs != branch.shape[-1]:
            raise AssertionError(
                f"Output size of the branch net does not equal to {trunk.shape[-1] // self.net.num_outputs}."
            )
        trunk_groups = tf.split(trunk, num_or_size_splits=self.net.num_outputs, axis=1)
        ys = []
        for i in range(self.net.num_outputs):
            y = self.net.merge_branch_trunk(branch, trunk_groups[i])
            ys.append(y)
        return self.net.concatenate_outputs(ys)


class DeepONet(NN):
    """Deep operator network.

    `Lu et al. Learning nonlinear operators via DeepONet based on the universal
    approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected
            network, or `(dim, f)` where `dim` is the input dimension and `f` is a
            network function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        dropout_rate: If `dropout_rate` is a ``float`` between 0 and 1, then the
            same rate is used in both trunk and branch nets. If `dropout_rate`
            is a ``dict``, then the trunk net uses the rate `dropout_rate["trunk"]`,
            and the branch net uses `dropout_rate["branch"]`. Both `dropout_rate["trunk"]`
            and `dropout_rate["branch"]` should be ``float`` or lists of ``float``.
            The list length should match the length of `layer_sizes_trunk` - 1 for the
            trunk net and `layer_sizes_branch` - 2 for the branch net.
        trainable_branch: Boolean.
        trainable_trunk: Boolean or a list of booleans.
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
        regularization=None,
        dropout_rate=0,
        use_bias=True,
        stacked=False,
        trainable_branch=True,
        trainable_trunk=True,
        num_outputs=1,
        multi_output_strategy=None,
    ):
        super().__init__()
        if isinstance(trainable_trunk, (list, tuple)):
            if len(trainable_trunk) != len(layer_sizes_trunk) - 1:
                raise ValueError("trainable_trunk does not match layer_sizes_trunk.")

        self.layer_size_func = layer_sizes_branch
        self.layer_size_loc = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        if stacked:
            self.kernel_initializer_stacked = initializers.get(
                "stacked " + kernel_initializer
            )
        self.regularizer = regularizers.get(regularization)
        if isinstance(dropout_rate, dict):
            self.dropout_rate_branch = dropout_rate["branch"]
            self.dropout_rate_trunk = dropout_rate["trunk"]
        else:
            self.dropout_rate_branch = self.dropout_rate_trunk = dropout_rate
        if isinstance(self.dropout_rate_branch, list):
            if not (len(layer_sizes_branch) - 2) == len(self.dropout_rate_branch):
                raise ValueError(
                    "Number of dropout rates of branch net must be "
                    f"equal to {len(layer_sizes_branch) - 2}"
                )
        else:
            self.dropout_rate_branch = [self.dropout_rate_branch] * (
                len(layer_sizes_branch) - 2
            )
        if isinstance(self.dropout_rate_trunk, list):
            if not (len(layer_sizes_trunk) - 1) == len(self.dropout_rate_trunk):
                raise ValueError(
                    "Number of dropout rates of trunk net must be "
                    f"equal to {len(layer_sizes_trunk) - 1}"
                )
        else:
            self.dropout_rate_trunk = [self.dropout_rate_trunk] * (
                len(layer_sizes_trunk) - 1
            )
        self.use_bias = use_bias
        self.stacked = stacked
        self.trainable_branch = trainable_branch
        self.trainable_trunk = trainable_trunk

        self._inputs = None
        self._X_func_default = None

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

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if value[1] is not None:
            raise ValueError("DeepONet does not support setting trunk net input.")
        self._X_func_default = value[0]
        self._inputs = self.X_loc

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    def _feed_dict_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            n = len(inputs)
            inputs = [np.tile(self._X_func_default, (n, 1)), inputs]
        return dict(zip([self.X_func, self.X_loc], inputs))

    @timing
    def build(self):
        print("Building DeepONet...")
        self.X_func = tf.placeholder(config.real(tf), [None, self.layer_size_func[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_size_loc[0]])
        self._inputs = [self.X_func, self.X_loc]

        self.y = self.multi_output_strategy.build()
        if self._output_transform is not None:
            self.y = self._output_transform(self._inputs, self.y)

        self.target = tf.placeholder(config.real(tf), [None, self.num_outputs])
        self.built = True

    def build_branch_net(self):
        if callable(self.layer_size_func[1]):
            # User-defined network
            return self.layer_size_func[1](self.X_func)

        if self.stacked:
            # Stacked fully connected network
            return self._build_stacked_branch_net()

        # Unstacked fully connected network
        return self._build_unstacked_branch_net()

    def _build_stacked_branch_net(self):
        y_func = self.X_func
        stack_size = self.layer_size_func[-1]

        for i in range(1, len(self.layer_size_func) - 1):
            y_func = self._stacked_dense(
                y_func,
                self.layer_size_func[i],
                stack_size=stack_size,
                activation=self.activation_branch,
                trainable=self.trainable_branch,
            )
            if self.dropout_rate_branch[i - 1] > 0:
                y_func = tf.layers.dropout(
                    y_func,
                    rate=self.dropout_rate_branch[i - 1],
                    training=self.training,
                )
        return self._stacked_dense(
            y_func,
            1,
            stack_size=stack_size,
            use_bias=self.use_bias,
            trainable=self.trainable_branch,
        )

    def _build_unstacked_branch_net(self):
        y_func = self.X_func

        for i in range(1, len(self.layer_size_func) - 1):
            y_func = self._dense(
                y_func,
                self.layer_size_func[i],
                activation=self.activation_branch,
                regularizer=self.regularizer,
                trainable=self.trainable_branch,
            )
            if self.dropout_rate_branch[i - 1] > 0:
                y_func = tf.layers.dropout(
                    y_func,
                    rate=self.dropout_rate_branch[i - 1],
                    training=self.training,
                )
        return self._dense(
            y_func,
            self.layer_size_func[-1],
            use_bias=self.use_bias,
            regularizer=self.regularizer,
            trainable=self.trainable_branch,
        )

    def build_trunk_net(self):
        y_loc = self.X_loc
        if self._input_transform is not None:
            y_loc = self._input_transform(y_loc)
        for i in range(1, len(self.layer_size_loc)):
            y_loc = self._dense(
                y_loc,
                self.layer_size_loc[i],
                activation=self.activation_trunk,
                regularizer=self.regularizer,
                trainable=self.trainable_trunk[i - 1]
                if isinstance(self.trainable_trunk, (list, tuple))
                else self.trainable_trunk,
            )
            if self.dropout_rate_trunk[i - 1] > 0:
                y_loc = tf.layers.dropout(
                    y_loc, rate=self.dropout_rate_trunk[i - 1], training=self.training
                )
        return y_loc

    def merge_branch_trunk(self, branch, trunk):
        # Dot product
        y = tf.einsum("bi,bi->b", branch, trunk)
        y = tf.expand_dims(y, axis=1)
        if self.use_bias:
            b = tf.Variable(tf.zeros(1, dtype=config.real(tf)))
            y += b
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return tf.concat(ys, axis=1)

    def _dense(
        self,
        inputs,
        units,
        activation=None,
        use_bias=True,
        regularizer=None,
        trainable=True,
    ):
        dense = tf.keras.layers.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            trainable=trainable,
        )
        out = dense(inputs)
        if regularizer:
            self.regularization_loss += tf.math.add_n(dense.losses)
        return out

    def _stacked_dense(
        self, inputs, units, stack_size, activation=None, use_bias=True, trainable=True
    ):
        """Stacked densely-connected NN layer.

        Args:
            inputs: If inputs is the NN input, then it is a 2D tensor with shape:
                `(batch_size, input_dim)`; otherwise, it is 3D tensor with shape:
                `(batch_size, stack_size, input_dim)`.

        Returns:
            tensor: outputs.

            If outputs is the NN output, i.e., units = 1,
            2D tensor with shape: `(batch_size, stack_size)`;
            otherwise, 3D tensor with shape: `(batch_size, stack_size, units)`.
        """
        shape = inputs.shape
        input_dim = shape[-1]
        if len(shape) == 2:
            # NN input layer
            W = tf.Variable(
                self.kernel_initializer_stacked([stack_size, input_dim, units]),
                trainable=trainable,
            )
            outputs = tf.einsum("bi,nij->bnj", inputs, W)
        elif units == 1:
            # NN output layer
            W = tf.Variable(
                self.kernel_initializer_stacked([stack_size, input_dim]),
                trainable=trainable,
            )
            outputs = tf.einsum("bni,ni->bn", inputs, W)
        else:
            W = tf.Variable(
                self.kernel_initializer_stacked([stack_size, input_dim, units]),
                trainable=trainable,
            )
            outputs = tf.einsum("bni,nij->bnj", inputs, W)
        if use_bias:
            if units == 1:
                # NN output layer
                b = tf.Variable(tf.zeros(stack_size), trainable=trainable)
            else:
                b = tf.Variable(tf.zeros([stack_size, units]), trainable=trainable)
            outputs += b
        if activation is not None:
            return activation(outputs)
        return outputs


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
        dropout_rate: If `dropout_rate` is a ``float`` between 0 and 1, then the
            same rate is used in both trunk and branch nets. If `dropout_rate`
            is a ``dict``, then the trunk net uses the rate `dropout_rate["trunk"]`,
            and the branch net uses `dropout_rate["branch"]`. Both `dropout_rate["trunk"]`
            and `dropout_rate["branch"]` should be ``float`` or lists of ``float``.
            The list length should match the length of `layer_sizes_trunk` - 1 for the
            trunk net and `layer_sizes_branch` - 2 for the branch net.
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
        regularization=None,
        dropout_rate=0,
        num_outputs=1,
        multi_output_strategy=None,
    ):
        super().__init__()
        self.layer_size_func = layer_sizes_branch
        self.layer_size_loc = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        if isinstance(dropout_rate, dict):
            self.dropout_rate_branch = dropout_rate["branch"]
            self.dropout_rate_trunk = dropout_rate["trunk"]
        else:
            self.dropout_rate_branch = self.dropout_rate_trunk = dropout_rate
        if isinstance(self.dropout_rate_branch, list):
            if not (len(layer_sizes_branch) - 2) == len(self.dropout_rate_branch):
                raise ValueError(
                    "Number of dropout rates of branch net must be "
                    f"equal to {len(layer_sizes_branch) - 2}"
                )
        else:
            self.dropout_rate_branch = [self.dropout_rate_branch] * (
                len(layer_sizes_branch) - 2
            )
        if isinstance(self.dropout_rate_trunk, list):
            if not (len(layer_sizes_trunk) - 1) == len(self.dropout_rate_trunk):
                raise ValueError(
                    "Number of dropout rates of trunk net must be "
                    f"equal to {len(layer_sizes_trunk) - 1}"
                )
        else:
            self.dropout_rate_trunk = [self.dropout_rate_trunk] * (
                len(layer_sizes_trunk) - 1
            )
        self._inputs = None

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

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building DeepONetCartesianProd...")
        self.X_func = tf.placeholder(config.real(tf), [None, self.layer_size_func[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_size_loc[0]])
        self._inputs = [self.X_func, self.X_loc]

        self.y = self.multi_output_strategy.build()
        if self._output_transform is not None:
            self.y = self._output_transform(self._inputs, self.y)

        if self.num_outputs > 1:
            self.target = tf.placeholder(config.real(tf), [None, None, None])
        else:
            self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def build_branch_net(self):
        y_func = self.X_func
        if callable(self.layer_size_func[1]):
            # User-defined network
            y_func = self.layer_size_func[1](y_func)
        else:
            # Fully connected network
            for i in range(1, len(self.layer_size_func) - 1):
                y_func = self._dense(
                    y_func,
                    self.layer_size_func[i],
                    activation=self.activation_branch,
                    regularizer=self.regularizer,
                )
                if self.dropout_rate_branch[i - 1] > 0:
                    y_func = tf.layers.dropout(
                        y_func,
                        rate=self.dropout_rate_branch[i - 1],
                        training=self.training,
                    )
            y_func = self._dense(
                y_func,
                self.layer_size_func[-1],
                regularizer=self.regularizer,
            )
        return y_func

    def build_trunk_net(self):
        # Trunk net to encode the domain of the output function
        y_loc = self.X_loc
        if self._input_transform is not None:
            y_loc = self._input_transform(y_loc)
        for i in range(1, len(self.layer_size_loc)):
            y_loc = self._dense(
                y_loc,
                self.layer_size_loc[i],
                activation=self.activation_trunk,
                regularizer=self.regularizer,
            )
            if self.dropout_rate_trunk[i - 1] > 0:
                y_loc = tf.layers.dropout(
                    y_loc, rate=self.dropout_rate_trunk[i - 1], training=self.training
                )
        return y_loc

    def merge_branch_trunk(self, branch, trunk):
        y = tf.einsum("bi,ni->bn", branch, trunk)
        # Add bias
        b = tf.Variable(tf.zeros(1, dtype=config.real(tf)))
        y += b
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return tf.stack(ys, axis=2)

    def _dense(
        self,
        inputs,
        units,
        activation=None,
        use_bias=True,
        regularizer=None,
        trainable=True,
    ):
        dense = tf.keras.layers.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            trainable=trainable,
        )
        out = dense(inputs)
        if regularizer:
            self.regularization_loss += tf.math.add_n(dense.losses)
        return out
