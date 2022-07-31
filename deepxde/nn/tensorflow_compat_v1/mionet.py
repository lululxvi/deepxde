from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class MIONet(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

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
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONetCartesianProd(MIONet):
    """MIONet with two input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True
