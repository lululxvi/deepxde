from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class BiONet(NN):
    """Deep operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super(BiONet, self).__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
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
        print("Building BiONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        y_func1 = self._branch_net(self.X_func1, self.layer_branch1[1:])
        # Branch net 2
        y_func2 = self._branch_net(self.X_func2, self.layer_branch2[1:])
        # Trunk net
        y_loc = self._trunk_net(self.X_loc, self.layer_trunk[1:])

        # Dot product
        y_loc = tf.reshape(y_loc, (-1, self.layer_branch1[-1], self.layer_branch2[-1]))
        self.y = tf.einsum("bji,bi->bj", y_loc, y_func2)
        self.y = tf.einsum("bi,bi->b", self.y, y_func1)
        self.y = tf.expand_dims(self.y, axis=1)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _branch_net(self, X_func, layer_branch):
        y_func = X_func
        for i in range(len(layer_branch) - 1):
            y_func = self._dense(
                y_func,
                layer_branch[i],
                activation=self.activation_branch,
                regularizer=self.regularizer,
            )
        return self._dense(y_func, layer_branch[-1], regularizer=self.regularizer)

    def _trunk_net(self, X_loc, layer_trunk):
        y_loc = X_loc
        for i in range(len(layer_trunk)):
            y_loc = self._dense(
                y_loc,
                layer_trunk[i],
                activation=self.activation_trunk,
                regularizer=self.regularizer,
            )
        return y_loc

    def _dense(
        self,
        inputs,
        units,
        activation=None,
        use_bias=True,
        regularizer=None,
        trainable=True,
    ):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            trainable=trainable,
        )
