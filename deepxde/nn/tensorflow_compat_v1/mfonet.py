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


class MfONet(NN):
    """Multifidelity DeepONet."""

    def __init__(
        self,
        layer_sizes_branch_low_fidelity,
        layer_sizes_trunk_low_fidelity,
        layer_sizes_high_fidelity_linear,
        layer_sizes_branch_high_fidelity_nonlinear,
        layer_sizes_trunk_high_fidelity_nonlinear,
        activation,
        kernel_initializer,
        regularization=None,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    ):
        super(MfONet, self).__init__()

        self.layer_branch_lo = layer_sizes_branch_low_fidelity
        self.layer_trunk_lo = layer_sizes_trunk_low_fidelity
        self.layer_hi_l = layer_sizes_high_fidelity_linear
        self.layer_branch_hi_nl = layer_sizes_branch_high_fidelity_nonlinear
        self.layer_trunk_hi_nl = layer_sizes_trunk_high_fidelity_nonlinear
        if isinstance(activation, dict):
            self.activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.trainable_lo = trainable_low_fidelity
        self.trainable_hi = trainable_high_fidelity

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return [self.y_lo, self.y_hi]

    @property
    def targets(self):
        return [self.target_lo, self.target_hi]

    @timing
    def build(self):
        print("Building multifidelity DeepONet...")
        self.X_func = tf.placeholder(config.real(tf), [None, self.layer_branch_lo[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk_lo[0]])
        self._inputs = [self.X_func, self.X_loc]

        # Low fidelity
        self.y_lo = self._onet(
            self.X_func,
            self.X_loc,
            self.layer_branch_lo[1:],
            self.layer_trunk_lo[1:],
            self.trainable_lo,
        )

        # High fidelity
        X_loc_hi = tf.concat([self.X_loc, self.y_lo], 1)

        # Linear
        # Branch net
        y_func = self._dense(self.X_func, self.layer_hi_l, trainable=self.trainable_hi)
        # Trunk net
        y_loc = self._dense(X_loc_hi, self.layer_hi_l, trainable=self.trainable_hi)
        # Dot product
        y_hi_l = tf.einsum("bi,bi->b", y_func, y_loc)
        y_hi_l = tf.expand_dims(y_hi_l, axis=1)
        # Add bias
        b = tf.Variable(tf.zeros(1), trainable=self.trainable_hi)
        y_hi_l += b

        # Nonlinear
        y_hi_nl = self._onet(
            self.X_func,
            X_loc_hi,
            self.layer_branch_hi_nl,
            self.layer_trunk_hi_nl,
            self.trainable_hi,
        )

        # Linear + nonlinear
        alphal = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
        alpha1 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
        alpha1 = activations.get("tanh")(alpha1)
        alpha2 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
        alpha2 = activations.get("tanh")(alpha2)
        self.y_hi = (1 + alphal) * self.y_lo + 0.1 * (
            alpha1 * y_hi_l + alpha2 * y_hi_nl
        )

        self.target_lo = tf.placeholder(config.real(tf), [None, 1])
        self.target_hi = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _onet(self, X_func, X_loc, layer_branch, layer_trunk, trainable):
        # Branch net: Unstacked fully connected network
        y_func = X_func
        for i in range(len(layer_branch) - 1):
            y_func = self._dense(
                y_func,
                layer_branch[i],
                activation=self.activation_branch,
                regularizer=self.regularizer,
                trainable=trainable,
            )
        y_func = self._dense(
            y_func, layer_branch[-1], regularizer=self.regularizer, trainable=trainable
        )

        # Trunk net
        y_loc = X_loc
        for i in range(len(layer_trunk)):
            y_loc = self._dense(
                y_loc,
                layer_trunk[i],
                activation=self.activation_trunk,
                regularizer=self.regularizer,
                trainable=trainable,
            )

        # Dot product
        y = tf.einsum("bi,bi->b", y_func, y_loc)
        y = tf.expand_dims(y, axis=1)
        # Add bias
        b = tf.Variable(tf.zeros(1), trainable=trainable)
        y += b
        return y

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
