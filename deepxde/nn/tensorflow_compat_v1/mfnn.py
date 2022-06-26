from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class MfNN(NN):
    """Multifidelity neural networks."""

    def __init__(
        self,
        layer_sizes_low_fidelity,
        layer_sizes_high_fidelity,
        activation,
        kernel_initializer,
        regularization=None,
        residue=False,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    ):
        super().__init__()
        self.layer_size_lo = layer_sizes_low_fidelity
        self.layer_size_hi = layer_sizes_high_fidelity
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.residue = residue
        self.trainable_lo = trainable_low_fidelity
        self.trainable_hi = trainable_high_fidelity

    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return [self.y_lo, self.y_hi]

    @property
    def targets(self):
        return [self.target_lo, self.target_hi]

    @timing
    def build(self):
        print("Building multifidelity neural network...")
        self.X = tf.placeholder(config.real(tf), [None, self.layer_size_lo[0]])

        # Low fidelity
        y = self.X
        for i in range(len(self.layer_size_lo) - 2):
            y = self._dense(
                y,
                self.layer_size_lo[i + 1],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_lo,
            )
        self.y_lo = self._dense(
            y,
            self.layer_size_lo[-1],
            regularizer=self.regularizer,
            trainable=self.trainable_lo,
        )

        # High fidelity
        X_hi = tf.concat([self.X, self.y_lo], 1)
        # Linear
        y_hi_l = self._dense(X_hi, self.layer_size_hi[-1], trainable=self.trainable_hi)
        # Nonlinear
        y = X_hi
        for i in range(len(self.layer_size_hi) - 1):
            y = self._dense(
                y,
                self.layer_size_hi[i],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_hi,
            )
        y_hi_nl = self._dense(
            y,
            self.layer_size_hi[-1],
            use_bias=False,
            regularizer=self.regularizer,
            trainable=self.trainable_hi,
        )
        # Linear + nonlinear
        if not self.residue:
            alpha = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha = activations.get("tanh")(alpha)
            self.y_hi = y_hi_l + alpha * y_hi_nl
        else:
            alpha1 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha1 = activations.get("tanh")(alpha1)
            alpha2 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha2 = activations.get("tanh")(alpha2)
            self.y_hi = self.y_lo + 0.1 * (alpha1 * y_hi_l + alpha2 * y_hi_nl)

        self.target_lo = tf.placeholder(config.real(tf), [None, self.layer_size_lo[-1]])
        self.target_hi = tf.placeholder(config.real(tf), [None, self.layer_size_hi[-1]])
        self.built = True

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
