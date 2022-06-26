from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing


class ResNet(NN):
    """Residual neural network."""

    def __init__(
        self,
        input_size,
        output_size,
        num_neurons,
        num_blocks,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = num_neurons
        self.num_blocks = num_blocks
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

    @property
    def inputs(self):
        return self.x

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.y_

    @timing
    def build(self):
        print("Building residual neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.input_size])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        y = self._dense(y, self.num_neurons, activation=self.activation)
        for _ in range(self.num_blocks):
            y = self._residual_block(y)
        self.y = self._dense(y, self.output_size)

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.output_size])
        self.built = True

    def _dense(self, inputs, units, activation=None, use_bias=True):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
        )

    def _residual_block(self, inputs):
        """A residual block in ResNet."""
        units = inputs.shape[1]

        x = self._dense(inputs, units, activation=self.activation)
        x = self._dense(x, units)

        x += inputs
        x = self.activation(x)

        return x
