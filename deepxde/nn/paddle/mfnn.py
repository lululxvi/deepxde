import paddle

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config


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
        self.initializer = initializers.get(kernel_initializer)
        self.trainable_lo = trainable_low_fidelity
        self.trainable_hi = trainable_high_fidelity
        self.residue = residue
        self.regularizer = regularizers.get(regularization)

        # low fidelity
        self.linears_lo = self._init_dense(self.layer_size_lo, self.trainable_lo)

        # high fidelity
        # linear part
        self.linears_hi_l = paddle.nn.Linear(
            in_features=self.layer_size_lo[0] + self.layer_size_lo[-1],
            out_features=self.layer_size_hi[-1],
            weight_attr=paddle.ParamAttr(initializer=self.initializer),
        )
        if not self.trainable_hi:
            for param in self.linears_hi_l.parameters():
                param.stop_gradient = False
        # nonlinear part
        self.layer_size_hi = [
            self.layer_size_lo[0] + self.layer_size_lo[-1]
        ] + self.layer_size_hi
        self.linears_hi = self._init_dense(self.layer_size_hi, self.trainable_hi)
        # linear + nonlinear
        if not self.residue:
            alpha = self._init_alpha(0.0, self.trainable_hi)
            self.add_parameter("alpha", alpha)
        else:
            alpha1 = self._init_alpha(0.0, self.trainable_hi)
            alpha2 = self._init_alpha(0.0, self.trainable_hi)
            self.add_parameter("alpha1", alpha1)
            self.add_parameter("alpha2", alpha2)

    def _init_dense(self, layer_size, trainable):
        linears = paddle.nn.LayerList()
        for i in range(len(layer_size) - 1):
            linear = paddle.nn.Linear(
                in_features=layer_size[i],
                out_features=layer_size[i + 1],
                weight_attr=paddle.ParamAttr(initializer=self.initializer),
            )
            if not trainable:
                for param in linear.parameters():
                    param.stop_gradient = False
            linears.append(linear)
        return linears

    def _init_alpha(self, value, trainable):
        alpha = paddle.create_parameter(
            shape=[1],
            dtype=config.real(paddle),
            default_initializer=paddle.nn.initializer.Constant(value),
        )
        alpha.stop_gradient = not trainable
        return alpha

    def forward(self, inputs):
        # low fidelity
        y = inputs
        for i, linear in enumerate(self.linears_lo):
            y = linear(y)
            if i != len(self.linears_lo) - 1:
                y = self.activation(y)
        y_lo = y

        # high fidelity
        x_hi = paddle.concat([inputs, y_lo], axis=1)
        # linear
        y_hi_l = self.linears_hi_l(x_hi)
        # nonlinear
        y = x_hi
        for i, linear in enumerate(self.linears_hi):
            y = linear(y)
            if i != len(self.linears_hi) - 1:
                y = self.activation(y)
        y_hi_nl = y
        # linear + nonlinear
        if not self.residue:
            alpha = paddle.tanh(self.alpha)
            y_hi = y_hi_l + alpha * y_hi_nl
        else:
            alpha1 = paddle.tanh(self.alpha1)
            alpha2 = paddle.tanh(self.alpha2)
            y_hi = y_lo + 0.1 * (alpha1 * y_hi_l + alpha2 * y_hi_nl)

        return y_lo, y_hi
