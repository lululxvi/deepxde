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
        self.activation_tanh = activations.get("tanh")
        self.initializer = initializers.get(kernel_initializer)
        self.initializer_zero = initializers.get("zeros")
        self.trainable_lo = trainable_low_fidelity
        self.trainable_hi = trainable_high_fidelity
        self.residue = residue
        self.regularizer = regularizers.get(regularization)
        self.regularizer_value = regularization[1:] if regularization is not None else None
        
        # low fidelity
        self.linears_lo = self.init_dense(self.layer_size_lo, self.trainable_lo)
        
        # high fidelity
        # linear part
        self.linears_hi_l = paddle.nn.Linear(
            in_features=self.layer_size_lo[0] + self.layer_size_lo[-1],
            out_features=self.layer_size_hi[-1],
            weight_attr=paddle.ParamAttr(initializer=self.initializer),
            bias_attr=paddle.ParamAttr(initializer=self.initializer_zero),
        )
        if not self.trainable_hi:
            for param in self.linears_hi_l.parameters():
                param.stop_gradient = False
        # nonlinear part
        self.layer_size_hi = [self.layer_size_lo[0] + self.layer_size_lo[-1]] + self.layer_size_hi
        self.linears_hi = self.init_dense(self.layer_size_hi, self.trainable_hi)
        # linear + nonlinear
        if not self.residue:
            alpha = self.init_alpha(0.0, self.trainable_hi)
            self.add_parameter("alpha",alpha)
        else:
            alpha1 = self.init_alpha(0.0, self.trainable_hi)
            alpha2 = self.init_alpha(0.0, self.trainable_hi)
            self.add_parameter("alpha1",alpha1)
            self.add_parameter("alpha2",alpha2)
    
    def init_dense(self, layer_size, trainable):
        linears = paddle.nn.LayerList()
        for i in range(len(layer_size) - 1):
            linear = paddle.nn.Linear(
                in_features=layer_size[i],
                out_features=layer_size[i + 1],
                weight_attr=paddle.ParamAttr(initializer=self.initializer),
                bias_attr=paddle.ParamAttr(initializer=self.initializer_zero),
            )
            if not trainable:
                for param in linear.parameters():
                    param.stop_gradient = False
            linears.append(linear)
        return linears

    def init_alpha(self, value, trainable):
        alpha = paddle.create_parameter(
            shape=[1], 
            dtype=config.real(paddle), 
            default_initializer=paddle.nn.initializer.Constant(value),
        )
        alpha.stop_gradient=not trainable
        return alpha

    def forward(self, inputs):
        x = inputs.astype(config.real(paddle))
        # low fidelity
        y = x
        for i, linear in enumerate(self.linears_lo):
            y = linear(y)
            if i != len(self.linears_lo) - 1:
                y = self.activation(y)
        y_lo = y

        # high fidelity
        x_hi = paddle.concat([x, y_lo], axis=1)
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
            alpha = self.activation_tanh(self.alpha)
            y_hi = y_hi_l + alpha * y_hi_nl
        else:
            alpha1 = self.activation_tanh(self.alpha1)
            alpha2 = self.activation_tanh(self.alpha2)
            y_hi = y_lo + 0.1 * (alpha1 * y_hi_l + alpha2 * y_hi_nl)

        return y_lo, y_hi
