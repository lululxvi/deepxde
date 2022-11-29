from paddle.nn.initializer import Normal

from ...backend import paddle
from .. import activations, initializers


class MsFFN(paddle.nn.Layer):
    """Multi-scale fourier feature networks.

    Args:
        sigmas: List of standard deviation of the distribution of fourier feature
            embeddings.

    References:
        `S. Wang, H. Wang, & P. Perdikaris. On the eigenvector bias of Fourier feature
        networks: From regression to solving multi-scale PDEs with physics-informed
        neural networks. Computer Methods in Applied Mechanics and Engineering, 384,
        113938, 2021 <https://doi.org/10.1016/j.cma.2021.113938>`_.
    """

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        sigmas,
        regularization=None,
        dropout_rate=0.0,
        kernel_constraint=None,
        use_bias=True
    ):
        super(MsFFN, self).__init__()
        self._input_transform = None
        self._output_transform = None

        self.activation = activations.get(activation)
        self.layer_size = layer_sizes
        self.dropout_rate = dropout_rate
        self.sigmas = sigmas  # list or tuple
        self.use_bias = use_bias
        self.fourier_feature_weights = None
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.b = []
        for i, sigma in enumerate(self.sigmas):
            self.b.append(
                self.create_parameter(
                    shape=[layer_sizes[0], self.layer_size[1] // 2], default_initializer=Normal(std=sigma),
                )
            )
            # freeze all parameter in self.b
            self.b[-1].trainable = False
            self.b[-1].stop_gradient = True

        self.linears = paddle.nn.LayerList()
        for i in range(2, len(layer_sizes) - 1):
            self.linears.append(
                paddle.nn.Linear(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

        self._dense = paddle.nn.Linear(
            layer_sizes[-2] * 2,
            layer_sizes[-1],
        )
        initializer(self._dense.weight)
        initializer_zero(self._dense.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        # fourier feature layer
        yb = [
            self._fourier_feature_forward(x, self.b[i])
            for i in range(len(self.sigmas))
        ]
        y = [elem[0] for elem in yb]
        self.fourier_feature_weights = [elem[1] for elem in yb]

        # fully-connected layers
        y = [self._fully_connected_forward(_y) for _y in y]

        # concatenate all the fourier features
        y = paddle.concat(y, axis=1)
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

    def _fourier_feature_forward(self, y, b):
        y = paddle.concat(
            [
                paddle.cos(paddle.matmul(y, b)),
                paddle.sin(paddle.matmul(y, b)),
            ],
            axis=1
        )
        return y, b

    def _fully_connected_forward(self, y):
        for i in range(len(self.linears)):
            y = self.linears[i](y)
            y = self.activation(y)
            if self.dropout_rate > 0:
                y = paddle.nn.functional.dropout(
                    y, p=self.dropout_rate, training=self.training
                )
        return y

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        # TODO: support input transform
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        # TODO: support output transform
        self._output_transform = transform


class STMsFFN(MsFFN):
    """Spatio-temporal multi-scale fourier feature networks.

    References:
        `S. Wang, H. Wang, & P. Perdikaris. On the eigenvector bias of Fourier feature
        networks: From regression to solving multi-scale PDEs with physics-informed
        neural networks. Computer Methods in Applied Mechanics and Engineering, 384,
        113938, 2021 <https://doi.org/10.1016/j.cma.2021.113938>`_.
    """

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        sigmas_x,
        sigmas_t,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
        layer_normalization=None,
        kernel_constraint=None,
        use_bias=True
    ):
        layer_sizes[0] = layer_sizes[0] // 2
        super(STMsFFN, self).__init__(
            layer_sizes,
            activation,
            kernel_initializer,
            sigmas_x + sigmas_t,
            regularization,
            dropout_rate,
            kernel_constraint,
            use_bias
        )
        self.sigmas_x = sigmas_x
        self.sigmas_t = sigmas_t

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            # The last column should be function of t.
            x = self._input_transform(x)

        x0, x1 = paddle.split(x, num_or_sections=2, axis=1)

        # fourier feature layer
        yb_x = [
            self._fourier_feature_forward(x0, self.b[i])
            for i in range(len(self.sigmas_x))
        ]
        yb_t = [
            self._fourier_feature_forward(x1, self.b[len(self.sigmas_x) + i])
            for i in range(len(self.sigmas_t))
        ]
        self.fourier_feature_weights = [elem[1] for elem in yb_x + yb_t]

        # fully-connected layers (reuse)
        y_x = [self._fully_connected_forward(_yb[0]) for _yb in yb_x]
        y_t = [self._fully_connected_forward(_yb[0]) for _yb in yb_t]

        # point-wise multiplication layer
        y = [paddle.multiply(_y_x, _y_t) for _y_x in y_x for _y_t in y_t]

        # concatenate all the fourier features
        y = paddle.concat(y, axis=1)
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)

        return y
