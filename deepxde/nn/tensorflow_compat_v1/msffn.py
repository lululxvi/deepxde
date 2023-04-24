from .fnn import FNN
from ... import config
from ...backend import tf


class MsFFN(FNN):
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
        dropout_rate=0,
        batch_normalization=None,
        layer_normalization=None,
        kernel_constraint=None,
        use_bias=True,
    ):
        super().__init__(
            layer_sizes,
            activation,
            kernel_initializer,
            regularization=regularization,
            dropout_rate=dropout_rate,
            batch_normalization=batch_normalization,
            layer_normalization=layer_normalization,
            kernel_constraint=kernel_constraint,
            use_bias=use_bias,
        )
        self.sigmas = sigmas  # list or tuple
        self.fourier_feature_weights = None

    def build(self):
        print("Building Multiscale Fourier Feature Network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)

        # fourier feature layer
        yb = [self._fourier_feature_forward(y, sigma) for sigma in self.sigmas]
        y = [elem[0] for elem in yb]
        self.fourier_feature_weights = [elem[1] for elem in yb]
        # fully-connected layers (reuse)
        y = [self._fully_connected_forward(_y) for _y in y]
        # concatenate all the fourier features
        y = tf.concat(y, axis=1)
        self.y = self._dense(y, self.layer_size[-1], use_bias=self.use_bias)

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True

    def _fourier_feature_forward(self, y, sigma):
        b = tf.Variable(
            tf.random_normal(
                [y.shape[1], self.layer_size[1] // 2], dtype=config.real(tf)
            )
            * sigma,
            dtype=config.real(tf),
            trainable=False,
        )
        y = tf.concat(
            [
                tf.cos(tf.matmul(y, b)),
                tf.sin(tf.matmul(y, b)),
            ],
            axis=1,
        )
        return y, b

    def _fully_connected_forward(self, y):
        with tf.variable_scope("fully_connected", reuse=tf.AUTO_REUSE):
            for i in range(1, len(self.layer_size) - 2):
                if (
                    self.batch_normalization is None
                    and self.layer_normalization is None
                ):
                    y = self._dense(
                        y,
                        self.layer_size[i + 1],
                        activation=self.activation,
                        use_bias=self.use_bias,
                    )
                elif self.batch_normalization and self.layer_normalization:
                    raise ValueError(
                        "Can not apply batch_normalization and layer_normalization at "
                        "the same time."
                    )
                elif self.batch_normalization == "before":
                    y = self._dense_batchnorm_v1(y, self.layer_size[i + 1])
                elif self.batch_normalization == "after":
                    y = self._dense_batchnorm_v2(y, self.layer_size[i + 1])
                elif self.layer_normalization == "before":
                    y = self._dense_layernorm_v1(y, self.layer_size[i + 1])
                elif self.layer_normalization == "after":
                    y = self._dense_layernorm_v2(y, self.layer_size[i + 1])
                else:
                    raise ValueError(
                        "batch_normalization: {}, layer_normalization: {}".format(
                            self.batch_normalization, self.layer_normalization
                        )
                        + "\n'before' and 'after' are the only acceptable values for"
                        + " batch_normalization and layer_normalization."
                    )
                if self.dropout_rate > 0:
                    y = tf.layers.dropout(
                        y, rate=self.dropout_rate, training=self.training
                    )
        return y


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
        use_bias=True,
    ):
        super().__init__(
            layer_sizes,
            activation,
            kernel_initializer,
            sigmas_x + sigmas_t,
            regularization,
            dropout_rate,
            batch_normalization,
            layer_normalization,
            kernel_constraint,
            use_bias,
        )
        self.sigmas_x = sigmas_x
        self.sigmas_t = sigmas_t

    def build(self):
        print("Building Spatio-temporal Multi-scale Fourier Feature Network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            # The last column should be function of t.
            y = self._input_transform(y)

        # fourier feature layer
        yb_x = [
            self._fourier_feature_forward(y[:, :-1], sigma) for sigma in self.sigmas_x
        ]
        yb_t = [
            self._fourier_feature_forward(y[:, -1:], sigma) for sigma in self.sigmas_t
        ]
        self.fourier_feature_weights = [elem[1] for elem in yb_x + yb_t]
        # fully-connected layers (reuse)
        y_x = [self._fully_connected_forward(_yb[0]) for _yb in yb_x]
        y_t = [self._fully_connected_forward(_yb[0]) for _yb in yb_t]
        # point-wise multiplication layer
        y = [tf.multiply(_y_x, _y_t) for _y_x in y_x for _y_t in y_t]
        # concatenate all the fourier features
        y = tf.concat(y, axis=1)
        self.y = self._dense(y, self.layer_size[-1], use_bias=self.use_bias)

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True
