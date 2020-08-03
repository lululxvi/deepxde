from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .fnn import FNN
from .. import config
from ..backend import tf
from ..utils import timing


class PFNN(FNN):
    """Parallel Feed-forward neural networks.

    Feed-forward neural networks that support independent "branches" or sub-network inside the network.

    Args:
        layer_size: A nested list to define the architecture of the neural network (how the layers are connected).
            If `layer_size[i]` is int, it represent one layer shared by all the outputs;
            if `layer_size[i]` is list, it represent `len(layer_size[i])` sub-layers, each of which exclusively used by one output.
            Note that `len(layer_size[i])` should equal to the number of outputs.
            Every number specify the number of neurons of that layer.
    """

    def __init__(
        self,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_normalization=None,
    ):
        super(PFNN, self).__init__(
            layer_size,
            activation,
            kernel_initializer,
            regularization,
            dropout_rate,
            batch_normalization,
        )

    @timing
    def build(self):
        def layer_map(_y, layer_size, net):
            if net.batch_normalization is None:
                _y = net.dense(_y, layer_size, activation=net.activation)
            elif net.batch_normalization == "before":
                _y = net.dense_batchnorm_v1(_y, layer_size)
            elif net.batch_normalization == "after":
                _y = net.dense_batchnorm_v2(_y, layer_size)
            else:
                raise ValueError("batch_normalization")
            if net.dropout_rate > 0:
                _y = tf.layers.dropout(_y, rate=net.dropout_rate, training=net.dropout)
            return _y

        print("Building feed-forward neural network...")
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])

        y = self.x
        if self._input_transform is not None:
            y = self._input_transform(y)
        # hidden layers
        for i_layer in range(len(self.layer_size) - 2):
            if isinstance(self.layer_size[i_layer + 1], (list, tuple)):
                if isinstance(y, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    if len(self.layer_size[i_layer + 1]) != len(
                        self.layer_size[i_layer]
                    ):
                        raise ValueError(
                            "Number of sub-layers should be the same when feed-forwarding"
                        )
                    y = [
                        layer_map(y[i_net], self.layer_size[i_layer + 1][i_net], self)
                        for i_net in range(len(self.layer_size[i_layer + 1]))
                    ]
                else:
                    # e.g. 64 -> [8, 8, 8]
                    y = [
                        layer_map(y, self.layer_size[i_layer + 1][i_net], self)
                        for i_net in range(len(self.layer_size[i_layer + 1]))
                    ]
            else:
                # e.g. 64 -> 64
                y = layer_map(y, self.layer_size[i_layer + 1], self)
        # output layers
        if isinstance(y, (list, tuple)):
            # e.g. [3, 3, 3] -> 3
            if len(self.layer_size[-2]) != self.layer_size[-1]:
                raise ValueError(
                    "Number of sub-layers should be the same as number of outputs"
                )
            y = [self.dense(y[i_net], 1) for i_net in range(len(y))]
            self.y = tf.concat(y, axis=1)
        else:
            self.y = self.dense(y, self.layer_size[-1])

        if self._output_transform is not None:
            self.y = self._output_transform(self.x, self.y)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])
        self.built = True
