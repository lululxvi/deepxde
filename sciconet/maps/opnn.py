from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from . import activations
from . import initializers
from . import regularizers
from .. import config
from ..utils import timing


class OpNN(object):
    """Operator neural networks
    """

    def __init__(
        self,
        layer_size_function,
        layer_size_location,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        self.layer_size_func = layer_size_function
        self.layer_size_loc = layer_size_location
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self.training = None
        self.dropout = None
        self.data_id = None
        self.X_func = None
        self.X_loc = None
        self.y = None
        self.target = None

        self.build()

    @property
    def inputs(self):
        return [self.X_func, self.X_loc]

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building operator neural network...")
        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.bool)
        self.data_id = tf.placeholder(tf.uint8)
        self.X_func = tf.placeholder(config.real(tf), [None, self.layer_size_func[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_size_loc[0]])

        # Function NN
        assert (
            len(self.layer_size_func) == 3
        ), "Only support function neural network of ONE hidden layer."
        W = tf.Variable(
            tf.truncated_normal(
                [
                    self.layer_size_func[0],
                    self.layer_size_func[2],
                    self.layer_size_func[1],
                ],
                stddev=math.sqrt(1 / self.layer_size_func[0]),
                dtype=config.real(tf),
            )
        )
        b = tf.Variable(tf.zeros([self.layer_size_func[2], self.layer_size_func[1]]))
        y_func = self.activation(tf.einsum("ai,ibk->abk", self.X_func, W) + b)
        W = tf.Variable(
            tf.truncated_normal(
                [self.layer_size_func[2], self.layer_size_func[1]],
                stddev=math.sqrt(1 / self.layer_size_func[1]),
            )
        )
        y_func = tf.einsum("abi,bi->ab", y_func, W)

        # Location NN
        y_loc = self.X_loc
        for i in range(1, len(self.layer_size_loc)):
            y_loc = self.dense(
                y_loc,
                self.layer_size_loc[i],
                activation=self.activation,
                regularizer=self.regularizer,
            )

        # Dot product
        self.y = tf.einsum("bi,bi->b", y_func, y_loc)
        self.y = tf.expand_dims(self.y, axis=1)

        self.target = tf.placeholder(config.real(tf), [None, 1])

    def dense(self, inputs, units, activation=None, use_bias=True, regularizer=None):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
        )
