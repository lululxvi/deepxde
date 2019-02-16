from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from . import activations
from . import initializers
from . import regularizers
from .. import config
from ..utils import timing


class FNN(object):
    """feed-forward neural networks
    """

    def __init__(self, layer_size, activation, kernel_initializer,
                 regularization=None, dropout_rate=0, batch_normalization=None):
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        self.training, self.dropout = None, None
        self.data_id = None  # 0: train data, 1: test data
        self.x, self.y, self.y_ = None, None, None
        self.build()

    @timing
    def build(self):
        print('Building feed-forward neural network...')
        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.bool)
        self.data_id = tf.placeholder(tf.uint8)
        self.x = tf.placeholder(config.real(tf), [None, self.layer_size[0]])
        y = self.x
        for i in range(len(self.layer_size) - 2):
            y = self.add_layer(y, self.layer_size[i + 1], False, self.training)
            if self.dropout_rate > 0:
                y = tf.layers.dropout(y, rate=self.dropout_rate, training=self.dropout)
        self.y = self.add_layer(y, self.layer_size[-1], True, self.training)

        self.y_ = tf.placeholder(config.real(tf), [None, self.layer_size[-1]])

    def dense(self, inputs, units, activation=None, use_bias=True,
              kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=None, bias_regularizer=None,
              activity_regularizer=None, kernel_constraint=None,
              bias_constraint=None, trainable=True, name=None, reuse=None):
        if False:
            shape = inputs.get_shape().as_list()
            fan_in = shape[1]
            W = tf.Variable(tf.random_normal([fan_in, units],
                                             stddev=math.sqrt(2 / fan_in)))
            # weight normalization
            g = tf.Variable(tf.ones(units))
            W = tf.nn.l2_normalize(W, axis=0) * g
            y = tf.matmul(inputs, W)
            if use_bias:
                b = tf.Variable(tf.zeros(units))
                y += b
            if activation is not None:
                return activation(y)
            return y
        return tf.layers.dense(
            inputs, units, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, trainable=trainable, name=name,
            reuse=reuse)

    def add_layer(self, inputs, units, last_layer, training):
        if last_layer:
            return self.dense(
                inputs, units, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
        if self.batch_normalization is None:
            return self.dense(
                inputs, units, activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
        if self.batch_normalization == 'before':
            # FC - BN - activation
            y = self.dense(
                inputs, units, use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.regularizer)
            y = tf.layers.batch_normalization(y, training=training)
            return self.activation(y)
        elif self.batch_normalization == 'after':
            # FC - activation - BN
            y = self.dense(
                inputs, units, activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            return tf.layers.batch_normalization(y, training=training)
        else:
            raise ValueError('batch_normalization')
