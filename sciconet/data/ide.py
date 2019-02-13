from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .. import config
from .. import losses
from ..utils import runifnone


class IDE(Data):
    """IDE solver.
    The current version only supports the 1D problem with initial condition at x = 0.
    """

    def __init__(self, geom, ide, func, nbc, quad_deg):
        self.geom = geom
        self.ide = ide
        self.func = func
        self.nbc = nbc
        self.quad_deg = quad_deg

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)

    def losses(self, y_true, y_pred, model):
        int_mat_train = self.get_int_matrix(model.batch_size, True)
        int_mat_test = self.get_int_matrix(model.ntest, False)
        f = tf.cond(tf.equal(tf.shape(y_pred)[0], np.shape(int_mat_train)[-1]),
                    lambda: self.ide(model.net.x, y_pred, int_mat_train),
                    lambda: self.ide(model.net.x, y_pred, int_mat_test))
        return [losses.get('MSE')(y_true[:self.nbc], y_pred[:self.nbc]),
                losses.get('MSE')(tf.zeros(tf.shape(f)), f)]

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y = self.gen_data(batch_size)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x, self.test_y = self.gen_data(n)
        return self.test_x, self.test_y

    def get_int_matrix(self, size, training):
        def get_quad_weights(x):
            return self.quad_w * x / 2

        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            x = self.train_x
        else:
            if self.test_x is None:
                self.test(size)
            x = self.test_x
        int_mat = np.zeros((size, x.size), dtype=config.real(np))
        for i in range(size):
            int_mat[i, size+self.quad_deg*i: size + self.quad_deg*(i+1)] = \
                get_quad_weights(x[i, 0])
        return int_mat

    def gen_data(self, size):
        def get_quad_points(x):
            return (self.quad_x + 1) * x / 2

        x = self.geom.uniform_points(size, True)
        quad_x = np.hstack(map(lambda xi: get_quad_points(xi[0]), x))
        x = np.vstack((x, quad_x[:, None]))
        return x, self.func(x)