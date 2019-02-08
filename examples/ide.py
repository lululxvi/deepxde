# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from nnlearn import DataIDE, FNN, Interval, saveplot


def main():
    def ide(x, y, int_mat):
        """int_0^x y(t)dt
        """
        lhs1 = tf.matmul(int_mat, y)
        lhs2 = tf.gradients(y, x)[0]
        rhs = 2 * np.pi * tf.cos(2 * np.pi * x) + tf.sin(np.pi * x)**2 / np.pi
        return lhs1 + (lhs2 - rhs)[:tf.size(lhs1)]

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return np.sin(2 * np.pi * x)

    x_dim, y_dim = 1, 1
    geom = Interval(0, 1)
    quad_deg = 16

    layer_size = [x_dim] + [20] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    optimizer = 'adam'
    batch_normalization = None
    nbc, ninside = 2, 14
    lr = 0.001
    nepoch = 100000
    ntest = 128

    mydata = DataIDE(ide, func, geom, nbc, quad_deg)
    mynn = FNN(layer_size, activation, initializer, optimizer,
               batch_normalization=batch_normalization)
    t = time.time()
    _, loss, res = mynn.train(mydata, nbc + ninside, lr, nepoch, ntest)
    print('Cost {} s'.format(time.time() - t))

    res = res[:ntest]
    res = res[res[:, 0].argsort()]
    saveplot(loss, res, x_dim, y_dim, issave=True, isplot=True)


if __name__ == '__main__':
    main()
