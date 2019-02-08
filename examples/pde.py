# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from nnlearn import DataPDE, FNN, Hypercube, Interval, saveplot


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        # return -dy_xx + tf.exp(-x) * x * (x**2 - 6 * x + 6)
        return -dy_xx - np.pi**2 * tf.sin(np.pi * x)
        # dy_x, dy_t = dy_x[:, 0], dy_x[:, 1]
        # dy_xx = tf.gradients(dy_x, x)[0][:, 0]
        # return dy_t - dy_xx + tf.exp(-x[:, 1])*(tf.sin(np.pi*x[:, 0]) - np.pi**2*tf.sin(np.pi*x[:, 0]))

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        # return np.exp(-x) * x**3
        return np.sin(np.pi * x)
        # return np.sin(np.pi*x[:, 0:1])*np.exp(-x[:, 1:])

    x_dim, y_dim = 1, 1
    geom = Interval(-1, 1)

    anchors = [[-1], [1]]
    batch_size = len(anchors) + 8
    ntest = 100

    activation = 'tanh'
    layer_size = [x_dim] + [50] * 3 + [y_dim]
    dropout = 0.01
    regularization = ['l2', 1e-6]
    lossweight = np.array([len(anchors), batch_size-len(anchors)]) * np.array([10, 1])
    uncertainty = True

    initializer = 'Glorot uniform'
    optimizer = 'adam'
    lr = 0.0002
    nepoch = 40000

    mydata = DataPDE(pde, func, geom, anchors)
    mynn = FNN(layer_size, activation, initializer, optimizer, dropout=dropout)
    t = time.time()
    _, loss, res = mynn.train(
        mydata, batch_size, lr, nepoch, ntest, uncertainty=uncertainty,
        regularization=regularization, lossweight=lossweight)
    print('Cost {} s'.format(time.time() - t))

    res = res[res[:, 0].argsort()]
    train = np.hstack((mydata.train_x, mydata.train_y))
    saveplot(loss, res, x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
