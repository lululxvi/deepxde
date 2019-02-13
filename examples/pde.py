from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        return -dy_xx - np.pi**2 * tf.sin(np.pi * x)
        # dy_x, dy_t = dy_x[:, 0], dy_x[:, 1]
        # dy_xx = tf.gradients(dy_x, x)[0][:, 0]
        # return dy_t - dy_xx + tf.exp(-x[:, 1])*(tf.sin(np.pi*x[:, 0]) - np.pi**2*tf.sin(np.pi*x[:, 0]))

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return np.sin(np.pi * x)
        # return np.sin(np.pi*x[:, 0:1])*np.exp(-x[:, 1:])

    geom = scn.geometry.Interval(-1, 1)

    nbc = 2
    data = scn.data.PDE(geom, pde, func, nbc)

    x_dim, y_dim = 1, 1
    layer_size = [x_dim] + [50] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = scn.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = 'adam'
    lr = 0.001
    batch_size = 16
    ntest = 100
    model.compile(optimizer, lr, batch_size, ntest, metrics=['l2 relative error'])

    epochs = 10000
    losshistory, training_state = model.train(epochs)

    train = np.hstack((data.train_x, data.train_y))
    scn.saveplot(losshistory, np.hstack((training_state.X_test, training_state.y_test, training_state.best_y)),
                 x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
