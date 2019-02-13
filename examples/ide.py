from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


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

    geom = scn.geometry.Interval(0, 1)

    nbc = 2
    quad_deg = 16
    data = scn.data.IDE(geom, ide, func, nbc, quad_deg)

    x_dim, y_dim = 1, 1
    layer_size = [x_dim] + [20] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = scn.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = 'adam'
    lr = 0.001
    batch_size = 16
    ntest = 128
    model.compile(optimizer, lr, batch_size, ntest, metrics=['l2 relative error'])

    epochs = 10000
    losshistory, training_state = model.train(epochs)

    train = np.hstack((data.train_x, data.train_y))
    scn.saveplot(losshistory, np.hstack((training_state.X_test, training_state.y_test, training_state.best_y)),
                 x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
