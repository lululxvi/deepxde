from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def ode_system(x, y):
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        y1, y2 = y[:, 0:1], y[:, 1:]
        dy1_x = tf.gradients(y1, x)[0]
        dy2_x = tf.gradients(y2, x)[0]
        return [dy1_x - y2, dy2_x + y1]

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x)))

    geom = scn.geometry.Interval(0, 10)
    data = scn.data.PDE(geom, ode_system, func, 0, anchors=[[0]])

    x_dim, y_dim = 1, 2
    layer_size = [x_dim] + [50] * 3 + [y_dim]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    batch_size = 30
    ntest = 100
    model.compile(optimizer, lr, batch_size, ntest, metrics=["l2 relative error"])

    epochs = 20000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
