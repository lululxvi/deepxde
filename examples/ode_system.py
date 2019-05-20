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

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x)))

    geom = scn.geometry.Interval(0, 10)
    bc1 = scn.DirichletBC(geom, lambda X: np.sin(X), boundary, component=0)
    bc2 = scn.DirichletBC(geom, lambda X: np.cos(X), boundary, component=1)
    data = scn.data.PDE(geom, 2, ode_system, [bc1, bc2], 35, 2, func=func, num_test=100)

    layer_size = [1] + [50] * 3 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    model.compile(optimizer, lr, metrics=["l2 relative error"])

    epochs = 20000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
