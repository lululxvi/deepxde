from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        return dy_xx - 2

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    def func(x):
        return (x + 1) ** 2

    geom = scn.geometry.Interval(-1, 1)
    bc_l = scn.DirichletBC(geom, func, boundary_l)
    bc_r = scn.NeumannBC(geom, lambda X: 2 * (X + 1), boundary_r)
    data = scn.data.PDE(geom, pde, [bc_l, bc_r], func, 16, 2)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    ntest = 100
    model.compile(optimizer, lr, ntest, metrics=["l2 relative error"])

    epochs = 10000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
