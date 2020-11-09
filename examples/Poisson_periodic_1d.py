from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    def func(x):
        return np.sin(np.pi * x)

    geom = dde.geometry.Interval(-1, 1)
    bc1 = dde.DirichletBC(geom, func, boundary_l)
    bc2 = dde.PeriodicBC(geom, 0, boundary_r)
    data = dde.data.PDE(geom, pde, [bc1, bc2], 16, 2, solution=func, num_test=100)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
