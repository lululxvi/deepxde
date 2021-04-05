from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def ddy(x, y):
        return dde.grad.hessian(y, x)

    def dddy(x, y):
        return tf.gradients(ddy(x, y), x)[0]

    def pde(x, y):
        dy_xxxx = tf.gradients(dddy(x, y), x)[0]
        return dy_xxxx + 1

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    def func(x):
        return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

    geom = dde.geometry.Interval(0, 1)

    bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
    bc3 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

    data = dde.data.PDE(
        geom,
        pde,
        [bc1, bc2, bc3, bc4],
        num_domain=10,
        num_boundary=2,
        solution=func,
        num_test=100,
    )
    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
