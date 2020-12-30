from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    C = tf.Variable(2.0)

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return (
            dy_t
            - C * dy_xx
            + tf.exp(-x[:, 1:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)

    observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
    observe_y = dde.PointSetBC(observe_x, func(observe_x), component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_y],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        anchors=observe_x,
        solution=func,
        num_test=10000,
    )

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    variable = dde.callbacks.VariableValue(C, period=1000)
    losshistory, train_state = model.train(epochs=50000, callbacks=[variable])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
