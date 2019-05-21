from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_t = dy_x[:, 0], dy_x[:, 1]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0]
        return (
            dy_t
            - dy_xx
            + tf.exp(-x[:, 1])
            * (tf.sin(np.pi * x[:, 0]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0]))
        )

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    geom = scn.geometry.Interval(-1, 1)
    timedomain = scn.geometry.TimeDomain(0, 1)
    geomtime = scn.geometry.GeometryXTime(geom, timedomain)

    bc = scn.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = scn.IC(geomtime, func, lambda _, on_initial: on_initial)
    num_domain, num_boundary, num_initial = 40, 20, 10
    data = scn.data.TimePDE(
        geomtime,
        1,
        pde,
        [bc, ic],
        num_domain,
        num_boundary,
        num_initial,
        func=func,
        num_test=10000,
    )

    layer_size = [2] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    model.compile(optimizer, lr, metrics=["l2 relative error"])

    epochs = 10000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
