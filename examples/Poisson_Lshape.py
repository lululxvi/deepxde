from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        dy_yy = tf.gradients(dy_y, x)[0][:, 1:]
        return -dy_xx - dy_yy - 1

    def boundary(_, on_boundary):
        return on_boundary

    def func(x):
        return np.zeros([len(x), 1])

    geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
    bc = dde.DirichletBC(geom, func, boundary)

    data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
    net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=50000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
