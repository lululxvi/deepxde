from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepxde as dde
import matplotlib.pyplot as plt
from deepxde.backend import tf
import numpy as np


# generate num equally-spaced points from -1 to 1
def gen_traindata(num):
    xvals = []
    yvals = []

    for i in range(0, num + 1):
        k = 2 / num * i - 1
        xvals.append(k)
        yvals.append(np.cos(k))
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def main():
    def pde(x, y):
        u, q = y[:, 0:1], y[:, 1:2]
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)

        #solution is u(x) = sin(pi*x^2), q(x) = cos(x)
        return (
            -du_xx
            + q * u
            + 2 * np.pi * tf.cos(np.pi * x ** 2)
            - x ** 2 * 4 * np.pi ** 2 * tf.sin(np.pi * x ** 2)
            - tf.cos(x) * tf.sin(np.pi * x ** 2)
        )

    def sol(x):
        return np.sin(np.pi * x ** 2)

    geom = dde.geometry.Interval(-1, 1)

    ob_x, ob_q = gen_traindata(50)

    bc = dde.DirichletBC(geom, sol, lambda x, on_boundary: on_boundary, component=0)
    observe_q = dde.PointSetBC(ob_x, ob_q, component=1)

    data = dde.data.PDE(
        geom,
        pde,
        [bc, observe_q],
        num_domain=100,
        num_boundary=2,
        anchors=ob_x,
        num_test=100,
    )

    net = dde.maps.PFNN([1, [50, 50], [50, 50], [50, 50], 2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)

    losshistory, train_state = model.train(epochs=20000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
