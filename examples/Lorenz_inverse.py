from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def gen_traindata():
    data = np.load("dataset/Lorenz.npz")
    return data["t"], data["y"]


def main():
    C1 = tf.Variable(1.0)
    C2 = tf.Variable(1.0)
    C3 = tf.Variable(1.0)

    def Lorenz_system(x, y):
        """Lorenz system.
        dy1/dx = 10 * (y2 - y1)
        dy2/dx = y1 * (28 - y3) - y2
        dy3/dx = y1 * y2 - 8/3 * y3
        """
        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
        dy1_x = dde.grad.jacobian(y, x, i=0)
        dy2_x = dde.grad.jacobian(y, x, i=1)
        dy3_x = dde.grad.jacobian(y, x, i=2)
        return [
            dy1_x - C1 * (y2 - y1),
            dy2_x - y1 * (C2 - y3) + y2,
            dy3_x - y1 * y2 + C3 * y3,
        ]

    def boundary(_, on_initial):
        return on_initial

    geom = dde.geometry.TimeDomain(0, 3)

    # Initial conditions
    ic1 = dde.IC(geom, lambda X: -8, boundary, component=0)
    ic2 = dde.IC(geom, lambda X: 7, boundary, component=1)
    ic3 = dde.IC(geom, lambda X: 27, boundary, component=2)

    # Get the train data
    observe_t, ob_y = gen_traindata()
    observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
    observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
    observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

    data = dde.data.PDE(
        geom,
        Lorenz_system,
        [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
        num_domain=400,
        num_boundary=2,
        anchors=observe_t,
    )

    net = dde.maps.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    variable = dde.callbacks.VariableValue(
        [C1, C2, C3], period=600, filename="variables.dat"
    )
    losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
