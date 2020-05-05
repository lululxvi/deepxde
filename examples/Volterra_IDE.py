from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def ide(x, y, int_mat):
        rhs = tf.matmul(int_mat, y)
        lhs1 = tf.gradients(y, x)[0]
        return (lhs1 + y)[: tf.size(rhs)] - rhs

    def kernel(x, s):
        return np.exp(s - x)

    def func(x):
        return np.exp(-x) * np.cosh(x)

    geom = dde.geometry.TimeDomain(0, 5)
    ic = dde.IC(geom, func, lambda _, on_initial: on_initial)

    quad_deg = 20
    data = dde.data.IDE(
        geom,
        ide,
        ic,
        quad_deg,
        kernel=kernel,
        num_domain=10,
        num_boundary=2,
        train_distribution="uniform",
    )

    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("L-BFGS-B")
    model.train()

    X = geom.uniform_points(100)
    y_true = func(X)
    y_pred = model.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

    plt.figure()
    plt.plot(X, y_true, "-")
    plt.plot(X, y_pred, "o")
    plt.show()
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


if __name__ == "__main__":
    main()
