from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def ide(x, y, int_mat):
        """int_0^x y(t)dt
        """
        lhs1 = tf.matmul(int_mat, y)
        lhs2 = tf.gradients(y, x)[0]
        rhs = 2 * np.pi * tf.cos(2 * np.pi * x) + tf.sin(np.pi * x) ** 2 / np.pi
        return lhs1 + (lhs2 - rhs)[: tf.size(lhs1)]

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return np.sin(2 * np.pi * x)

    geom = scn.geometry.Interval(0, 1)
    bc = scn.DirichletBC(geom, func, boundary)

    quad_deg = 16
    data = scn.data.IDE(
        geom, ide, bc, quad_deg, num_domain=16, num_boundary=2, func=func
    )

    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    X = geom.uniform_points(100, True)
    y_true = func(X)
    y_pred = model.predict(X)
    print("L2 relative error:", scn.metrics.l2_relative_error(y_true, y_pred))

    plt.figure()
    plt.plot(X, y_true, "-")
    plt.plot(X, y_pred, "o")
    plt.show()
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


if __name__ == "__main__":
    main()
