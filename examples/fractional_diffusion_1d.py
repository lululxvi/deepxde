from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import gamma

import deepxde as dde
from deepxde.backend import tf


def main():
    alpha = 1.8

    def fpde(x, y, dy_t, int_mat):
        """du/dt + (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)
        """
        if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
            int_mat = tf.SparseTensor(*int_mat)
            lhs = -tf.sparse_tensor_dense_matmul(int_mat, y)
        else:
            lhs = -tf.matmul(int_mat, y)
        x, t = x[:, :-1], x[:, -1:]
        rhs = -dy_t - tf.exp(-t) * (
            x ** 3 * (1 - x) ** 3
            + gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
            - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
            + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
            - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
        )
        return lhs - rhs[: tf.size(lhs)]

    def func(x):
        x, t = x[:, :-1], x[:, -1:]
        return np.exp(-t) * x ** 3 * (1 - x) ** 3

    geom = dde.geometry.Interval(0, 1)
    t_min, t_max = 0, 1

    # Static auxiliary points
    disc = dde.data.fpde.Discretization(1, 'static', [50])
    data = dde.data.TimeFPDE(
        fpde, alpha, func, geom, t_min, t_max, disc, batch_size=400, ntest=400
    )
    # Dynamic auxiliary points
    # disc = dde.data.fpde.Discretization(1, "dynamic", [100])
    # data = dde.data.TimeFPDE(
    #     fpde, alpha, func, geom, t_min, t_max, disc, batch_size=200, ntest=500
    # )

    net = dde.maps.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(
        lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * y
        + x[:, 0:1] ** 3 * (1 - x[:, 0:1]) ** 3
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(epochs=10000)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    X = geomtime.random_points(1000)
    y_true = func(X)
    y_pred = model.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


if __name__ == "__main__":
    main()
