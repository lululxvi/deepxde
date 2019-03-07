from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sciconet as scn


def main():
    def func_lo(x):
        A, B, C = 0.5, 10, -5
        return A * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5) + C

    def func_hi(x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    geom = scn.geometry.Interval(0, 1)
    data = scn.data.MfFunc(geom, func_lo, func_hi, 5)

    x_dim, y_dim = 1, 1
    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 0.01]
    net = scn.maps.MfNN(
        [x_dim] + [20] * 4 + [y_dim],
        [10] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
    )

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    batch_size = 51
    ntest = 1000
    model.compile(optimizer, lr, batch_size, ntest, metrics=["l2 relative error"])

    epochs = 80000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
