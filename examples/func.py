from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde


def main():
    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return x * np.sin(5 * x)

    geom = dde.geometry.Interval(-1, 1)
    num_train = 16
    num_test = 100
    data = dde.data.Func(geom, func, num_train, num_test)

    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN([1] + [20] * 3 + [1], activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
