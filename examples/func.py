from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sciconet as scn


def main():
    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return x * np.sin(5 * x)

    geom = scn.geometry.Interval(-1, 1)
    num_train = 16
    num_test = 100
    data = scn.data.Func(geom, func, num_train, num_test)

    x_dim, y_dim = 1, 1
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN([x_dim] + [20] * 3 + [y_dim], activation, initializer)
    # net = scn.maps.ResNet(x_dim, y_dim, 20, 1, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    model.compile(optimizer, lr, metrics=["l2 relative error"])

    epochs = 10000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
