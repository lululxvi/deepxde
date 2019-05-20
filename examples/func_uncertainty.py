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
    num_train = 10
    num_test = 1000
    data = scn.data.Func(geom, func, num_train, num_test)

    x_dim, y_dim = 1, 1
    layer_size = [x_dim] + [50] * 3 + [y_dim]
    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 1e-5]
    dropout_rate = 0.01
    net = scn.maps.FNN(
        layer_size,
        activation,
        initializer,
        regularization=regularization,
        dropout_rate=dropout_rate,
    )

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    model.compile(optimizer, lr, metrics=["l2 relative error"])

    epochs = 30000
    losshistory, train_state = model.train(epochs, uncertainty=True)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
