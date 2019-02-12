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
    data = scn.data.Func(geom, func)

    x_dim, y_dim = 1, 1
    layer_size = [x_dim] + [20] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = scn.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = 'adam'
    lr = 0.001
    batch_size = 16
    ntest = 100
    model.compile(optimizer, lr, batch_size, ntest, metrics=['l2 relative error'])

    epochs = 10000
    losshistory, training_state = model.train(epochs)

    train = np.hstack((data.train_x, data.train_y))
    scn.saveplot(losshistory, np.hstack((training_state.X_test, training_state.y_test, training_state.best_y)),
                 x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
