from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sciconet import DataFunc, FNN, Interval, Model, saveplot


def main():
    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return x * np.sin(5 * x)

    x_dim, y_dim = 1, 1
    geom = Interval(-1, 1)
    mydata = DataFunc(func, geom)

    layer_size = [x_dim] + [20] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = FNN(layer_size, activation, initializer)

    model = Model(mydata, net)

    optimizer = 'adam'
    lr = 0.001
    batch_size = 16
    ntest = 100
    model.compile(optimizer, lr, batch_size, ntest)

    nepoch = 10000
    losshistory, training_state = model.train(nepoch)

    train = np.hstack((mydata.train_x, mydata.train_y))
    saveplot(losshistory, np.hstack((training_state.X_test, training_state.y_test, training_state.best_y)),
             x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
