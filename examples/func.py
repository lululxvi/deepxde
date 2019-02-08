# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import time

import numpy as np

from nnlearn import DataFunc, FNN, Interval, saveplot


def main():
    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        # return np.abs(x)
        # return x * np.sin(5 * x)
        # return np.heaviside(x, 0.5) + np.sin(5 * x) / 10
        # return np.abs(np.vstack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))).T
        return np.exp(-x) * x**3

    x_dim, y_dim = 1, 1
    geom = Interval(0, 1)

    layer_size = [x_dim] + [20] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    optimizer = 'adam'
    dropout = 0.1
    regularization = ['l2', 1e-3]
    batch_normalization = None
    batch_size = 16
    lr = 0.001
    nepoch = 100000
    ntest = 100

    mydata = DataFunc(func, geom)
    mynn = FNN(layer_size, activation, initializer, optimizer,
               dropout=dropout, batch_normalization=batch_normalization)
    t = time.time()
    _, loss, res = mynn.train(mydata, batch_size, lr, nepoch, ntest,
                              uncertainty=True, regularization=regularization)
    print('Cost {} s'.format(time.time() - t))

    train = np.hstack((mydata.train_x, mydata.train_y))
    saveplot(loss, res, x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')
