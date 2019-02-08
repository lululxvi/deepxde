from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

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

    optimizer = 'adam'
    model = Model(net, optimizer)

    batch_size = 16
    lr = 0.001
    nepoch = 10000
    ntest = 100

    t = time.time()
    _, loss, _, res = model.train(mydata, batch_size, lr, nepoch, ntest)
    print('Cost {} s'.format(time.time() - t))

    train = np.hstack((mydata.train_x, mydata.train_y))
    saveplot(loss, res, x_dim, y_dim, train=train, issave=True, isplot=True)


if __name__ == '__main__':
    main()
