from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

import sciconet as scn


def main():
    fname_train = 'examples/dataset.train'
    fname_test = 'examples/dataset.test'
    data = scn.data.DataSet(fname_train=fname_train, fname_test=fname_test, col_x=(0,), col_y=(1,))

    x_dim, y_dim = 1, 1
    layer_size = [x_dim] + [50] * 3 + [y_dim]
    activation = 'tanh'
    initializer = 'Glorot normal'
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = 'adam'
    lr = 0.001
    batch_size = 0
    ntest = 0
    model.compile(optimizer, lr, batch_size, ntest, metrics=['l2 relative error'])

    epochs = 50000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == '__main__':
    main()
