# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import time

import numpy as np

from nnlearn import DataFunc, FNN, saveplot


def main():
    fname_train = 'train2.txt'
    fname_test = 'test2.txt'

    layer_size = [1] + [20] * 3 + [1]
    activation = 'tanh'
    initializer = 'Glorot normal'
    optimizer = 'adam'
    dropout = None
    batch_normalization = None
    batch_size = 256
    lr = 0.001
    nepoch = 50000
    ntest = 512

    mydata = DataSet(fname_train, fname_test)
    mynn = FNN(layer_size, activation, initializer, optimizer,
               dropout=dropout, batch_normalization=batch_normalization)
    t = time.time()
    _, loss, res = mynn.train(mydata, batch_size, lr, nepoch, ntest,
                              print_model=False)
    print('Cost {} s'.format(time.time() - t))

    saveplot(loss, res, 1, 1, issave=True, isplot=True)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')
