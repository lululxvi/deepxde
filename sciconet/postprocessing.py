from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def plotresult(test, x_dim, y_dim, train):
    plt.figure()
    for i in range(y_dim):
        if train is not None:
            plt.plot(train[:, 0], train[:, x_dim + i], 'ok')
        plt.plot(test[:, 0], test[:, x_dim + i], '-k')
        plt.plot(test[:, 0], test[:, x_dim + y_dim + i], '--r')
        if test.shape[1] > x_dim + 2 * y_dim:
            plt.plot(test[:, 0],
                     test[:, x_dim+y_dim+i]+test[:, x_dim+y_dim*2+i], '-b')
            plt.plot(test[:, 0],
                     test[:, x_dim+y_dim+i]-test[:, x_dim+y_dim*2+i], '-b')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure()
    lims = [np.min(test[:, x_dim:]), np.max(test[:, x_dim:])]
    plt.plot(lims, lims, '-k')
    for i in range(y_dim):
        plt.scatter(test[:, x_dim + i], test[:, x_dim + y_dim + i])
    plt.xlabel('Truth')
    plt.ylabel('Prediction')

    if test.shape[1] > x_dim + 2 * y_dim:
        plt.figure()
        for i in range(y_dim):
            plt.plot(test[:, 0], test[:, x_dim+y_dim*2+i], '-')
        plt.xlabel('x')
        plt.ylabel('std(y)')


def plotloss(loss):
    plt.figure()
    for i in range(1, loss.shape[1]):
        plt.semilogy(loss[:, 0], loss[:, i], label='loss {}'.format(i))
    plt.legend()


def saveplot(loss, test, x_dim, y_dim, train=None, issave=True, isplot=True,
             lowest_err=True):
    if loss is None:
        err = np.linalg.norm(test[:, x_dim: x_dim+y_dim] - test[:, x_dim+y_dim: x_dim+2*y_dim]) \
                / np.linalg.norm(test[:, x_dim+y_dim: x_dim+2*y_dim])
        print('l2 loss:', err)
    if lowest_err:
        if loss is not None:
            totalloss = np.sum(loss[:, 1:-1], axis=1)
            n = totalloss.argmin()
            print('Smallest loss:', loss[n])
    if test.shape[1] > x_dim + 2 * y_dim:
        uncertainty = test[:, x_dim+2*y_dim]
        print('Uncertainty:')
        print('  l2:', np.linalg.norm(uncertainty))
        print('  l_infinity:', np.linalg.norm(uncertainty, ord=np.inf))
        print('  max uncertainty:', test[np.argmax(uncertainty)])
    if issave:
        if loss is not None:
            np.savetxt('loss.dat', loss, header='step, loss')
        if train is not None:
            np.savetxt('train.dat', train, header='x, y')
        np.savetxt('test.dat', test, header='x, truth, pred, std')
    if isplot:
        if loss is not None:
            plotloss(loss)
        plotresult(test, x_dim, y_dim, train)
        plt.show()
