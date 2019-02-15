from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def saveplot(losshistory, train_state, issave=True, isplot=True):
    print('Best training: loss: %s, val_loss: %s, val_metric: %s' % (
          train_state.loss_train, train_state.loss_test, train_state.metrics_test))

    if train_state.best_ystd is not None:
        print('Uncertainty:')
        print('  l2:', np.linalg.norm(train_state.best_ystd))
        print('  l_infinity:', np.linalg.norm(train_state.best_ystd, ord=np.inf))
        print('  max uncertainty location:', train_state.X_test[np.argmax(train_state.best_ystd)])

    if isplot:
        losshistory.plot()
        train_state.plot()
        plt.show()

    if issave:
        losshistory.savetxt('loss.dat')
        train_state.savetxt('train.dat', 'test.dat')