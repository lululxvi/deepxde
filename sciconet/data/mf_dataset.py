from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

from .data import Data
from .. import losses


class MfDataSet(Data):
    """Multifidelity function approximation from data set.
    """

    def __init__(
        self,
        X_lo_train=None,
        X_hi_train=None,
        y_lo_train=None,
        y_hi_train=None,
        X_hi_test=None,
        y_hi_test=None,
        fname_lo_train=None,
        fname_hi_train=None,
        fname_hi_test=None,
        col_x=None,
        col_y=None,
    ):
        """
        # Arguments
            col_x: List of integers
            col_y: List of integers.
        """
        if X_lo_train is not None:
            self.X_lo_train = X_lo_train
            self.X_hi_train = X_hi_train
            self.y_lo_train = y_lo_train
            self.y_hi_train = y_hi_train
            self.X_hi_test = X_hi_test
            self.y_hi_test = y_hi_test
        elif fname_lo_train is not None:
            data = np.loadtxt(fname_lo_train)
            self.X_lo_train = data[:, col_x]
            self.y_lo_train = data[:, col_y]
            data = np.loadtxt(fname_hi_train)
            self.X_hi_train = data[:, col_x]
            self.y_hi_train = data[:, col_y]
            data = np.loadtxt(fname_hi_test)
            self.X_hi_test = data[:, col_x]
            self.y_hi_test = data[:, col_y]
        else:
            raise ValueError("No training data.")

        self.X_train = None
        self.scaler_x = None
        self._standardize()

    def losses(self, targets, outputs, loss, model):
        n = len(self.X_lo_train)
        loss_lo = losses.get(loss)(targets[0][:n], outputs[0][:n])
        loss_hi = losses.get(loss)(targets[1][n:], outputs[1][n:])
        return [loss_lo, loss_hi]

    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.X_train is not None:
            return self.X_train, [self.y_lo_train, self.y_hi_train]

        self.X_train = np.vstack((self.X_lo_train, self.X_hi_train))
        self.y_lo_train, self.y_hi_train = (
            np.vstack((self.y_lo_train, np.zeros_like(self.y_hi_train))),
            np.vstack((np.zeros_like(self.y_lo_train), self.y_hi_train)),
        )
        return self.X_train, [self.y_lo_train, self.y_hi_train]

    def test(self, n, *args, **kwargs):
        return self.X_hi_test, [self.y_hi_test, self.y_hi_test]

    def _standardize(self):
        self.scaler_x = preprocessing.StandardScaler(with_mean=True, with_std=True)
        self.X_lo_train = self.scaler_x.fit_transform(self.X_lo_train)
        self.X_hi_train = self.scaler_x.transform(self.X_hi_train)
        self.X_hi_test = self.scaler_x.transform(self.X_hi_test)
