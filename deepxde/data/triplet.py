from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

from .data import Data
from .sampler import BatchSampler


class Triplet(Data):
    """Dataset with each data point as a triplet.

    This dataset can be used with the network ``DeepONet`` for operator learning. `Lu et al. Learning nonlinear
    operators via DeepONet based on the universal approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        X_train: A tuple of two NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        standardize=False,
    ):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.scaler_x = None
        if standardize:
            self._standardize()

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss, model):
        return [loss(targets, outputs)]

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices]),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y

    def transform_inputs(self, x):
        if self.scaler_x is None:
            return x
        return list(map(lambda scaler, xi: scaler.transform(xi), self.scaler_x, x))

    def _standardize(self):
        def standardize_one(X1, X2):
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
            X1 = scaler.fit_transform(X1)
            X2 = scaler.transform(X2)
            return scaler, X1, X2

        self.scaler_x, self.train_x, self.test_x = zip(
            *map(standardize_one, self.train_x, self.test_x)
        )
