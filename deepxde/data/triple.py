from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data import Data
from .sampler import BatchSampler


class Triple(Data):
    """Dataset with each data point as a triple.

    The couple of the first two elements are the input, and the third element is the
    output. This dataset can be used with the network ``DeepONet`` for operator
    learning. Reference: `Lu et al. Learning nonlinear operators via DeepONet based on
    the universal approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        X_train: A tuple of two NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

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


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`). The mini-batch
            is only applied to `N1`.
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if len(X_train[0]) * len(X_train[1]) != y_train.size:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) * len(X_test[1]) != y_test.size:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = BatchSampler(len(X_train[0]), shuffle=True)

    def losses(self, targets, outputs, loss, model):
        return [loss(targets, outputs)]

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y
