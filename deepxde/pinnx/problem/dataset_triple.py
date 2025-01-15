# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Sequence

import brainstate as bst

from pinnx.utils.sampler import BatchSampler
from .base import Problem

__all__ = [
    'TripleDataset',
    'TripleCartesianProd'
]


class TripleDataset(Problem):
    """
    Dataset with each data point as a triple.

    The couple of the first two elements are the input, and the third element is the
    output. This dataset can be used with the network ``DeepONet`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays.
        y_train: A NumPy array.

    References:
        `L. Lu, P. Jin, G. Pang, Z. Zhang, & G. E. Karniadakis. Learning nonlinear
        operators via DeepONet based on the universal approximation theorem of
        operators. Nature Machine Intelligence, 3, 218--229, 2021
        <https://doi.org/10.1038/s42256-021-00302-5>`_.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, inputs, outputs, targets, **kwargs):
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices],
             self.train_x[1][indices]),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y


class TripleCartesianProd(Problem):
    """
    Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``TripleDataset`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )

        if len(X_train[0]) != y_train.shape[0] or len(X_train[1]) != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) != y_test.shape[0] or len(X_test[1]) != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, inputs, outputs, targets, **kwargs):
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            (self.train_x[0][indices_branch],
             self.train_x[1][indices_trunk],),
            self.train_y[indices_branch, indices_trunk]
        )

    def test(self):
        return self.test_x, self.test_y
