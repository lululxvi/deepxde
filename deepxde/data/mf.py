import numpy as np

from .data import Data
from ..backend import tf
from ..utils import run_if_any_none, standardize


class MfFunc(Data):
    """Multifidelity function approximation."""

    def __init__(
        self, geom, func_lo, func_hi, num_lo, num_hi, num_test, dist_train="uniform"
    ):
        self.geom = geom
        self.func_lo = func_lo
        self.func_hi = func_hi
        self.num_lo = num_lo
        self.num_hi = num_hi
        self.num_test = num_test
        self.dist_train = dist_train

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        loss_lo = loss_fn(targets[0][: self.num_lo], outputs[0][: self.num_lo])
        loss_hi = loss_fn(targets[1][self.num_lo :], outputs[1][self.num_lo :])
        return [loss_lo, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        if self.dist_train == "uniform":
            self.X_train = np.vstack(
                (
                    self.geom.uniform_points(self.num_lo, True),
                    self.geom.uniform_points(self.num_hi, True),
                )
            )
        else:
            self.X_train = np.vstack(
                (
                    self.geom.random_points(self.num_lo, random=self.dist_train),
                    self.geom.random_points(self.num_hi, random=self.dist_train),
                )
            )
        y_lo_train = self.func_lo(self.X_train)
        y_hi_train = self.func_hi(self.X_train)
        self.y_train = [y_lo_train, y_hi_train]
        return self.X_train, self.y_train

    @run_if_any_none("X_test", "y_test")
    def test(self):
        self.X_test = self.geom.uniform_points(self.num_test, True)
        y_lo_test = self.func_lo(self.X_test)
        y_hi_test = self.func_hi(self.X_test)
        self.y_test = [y_lo_test, y_hi_test]
        return self.X_test, self.y_test


class MfDataSet(Data):
    """Multifidelity function approximation from data set.

    Args:
        col_x: List of integers.
        col_y: List of integers.
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
        standardize=False,
    ):
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
        self.y_train = None

        self.scaler_x = None
        if standardize:
            self._standardize()

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        n = len(self.X_lo_train)
        loss_lo = loss_fn(targets[0][:n], outputs[0][:n])
        loss_hi = loss_fn(targets[1][n:], outputs[1][n:])
        return [loss_lo, loss_hi]

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return [0, loss_fn(targets[1], outputs[1])]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        self.X_train = np.vstack((self.X_lo_train, self.X_hi_train))
        self.y_lo_train, self.y_hi_train = (
            np.vstack((self.y_lo_train, np.zeros_like(self.y_hi_train))),
            np.vstack((np.zeros_like(self.y_lo_train), self.y_hi_train)),
        )
        self.y_train = [self.y_lo_train, self.y_hi_train]
        return self.X_train, self.y_train

    def test(self):
        return self.X_hi_test, [self.y_hi_test, self.y_hi_test]

    def _standardize(self):
        self.scaler_x, self.X_lo_train, self.X_hi_train = standardize(
            self.X_lo_train, self.X_hi_train
        )
        self.X_hi_test = self.scaler_x.transform(self.X_hi_test)
