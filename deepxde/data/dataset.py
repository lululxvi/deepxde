import numpy as np

from .data import Data
from .. import config
from .. import utils


class DataSet(Data):
    """Fitting Data set.

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        fname_train=None,
        fname_test=None,
        col_x=None,
        col_y=None,
        standardize=False,
    ):
        if X_train is not None:
            self.train_x = X_train.astype(config.real(np))
            self.train_y = y_train.astype(config.real(np))
            self.test_x = X_test.astype(config.real(np))
            self.test_y = y_test.astype(config.real(np))
        elif fname_train is not None:
            train_data = np.loadtxt(fname_train)
            self.train_x = train_data[:, col_x].astype(config.real(np))
            self.train_y = train_data[:, col_y].astype(config.real(np))
            test_data = np.loadtxt(fname_test)
            self.test_x = test_data[:, col_x].astype(config.real(np))
            self.test_y = test_data[:, col_y].astype(config.real(np))
        else:
            raise ValueError("No training data.")

        self.scaler_x = None
        if standardize:
            self.scaler_x, self.train_x, self.test_x = utils.standardize(
                self.train_x, self.test_x
            )

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y

    def transform_inputs(self, x):
        if self.scaler_x is None:
            return x
        return self.scaler_x.transform(x)
