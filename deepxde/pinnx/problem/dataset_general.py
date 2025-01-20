# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Sequence, Dict

import brainstate as bst
import jax
import numpy as np

from deepxde.pinnx import utils
from .base import Problem

__all__ = [
    'DataSet'
]


class DataSet(Problem):
    """Fitting Problem set.

    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Testing input data.
        y_test (np.ndarray): Testing output data.
        standardize (bool, optional): Standardize input data. Defaults to False.
    """

    def __init__(
        self,
        X_train: Dict[str, bst.typing.ArrayLike],
        y_train: Dict[str, bst.typing.ArrayLike],
        X_test: Dict[str, bst.typing.ArrayLike],
        y_test: Dict[str, bst.typing.ArrayLike],
        standardize: bool = False,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(approximator=approximator, loss_fn=loss_fn, loss_weights=loss_weights)

        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test
        self.scaler_x = None
        if standardize:
            r = jax.tree.map(
                lambda train, test: utils.standardize(train, test),
                self.train_x, self.test_x
            )
            self.train_x = dict()
            self.test_x = dict()
            for key, val in r.items():
                self.train_x[key] = val[0]
                self.test_x[key] = val[1]

    def losses(self, inputs, outputs, targets, **kwargs):
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y
