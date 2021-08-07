from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .data import Data
from ..backend import tf
from ..utils import run_if_any_none


class MfOpDataSet(Data):
    """Multifidelity DeepONet dataset."""

    def __init__(
        self,
        X_lo_train=None,
        X_hi_train=None,
        y_lo_train=None,
        y_hi_train=None,
        X_hi_test=None,
        y_hi_test=None,
    ):
        self.X_lo_train = X_lo_train
        self.X_hi_train = X_hi_train
        self.y_lo_train = y_lo_train
        self.y_hi_train = y_hi_train
        self.X_hi_test = X_hi_test
        self.y_hi_test = y_hi_test

        self.X_train = None
        self.y_train = None

    def losses(self, targets, outputs, loss, model):
        n = tf.cond(model.net.training, lambda: len(self.X_lo_train[0]), lambda: 0)
        loss_lo = loss(targets[0][:n], outputs[0][:n])
        loss_hi = loss(targets[1][n:], outputs[1][n:])
        return [loss_lo, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        self.X_train = (
            np.vstack((self.X_lo_train[0], self.X_hi_train[0])),
            np.vstack((self.X_lo_train[1], self.X_hi_train[1])),
        )
        y_lo_train = np.vstack((self.y_lo_train, np.zeros_like(self.y_hi_train)))
        y_hi_train = np.vstack((np.zeros_like(self.y_lo_train), self.y_hi_train))
        self.y_train = [y_lo_train, y_hi_train]
        return self.X_train, self.y_train

    def test(self):
        return self.X_hi_test, [self.y_hi_test, self.y_hi_test]
