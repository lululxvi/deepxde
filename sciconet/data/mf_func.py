from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .data import Data
from .. import losses


class MfFunc(Data):
    """Multifidelity function approximation.
    """

    def __init__(self, geom, func_lo, func_hi, num_hi, dist_train="uniform"):
        self.geom = geom
        self.func_lo = func_lo
        self.func_hi = func_hi
        self.num_hi = num_hi
        self.dist_train = dist_train

        self.X_train = None
        self.y_lo_train = None
        self.y_hi_train = None
        self.X_test = None
        self.y_lo_test = None
        self.y_hi_test = None

    def losses(self, targets, outputs, loss, model):
        loss_lo = losses.get(loss)(
            targets[0][: model.batch_size], outputs[0][: model.batch_size]
        )
        loss_hi = losses.get(loss)(
            targets[1][model.batch_size :], outputs[1][model.batch_size :]
        )
        return [loss_lo, loss_hi]

    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.X_train is not None:
            return self.X_train, [self.y_lo_train, self.y_hi_train]

        if self.dist_train == "uniform":
            self.X_train = np.vstack(
                (
                    self.geom.uniform_points(batch_size, True),
                    self.geom.uniform_points(self.num_hi, True),
                )
            )
        else:
            self.X_train = np.vstack(
                (
                    self.geom.random_points(batch_size, "sobol"),
                    self.geom.random_points(self.num_hi, "sobol"),
                )
            )
        self.y_lo_train = self.func_lo(self.X_train)
        self.y_hi_train = self.func_hi(self.X_train)
        return self.X_train, [self.y_lo_train, self.y_hi_train]

    def test(self, n, *args, **kwargs):
        if self.X_test is not None:
            return self.X_test, [self.y_lo_test, self.y_hi_test]

        self.X_test = self.geom.uniform_points(n, True)
        self.y_lo_test = self.func_lo(self.X_test)
        self.y_hi_test = self.func_hi(self.X_test)
        return self.X_test, [self.y_lo_test, self.y_hi_test]
