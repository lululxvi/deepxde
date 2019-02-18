from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data import Data
from .. import losses
from ..utils import runifnone


class Func(Data):
    """Function approximation.
    """

    def __init__(self, geom, func, dist_train="uniform", online=False):
        self.geom = geom
        self.func = func
        self.dist_train = dist_train
        self.online = online

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, model):
        return [losses.get("MSE")(y_true, y_pred)]

    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.online:
            self.train_x = self.geom.random_points(batch_size, "pseudo")
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            if self.dist_train == "uniform":
                self.train_x = self.geom.uniform_points(batch_size, True)
            else:
                self.train_x = self.geom.random_points(batch_size, "sobol")
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone("test_x", "test_y")
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
