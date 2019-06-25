from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data import Data
from ..utils import run_if_any_none


class Func(Data):
    """Function approximation.
    """

    def __init__(
        self, geom, func, num_train, num_test, dist_train="uniform", online=False
    ):
        self.geom = geom
        self.func = func
        self.num_train = num_train
        self.num_test = num_test
        self.dist_train = dist_train
        self.online = online

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, targets, outputs, loss, model):
        return [loss(targets, outputs)]

    def train_next_batch(self, batch_size=None):
        if self.online:
            self.train_x = self.geom.random_points(batch_size, "pseudo")
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            if self.dist_train == "uniform":
                self.train_x = self.geom.uniform_points(self.num_train, True)
            else:
                self.train_x = self.geom.random_points(self.num_train, "sobol")
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.geom.uniform_points(self.num_test, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
