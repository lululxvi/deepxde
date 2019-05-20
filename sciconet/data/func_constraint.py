from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .. import losses
from ..utils import runifnone


class FuncConstraint(Data):
    """Function approximation with constraints.
    """

    def __init__(
        self, geom, constraint, func, num_train, anchors, dist_train="uniform"
    ):
        self.geom = geom
        self.constraint = constraint
        self.func = func
        self.num_train = num_train
        self.anchors = anchors
        self.dist_train = dist_train

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, loss, model):
        if self.train_x is None:
            self.train_next_batch(self.num_train)
        if self.test_x is None:
            self.test(model.ntest)

        n = 0
        if self.anchors is not None:
            n += len(self.anchors)

        f = tf.cond(
            tf.equal(model.net.data_id, 0),
            lambda: self.constraint(model.net.x, y_pred, self.train_x),
            lambda: self.constraint(model.net.x, y_pred, self.test_x),
        )
        return [
            losses.get(loss)(y_true[:n], y_pred[:n]),
            losses.get(loss)(tf.zeros(tf.shape(f)), f),
        ]

    @runifnone("train_x", "train_y")
    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.dist_train == "log uniform":
            self.train_x = self.geom.log_uniform_points(self.num_train, False)
        elif self.dist_train == "random":
            self.train_x = self.geom.random_points(self.num_train, "sobol")
        else:
            self.train_x = self.geom.uniform_points(self.num_train, False)
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone("test_x", "test_y")
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
