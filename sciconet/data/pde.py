from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .. import losses
from ..utils import runifnone


class PDE(Data):
    """PDE solver.
    """

    def __init__(self, geom, pde, func, nbc, anchors=None):
        self.geom = geom
        self.pde = pde
        self.func = func
        self.nbc = nbc
        self.anchors = anchors

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, model):
        n = self.nbc
        if self.anchors is not None:
            n += len(self.anchors)
        f = self.pde(model.net.x, y_pred)[n:]
        return [losses.get('MSE')(y_true[:n], y_pred[:n]),
                losses.get('MSE')(tf.zeros(tf.shape(f)), f)]

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x = self.geom.uniform_points(batch_size, True)
        if self.nbc > 0:
            self.train_x = np.vstack(
                (self.geom.uniform_boundary_points(self.nbc), self.train_x))
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
