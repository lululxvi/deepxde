from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .. import losses
from ..geometry import GeometryXTime
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

    def losses(self, y_true, y_pred, loss, model):
        n = self.nbc
        if self.anchors is not None:
            n += len(self.anchors)
        f = self.pde(model.net.x, y_pred)
        if not isinstance(f, list):
            f = [f]
        f = [fi[n:] for fi in f]
        return [losses.get(loss)(y_true[:n], y_pred[:n])] + [
            losses.get(loss)(tf.zeros(tf.shape(fi)), fi) for fi in f
        ]

    @runifnone("train_x", "train_y")
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x = self.geom.uniform_points(batch_size, True)
        if self.nbc > 0:
            self.train_x = np.vstack(
                (self.geom.uniform_boundary_points(self.nbc), self.train_x)
            )
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone("test_x", "test_y")
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y


class TimePDE(Data):
    """Time-dependent PDE solver.

    Args:
        nbc: Number of boundary condition points on the geometry boundary.
        nic: Number of initial condition points inside the geometry.
        nt: Number of time points.
    """

    def __init__(self, geom, timedomain, pde, func, nbc, nic, nt, anchors=None):
        self.geomtime = GeometryXTime(geom, timedomain)
        self.pde = pde
        self.func = func
        self.nbc = nbc
        self.nic = nic
        self.nt = nt
        self.anchors = anchors

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, loss, model):
        n = self.nbc * self.nt + self.nic
        if self.anchors is not None:
            n += len(self.anchors)
        f = self.pde(model.net.x, y_pred)[n:]
        return [
            losses.get(loss)(y_true[:n], y_pred[:n]),
            losses.get(loss)(tf.zeros(tf.shape(f)), f),
        ]

    @runifnone("train_x", "train_y")
    def train_next_batch(self, batch_size, *args, **kwargs):
        """
        Args:
            batch_size: Number of f training points.
        """
        self.train_x = self.geomtime.random_points(batch_size)
        if self.nbc > 0:
            self.train_x = np.vstack(
                (self.geomtime.uniform_boundary_points(self.nbc, self.nt), self.train_x)
            )
        if self.nic > 0:
            self.train_x = np.vstack(
                (self.geomtime.uniform_initial_points(self.nic), self.train_x)
            )
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone("test_x", "test_y")
    def test(self, n, *args, **kwargs):
        self.test_x = self.geomtime.random_points(n)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
