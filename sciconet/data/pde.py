from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .. import losses as losses_module
from ..geometry import GeometryXTime
from ..utils import runifnone


class PDE(Data):
    """PDE solver.
    """

    def __init__(self, geom, pde, bcs, func, num_boundary, anchors=None):
        self.geom = geom
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, list) else [bcs]
        self.func = func
        self.num_boundary = num_boundary
        self.anchors = anchors

        self.num_bcs = None
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, loss_type, model):
        self.train_next_batch(model.batch_size)
        bcs_start = np.cumsum([0] + self.num_bcs)
        loss_f = losses_module.get(loss_type)

        f = self.pde(model.net.x, y_pred)
        if not isinstance(f, list):
            f = [f]
        f = [fi[bcs_start[-1] :] for fi in f]
        loss = [loss_f(tf.zeros(tf.shape(fi)), fi) for fi in f]

        for i in range(len(self.bcs)):
            loss.append(
                loss_f(
                    tf.zeros((self.num_bcs[i], 1)),
                    self.bcs[i].error(
                        self.train_x[bcs_start[i] : bcs_start[i + 1]],
                        model.net.x[bcs_start[i] : bcs_start[i + 1]],
                        y_pred[bcs_start[i] : bcs_start[i + 1]],
                    ),
                )
            )
        return loss

    @runifnone("train_x", "train_y")
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x = self.geom.uniform_points(batch_size, False)
        if self.num_boundary > 0:
            self.train_x = np.vstack(
                (self.geom.uniform_boundary_points(self.num_boundary), self.train_x)
            )
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))

        x_bcs = [bc.filter(self.geom, self.train_x) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))

        self.train_x = np.vstack(x_bcs + [self.train_x])
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
            losses_module.get(loss)(y_true[:n], y_pred[:n]),
            losses_module.get(loss)(tf.zeros(tf.shape(f)), f),
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
