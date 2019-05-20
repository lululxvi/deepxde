from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .helper import zero_function
from .. import boundary_conditions
from .. import losses as losses_module
from ..geometry import GeometryXTime
from ..utils import run_if_any_none


class PDE(Data):
    """PDE solver.
    """

    def __init__(
        self,
        geom,
        num_outputs,
        pde,
        bcs,
        num_domain,
        num_boundary,
        anchors=None,
        func=None,
        num_test=None,
    ):
        self.geom = geom
        self.num_outputs = num_outputs
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, list) else [bcs]

        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.anchors = anchors

        self.func = func if func is not None else zero_function(self.num_outputs)
        self.num_test = num_test

        self.num_bcs = None
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, y_true, y_pred, loss_type, model):
        self.train_next_batch(None)
        self.test()
        bcs_start = np.cumsum([0] + self.num_bcs)
        loss_f = losses_module.get(loss_type)

        f = self.pde(model.net.x, y_pred)
        if not isinstance(f, list):
            f = [f]
        f = [fi[bcs_start[-1] :] for fi in f]
        loss = [loss_f(tf.zeros(tf.shape(fi)), fi) for fi in f]

        for i in range(len(self.bcs)):
            if isinstance(self.bcs[i], boundary_conditions.PeriodicBC):
                loss_one = loss_f(
                    tf.zeros((self.num_bcs[i] // 2, 1)),
                    tf.cond(
                        tf.equal(model.net.data_id, 0),
                        lambda: self.bcs[i].error(
                            self.train_x[bcs_start[i] : bcs_start[i + 1]],
                            model.net.x[bcs_start[i] : bcs_start[i + 1]],
                            y_pred[bcs_start[i] : bcs_start[i + 1]],
                        ),
                        lambda: self.bcs[i].error(
                            self.test_x[bcs_start[i] : bcs_start[i + 1]],
                            model.net.x[bcs_start[i] : bcs_start[i + 1]],
                            y_pred[bcs_start[i] : bcs_start[i + 1]],
                        ),
                    ),
                )
            else:
                loss_one = loss_f(
                    tf.zeros((self.num_bcs[i], 1)),
                    tf.cond(
                        tf.equal(model.net.data_id, 0),
                        lambda: self.bcs[i].error(self.train_x, model.net.x, y_pred),
                        lambda: self.bcs[i].error(self.test_x, model.net.x, y_pred),
                    )[bcs_start[i] : bcs_start[i + 1]],
                )
            loss.append(loss_one)
        return loss

    @run_if_any_none("train_x", "train_y")
    def train_next_batch(self, batch_size):
        self.train_x = self.geom.uniform_points(self.num_domain, False)
        if self.num_boundary > 0:
            self.train_x = np.vstack(
                (self.geom.uniform_boundary_points(self.num_boundary), self.train_x)
            )
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))

        x_bcs = [bc.filter(self.train_x) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))

        self.train_x = np.vstack(x_bcs + [self.train_x])
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_y = self.train_y
        else:
            self.test_x = self.geom.uniform_points(self.num_test, True)
            self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y


class TimePDE(Data):
    """Time-dependent PDE solver.

    Args:
        num_domain: Number of f training points.
        nbc: Number of boundary condition points on the geometry boundary.
        nic: Number of initial condition points inside the geometry.
        nt: Number of time points.
    """

    def __init__(
        self,
        geom,
        timedomain,
        pde,
        func,
        num_domain,
        nbc,
        nic,
        nt,
        num_test,
        anchors=None,
    ):
        self.geomtime = GeometryXTime(geom, timedomain)
        self.pde = pde
        self.func = func
        self.num_domain = num_domain
        self.nbc = nbc
        self.nic = nic
        self.nt = nt
        self.num_test = num_test
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

    @run_if_any_none("train_x", "train_y")
    def train_next_batch(self, batch_size):
        self.train_x = self.geomtime.random_points(self.num_domain)
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

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.geomtime.random_points(self.num_test)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
