from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .data import Data
from .helper import zero_function
from .. import losses as losses_module
from ..geometry import GeometryXTime
from ..utils import run_if_any_none


class PDE(Data):
    """ODE or time-independent PDE solver.
    """

    def __init__(
        self,
        geom,
        num_outputs,
        pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        train_distribution="random",
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
        self.train_distribution = train_distribution
        self.anchors = anchors

        self.func = func or zero_function(self.num_outputs)
        self.num_test = num_test

        self.num_bcs = None
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.train_next_batch()
        self.test()

    def losses(self, y_true, y_pred, loss_type, model):
        bcs_start = np.cumsum([0] + self.num_bcs)
        loss_f = losses_module.get(loss_type)

        f = self.pde(model.net.x, y_pred)
        if not isinstance(f, list):
            f = [f]
        f = [fi[bcs_start[-1] :] for fi in f]
        loss = [loss_f(tf.zeros(tf.shape(fi)), fi) for fi in f]

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = (
                tf.cond(
                    tf.equal(model.net.data_id, 0),
                    lambda: bc.error(self.train_x, model.net.x, y_pred, beg, end),
                    lambda: bc.error(self.test_x, model.net.x, y_pred, beg, end),
                ),
            )
            loss.append(loss_f(tf.zeros(tf.shape(error)), error))
        return loss

    @run_if_any_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        self.train_x = self.train_points()
        self.train_x = np.vstack((self.bc_points(), self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_y = self.train_y
        else:
            self.test_x = self.test_points()
            self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y

    def add_anchors(self, anchors):
        if self.anchors is None:
            self.anchors = anchors
        else:
            self.anchors = np.vstack((anchors, self.anchors))
        self.train_x = np.vstack((anchors, self.train_x[sum(self.num_bcs) :]))
        self.train_x = np.vstack((self.bc_points(), self.train_x))
        self.train_y = self.func(self.train_x)

    def train_points(self):
        X = np.empty((0, self.geom.dim))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(self.num_domain, random="sobol")
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random="sobol"
                )
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        return X

    def bc_points(self):
        x_bcs = [bc.collocation_points(self.train_x) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))
        return np.vstack(x_bcs)

    def test_points(self):
        return self.geom.uniform_points(self.num_test, True)


class TimePDE(PDE):
    """Time-dependent PDE solver.

    Args:
        num_domain: Number of f training points.
        num_boundary: Number of boundary condition points on the geometry boundary.
        num_initial: Number of initial condition points.
    """

    def __init__(
        self,
        geomtime,
        num_outputs,
        pde,
        ic_bcs,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        train_distribution="random",
        anchors=None,
        func=None,
        num_test=None,
    ):
        self.num_initial = num_initial
        super(TimePDE, self).__init__(
            geomtime,
            num_outputs,
            pde,
            ic_bcs,
            num_domain,
            num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            func=func,
            num_test=num_test,
        )

    def train_points(self):
        X = np.empty((0, self.geom.dim))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(self.num_domain, random="sobol")
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random="sobol"
                )
            X = np.vstack((tmp, X))
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(self.num_initial, random="sobol")
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        return X

    def test_points(self):
        return self.geom.uniform_points(self.num_test)
