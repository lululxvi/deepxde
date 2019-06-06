from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from .helper import one_function
from .pde import PDE
from .. import config
from .. import losses as losses_module
from ..utils import run_if_any_none


class IDE(PDE):
    """IDE solver.
    The current version only supports 1D problems with initial condition at x = 0.

    Args:
        kernel: (x, s) --> R
    """

    def __init__(
        self,
        geom,
        ide,
        bcs,
        quad_deg,
        kernel=None,
        num_domain=0,
        num_boundary=0,
        train_distribution="random",
        anchors=None,
        func=None,
        num_test=None,
    ):
        self.kernel = kernel or one_function(1)
        self.quad_deg = quad_deg
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

        super(IDE, self).__init__(
            geom,
            1,
            ide,
            bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            func=func,
            num_test=num_test,
        )

    def losses(self, y_true, y_pred, loss_type, model):
        bcs_start = np.cumsum([0] + self.num_bcs)
        loss_f = losses_module.get(loss_type)

        int_mat_train = self.get_int_matrix(True)
        int_mat_test = self.get_int_matrix(False)
        f = tf.cond(
            tf.equal(model.net.data_id, 0),
            lambda: self.pde(model.net.x, y_pred, int_mat_train),
            lambda: self.pde(model.net.x, y_pred, int_mat_test),
        )
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
        x_bc = self.bc_points()
        x_quad = self.quad_points(self.train_x)
        self.train_x = np.vstack((x_bc, self.train_x, x_quad))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_y = self.train_y
        else:
            self.test_x = self.test_points()
            x_quad = self.quad_points(self.test_x)
            self.test_x = np.vstack(
                (self.train_x[: sum(self.num_bcs)], self.test_x, x_quad)
            )
            self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y

    def quad_points(self, X):
        def get_quad_points(x):
            return (self.quad_x + 1) * x / 2

        return np.hstack(list(map(lambda xi: get_quad_points(xi[0]), X)))[:, None]

    def get_int_matrix(self, training):
        def get_quad_weights(x):
            return self.quad_w * x / 2

        if training or self.num_test is None:
            size = self.num_domain + self.num_boundary
            if self.anchors is not None:
                size += len(self.anchors)
            X = self.train_x
        else:
            size = self.num_test
            X = self.test_x
        num_bc = sum(self.num_bcs)
        int_mat = np.zeros((size + num_bc, X.size), dtype=config.real(np))
        for i in range(size):
            x = X[i + num_bc, 0]
            beg = size + num_bc + self.quad_deg * i
            end = beg + self.quad_deg
            K = np.ravel(self.kernel(np.full((self.quad_deg, 1), x), X[beg:end]))
            int_mat[i + num_bc, beg:end] = get_quad_weights(x) * K
        return int_mat
