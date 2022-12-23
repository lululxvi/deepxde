import numpy as np

from .helper import one_function
from .pde import PDE
from .. import backend as bkd
from .. import config
from ..utils import run_if_all_none


class IDE(PDE):
    """IDE solver.

    The current version only supports 1D problems with the integral int_0^x K(x, t) y(t) dt.

    Args:
        kernel: (x, t) --> R.
    """

    def __init__(
        self,
        geometry,
        ide,
        bcs,
        quad_deg,
        kernel=None,
        num_domain=0,
        num_boundary=0,
        train_distribution="Hammersley",
        anchors=None,
        solution=None,
        num_test=None,
    ):
        self.kernel = kernel or one_function(1)
        self.quad_deg = quad_deg
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)
        self.quad_x = self.quad_x.astype(config.real(np))
        self.quad_w = self.quad_w.astype(config.real(np))

        super().__init__(
            geometry,
            ide,
            bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            solution=solution,
            num_test=num_test,
        )

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        bcs_start = np.cumsum([0] + self.num_bcs)
        int_mat = self.get_int_matrix(True)
        f = self.pde(inputs, outputs, int_mat)
        if not isinstance(f, (list, tuple)):
            f = [f]
        f = [fi[bcs_start[-1] :] for fi in f]
        losses = [loss_fn(bkd.zeros_like(fi), fi) for fi in f]

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(loss_fn(bkd.zeros_like(error), error))
        return losses

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        int_mat = self.get_int_matrix(False)
        f = self.pde(inputs, outputs, int_mat)
        if not isinstance(f, (list, tuple)):
            f = [f]
        return [
            loss_fn(bkd.zeros_like(fi), fi) for fi in f
        ] + [bkd.as_tensor(0, dtype=config.real(bkd.lib)) for _ in self.bcs]

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = self.train_points()
        x_bc = self.bc_points()
        x_quad = self.quad_points(self.train_x_all)
        self.train_x = np.vstack((x_bc, self.train_x_all, x_quad))
        self.train_y = self.soln(self.train_x) if self.soln else None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x_all
        else:
            self.test_x = self.test_points()
        x_quad = self.quad_points(self.test_x)
        self.test_x = np.vstack((self.test_x, x_quad))
        self.test_y = self.soln(self.test_x) if self.soln else None
        return self.test_x, self.test_y

    def test_points(self):
        return self.geom.uniform_points(self.num_test, True)

    def quad_points(self, X):
        def get_quad_points(x):
            return (self.quad_x + 1) * x / 2

        return np.hstack(list(map(lambda xi: get_quad_points(xi[0]), X)))[:, None]

    def get_int_matrix(self, training):
        def get_quad_weights(x):
            return self.quad_w * x / 2

        if training:
            num_bc = sum(self.num_bcs)
            X = self.train_x
        else:
            num_bc = 0
            X = self.test_x
        if training or self.num_test is None:
            num_f = len(self.train_x_all)
        else:
            num_f = self.num_test

        int_mat = np.zeros((num_bc + num_f, X.size), dtype=config.real(np))
        for i in range(num_f):
            x = X[i + num_bc, 0]
            beg = num_f + num_bc + self.quad_deg * i
            end = beg + self.quad_deg
            K = np.ravel(self.kernel(np.full((self.quad_deg, 1), x), X[beg:end]))
            int_mat[i + num_bc, beg:end] = get_quad_weights(x) * K
        return int_mat
