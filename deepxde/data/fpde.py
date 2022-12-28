__all__ = ["Scheme", "FPDE", "TimeFPDE"]

import math

import numpy as np

from .pde import PDE
from .. import backend as bkd
from .. import config
from ..backend import is_tensor, backend_name
from ..utils import array_ops_compat, run_if_all_none


class Scheme:
    """Fractional Laplacian discretization.

    Discretize fractional Laplacian uisng quadrature rule for the integral with respect to the directions
    and Grunwald-Letnikov (GL) formula for the Riemann-Liouville directional fractional derivative.

    Args:
        meshtype (string): "static" or "dynamic".
        resolution: A list of integer. The first number is the number of quadrature points in the first direction, ...,
            and the last number is the GL parameter.

    References:
        `G. Pang, L. Lu, & G. E. Karniadakis. fPINNs: Fractional physics-informed neural
        networks. SIAM Journal on Scientific Computing, 41(4), A2603--A2626, 2019
        <https://doi.org/10.1137/18M1229845>`_.
    """

    def __init__(self, meshtype, resolution):
        self.meshtype = meshtype
        self.resolution = resolution

        self.dim = len(resolution)
        self._check()

    def _check(self):
        if self.meshtype not in ["static", "dynamic"]:
            raise ValueError("Wrong meshtype %s" % self.meshtype)
        if self.dim >= 2 and self.meshtype == "static":
            raise ValueError(
                "Do not support meshtype static for dimension %d" % self.dim
            )


class FPDE(PDE):
    r"""Fractional PDE solver.

    D-dimensional fractional Laplacian of order alpha/2 (1 < alpha < 2) is defined as:
    (-Delta)^(alpha/2) u(x) = C(alpha, D) \int_{||theta||=1} D_theta^alpha u(x) d theta,
    where C(alpha, D) = gamma((1-alpha)/2) * gamma((D+alpha)/2) / (2 pi^((D+1)/2)),
    D_theta^alpha is the Riemann-Liouville directional fractional derivative,
    and theta is the differentiation direction vector.
    The solution u(x) is assumed to be identically zero in the boundary and exterior of the domain.
    When D = 1, C(alpha, D) = 1 / (2 cos(alpha * pi / 2)).

    This solver does not consider C(alpha, D) in the fractional Laplacian,
    and only discretizes \int_{||theta||=1} D_theta^alpha u(x) d theta.
    D_theta^alpha is approximated by Grunwald-Letnikov formula.

    References:
        `G. Pang, L. Lu, & G. E. Karniadakis. fPINNs: Fractional physics-informed neural
        networks. SIAM Journal on Scientific Computing, 41(4), A2603--A2626, 2019
        <https://doi.org/10.1137/18M1229845>`_.
    """

    def __init__(
        self,
        geometry,
        fpde,
        alpha,
        bcs,
        resolution,
        meshtype="dynamic",
        num_domain=0,
        num_boundary=0,
        train_distribution="Hammersley",
        anchors=None,
        solution=None,
        num_test=None,
    ):
        self.alpha = alpha
        self.disc = Scheme(meshtype, resolution)
        self.frac_train, self.frac_test = None, None

        super().__init__(
            geometry,
            fpde,
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
        losses = [
            loss_fn(bkd.zeros_like(fi), fi) for fi in f
        ]

        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            error = bc.error(self.train_x, inputs, outputs, beg, end)
            losses.append(
                loss_fn(bkd.zeros_like(error), error)
            )
        return losses

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        int_mat = self.get_int_matrix(False)
        f = self.pde(inputs, outputs, int_mat)
        if not isinstance(f, (list, tuple)):
            f = [f]
        return [
            loss_fn(bkd.zeros_like(fi), fi) for fi in f
        ] + [bkd.as_tensor(0, dtype=config.real(bkd.lib)) for _ in self.bcs]

    def train_next_batch(self, batch_size=None):
        # do not cache train data when alpha is a learnable parameter
        if not is_tensor(self.alpha) or backend_name == "tensorflow.compat.v1":
            if self.train_x is not None:
                return self.train_x, self.train_y
        if self.disc.meshtype == "static":
            if self.geom.idstr != "Interval":
                raise ValueError("Only Interval supports static mesh.")

            self.frac_train = Fractional(self.alpha, self.geom, self.disc, None)
            X = self.frac_train.get_x()
            # FPDE is only applied to the domain points.
            # Boundary points are auxiliary points, and appended in the end.
            X = np.roll(X, -1)
            self.train_x_all = X
            if self.anchors is not None:
                self.train_x_all = np.vstack((self.anchors, self.train_x_all))
            x_bc = self.bc_points()
        elif self.disc.meshtype == "dynamic":
            self.train_x_all = self.train_points()
            x_bc = self.bc_points()
            # FPDE is only applied to the domain points.
            x_f = self.train_x_all[~self.geom.on_boundary(self.train_x_all)]
            self.frac_train = Fractional(self.alpha, self.geom, self.disc, x_f)
            X = self.frac_train.get_x()

        self.train_x = np.vstack((x_bc, X))
        self.train_y = self.soln(self.train_x) if self.soln else None
        return self.train_x, self.train_y

    def test(self):
        # do not cache test data when alpha is a learnable parameter
        if not is_tensor(self.alpha) or backend_name == "tensorflow.compat.v1":
            if self.test_x is not None:
                return self.test_x, self.test_y

        if self.disc.meshtype == "static" and self.num_test is not None:
            raise ValueError("Cannot use test points in static mesh.")

        if self.num_test is None:
            self.test_x = self.train_x[sum(self.num_bcs) :]
            self.frac_test = self.frac_train
        else:
            self.test_x = self.test_points()
            x_f = self.test_x[~self.geom.on_boundary(self.test_x)]
            self.frac_test = Fractional(self.alpha, self.geom, self.disc, x_f)
            self.test_x = self.frac_test.get_x()
        self.test_y = self.soln(self.test_x) if self.soln else None
        return self.test_x, self.test_y

    def test_points(self):
        return self.geom.uniform_points(self.num_test, True)

    def get_int_matrix(self, training):
        if training:
            int_mat = self.frac_train.get_matrix(sparse=True)
            num_bc = sum(self.num_bcs)
        else:
            int_mat = self.frac_test.get_matrix(sparse=True)
            num_bc = 0

        if self.disc.meshtype == "static":
            int_mat = array_ops_compat.roll(int_mat, -1, 1)
            int_mat = int_mat[1:-1]

        int_mat = array_ops_compat.zero_padding(int_mat, ((num_bc, 0), (num_bc, 0)))
        return int_mat


class TimeFPDE(FPDE):
    r"""Time-dependent fractional PDE solver.

    D-dimensional fractional Laplacian of order alpha/2 (1 < alpha < 2) is defined as:
    (-Delta)^(alpha/2) u(x) = C(alpha, D) \int_{||theta||=1} D_theta^alpha u(x) d theta,
    where C(alpha, D) = gamma((1-alpha)/2) * gamma((D+alpha)/2) / (2 pi^((D+1)/2)),
    D_theta^alpha is the Riemann-Liouville directional fractional derivative,
    and theta is the differentiation direction vector.
    The solution u(x) is assumed to be identically zero in the boundary and exterior of the domain.
    When D = 1, C(alpha, D) = 1 / (2 cos(alpha * pi / 2)).

    This solver does not consider C(alpha, D) in the fractional Laplacian,
    and only discretizes \int_{||theta||=1} D_theta^alpha u(x) d theta.
    D_theta^alpha is approximated by Grunwald-Letnikov formula.

    References:
        `G. Pang, L. Lu, & G. E. Karniadakis. fPINNs: Fractional physics-informed neural
        networks. SIAM Journal on Scientific Computing, 41(4), A2603--A2626, 2019
        <https://doi.org/10.1137/18M1229845>`_.
    """

    def __init__(
        self,
        geometryxtime,
        fpde,
        alpha,
        ic_bcs,
        resolution,
        meshtype="dynamic",
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        train_distribution="Hammersley",
        anchors=None,
        solution=None,
        num_test=None,
    ):
        self.num_initial = num_initial
        super().__init__(
            geometryxtime,
            fpde,
            alpha,
            ic_bcs,
            resolution,
            meshtype=meshtype,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            solution=solution,
            num_test=num_test,
        )

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        if self.disc.meshtype == "static":
            if self.geom.geometry.idstr != "Interval":
                raise ValueError("Only Interval supports static mesh.")

            nt = int(round(self.num_domain / (self.disc.resolution[0] - 2))) + 1
            self.frac_train = FractionalTime(
                self.alpha,
                self.geom.geometry,
                self.geom.timedomain.t0,
                self.geom.timedomain.t1,
                self.disc,
                nt,
                None,
            )
            X = self.frac_train.get_x()
            self.train_x_all = X
            if self.anchors is not None:
                self.train_x_all = np.vstack((self.anchors, self.train_x_all))
            x_bc = self.bc_points()
            # Remove the initial and boundary points at the beginning of X,
            # which are not considered in the integral matrix.
            X = X[self.disc.resolution[0] + 2 * nt - 2 :, :]
        elif self.disc.meshtype == "dynamic":
            self.train_x_all = self.train_points()
            x_bc = self.bc_points()
            # FPDE is only applied to the non-boundary points.
            x_f = self.train_x_all[~self.geom.on_boundary(self.train_x_all)]
            self.frac_train = FractionalTime(
                self.alpha,
                self.geom.geometry,
                self.geom.timedomain.t0,
                self.geom.timedomain.t1,
                self.disc,
                None,
                x_f,
            )
            X = self.frac_train.get_x()

        self.train_x = np.vstack((x_bc, X))
        self.train_y = self.soln(self.train_x) if self.soln else None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        if self.disc.meshtype == "static" and self.num_test is not None:
            raise ValueError("Cannot use test points in static mesh.")

        if self.num_test is None:
            self.test_x = self.train_x[sum(self.num_bcs) :]
            self.frac_test = self.frac_train
        else:
            self.test_x = self.test_points()
            x_f = self.test_x[~self.geom.on_boundary(self.test_x)]
            self.frac_test = FractionalTime(
                self.alpha,
                self.geom.geometry,
                self.geom.timedomain.t0,
                self.geom.timedomain.t1,
                self.disc,
                None,
                x_f,
            )
            self.test_x = self.frac_test.get_x()
        self.test_y = self.soln(self.test_x) if self.soln else None
        return self.test_x, self.test_y

    def train_points(self):
        X = super().train_points()
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(
                    self.num_initial, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        return X

    def get_int_matrix(self, training):
        if training:
            int_mat = self.frac_train.get_matrix(sparse=True)
            num_bc = sum(self.num_bcs)
        else:
            int_mat = self.frac_test.get_matrix(sparse=True)
            num_bc = 0

        int_mat = array_ops_compat.zero_padding(int_mat, ((num_bc, 0), (num_bc, 0)))
        return int_mat


class Fractional:
    """Fractional derivative.

    Args:
        x0: If ``disc.meshtype = static``, then x0 should be None;
            if ``disc.meshtype = 'dynamic'``, then x0 are non-boundary points.
    """

    def __init__(self, alpha, geom, disc, x0):
        if (disc.meshtype == "static" and x0 is not None) or (
            disc.meshtype == "dynamic" and x0 is None
        ):
            raise ValueError("disc.meshtype and x0 do not match.")

        self.alpha, self.geom = alpha, geom
        self.disc, self.x0 = disc, x0
        if disc.meshtype == "dynamic":
            self._check_dynamic_stepsize()

        self.x, self.xindex_start, self.w = None, None, None
        self._w_init = self._init_weights()

    def _check_dynamic_stepsize(self):
        h = 1 / self.disc.resolution[-1]
        min_h = self.geom.mindist2boundary(self.x0)
        if min_h < h:
            print(
                "Warning: mesh step size %f is larger than the boundary distance %f."
                % (h, min_h)
            )

    def _init_weights(self):
        """If ``disc.meshtype = 'static'``, then n is number of points;
        if ``disc.meshtype = 'dynamic'``, then n is resolution lambda.
        """
        n = (
            self.disc.resolution[0]
            if self.disc.meshtype == "static"
            else self.dynamic_dist2npts(self.geom.diam) + 1
        )
        w = [bkd.as_tensor(1.0, dtype=config.real(bkd.lib)) if bkd.is_tensor(self.alpha) else 1.0]
        for j in range(1, n):
            w.append(w[-1] * (j - 1 - self.alpha) / j)
        return array_ops_compat.convert_to_array(w)

    def get_x(self):
        self.x = (
            self.get_x_static()
            if self.disc.meshtype == "static"
            else self.get_x_dynamic()
        )
        return self.x

    def get_matrix(self, sparse=False):
        return (
            self.get_matrix_static()
            if self.disc.meshtype == "static"
            else self.get_matrix_dynamic(sparse)
        )

    def get_x_static(self):
        return self.geom.uniform_points(self.disc.resolution[0], True)

    def dynamic_dist2npts(self, dx):
        return int(math.ceil(self.disc.resolution[-1] * dx))

    def get_x_dynamic(self):
        if np.any(self.geom.on_boundary(self.x0)):
            raise ValueError("x0 contains boundary points.")
        if self.geom.dim == 1:
            dirns, dirn_w = [-1, 1], [1, 1]
        elif self.geom.dim == 2:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(self.disc.resolution[0])
            gauss_x, gauss_w = gauss_x.astype(config.real(np)), gauss_w.astype(config.real(np))
            thetas = np.pi * gauss_x + np.pi
            dirns = np.vstack((np.cos(thetas), np.sin(thetas))).T
            dirn_w = np.pi * gauss_w
        elif self.geom.dim == 3:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(
                max(self.disc.resolution[:2])
            )
            gauss_x, gauss_w = gauss_x.astype(config.real(np)), gauss_w.astype(config.real(np))
            thetas = (np.pi * gauss_x[: self.disc.resolution[0]] + np.pi) / 2
            phis = np.pi * gauss_x[: self.disc.resolution[1]] + np.pi
            dirns, dirn_w = [], []
            for i in range(self.disc.resolution[0]):
                for j in range(self.disc.resolution[1]):
                    dirns.append(
                        [
                            np.sin(thetas[i]) * np.cos(phis[j]),
                            np.sin(thetas[i]) * np.sin(phis[j]),
                            np.cos(thetas[i]),
                        ]
                    )
                    dirn_w.append(gauss_w[i] * gauss_w[j] * np.sin(thetas[i]))
            dirn_w = np.pi ** 2 / 2 * np.array(dirn_w)
        x, self.w = [], []
        for x0i in self.x0:
            xi = list(
                map(
                    lambda dirn: self.geom.background_points(
                        x0i, dirn, self.dynamic_dist2npts, 0
                    ),
                    dirns,
                )
            )
            wi = list(
                map(
                    lambda i: dirn_w[i]
                    * np.linalg.norm(xi[i][1] - xi[i][0]) ** (-self.alpha)
                    * self.get_weight(len(xi[i]) - 1),
                    range(len(dirns)),
                )
            )
            # first order
            xi, wi = zip(*map(self.modify_first_order, xi, wi))
            # second order
            # xi, wi = zip(*map(self.modify_second_order, xi, wi))
            # third order
            # xi, wi = zip(*map(self.modify_third_order, xi, wi))
            x.append(np.vstack(xi))
            self.w.append(array_ops_compat.hstack(wi))
        self.xindex_start = np.hstack(([0], np.cumsum(list(map(len, x))))) + len(
            self.x0
        )
        return np.vstack([self.x0] + x)

    def modify_first_order(self, x, w):
        x = np.vstack(([2 * x[0] - x[1]], x[:-1]))
        if not self.geom.inside(x[0:1])[0]:
            return x[1:], w[1:]
        return x, w

    def modify_second_order(self, x=None, w=None):
        w0 = np.hstack(([config.real(np)(0)], w))
        w1 = np.hstack((w, [config.real(np)(0)]))
        beta = 1 - self.alpha / 2
        w = beta * w0 + (1 - beta) * w1
        if x is None:
            return w
        x = np.vstack(([2 * x[0] - x[1]], x))
        if not self.geom.inside(x[0:1])[0]:
            return x[1:], w[1:]
        return x, w

    def modify_third_order(self, x=None, w=None):
        w0 = np.hstack(([config.real(np)(0)], w))
        w1 = np.hstack((w, [config.real(np)(0)]))
        w2 = np.hstack(([config.real(np)(0)] * 2, w[:-1]))
        beta = 1 - self.alpha / 2
        w = (
            (-6 * beta ** 2 + 11 * beta + 1) / 6 * w0
            + (11 - 6 * beta) * (1 - beta) / 12 * w1
            + (6 * beta + 1) * (beta - 1) / 12 * w2
        )
        if x is None:
            return w
        x = np.vstack(([2 * x[0] - x[1]], x))
        if not self.geom.inside(x[0:1])[0]:
            return x[1:], w[1:]
        return x, w

    def get_weight(self, n):
        return self._w_init[: n + 1]

    def get_matrix_static(self):
        if not bkd.is_tensor(self.alpha):
            int_mat = np.zeros(
                (self.disc.resolution[0], self.disc.resolution[0]),
                dtype=config.real(np),
            )
            h = self.geom.diam / (self.disc.resolution[0] - 1)
            for i in range(1, self.disc.resolution[0] - 1):
                # first order
                int_mat[i, 1 : i + 2] = np.flipud(self.get_weight(i))
                int_mat[i, i - 1 : -1] += self.get_weight(
                    self.disc.resolution[0] - 1 - i
                )
                # second order
                # int_mat[i, 0:i+2] = np.flipud(self.modify_second_order(w=self.get_weight(i)))
                # int_mat[i, i-1:] += self.modify_second_order(w=self.get_weight(self.disc.resolution[0]-1-i))
                # third order
                # int_mat[i, 0:i+2] = np.flipud(self.modify_third_order(w=self.get_weight(i)))
                # int_mat[i, i-1:] += self.modify_third_order(w=self.get_weight(self.disc.resolution[0]-1-i))
            return h ** (-self.alpha) * int_mat
        int_mat = bkd.zeros((1, self.disc.resolution[0]), dtype=config.real(bkd.lib))
        for i in range(1, self.disc.resolution[0] - 1):
            if True:
                # shifted
                row = bkd.concat(
                    [
                        bkd.zeros(1, dtype=config.real(bkd.lib)),
                        bkd.reverse(self.get_weight(i), [0]),
                        bkd.zeros(
                            self.disc.resolution[0] - i - 2, dtype=config.real(bkd.lib)
                        ),
                    ],
                    0,
                )
                row += bkd.concat(
                    [
                        bkd.zeros(i - 1, dtype=config.real(bkd.lib)),
                        self.get_weight(self.disc.resolution[0] - 1 - i),
                        bkd.zeros(1, dtype=config.real(bkd.lib)),
                    ],
                    0,
                )
            else:
                # not shifted
                row = bkd.concat(
                    [
                        bkd.reverse(self.get_weight(i), [0]),
                        bkd.zeros(self.disc.resolution[0] - i - 1),
                    ],
                    0,
                )
                row += bkd.concat(
                    [bkd.zeros(i), self.get_weight(self.disc.resolution[0] - 1 - i)], 0
                )
            row = bkd.expand_dims(row, 0)
            int_mat = bkd.concat([int_mat, row], 0)
        int_mat = bkd.concat(
            [int_mat, bkd.zeros([1, self.disc.resolution[0]], dtype=config.real(bkd.lib))], 0
        )
        h = self.geom.diam / (self.disc.resolution[0] - 1)
        return h ** (-self.alpha) * int_mat

    def get_matrix_dynamic(self, sparse):
        if self.x is None:
            raise AssertionError("No dynamic points")

        if sparse:
            print("Generating sparse fractional matrix...")
            dense_shape = (self.x0.shape[0], self.x.shape[0])
            indices, values = [], []
            beg = self.x0.shape[0]
            for i in range(self.x0.shape[0]):
                for _ in range(self.w[i].shape[0]):
                    indices.append([i, beg])
                    beg += 1
                values = array_ops_compat.hstack((values, self.w[i]))
            return indices, values, dense_shape

        print("Generating dense fractional matrix...")
        int_mat = np.zeros((self.x0.shape[0], self.x.shape[0]), dtype=config.real(np))
        beg = self.x0.shape[0]
        for i in range(self.x0.shape[0]):
            int_mat[i, beg : beg + self.w[i].size] = self.w[i]
            beg += self.w[i].size
        return int_mat


class FractionalTime:
    """Fractional derivative with time.

    Args:
        nt: If ``disc.meshtype = static``, then nt is the number of t points;
            if ``disc.meshtype = 'dynamic'``, then nt is None.
        x0: If ``disc.meshtype = static``, then x0 should be None;
            if ``disc.meshtype = 'dynamic'``, then x0 are non-boundary points.

    Attributes:
        nx: If ``disc.meshtype = static``, then nx is the number of x points;
            if ``disc.meshtype = dynamic``, then nx is the resolution lambda.
    """

    def __init__(self, alpha, geom, tmin, tmax, disc, nt, x0):
        self.alpha = alpha
        self.geom, self.tmin, self.tmax = geom, tmin, tmax
        self.disc, self.nt, self.x0 = disc, nt, x0

        self.x, self.fracx = None, None

    def get_x(self):
        self.x = (
            self.get_x_static()
            if self.disc.meshtype == "static"
            else self.get_x_dynamic()
        )
        return self.x

    def get_matrix(self, sparse=False):
        return (
            self.get_matrix_static()
            if self.disc.meshtype == "static"
            else self.get_matrix_dynamic(sparse)
        )

    def get_x_static(self):
        # Points are ordered as initial --> boundary --> inside
        x = self.geom.uniform_points(self.disc.resolution[0], True)
        x = np.roll(x, 1)[:, 0]
        dt = (self.tmax - self.tmin) / (self.nt - 1)
        d = np.empty((self.disc.resolution[0] * self.nt, self.geom.dim + 1), dtype=x.dtype)
        d[0 : self.disc.resolution[0], 0] = x
        d[0 : self.disc.resolution[0], 1] = self.tmin
        beg = self.disc.resolution[0]
        for i in range(1, self.nt):
            d[beg : beg + 2, 0] = x[:2]
            d[beg : beg + 2, 1] = self.tmin + i * dt
            beg += 2
        for i in range(1, self.nt):
            d[beg : beg + self.disc.resolution[0] - 2, 0] = x[2:]
            d[beg : beg + self.disc.resolution[0] - 2, 1] = self.tmin + i * dt
            beg += self.disc.resolution[0] - 2
        return d

    def get_x_dynamic(self):
        self.fracx = Fractional(self.alpha, self.geom, self.disc, self.x0[:, :-1])
        xx = self.fracx.get_x()
        x = np.empty((len(xx), self.geom.dim + 1), dtype=xx.dtype)
        x[: len(self.x0)] = self.x0
        beg = len(self.x0)
        for i in range(len(self.x0)):
            tmp = xx[self.fracx.xindex_start[i] : self.fracx.xindex_start[i + 1]]
            x[beg : beg + len(tmp), :1] = tmp
            x[beg : beg + len(tmp), -1] = self.x0[i, -1]
            beg += len(tmp)
        return x

    def get_matrix_static(self):
        # Only consider the inside points
        print("Warning: assume zero boundary condition.")
        n = (self.disc.resolution[0] - 2) * (self.nt - 1)
        int_mat = np.zeros((n, n), dtype=config.real(np))
        self.fracx = Fractional(self.alpha, self.geom, self.disc, None)
        int_mat_one = self.fracx.get_matrix()
        beg = 0
        for _ in range(self.nt - 1):
            int_mat[
                beg : beg + self.disc.resolution[0] - 2,
                beg : beg + self.disc.resolution[0] - 2,
            ] = int_mat_one[1:-1, 1:-1]
            beg += self.disc.resolution[0] - 2
        return int_mat

    def get_matrix_dynamic(self, sparse):
        return self.fracx.get_matrix(sparse)
