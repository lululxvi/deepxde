# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

import warnings
from typing import Callable, Sequence, Optional, Dict, Any

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from deepxde.data.fpde import Scheme, Fractional as FractionalBase, FractionalTime as FractionalTimeBase
from deepxde.pinnx.geometry import GeometryXTime, DictPointGeometry
from deepxde.pinnx.icbc.base import ICBC
from deepxde.pinnx.utils import array_ops
from deepxde.utils.internal import run_if_all_none
from .pde import PDE

__all__ = [
    "FPDE",
    "TimeFPDE"
]

X = Dict[str, bst.typing.ArrayLike]
Y = Dict[str, bst.typing.ArrayLike]
InitMat = bst.typing.ArrayLike


class FPDE(PDE):
    r"""
    Fractional PDE solver.

    This class implements a solver for Fractional Partial Differential Equations (FPDEs) using the Physics-Informed Neural Network (PINN) approach.

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

    Parameters:
    -----------
    geometry : DictPointGeometry
        The geometry of the problem domain.
    pde : Callable[[X, Y, InitMat], Any]
        The PDE to be solved.
    alpha : float | bst.State[float]
        The order of the fractional derivative.
    constraints : ICBC | Sequence[ICBC]
        The initial and boundary conditions.
    resolution : Sequence[int]
        The resolution for discretization.
    approximator : Optional[bst.nn.Module], default=None
        The neural network approximator.
    meshtype : str, default="dynamic"
        The type of mesh to use ("static" or "dynamic").
    num_domain : int, default=0
        The number of domain points.
    num_boundary : int, default=0
        The number of boundary points.
    train_distribution : str, default="Hammersley"
        The distribution method for training points.
    anchors : Any, default=None
        Anchor points for the domain.
    solution : Callable[[Dict], Dict], default=None
        The analytical solution of the PDE, if available.
    num_test : int, default=None
        The number of test points.
    loss_fn : str | Callable, default='MSE'
        The loss function to use.
    loss_weights : Sequence[float], default=None
        The weights for different components of the loss.

    References:
    -----------
    G. Pang, L. Lu, & G. E. Karniadakis. fPINNs: Fractional physics-informed neural
    networks. SIAM Journal on Scientific Computing, 41(4), A2603--A2626, 2019
    <https://doi.org/10.1137/18M1229845>.
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        pde: Callable[[X, Y, InitMat], Any],
        alpha: float | bst.State[float],
        constraints: ICBC | Sequence[ICBC],
        resolution: Sequence[int],
        approximator: Optional[bst.nn.Module] = None,
        meshtype: str = "dynamic",
        num_domain: int = 0,
        num_boundary: int = 0,
        train_distribution: str = "Hammersley",
        anchors=None,
        solution: Callable[[Dict], Dict] = None,
        num_test: int = None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        self.alpha = alpha
        self.disc = Scheme(meshtype, resolution)
        self.frac_train, self.frac_test = None, None
        self.int_mat_train = None

        super().__init__(
            geometry,
            pde,
            constraints,
            approximator=approximator,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            solution=solution,
            num_test=num_test,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
        )

    def call_pde_errors(self, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)

        # # PDE inputs and outputs
        # pde_inputs = jax.tree.map(lambda x: x[bcs_start[-1]:], inputs)
        # pde_outputs = jax.tree.map(lambda x: x[bcs_start[-1]:], outputs)

        # do not cache int_mat when alpha is a learnable parameter
        fit = bst.environ.get('fit')

        if fit:
            if isinstance(self.alpha, bst.State):
                int_mat = self.get_int_matrix(True)
            else:
                if self.int_mat_train is not None:
                    # use cached int_mat
                    int_mat = self.int_mat_train
                else:
                    # initialize self.int_mat_train with int_mat
                    int_mat = self.get_int_matrix(True)
                    self.int_mat_train = int_mat
        else:
            int_mat = self.get_int_matrix(False)

        # computing PDE losses
        # pde_errors = self.pde(pde_inputs, pde_outputs, int_mat, **kwargs)
        # return pde_errors
        pde_errors = self.pde(inputs, outputs, int_mat, **kwargs)
        return jax.tree.map(lambda x: x[bcs_start[-1]:], pde_errors)

    def call_bc_errors(self, loss_fns, loss_weights, inputs, outputs, **kwargs):
        return super().call_bc_errors(loss_fns, loss_weights, inputs, outputs, **kwargs)
        # fit = bst.environ.get('fit')
        # if fit:
        #     return super().call_bc_errors(loss_fns, loss_weights, inputs, outputs, **kwargs)
        # else:
        #     return [u.math.zeros((), dtype=bst.environ.dftype()) for _ in self.constraints]

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        alpha = self.alpha.value if isinstance(self.alpha, bst.State) else self.alpha

        # do not cache train data when alpha is a learnable parameter
        if self.disc.meshtype == "static":
            if self.geometry.geom.idstr != "Interval":
                raise ValueError("Only Interval supports static mesh.")

            self.frac_train = Fractional(alpha, self.geometry.geom, self.disc, None)
            X = self.frac_train.get_x()
            X = self.geometry.arr_to_dict(u.math.roll(X, -1))

            # FPDE is only applied to the domain points.
            # Boundary points are auxiliary points, and appended in the end.
            self.train_x_all = X
            if self.anchors is not None:
                self.train_x_all = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                                self.anchors,
                                                self.train_x_all)
            x_bc = self.bc_points()

        elif self.disc.meshtype == "dynamic":
            self.train_x_all = self.train_points()
            x_bc = self.bc_points()

            # FPDE is only applied to the domain points.
            train_x_all = self.geometry.dict_to_arr(self.train_x_all)
            x_f = train_x_all[~self.geometry.on_boundary(self.train_x_all)]
            self.frac_train = Fractional(alpha, self.geometry.geom, self.disc, x_f)
            X = self.geometry.arr_to_dict(self.frac_train.get_x())

        else:
            raise ValueError("Unknown meshtype %s" % self.disc.meshtype)

        self.train_x = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                    x_bc,
                                    X,
                                    is_leaf=u.math.is_quantity)
        self.train_y = self.solution(self.train_x) if self.solution else None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        # do not cache test data when alpha is a learnable parameter
        if self.disc.meshtype == "static" and self.num_test is not None:
            raise ValueError("Cannot use test points in static mesh.")

        if self.num_test is None:
            # assign the training points to the testing points
            num_bc = sum(self.num_bcs)
            self.test_x = jax.tree_map(lambda x: x[num_bc:], self.train_x)
            self.frac_test = self.frac_train
        else:
            alpha = self.alpha.value if isinstance(self.alpha, bst.State) else self.alpha

            # Generate `self.test_x`, resampling the test points
            self.test_x = self.test_points()
            not_boundary = ~self.geometry.on_boundary(self.test_x)
            x_f = self.geometry.dict_to_arr(self.test_x)[not_boundary]
            self.frac_test = Fractional(alpha, self.geometry.geom, self.disc, x_f)
            self.test_x = self.geometry.arr_to_dict(self.frac_test.get_x())

        self.test_y = self.solution(self.test_x) if self.solution else None
        return self.test_x, self.test_y

    def test_points(self):
        return self.geometry.uniform_points(self.num_test, True)

    def get_int_matrix(self, training):
        if training:
            int_mat = self.frac_train.get_matrix(sparse=True)
            num_bc = sum(self.num_bcs)
        else:
            int_mat = self.frac_test.get_matrix(sparse=True)
            num_bc = 0

        if self.disc.meshtype == "static":
            int_mat = np.roll(int_mat, -1, 1)
            int_mat = int_mat[1:-1]

        int_mat = array_ops.zero_padding(int_mat, ((num_bc, 0), (num_bc, 0)))
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
        geometry: DictPointGeometry,
        pde: Callable[[X, Y, InitMat], Any],
        alpha: float | bst.State[float],
        constraints: ICBC | Sequence[ICBC],
        resolution: Sequence[int],
        approximator: Optional[bst.nn.Module] = None,
        meshtype: str = "dynamic",
        num_domain: int = 0,
        num_boundary: int = 0,
        num_initial: int = 0,
        train_distribution: str = "Hammersley",
        anchors=None,
        solution=None,
        num_test: int = None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        self.num_initial = num_initial
        assert isinstance(geometry, DictPointGeometry), f"DictPointGeometry is required. But got {geometry}"
        super().__init__(
            geometry,
            pde,
            alpha,
            constraints,
            resolution,
            approximator=approximator,
            meshtype=meshtype,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            solution=solution,
            num_test=num_test,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
        )

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        assert isinstance(self.geometry.geom, GeometryXTime), "GeometryXTime is required."
        geometry = self.geometry.geom
        alpha = self.alpha.value if isinstance(self.alpha, bst.State) else self.alpha

        if self.disc.meshtype == "static":
            if geometry.geometry.idstr != "Interval":
                raise ValueError("Only Interval supports static mesh.")

            nt = int(round(self.num_domain / (self.disc.resolution[0] - 2))) + 1
            self.frac_train = FractionalTime(
                alpha,
                geometry.geometry,
                geometry.timedomain.t0,
                geometry.timedomain.t1,
                self.disc,
                nt,
                None,
            )
            X = self.geometry.arr_to_dict(self.frac_train.get_x())
            self.train_x_all = X
            if self.anchors is not None:
                self.train_x_all = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                                self.anchors,
                                                self.train_x_all)
            x_bc = self.bc_points()

            # Remove the initial and boundary points at the beginning of X,
            # which are not considered in the integral matrix.
            n_start = self.disc.resolution[0] + 2 * nt - 2
            X = jax.tree.map(lambda x: x[n_start:], X)

        elif self.disc.meshtype == "dynamic":
            self.train_x_all = self.train_points()
            train_x_all = self.geometry.dict_to_arr(self.train_x_all)
            x_bc = self.bc_points()

            # FPDE is only applied to the non-boundary points.
            x_f = train_x_all[~geometry.on_boundary(train_x_all)]
            self.frac_train = FractionalTime(
                alpha,
                geometry.geometry,
                geometry.timedomain.t0,
                geometry.timedomain.t1,
                self.disc,
                None,
                x_f,
            )
            X = self.geometry.arr_to_dict(self.frac_train.get_x())

        else:
            raise ValueError("Unknown meshtype %s" % self.disc.meshtype)

        self.train_x = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                    x_bc,
                                    X,
                                    is_leaf=u.math.is_quantity)
        self.train_y = self.solution(self.train_x) if self.solution else None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        alpha = self.alpha.value if isinstance(self.alpha, bst.State) else self.alpha
        assert isinstance(self.geometry.geom, GeometryXTime), "GeometryXTime is required."
        geometry = self.geometry.geom
        if self.disc.meshtype == "static" and self.num_test is not None:
            raise ValueError("Cannot use test points in static mesh.")

        if self.num_test is None:
            n_bc = sum(self.num_bcs)
            self.test_x = jax.tree.map(lambda x: x[n_bc:], self.train_x)
            self.frac_test = self.frac_train

        else:
            self.test_x = self.test_points()
            test_x = self.geometry.dict_to_arr(self.test_x)
            x_f = test_x[~geometry.on_boundary(test_x)]
            self.frac_test = FractionalTime(
                alpha,
                geometry.geometry,
                geometry.timedomain.t0,
                geometry.timedomain.t1,
                self.disc,
                None,
                x_f,
            )
            self.test_x = self.geometry.arr_to_dict(self.frac_test.get_x())
        self.test_y = self.solution(self.test_x) if self.solution else None
        return self.test_x, self.test_y

    def train_points(self):
        X = super().train_points()
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geometry.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geometry.random_initial_points(
                    self.num_initial,
                    random=self.train_distribution
                )
            X = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                             tmp,
                             X,
                             is_leaf=u.math.is_quantity)
        return X

    def get_int_matrix(self, training):
        if training:
            int_mat = self.frac_train.get_matrix(sparse=True)
            num_bc = sum(self.num_bcs)
        else:
            int_mat = self.frac_test.get_matrix(sparse=True)
            num_bc = 0

        int_mat = array_ops.zero_padding(int_mat, ((num_bc, 0), (num_bc, 0)))
        return int_mat


class Fractional(FractionalBase):
    """Fractional derivative.

    Args:
        x0: If ``disc.meshtype = static``, then x0 should be None;
            if ``disc.meshtype = 'dynamic'``, then x0 are non-boundary points.
    """

    def _check_dynamic_stepsize(self):
        h = 1 / self.disc.resolution[-1]
        min_h = self.geom.mindist2boundary(self.x0)
        if min_h < h:
            warnings.warn(
                "Warning: mesh step size %f is larger than the boundary distance %f."
                % (h, min_h),
                UserWarning,
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
        w = [1.0]
        for j in range(1, n):
            w.append(w[-1] * (j - 1 - self.alpha) / j)
        return np.asarray(w)

    def get_x_dynamic(self):
        if np.any(self.geom.on_boundary(self.x0)):
            raise ValueError("x0 contains boundary points.")
        if self.geom.dim == 1:
            dirns, dirn_w = [-1, 1], [1, 1]
        elif self.geom.dim == 2:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(self.disc.resolution[0])
            gauss_x, gauss_w = gauss_x.astype(bst.environ.dftype()), gauss_w.astype(bst.environ.dftype())
            thetas = np.pi * gauss_x + np.pi
            dirns = np.vstack((np.cos(thetas), np.sin(thetas))).T
            dirn_w = np.pi * gauss_w
        elif self.geom.dim == 3:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(
                max(self.disc.resolution[:2])
            )
            gauss_x, gauss_w = gauss_x.astype(bst.environ.dftype()), gauss_w.astype(bst.environ.dftype())
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
                    lambda dirn: self.geom.background_points(x0i, dirn, self.dynamic_dist2npts, 0),
                    dirns,
                )
            )
            wi = list(
                map(
                    lambda i: dirn_w[i] * np.linalg.norm(xi[i][1] - xi[i][0]) ** (-self.alpha)
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
            self.w.append(array_ops.hstack(wi))
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
        w0 = np.hstack(([bst.environ.dftype()(0)], w))
        w1 = np.hstack((w, [bst.environ.dftype()(0)]))
        beta = 1 - self.alpha / 2
        w = beta * w0 + (1 - beta) * w1
        if x is None:
            return w
        x = np.vstack(([2 * x[0] - x[1]], x))
        if not self.geom.inside(x[0:1])[0]:
            return x[1:], w[1:]
        return x, w

    def modify_third_order(self, x=None, w=None):
        w0 = np.hstack(([bst.environ.dftype()(0)], w))
        w1 = np.hstack((w, [bst.environ.dftype()(0)]))
        w2 = np.hstack(([bst.environ.dftype()(0)] * 2, w[:-1]))
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

    def get_matrix_static(self):
        if not isinstance(self.alpha, (np.ndarray, jax.Array)):
            int_mat = np.zeros(
                (self.disc.resolution[0], self.disc.resolution[0]),
                dtype=bst.environ.dftype(),
            )
            h = self.geom.diam / (self.disc.resolution[0] - 1)
            for i in range(1, self.disc.resolution[0] - 1):
                # first order
                int_mat[i, 1: i + 2] = np.flipud(self.get_weight(i))
                int_mat[i, i - 1: -1] += self.get_weight(
                    self.disc.resolution[0] - 1 - i
                )
                # second order
                # int_mat[i, 0:i+2] = np.flipud(self.modify_second_order(w=self.get_weight(i)))
                # int_mat[i, i-1:] += self.modify_second_order(w=self.get_weight(self.disc.resolution[0]-1-i))
                # third order
                # int_mat[i, 0:i+2] = np.flipud(self.modify_third_order(w=self.get_weight(i)))
                # int_mat[i, i-1:] += self.modify_third_order(w=self.get_weight(self.disc.resolution[0]-1-i))
            return h ** (-self.alpha) * int_mat
        int_mat = np.zeros((1, self.disc.resolution[0]), dtype=bst.environ.dftype())
        for i in range(1, self.disc.resolution[0] - 1):
            # shifted
            row = np.concatenate(
                [
                    np.zeros(1, dtype=bst.environ.dftype()),
                    np.flip(self.get_weight(i), (0,)),
                    np.zeros(self.disc.resolution[0] - i - 2, dtype=bst.environ.dftype()),
                ],
                0,
            )
            row += np.concatenate(
                [
                    np.zeros(i - 1, dtype=bst.environ.dftype()),
                    self.get_weight(self.disc.resolution[0] - 1 - i),
                    np.zeros(1, dtype=bst.environ.dftype()),
                ],
                0,
            )
            row = np.expand_dims(row, 0)
            int_mat = np.concatenate([int_mat, row], 0)
        int_mat = np.concatenate(
            [int_mat, np.zeros([1, self.disc.resolution[0]], dtype=bst.environ.dftype())], 0
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
                values = array_ops.hstack((values, self.w[i]))
            return indices, values, dense_shape

        print("Generating dense fractional matrix...")
        int_mat = np.zeros((self.x0.shape[0], self.x.shape[0]), dtype=bst.environ.dftype())
        beg = self.x0.shape[0]
        for i in range(self.x0.shape[0]):
            int_mat[i, beg: beg + self.w[i].size] = self.w[i]
            beg += self.w[i].size
        return int_mat


class FractionalTime(FractionalTimeBase):
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

    def get_x_static(self):
        # Points are ordered as initial --> boundary --> inside
        x = self.geom.uniform_points(self.disc.resolution[0], True)
        x = np.roll(x, 1)[:, 0]
        dt = (self.tmax - self.tmin) / (self.nt - 1)
        d = np.empty((self.disc.resolution[0] * self.nt, self.geom.dim + 1), dtype=x.dtype)
        d[0: self.disc.resolution[0], 0] = x
        d[0: self.disc.resolution[0], 1] = self.tmin
        beg = self.disc.resolution[0]
        for i in range(1, self.nt):
            d[beg: beg + 2, 0] = x[:2]
            d[beg: beg + 2, 1] = self.tmin + i * dt
            beg += 2
        for i in range(1, self.nt):
            d[beg: beg + self.disc.resolution[0] - 2, 0] = x[2:]
            d[beg: beg + self.disc.resolution[0] - 2, 1] = self.tmin + i * dt
            beg += self.disc.resolution[0] - 2
        return d

    def get_x_dynamic(self):
        self.fracx = Fractional(self.alpha, self.geom, self.disc, self.x0[:, :-1])
        xx = self.fracx.get_x()
        x = np.empty((len(xx), self.geom.dim + 1), dtype=xx.dtype)
        x[: len(self.x0)] = self.x0
        beg = len(self.x0)
        for i in range(len(self.x0)):
            tmp = xx[self.fracx.xindex_start[i]: self.fracx.xindex_start[i + 1]]
            x[beg: beg + len(tmp), :1] = tmp
            x[beg: beg + len(tmp), -1] = self.x0[i, -1]
            beg += len(tmp)
        return x

    def get_matrix_static(self):
        # Only consider the inside points
        print("Warning: assume zero boundary condition.")
        n = (self.disc.resolution[0] - 2) * (self.nt - 1)
        int_mat = np.zeros((n, n), dtype=bst.environ.dftype())
        self.fracx = Fractional(self.alpha, self.geom, self.disc, None)
        int_mat_one = self.fracx.get_matrix()
        beg = 0
        for _ in range(self.nt - 1):
            int_mat[
            beg: beg + self.disc.resolution[0] - 2,
            beg: beg + self.disc.resolution[0] - 2,
            ] = int_mat_one[1:-1, 1:-1]
            beg += self.disc.resolution[0] - 2
        return int_mat
