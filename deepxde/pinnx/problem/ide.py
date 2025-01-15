# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

from typing import Callable, Sequence, Union, Optional, Dict, Any

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from pinnx import utils
from pinnx.geometry import DictPointGeometry
from pinnx.icbc.base import ICBC
from .pde import PDE

__all__ = [
    "IDE",
]

X = Dict[str, bst.typing.ArrayLike]
Y = Dict[str, bst.typing.ArrayLike]
InitMat = Any


class IDE(PDE):
    """IDE solver.

    The current version only supports 1D problems with the integral int_0^x K(x, t) y(t) dt.

    Args:
        kernel: (x, t) --> R.
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        ide: Callable[[X, Y, InitMat], Any],
        constraints: Union[ICBC, Sequence[ICBC]],
        quad_deg: int,
        approximator: Optional[bst.nn.Module] = None,
        kernel: Callable = None,
        num_domain: int = 0,
        num_boundary: int = 0,
        train_distribution: str = "Hammersley",
        anchors=None,
        solution=None,
        num_test: int = None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        self.kernel = kernel or (lambda x, *args: np.ones((len(x), 1)))
        self.quad_deg = quad_deg
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)
        self.quad_x = self.quad_x.astype(bst.environ.dftype())
        self.quad_w = self.quad_w.astype(bst.environ.dftype())

        super().__init__(
            geometry,
            ide,
            constraints,
            approximator=approximator,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            solution=solution,
            num_test=num_test,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )

    def call_pde_errors(self, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)
        fit = bst.environ.get('fit')
        int_mat = self.get_int_matrix(fit)
        pde_errors = self.pde(inputs, outputs, int_mat, **kwargs)
        return jax.tree.map(lambda x: x[bcs_start[-1]:], pde_errors)

    @utils.run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = self.train_points()
        x_bc = self.bc_points()
        x_quad = self.quad_points(self.train_x_all)
        self.train_x = jax.tree.map(
            lambda x, y, z: u.math.concatenate((x, y, z), axis=0),
            x_bc,
            self.train_x_all,
            x_quad,
            is_leaf=u.math.is_quantity
        )
        self.train_y = self.solution(self.train_x) if self.solution else None
        return self.train_x, self.train_y

    @utils.run_if_all_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x_all
        else:
            self.test_x = self.test_points()
        x_quad = self.quad_points(self.test_x)
        self.test_x = jax.tree.map(
            lambda x, y: u.math.concatenate((x, y), axis=0),
            self.test_x,
            x_quad,
            is_leaf=u.math.is_quantity
        )
        self.test_y = self.solution(self.test_x) if self.solution else None
        return self.test_x, self.test_y

    def test_points(self):
        return self.geometry.uniform_points(self.num_test, True)

    def quad_points(self, X):
        fn = lambda xs: (jax.vmap(lambda x: (self.quad_x + 1) * x / 2)(xs)).flatten()
        return jax.tree.map(
            fn,
            X,
            is_leaf=u.math.is_quantity
        )

    def get_int_matrix(self, training):
        def get_quad_weights(x):
            return self.quad_w * x / 2

        with jax.ensure_compile_time_eval():
            if training:
                num_bc = sum(self.num_bcs)
                X = self.train_x
            else:
                num_bc = 0
                X = self.test_x

            X = np.asarray(self.geometry.dict_to_arr(X))
            if training or self.num_test is None:
                num_f = tuple(self.train_x_all.values())[0].shape[0]
            else:
                num_f = self.num_test

            int_mat = np.zeros((num_bc + num_f, X.size), dtype=bst.environ.dftype())
            for i in range(num_f):
                x = X[i + num_bc, 0]
                beg = num_f + num_bc + self.quad_deg * i
                end = beg + self.quad_deg
                K = np.ravel(self.kernel(np.full((self.quad_deg, 1), x), X[beg:end]))
                int_mat[i + num_bc, beg:end] = get_quad_weights(x) * K
            return int_mat
