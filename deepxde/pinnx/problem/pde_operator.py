# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

from typing import Callable, Sequence, Union, Optional, Any, Dict

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from deepxde.data.function_spaces import FunctionSpace
from deepxde.pinnx.geometry import DictPointGeometry
from deepxde.pinnx.icbc.base import ICBC
from deepxde.pinnx.utils import run_if_all_none
from deepxde.pinnx.utils.sampler import BatchSampler
from .pde import TimePDE

__all__ = [
    'PDEOperator',
    'PDEOperatorCartesianProd',
]

Inputs = Any
Outputs = Any
Auxiliary = Any
Residual = Any


class PDEOperator(TimePDE):
    """
    PDE solution operator.

    Args:
        function_space: Instance of ``pinnx.fnspace.FunctionSpace``.
        evaluation_points: A NumPy array of shape (n_points, dim). Discretize the input
            function sampled from `function_space` using point-wise evaluations at a set
            of points as the input of the branch net.
        num_function (int): The number of functions for training.
        function_variables: ``None`` or a list of integers. The functions in the
            `function_space` may not have the same domain as the PDE. For example, the
            PDE is defined on a spatio-temporal domain (`x`, `t`), but the function is
            IC, which is only a function of `x`. In this case, we need to specify the
            variables of the function by `function_variables=[0]`, where `0` indicates
            the first variable `x`. If ``None``, then we assume the domains of the
            function and the PDE are the same.
        num_fn_test: The number of functions for testing PDE loss. The testing functions
            for BCs/ICs are the same functions used for training. If ``None``, then the
            training functions will be used for testing.
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        pde: Callable[[Inputs, Outputs, Auxiliary], Residual],
        constraints: Union[ICBC, Sequence[ICBC]],
        function_space: FunctionSpace,
        evaluation_points,
        num_function: int,
        function_variables: Optional[Sequence[int]] = None,
        num_test: int = None,
        approximator: Optional[bst.nn.Module] = None,
        solution: Callable[[bst.typing.PyTree], bst.typing.PyTree] = None,
        num_domain: int = 0,  # for space PDE
        num_boundary: int = 0,  # for space PDE
        num_initial: int = 0,  # for time PDE
        num_fn_test: int = None,
        train_distribution: str = "Hammersley",
        anchors: Optional[bst.typing.ArrayLike] = None,
        exclusions=None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
    ):

        assert isinstance(function_space, FunctionSpace), (
            f"Expected `function_space` to be an instance of `FunctionSpace`, "
            f"but got {type(function_space)}."
        )
        self.fn_space = function_space
        self.eval_pts = evaluation_points
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(geometry.dim))
        )

        self.num_fn = num_function
        self.num_fn_test = num_fn_test

        self.fn_train_bc = None
        self.fn_train_x = None
        self.fn_train_y = None
        self.fn_train_aux_vars = None
        self.fn_test_x = None
        self.fn_test_y = None
        self.fn_test_aux_vars = None

        super().__init__(
            geometry=geometry,
            pde=pde,
            constraints=constraints,
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            num_initial=num_initial,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
        )

    def call_pde_errors(self, inputs, outputs, **kwargs):
        num_bcs = self.num_bcs
        self.num_bcs = self.num_fn_bcs
        losses = super().call_pde_errors(inputs, outputs, **kwargs)
        self.num_bcs = num_bcs
        return losses

    def call_bc_errors(self, loss_fns, loss_weights, inputs, outputs, **kwargs):
        num_bcs = self.num_bcs
        self.num_bcs = self.num_fn_bcs
        losses = super().call_bc_errors(loss_fns, loss_weights, inputs, outputs, **kwargs)
        self.num_bcs = num_bcs
        return losses

    @run_if_all_none("fn_train_x", "fn_train_y", "fn_train_aux_vars")
    def train_next_batch(self, batch_size=None):
        super().train_next_batch(batch_size)

        self.num_fn_bcs = [n * self.num_fn for n in self.num_bcs]
        func_feats = self.fn_space.random(self.num_fn)
        func_vals = self.fn_space.eval_batch(func_feats, self.eval_pts)
        v, x, vx = self.bc_inputs(func_feats, func_vals)

        if self._pde is not None:
            v_pde, x_pde, vx_pde = self.gen_inputs(
                func_feats,
                func_vals,
                self.geometry.dict_to_arr(self.train_x_all)
            )
            v = np.vstack((v, v_pde))
            x = np.vstack((x, x_pde))
            vx = np.vstack((vx, vx_pde))
        self.fn_train_x = (v, x)
        self.fn_train_aux_vars = {'aux': vx}
        return self.fn_train_x, self.fn_train_x, self.fn_train_aux_vars

    @run_if_all_none("fn_test_x", "fn_test_y", "fn_test_aux_vars")
    def test(self):
        super().test()

        if self.num_fn_test is None:
            self.fn_test_x = self.fn_train_x
            self.fn_test_aux_vars = self.fn_train_aux_vars

        else:
            func_feats = self.fn_space.random(self.num_fn_test)
            func_vals = self.fn_space.eval_batch(func_feats, self.eval_pts)
            # TODO: Use different BC data from self.fn_train_x
            v, x, vx = self.train_bc
            if self._pde is not None:
                test_x = self.geometry.dict_to_arr(self.test_x)
                v_pde, x_pde, vx_pde = self.gen_inputs(
                    func_feats,
                    func_vals,
                    test_x[sum(self.num_bcs):]
                )
                v = np.vstack((v, v_pde))
                x = np.vstack((x, x_pde))
                vx = np.vstack((vx, vx_pde))
            self.fn_test_x = (v, x)
            self.fn_test_aux_vars = {'aux': vx}
        return self.fn_test_x, self.fn_test_y, self.fn_test_aux_vars

    def gen_inputs(self, func_feats, func_vals, points):
        # Format:
        # v1, x_1
        # ...
        # v1, x_N1
        # v2, x_1
        # ...
        # v2, x_N1
        v = np.repeat(func_vals, len(points), axis=0)
        x = np.tile(points, (len(func_feats), 1))
        vx = self.fn_space.eval_batch(func_feats, points[:, self.func_vars]).reshape(-1, 1)
        return v, x, vx

    def bc_inputs(self, func_feats, func_vals):
        if len(self.constraints) == 0:
            self.train_bc = (
                np.empty((0, len(self.eval_pts)), dtype=bst.environ.dftype()),
                np.empty((0, self.geometry.dim), dtype=bst.environ.dftype()),
                np.empty((0, 1), dtype=bst.environ.dftype()),
            )
            return self.train_bc

        v, x, vx = [], [], []
        bcs_start = np.cumsum([0] + self.num_bcs)
        train_x_bc = self.geometry.dict_to_arr(self.train_x_bc)
        for i, _ in enumerate(self.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            vi, xi, vxi = self.gen_inputs(func_feats, func_vals, train_x_bc[beg:end])
            v.append(vi)
            x.append(xi)
            vx.append(vxi)
        self.train_bc = (np.vstack(v), np.vstack(x), np.vstack(vx))
        return self.train_bc

    def resample_train_points(self, pde_points=True, bc_points=True):
        """
        Resample the training points for the operator.
        """
        super().resample_train_points(pde_points=pde_points, bc_points=bc_points)

        self.fn_train_x, self.fn_train_x, self.fn_train_aux_vars = None, None, None
        self.train_next_batch()


class PDEOperatorCartesianProd(TimePDE):
    """
    PDE solution operator with problem in the format of Cartesian product.

    Args:
        pde: Instance of ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``.
        function_space: Instance of ``pinnx.problem.FunctionSpace``.
        evaluation_points: A NumPy array of shape (n_points, dim). Discretize the input
            function sampled from `function_space` using pointwise evaluations at a set
            of points as the input of the branch net.
        num_function (int): The number of functions for training.
        function_variables: ``None`` or a list of integers. The functions in the
            `function_space` may not have the same domain as the PDE. For example, the
            PDE is defined on a spatio-temporal domain (`x`, `t`), but the function is
            IC, which is only a function of `x`. In this case, we need to specify the
            variables of the function by `function_variables=[0]`, where `0` indicates
            the first variable `x`. If ``None``, then we assume the domains of the
            function and the PDE are the same.
        num_test: The number of functions for testing PDE loss. The testing functions
            for BCs/ICs are the same functions used for training. If ``None``, then the
            training functions will be used for testing.
        batch_size: Integer or ``None``.

    Attributes:
        train_x: A tuple of two Numpy arrays (v, x) fed into PIDeepONet for training. v
            is the function input to the branch net and has the shape (`N1`, `dim1`); x
            is the point input to the trunk net and has the shape (`N2`, `dim2`).
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        pde: Callable[[Inputs, Outputs, Auxiliary], Residual],
        constraints: Union[ICBC, Sequence[ICBC]],
        function_space: FunctionSpace,
        evaluation_points,
        num_function: int,
        function_variables: Optional[Sequence[int]] = None,
        num_test: int = None,
        approximator: Optional[bst.nn.Module] = None,
        solution: Callable[[bst.typing.PyTree], bst.typing.PyTree] = None,
        num_domain: int = 0,  # for space PDE
        num_boundary: int = 0,  # for space PDE
        num_initial: int = 0,  # for time PDE
        num_fn_test: int = None,  # for function space
        train_distribution: str = "Hammersley",
        anchors: Optional[bst.typing.ArrayLike] = None,
        exclusions=None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
        batch_size: int = None,
    ):

        assert isinstance(function_space, FunctionSpace), (
            f"Expected `function_space` to be an instance of `FunctionSpace`, "
            f"but got {type(function_space)}."
        )
        self.fn_space = function_space
        self.eval_pts = evaluation_points
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(geometry.dim))
        )
        self.num_fn = num_function
        self.num_fn_test = num_fn_test

        self.train_sampler = BatchSampler(self.num_fn, shuffle=True)
        self.batch_size = batch_size

        self.fn_train_bc = None
        self.fn_train_x = None
        self.fn_train_y = None
        self.fn_train_aux_vars = None
        self.fn_test_x = None
        self.fn_test_y = None
        self.fn_test_aux_vars = None

        super().__init__(
            geometry=geometry,
            pde=pde,
            constraints=constraints,
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            num_initial=num_initial,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
        )

    def call_pde_errors(self, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)

        # PDE inputs and outputs, computing PDE losses
        pde_inputs = (inputs[0], jax.tree.map(lambda x: x[bcs_start[-1]:], inputs[1]))
        pde_outputs = jax.tree.map(lambda x: x[:, bcs_start[-1]:], outputs)
        pde_kwargs = jax.tree.map(lambda x: x[:, bcs_start[-1]:], kwargs)

        # error
        pde_errors = self.pde(pde_inputs, pde_outputs, **pde_kwargs)
        return pde_errors

    def call_bc_errors(self, loss_fns, loss_weights, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)
        losses = []
        for i, bc in enumerate(self.constraints):
            # ICBC inputs and outputs, computing ICBC losses
            beg, end = bcs_start[i], bcs_start[i + 1]
            icbc_inputs = (inputs[0], jax.tree.map(lambda x: x[beg:end], inputs[1]))
            icbc_outputs = jax.tree.map(lambda x: x[:, beg:end], outputs)
            icbc_kwargs = jax.tree.map(lambda x: x[:, beg:end], kwargs)

            # error
            error: Dict = bc.error(icbc_inputs, icbc_outputs, **icbc_kwargs)

            # loss and weights
            f_loss = loss_fns[i]
            if loss_weights is not None:
                w = loss_weights[i]
                bc_loss = jax.tree.map(lambda err: f_loss(u.math.zeros_like(err), err) * w, error)
            else:
                bc_loss = jax.tree.map(lambda err: f_loss(u.math.zeros_like(err), err), error)

            # append to losses
            losses.append({f'ibc{i}': bc_loss})
        return losses

    # def _losses(self, inputs, outputs, num_fn):
    #     bcs_start = np.cumsum([0] + self.num_bcs)
    #
    #     losses = []
    #     for i in range(num_fn):
    #         out = outputs[i]
    #         # Single output
    #         if u.math.ndim(out) == 1:
    #             out = out[:, None]
    #         f = []
    #         if self.pde.pde is not None:
    #             f = self.pde.pde(partial(model.fn_outputs, True), inputs[1])
    #             if not isinstance(f, (list, tuple)):
    #                 f = [f]
    #         error_f = [fi[bcs_start[-1]:] for fi in f]
    #         losses_i = [loss_fn(u.math.zeros_like(error), error) for error in error_f]
    #
    #         for j, bc in enumerate(self.constraints):
    #             beg, end = bcs_start[j], bcs_start[j + 1]
    #             # The same BC points are used for training and testing.
    #             error = bc.error(
    #                 self.fn_train_x[1],
    #                 inputs[1],
    #                 out,
    #                 beg,
    #                 end,
    #                 aux_var=model.net.auxiliary_vars[i][:, None],
    #             )
    #             losses_i.append(loss_fn(u.math.zeros_like(error), error))
    #
    #         losses.append(losses_i)
    #
    #     losses = zip(*losses)
    #     # Use stack instead of as_tensor to keep the gradients.
    #     losses = [u.math.mean(u.math.stack(loss, 0)) for loss in losses]
    #     return losses
    #
    # def losses_train(self, inputs, outputs, targets, **kwargs):
    #     num_fn = self.num_fn if self.batch_size is None else self.batch_size
    #     return self._losses(outputs, inputs, num_fn)
    #
    # def losses_test(self, inputs, outputs, targets, **kwargs):
    #     return self._losses(outputs, inputs, len(self.test_x[0]))

    def train_next_batch(self, batch_size=None):
        super().train_next_batch(batch_size)

        if self.fn_train_x is None:
            train_x = self.geometry.dict_to_arr(self.train_x)
            func_feats = self.fn_space.random(self.num_fn)
            func_vals = self.fn_space.eval_batch(func_feats, self.eval_pts)
            vx = self.fn_space.eval_batch(func_feats, train_x[:, self.func_vars])
            self.fn_train_x = (func_vals, train_x)
            self.fn_train_aux_vars = {'aux': vx}

        if self.batch_size is None:
            return self.fn_train_x, self.train_y, self.fn_train_aux_vars

        indices = self.train_sampler.get_next(self.batch_size)
        train_x = (self.fn_train_x[0][indices], self.fn_train_x[1])
        return train_x, self.train_y, {'aux': self.fn_train_aux_vars['aux'][indices]}

    @run_if_all_none("fn_test_x", "test_y", "fn_test_aux_vars")
    def test(self):
        super().test()

        if self.num_fn_test is None:
            self.fn_test_x = self.fn_train_x
            self.fn_test_aux_vars = self.fn_train_aux_vars
        else:
            test_x = self.geometry.dict_to_arr(self.test_x)
            func_feats = self.fn_space.random(self.num_fn_test)
            func_vals = self.fn_space.eval_batch(func_feats, self.eval_pts)
            vx = self.fn_space.eval_batch(func_feats, test_x[:, self.func_vars])
            self.fn_test_x = (func_vals, test_x)
            self.fn_test_aux_vars = {'aux': vx}
        return self.fn_test_x, self.test_y, {'aux': self.fn_test_aux_vars}
