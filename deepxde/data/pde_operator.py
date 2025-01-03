import numpy as np

from .data import Data
from .sampler import BatchSampler
from .. import backend as bkd
from .. import config
from ..utils import run_if_all_none


class PDEOperator(Data):
    """PDE solution operator.

    Args:
        pde: Instance of ``dde.data.PDE`` or ``dde.data.TimePDE``.
        function_space: Instance of ``dde.data.FunctionSpace``.
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

    Attributes:
        train_bc: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            training BCs/ICs.
        num_bcs (list): `num_bcs[i]` is the number of points for `bcs[i]`.
        train_x: A tuple of two Numpy arrays (v, x) fed into PIDeepONet for training. v
            is the function input to the branch net; x is the point input to
            the trunk net. `train_x` is ordered from BCs/ICs (`train_bc`) to PDEs.
        train_aux_vars: v(x), i.e., the value of v evaluated at x.
    """

    def __init__(
        self,
        pde,
        function_space,
        evaluation_points,
        num_function,
        function_variables=None,
        num_test=None,
    ):
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(pde.geom.dim))
        )
        self.num_test = num_test

        self.num_bcs = [n * self.num_func for n in self.pde.num_bcs]
        self.train_bc = None
        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.train_next_batch()
        self.test()

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(inputs[1], outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        bcs_start = np.cumsum([0] + self.num_bcs)
        error_f = [fi[bcs_start[-1] :] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(
                self.train_x[1],
                inputs[1],
                outputs,
                beg,
                end,
                aux_var=self.train_aux_vars,
            )
            losses.append(loss_fn(bkd.zeros_like(error), error))
        return losses

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        func_feats = self.func_space.random(self.num_func)
        func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
        v, x, vx = self.bc_inputs(func_feats, func_vals)
        if self.pde.pde is not None:
            v_pde, x_pde, vx_pde = self.gen_inputs(
                func_feats, func_vals, self.pde.train_x_all
            )
            v = np.vstack((v, v_pde))
            x = np.vstack((x, x_pde))
            vx = np.vstack((vx, vx_pde))
        self.train_x = (v, x)
        self.train_aux_vars = vx
        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
        else:
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            # TODO: Use different BC data from self.train_x
            v, x, vx = self.train_bc
            if self.pde.pde is not None:
                v_pde, x_pde, vx_pde = self.gen_inputs(
                    func_feats, func_vals, self.pde.test_x[sum(self.pde.num_bcs) :]
                )
                v = np.vstack((v, v_pde))
                x = np.vstack((x, x_pde))
                vx = np.vstack((vx, vx_pde))
            self.test_x = (v, x)
            self.test_aux_vars = vx
        return self.test_x, self.test_y, self.test_aux_vars

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
        vx = self.func_space.eval_batch(func_feats, points[:, self.func_vars]).reshape(
            -1, 1
        )
        return v, x, vx

    def bc_inputs(self, func_feats, func_vals):
        if not self.pde.bcs:
            self.train_bc = (
                np.empty((0, len(self.eval_pts)), dtype=config.real(np)),
                np.empty((0, self.pde.geom.dim), dtype=config.real(np)),
                np.empty((0, 1), dtype=config.real(np)),
            )
            return self.train_bc
        v, x, vx = [], [], []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, _ in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            vi, xi, vxi = self.gen_inputs(
                func_feats, func_vals, self.pde.train_x_bc[beg:end]
            )
            v.append(vi)
            x.append(xi)
            vx.append(vxi)
        self.train_bc = (np.vstack(v), np.vstack(x), np.vstack(vx))
        return self.train_bc

    def resample_train_points(self, pde_points=True, bc_points=True):
        """Resample the training points for the operator."""
        self.pde.resample_train_points(pde_points, bc_points)
        self.train_x, self.train_y, self.train_aux_vars = None, None, None
        self.train_next_batch()


class PDEOperatorCartesianProd(Data):
    """PDE solution operator with data in the format of Cartesian product.

    Args:
        pde: Instance of ``dde.data.PDE`` or ``dde.data.TimePDE``.
        function_space: Instance of ``dde.data.FunctionSpace``.
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
        train_aux_vars: v(x), i.e., the value of v evaluated at x, has the shape (`N1`,
            `N2`).
    """

    def __init__(
        self,
        pde,
        function_space,
        evaluation_points,
        num_function,
        function_variables=None,
        num_test=None,
        batch_size=None,
    ):
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(pde.geom.dim))
        )
        self.num_test = num_test
        self.batch_size = batch_size

        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.train_sampler = BatchSampler(self.num_func, shuffle=True)
        self.train_next_batch()
        self.test()

    def _losses(self, outputs, loss_fn, inputs, model, num_func, aux=None):
        bcs_start = np.cumsum([0] + self.pde.num_bcs)

        losses = []
        # PDE loss
        if config.autodiff == "reverse":  # reverse mode AD
            for i in range(num_func):
                out = outputs[i]
                # Single output
                if bkd.ndim(out) == 1:
                    out = out[:, None]
                f = []
                if self.pde.pde is not None:
                    f = self.pde.pde(
                        inputs[1], out, model.net.auxiliary_vars[i][:, None]
                    )
                    if not isinstance(f, (list, tuple)):
                        f = [f]
                error_f = [fi[bcs_start[-1] :] for fi in f]
                losses_i = [loss_fn(bkd.zeros_like(error), error) for error in error_f]
                losses.append(losses_i)

            losses = zip(*losses)
            # Use stack instead of as_tensor to keep the gradients.
            losses = [bkd.reduce_mean(bkd.stack(loss, 0)) for loss in losses]
        elif config.autodiff == "forward":  # forward mode AD

            def forward_call(trunk_input):
                return aux[0]((inputs[0], trunk_input))

            f = []
            if self.pde.pde is not None:
                # Each f has the shape (N1, N2)
                f = self.pde.pde(
                    inputs[1], (outputs, forward_call), model.net.auxiliary_vars
                )
                if not isinstance(f, (list, tuple)):
                    f = [f]
            # Each error has the shape (N1, ~N2)
            error_f = [fi[:, bcs_start[-1] :] for fi in f]
            for error in error_f:
                error_i = []
                for i in range(num_func):
                    error_i.append(loss_fn(bkd.zeros_like(error[i]), error[i]))
                losses.append(bkd.reduce_mean(bkd.stack(error_i, 0)))

        # BC loss
        losses_bc = []
        for i in range(num_func):
            losses_i = []
            out = outputs[i]
            if bkd.ndim(out) == 1:
                out = out[:, None]
            for j, bc in enumerate(self.pde.bcs):
                beg, end = bcs_start[j], bcs_start[j + 1]
                # The same BC points are used for training and testing.
                error = bc.error(
                    self.train_x[1],
                    inputs[1],
                    out,
                    beg,
                    end,
                    aux_var=model.net.auxiliary_vars[i][:, None],
                )
                losses_i.append(loss_fn(bkd.zeros_like(error), error))

            losses_bc.append(losses_i)

        losses_bc = zip(*losses_bc)
        losses_bc = [bkd.reduce_mean(bkd.stack(loss, 0)) for loss in losses_bc]
        losses.extend(losses_bc)
        return losses

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        num_func = self.num_func if self.batch_size is None else self.batch_size
        return self._losses(outputs, loss_fn, inputs, model, num_func, aux=aux)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self._losses(
            outputs, loss_fn, inputs, model, len(self.test_x[0]), aux=aux
        )

    def train_next_batch(self, batch_size=None):
        if self.train_x is None:
            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(
                func_feats, self.pde.train_x[:, self.func_vars]
            )
            self.train_x = (func_vals, self.pde.train_x)
            self.train_aux_vars = vx

        if self.batch_size is None:
            return self.train_x, self.train_y, self.train_aux_vars

        indices = self.train_sampler.get_next(self.batch_size)
        traix_x = (self.train_x[0][indices], self.train_x[1])
        return traix_x, self.train_y, self.train_aux_vars[indices]

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
        else:
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(
                func_feats, self.pde.test_x[:, self.func_vars]
            )
            self.test_x = (func_vals, self.pde.test_x)
            self.test_aux_vars = vx
        return self.test_x, self.test_y, self.test_aux_vars
