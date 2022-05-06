import numpy as np

from .data import Data
from .. import backend as bkd
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

    Attributes:
        train_x_bc: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            training BCs/ICs.
        num_bcs (list): `num_bcs[i]` is the number of points for `bcs[i]`.
        train_x: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            training. v is the function input to the branch net; x is the point input to
            the trunk net; vx is the value of v evaluated at x, i.e., v(x). `train_x` is
            ordered from BCs/ICs (`train_x_bc`) to PDEs.
        test_x: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            testing.
    """

    def __init__(
        self,
        pde,
        function_space,
        evaluation_points,
        num_function,
        function_variables=None,
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

        self.num_bcs = [n * self.num_func for n in self.pde.num_bcs]
        self.train_x_bc = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.train_next_batch()
        self.test()

    def losses(self, targets, outputs, loss, model):
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(model.net.inputs[1], outputs, model.net.inputs[2])
            if not isinstance(f, (list, tuple)):
                f = [f]

        bcs_start = np.cumsum([0] + self.num_bcs)
        error_f = [fi[bcs_start[-1] :] for fi in f]
        losses = [loss(bkd.zeros_like(error), error) for error in error_f]
        for i, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(
                self.train_x[1],
                model.net.inputs[1],
                outputs,
                beg,
                end,
                aux_var=self.train_x[2],
            )
            losses.append(loss(bkd.zeros_like(error), error))
        return losses

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        func_feats = self.func_space.random(self.num_func)
        func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)

        v, x, vx = self.bc_inputs(func_feats, func_vals)
        if self.pde.pde is not None:
            # Branch input: v
            v_pde = np.repeat(func_vals, len(self.pde.train_x_all), axis=0)
            v = np.vstack((v, v_pde))
            # Trunk input: x
            x_pde = np.tile(self.pde.train_x_all, (self.num_func, 1))
            x = np.vstack((x, x_pde))
            # vx
            vx_pde = self.func_space.eval_batch(
                func_feats, self.pde.train_x_all[:, self.func_vars]
            ).reshape(-1, 1)
            vx = np.vstack((vx, vx_pde))

        self.train_x = (v, x, vx)
        self.train_y = None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        # TODO: Use different BC data from self.train_x
        # TODO
        self.test_x = self.train_x
        self.test_y = self.train_y
        return self.test_x, self.test_y

    def bc_inputs(self, func_feats, func_vals):
        # Format:
        # v1, x_bc1_1
        # ...
        # v1, x_bc1_N1
        # v2, x_bc1_1
        # ...
        # v2, x_bc1_N1
        v, x, vx = [], [], []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, num_bc in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # Branch input: v
            v.append(np.repeat(func_vals, num_bc, axis=0))
            # Trunk input: x
            x.append(np.tile(self.pde.train_x_bc[beg:end], (self.num_func, 1)))
            # vx
            vx.append(
                self.func_space.eval_batch(
                    func_feats, self.pde.train_x_bc[beg:end, self.func_vars]
                ).reshape(-1, 1)
            )
        self.train_x_bc = (np.vstack(v), np.vstack(x), np.vstack(vx))
        return self.train_x_bc
