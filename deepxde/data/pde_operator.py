import numpy as np

from .data import Data
from .. import backend as bkd
from ..utils import run_if_all_none


class PDEOperator(Data):
    """PDE solution operator.

    Args:
        pde: Instance of ``dde.data.PDE`` or ``dde.data.TimePDE``.
        function_space: Instance of ``dde.data.FunctionSpace``.
        num_function (int): The number of functions for training.
        num_sensor (int): The number of sensors to evaluate the input function as the
            input of the branch net.

    Attributes:
        train_x: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            training. v is the function input to the branch net; x is the point input to
            the trunk net; vx is the value of v evaluated at x, i.e., v(x). `train_x` is
            ordered from BCs to PDE.
        test_x: A triple of three Numpy arrays (v, x, vx) fed into PIDeepONet for
            testing.
        num_bcs (list): `num_bcs[i]` is the number of points for `bcs[i]`.
    """

    def __init__(self, pde, function_space, num_function, num_sensor):
        self.pde = pde
        self.func_space = function_space
        self.num_func = num_function
        self.num_sensor = num_sensor

        self.num_bcs = [n * self.num_func for n in self.pde.num_bcs]
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
            # TODO BC cannot have v
            error = bc.error(self.train_x[1], model.net.inputs[1], outputs, beg, end)
            losses.append(loss(bkd.zeros_like(error), error))
        return losses

    @run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        # Branch input: v
        func_feats = self.func_space.random(self.num_func)
        # TODO FunctionSpace should have a domain variable via dde.geometry
        # Assume v is in [0, 1]
        xs = np.linspace(0, 1, num=self.num_sensor)[:, None]
        v = self.func_space.eval_batch(func_feats, xs)
        # BC
        # Format:
        # v1, x_bc1_1
        # ...
        # v1, x_bc1_N1
        # v2, x_bc1_1
        # ...
        # v2, x_bc1_N1
        v_bc = []
        for num_bc in self.pde.num_bcs:
            v_bc.append(np.repeat(v, num_bc, axis=0))
        v_bc = np.vstack(v_bc)
        # PDE
        if self.pde.pde is not None:
            v_pde = np.repeat(v, len(self.pde.train_x_all), axis=0)
            v = np.vstack((v_bc, v_pde))
        else:
            v = v_bc

        # Trunk input: x
        # BC
        x_bc = []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, _ in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            x_bc.append(np.tile(self.pde.train_x_bc[beg:end], (self.num_func, 1)))
        x = np.vstack(x_bc)
        # PDE
        if self.pde.pde is not None:
            x_pde = np.tile(self.pde.train_x_all, (self.num_func, 1))
            x = np.vstack((x, x_pde))

        # vx
        # TODO: Assume v is only a function of x1
        # BC
        vx_bc = []
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        for i, _ in enumerate(self.pde.num_bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            vx_bc.append(
                self.func_space.eval_batch(
                    func_feats, self.pde.train_x_bc[beg:end, :1]
                ).reshape(-1, 1)
            )
        vx = np.vstack(vx_bc)
        # PDE
        if self.pde.pde is not None:
            vx_pde = self.func_space.eval_batch(
                func_feats, self.pde.train_x_all[:, :1]
            ).reshape(-1, 1)
            vx = np.vstack((vx, vx_pde))

        self.train_x = (v, x, vx)
        self.train_y = None
        return self.train_x, self.train_y

    @run_if_all_none("test_x", "test_y")
    def test(self):
        # TODO
        self.test_x = self.train_x
        self.test_y = self.train_y
        return self.test_x, self.test_y
