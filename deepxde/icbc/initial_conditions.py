"""Initial conditions."""

__all__ = ["IC"]

import numpy as np

from .boundary_conditions import _check_func_output, npfunc_range_autocache
from .. import backend as bkd
from .. import utils


class IC:
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom
        self.func = npfunc_range_autocache(utils.return_tensor(func))
        self.on_initial = lambda x, on: np.array(
            [on_initial(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter(self, X):
        return X[self.on_initial(X, self.geom.on_initial(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        _check_func_output(values, "IC")
        return outputs[beg:end, self.component : self.component + 1] - values
