"""Initial conditions."""

__all__ = ["IC"]

import numpy as np

from .boundary_conditions import npfunc_range_autocache
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
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "IC function should return an array of shape N by 1 for each component."
                "Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values
