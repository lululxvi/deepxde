"""Initial conditions."""

__all__ = ["IC"]

import numpy as np

from .boundary_conditions import npfunc_range_autocache
from .. import backend as bkd
from .. import utils


class IC(object):
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

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        if bkd.ndim(values) > 0 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "IC func should return an array of shape N by 1 for a single component."
                "Use argument 'component' for different components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values
