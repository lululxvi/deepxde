"""Initial conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from .. import backend as bkd
from .. import config


class IC(object):
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(self, geom, func, on_initial, component=0):
        self.geom = geom
        self.func = func
        self.on_initial = lambda x, on: np.array(
            [on_initial(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter(self, X):
        return X[self.on_initial(X, self.geom.on_initial(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end):
        targets = self.func(X[beg:end])
        if not isinstance(targets, numbers.Number) and targets.shape[1] != 1:
            raise RuntimeError(
                "IC func should return an array of shape N by 1 for a single component."
                "Use argument 'component' for different components."
            )
        targets = bkd.as_tensor(targets, dtype=config.real(bkd.lib))
        return outputs[beg:end, self.component : self.component + 1] - targets
