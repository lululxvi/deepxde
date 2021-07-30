"""Initial conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ..backend import backend_name, torch


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
        # TODO: For PyTorch, this is recomputed in each iteration.
        targets = self.func(X[beg:end])
        if backend_name == "pytorch":
            targets = torch.from_numpy(targets)
        return outputs[beg:end, self.component : self.component + 1] - targets
