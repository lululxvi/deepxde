from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class IC(object):
    """Initial conditions: y([x, t0]) = func([x, t0]).
    """

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
        return outputs[beg:end, self.component : self.component + 1] - self.func(
            X[beg:end]
        )
