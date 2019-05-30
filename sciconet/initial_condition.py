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
        self.on_initial = on_initial
        self.component = component

    def filter(self, X):
        X = np.array([x for x in X if self.on_initial(x, self.geom.on_initial(x))])
        return X if len(X) > 0 else np.empty((0, self.geom.dim))

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component : self.component + 1] - self.func(
            X[beg:end]
        )
