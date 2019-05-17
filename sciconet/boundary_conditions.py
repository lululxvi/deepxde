from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class BC(object):
    """Boundary conditions.
    """

    def __init__(self, func, on_boundary, component):
        self.func = func
        self.on_boundary = on_boundary
        self.component = component

    def filter(self, geom, X):
        return np.array([x for x in X if self.on_boundary(x, geom.on_boundary(x))])

    def error(self, X, inputs, outputs):
        raise NotImplementedError(
            "{}.error to be implemented".format(type(self).__name__)
        )


class DirichletBC(BC):
    """Dirichlet boundary conditions.
    """

    def __init__(self, func, on_boundary, component=0):
        super(DirichletBC, self).__init__(func, on_boundary, component)

    def error(self, X, inputs, outputs):
        return outputs[:, self.component : self.component + 1] - self.func(X)
