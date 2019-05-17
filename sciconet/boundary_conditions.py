from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class BC(object):
    """Boundary conditions.

    Args:
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False
    """

    def __init__(self, geom, func, on_boundary, component):
        self.geom = geom
        self.func = func
        self.on_boundary = on_boundary
        self.component = component

    def filter(self, X):
        return np.array([x for x in X if self.on_boundary(x, self.geom.on_boundary(x))])

    def error(self, X, inputs, outputs):
        raise NotImplementedError(
            "{}.error to be implemented".format(type(self).__name__)
        )


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(DirichletBC, self).__init__(geom, func, on_boundary, component)

    def error(self, X, inputs, outputs):
        return outputs[:, self.component : self.component + 1] - self.func(X)


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(NeumannBC, self).__init__(geom, func, on_boundary, component)

    def error(self, X, inputs, outputs):
        dydx = tf.gradients(outputs[:, self.component : self.component + 1], inputs)[0]
        n = np.array(list(map(self.geom.boundary_normal, X)))
        return tf.reduce_sum(dydx * n, axis=1, keepdims=True) - self.func(X)
