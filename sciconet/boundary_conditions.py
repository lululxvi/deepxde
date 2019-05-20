from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class BC(object):
    """Boundary conditions.

    Args:
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, func, on_boundary, component):
        self.geom = geom
        self.func = func
        self.on_boundary = on_boundary
        self.component = component

    def filter(self, X):
        return np.array([x for x in X if self.on_boundary(x, self.geom.on_boundary(x))])

    def normal_derivative(self, X, inputs, outputs):
        dydx = tf.gradients(outputs[:, self.component : self.component + 1], inputs)[0]
        n = np.array(list(map(self.geom.boundary_normal, X)))
        return tf.reduce_sum(dydx * n, axis=1, keepdims=True)

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
        return self.normal_derivative(X, inputs, outputs) - self.func(X)


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(RobinBC, self).__init__(geom, func, on_boundary, component)

    def error(self, X, inputs, outputs):
        return self.normal_derivative(X, inputs, outputs) - self.func(X, outputs)


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x.
    """

    def __init__(self, geom, component_x, on_boundary, component=0):
        super(PeriodicBC, self).__init__(geom, None, on_boundary, component)
        self.component_x = component_x

    def filter(self, X):
        X1 = np.array([x for x in X if self.on_boundary(x, self.geom.on_boundary(x))])
        # auxiliary_points
        X2 = np.array([self.geom.periodic_point(x, self.component_x) for x in X1])
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs):
        outputs = outputs[:, self.component : self.component + 1]
        outputs = tf.reshape(outputs, [-1, 2])
        return outputs[:, 0:1] - outputs[:, 1:]
