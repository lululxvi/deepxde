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

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = on_boundary
        self.component = component

    def filter(self, X):
        X = np.array([x for x in X if self.on_boundary(x, self.geom.on_boundary(x))])
        return X if len(X) > 0 else np.empty((0, self.geom.dim))

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        outputs = outputs[:, self.component : self.component + 1]
        dydx = tf.gradients(outputs, inputs)[0][beg:end]
        n = np.array(list(map(self.geom.boundary_normal, X[beg:end])))
        return tf.reduce_sum(dydx * n, axis=1, keepdims=True)

    def error(self, X, inputs, outputs, beg, end):
        raise NotImplementedError(
            "{}.error to be implemented".format(type(self).__name__)
        )


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(DirichletBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component : self.component + 1] - self.func(
            X[beg:end]
        )


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(NeumannBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end]
        )


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super(RobinBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x.
    """

    def __init__(self, geom, component_x, on_boundary, component=0):
        super(PeriodicBC, self).__init__(geom, on_boundary, component)
        self.component_x = component_x

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = np.array([self.geom.periodic_point(x, self.component_x) for x in X1])
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end):
        outputs = outputs[beg:end, self.component : self.component + 1]
        outputs = tf.reshape(outputs, [-1, 2])
        return outputs[:, 0:1] - outputs[:, 1:]


class PointSet(object):
    """A set of points.
    """

    def __init__(self, points):
        self.points = np.array(points)

    def inside(self, x):
        return np.any(np.all(np.isclose(x, self.points), axis=1))

    def values_to_func(self, values):
        zero = np.zeros(len(values[0]))

        def func(x):
            if not self.inside(x):
                return zero
            idx = np.argwhere(np.all(np.isclose(x, self.points), axis=1))[0, 0]
            return values[idx]

        return lambda X: np.array(list(map(func, X)))
