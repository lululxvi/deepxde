from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from . import gradients as grad
from .backend import tf


class BC(object):
    """Boundary conditions.

    Args:
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.geom.boundary_normal(X[beg:end])
        return tf.reduce_sum(dydx * n, axis=1, keepdims=True)

    def error(self, X, inputs, outputs, beg, end):
        raise NotImplementedError(
            "{}.error to be implemented".format(type(self).__name__)
        )


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super(DirichletBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X[beg:end])
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "DirichletBC should output 1D values. Use argument 'component' for different components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super(NeumannBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end]
        )


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super(RobinBC, self).__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super(PeriodicBC, self).__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component : self.component + 1]
            yright = outputs[mid:end, self.component : self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return yleft - yright


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geom: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors, respectively;
            `X` are the values of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(self, geom, func, on_boundary):
        super(OperatorBC, self).__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.func(inputs, outputs, X)[beg:end]


class PointSet(object):
    """A set of points."""

    def __init__(self, points):
        self.points = np.array(points)

    def inside(self, x):
        return np.any(
            np.all(np.isclose(x[:, np.newaxis, :], self.points), axis=-1),
            axis=-1,
        )

    def values_to_func(self, values, default_value=0):
        def func(x):
            pt_equal = np.all(np.isclose(x[:, np.newaxis, :], self.points), axis=-1)
            not_inside = np.logical_not(np.any(pt_equal, axis=-1, keepdims=True))
            return np.matmul(pt_equal, values) + default_value * not_inside

        return func


class PointSetBC(object):
    """Dirichlet boundary condition for a set of points.
    Compare the output (that associates with `points`) with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are known and used for training.
        values: An array of values that gives the exact solution of the problem.
        component: The output component satisfying this BC.
    """

    def __init__(self, points, values, component=0):
        self.points = np.array(points)
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "PointSetBC should output 1D values. Use argument 'component' for different components."
            )
        self.values = values
        self.component = component

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end):
        return outputs[beg:end, self.component : self.component + 1] - self.values
