"""Boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "Interface2DBC",
    "NeumannBC",
    "OperatorBC",
    "PeriodicBC",
    "PointSetBC",
    "PointSetOperatorBC",
    "RobinBC",
]

import numbers
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from .. import backend as bkd
from .. import config
from .. import data
from .. import gradients as grad
from .. import utils
from ..backend import backend_name


class BC(ABC):
    """Boundary condition base class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

        self.boundary_normal = npfunc_range_autocache(
            utils.return_tensor(self.geom.boundary_normal)
        )

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return bkd.sum(dydx * n, 1, keepdims=True)

    @abstractmethod
    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Returns the loss."""
        # aux_var is used in PI-DeepONet, where aux_var is the input function evaluated
        # at x.


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return self.normal_derivative(X, inputs, outputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
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

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
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
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    Warning:
        If you use `X` in `func`, then do not set ``num_test`` when you define
        ``dde.data.PDE`` or ``dde.data.TimePDE``, otherwise DeepXDE would throw an
        error. In this case, the training points will be used for testing, and this will
        not affect the network training and training loss. This is a bug of DeepXDE,
        which cannot be fixed in an easy way for all backends.
    """

    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, X)[beg:end]


class PointSetBC:
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'dde.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, component=0, batch_size=None, shuffle=True):
        self.points = np.array(points, dtype=config.real(np))
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component
        if isinstance(component, list) and backend_name != "pytorch":
            # TODO: Add support for multiple components in other backends
            raise RuntimeError(
                "multiple components only implemented for pytorch backend"
            )
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            if backend_name not in ["pytorch", "paddle"]:
                raise RuntimeError(
                    "batch_size only implemented for pytorch and paddle backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if self.batch_size is not None:
            if isinstance(self.component, numbers.Number):
                return (
                    outputs[beg:end, self.component : self.component + 1]
                    - self.values[self.batch_indices]
                )
            return outputs[beg:end, self.component] - self.values[self.batch_indices]
        if isinstance(self.component, numbers.Number):
            return outputs[beg:end, self.component : self.component + 1] - self.values
        # When a concat is provided, the following code works 'fast' in paddle cpu,
        # and slow in both tensorflow backends, jax untested.
        # tf.gather can be used instead of for loop but is also slow
        # if len(self.component) > 1:
        #    calculated_error = outputs[beg:end, self.component[0]] - self.values[:,0]
        #    for i in range(1,len(self.component)):
        #        tmp = outputs[beg:end, self.component[i]] - self.values[:,i]
        #        calculated_error = bkd.lib.concat([calculated_error,tmp],axis=0)
        # else:
        #    calculated_error = outputs[beg:end, self.component[0]] - self.values
        # return calculated_error
        return outputs[beg:end, self.component] - self.values


class PointSetOperatorBC:
    """General operator boundary conditions for a set of points.

    Compare the function output, func, (that associates with `points`)
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of
            `inputs`. `inputs` and `outputs` are the network input and output
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(self, points, values, func):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError("PointSetOperatorBC should output 1D values")
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.func = func

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, X)[beg:end] - self.values


class Interface2DBC:
    """2D interface boundary condition.

    This BC applies to the case with the following conditions:
    (1) the network output has two elements, i.e., output = [y1, y2],
    (2) the 2D geometry is ``dde.geometry.Rectangle`` or ``dde.geometry.Polygon``, which has two edges of the same length,
    (3) uniform boundary points are used, i.e., in ``dde.data.PDE`` or ``dde.data.TimePDE``, ``train_distribution="uniform"``.
    For a pair of points on the two edges, compute <output_1, d1> for the point on the first edge
    and <output_2, d2> for the point on the second edge in the n/t direction ('n' for normal or 't' for tangent).
    Here, <v1, v2> is the dot product between vectors v1 and v2;
    and d1 and d2 are the n/t vectors of the first and second edges, respectively.
    In the normal case, d1 and d2 are the outward normal vectors;
    and in the tangent case, d1 and d2 are the outward normal vectors rotated 90 degrees clockwise.
    The points on the two edges are paired as follows: the boundary points on one edge are sampled clockwise,
    and the points on the other edge are sampled counterclockwise. Then, compare the sum with 'values',
    i.e., the error is calculated as <output_1, d1> + <output_2, d2> - values,
    where 'values' is the argument `func` evaluated on the first edge.

    Args:
        geom: a ``dde.geometry.Rectangle`` or ``dde.geometry.Polygon`` instance.
        func: the target discontinuity between edges, evaluated on the first edge,
            e.g., ``func=lambda x: 0`` means no discontinuity is wanted.
        on_boundary1: First edge func. (x, Geometry.on_boundary(x)) -> True/False.
        on_boundary2: Second edge func. (x, Geometry.on_boundary(x)) -> True/False.
        direction (string): "normal" or "tangent".
    """

    def __init__(self, geom, func, on_boundary1, on_boundary2, direction="normal"):
        self.geom = geom
        self.func = npfunc_range_autocache(utils.return_tensor(func))
        self.on_boundary1 = lambda x, on: np.array(
            [on_boundary1(x[i], on[i]) for i in range(len(x))]
        )
        self.on_boundary2 = lambda x, on: np.array(
            [on_boundary2(x[i], on[i]) for i in range(len(x))]
        )
        self.direction = direction

        self.boundary_normal = npfunc_range_autocache(
            utils.return_tensor(self.geom.boundary_normal)
        )

    def collocation_points(self, X):
        on_boundary = self.geom.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when dde.geometry.Polygon is used
        if self.geom.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        mid = beg + (end - beg) // 2
        if not mid - beg == end - mid:
            raise RuntimeError(
                "There is a different number of points on each edge,\n\
                this is likely because the chosen edges do not have the same length."
            )
        values = self.func(X, beg, mid, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError("BC function should return an array of shape N by 1")
        left_n = self.boundary_normal(X, beg, mid, None)
        right_n = self.boundary_normal(X, mid, end, None)
        if self.direction == "normal":
            left_side = outputs[beg:mid, :]
            right_side = outputs[mid:end, :]
            left_values = bkd.sum(left_side * left_n, 1, keepdims=True)
            right_values = bkd.sum(right_side * right_n, 1, keepdims=True)

        elif self.direction == "tangent":
            # Tangent vector is [n[1],-n[0]] on edge 1
            left_side1 = outputs[beg:mid, 0:1]
            left_side2 = outputs[beg:mid, 1:2]
            right_side1 = outputs[mid:end, 0:1]
            right_side2 = outputs[mid:end, 1:2]
            left_values_1 = bkd.sum(left_side1 * left_n[:, 1:2], 1, keepdims=True)
            left_values_2 = bkd.sum(-left_side2 * left_n[:, 0:1], 1, keepdims=True)
            left_values = left_values_1 + left_values_2
            right_values_1 = bkd.sum(right_side1 * right_n[:, 1:2], 1, keepdims=True)
            right_values_2 = bkd.sum(-right_side2 * right_n[:, 0:1], 1, keepdims=True)
            right_values = right_values_1 + right_values_2

        return left_values + right_values - values


def npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for dde.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end, _):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_nocache_auxiliary(X, beg, end, aux_var):
        return func(X[beg:end], aux_var[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end, _):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    @wraps(func)
    def wrapper_cache_auxiliary(X, beg, end, aux_var):
        # Even if X is the same one, aux_var could be different
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end], aux_var[beg:end])
        return cache[key]

    if backend_name in ["tensorflow.compat.v1", "tensorflow", "jax"]:
        if utils.get_num_args(func) == 1:
            return wrapper_nocache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
    if backend_name in ["pytorch", "paddle"]:
        if utils.get_num_args(func) == 1:
            return wrapper_cache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
