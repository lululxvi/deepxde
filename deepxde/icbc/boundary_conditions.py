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
        """Extracts points from a set that satisfy the boundary condition.

        Args:
            X (np.ndarray): An array of points (coordinates).

        Returns:
            np.ndarray: A subset of ``X`` containing only points that lie
            on the specific boundary defined by ``self.on_boundary``.
        """
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        """Returns the points where the boundary condition error will be evaluated.

        For standard BCs, this is identical to the filtered boundary points.
        Subclasses like ``PeriodicBC`` may override this to include
        paired points.

        Args:
            X (np.ndarray): The full set of available boundary points.

        Returns:
            np.ndarray: The subset of points designated for BC loss calculation.
        """
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        r"""Computes the directional derivative along the outward normal vector.

        This is used for Neumann and Robin boundary conditions to calculate
        :math:`\frac{\partial \hat{y}}{\partial n} = \nabla \hat{y} \cdot \mathbf{n}`.

        Args:
            X (np.ndarray): The coordinates of the boundary points.
            inputs (Tensor): The input tensor to the neural network.
            outputs (Tensor): The output tensor from the neural network.
            beg (int): The starting index of the points in the current batch.
            end (int): The ending index of the points in the current batch.

        Returns:
            Tensor: A column vector representing the normal derivative at
            each point in the range [beg, end].
        """
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return bkd.sum(dydx * n, 1, keepdims=True)

    @abstractmethod
    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Calculates the residual (loss) for the boundary condition.

        This method must be implemented by subclasses to define the
        specific physics of the boundary (e.g., :math:`\hat{y} - y_{true}`).

        Args:
            X (np.ndarray): Boundary point coordinates.
            inputs (Tensor): Neural network input tensor.
            outputs (Tensor): Neural network output tensor.
            beg (int): Start index for the point batch.
            end (int): End index for the point batch.
            aux_var (Tensor, optional): The input function evaluated at x. Only used in PI-DeepONet architectures.

        Returns:
            Tensor: The computed residual which the optimizer will
            attempt to minimize toward zero.
        """


class DirichletBC(BC):
    """Dirichlet boundary conditions: $y(x) = f(x)$.

    Enforces the value of the solution at the boundary.

    Args:
        geom: A ``dde.geometry.Geometry`` instance.
        func: A function returning the boundary values.
            Signature: ``(x) -> array_like (N, 1)``.
        on_boundary: Function identifying the target boundary.
        component (int): Output component to which this BC applies.
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Returns the Dirichlet BC residual: $\hat{y}(x) - f(x)$."""
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x).

    Enforces the value of the derivative of the solution in the direction
    of the outward normal.

    Args:
        geom: A ``dde.geometry.Geometry`` instance.
        func: A function returning the boundary values.
            Signature: ``(x) -> array_like (N, 1)``.
        on_boundary: Function identifying the target boundary.
        component (int): Output component to which this BC applies.
    """

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        r"""Computes the Neumann residual: $\frac{\partial \hat{y}}{\partial n} - f(x)$."""
        values = self.func(X, beg, end, aux_var)
        return self.normal_derivative(X, inputs, outputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary condition: $\frac{\partial y}{\partial n}(x) = f(x, y)$."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Computes the Robin residual: $\frac{\partial \hat{y}}{\partial n} - f(x, \hat{y})$."""
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary condition: $y(x_{left}) = y(x_{right})$.

    Setting derivative_order=1 enforces periodicity of the first derivative, setting derivative_order=0 enforces periodicity of the function values.
    """

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def collocation_points(self, X):
        """Generates pairs of points on opposing periodic boundaries.

        Note that for Periodic Boundary Conditions, the collocation points are pairs rather than individual points on a single boundary.
        This method identifies points on the first boundary and maps them to their corresponding points on the second boundary
        using the geometry's periodic mapping function.

        Returns:
            np.ndarray: A vertical stack of points from boundary 1 and their
                corresponding mapped points on boundary 2.
        """
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Computes the difference in first derivative (if self.derivative_order == 1)
        or function value (if self.derivative_order == 0) between the two periodic edges.
        """
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
        """Computes the residual of the operator BC: func(inputs, outputs, X)."""
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
        """Returns the total number of points in the BC."""
        return self.points.shape[0]

    def collocation_points(self, X):
        """Retrieves points for the current training iteration."""
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Computes the residual between network output and ground truth data."""
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
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'dde.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, func, batch_size=None, shuffle=True):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError("PointSetOperatorBC should output 1D values")
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.func = func
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            if backend_name not in ["pytorch", "paddle"]:
                raise RuntimeError(
                    "batch_size only implemented for pytorch and paddle backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        """Returns the total number of points in the BC."""

        return self.points.shape[0]

    def collocation_points(self, X):
        """Retrieves points for the current training iteration."""

        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Computes user-defined operator residual minus target values."""

        if self.batch_size is not None:
            return self.func(inputs, outputs, X)[beg:end] - self.values[self.batch_indices]
        return self.func(inputs, outputs, X)[beg:end] - self.values


class Interface2DBC:
    """2D interface boundary condition for vector-valued outputs.

    This boundary condition (BC) is designed for scenarios where specific jump conditions
    or continuities are required across two matching edges of a geometry.

    * **Network Output:** The model must have exactly two output elements, i.e., :math:`\\mathbf{y} = [y_1, y_2]`.
    * **Geometry:** Must be a ``dde.geometry.Rectangle`` or ``dde.geometry.Polygon`` with two edges of identical length.
    * **Sampling:** Use uniform boundary points (``train_distribution="uniform"``) in ``dde.data.PDE`` or ``dde.data.TimePDE``.

    For a pair of points :math:`x_1` and :math:`x_2` located on the two specified edges, the BC computes the
    dot product of the output and the direction vector :math:`\mathbf{d}`:

    .. math:: \langle \mathbf{y}_1, \mathbf{d}_1 \\rangle + \langle \mathbf{y}_2, \mathbf{d}_2 \\rangle = \\text{values}

    Where:

    * :math:`\mathbf{d}_1, \mathbf{d}_2`: The unit vectors based on the ``direction`` argument.
    * **Normal Case:** :math:`\mathbf{d}` represents the outward normal vectors.
    * **Tangent Case:** :math:`\mathbf{d}` represents the outward normal vectors rotated 90° clockwise.
    * **Point Pairing:** Points on the first edge are sampled clockwise; points on the second edge are sampled counter-clockwise to ensure they are correctly mapped spatially.

    Args:
        geom: A ``dde.geometry.Rectangle`` or ``dde.geometry.Polygon`` instance.
        func (callable): The target discontinuity (jump) between edges, evaluated on the
            first edge. For example, ``lambda x: 0`` enforces continuity.
        on_boundary1 (callable): Function identifying the first edge.
            Signature: ``(x, on_boundary) -> bool``.
        on_boundary2 (callable): Function identifying the second edge.
            Signature: ``(x, on_boundary) -> bool``.
        direction (str): The vector component to constrain. Options are ``"normal"``
            or ``"tangent"``.
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
        """Pairs points on two matching edges, ensuring spatial alignment.

        Reverses the order of points on the second edge for polygons to 
        correctly match point-to-point across the interface if dde.geometry.Polygon is used.
        """
        on_boundary = self.geom.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when dde.geometry.Polygon is used
        if self.geom.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Computes the jump residual based on projected vector components."""
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
