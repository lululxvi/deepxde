# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

from typing import Callable, Dict

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from deepxde.data.sampler import BatchSampler
from deepxde.pinnx import utils
from deepxde.pinnx.nn.model import Model
from .base import ICBC

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

X = Dict[str, bst.typing.ArrayLike]
Y = Dict[str, bst.typing.ArrayLike]
F = Dict[str, bst.typing.ArrayLike]
Boundary = Dict[str, bst.typing.ArrayLike]


class BC(ICBC):
    """
    Boundary condition base class.

    Args:
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(
        self,
        on_boundary: Callable[[X, np.array], np.array],
    ):
        self.on_boundary = lambda x, on: jax.vmap(on_boundary)(x, on)

    @utils.check_not_none('geometry')
    def filter(self, X):
        """
        Filter the collocation points for boundary conditions.

        Args:
            X: Collocation points.

        Returns:
            Filtered collocation points.
        """
        positions = self.on_boundary(X, self.geometry.on_boundary(X))
        return jax.tree.map(lambda x: x[positions], X)

    def collocation_points(self, X):
        """
        Return the collocation points for boundary conditions.

        Args:
            X: Collocation points.

        Returns:
            Collocation points for boundary conditions.
        """
        return self.filter(X)

    def normal_derivative(self, inputs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Compute the normal derivative of the output.
        """
        # first order derivative
        assert isinstance(self.problem.approximator, Model), ("Normal derivative is only supported "
                                                              "for Sequential approximator.")
        dydx = self.problem.approximator.jacobian(inputs)

        # boundary normal
        n = self.geometry.boundary_normal(inputs)

        assert isinstance(n, dict), "Boundary normal should be a dictionary."
        assert isinstance(dydx, dict), "dydx should be a dictionary."
        norms = dict()
        for y in dydx:
            norm = None
            for x in dydx[y]:
                if norm is None:
                    norm = dydx[y][x] * n[x]
                else:
                    norm += dydx[y][x] * n[x]
            norms[y] = norm
        return norms


class DirichletBC(BC):
    """
    Dirichlet boundary conditions: ``y(x) = func(x)``.

    Args:
        func: A function that takes an array of points and returns an array of values.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    """

    def __init__(
        self,
        func: Callable[[X, ...], F] | Callable[[X], F] | F,
        on_boundary: Callable[[X, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func if callable(func) else lambda x: func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs, **kwargs)
        errors = dict()
        for component in values.keys():
            errors[component] = bc_outputs[component] - values[component]
        return errors


class NeumannBC(BC):
    """
    Neumann boundary conditions: ``dy/dn(x) = func(x)``.

    Args:
        func: A function that takes an array of points and returns an array of values.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(
        self,
        func: Callable[[X, ...], F] | Callable[[X], F],
        on_boundary: Callable[[X, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs, **kwargs)
        normals = self.normal_derivative(bc_inputs)
        return {
            component: normals[component] - values[component]
            for component in values.keys()
        }


class RobinBC(BC):
    """
    Robin boundary conditions: dy/dn(x) = func(x, y).
    """

    def __init__(
        self,
        func: Callable[[X, Y, ...], F] | Callable[[X, Y], F],
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs, bc_outputs, **kwargs)
        normals = self.normal_derivative(bc_inputs)
        return {
            component: normals[component] - values[component]
            for component in values.keys()
        }


class PeriodicBC(BC):
    """
    Periodic boundary conditions.

    Args:
        component_y: The component of the output.
        component_x: The component of the input.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
        derivative_order: The order of the derivative. Can be 0 or 1.
    """

    def __init__(
        self,
        component_y: str,
        component_x: str,
        on_boundary: Callable[[X, np.array], np.array] = lambda x, on: on,
        derivative_order: int = 0,
    ):
        super().__init__(on_boundary)
        self.component_y = component_y
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError("PeriodicBC only supports derivative_order 0 or 1.")

    @utils.check_not_none('geometry')
    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geometry.periodic_point(X1, self.component_x)
        return jax.tree.map(
            lambda x1, x2: utils.smart_numpy(x1).concatenate((x1, x2), axis=-1),
            X1,
            X2,
            is_leaf=u.math.is_quantity
        )

    def error(self, bc_inputs, bc_outputs, **kwargs):
        n_batch = bc_inputs[self.component_x].shape[0]
        mid = n_batch // 2
        if self.derivative_order == 0:
            yleft = bc_outputs[self.component_y][:mid]
            yright = bc_outputs[self.component_y][mid:]
        else:
            dydx = self.problem.approximator.jacobian(bc_outputs, y=self.component_y, x=self.component_x)
            dydx = dydx[self.component_y][self.component_x]
            yleft = dydx[:mid]
            yright = dydx[mid:]
        return {self.component_y: {self.component_x: yleft - yright}}


class OperatorBC(BC):
    """
    General operator boundary conditions: func(inputs, outputs) = 0.

    Args:
        func: A function takes arguments (`inputs`, `outputs`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    Warning:
        If you use `X` in `func`, then do not set ``num_test`` when you define
        ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``, otherwise DeepXDE would throw an
        error. In this case, the training points will be used for testing, and this will
        not affect the network training and training loss. This is a bug of DeepXDE,
        which cannot be fixed in an easy way for all backends.
    """

    def __init__(
        self,
        func: Callable[[X, Y, ...], F] | Callable[[X, Y], F],
        on_boundary: Callable[[X, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        return self.func(bc_inputs, bc_outputs, **kwargs)


class PointSetBC(BC):
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'pinnx.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(
        self,
        points: Dict[str, bst.typing.ArrayLike],
        values: Dict[str, bst.typing.ArrayLike],
        batch_size: int = None,
        shuffle: bool = True
    ):
        super().__init__(lambda x, on: on)

        self.points = points
        self.values = values
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            self.batch_sampler = BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        v = tuple(self.points.values())[0]
        return v.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return jax.tree.map(lambda x: x[self.batch_indices], self.points)
        return self.points

    def error(self, bc_inputs, bc_outputs, **kwargs):
        if self.batch_size is not None:
            return {
                k: bc_outputs[k] - self.values[k][self.batch_indices]
                for k in self.values.keys()
            }
        else:
            return {
                k: bc_outputs[k] - self.values[k]
                for k in self.values.keys()
            }


class PointSetOperatorBC(BC):
    """
    General operator boundary conditions for a set of points.

    Compare the function output, func, (that associates with `points`)
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`,)
            and outputs a tensor of size `N x 1`, where `N` is the length of
            `inputs`. `inputs` and `outputs` are the network input and output
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(
        self,
        points: Dict[str, bst.typing.ArrayLike],
        values: Dict[str, bst.typing.ArrayLike],
        func: Callable[[X, Y], F]
    ):
        super().__init__(lambda x, on: on)
        self.points = points
        self.values = values
        self.func = func

    def collocation_points(self, X):
        return self.points

    def error(self, bc_inputs, bc_outputs, **kwargs):
        outs = self.func(bc_inputs, bc_outputs)
        return {
            component: outs[component] - self.values[component]
            for component in outs.keys()
        }


class Interface2DBC(BC):
    """2D interface boundary condition.

    This BC applies to the case with the following conditions:
    (1) the network output has two elements, i.e., output = [y1, y2],
    (2) the 2D geometry is ``pinnx.geometry.Rectangle`` or ``pinnx.geometry.Polygon``, which has two edges of the same length,
    (3) uniform boundary points are used, i.e., in ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``, ``train_distribution="uniform"``.
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
        func: the target discontinuity between edges, evaluated on the first edge,
            e.g., ``func=lambda x: 0`` means no discontinuity is wanted.
        on_boundary1: First edge func. (x, Geometry.on_boundary(x)) -> True/False.
        on_boundary2: Second edge func. (x, Geometry.on_boundary(x)) -> True/False.
        direction (string): "normal" or "tangent".
    """

    def __init__(
        self,
        func: Callable[[X, ...], F] | Callable[[X], F],
        on_boundary1: Callable[[X, np.array], np.array] = lambda x, on: on,
        on_boundary2: Callable[[X, np.array], np.array] = lambda x, on: on,
        direction: str = "normal"
    ):
        super().__init__(lambda x, on: on)

        self.func = utils.return_tensor(func)
        self.on_boundary1 = lambda x, on: np.array([on_boundary1(x[i], on[i]) for i in range(len(x))])
        self.on_boundary2 = lambda x, on: np.array([on_boundary2(x[i], on[i]) for i in range(len(x))])
        self.direction = direction

    @utils.check_not_none('geometry')
    def collocation_points(self, X):
        on_boundary = self.geometry.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when pinnx.geometry.Polygon is used
        if self.geometry.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    @utils.check_not_none('geometry')
    def error(self, bc_inputs, bc_outputs, **kwargs):
        mid = bc_inputs.shape[0] // 2
        if bc_inputs.shape[0] % 2 != 0:
            raise RuntimeError("There is a different number of points on each edge,\n "
                               "this is likely because the chosen edges do not have the same length.")
        aux_var = None
        values = self.func(bc_inputs[: mid], **kwargs)
        if np.ndim(values) == 2 and np.shape(values)[1] != 1:
            raise RuntimeError("BC function should return an array of shape N by 1")
        left_n = self.geometry.boundary_normal(bc_inputs[: mid])
        right_n = self.geometry.boundary_normal(bc_inputs[: mid])
        if self.direction == "normal":
            left_side = bc_outputs[:mid, :]
            right_side = bc_outputs[mid:, :]
            left_values = u.math.sum(left_side * left_n, 1, keepdims=True)
            right_values = u.math.sum(right_side * right_n, 1, keepdims=True)

        elif self.direction == "tangent":
            # Tangent vector is [n[1],-n[0]] on edge 1
            left_side1 = bc_outputs[:mid, 0:1]
            left_side2 = bc_outputs[:mid, 1:2]
            right_side1 = bc_outputs[mid:, 0:1]
            right_side2 = bc_outputs[mid:, 1:2]
            left_values_1 = u.math.sum(left_side1 * left_n[:, 1:2], 1, keepdims=True)
            left_values_2 = u.math.sum(-left_side2 * left_n[:, 0:1], 1, keepdims=True)
            left_values = left_values_1 + left_values_2
            right_values_1 = u.math.sum(right_side1 * right_n[:, 1:2], 1, keepdims=True)
            right_values_2 = u.math.sum(-right_side2 * right_n[:, 0:1], 1, keepdims=True)
            right_values = right_values_1 + right_values_2

        else:
            raise ValueError("Invalid direction, must be 'normal' or 'tangent'.")

        return left_values + right_values - values
