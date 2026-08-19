from __future__ import annotations

from typing import Callable, Dict

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from deepxde.data.sampler import BatchSampler
from deepxde.experimental import utils
from deepxde.experimental.nn.model import Model
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

    This class serves as the foundation for implementing various boundary conditions in the DeepXDE framework.
    It provides methods for filtering collocation points, computing normal derivatives, and handling boundary-related operations.

    Args:
        on_boundary (Callable[[X, np.array], np.array]): A function that takes two arguments:
            - x: The input points.
            - on: A boolean array indicating whether each point is on the boundary.
            The function should return a boolean array indicating which points satisfy the boundary condition.

    Attributes:
        on_boundary (Callable): A vectorized version of the input `on_boundary` function.
    """

    def __init__(
        self,
        on_boundary: Callable[[X, np.array], np.array],
    ):
        self.on_boundary = lambda x, on: jax.vmap(on_boundary)(x, on)

    @utils.check_not_none("geometry")
    def filter(self, X):
        """
        Filter the collocation points for boundary conditions.

        This method applies the boundary condition filter to the given collocation points.

        Args:
            X (Dict[str, bst.typing.ArrayLike]): A dictionary of collocation points.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary of filtered collocation points that satisfy the boundary condition.
        """
        positions = self.on_boundary(X, self.geometry.on_boundary(X))
        return jax.tree.map(lambda x: x[positions], X)

    def collocation_points(self, X):
        """
        Return the collocation points for boundary conditions.

        This method filters the input collocation points to return only those that satisfy the boundary condition.

        Args:
            X (Dict[str, bst.typing.ArrayLike]): A dictionary of collocation points.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary of collocation points that satisfy the boundary condition.
        """
        return self.filter(X)

    def normal_derivative(self, inputs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Compute the normal derivative of the output.

        This method calculates the normal derivative of the output with respect to the input at the boundary.

        Args:
            inputs (Dict[str, bst.typing.ArrayLike]): A dictionary of input points.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the normal derivatives of the output
            with respect to each input variable.

        Raises:
            AssertionError: If the problem approximator is not an instance of the Model class,
            or if the boundary normal or jacobian are not dictionaries.
        """
        # first order derivative
        assert isinstance(self.problem.approximator, Model), (
            "Normal derivative is only supported " "for Sequential approximator."
        )
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

    This class implements Dirichlet boundary conditions, where the solution is specified
    on the boundary of the domain.

    Args:
        func (Callable[[X, ...], F] | Callable[[X], F] | F): A function that takes an array of points
            and returns an array of values, or a constant value to be applied at all boundary points.
        on_boundary (Callable[[X, np.array], np.array], optional): A function that takes two arguments:
            x (the input points) and on (a boolean array indicating whether each point is on the boundary).
            It should return a boolean array indicating which points satisfy the boundary condition.
            Defaults to a function that returns the input 'on' array.

    """

    def __init__(
        self,
        func: Callable[[X, ...], F] | Callable[[X], F] | F,
        on_boundary: Callable[[X, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func if callable(func) else lambda x: func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        """
        Calculate the error between the predicted and true values at the boundary.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): Input points on the boundary.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): Predicted output values at the boundary points.
            **kwargs: Additional keyword arguments to be passed to self.func.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the errors for each output component.
                The keys are the component names, and the values are the differences between
                the predicted and true values at the boundary points.
        """
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
        """
        Calculate the error for Neumann boundary conditions.

        This method computes the difference between the normal derivative of the solution
        and the specified function values at the boundary points.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): Input points on the boundary.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): Predicted output values at the boundary points.
            **kwargs: Additional keyword arguments to be passed to self.func.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the errors for each output component.
                The keys are the component names, and the values are the differences between
                the normal derivatives and the specified function values at the boundary points.
        """
        values = self.func(bc_inputs, **kwargs)
        normals = self.normal_derivative(bc_inputs)
        return {
            component: normals[component] - values[component]
            for component in values.keys()
        }


class RobinBC(BC):
    """
    Robin boundary conditions: dy/dn(x) = func(x, y).

    This class implements Robin boundary conditions, which are a combination of
    Dirichlet and Neumann boundary conditions.

    Attributes:
        func (Callable): The function defining the Robin boundary condition.
    """

    def __init__(
        self,
        func: Callable[[X, Y, ...], F] | Callable[[X, Y], F],
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        """
        Initialize the RobinBC class.

        Args:
            func (Callable[[X, Y, ...], F] | Callable[[X, Y], F]): A function that takes
                input points (X) and output values (Y) and returns the right-hand side
                of the Robin boundary condition equation.
            on_boundary (Callable[[Dict, np.array], np.array], optional): A function that
                determines which points are on the boundary. Defaults to a function that
                returns the input 'on' array.
        """
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        """
        Calculate the error for the Robin boundary condition.

        This method computes the difference between the normal derivative of the solution
        and the specified function values at the boundary points.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): Input points on the boundary.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): Predicted output values at the boundary points.
            **kwargs: Additional keyword arguments to be passed to self.func.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the errors for each output component.
                The keys are the component names, and the values are the differences between
                the normal derivatives and the specified function values at the boundary points.
        """
        values = self.func(bc_inputs, bc_outputs, **kwargs)
        normals = self.normal_derivative(bc_inputs)
        return {
            component: normals[component] - values[component]
            for component in values.keys()
        }


class PeriodicBC(BC):
    """
    Implements periodic boundary conditions for a specified component of the solution.

    This class enforces periodicity by ensuring that the values (or derivatives) of the solution
    at corresponding points on opposite boundaries are equal.

    Args:
        component_y (str): The name of the output component to which the periodic condition is applied.
        component_x (str): The name of the input component along which the periodicity is enforced.
        on_boundary (Callable[[X, np.array], np.array], optional): A function that takes two arguments:
            x (the input points) and on (a boolean array indicating whether each point is on the boundary).
            It should return a boolean array indicating which points satisfy the boundary condition.
            Defaults to a function that returns the input 'on' array.
        derivative_order (int, optional): The order of the derivative for which periodicity is enforced.
            Can be 0 (for function values) or 1 (for first derivatives). Defaults to 0.

    Raises:
        NotImplementedError: If derivative_order is greater than 1.
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
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    @utils.check_not_none("geometry")
    def collocation_points(self, X):
        """
        Generates collocation points for enforcing periodic boundary conditions.

        This method filters the input points, identifies the periodic points, and concatenates
        them to create pairs of points for enforcing periodicity.

        Args:
            X (Dict[str, bst.typing.ArrayLike]): A dictionary of input points.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary of collocation points, where each entry
            is the concatenation of points on one boundary and their periodic counterparts.
        """
        X1 = self.filter(X)
        X2 = self.geometry.periodic_point(X1, self.component_x)
        return jax.tree.map(
            lambda x1, x2: utils.smart_numpy(x1).concatenate((x1, x2), axis=-1),
            X1,
            X2,
            is_leaf=u.math.is_quantity,
        )

    def error(self, bc_inputs, bc_outputs, **kwargs):
        """
        Calculates the error for periodic boundary conditions.

        This method computes the difference between the values (or derivatives) of the solution
        at corresponding points on opposite boundaries.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): Input points on the boundary.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): Predicted output values at the boundary points.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            Dict[str, Dict[str, bst.typing.ArrayLike]]: A nested dictionary containing the errors.
            The outer key is the output component name, and the inner key is the input component name.
            The value is the difference between the left and right boundary values or derivatives.
        """
        n_batch = bc_inputs[self.component_x].shape[0]
        mid = n_batch // 2
        if self.derivative_order == 0:
            yleft = bc_outputs[self.component_y][:mid]
            yright = bc_outputs[self.component_y][mid:]
        else:
            dydx = self.problem.approximator.jacobian(
                bc_outputs, y=self.component_y, x=self.component_x
            )
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
        ``experimental.problem.PDE`` or ``experimental.problem.TimePDE``, otherwise DeepXDE would throw an
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
        """
        Calculate the error for the operator boundary condition.

        This method applies the operator function to the boundary inputs and outputs
        to compute the error of the boundary condition.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): A dictionary of input values at the boundary points.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): A dictionary of output values at the boundary points.
            **kwargs: Additional keyword arguments to be passed to the operator function.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the computed error values
            for each component of the boundary condition.
        """
        return self.func(bc_inputs, bc_outputs, **kwargs)


class PointSetBC(BC):
    """
    Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points (Dict[str, bst.typing.ArrayLike]): A dictionary of arrays representing points
            where the corresponding target values are known and used for training.
        values (Dict[str, bst.typing.ArrayLike]): A dictionary of scalars or 2D-arrays
            representing the exact solution of the problem at the given points.
        batch_size (int, optional): The number of points per minibatch, or None to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle. Defaults to None.
        shuffle (bool, optional): Whether to randomize the order on each pass through the data
            when batching. Defaults to True.

    Note:
        If you want to use batch size here, you should also set callback
        'experimental.callbacks.PDEPointResampler(bc_points=True)' in training.
    """

    def __init__(
        self,
        points: Dict[str, bst.typing.ArrayLike],
        values: Dict[str, bst.typing.ArrayLike],
        batch_size: int = None,
        shuffle: bool = True,
    ):
        super().__init__(lambda x, on: on)

        self.points = points
        self.values = values
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            self.batch_sampler = BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        """
        Get the number of points in the PointSetBC.

        Returns:
            int: The number of points in the first value of the points dictionary.
        """
        v = tuple(self.points.values())[0]
        return v.shape[0]

    def collocation_points(self, X):
        """
        Get the collocation points for the boundary condition.

        If batch_size is set, returns a batch of points. Otherwise, returns all points.

        Args:
            X: Unused in this method, kept for compatibility with parent class.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary of collocation points,
            either a batch or all points depending on the batch_size setting.
        """
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return jax.tree.map(lambda x: x[self.batch_indices], self.points)
        return self.points

    def error(self, bc_inputs, bc_outputs, **kwargs):
        """
        Calculate the error between the predicted and true values at the boundary points.

        Args:
            bc_inputs: Unused in this method, kept for compatibility with parent class.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): A dictionary of predicted output values
                at the boundary points.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the errors for each output component.
                The keys are the component names, and the values are the differences between
                the predicted and true values at the boundary points.
        """
        if self.batch_size is not None:
            return {
                k: bc_outputs[k] - self.values[k][self.batch_indices]
                for k in self.values.keys()
            }
        else:
            return {k: bc_outputs[k] - self.values[k] for k in self.values.keys()}


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
        func: Callable[[X, Y], F],
    ):
        super().__init__(lambda x, on: on)
        self.points = points
        self.values = values
        self.func = func

    def collocation_points(self, X):
        """
        Return the collocation points for the boundary condition.

        Args:
            X: Unused input parameter, kept for compatibility with parent class.

        Returns:
            Dict[str, bst.typing.ArrayLike]: The points where the boundary condition is applied.
        """
        return self.points

    def error(self, bc_inputs, bc_outputs, **kwargs):
        """
        Calculate the error for the operator boundary condition.

        This method applies the operator function to the boundary inputs and outputs,
        then computes the difference between the function output and the target values.

        Args:
            bc_inputs (Dict[str, bst.typing.ArrayLike]): Input values at the boundary points.
            bc_outputs (Dict[str, bst.typing.ArrayLike]): Output values at the boundary points.
            **kwargs: Additional keyword arguments to be passed to the operator function.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the computed error values
            for each component of the boundary condition.
        """
        outs = self.func(bc_inputs, bc_outputs)
        return {
            component: outs[component] - self.values[component]
            for component in outs.keys()
        }


class Interface2DBC(BC):
    """2D interface boundary condition.

    This BC applies to the case with the following conditions:
    (1) the network output has two elements, i.e., output = [y1, y2],
    (2) the 2D geometry is ``experimental.geometry.Rectangle`` or ``experimental.geometry.Polygon``, which has two edges of the same length,
    (3) uniform boundary points are used, i.e., in ``experimental.problem.PDE`` or ``experimental.problem.TimePDE``, ``train_distribution="uniform"``.
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
        direction: str = "normal",
    ):
        super().__init__(lambda x, on: on)

        self.func = utils.return_tensor(func)
        self.on_boundary1 = lambda x, on: np.array(
            [on_boundary1(x[i], on[i]) for i in range(len(x))]
        )
        self.on_boundary2 = lambda x, on: np.array(
            [on_boundary2(x[i], on[i]) for i in range(len(x))]
        )
        self.direction = direction

    @utils.check_not_none("geometry")
    def collocation_points(self, X):
        on_boundary = self.geometry.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when experimental.geometry.Polygon is used
        if self.geometry.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    @utils.check_not_none("geometry")
    def error(self, bc_inputs, bc_outputs, **kwargs):
        mid = bc_inputs.shape[0] // 2
        if bc_inputs.shape[0] % 2 != 0:
            raise RuntimeError(
                "There is a different number of points on each edge,\n "
                "this is likely because the chosen edges do not have the same length."
            )
        aux_var = None
        values = self.func(bc_inputs[:mid], **kwargs)
        if np.ndim(values) == 2 and np.shape(values)[1] != 1:
            raise RuntimeError("BC function should return an array of shape N by 1")
        left_n = self.geometry.boundary_normal(bc_inputs[:mid])
        right_n = self.geometry.boundary_normal(bc_inputs[:mid])
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
            right_values_2 = u.math.sum(
                -right_side2 * right_n[:, 0:1], 1, keepdims=True
            )
            right_values = right_values_1 + right_values_2

        else:
            raise ValueError("Invalid direction, must be 'normal' or 'tangent'.")

        return left_values + right_values - values
