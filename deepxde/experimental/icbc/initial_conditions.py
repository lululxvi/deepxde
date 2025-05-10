from __future__ import annotations

from typing import Callable, Dict

import brainstate as bst
import jax
import numpy as np

from .base import ICBC

__all__ = ["IC"]


class IC(ICBC):
    """
    Represents the Initial Conditions (IC) for a differential equation.

    This class defines and handles the initial conditions of the form:
    y([x, t0]) = func([x, t0]), where func is a user-defined function.

    Args:
        func (Callable[[Dict, ...], Dict] | Callable[[Dict], Dict]): A function that returns the initial conditions.
            This function should take a dictionary of collocation points and
            return a dictionary of initial conditions. For example:
                import brainunit as u
                def func(x):
                    return {'y': -u.math.sin(np.pi * x['x'] / u.meter) * u.meter / u.second}
        on_initial (Callable[[Dict, np.array], np.array], optional): A filter function for initial conditions.
            This function should take a dictionary of collocation points and
            return a boolean array indicating whether the points are initial conditions.
            Defaults to lambda x, on: on. For example:
                def on_initial(x, on):
                    return on
    """

    def __init__(
        self,
        func: Callable[[Dict, ...], Dict] | Callable[[Dict], Dict],
        on_initial: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        self.func = func
        self.on_initial = lambda x, on: jax.vmap(on_initial)(x, on)

    def filter(self, X):
        """
        Filters the collocation points for initial conditions.

        Args:
            X (Dict): A dictionary of collocation points.

        Returns:
            Dict: Filtered collocation points that satisfy the initial conditions.
        """
        # the "geometry" should be "TimeDomain" or "GeometryXTime"
        positions = self.on_initial(X, self.geometry.on_initial(X))
        return jax.tree.map(lambda x: x[positions], X)

    def collocation_points(self, X):
        """
        Returns the collocation points for initial conditions.

        Args:
            X (Dict): A dictionary of collocation points.

        Returns:
            Dict: Collocation points that satisfy the initial conditions.
        """
        return self.filter(X)

    def error(self, inputs, outputs, **kwargs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Calculates the error for initial conditions.

        This method compares the initial conditions with the outputs to compute the error.

        Args:
            inputs (Dict): A dictionary of collocation points.
            outputs (Dict): A dictionary of collocation values.
            **kwargs: Additional keyword arguments to be passed to the func method.

        Returns:
            Dict[str, bst.typing.ArrayLike]: A dictionary containing the errors for each variable.
                The keys correspond to the variable names, and the values are the computed errors.
        """
        values = self.func(inputs, **kwargs)
        errors = dict()
        for key, value in values.items():
            errors[key] = outputs[key] - value
        return errors
