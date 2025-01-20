# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

from __future__ import annotations

from typing import Callable, Dict

import brainstate as bst
import jax
import numpy as np

from .base import ICBC

__all__ = ["IC"]


class IC(ICBC):
    """
    Initial conditions: ``y([x, t0]) = func([x, t0])``.

    Args:
        func: Function that returns the initial conditions.
            This function should take a dictionary of collocation points and
            return a dictionary of initial conditions. For example::

                import brainunit as u
                def func(x):
                    return {'y': -u.math.sin(np.pi * x['x'] / u.meter) * u.meter / u.second}
        on_initial: Filter function for initial conditions.
            This function should take a dictionary of collocation points and
            return a boolean array indicating whether the points are initial conditions.
            For example::

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
        Filter the collocation points for initial conditions.

        Args:
            X: Collocation points.

        Returns:
            Filtered collocation points.
        """
        # the "geometry" should be "TimeDomain" or "GeometryXTime"
        positions = self.on_initial(X, self.geometry.on_initial(X))
        return jax.tree.map(lambda x: x[positions], X)

    def collocation_points(self, X):
        """
        Return the collocation points for initial conditions.

        Args:
            X: Collocation points.

        Returns:
            Collocation points for initial conditions.
        """
        return self.filter(X)

    def error(self, inputs, outputs, **kwargs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Error for initial conditions.

        Compare the initial conditions with the outputs.

        Args:
            inputs: Collocation points.
            outputs: Collocation values.

        Returns:
            Error for initial conditions.
        """
        values = self.func(inputs, **kwargs)
        errors = dict()
        for key, value in values.items():
            errors[key] = outputs[key] - value
        return errors
