# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import abc
from typing import Optional, Dict

import brainstate as bst

from deepxde.pinnx.geometry.base import GeometryPINNx


class ICBC(abc.ABC):
    """
    Base class for initial and boundary conditions.
    """

    # A ``pinnx.geometry.Geometry`` instance.
    geometry: Optional[GeometryPINNx]
    problem: Optional['Problem']

    def apply_geometry(self, geom: GeometryPINNx):
        assert isinstance(geom, GeometryPINNx), 'geometry must be an instance of AbstractGeometry.'
        self.geometry = geom

    def apply_problem(self, problem: 'Problem'):
        from deepxde.pinnx.problem.base import Problem
        assert isinstance(problem, Problem), 'problem must be an instance of Problem.'
        self.problem = problem

    @abc.abstractmethod
    def filter(self, X):
        """
        Filters the input data.
        """
        pass

    @abc.abstractmethod
    def collocation_points(self, X):
        """
        Returns the collocation points.
        """
        pass

    @abc.abstractmethod
    def error(self, inputs, outputs, **kwargs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Returns the loss for each component at the initial or boundary conditions.
        """
