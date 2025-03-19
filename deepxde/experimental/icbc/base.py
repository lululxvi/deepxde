import abc
from typing import Optional, Dict

import brainstate as bst

from deepxde.experimental.geometry.base import GeometryExperimental


class ICBC(abc.ABC):
    """
    Base class for initial and boundary conditions.
    """

    # A ``experimental.geometry.Geometry`` instance.
    geometry: Optional[GeometryExperimental]
    problem: Optional['Problem']

    def apply_geometry(self, geom: GeometryExperimental):
        """
        Applies a geometry to the ICBC instance.

        Parameters:
        -----------
        geom : GeometryExperimental
            The geometry to be applied to the ICBC instance.

        Raises:
        -------
        AssertionError
            If the provided geometry is not an instance of AbstractGeometry.
        """
        assert isinstance(geom, GeometryExperimental), 'geometry must be an instance of AbstractGeometry.'
        self.geometry = geom

    def apply_problem(self, problem: 'Problem'):
        """
        Applies a problem to the ICBC instance.

        Parameters:
        -----------
        problem : Problem
            The problem to be applied to the ICBC instance.

        Raises:
        -------
        AssertionError
            If the provided problem is not an instance of Problem.
        """
        from deepxde.experimental.problem.base import Problem
        assert isinstance(problem, Problem), 'problem must be an instance of Problem.'
        self.problem = problem

    @abc.abstractmethod
    def filter(self, X):
        """
        Filters the input data.

        Parameters:
        -----------
        X : array-like
            The input data to be filtered.

        Returns:
        --------
        array-like
            The filtered input data.
        """
        pass

    @abc.abstractmethod
    def collocation_points(self, X):
        """
        Returns the collocation points.

        Parameters:
        -----------
        X : array-like
            The input data for which to compute collocation points.

        Returns:
        --------
        array-like
            The computed collocation points.
        """
        pass

    @abc.abstractmethod
    def error(self, inputs, outputs, **kwargs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Returns the loss for each component at the initial or boundary conditions.

        Parameters:
        -----------
        inputs : array-like
            The input data.
        outputs : array-like
            The output data.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        Dict[str, bst.typing.ArrayLike]
            A dictionary containing the loss for each component at the initial or boundary conditions.
        """
        pass
