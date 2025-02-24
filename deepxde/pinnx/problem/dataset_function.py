# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Callable, Sequence

import brainstate as bst

from deepxde.pinnx.geometry.base import GeometryPINNx
from deepxde.utils.internal import run_if_any_none
from .base import Problem

__all__ = [
    'Function',
]


class Function(Problem):
    """
    Approximate a function via a network.

    Args:
        geometry: The domain of the function. Instance of ``Geometry``.
        function: The function to be approximated. A callable function takes a NumPy array as the input and returns the
            a NumPy array of corresponding function values.
        num_train (int): The number of training points sampled inside the domain.
        num_test (int). The number of points for testing.
        train_distribution (string): The distribution to sample training points. One of the following: "uniform"
            (equispaced grid), "pseudo" (pseudorandom), "LHS" (Latin hypercube sampling), "Halton" (Halton sequence),
            "Hammersley" (Hammersley sequence), or "Sobol" (Sobol sequence).
        online (bool): If ``True``, resample the pseudorandom training points every training step, otherwise, use the
            same training points.
    """

    def __init__(
        self,
        geometry: GeometryPINNx,
        function: Callable,
        num_train: int,
        num_test: int,
        train_distribution: str = "uniform",
        online: bool = False,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(approximator=approximator, loss_fn=loss_fn, loss_weights=loss_weights)

        self.geom = geometry
        self.func = function
        self.num_train = num_train
        self.num_test = num_test
        self.dist_train = train_distribution
        self.online = online

        if online and train_distribution != "pseudo":
            print("Warning: Online learning should use pseudorandom sampling.")
            self.dist_train = "pseudo"

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def losses(self, inputs, outputs, targets, **kwargs):
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if self.train_x is None or self.online:
            if self.dist_train == "uniform":
                self.train_x = self.geom.uniform_points(self.num_train, boundary=True)
            else:
                self.train_x = self.geom.random_points(self.num_train, random=self.dist_train)
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.geom.uniform_points(self.num_test, boundary=True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
