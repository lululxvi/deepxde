# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from typing import Callable, Sequence

import brainstate as bst

from deepxde.experimental.geometry.base import GeometryExperimental
from deepxde.utils.internal import run_if_any_none
from .base import Problem

__all__ = [
    'Function',
]


class Function(Problem):
    """
    Approximate a function via a network.

    Args:
        geometry (GeometryExperimental): The domain of the function. Instance of ``Geometry``.
        function (Callable): The function to be approximated. A callable function takes a NumPy array as the input and returns the
            a NumPy array of corresponding function values.
        num_train (int): The number of training points sampled inside the domain.
        num_test (int): The number of points for testing.
        train_distribution (str, optional): The distribution to sample training points. One of the following: "uniform"
            (equispaced grid), "pseudo" (pseudorandom), "LHS" (Latin hypercube sampling), "Halton" (Halton sequence),
            "Hammersley" (Hammersley sequence), or "Sobol" (Sobol sequence). Defaults to "uniform".
        online (bool, optional): If ``True``, resample the pseudorandom training points every training step, otherwise, use the
            same training points. Defaults to False.
        approximator (bst.nn.Module, optional): The neural network module to use as an approximator. Defaults to None.
        loss_fn (str, optional): The loss function to use. Defaults to 'MSE'.
        loss_weights (Sequence[float], optional): The weights for different loss components. Defaults to None.
    """

    def __init__(
        self,
        geometry: GeometryExperimental,
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
        """
        Compute the loss between the predicted outputs and the target values.

        Args:
            inputs: The input data.
            outputs: The predicted output from the model.
            targets: The target values.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        """
        Generate the next batch of training data.

        Args:
            batch_size (int, optional): The size of the batch to generate. Defaults to None.

        Returns:
            tuple: A tuple containing the input features (train_x) and target values (train_y) for training.
        """
        if self.train_x is None or self.online:
            if self.dist_train == "uniform":
                self.train_x = self.geom.uniform_points(self.num_train, boundary=True)
            else:
                self.train_x = self.geom.random_points(self.num_train, random=self.dist_train)
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        """
        Generate test data points and their corresponding function values.

        Returns:
            tuple: A tuple containing the test input features (test_x) and their corresponding function values (test_y).
        """
        self.test_x = self.geom.uniform_points(self.num_test, boundary=True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
