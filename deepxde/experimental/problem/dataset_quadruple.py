from typing import Sequence

import brainstate as bst

from deepxde.data.sampler import BatchSampler
from .base import Problem

__all__ = [
    "QuadrupleDataset",
]


class QuadrupleDataset(Problem):
    """
    Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train (tuple): A tuple of three NumPy arrays representing the input training data.
        y_train (numpy.ndarray): A NumPy array representing the output training data.
        X_test (tuple): A tuple of three NumPy arrays representing the input testing data.
        y_test (numpy.ndarray): A NumPy array representing the output testing data.
        approximator (bst.nn.Module, optional): The neural network module used for approximation. Defaults to None.
        loss_fn (str, optional): The loss function to be used. Defaults to 'MSE'.
        loss_weights (Sequence[float], optional): Weights for the loss function. Defaults to None.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        approximator: bst.nn.Module = None,
        loss_fn: str = "MSE",
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(
            approximator=approximator, loss_fn=loss_fn, loss_weights=loss_weights
        )
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Calculate the loss between the predicted outputs and the target values.

        Args:
            inputs: The input data (not used in this method).
            outputs: The predicted output values.
            targets: The target output values.
            **kwargs: Additional keyword arguments.

        Returns:
            The calculated loss value.
        """
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        """
        Get the next batch of training data.

        Args:
            batch_size (int, optional): The size of the batch to return. If None, returns all training data.

        Returns:
            tuple: A tuple containing the input data (as a tuple of arrays) and the corresponding output data.
        """
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices]),
            self.train_x[2][indices],
            self.train_y[indices],
        )

    def test(self):
        """
        Get the testing data.

        Returns:
            tuple: A tuple containing the input testing data and the corresponding output testing data.
        """
        return self.test_x, self.test_y
