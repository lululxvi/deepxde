from typing import Sequence, Dict

import brainstate as bst
import jax
import numpy as np

from deepxde.experimental import utils
from .base import Problem

__all__ = ["DataSet"]


class DataSet(Problem):
    """
    Fitting Problem set for handling dataset-based machine learning problems.

    This class extends the Problem class to handle dataset-based machine learning tasks,
    including data preprocessing, loss calculation, and batch generation for training.

    Args:
        X_train (Dict[str, bst.typing.ArrayLike]): Dictionary of training input data.
        y_train (Dict[str, bst.typing.ArrayLike]): Dictionary of training output data.
        X_test (Dict[str, bst.typing.ArrayLike]): Dictionary of testing input data.
        y_test (Dict[str, bst.typing.ArrayLike]): Dictionary of testing output data.
        standardize (bool, optional): Whether to standardize input data. Defaults to False.
        approximator (bst.nn.Module, optional): The neural network module to use. Defaults to None.
        loss_fn (str, optional): The loss function to use. Defaults to 'MSE'.
        loss_weights (Sequence[float], optional): Weights for different loss components. Defaults to None.

    Attributes:
        train_x (Dict[str, bst.typing.ArrayLike]): Processed training input data.
        train_y (Dict[str, bst.typing.ArrayLike]): Processed training output data.
        test_x (Dict[str, bst.typing.ArrayLike]): Processed testing input data.
        test_y (Dict[str, bst.typing.ArrayLike]): Processed testing output data.
        scaler_x (object): Scaler used for standardization, if applied.
    """

    def __init__(
        self,
        X_train: Dict[str, bst.typing.ArrayLike],
        y_train: Dict[str, bst.typing.ArrayLike],
        X_test: Dict[str, bst.typing.ArrayLike],
        y_test: Dict[str, bst.typing.ArrayLike],
        standardize: bool = False,
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
        self.scaler_x = None
        if standardize:
            r = jax.tree.map(
                lambda train, test: utils.standardize(train, test),
                self.train_x,
                self.test_x,
            )
            self.train_x = dict()
            self.test_x = dict()
            for key, val in r.items():
                self.train_x[key] = val[0]
                self.test_x[key] = val[1]

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Calculate the loss between the model outputs and the target values.

        Args:
            inputs: The input data (not used in this method).
            outputs: The model's output predictions.
            targets: The true target values.
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
            tuple: A tuple containing the batch of training inputs (self.train_x) and outputs (self.train_y).
        """
        return self.train_x, self.train_y

    def test(self):
        """
        Get the test dataset.

        Returns:
            tuple: A tuple containing the test inputs (self.test_x) and outputs (self.test_y).
        """
        return self.test_x, self.test_y
