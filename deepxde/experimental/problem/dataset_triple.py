from typing import Sequence

import brainstate as bst

from deepxde.data.sampler import BatchSampler
from .base import Problem

__all__ = [
    'TripleDataset',
    'TripleCartesianProd'
]


class TripleDataset(Problem):
    """
    Dataset with each data point as a triple.

    The couple of the first two elements are the input, and the third element is the
    output. This dataset can be used with the network ``DeepONet`` for operator
    learning.

    Args:
        X_train (tuple): A tuple of two NumPy arrays representing the input training data.
        y_train (numpy.ndarray): A NumPy array representing the output training data.
        X_test (tuple): A tuple of two NumPy arrays representing the input testing data.
        y_test (numpy.ndarray): A NumPy array representing the output testing data.
        approximator (bst.nn.Module, optional): The neural network module used for approximation. Defaults to None.
        loss_fn (str, optional): The loss function to be used. Defaults to 'MSE'.
        loss_weights (Sequence[float], optional): Weights for the loss function. Defaults to None.

    References:
        `L. Lu, P. Jin, G. Pang, Z. Zhang, & G. E. Karniadakis. Learning nonlinear
        operators via DeepONet based on the universal approximation theorem of
        operators. Nature Machine Intelligence, 3, 218--229, 2021
        <https://doi.org/10.1038/s42256-021-00302-5>`_.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Compute the loss between the model outputs and the targets.

        Args:
            inputs: The input data (not used in this method).
            outputs: The model outputs.
            targets: The target values.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        """
        Get the next batch of training data.

        Args:
            batch_size (int, optional): The size of the batch to return. If None, returns all training data.

        Returns:
            tuple: A tuple containing two elements:
                - A tuple of two arrays representing the input training data for the batch.
                - An array representing the output training data for the batch.
        """
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices],
             self.train_x[1][indices]),
            self.train_y[indices],
        )

    def test(self):
        """
        Get the testing data.

        Returns:
            tuple: A tuple containing two elements:
                - The input testing data.
                - The output testing data.
        """
        return self.test_x, self.test_y


class TripleCartesianProd(Problem):
    """
    Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``TripleDataset`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        """
        Initialize the TripleCartesianProd dataset.

        Args:
            X_train (tuple): A tuple of two NumPy arrays for training input data.
            y_train (numpy.ndarray): A NumPy array for training output data.
            X_test (tuple): A tuple of two NumPy arrays for testing input data.
            y_test (numpy.ndarray): A NumPy array for testing output data.
            approximator (bst.nn.Module, optional): The neural network module used for approximation. Defaults to None.
            loss_fn (str, optional): The loss function to be used. Defaults to 'MSE'.
            loss_weights (Sequence[float], optional): Weights for the loss function. Defaults to None.

        Raises:
            ValueError: If the training or testing dataset does not have the format of Cartesian product.
        """
        super().__init__(
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )

        if len(X_train[0]) != y_train.shape[0] or len(X_train[1]) != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) != y_test.shape[0] or len(X_test[1]) != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Compute the loss between the model outputs and the targets.

        Args:
            inputs: The input data (not used in this method).
            outputs: The model outputs.
            targets: The target values.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        return self.loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        """
        Get the next batch of training data.

        Args:
            batch_size (int, tuple, or list, optional): The size of the batch to return. 
                If None, returns all training data. 
                If int, returns a batch with the specified size for branch data and all trunk data.
                If tuple or list, returns a batch with specified sizes for both branch and trunk data.

        Returns:
            tuple: A tuple containing two elements:
                - A tuple of two arrays representing the input training data for the batch.
                - An array representing the output training data for the batch.
        """
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            (self.train_x[0][indices_branch],
             self.train_x[1][indices_trunk],),
            self.train_y[indices_branch, indices_trunk]
        )

    def test(self):
        """
        Get the testing data.

        Returns:
            tuple: A tuple containing two elements:
                - The input testing data.
                - The output testing data.
        """
        return self.test_x, self.test_y
