# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

from __future__ import annotations

import abc
from typing import Callable, Sequence, Any, Tuple

import brainstate as bst
import jax

from deepxde.pinnx.utils.losses import get_loss

Inputs = Any
Targets = Any
Auxiliary = Any
Outputs = Any
LOSS = jax.typing.ArrayLike

__all__ = [
    'Problem',
]


class Problem(abc.ABC):
    """
    Base Problem Class.

    A problem is defined by the approximator and the loss function.

    Attributes:
        approximator: The approximator.
        loss_fn: The loss function.
        loss_weights: A list specifying scalar coefficients (Python floats) to
            weight the loss contributions. The loss value that will be minimized by
            the trainer will then be the weighted sum of all individual losses,
            weighted by the `loss_weights` coefficients.
    """

    approximator: bst.nn.Module
    loss_fn: Callable | Sequence[Callable]

    def __init__(
        self,
        approximator: bst.nn.Module = None,
        loss_fn: str | Callable[[Inputs, Outputs], LOSS] = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        """
        Initialize the problem.

        Args:
            approximator (bst.nn.Module, optional): The approximator. Defaults to None.
            loss_fn (str | Callable[[Inputs, Outputs], LOSS], optional): The loss function. 
                If the same loss is used for all errors, then `loss` is a String name of a loss function 
                or a loss function. If different errors use different losses, then `loss` is a list
                whose size is equal to the number of errors. Defaults to 'MSE'.
            loss_weights (Sequence[float], optional): A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the trainer will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients. Defaults to None.
        """
        # Implementation details...

    def define_approximator(
        self,
        approximator: bst.nn.Module,
    ) -> Problem:
        """
        Define the approximator for the problem.

        Args:
            approximator (bst.nn.Module): The approximator to be used in the problem.

        Returns:
            Problem: The current Problem instance with the defined approximator.

        Raises:
            AssertionError: If the approximator is not an instance of bst.nn.Module.
        """
        # Implementation details...

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Calculate and return a list of losses (constraints) for the problem.

        Args:
            inputs: The input data.
            outputs: The output data.
            targets: The target data.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of calculated losses.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Problem.losses is not implemented.")

    def losses_train(self, inputs, outputs, targets, **kwargs):
        """
        Calculate and return a list of losses for the training dataset.

        This method sets the environment context to training mode before calculating losses.

        Args:
            inputs: The input data for training.
            outputs: The output data for training.
            targets: The target data for training.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of calculated losses for the training dataset.
        """
        with bst.environ.context(fit=True):
            return self.losses(inputs, outputs, targets, **kwargs)

    def losses_test(self, inputs, outputs, targets, **kwargs):
        """
        Calculate and return a list of losses for the test dataset.

        This method sets the environment context to testing mode before calculating losses.

        Args:
            inputs: The input data for testing.
            outputs: The output data for testing.
            targets: The target data for testing.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of calculated losses for the test dataset.
        """
        with bst.environ.context(fit=False):
            return self.losses(inputs, outputs, targets, **kwargs)

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None) -> Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]:
        """
        Generate and return the next batch of training data.

        This method should be implemented by subclasses to provide the next batch of training data.

        Args:
            batch_size (int, optional): The size of the batch to be returned. Defaults to None.

        Returns:
            Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]: A tuple containing the inputs and targets
            for the next training batch. May also include auxiliary data if applicable.
        """

    @abc.abstractmethod
    def test(self) -> Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]:
        """
        Generate and return the test dataset.

        This method should be implemented by subclasses to provide the test dataset.

        Returns:
            Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]: A tuple containing the inputs and targets
            for the test dataset. May also include auxiliary data if applicable.
        """
