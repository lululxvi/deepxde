from typing import Optional, Callable

import brainstate as bst
import jax.tree


class NN(bst.nn.Module):
    """Base class for all neural network modules."""

    def __init__(
        self,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        """
        Initialize the NN class.

        Parameters:
        -----------
        input_transform : Optional[Callable], default=None
            A callable that transforms the input before it's passed to the network.
        output_transform : Optional[Callable], default=None
            A callable that transforms the output after it's produced by the network.

        Returns:
        --------
        None
        """
        super().__init__()
        self.regularization = None
        self._input_transform = input_transform
        self._output_transform = output_transform

    def apply_feature_transform(self, transform):
        """
        Compute the features by applying a transform to the network inputs.

        This method sets the input transform function, which is applied before
        the input is passed to the network, i.e., ``features = transform(inputs)``.
        Then, ``outputs = network(features)``.

        Parameters:
        -----------
        transform : Callable
            The transform function to be applied to the inputs.

        Returns:
        --------
        None
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """
        Apply a transform to the network outputs.

        This method sets the output transform function, which is applied after
        the network produces its output, i.e., ``outputs = transform(inputs, outputs)``.

        Parameters:
        -----------
        transform : Callable
            The transform function to be applied to the outputs.

        Returns:
        --------
        None
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """
        Evaluate the number of trainable parameters for the NN.

        This method calculates the total number of trainable parameters in the neural network
        by iterating through all parameters in the network's state.

        Parameters:
        -----------
        None

        Returns:
        --------
        int
            The total number of trainable parameters in the neural network.
        """
        n_param = 0
        for key, val in self.states(bst.ParamState).items():
            n_param += [v.size for v in jax.tree_leaves(val)]
        return n_param
