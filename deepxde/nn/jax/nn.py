from flax import linen as nn
from typing import Any


class NN(nn.Module):
    """Base class for all neural network modules."""

    training: Any = True
    regularizer: Any = None
    inputs: Any = None
    targets: Any = None
    auxiliary_vars: Any = None
    params: Any = None
    _input_transform: Any = None
    _output_transform: Any = None

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform
