from flax import linen as nn

from .. import regularizers


class NN(nn.Module):
    """Base class for all neural network modules."""

    # All sub-modules should have the following variables:
    # regularization: Any = None
    # params: Any = None
    # _input_transform: Optional[Callable] = None
    # _output_transform: Optional[Callable] = None
    
    @property
    def regularizer(self):
        """Dynamically compute and return the regularizer function based on regularization."""
        return regularizers.get(self.regularization)
    
    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """

        def transform_handling_flat(x):
            """Handle inputs of shape (n,)"""
            # TODO: Support tuple or list inputs.
            if isinstance(x, (list, tuple)):
                return transform(x)
            if x.ndim == 1:
                return transform(x.reshape(1, -1)).reshape(-1)
            return transform(x)

        self._input_transform = transform_handling_flat

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """

        def transform_handling_flat(inputs, outputs):
            """Handle inputs of shape (n,)"""
            # TODO: Support tuple or list inputs.
            if isinstance(inputs, (list, tuple)):
                return transform(inputs, outputs)
            if inputs.ndim == 1:
                return transform(inputs.reshape(1, -1), outputs.reshape(1, -1)).reshape(-1)
            return transform(inputs, outputs)

        self._output_transform = transform_handling_flat
