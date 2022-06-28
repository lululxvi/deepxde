import torch


class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._input_transform = None
        self._output_transform = None

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

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)
