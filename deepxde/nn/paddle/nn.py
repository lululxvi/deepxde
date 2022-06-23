import paddle


class NN(paddle.nn.Layer):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self._input_transform = None
        self._output_transform = None

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        # TODO: support input transform
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        # TODO: support output transform
        self._output_transform = transform
