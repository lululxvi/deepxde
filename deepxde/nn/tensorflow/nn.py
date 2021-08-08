from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ...backend import tf


class NN(tf.keras.Model):
    """Base class for all neural network modules."""

    def __init__(self):
        super(NN, self).__init__()
        self._inputs = None
        self._targets = None
        self._auxiliary_vars = None

        self.regularizer = None

        self._input_transform = None
        self._output_transform = None

    @property
    def inputs(self):
        """Return the net inputs (Tensors)."""
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def targets(self):
        """Return the targets of the net outputs (Tensors)."""
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

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
