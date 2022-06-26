import numpy as np

from ... import config
from ...backend import tf
from ...utils import make_dict, timing


class NN:
    """Base class for all neural network modules."""

    def __init__(self):
        self.training = tf.placeholder(tf.bool)
        self.regularizer = None

        self._auxiliary_vars = tf.placeholder(config.real(tf), [None, None])
        self._input_transform = None
        self._output_transform = None
        self._built = False  # The property will be set upon call of self.build()

    @property
    def inputs(self):
        """Return the net inputs (placeholders)."""

    @property
    def outputs(self):
        """Return the net outputs (tf.Tensor)."""

    @property
    def targets(self):
        """Return the targets of the net outputs (placeholders)."""

    @property
    def auxiliary_vars(self):
        """Return additional variables needed (placeholders)."""
        return self._auxiliary_vars

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    def feed_dict(self, training, inputs, targets=None, auxiliary_vars=None):
        """Construct a feed_dict to feed values to TensorFlow placeholders."""
        feed_dict = {self.training: training}
        feed_dict.update(self._feed_dict_inputs(inputs))
        if targets is not None:
            feed_dict.update(self._feed_dict_targets(targets))
        if auxiliary_vars is not None:
            feed_dict.update(self._feed_dict_auxiliary_vars(auxiliary_vars))
        return feed_dict

    def _feed_dict_inputs(self, inputs):
        return make_dict(self.inputs, inputs)

    def _feed_dict_targets(self, targets):
        return make_dict(self.targets, targets)

    def _feed_dict_auxiliary_vars(self, auxiliary_vars):
        return make_dict(self.auxiliary_vars, auxiliary_vars)

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
        """Evaluate the number of trainable parameters for the NN.

        Notice that the function returns the number of trainable parameters for the
        whole tf.Session, so that it will not be correct if several nets are defined
        within the same tf.Session.
        """
        return np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
        )

    @timing
    def build(self):
        """Construct the network."""
        self.built = True
