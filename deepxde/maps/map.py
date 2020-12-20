from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import config
from ..backend import tf
from ..utils import make_dict, timing


class Map(object):
    """Map base class."""

    def __init__(self):
        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.bool)
        self.data_id = tf.placeholder(tf.uint8)  # 0: train data, 1: test data
        self._auxiliary_vars = tf.placeholder(config.real(tf))

        self.regularizer = None

        # The property will be set upon call of self.build()
        self._built = False

        self._input_transform = None
        self._output_transform = None

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    @property
    def inputs(self):
        """Mapping inputs."""

    @property
    def outputs(self):
        """Mapping outputs."""

    @property
    def targets(self):
        """Targets of the mapping outputs."""

    @property
    def auxiliary_vars(self):
        """Any additional variables needed."""
        return self._auxiliary_vars

    def feed_dict(
        self, training, dropout, data_id, inputs, targets=None, auxiliary_vars=None
    ):
        """Construct a feed_dict to feed values to TensorFlow placeholders."""
        feed_dict = {
            self.training: training,
            self.dropout: dropout,
            self.data_id: data_id,
        }
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

    @timing
    def build(self):
        """Construct the mapping."""
        self.built = True
