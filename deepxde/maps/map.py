from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..utils import make_dict, timing


class Map(object):
    """Map base class."""

    def __init__(self):
        if not hasattr(self, "regularizer"):
            self.regularizer = None

        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.bool)
        self.data_id = tf.placeholder(tf.uint8)  # 0: train data, 1: test data

        self.build()

    @property
    def inputs(self):
        """Mapping inputs."""

    @property
    def outputs(self):
        """Mapping outputs."""

    @property
    def targets(self):
        """Targets of the mapping outputs."""

    def feed_dict(self, training, dropout, data_id, inputs, targets=None):
        """Construct a feed_dict to feed values to TensorFlow placeholders."""
        feed_dict = {
            self.training: training,
            self.dropout: dropout,
            self.data_id: data_id,
        }
        feed_dict.update(self._feed_dict_inputs(inputs))
        if targets is not None:
            feed_dict.update(self._feed_dict_targets(targets))
        return feed_dict

    def _feed_dict_inputs(self, inputs):
        return make_dict(self.inputs, inputs)

    def _feed_dict_targets(self, targets):
        return make_dict(self.targets, targets)

    @timing
    def build(self):
        """Construct the mapping."""
