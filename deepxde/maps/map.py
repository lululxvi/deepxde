from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..utils import timing


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
        """Return the mapping inputs."""

    @property
    def outputs(self):
        """Return the mapping outputs."""

    @property
    def targets(self):
        """Return the targets of the mapping outputs."""

    @timing
    def build(self):
        """Construct the mapping."""
