from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Real(object):
    def __init__(self, precision):
        self.precision = precision
        self.reals = {
            32: {np: np.float32, tf: tf.float32},
            64: {np: np.float64, tf: tf.float64},
        }[precision]

    def __call__(self, package):
        return self.reals[package]
