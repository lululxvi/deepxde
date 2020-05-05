from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .backend import tf


class Real(object):
    def __init__(self, precision):
        self.set_precision(precision)

    def set_precision(self, precision):
        self.precision = precision
        self.reals = {
            32: {np: np.float32, tf: tf.float32},
            64: {np: np.float64, tf: tf.float64},
        }[precision]

    def __call__(self, package):
        return self.reals[package]

    def set_float32(self):
        print("Set float to float32")
        self.set_precision(32)

    def set_float64(self):
        print("Set float to float64")
        self.set_precision(64)
