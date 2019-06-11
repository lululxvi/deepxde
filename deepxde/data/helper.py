from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def zero_function(dim_outputs):
    def zero(X):
        return np.zeros((len(X), dim_outputs))

    return zero


def one_function(dim_outputs):
    def one(X, *args):
        return np.ones((len(X), dim_outputs))

    return one
