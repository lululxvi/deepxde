from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
