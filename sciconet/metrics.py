from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)


def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "MAPE": mean_absolute_percentage_error,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret metric function identifier:", identifier)
