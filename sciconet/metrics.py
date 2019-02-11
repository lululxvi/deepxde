from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def get_metrics(names):
    metric_identifier = {
        'accuracy': accuracy,
        'l2 relative error': l2_relative_error,
        'MAPE': mean_absolute_percentage_error
    }

    metrics = []
    for name in names:
        if isinstance(name, str):
            metrics.append(metric_identifier[name])
        else:
            metrics.append(name)
    return metrics