from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mean_squared_error(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * tf.reduce_mean(tf.abs(y_true - y_pred) / y_true)


def softmax_cross_entropy(y_true, y_pred):
    return tf.losses.softmax_cross_entropy(y_true, y_pred)


def get(identifier):
    loss_identifier = {
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "mean absolute percentage error": mean_absolute_percentage_error,
        "MAPE": mean_absolute_percentage_error,
        "mape": mean_absolute_percentage_error,
        "softmax cross entropy": softmax_cross_entropy,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret loss function identifier:", identifier)
