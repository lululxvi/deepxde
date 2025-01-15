# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import brainunit as u
import jax

__all__ = [
    'accuracy',
    'l2_relative_error',
    'nanl2_relative_error',
    'mean_l2_relative_error',
    'mean_squared_error',
    'mean_absolute_percentage_error',
    'max_absolute_percentage_error',
    'absolute_percentage_error_std',
]


def _accuracy(y_true, y_pred):
    return u.math.mean(u.math.equal(u.math.argmax(y_pred, axis=-1),
                                    u.math.argmax(y_true, axis=-1)))


def accuracy(y_true, y_pred):
    """Computes accuracy across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(_accuracy, y_true, y_pred, is_leaf=u.math.is_quantity)


def _l2_relative_error(y_true, y_pred):
    return u.linalg.norm(y_true - y_pred) / u.linalg.norm(y_true)


def l2_relative_error(y_true, y_pred):
    """Computes L2 relative error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(_l2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _nanl2_relative_error(y_true, y_pred):
    err = y_true - y_pred
    err = u.math.nan_to_num(err)
    y_true = u.math.nan_to_num(y_true)
    return u.linalg.norm(err) / u.linalg.norm(y_true)


def nanl2_relative_error(y_true, y_pred):
    """Computes L2 relative error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(_nanl2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _mean_l2_relative_error(y_true, y_pred):
    return u.math.mean(
        u.linalg.norm(y_true - y_pred, axis=1) /
        u.linalg.norm(y_true, axis=1)
    )


def mean_l2_relative_error(y_true, y_pred):
    """Computes mean L2 relative error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(_mean_l2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _absolute_percentage_error(y_true, y_pred):
    return 100 * u.math.abs((y_true - y_pred) / u.math.abs(y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    """Computes mean absolute percentage error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(lambda x, y: _absolute_percentage_error(x, y).mean(),
                                  y_true,
                                  y_pred,
                                  is_leaf=u.math.is_quantity)


def max_absolute_percentage_error(y_true, y_pred):
    return jax.tree_util.tree_map(lambda x, y: _absolute_percentage_error(x, y).max(),
                                  y_true,
                                  y_pred,
                                  is_leaf=u.math.is_quantity)


def absolute_percentage_error_std(y_true, y_pred):
    """Computes standard deviation of absolute percentage error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(lambda x, y: _absolute_percentage_error(x, y).std(),
                                  y_true,
                                  y_pred,
                                  is_leaf=u.math.is_quantity)


def _mean_squared_error(y_true, y_pred):
    return u.math.mean(u.math.square(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """Computes mean squared error across nested structures of labels and predictions."""
    return jax.tree_util.tree_map(_mean_squared_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "mean l2 relative error": mean_l2_relative_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)
