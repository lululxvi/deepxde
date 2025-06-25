import braintools
import brainunit as u
import jax


def mean_absolute_error(y_true, y_pred):
    return jax.tree.map(
        lambda x, y: braintools.metric.absolute_error(x, y).mean(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity,
    )


def mean_squared_error(y_true, y_pred):
    return jax.tree.map(
        lambda x, y: braintools.metric.squared_error(x, y).mean(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity,
    )


def mean_l2_relative_error(y_true, y_pred):
    return jax.tree.map(
        lambda x, y: braintools.metric.l2_norm(x, y).mean(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity,
    )


def softmax_cross_entropy(y_true, y_pred):
    return jax.tree.map(
        lambda x, y: braintools.metric.softmax_cross_entropy(x, y).mean(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity,
    )


LOSS_DICT = {
    # mean absolute error
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    # mean squared error
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    # mean l2 relative error
    "mean l2 relative error": mean_l2_relative_error,
    # softmax cross entropy
    "softmax cross entropy": softmax_cross_entropy,
}


def get_loss(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get_loss, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
