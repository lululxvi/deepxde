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
    """
    Computes accuracy across nested structures of labels and predictions.

    This function calculates the accuracy by comparing the predicted labels
    with the true labels. It can handle nested structures of data.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true labels or ground truth values. Can be a single array or a
        nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted labels or values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed accuracy. If the input is a nested structure, the output
        will have the same structure with accuracy values for each leaf node.
    """
    return jax.tree_util.tree_map(_accuracy, y_true, y_pred, is_leaf=u.math.is_quantity)


def _l2_relative_error(y_true, y_pred):
    return u.linalg.norm(y_true - y_pred) / u.linalg.norm(y_true)


def l2_relative_error(y_true, y_pred):
    """
    Computes L2 relative error across nested structures of labels and predictions.

    This function calculates the L2 relative error between true values and predicted values.
    It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed L2 relative error. If the input is a nested structure, the output
        will have the same structure with L2 relative error values for each leaf node.
    """
    return jax.tree_util.tree_map(_l2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _nanl2_relative_error(y_true, y_pred):
    err = y_true - y_pred
    err = u.math.nan_to_num(err)
    y_true = u.math.nan_to_num(y_true)
    return u.linalg.norm(err) / u.linalg.norm(y_true)


def nanl2_relative_error(y_true, y_pred):
    """
    Computes L2 relative error across nested structures of labels and predictions,
    handling NaN values.

    This function calculates the L2 relative error between true values and predicted values,
    treating NaN values as zeros. It can handle nested structures of data by applying
    the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
        May contain NaN values.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.
        May contain NaN values.

    Returns:
    --------
    float or nested structure
        The computed L2 relative error with NaN handling. If the input is a nested structure,
        the output will have the same structure with L2 relative error values for each leaf node.
    """
    return jax.tree_util.tree_map(_nanl2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _mean_l2_relative_error(y_true, y_pred):
    return u.math.mean(
        u.linalg.norm(y_true - y_pred, axis=1) /
        u.linalg.norm(y_true, axis=1)
    )


def mean_l2_relative_error(y_true, y_pred):
    """
    Computes mean L2 relative error across nested structures of labels and predictions.

    This function calculates the mean L2 relative error between true values and predicted values.
    It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed mean L2 relative error. If the input is a nested structure, the output
        will have the same structure with mean L2 relative error values for each leaf node.
    """
    return jax.tree_util.tree_map(_mean_l2_relative_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def _absolute_percentage_error(y_true, y_pred):
    return 100 * u.math.abs((y_true - y_pred) / u.math.abs(y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes mean absolute percentage error across nested structures of labels and predictions.

    This function calculates the mean absolute percentage error between true values and predicted values.
    It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed mean absolute percentage error. If the input is a nested structure, the output
        will have the same structure with mean absolute percentage error values for each leaf node.
    """
    return jax.tree_util.tree_map(
        lambda x, y: _absolute_percentage_error(x, y).mean(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity
    )


def max_absolute_percentage_error(y_true, y_pred):
    """
    Computes maximum absolute percentage error across nested structures of labels and predictions.

    This function calculates the maximum absolute percentage error between true values and predicted values.
    It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed maximum absolute percentage error. If the input is a nested structure, the output
        will have the same structure with maximum absolute percentage error values for each leaf node.
    """
    return jax.tree_util.tree_map(
        lambda x, y: _absolute_percentage_error(x, y).max(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity
    )


def absolute_percentage_error_std(y_true, y_pred):
    """
    Computes standard deviation of absolute percentage error across nested structures of labels and predictions.

    This function calculates the standard deviation of the absolute percentage error between true values
    and predicted values. It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed standard deviation of absolute percentage error. If the input is a nested structure,
        the output will have the same structure with standard deviation values for each leaf node.
    """
    return jax.tree_util.tree_map(
        lambda x, y: _absolute_percentage_error(x, y).std(),
        y_true,
        y_pred,
        is_leaf=u.math.is_quantity
    )


def _mean_squared_error(y_true, y_pred):
    return u.math.mean(u.math.square(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Computes mean squared error across nested structures of labels and predictions.

    This function calculates the mean squared error between true values and predicted values.
    It can handle nested structures of data by applying the calculation to each leaf node.

    Parameters:
    -----------
    y_true : array_like or nested structure
        The true values or ground truth. Can be a single array or a nested structure of arrays.
    y_pred : array_like or nested structure
        The predicted values. Should have the same structure as y_true.

    Returns:
    --------
    float or nested structure
        The computed mean squared error. If the input is a nested structure, the output
        will have the same structure with mean squared error values for each leaf node.
    """
    return jax.tree_util.tree_map(_mean_squared_error, y_true, y_pred, is_leaf=u.math.is_quantity)


def get(identifier):
    """
    Retrieves a metric function based on the provided identifier.

    This function maps string identifiers to their corresponding metric functions
    or returns the function if a callable is provided directly.

    Parameters:
    -----------
    identifier : str or callable
        A string identifier for a predefined metric function or a callable metric function.
        Accepted string identifiers include:
        - "accuracy"
        - "l2 relative error"
        - "nanl2 relative error"
        - "mean l2 relative error"
        - "mean squared error" (also "MSE" or "mse")
        - "MAPE"
        - "max APE"
        - "APE SD"

    Returns:
    --------
    callable
        The metric function corresponding to the provided identifier.

    Raises:
    -------
    ValueError
        If the provided identifier is neither a recognized string nor a callable.
    """
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
