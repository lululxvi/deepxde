__all__ = ["get", "is_external_optimizer"]

from .tfp_optimizer import lbfgs_minimize
from ...backend import tf


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(optimizer, learning_rate=None, decay=None):
    """Retrieves a Keras Optimizer instance."""
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer
    if is_external_optimizer(optimizer):
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return lbfgs_minimize

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    lr_schedule = _get_learningrate(learning_rate, decay)
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if optimizer == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    raise NotImplementedError(f"{optimizer} to be implemented for backend tensorflow.")


def _get_learningrate(lr, decay):
    if decay is None:
        return lr

    if decay[0] == "inverse time":
        return tf.keras.optimizers.schedules.InverseTimeDecay(lr, decay[1], decay[2])
    if decay[0] == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(lr, decay[1], alpha=decay[2])

    raise NotImplementedError(
        f"{decay[0]} learning rate decay to be implemented for backend tensorflow."
    )
