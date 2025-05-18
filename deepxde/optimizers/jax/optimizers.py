__all__ = ["get", "is_external_optimizer", "apply_updates"]

import jax
import optax


apply_updates = optax.apply_updates


def is_external_optimizer(optimizer):
    # TODO: add external optimizers
    return False


def get(optimizer, learning_rate=None, decay=None):
    """Retrieves an optax Optimizer instance."""
    if isinstance(optimizer, optax._src.base.GradientTransformation):
        return optimizer
    if is_external_optimizer(optimizer):
        raise NotImplementedError(f"{optimizer} to be implemented for backend jax.")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    lr_schedule = _get_learningrate(learning_rate, decay)
    if optimizer == "adam":
        return optax.adam(lr_schedule)
    if optimizer == "rmsprop":
        return optax.rmsprop(lr_schedule)
    if optimizer == "sgd":
        return optax.sgd(lr_schedule)

    raise NotImplementedError(f"{optimizer} to be implemented for backend jax.")


def _get_learningrate(lr, decay):
    if decay is None:
        return lr
    if decay[0] == "linear":
        return optax.linear_schedule(lr, decay[1], decay[2])
    if decay[0] == "cosine":
        return optax.cosine_decay_schedule(lr, decay[1], decay[2])
    if decay[0] == "exponential":
        return optax.exponential_decay(lr, decay[1], decay[2])

    raise NotImplementedError(
        f"{decay[0]} learning rate decay to be implemented for backend jax."
    )
