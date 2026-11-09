__all__ = ["get", "is_external_optimizer", "apply_updates"]

import jax
import optax
from optax import schedules


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
        return schedules.linear_schedule(lr, *decay[1:])
    if decay[0] == "cosine":
        return schedules.cosine_decay_schedule(lr, *decay[1:])
    if decay[0] == "exponential":
        return schedules.exponential_decay(lr, *decay[1:])
    if decay[0] == "warmup_cosine":
        return schedules.warmup_cosine_decay_schedule(lr, *decay[1:])
    if decay[0] == "warmup_exponential":
        return schedules.warmup_exponential_decay_schedule(lr, *decay[1:])

    raise NotImplementedError(
        f"Unknown decay schedule '{decay[0]}' for JAX backend. "
        f"Supported: 'linear', 'cosine', 'exponential', 'warmup_cosine', 'warmup_exponential'."
    )
