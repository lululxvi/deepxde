__all__ = ["get", "is_external_optimizer"]

import jax


def is_external_optimizer(optimizer):
    # TODO: add external optimizers
    return False


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    # TODO: add optimizers for jax
    raise NotImplementedError(f"{optimizer} to be implemented for backend jax.")
