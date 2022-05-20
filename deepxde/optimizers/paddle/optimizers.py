__all__ = ["get", "is_external_optimizer"]

import paddle


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        # TODO: add support for L-BFGS and L-BFGS-B
        raise ValueError("L-BFGS to be implemented for backend Paddle.")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        raise NotImplementedError(
            "learning rate decay to be implemented for backend Paddle."
        )
    if optimizer == "adam":
        return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    raise NotImplementedError(f"{optimizer} to be implemented for backend Paddle.")
