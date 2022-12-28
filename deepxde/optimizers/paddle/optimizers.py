__all__ = ["get", "is_external_optimizer"]

import paddle


def _get_lr_scheduler(lr, decay):
    if decay[0] == "inverse time":
        lr_sch = paddle.optimizer.lr.InverseTimeDecay(lr, decay[1], verbose=False)
    else:
        raise NotImplementedError(
            f"{decay[0]} decay is not implemented in PaddlePaddle"
        )
    return lr_sch


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if is_external_optimizer(optimizer):
        # TODO: add support for L-BFGS and L-BFGS-B
        raise NotImplementedError(f"{optimizer} is not implemented in PaddlePaddle")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        learning_rate = _get_lr_scheduler(learning_rate, decay)

    if optimizer == "adam":
        return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    raise NotImplementedError(f"{optimizer} is not implemented in PaddlePaddle")
