__all__ = ["get", "is_external_optimizer"]

import paddle

from ..config import LBFGS_options


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

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = paddle.optimizer.LBFGS(
            learning_rate=1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=None,
            parameters=params,
        )
        return optim

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        learning_rate = _get_lr_scheduler(learning_rate, decay)

    if optimizer == "adam":
        return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    raise NotImplementedError(f"{optimizer} to be implemented for backend Paddle.")
