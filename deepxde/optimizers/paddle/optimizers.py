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


def get(params, optimizer, learning_rate=None, decay=None, weight_decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        if weight_decay is not None:
            raise ValueError("L-BFGS optimizer doesn't support weight_decay")
        optim = paddle.optimizer.LBFGS(
            learning_rate=1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=("strong_wolfe" if LBFGS_options["maxls"] > 0 else None),
            parameters=params,
        )
        return optim

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        learning_rate = _get_lr_scheduler(learning_rate, decay)

    if optimizer == "adam":
        return paddle.optimizer.Adam(
            learning_rate=learning_rate, parameters=params, weight_decay=weight_decay
        )
    if optimizer == "sgd":
        return paddle.optimizer.SGD(
            learning_rate=learning_rate, parameters=params, weight_decay=weight_decay
        )
    if optimizer == "rmsprop":
        return paddle.optimizer.RMSProp(
            learning_rate=learning_rate,
            parameters=params,
            weight_decay=weight_decay,
        )
    if optimizer == "adamw":
        if (
            not isinstance(weight_decay, paddle.regularizer.L2Decay)
            or weight_decay._coeff == 0
        ):
            raise ValueError("AdamW optimizer requires non-zero L2 regularizer")
        return paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=params,
            weight_decay=weight_decay._coeff,
        )
    raise NotImplementedError(f"{optimizer} to be implemented for backend Paddle.")
