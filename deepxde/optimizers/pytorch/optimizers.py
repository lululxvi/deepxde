__all__ = ["get", "is_external_optimizer"]

import torch

from ..config import LBFGS_options


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    # Custom Optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optim = optimizer
    elif optimizer in ["L-BFGS", "L-BFGS-B"]:
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = torch.optim.LBFGS(
            params,
            lr=1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=("strong_wolfe" if LBFGS_options["maxls"] > 0 else None),
        )
    else:
        if learning_rate is None:
            raise ValueError("No learning rate for {}.".format(optimizer))
        if optimizer == "sgd":
            optim = torch.optim.SGD(params, lr=learning_rate)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(params, lr=learning_rate)
        elif optimizer == "adam":
            optim = torch.optim.Adam(params, lr=learning_rate)
        elif optimizer == "adamw":
            optim = torch.optim.AdamW(params, lr=learning_rate)
        else:
            raise NotImplementedError(
                f"{optimizer} to be implemented for backend pytorch."
            )
    lr_scheduler = _get_learningrate_scheduler(optim, decay)
    return optim, lr_scheduler


def _get_learningrate_scheduler(optim, decay):
    if decay is None:
        return None

    if decay[0] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=decay[1], gamma=decay[2]
        )
    elif decay[0] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, decay[1], eta_min=decay[2]
        )
    elif decay[0] == "inverse time":
        return torch.optim.lr_scheduler.LambdaLR(
            optim, lambda step: 1 / (1 + decay[2] * (step / decay[1]))
        )
    elif decay[0] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optim, decay[1])
    elif decay[0] == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(optim, decay[1])

    # TODO: More learning rate scheduler
    raise NotImplementedError(
        f"{decay[0]} learning rate scheduler to be implemented for backend pytorch."
    )
