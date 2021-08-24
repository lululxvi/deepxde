__all__ = ["get", "is_external_optimizer"]

import torch

from ..config import LBFGS_options


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return torch.optim.LBFGS(
            params,
            lr=1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=None,
        )

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        raise NotImplementedError(
            "learning rate decay to be implemented for backend pytorch."
        )
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    raise NotImplementedError(f"{optimizer} to be implemented for backend pytorch.")
