import torch


__all__ = ["get", "is_external_optimizer"]


def is_external_optimizer(optimizer):
    return False


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if learning_rate is None and optimizer not in ["L-BFGS", "L-BFGS-B"]:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        raise NotImplementedError(
            "learning rate decay to be implemented for backend pytorch."
        )
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        # TODO: L-BFGS parameters
        return torch.optim.LBFGS(params, lr=1, max_iter=20)
    raise NotImplementedError(f"{optimizer} to be implemented for backend pytorch.")
