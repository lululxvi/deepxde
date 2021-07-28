import torch


__all__ = ["get", "is_external_optimizer"]


def is_external_optimizer(optimizer):
    return False


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    # TODO: decay
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    raise NotImplementedError(f"{optimizer} to be implemented for backend pytorch.")
