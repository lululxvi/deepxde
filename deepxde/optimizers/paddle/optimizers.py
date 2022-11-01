__all__ = ["get", "is_external_optimizer"]

import paddle
from .lbfgs_optimizer import lbfgs_minimize

def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if not paddle.in_dynamic_mode():
            raise ValueError("L-BFGS can not used for backend Paddle in static mode.")
        else:
            if learning_rate is not None or decay is not None:
                print("Warning: learning rate is ignored for {}".format(optimizer))
            return lbfgs_minimize

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        if decay[0] == 'inverse time':
            scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=learning_rate, gamma=decay[2], verbose=False)
        else:
            raise NotImplementedError(
                f"{decay[0]} to be implemented for backend Paddle."
            )
    
    if optimizer == "adam":
        if decay is not None and decay[0] == 'inverse time':
            return paddle.optimizer.Adam(learning_rate=scheduler, parameters=params), scheduler
        else:
            return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    
    raise NotImplementedError(f"{optimizer} to be implemented for backend Paddle.")
