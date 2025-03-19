
"""Internal utilities."""
from functools import wraps
from typing import Callable, Union

import brainstate as bst
import brainunit as u
import numpy as np


def check_not_none(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            is_none = []
            for a in attr:
                if not hasattr(self, a):
                    raise ValueError(f"{a} must be an attribute of the class.")
                is_none.append(getattr(self, a) is None)
            if any(is_none):
                raise ValueError(f"{attr} must not be None.")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def return_tensor(func):
    """Convert the output to a Tensor."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return u.math.asarray(func(*args, **kwargs), dtype=bst.environ.dftype())

    return wrapper


def tree_repr(tree, precision: int = 2):
    with np.printoptions(precision=precision, suppress=True, threshold=5):
        return repr(tree)
        # return repr(jax.tree.map(lambda x: repr(x), tree, is_leaf=u.math.is_quantity))


def get_activation(activation: Union[str, Callable]):
    """Get the activation function."""
    if isinstance(activation, str):
        return getattr(bst.functional, activation)
    else:
        return activation
