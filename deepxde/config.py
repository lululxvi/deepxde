import os
import random

import numpy as np

from .backend import tf, torch, backend_name
from .real import Real


real = Real(32)


def default_float():
    """Returns the default float type, as a string."""
    if real.precision == 64:
        return "float64"
    return "float32"


def set_default_float(value):
    """Sets the default float type.

    The default floating point type is 'float32'.

    Args:
        value (String): 'float32' or 'float64'.
    """
    if value == "float32":
        print("Set the default float type to float32")
        real.set_float32()
    elif value == "float64":
        print("Set the default float type to float64")
        real.set_float64()
    tf.keras.backend.set_floatx(value)


def set_random_seed(seed):
    """Set the global random seed.

    For reproductibility purposes, one has to set the random, numpy and backend seeds.
    """
    random.seed(seed)  #  by python Set random seeds
    np.random.seed(seed)  #  by numpy Set random seeds

    if backend_name == "pytorch":
        torch.manual_seed(seed)
    elif backend_name == "tensorflow":
        tf.random.set_seed(seed)  # tf cpu fix seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    elif backend_name == "tensorflow.compat.v1":
        tf.set_random_seed(seed)  # tf cpu fix seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
