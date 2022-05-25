import os
import random

import numpy as np

from . import backend as bkd
from .backend import backend_name, tf, torch, paddle
from .real import Real

hvd_dist = False
real = Real(32)
random_seed = None
if backend_name == "jax":
    iinfo = np.iinfo(int)
    jax_random_seed = random.randint(iinfo.min, iinfo.max)
xla_jit = None
if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    xla_jit = bkd.is_gpu_available()


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
    if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
        tf.keras.backend.set_floatx(value)
    # TODO: support jax.numpy.float64, which is not automatically enabled by default,
    # and will be truncated to jax.numpy.float32 for now.
    # - https://github.com/google/jax#current-gotchas
    # - https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision


def set_random_seed(seed):
    """Set the global random seeds of random, numpy, and backend.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)  # python random
    np.random.seed(seed)  # numpy
    if backend_name == "tensorflow.compat.v1":
        tf.set_random_seed(seed)  # tf CPU seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    elif backend_name == "tensorflow":
        tf.random.set_seed(seed)  # tf CPU seed
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    elif backend_name == "pytorch":
        torch.manual_seed(seed)
    elif backend_name == "jax":
        global jax_random_seed
        jax_random_seed = seed
    elif backend_name == "paddle":
        paddle.seed(seed)
    global random_seed
    random_seed = seed


def set_hvd():
    """Initialize horovod."""
    import horovod.tensorflow as hvd

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    global hvd_dist
    hvd_dist = True


def enable_xla_jit(mode=True):
    """Enables just-in-time compilation with XLA.

    - For backend TensorFlow 1.x, by default, compiles with XLA when running on GPU.
      XLA compilation can only be enabled when running on GPU.
    - For backend TensorFlow 2.x, by default, compiles with XLA when running on GPU.
    - Backend JAX always uses XLA.
    - Backends PyTorch and PaddlePaddle do not support XLA.

    Args:
        mode (bool): Whether to enable just-in-time compilation with XLA (``True``) or
            disable just-in-time compilation with XLA (``False``).
    """
    global xla_jit
    xla_jit = mode


def disable_xla_jit():
    """Disables just-in-time compilation with XLA.

    - For backend TensorFlow 1.x, by default, compiles with XLA when running on GPU.
      XLA compilation can only be enabled when running on GPU.
    - For backend TensorFlow 2.x, by default, compiles with XLA when running on GPU.
    - Backend JAX always uses XLA.
    - Backends PyTorch and PaddlePaddle do not support XLA.

    This is equivalent with ``enable_xla_jit(False)``.
    """
    enable_xla_jit(False)
