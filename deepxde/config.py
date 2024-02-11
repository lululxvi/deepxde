import os
import random
import sys

import numpy as np

from . import backend as bkd
from .backend import backend_name, tf, torch, paddle
from .real import Real

# Data parallel
parallel_scaling = None
# Data parallel via Horovod
hvd = None
comm = None
world_size = 1
rank = 0
if "OMPI_COMM_WORLD_SIZE" in os.environ:
    if backend_name == "tensorflow.compat.v1":
        import horovod.tensorflow as hvd

        hvd.init()
        world_size = hvd.size()
        if world_size > 1:
            from mpi4py import MPI

            parallel_scaling = "weak"
            comm = MPI.COMM_WORLD
            tf.compat.v1.disable_eager_execution()  # Without this line, Horovod broadcasting fails.
            rank = hvd.rank()  # Only single node acceleration supported so far.
            if rank == 0:
                print(f"\nParallel training with {world_size} processes.\n")
        else:
            hvd = None
    else:
        raise NotImplementedError(
            "Parallel training via Horovod is only implemented in backend tensorflow.compat.v1"
        )


# Default float type
real = Real(32)
# Using mixed precision
mixed = False
# Random seed
random_seed = None
if backend_name == "jax":
    iinfo = np.iinfo(int)
    jax_random_seed = random.randint(iinfo.min, iinfo.max)
# XLA
xla_jit = False
if backend_name in ["tensorflow.compat.v1", "tensorflow"] and hvd is None:
    # Note: Horovod with tensorflow.compat.v1 does not support XLA.
    xla_jit = bkd.is_gpu_available()
elif backend_name == "jax":
    xla_jit = True
if xla_jit:
    print("Enable just-in-time compilation with XLA.\n", file=sys.stderr, flush=True)
# Automatic differentiation
autodiff = "reverse"


def default_float():
    """Returns the default float type, as a string."""
    if real.precision == 64:
        return "float64"
    elif real.precision == 32:
        return "float32"
    elif real.precision == 16:
        return "float16"


def set_default_float(value):
    """Sets the default float type.

    The default floating point type is 'float32'. Mixed precision uses the method in the paper:
    `J. Hayford, J. Goldman-Wetzler, E. Wang, & L. Lu. Speeding up and reducing memory usage for scientific machine learning via mixed precision.
    Computer Methods in Applied Mechanics and Engineering, 428, 117093, 2024 <https://doi.org/10.1016/j.cma.2024.117093>`_.

    Args:
        value (String): 'float16', 'float32', 'float64', or 'mixed' (mixed precision).
    """
    global mixed
    if value == "float16":
        print("Set the default float type to float16")
        real.set_float16()
    elif value == "float32":
        print("Set the default float type to float32")
        real.set_float32()
    elif value == "float64":
        print("Set the default float type to float64")
        real.set_float64()
    elif value == "mixed":
        print("Set the float type to mixed precision of float16 and float32")
        mixed = True
        if backend_name == "tensorflow":
            real.set_float16()
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            return # don't try to set it again below
        if backend_name == "pytorch":
            # Use float16 during the forward and backward passes, but store in float32
            real.set_float32()
        else:
            raise ValueError(
                f"{backend_name} backend does not currently support mixed precision."
            )
    else:
        raise ValueError(f"{value} not supported in deepXDE")
    if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
        tf.keras.backend.set_floatx(value)
    elif backend_name == "pytorch":
        torch.set_default_dtype(real(torch))
    elif backend_name == "paddle":
        paddle.set_default_dtype(value)
    # TODO: support jax.numpy.float64, which is not automatically enabled by default,
    # and will be truncated to jax.numpy.float32 for now.
    # - https://github.com/google/jax#current-gotchas
    # - https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision


def set_random_seed(seed):
    """Sets all random seeds for the program (Python random, NumPy, and backend), and
    configures the program to run deterministically.

    You can use this to make the program fully deterministic. This means that if the
    program is run multiple times with the same inputs on the same hardware, it will
    have the exact same outputs each time. This is useful for debugging models, and for
    obtaining fully reproducible results.

    - For backend TensorFlow 2.x: Results might change if you run the model several
      times in the same terminal.

    Warning:
        Note that determinism in general comes at the expense of lower performance and
        so your model may run slower when determinism is enabled.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    if backend_name == "tensorflow.compat.v1":
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        # Based on https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_19.06.html,
        # if we set TF_DETERMINISTIC_OPS=1, then there is no need to also set
        # TF_CUDNN_DETERMINISTIC=1. However, our experiment shows that
        # TF_CUDNN_DETERMINISTIC=1 is still required.
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        tf.set_random_seed(seed)
    elif backend_name == "tensorflow":
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        tf.random.set_seed(seed)
    elif backend_name == "pytorch":
        torch.manual_seed(seed)
    elif backend_name == "jax":
        global jax_random_seed
        jax_random_seed = seed
    elif backend_name == "paddle":
        paddle.seed(seed)
    global random_seed
    random_seed = seed


def enable_xla_jit(mode=True):
    """Enables just-in-time compilation with XLA.

    - For backend TensorFlow 1.x, by default, compiles with XLA when running on GPU.
      XLA compilation can only be enabled when running on GPU.
    - For backend TensorFlow 2.x, by default, compiles with XLA when running on GPU. If
      compilation with XLA makes your code slower on GPU, in addition to calling
      ``disable_xla_jit``, you may simultaneously try XLA with auto-clustering via

          $ TF_XLA_FLAGS=--tf_xla_auto_jit=2 path/to/your/program

    - Backend JAX always uses XLA.
    - Backends PyTorch and PaddlePaddle do not support XLA.

    Args:
        mode (bool): Whether to enable just-in-time compilation with XLA (``True``) or
            disable just-in-time compilation with XLA (``False``).
    """
    if backend_name == "tensorflow.compat.v1":
        if mode and not bkd.is_gpu_available():
            raise ValueError(
                "For backend TensorFlow 1.x, XLA compilation can only be enabled when "
                "running on GPU."
            )
    elif backend_name == "tensorflow":
        if not mode:
            mode = None
    elif backend_name == "pytorch":
        if mode:
            raise ValueError("Backend PyTorch does not support XLA.")
    elif backend_name == "jax":
        if not mode:
            raise ValueError("Backend JAX always uses XLA.")
    elif backend_name == "paddle":
        if mode:
            raise ValueError("Backend PaddlePaddle does not support XLA.")

    global xla_jit
    xla_jit = mode
    if xla_jit:
        print("Enable just-in-time compilation with XLA.", file=sys.stderr, flush=True)
    else:
        print("Disable just-in-time compilation with XLA.", file=sys.stderr, flush=True)


def disable_xla_jit():
    """Disables just-in-time compilation with XLA.

    - For backend TensorFlow 1.x, by default, compiles with XLA when running on GPU.
      XLA compilation can only be enabled when running on GPU.
    - For backend TensorFlow 2.x, by default, compiles with XLA when running on GPU. If
      compilation with XLA makes your code slower on GPU, in addition to calling
      ``disable_xla_jit``, you may simultaneously try XLA with auto-clustering via

          $ TF_XLA_FLAGS=--tf_xla_auto_jit=2 path/to/your/program

    - Backend JAX always uses XLA.
    - Backends PyTorch and PaddlePaddle do not support XLA.

    This is equivalent with ``enable_xla_jit(False)``.
    """
    enable_xla_jit(False)


def set_default_autodiff(value):
    """Sets the default automatic differentiation mode.

    The default automatic differentiation uses reverse mode.

    Args:
        value (String): 'reverse' or 'forward'.
    """
    global autodiff
    autodiff = value
    print(f"Set the default automatic differentiation to {value} mode.")


def set_parallel_scaling(scaling_mode):
    """Sets the scaling mode for data parallel acceleration.
    Weak scaling involves increasing the problem size proportionally with the number of processors,
    while strong scaling involves keeping the problem size fixed and increasing the number of processors.

    Args:
        scaling_mode (str): Whether 'weak' or 'strong'
    """
    global parallel_scaling
    parallel_scaling = scaling_mode
