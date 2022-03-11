__all__ = ["get", "VarianceScalingStacked"]

import math

from .. import config
from ..backend import backend_name, tf, torch, jax, paddle


class VarianceScalingStacked:
    """Initializer capable of adapting its scale to the shape of weights tensors.

    With `distribution="truncated_normal" or "untruncated_normal"`,
    samples are drawn from a truncated/untruncated normal
    distribution with a mean of zero and a standard deviation (after truncation,
    if used) `stddev = sqrt(scale / n)`
    where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    Args:
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to create random seeds. See
            `tf.set_random_seed`
            for behavior.
        dtype: Default data type, used if no `dtype` argument is provided when
            calling the initializer. Only floating point types are supported.

    Raises:
        ValueError: In case of an invalid value for the "scale", mode" or
            "distribution" arguments.
    """

    def __init__(
        self,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
        seed=None,
    ):
        if scale <= 0.0:
            raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)
        distribution = distribution.lower()
        if distribution not in {
            "normal",
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }:
            raise ValueError("Invalid `distribution` argument:", distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self.dtype = config.real(tf)

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = self.dtype
        scale = self.scale
        fan_in, fan_out = _compute_fans_stacked(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "normal" or self.distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / 0.87962566103423978
            return tf.truncated_normal(shape, 0.0, stddev, dtype, seed=self.seed)
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return tf.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)
        else:
            limit = math.sqrt(3.0 * scale)
            return tf.random_uniform(shape, -limit, limit, dtype, seed=self.seed)


def _compute_fans_stacked(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple or TF tensor shape.

    Returns:
        A tuple of scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        # Assuming stacked NN.
        # kernel shape: (num_stack, fan_in)
        fan_in = shape[1]
        fan_out = 1
    else:
        # Assuming stacked NN.
        # kernel shape: (..., fan_in, fan_out)
        fan_in = shape[-2]
        fan_out = shape[-1]
    return fan_in, fan_out


def initializer_dict_tf():
    return {
        "Glorot normal": tf.keras.initializers.glorot_normal(),
        "Glorot uniform": tf.keras.initializers.glorot_uniform(),
        "He normal": tf.keras.initializers.he_normal(),
        "He uniform": tf.keras.initializers.he_uniform(),
        "LeCun normal": tf.keras.initializers.lecun_normal(),
        "LeCun uniform": tf.keras.initializers.lecun_uniform(),
        "Orthogonal": tf.keras.initializers.Orthogonal(),
        "zeros": tf.zeros_initializer(),
        # Initializers of stacked DeepONet
        "stacked He normal": VarianceScalingStacked(scale=2.0),
        "stacked He uniform": VarianceScalingStacked(scale=2.0, distribution="uniform"),
        "stacked LeCun normal": VarianceScalingStacked(),
        "stacked LeCun uniform": VarianceScalingStacked(distribution="uniform"),
    }


def initializer_dict_torch():
    return {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }


def initializer_dict_jax():
    return {
        "Glorot normal": jax.nn.initializers.glorot_normal(),
        "Glorot uniform": jax.nn.initializers.glorot_uniform(),
        "He normal": jax.nn.initializers.he_normal(),
        "He uniform": jax.nn.initializers.he_uniform(),
        "Lecun normal": jax.nn.initializers.lecun_normal(),
        "Lecun uniform": jax.nn.initializers.lecun_uniform(),
        "zeros": jax.nn.initializers.zeros,
    }


def initializer_dict_paddle():
    return {
        "Glorot normal": paddle.nn.initializer.XavierNormal(),
        "Glorot uniform": paddle.nn.initializer.XavierUniform(),
        "He normal": paddle.nn.initializer.KaimingNormal(),
        "He uniform": paddle.nn.initializer.KaimingUniform(),
        "zeros": paddle.nn.initializer.Constant(0.0),
    }


if backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    INITIALIZER_DICT = initializer_dict_tf()
elif backend_name == "pytorch":
    INITIALIZER_DICT = initializer_dict_torch()
elif backend_name == "jax":
    INITIALIZER_DICT = initializer_dict_jax()
elif backend_name == "paddle":
    INITIALIZER_DICT = initializer_dict_paddle()


def get(identifier):
    """Retrieve an initializer by the identifier.

    Args:
        identifier: String that contains the initializer name or an initializer
            function.

    Returns:
        Initializer instance base on the input identifier.
    """
    if isinstance(identifier, str):
        return INITIALIZER_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret initializer identifier: " + str(identifier))
