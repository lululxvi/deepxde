from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


class VarianceScalingStacked(object):
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
        dtype=tf.float32,
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
        self.dtype = _assert_float_dtype(tf.as_dtype(dtype))

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


def get(identifier):
    identifiers = {
        "zeros": tf.zeros_initializer(),
        "He normal": tf.variance_scaling_initializer(scale=2.0),
        "He uniform": tf.variance_scaling_initializer(
            scale=2.0, distribution="uniform"
        ),
        "LeCun normal": tf.variance_scaling_initializer(),
        "LeCun uniform": tf.variance_scaling_initializer(distribution="uniform"),
        "Glorot normal": tf.glorot_normal_initializer(),
        "Glorot uniform": tf.glorot_uniform_initializer(),
        "Orthogonal": tf.orthogonal_initializer(),
    }
    identifiers_stacked = {
        "He normal": VarianceScalingStacked(scale=2.0),
        "He uniform": VarianceScalingStacked(scale=2.0, distribution="uniform"),
        "LeCun normal": VarianceScalingStacked(),
        "LeCun uniform": VarianceScalingStacked(distribution="uniform"),
    }

    if isinstance(identifier, str):
        if "stacked" in identifier:
            identifier = identifier.replace("stacked", "")
            return identifiers_stacked[identifier]
        else:
            return identifiers[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            "Could not interpret initializer identifier: " + str(identifier)
        )


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


def _assert_float_dtype(dtype):
    """Validate and return floating point type based on `dtype`.

    `dtype` must be a floating point type.

    Args:
        dtype: The data type to validate.

    Returns:
        Validated type.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype
