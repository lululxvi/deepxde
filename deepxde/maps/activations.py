from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import config
from ..backend import tf


def linear(x):
    return x


def swish(x):
    return x * tf.math.sigmoid(x)


def layer_wise_locally_adaptive(activation, n=1):
    """Layer-wise locally adaptive activation functions (L-LAAF).

    Examples:

    To define a L-LAAF ReLU with the scaling factor ``n = 10``:

    .. code-block:: python

        n = 10
        activation = f"LAAF-{n} relu"  # "LAAF-10 relu"

    References: `Jagtap et al., 2019 <https://arxiv.org/abs/1909.12228>`_.
    """
    a = tf.Variable(1 / n, dtype=config.real(tf))
    return lambda x: activation(n * a * x)


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        if identifier.startswith("LAAF"):
            identifier = identifier.split()
            n = float(identifier[0].split("-")[1])
            return layer_wise_locally_adaptive(get(identifier[1]), n=n)
        return {
            "elu": tf.nn.elu,
            "relu": tf.nn.relu,
            "selu": tf.nn.selu,
            "sigmoid": tf.nn.sigmoid,
            "sin": tf.sin,
            "swish": swish,
            "tanh": tf.nn.tanh,
        }[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            "Could not interpret activation function identifier:", identifier
        )
