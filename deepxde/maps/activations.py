from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend as bkd
from .. import config
from ..backend import backend_name, tf


def linear(x):
    return x


def layer_wise_locally_adaptive(activation, n=1):
    """Layer-wise locally adaptive activation functions (L-LAAF).

    Examples:

    To define a L-LAAF ReLU with the scaling factor ``n = 10``:

    .. code-block:: python

        n = 10
        activation = f"LAAF-{n} relu"  # "LAAF-10 relu"

    References: `Jagtap et al., 2019 <https://arxiv.org/abs/1909.12228>`_.
    """
    # TODO: other backends
    if backend_name != "tensorflow.compat.v1":
        raise NotImplementedError("Only tensorflow.compat.v1 backend supports L-LAAF.")
    a = tf.Variable(1 / n, dtype=config.real(tf))
    return lambda x: activation(n * a * x)


def get(identifier):
    """Returns function.

    Args:
        identifier: Function or string.

    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        if identifier.startswith("LAAF"):
            identifier = identifier.split()
            n = float(identifier[0].split("-")[1])
            return layer_wise_locally_adaptive(get(identifier[1]), n=n)
        return {
            "elu": bkd.elu,
            "relu": bkd.relu,
            "selu": bkd.selu,
            "sigmoid": bkd.sigmoid,
            "silu": bkd.silu,
            "sin": bkd.sin,
            "swish": bkd.silu,
            "tanh": bkd.tanh,
        }[identifier]
    if callable(identifier):
        return identifier
    raise TypeError(
        "Could not interpret activation function identifier: {}".format(identifier)
    )
