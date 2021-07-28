from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .backend import tf
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
