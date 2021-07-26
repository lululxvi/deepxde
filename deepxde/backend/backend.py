"""This file defines the unified tensor framework interface required by DeepXDE.

The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by DeepXDE system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.

It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
that returns whether the interface is supported by the framework or not.
"""

# The backend currently being used
# lib will be set to one of the following backends.
lib = None
# All possible backends to use explicitly
tf = None
torch = None

###############################################################################
# Tensor, data type and context interfaces


def data_type_dict():
    """Returns a dictionary from data type string to the data type.

    The dictionary should include at least:
    float16
    float32
    float64
    uint8
    int8
    int16
    int32
    int64
    bool

    This function will be called only *once* during the initialization of the backend
    module. The returned dictionary will become the attributes of the backend module.

    Examples:
        >>> import tensorflow as tf
        >>> def data_type_dict():
        >>>     return {'float16': tf.float16, 'float32': tf.float32, ...}

        After the module is initialized.

        >>> import backend as bkd
        >>> bkd.float16  # this will point to tf.float16

    Returns:
        dict of str to data type. The data type dict.
    """


def is_tensor(obj):
    """Returns True if `obj` is a backend-native type tensor."""
