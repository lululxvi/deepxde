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


def shape(input_tensor):
    """Return the shape of the tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        list of ints: The tensor shape.
    """


def as_tensor(data, dtype=None):
    """Convert the data to a Tensor.

    If the data is already a tensor and has the same dtype, directly return.

    Args:
        data. Tensor object, numpy array, Python list, and Python scalar.
    dtype (data type, optional). It should be one of the values in the data type dict.
        If None, infers data type from data.

    Returns:
        Tensor. A framework-specific tensor.
    """


def from_numpy(np_array):
    """Create a tensor that shares the underlying numpy array memory, if possible.

    Args:
        np_array (numpy.ndarray). The numpy ndarray.

    Returns:
        Tensor. A framework-specific tensor.
    """


###############################################################################
# Element-wise math functions
# ---------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.


def elu(x):
    """Computes the exponential linear function."""


def relu(x):
    """Applies the rectified linear unit activation function."""


def selu(x):
    """Computes scaled exponential linear."""


def sigmoid(x):
    """Computes sigmoid of x element-wise."""


def silu(x):
    """Sigmoid Linear Unit (SiLU) function, also known as the swish function.
    silu(x) = x * sigmoid(x).
    """


def sin(x):
    """Computes sine of x element-wise."""


def square(x):
    """Returns the square of the elements of input."""


def tanh(x):
    """Computes hyperbolic tangent of x element-wise."""


###############################################################################
# Tensor functions on feature data
# --------------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.


def mean(input_tensor, dim, keepdims=False):
    """Returns the mean value of the input tensor in the given dimension dim."""


def reduce_mean(input_tensor):
    """Returns the mean value of all elements in the input tensor."""


def sum(input_tensor, dim, keepdims=False):
    """Returns the sum of the input tensor along the given dim.

    Args:
        input_tensor (Tensor). The input tensor.
        dim (int). The reduce dim.
        keepdims (bool). Whether to keep the summed dimension.

    Returns:
        Tensor: A framework-specific tensor.
    """


def reduce_sum(input_tensor):
    """Returns the sum of all elements in the input tensor.

    Args:
        input_tensor (Tensor). The input tensor.

    Returns:
        Tensor.
    """


def zeros(shape, dtype):
    """Creates a tensor with all elements set to zero.

    Args:
        shape (tuple of int). The tensor shape.
        dtype (data type). It should be one of the values in the data type dict.

    Returns:
        Tensor. The zero tensor.
    """


def zeros_like(input_tensor):
    """Create a zero tensor with the same shape, dtype and context of the given tensor.

    Args:
        input_tensor (Tensor).

    Returns:
        Tensor: The result.
    """
