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
jax = None
paddle = None

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


def is_gpu_available():
    """Returns a bool indicating if GPU is currently available.

    Returns:
        True if a GPU device is available.
    """


def is_tensor(obj):
    """Returns True if `obj` is a backend-native type tensor."""


def shape(input_tensor):
    """Return the shape of the tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        tuple or list of ints: The tensor shape.
    """


def size(input_tensor):
    """Return the total number of elements in the input tensor.

    Args:
        input_tensor (Tensor). The input tensor.

    Returns:
        int: The total number of elements in the input tensor.
    """


def ndim(input_tensor):
    """Returns the number of dimensions of the tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        int: The number of dimensions.
    """


def transpose(tensor, axes=None):
    """Reverse or permute the axes of a tensor; returns the modified array.

    For a tensor with two axes, transpose gives the matrix transpose.

    Args:
        tensor (Tensor): Input tensor.
        axes (tuple of ints): A permutation of the dimensions.

    Returns:
        A tensor with its axes permuted. A view is returned whenever possible.
    """


def reshape(tensor, shape):
    """Gives a new shape to a tensor without changing its data.

    Args:
        tensor (Tensor): The tensor to be reshaped.
        shape (tuple of ints): The new shape should be compatible with the original
            shape.

    Returns:
        Reshaped tensor. This will be a new view object if possible.
    """


def Variable(initial_value, dtype=None):
    """Return a trainable variable.

    Args:
        initial_value: The initial value of the variable.
        dtype: The desired data type of returned tensor. Default: if None, infers data
            type from data.
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


def sparse_tensor(indices, values, shape):
    """Construct a sparse tensor based on given indices, values and shape.

    Args:
        indices (list of tuple). A 2-D int list of shape [N, ndims], which specifies
            the indices of the elements in the sparse tensor that contain nonzero values
            (elements are zero-indexed), such as [(x1, y1), (x2, y2), ..., (xN, yN)].
        values (Tensor). Values of non-zero elements, with shape of [N].
        shape (list or tuple). Dense shape of constructed tensor.

    Returns:
        SparseTensor: A sparse tensor.
    """


def from_numpy(np_array):
    """Create a tensor that shares the underlying numpy array memory, if possible.

    Args:
        np_array (numpy.ndarray). The numpy ndarray.

    Returns:
        Tensor. A framework-specific tensor.
    """


def to_numpy(input_tensor):
    """Create a numpy ndarray that shares the same underlying storage, if possible.

    Args:
        input_tensor (Tensor).

    Returns:
        np_array (numpy.ndarray). The numpy ndarray.
    """


def concat(values, axis):
    """Returns the concatenation of the input tensors along the given dim.

    Args:
        values (list or tuple of Tensor). The input tensors in list or tuple.
        axis (int). The concatenating dim.

    Returns:
        Tensor: Concatenated tensor.
    """


def stack(values, axis):
    """Returns the stack of the input tensors along the given dim.

    Args:
        values (list or tuple of Tensor). The input tensors in list or tuple.
        axis (int). The stacking dim.

    Returns:
        Tensor: Stacked tensor.
    """


def expand_dims(tensor, axis):
    """Expand dim for tensor along given axis.

    Args:
        tensor (Tensor). The input tensor.
        axis (int). Axis to expand.

    Returns:
        Tensor: Expanded tensor.
    """


def reverse(tensor, axis):
    """Reverse the order of elements along the given axis.

    Args:
        tensor (Tensor). The input tensor.
        axis (int). Axis to flip on.

    Returns:
        Tensor: Tensor which is flipped along given axis.
    """


def roll(tensor, shift, axis):
    """Roll the tensor along the given axis (axes).

    Args:
        tensor (Tensor). The input tensor.
        shift (int or tuple of ints). The number of places by which the elements of the
            tensor are shifted.
        axis (int or tuple of ints). Axis (axes) along which to roll.

    Returns:
        Tensor: Rolled tensor.
    """


###############################################################################
# Element-wise math functions
# ---------------------------
# These functions are performance critical, so it's better to have efficient
# implementation in each framework.


def lgamma(x):
    """Computes the natural logarithm of the absolute value of the gamma function of x
    element-wise.
    """


def elu(x):
    """Computes the exponential linear function."""


def relu(x):
    """Applies the rectified linear unit activation function."""


def gelu(x):
    """Computes Gaussian Error Linear Unit function."""


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


def cos(x):
    """Computes cosine of x element-wise."""


def exp(x):
    """Computes exponential of x element-wise."""


def square(x):
    """Returns the square of the elements of input."""


def abs(x):
    """Computes the absolute value element-wise."""


def minimum(x, y):
    """Returns the minimum of x and y (i.e. x < y ? x : y) element-wise."""


def tanh(x):
    """Computes hyperbolic tangent of x element-wise."""


def pow(x, y):
    """Computes the power of one value to another: x ^ y."""


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


def prod(input_tensor, dim, keepdims=False):
    """Returns the product of the input tensor along the given dim.

    Args:
        input_tensor (Tensor). The input tensor.
        dim (int). The reduce dim.
        keepdims (bool). Whether to keep the product dimension.

    Returns:
        Tensor: A framework-specific tensor.
    """


def reduce_prod(input_tensor):
    """Returns the product of all elements in the input tensor.

    Args:
        input_tensor (Tensor). The input tensor.

    Returns:
        Tensor.
    """


def min(input_tensor, dim, keepdims=False):
    """Returns the minimum of the input tensor along the given dim.

    Args:
        input_tensor (Tensor). The input tensor.
        dim (int). The reduce dim.
        keepdims (bool). Whether to keep the dimension.

    Returns:
        Tensor: A framework-specific tensor.
    """


def reduce_min(input_tensor):
    """Returns the minimum of all elements in the input tensor.

    Args:
        input_tensor (Tensor). The input tensor.

    Returns:
        Tensor.
    """


def max(input_tensor, dim, keepdims=False):
    """Returns the maximum of the input tensor along the given dim.

    Args:
        input_tensor (Tensor). The input tensor.
        dim (int). The reduce dim.
        keepdims (bool). Whether to keep the dimension.

    Returns:
        Tensor: A framework-specific tensor.
    """


def reduce_max(input_tensor):
    """Returns the maximum of all elements in the input tensor.

    Args:
        input_tensor (Tensor). The input tensor.

    Returns:
        Tensor.
    """


def norm(tensor, ord=None, axis=None, keepdims=False):
    """Computes a vector norm.

    Due to the incompatibility of different backends, only some vector norms are 
    supported, and matrix norm is not supported now. This API follows numpy.linalg.norm().

    Args:
        tensor (Tensor). If axis is None, tensor must be 1-D.
        ord (int, float, inf). Order of the norm. For vector norm, supported values are 
            1, 2, inf, and any positive real number. Default is 2-norm for vectors.
        axis (int). If axis is an integer, it specifies the axis of tensor along which 
            to compute the vector norms. If axis is None, then a vector norm 
            (when tensor is 1-D) is returned. The default is None.
        keepdims (bool). If this is set to True, the axes which are normed over are left
            in the result as dimensions with size one.

    Returns:
        Tensor: Norm of the vector(s).
    """


def zeros(shape, dtype):
    """Creates a tensor with all elements set to zero.

    Args:
        shape (tuple of ints). The tensor shape.
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


def matmul(x, y):
    """Compute matrix multiplication for two matrices x and y.

    Args:
        x (Tensor). The first matrix to be matrix multiplied.
        y (Tensor). The second matrix to be matrix multiplied.

    Returns:
        Tensor: The multiplication result.
    """


def sparse_dense_matmul(x, y):
    """Compute matrix multiplication of a sparse matrix x and a sparse/dense matrix y.

    Args:
        x (Sparse Tensor). The first sparse matrix to be multiplied.
        y (Sparse Tensor or Tensor). The second matrix to be multiplied, which could be sparse or dense.

    Returns:
        Tensor: The multiplication result.
    """
