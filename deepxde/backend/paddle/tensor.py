"""paddle backend implementation"""
from packaging.version import Version
import paddle

if Version(paddle.__version__) < Version("2.3.0") and Version(paddle.__version__) != Version("0.0.0") :
    raise RuntimeError("DeepXDE requires PaddlePaddle>=2.3.0")

if paddle.device.is_compiled_with_cuda():
    paddle.device.set_device("gpu")

lib = paddle


def data_type_dict():
    return {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "bool": paddle.bool,
    }


def is_tensor(obj):
    return paddle.is_tensor(obj)


def shape(input_tensor):
    return input_tensor.shape


def tensor_shape(input_tensor):
    return input_tensor.shape


def ndim(input_tensor):
    return input_tensor.ndim


def transpose(tensor, axes=None):
    if axes is None:
        axes = tuple(range(tensor.ndim)[::-1])
    return paddle.transpose(tensor, axes)


def reshape(tensor, shape):
    return paddle.reshape(tensor, shape)


def Variable(initial_value, dtype=None):
    if paddle.in_dynamic_mode():
        return paddle.create_parameter(
            shape=[1],
            dtype="float32" if dtype is None else dtype,
            default_initializer=paddle.nn.initializer.Constant(value=initial_value))
    else:
        return paddle.static.create_parameter(
            shape=[1],
            dtype="float32" if dtype is None else dtype,
            default_initializer=paddle.nn.initializer.Constant(value=initial_value))


def as_tensor(data, dtype=None):
    if paddle.is_tensor(data):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return paddle.to_tensor(data, dtype=dtype)


def from_numpy(np_array):
    return paddle.to_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def elu(x):
    return paddle.nn.functional.elu(x)


def relu(x):
    return paddle.nn.functional.relu(x)


def selu(x):
    return paddle.nn.functional.selu(x)


def sigmoid(x):
    return paddle.nn.functional.sigmoid(x)


def silu(x):
    return paddle.nn.functional.silu(x)


def sin(x):
    return paddle.sin(x)


def exp(x):
    return paddle.exp(x)


def pow(x, y):
    return paddle.pow(x, y)


def square(x):
    return paddle.square(x)


def tanh(x):
    return paddle.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return paddle.mean(input_tensor, axis=dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return paddle.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return paddle.sum(input_tensor, axis=dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return paddle.sum(input_tensor)


def norm(x, p=None, axis=None, keepdims=False):
    return paddle.linalg.norm(x, p=p, axis=axis, keepdim=keepdims)


def zeros(shape, dtype):
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return paddle.full_like(input_tensor, 0.0)


def lgamma(tensor):
    return paddle.lgamma(tensor)


def matmul(x, y):
    return paddle.matmul(x, y)


def size(tensor):
    return paddle.numel(tensor)


def sparse_tensor(indices, values, shape):
    x = [p[0] for p in indices]  # [num_of_nonzeros, ]
    y = [p[1] for p in indices]  # [num_of_nonzeros, ]
    indices = paddle.stack(
        [paddle.to_tensor(x), paddle.to_tensor(y)]
    )  # [2, num_of_nonzeros]
    return paddle.sparse.sparse_coo_tensor(indices=indices, values=values, shape=list(shape), stop_gradient=False)


def sparse_tensor_dense_matmul(x, y):
    return paddle.sparse.matmul(x, y)


def ones(shape, dtype):
    return paddle.ones(shape=shape, dtype=dtype)


def constant(values, dtype):
    return paddle.to_tensor(values, dtype=dtype)


def concat(values, axis):
    return paddle.concat(values, axis=axis)


def reverse(tensor, axis):
    return paddle.flip(tensor, axis)


def expand_dims(tensor, axis):
    return paddle.unsqueeze(tensor, axis=axis)


def cos(tensor):
    return paddle.cos(tensor)


def roll(tensor, shift, axis=None):
    return paddle.roll(tensor, shift, axis)


def gradients(outputs, inputs):
    if paddle.in_dynamic_mode():
        # for dynamic graph
        # NOTE: set create_graph=True to enable high-order differentiation
        return paddle.grad(outputs, inputs, create_graph=True)
    else:
        # for static graph
        return paddle.static.gradients(outputs, inputs)
