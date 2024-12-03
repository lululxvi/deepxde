"""paddle backend implementation"""
from packaging.version import Version

import paddle

if Version(paddle.__version__) < Version("2.6.0") and Version(paddle.__version__) != Version("0.0.0"):
    raise RuntimeError("DeepXDE requires PaddlePaddle>=2.6.0 or PaddlePaddle==0.0.0(develop).")

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


def is_gpu_available():
    device = paddle.device.get_device()
    # "cpu"/"gpu:x"/"xpu:x"/"mlu:x"/"npu:x"
    return "gpu" in device


def is_tensor(obj):
    return paddle.is_tensor(obj)


def shape(input_tensor):
    return input_tensor.shape


def size(input_tensor):
    return int(paddle.numel(input_tensor))


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
        return paddle.to_tensor(initial_value, dtype=dtype, stop_gradient=False)
    return paddle.create_parameter(
        shape=[1],
        dtype=paddle.get_default_dtype() if dtype is None else dtype,
        default_initializer=paddle.nn.initializer.Constant(value=initial_value),
    )


def as_tensor(data, dtype=None):
    if paddle.is_tensor(data):
        if dtype is None or data.dtype == dtype:
            return data
        return data.astype(dtype)
    return paddle.to_tensor(data, dtype=dtype)


def sparse_tensor(indices, values, shape):
    return paddle.sparse.sparse_coo_tensor(
        list(zip(*indices)), values, shape, stop_gradient=False
    )


def from_numpy(np_array):
    return paddle.to_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def concat(values, axis):
    return paddle.concat(values, axis=axis)


def stack(values, axis):
    return paddle.stack(values, axis=axis)


def expand_dims(tensor, axis):
    return paddle.unsqueeze(tensor, axis=axis)


def reverse(tensor, axis):
    return paddle.flip(tensor, axis)


def roll(tensor, shift, axis):
    return paddle.roll(tensor, shift, axis)


def lgamma(tensor):
    return paddle.lgamma(tensor)


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


def cos(x):
    return paddle.cos(x)


def exp(x):
    return paddle.exp(x)


def square(x):
    return paddle.square(x)


# pylint: disable=redefined-builtin
def abs(x):
    return paddle.abs(x)


def minimum(x, y):
    return paddle.minimum(x, y)


def tanh(x):
    return paddle.tanh(x)


def pow(x, y):
    return paddle.pow(x, y)


def mean(input_tensor, dim, keepdims=False):
    return paddle.mean(input_tensor, axis=dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return paddle.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return paddle.sum(input_tensor, axis=dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return paddle.sum(input_tensor)


def prod(input_tensor, dim, keepdims=False):
    return paddle.prod(input_tensor, axis=dim, keepdim=keepdims)


def reduce_prod(input_tensor):
    return paddle.prod(input_tensor)


# pylint: disable=redefined-builtin
def min(input_tensor, dim, keepdims=False):
    return paddle.min(input_tensor, axis=dim, keepdim=keepdims)


def reduce_min(input_tensor):
    return paddle.min(input_tensor)


# pylint: disable=redefined-builtin
def max(input_tensor, dim, keepdims=False):
    return paddle.max(input_tensor, axis=dim, keepdim=keepdims)


def reduce_max(input_tensor):
    return paddle.max(input_tensor)


def norm(x, ord=None, axis=None, keepdims=False):
    if ord is None:
        ord = 2
    return paddle.linalg.norm(x, p=ord, axis=axis, keepdim=keepdims)


def zeros(shape, dtype):
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return paddle.zeros_like(input_tensor)


def matmul(x, y):
    return paddle.mm(x, y)


def sparse_dense_matmul(x, y):
    return paddle.sparse.matmul(x, y)


def l1_regularization(l1):
    return paddle.regularizer.L1Decay(coeff=l1)


def l2_regularization(l2):
    return paddle.regularizer.L2Decay(coeff=l2)
