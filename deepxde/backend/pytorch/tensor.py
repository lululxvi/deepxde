"""pytorch backend implementation"""
from distutils.version import LooseVersion

import torch


if LooseVersion(torch.__version__) < LooseVersion("1.9.0"):
    raise RuntimeError("DeepXDE requires PyTorch>=1.9.0.")

# To write device-agnostic (CPU or GPU) code, a common pattern is to first determine
# torch.device and then use it for all the tensors.
# https://pytorch.org/docs/stable/notes/cuda.html
# >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# >>> tensor.to(device=device)
# But, taking care of all tensors requires a lot of work.
# An alternative way is to use GPU by default if GPU is available, which is similar to
# TensorFlow.
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


lib = torch


def data_type_dict():
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }


def is_gpu_available():
    return torch.cuda.is_available()


def is_tensor(obj):
    return torch.is_tensor(obj)


def shape(input_tensor):
    return list(input_tensor.shape)


def tensor_shape(input_tensor):
    return list(input_tensor.shape)


def ndim(input_tensor):
    return input_tensor.dim()


def transpose(tensor, axes=None):
    if axes is None:
        axes = tuple(range(tensor.dim())[::-1])
    return torch.permute(tensor, axes)


def reshape(tensor, shape):
    return torch.reshape(tensor, shape)


def Variable(initial_value, dtype=None):
    return torch.tensor(initial_value, dtype=dtype, requires_grad=True)


def as_tensor(data, dtype=None):
    if isinstance(data, torch.Tensor):
        if dtype is None or data.dtype == dtype:
            return data
        return data.type(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


def from_numpy(np_array):
    # Both torch.from_numpy and torch.as_tensor work without memory copy.
    # https://discuss.pytorch.org/t/from-numpy-vs-as-tensor/79932
    # https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
    # But torch.from_numpy cannot handle device.
    return torch.as_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def elu(x):
    return torch.nn.functional.elu(x)


def relu(x):
    return torch.nn.functional.relu(x)


def selu(x):
    return torch.nn.functional.selu(x)


def sigmoid(x):
    return torch.nn.functional.sigmoid(x)


def silu(x):
    return torch.nn.functional.silu(x)


def sin(x):
    return torch.sin(x)


def exp(x):
    return torch.exp(x)


def pow(x, y):
    return torch.pow(x, y)


def square(x):
    return torch.square(x)


def tanh(x):
    return torch.tanh(x)


def mean(input_tensor, dim, keepdims=False):
    return torch.mean(input_tensor, dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return torch.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return torch.sum(input_tensor, dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return torch.sum(input_tensor)


def norm(tensor, ord=None, axis=None, keepdims=False):
    return torch.linalg.norm(tensor, ord=ord, dim=axis, keepdim=keepdims)


def zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return torch.zeros_like(input_tensor)


def lgamma(x):
    return torch.lgamma(x)


def matmul(x, y):
    return torch.matmul(x, y)


def size(tensor):
    return torch.numel(tensor)


def sparse_tensor(indices, values, shape):
    x = [p[0] for p in indices]  # [num_of_nonzeros, ]
    y = [p[1] for p in indices]  # [num_of_nonzeros, ]
    indices = torch.stack(
        [torch.tensor(x), torch.tensor(y)]
    )  # [2, num_of_nonzeros]
    coo_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return coo_tensor


def sparse_tensor_dense_matmul(x, y):
    if not x.is_sparse_csr:
        # NOTE: only support CSR tensor multiplication now
        # refer to https://pytorch.org/docs/stable/sparse.html#csr-tensor-operations
        x = x.to_sparse_csr()
    return torch.matmul(x, y)


def constant(values, dtype):
    return torch.tensor(values, dtype=dtype)


def concat(values, axis):
    return torch.concat(values, axis)


def reverse(tensor, axis):
    return torch.flip(tensor, axis)


def expand_dims(tensor, axis):
    return torch.unsqueeze(tensor, axis)


def cos(x):
    return torch.cos(x)


def roll(tensor, shift, axis=None):
    return torch.roll(tensor, shift, axis)


def gradients(outputs, inputs):
    # NOTE: set create_graph=True to enable high-order differentiation
    return torch.autograd.grad(outputs, inputs, create_graph=True)
