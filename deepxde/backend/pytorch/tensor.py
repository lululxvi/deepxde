"""pytorch backend implementation"""
from packaging.version import Version

import torch


if Version(torch.__version__) < Version("2.0.0"):
    raise RuntimeError("DeepXDE requires PyTorch>=2.0.0.")

# To write device-agnostic (CPU or GPU) code, a common pattern is to first determine
# torch.device and then use it for all the tensors.
# https://pytorch.org/docs/stable/notes/cuda.html
# >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# >>> tensor.to(device=device)
# But, taking care of all tensors requires a lot of work.
# An alternative way is to use GPU by default if GPU is available, which is similar to
# TensorFlow.
if torch.cuda.is_available():
    if Version(torch.__version__) >= Version("2.1.0"):
        torch.set_default_device("cuda")
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
elif torch.backends.mps.is_available():
    fallback_device = torch.get_default_device()
    torch.set_default_device("mps")
    
    # As of March 2025, the macOS X-based GitHub Actions building environment sees
    # the MPS GPU, but cannot access it. So, a try-except workaround is applied.
    try:
        # A temporary trick to evade the Pytorch optimizer bug on MPS GPUs
        # See https://github.com/pytorch/pytorch/issues/149184
        torch._dynamo.disable()
        
        # If the Pytorch optimizer bug is fixed and the line above is removed,
        # the following code will perform a simple check of the MPS GPU
        test_nn = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Tanh(),
        )
        test_input = torch.randn(3, 1)
        test_run = test_nn(test_input)
        del test_nn, test_input, test_run
        torch.mps.empty_cache()
        
    except Exception as e:
        import warnings
        warnings.warn(
            f'An MPS GPU has been detected, but cannot be used. '
            f'Falling back to the CPU.\nThe exception message is:\n    {e}'
        )
        torch.set_default_device(fallback_device)


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


def size(tensor):
    return torch.numel(tensor)


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


def sparse_tensor(indices, values, shape):
    return torch.sparse_coo_tensor(list(zip(*indices)), values, shape, requires_grad=True)


def from_numpy(np_array):
    # Both torch.from_numpy and torch.as_tensor work without memory copy.
    # https://discuss.pytorch.org/t/from-numpy-vs-as-tensor/79932
    # https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
    # But torch.from_numpy cannot handle device.
    return torch.as_tensor(np_array)


def to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def concat(values, axis):
    return torch.cat(values, axis)


def stack(values, axis):
    return torch.stack(values, axis)


def expand_dims(tensor, axis):
    return torch.unsqueeze(tensor, axis)


def reverse(tensor, axis):
    return torch.flip(tensor, axis)


def roll(tensor, shift, axis):
    return torch.roll(tensor, shift, axis)


def lgamma(x):
    return torch.lgamma(x)


def elu(x):
    return torch.nn.functional.elu(x)


def relu(x):
    return torch.nn.functional.relu(x)


def gelu(x):
    return torch.nn.functional.gelu(x)


def selu(x):
    return torch.nn.functional.selu(x)


def sigmoid(x):
    return torch.nn.functional.sigmoid(x)


def silu(x):
    return torch.nn.functional.silu(x)


def sin(x):
    return torch.sin(x)


def cos(x):
    return torch.cos(x)


def exp(x):
    return torch.exp(x)


def square(x):
    return torch.square(x)


# pylint: disable=redefined-builtin
def abs(x):
    return torch.abs(x)


def minimum(x, y):
    return torch.minimum(x, y)


def tanh(x):
    return torch.tanh(x)


def pow(x, y):
    return torch.pow(x, y)


def mean(input_tensor, dim, keepdims=False):
    return torch.mean(input_tensor, dim, keepdim=keepdims)


def reduce_mean(input_tensor):
    return torch.mean(input_tensor)


def sum(input_tensor, dim, keepdims=False):
    return torch.sum(input_tensor, dim, keepdim=keepdims)


def reduce_sum(input_tensor):
    return torch.sum(input_tensor)


def prod(input_tensor, dim, keepdims=False):
    return torch.prod(input_tensor, dim, keepdim=keepdims)


def reduce_prod(input_tensor):
    return torch.prod(input_tensor)


# pylint: disable=redefined-builtin
def min(input_tensor, dim, keepdims=False):
    return torch.amin(input_tensor, dim, keepdim=keepdims)


def reduce_min(input_tensor):
    return torch.min(input_tensor)


# pylint: disable=redefined-builtin
def max(input_tensor, dim, keepdims=False):
    return torch.amax(input_tensor, dim, keepdim=keepdims)


def reduce_max(input_tensor):
    return torch.max(input_tensor)


def norm(tensor, ord=None, axis=None, keepdims=False):
    return torch.linalg.norm(tensor, ord=ord, dim=axis, keepdim=keepdims)


def zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return torch.zeros_like(input_tensor)


def matmul(x, y):
    return torch.mm(x, y)


def sparse_dense_matmul(x, y):
    return torch.sparse.mm(x, y)
