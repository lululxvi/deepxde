"""pytorch backend implementation"""
import torch


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


def is_tensor(obj):
    return torch.is_tensor(obj)


def shape(input_tensor):
    return list(input_tensor.shape)


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
    return torch.from_numpy(np_array)


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


def zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def zeros_like(input_tensor):
    return torch.zeros_like(input_tensor)
