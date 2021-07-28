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
