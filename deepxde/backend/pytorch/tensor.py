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


def square(input):
    return torch.square(input)


def mean(input, dim, keepdims=False):
    return torch.mean(input, dim, keepdim=keepdims)


def reduce_mean(input):
    return torch.mean(input)
