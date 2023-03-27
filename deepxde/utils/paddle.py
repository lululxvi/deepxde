"""Utilities of paddle."""

import paddle


def all_gather(tensor, concat=True, axis=0):
    """Gather tensor from all devices, concatenate them along given axis(if specified).

    Args:
        tensor (Tensor): Tensor to be gathered from all GPUs.
        concat (bool, optional): Whether to concatenate gathered Tensors. Defaults to True.
        axis (int, optional): Axis which concatenated along. Defaults to 0.

    Returns:
        Tensor or list of Tensors: Gathered Tensors.
    """
    tensor_list = []
    paddle.distributed.all_gather(tensor_list, tensor)
    if concat:
        return paddle.concat(tensor_list, axis)
    return tensor_list
