"""Utilities of paddle."""
import paddle


def all_gather(tensor, concat=True, axis=0):
    """Gather tensor from all devices, concatenate them along given axis(if specified)

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


def get_world_size():
    """Get world size

    Returns:
        int: world size
    """
    return paddle.distributed.get_world_size()


def get_rank():
    """Get current rank

    Returns:
        int: current rank
    """
    return paddle.distributed.get_rank()


def get_dist_info():
    """Get world size and current rank

    Returns:
        Tuple of ints: (world size, current rank)
    """
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    return world_size, rank


