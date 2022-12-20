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


def get_nprocs():
    """Get number of replica(s)

    Returns:
        int: number of replica(s)
    """
    return paddle.distributed.get_world_size()


def get_rank():
    """Get current rank

    Returns:
        int: current rank
    """
    return paddle.distributed.get_rank()


def get_nprocs_and_rank():
    """Get number of replica(s) and current rank

    Returns:
        Tuple of ints: (number of replica(s), current rank)
    """
    nprocs = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    return nprocs, rank


