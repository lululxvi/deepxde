# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


"""Internal utilities."""
import inspect
import sys
import timeit
from functools import wraps
from typing import Callable, Union

import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .external import apply


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()
        result = f(*args, **kwargs)
        te = timeit.default_timer()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper


def merge_dict(*dicts):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k in res:
                raise ValueError(f"Key {k} is duplicated.")
            res[k] = v

    return res


def subdict(dict_, keys):
    """Get a sub-dictionary from a dictionary."""
    return {k: dict_[k] for k in keys}


def check_not_none(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            is_none = []
            for a in attr:
                if not hasattr(self, a):
                    raise ValueError(f"{a} must be an attribute of the class.")
                is_none.append(getattr(self, a) is None)
            if any(is_none):
                raise ValueError(f"{attr} must not be None.")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def run_if_all_none(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            x = [getattr(self, a) for a in attr]
            if all(i is None for i in x):
                return func(self, *args, **kwargs)
            else:
                return x if len(x) > 1 else x[0]

        return wrapper

    return decorator


def run_if_any_none(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            x = [getattr(self, a) for a in attr]
            if any(i is None for i in x):
                return func(self, *args, **kwargs)
            return x if len(x) > 1 else x[0]

        return wrapper

    return decorator


def vectorize(**kwargs):
    """numpy.vectorize wrapper that works with instance methods.

    References:

    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
    - https://github.com/numpy/numpy/issues/9477
    """

    def decorator(fn):
        vectorized = np.vectorize(fn, **kwargs)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


def return_tensor(func):
    """Convert the output to a Tensor."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return u.math.asarray(func(*args, **kwargs), dtype=bst.environ.dftype())

    return wrapper


def to_numpy(tensors):
    """Create numpy ndarrays that shares the same underlying storage, if possible.

    Args:
        tensors. A Tensor or a list of Tensor.

    Returns:
        A numpy ndarray or a list of numpy ndarray.
    """
    if isinstance(tensors, (list, tuple)):
        return [np.asarray(tensor) for tensor in tensors]
    return np.asarray(tensors)


def make_dict(keys, values):
    """Convert two lists or two variables into a dictionary."""
    if isinstance(keys, (list, tuple)):
        if len(keys) != len(values):
            raise ValueError("keys and values have different length.")
        return dict(zip(keys, values))
    return {keys: values}


def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    apply(
        _save_animation,
        args=(filename, xdata, ydata),
        kwds={"y_reference": y_reference, "logy": logy},
    )


def _save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    """The animation figure window cannot be closed automatically.

    References:

    - https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
    """
    fig, ax = plt.subplots()
    if y_reference is not None:
        plt.plot(xdata, y_reference, "k-")
    (ln,) = plt.plot([], [], "r-o")

    def init():
        ax.set_xlim(np.min(xdata), np.max(xdata))
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(max(1e-4, np.min(ydata)), np.max(ydata))
        else:
            ax.set_ylim(np.min(ydata), np.max(ydata))
        return (ln,)

    def update(frame):
        ln.set_data(xdata, ydata[frame])
        return (ln,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(ydata), init_func=init, blit=True
    )
    ani.save(filename, writer="imagemagick", fps=30)
    plt.close()


def list_to_str(nums, precision=2):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


def tree_repr(tree, precision: int = 2):
    with np.printoptions(precision=precision, suppress=True, threshold=5):
        return repr(tree)
        # return repr(jax.tree.map(lambda x: repr(x), tree, is_leaf=u.math.is_quantity))


def get_num_args(func):
    """Get the number of arguments of a Python function.

    References:

    - https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function
    """
    # If the function is a class method decorated with functools.wraps, then "self" will
    # be in parameters, as inspect.signature follows wrapper chains by default, see
    # https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    #
    # Example:
    #
    # import inspect
    # from functools import wraps
    #
    # def dummy(f):
    #     print(f)
    #     print(inspect.signature(f))
    #
    #     @wraps(f)
    #     def wrapper(*args, **kwargs):
    #         f(*args, **kwargs)
    #
    #     print(wrapper)
    #     print(inspect.signature(wrapper))
    #     return wrapper
    #
    # class A:
    #     @dummy  # See the difference by commenting out this line
    #     def f(self, x):
    #         pass
    #
    # print(A.f)
    # print(inspect.signature(A.f))
    #
    # a = A()
    # g = dummy(a.f)
    params = inspect.signature(func).parameters
    return len(params) - ("self" in params)


def get_activation(activation: Union[str, Callable]):
    """Get the activation function."""
    if isinstance(activation, str):
        return getattr(bst.functional, activation)
    else:
        return activation
