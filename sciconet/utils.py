from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from functools import wraps
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation


def run_if_all_none(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            x = [getattr(self, a) for a in attr]
            if all(i is None for i in x):
                return func(self, *args, **kwargs)
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


def timing(f):
    """decorator for measuring the execution time of methods"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s" % (f.__name__, te - ts))
        return result

    return wrapper


def apply(func, args=None, kwds=None):
    """Clear Tensorflow GPU memory after model execution.

    Reference: https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def guarantee_initialized_variables(session, var_list=None):
    """Guarantee that all the specified variables are initialized.
    If a variable is already initialized, leave it alone. Otherwise, initialize it.
    If no variables are specified, checks all variables in the default graph.

    Args:
        var_list: List of Variable objects.

    References:
        https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
        https://www.programcreek.com/python/example/90525/tensorflow.report_uninitialized_variables
    """
    name_to_var = {v.op.name: v for v in tf.global_variables() + tf.local_variables()}
    uninitialized_variables = [
        name_to_var[name.decode("utf-8")]
        for name in session.run(tf.report_uninitialized_variables(var_list))
    ]
    session.run(tf.variables_initializer(uninitialized_variables))
    return uninitialized_variables


def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    apply(
        _save_animation,
        args=(filename, xdata, ydata),
        kwds={"y_reference": y_reference, "logy": logy},
    )


def _save_animation(filename, xdata, ydata, y_reference=None, logy=False):
    """The animation figure window cannot be closed automatically.
    https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
    """
    fig, ax = plt.subplots()
    if y_reference is not None:
        plt.plot(xdata, y_reference, "k-")
    ln, = plt.plot([], [], "r-o")

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
