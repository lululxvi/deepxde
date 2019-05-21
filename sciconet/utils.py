from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from functools import wraps
from multiprocessing import Pool

import tensorflow as tf


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
