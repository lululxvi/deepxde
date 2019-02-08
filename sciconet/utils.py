from __future__ import division
from __future__ import print_function

import time
from functools import wraps


def runifnone(*attr):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            x = map(lambda i: getattr(self, i), attr)
            if len(filter(lambda i: i is None, x)) > 0:
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
        print('%r took %f s' % (f.__name__, te - ts))
        return result
    return wrapper