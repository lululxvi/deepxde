from __future__ import division
from __future__ import print_function

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
