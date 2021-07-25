"""Internal utilities."""

import importlib
import sys

from . import array_ops_compat
from .external import *
from .internal import *
from ..backend import backend_name


def _load_backend(mod_name):
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


_load_backend(backend_name.replace(".", "_"))
