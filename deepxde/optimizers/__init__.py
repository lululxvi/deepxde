import importlib
import sys

from .config import LBFGS_options, set_LBFGS_options
from ..backend import backend_name
from ..config import hvd

if hvd is not None:
    from .config import hvd_opt_options, set_hvd_opt_options

def _load_backend(mod_name):
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)


_load_backend(backend_name.replace(".", "_"))
