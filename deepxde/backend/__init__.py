# References: https://github.com/dmlc/dgl/tree/master/python/dgl/backend

import importlib
import json
import os
import sys

from . import backend
from .set_default_backend import set_default_backend

_enabled_apis = set()


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError(
            'API "%s" is not supported by backend "%s".'
            " You can switch to other backends by setting"
            " the DDE_BACKEND environment." % (api, mod_name)
        )

    return _missing_api


def notice_backend(backend_name):
    """Show notice about backend

    Args:
        backend_name: which backend used
    """
    if backend_name == "paddle":
        print("Using backend: paddle, which is default. tensorflow and pytorch also be available",
                "you can switch to them by using 'DDE_BACKEND=xxx'.\n", file=sys.stderr, flush=True)
    elif backend_name == "tensorflow.compat.v1":
        print("Using backend: tensorflow.compat.v1. paddle is default, which can support more examples now and under monitoring",
            "you can switch to it by using 'DDE_BACKEND=paddle'.\n", file=sys.stderr, flush=True)
    elif backend_name == "tensorflow":
        print("Using backend: tensorflow. paddle is default, which can support more examples now and under monitoring",
            "you can switch to it by using 'DDE_BACKEND=paddle'.\n", file=sys.stderr, flush=True)
    elif backend_name == "pytorch":
        print("Using backend: pytorch. paddle is default, which can support more examples now and under monitoring",
            "you can switch to it by using 'DDE_BACKEND=paddle'.\n", file=sys.stderr, flush=True)
    elif backend_name == "jax":
        print("Using backend: jax. paddle is default, which can support more examples now and under monitoring",
            "you can switch to it by using 'DDE_BACKEND=paddle'.\n", file=sys.stderr, flush=True)


def load_backend(mod_name):
    if mod_name not in [
        "tensorflow.compat.v1",
        "tensorflow",
        "pytorch",
        "jax",
        "paddle",
    ]:
        raise NotImplementedError("Unsupported backend: %s" % mod_name)

    notice_backend(mod_name)
    set_default_backend(mod_name)

    mod = importlib.import_module(".%s" % mod_name.replace(".", "_"), __name__)
    thismod = sys.modules[__name__]
    # log backend name
    setattr(thismod, "backend_name", mod_name)
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for backend "%s".'
                    % mod_name
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

            # override data type dict function
            setattr(thismod, "data_type_dict", data_type_dict)
            setattr(
                thismod,
                "reverse_data_type_dict",
                {v: k for k, v in data_type_dict.items()},
            )
        else:
            # load functions
            if api in mod.__dict__:
                _enabled_apis.add(api)
                setattr(thismod, api, mod.__dict__[api])
            else:
                setattr(thismod, api, _gen_missing_api(api, mod_name))


def check_backend(backend_name=None):
    """Detect user's package to select backend.

    Return:
        backend name or None
    """
    # pylint: disable=C0415
    if backend_name in ["paddle", None]:
        try:
            import paddle
            assert paddle   # silence pyflakes
            return "paddle"
        except ImportError:
            pass
    
    if backend_name in ["tensorflow.compat.v1", "tensorflow", None]:
        try:
            import tensorflow.compat.v1
            assert tensorflow.compat.v1   # silence pyflakes
            print("Backend be set to 'tensorflow.compat.v1' automaticlly, if you want to set it to 'tensorflow',", 
                "please using 'DDE_BACKEND=tensorflow'.\n", file=sys.stderr, flush=True)
            return "tensorflow.compat.v1"
        except ImportError:
            pass

    if backend_name in ["pytorch", None]:
        try:
            import torch
            assert torch   # silence pyflakes
            return "pytorch"
        except ImportError:
            pass

    if backend_name in ["jax", None]:
        try:
            import jax
            assert jax   # silence pyflakes
            return "jax"
        except ImportError:
            pass

    return None


def ask_install(backend_name):
    """Ask the user to install the backend.

    Args:
        backend_name: which backend detected.
    """
    if backend_name == 'paddle':
        return backend_name
        
    if backend_name in [
        "tensorflow.compat.v1",
        "tensorflow",
        "pytorch",
        "jax",
    ]:
        notice = "{}{}{}".format("Your backend is ", backend_name, 
                " now. paddle is default which can support more examples and under monitoring, do you want to try and install it (y/n): ")
    else:
        notice = "No available backend found. paddle is default which can support more examples and under monitoring, do you want to install it (y/n): "
    
    try:
        msg = input(notice)
    except EOFError:
        msg = "n"
    
    cnt = 0
    while cnt < 3:
        if msg == 'y':
            install_backend()
            return "paddle"
        if msg == 'n':
            break
        cnt += 1
        msg = input("Please enter correctly (y/n): ")

    if backend_name in [
        "tensorflow.compat.v1",
        "tensorflow",
        "pytorch",
        "jax",
    ]:
        print("No installing required, continue running...\n", file=sys.stderr, flush=True)
    else:
        sys.exit("No available backend found, you can manually install one of paddle/tensorflow.compat.va/tensorflow/pytorch/jax ",
                 "and select backend by using 'DDE_BACKEND=xxx', Running stopped!")

    return backend_name


def run_install(command):
    """Send command to terminal and print it.

    Args:
        command: command to be sent to terminal.
    """
    print("Install command:", command,"\n")
    installed = os.system(command)
    if installed == 0:
        print("Paddle installed successfully!\n", file=sys.stderr, flush=True)
    else:
        sys.exit("Paddle installed failed. You can visit https://www.paddlepaddle.org.cn to install it manually. Running stopped!")


def install_backend():
    """Generate command and install paddle.
    """
    # get user's platform
    platform = sys.platform
    if platform in ["win32", "cygwin"]:
        platform = 'windows'
    elif platform in ["linux","linux2"]:
        platform = 'linux'
    elif platform == "darwin":
        print("Paddle only can be installed in macOS with cpu by pip now, installing cpu version...", file=sys.stderr, flush=True)
        cmd = "python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html"
        run_install(cmd)
    else:
        sys.exit("your system is", platform, ", paddle only supported Window/Linux/macOS now, sorry. Running stopped!")

    # get user's device
    nvcc_text = os.popen("nvcc -V").read()
    if nvcc_text != '':
        cuda_version = nvcc_text.split("Cuda compilation tools, release ")[-1].split(",")[0]
        version = int(float(cuda_version)*10)
        if version not in [102,112,116,117]:
            print("{}{}{}".format("There is no avaliable Cuda, your Cuda version is ",cuda_version,
                    " however paddle only supports Cuda 10.2/11.2/11.6/11.7 now.\n"), file=sys.stderr, flush=True)
        else:
            print("Installing cuda", cuda_version, "version...", file=sys.stderr, flush=True)
            cmd = "{}{}{}{}{}".format("python -m pip install paddlepaddle-gpu==0.0.0.post", version, " -f https://www.paddlepaddle.org.cn/whl/", platform, "/gpu/develop.html")
            run_install(cmd)
    if platform == 'linux':
        roc_text1 = os.popen("/opt/rocm/bin/rocminfo").read()
        roc_text2 = os.popen("/opt/rocm/opencl/bin/clinfo").read()
        if roc_text1 != '' and roc_text2 != '':
            print("Installing ROCm4.0 version...", file=sys.stderr, flush=True)
            cmd = "pip install --pre paddlepaddle-rocm -f https://www.paddlepaddle.org.cn/whl/rocm/develop.html"
            run_install(cmd)
        else:
            print("There is no avaliable ROCm4.0.\n", file=sys.stderr, flush=True)

    print("Installing cpu version...", file=sys.stderr, flush=True)
    cmd = "{}{}{}".format("python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/", platform, "/cpu-mkl-avx/develop.html")
    run_install(cmd)


def get_preferred_backend():
    backend_name = None
    config_path = os.path.join(os.path.expanduser("~"), ".deepxde", "config.json")
    if "DDE_BACKEND" in os.environ:
        backend_name = check_backend(os.getenv("DDE_BACKEND"))
    # Backward compatibility
    elif "DDEBACKEND" in os.environ:
        backend_name = check_backend(os.getenv("DDEBACKEND"))
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = check_backend(config_dict.get("backend", "").lower())
    
    # no backend config or config error
    if backend_name is None:
        backend_name = check_backend()
    
    backend_name = ask_install(backend_name)
    return backend_name


load_backend(get_preferred_backend())


def is_enabled(api):
    """Return true if the api is enabled by the current backend.

    Args:
        api (string): The api name.

    Returns:
        bool: ``True`` if the API is enabled by the current backend.
    """
    return api in _enabled_apis
