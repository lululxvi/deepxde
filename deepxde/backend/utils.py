import os
import sys


# Verify if the backend is available/importable
def import_tensorflow_compat_v1():
    # pylint: disable=import-outside-toplevel
    try:
        import tensorflow.compat.v1

        assert tensorflow.compat.v1  # silence pyflakes
        return True
    except ImportError:
        return False


def import_tensorflow():
    # pylint: disable=import-outside-toplevel
    try:
        import tensorflow

        assert tensorflow  # silence pyflakes
        return True
    except ImportError:
        return False


def import_pytorch():
    # pylint: disable=import-outside-toplevel
    try:
        import torch

        assert torch  # silence pyflakes
        return True
    except ImportError:
        return False


def import_jax():
    # pylint: disable=import-outside-toplevel
    try:
        import jax

        assert jax  # silence pyflakes
        return True
    except ImportError:
        return False


def import_paddle():
    # pylint: disable=import-outside-toplevel
    try:
        import paddle

        assert paddle  # silence pyflakes
        return True
    except ImportError:
        return False


def verify_backend(backend_name):
    """Verify if the backend is available. If it is available,
    do nothing, otherwise, raise RuntimeError.
    """
    import_funcs = {
        "tensorflow.compat.v1": import_tensorflow_compat_v1,
        "tensorflow": import_tensorflow,
        "pytorch": import_pytorch,
        "jax": import_jax,
        "paddle": import_paddle,
    }
    if backend_name not in import_funcs:
        raise NotImplementedError(
            f"Unsupported backend: {backend_name}.\n"
            "Please select backend from tensorflow.compat.v1, tensorflow, pytorch, jax or paddle."
        )
    if not import_funcs[backend_name]():
        raise RuntimeError(
            f"Backend is set as {backend_name}, but '{backend_name}' failed to import."
        )


def get_available_backend():
    backends = ["tensorflow.compat.v1", "tensorflow", "pytorch", "jax", "paddle"]
    import_funcs = {
        "tensorflow.compat.v1": import_tensorflow_compat_v1,
        "tensorflow": import_tensorflow,
        "pytorch": import_pytorch,
        "jax": import_jax,
        "paddle": import_paddle,
    }
    for backend in backends:
        if import_funcs[backend]():
            return backend
    return None


# Ask user if install paddle and install it
def run_install(command):
    """Send command to terminal and print it.

    Args:
        command (str): command to be sent to terminal.
    """
    print("Install command:", command)
    installed = os.system(command)
    if installed == 0:
        print("Paddle installed successfully!\n", file=sys.stderr, flush=True)
    else:
        sys.exit(
            "Paddle installed failed!\n"
            "Please visit https://www.paddlepaddle.org.cn/en for help and install it manually, "
            "or use another backend."
        )


def get_platform():
    """Get user's platform.

    Returns:
        platform (str): "windows", "linux" or "darwin"
    """
    if sys.platform in ["win32", "cygwin"]:
        return "windows"
    if sys.platform in ["linux", "linux2"]:
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    sys.exit(
        f"Your system {sys.platform} is not supported by Paddle. Paddle installation stopped.\n"
        "Please use another backend."
    )


def get_cuda(platform):
    """Check whether cuda is avaliable and get its version.

    Returns:
        cuda_verion (str) or None
    """
    if platform == "linux":
        cuda_list = [101, 102, 110, 111, 112, 116, 117, 118, 120]
    elif platform == "windows":
        cuda_list = [101, 102, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120]
    nvcc_text = os.popen("nvcc -V").read()
    if nvcc_text != "":
        cuda_version = nvcc_text.split("Cuda compilation tools, release ")[-1].split(
            ","
        )[0]
        version = int(float(cuda_version) * 10)
        if version not in cuda_list:
            cuda_list_str = [str(i / 10) for i in cuda_list]
            msg_cl = "/".join(cuda_list_str)
            print(
                f"Your CUDA version is {cuda_version},",
                f"but Paddle only supports CUDA {msg_cl} for {platform} now.",
                file=sys.stderr,
                flush=True,
            )
        else:
            return cuda_version
    return None


def get_rocm():
    """Check whether ROCm4.0 is avaliable.

    Returns:
        bool
    """
    roc_text1 = os.popen("/opt/rocm/bin/rocminfo").read()
    roc_text2 = os.popen("/opt/rocm/opencl/bin/clinfo").read()
    if roc_text1 != "" and roc_text2 != "":
        return True
    print("There is no avaliable ROCm4.0.", file=sys.stderr, flush=True)
    return False


def check_avx(platform):
    """Check whether avx is supported."""
    avx_text1 = avx_text2 = ""
    if platform == "darwin":
        avx_text1 = os.popen("sysctl machdep.cpu.features | grep -i avx").read()
        avx_text2 = os.popen("sysctl machdep.cpu.leaf7_features | grep -i avx").read()
    elif platform == "linux":
        avx_text1 = os.popen("cat /proc/cpuinfo | grep -i avx").read()
    elif platform == "windows":
        return

    if avx_text1 == "" and avx_text2 == "":
        sys.exit(
            "Your machine doesn't support AVX, which is required by PaddlePaddle (develop version). "
            "Paddle installation stopped.\n"
            "Please use another backend."
        )


def get_python_executable():
    """Get user's python executable.

    Returns:
        str: python exection path
    """
    return sys.executable


def generate_cmd(py_exec, platform, cuda_version=None, has_rocm=False):
    """Generate command.

    Args:
        py_exec (str): python executable path.
        platform (str): User's platform.
        cuda_version (str): Whether cuda is avaliable and its version if it is.
        has_rocm (bool): Whether ROCm4.0 has been installed.
    """
    if platform == "darwin":
        print(
            "Paddle can only be installed in macOS with CPU version now. ",
            "Installing CPU version...",
            file=sys.stderr,
            flush=True,
        )
        cmd = "{}{}{}".format(
            py_exec,
            " -m pip install paddlepaddle==0.0.0 -f ",
            "https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html",
        )
        return cmd

    if cuda_version is not None:
        print(f"Installing CUDA {cuda_version} version...", file=sys.stderr, flush=True)
        cmd = "{}{}{}{}{}{}".format(
            py_exec,
            " -m pip install paddlepaddle-gpu==0.0.0.post",
            int(float(cuda_version) * 10),
            " -f https://www.paddlepaddle.org.cn/whl/",
            platform,
            "/gpu/develop.html",
        )
        return cmd

    if platform == "linux" and has_rocm:
        print("Installing ROCm4.0 version...", file=sys.stderr, flush=True)
        cmd = "{}{}{}".format(
            py_exec,
            " -m pip install --pre paddlepaddle-rocm -f ",
            "https://www.paddlepaddle.org.cn/whl/rocm/develop.html",
        )
        return cmd

    print("Installing CPU version...", file=sys.stderr, flush=True)
    cmd = "{}{}".format(
        py_exec,
        " -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/",
    )
    if platform == "windows":
        cmd += "windows/cpu-mkl-avx/develop.html"
    elif platform == "linux":
        cmd += "linux/cpu-mkl/develop.html"
    return cmd


def install_paddle():
    """Generate command and install paddle."""
    # get user's platform
    platform = get_platform()
    # check avx
    check_avx(platform)
    # check python version
    py_exec = get_python_executable()

    # get user's device and generate cmd
    if platform == "darwin":
        cmd = generate_cmd(py_exec, platform)
    else:
        cuda_version = get_cuda(platform)
        has_rocm = get_rocm() if platform == "linux" and cuda_version is None else False
        cmd = generate_cmd(py_exec, platform, cuda_version, has_rocm)

    # run command
    run_install(cmd)


def interactive_install_paddle():
    """Ask the user for installing paddle."""
    try:
        notice = "Do you want to install the recommended backend Paddle (y/n): "
        msg = input(notice)
    except EOFError:
        msg = "n"

    cnt = 0
    while cnt < 3:
        if msg == "y":
            install_paddle()
            return
        if msg == "n":
            break
        cnt += 1
        msg = input("Please enter correctly (y/n): ")

    sys.exit(
        "No available backend found.\n"
        "Please manually install a backend, and run DeepXDE again."
    )
