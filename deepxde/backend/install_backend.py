import os
import sys


def exit_install(state=0, arg=None):
    """Exit running.

    Args:
        state (int): 0, 1, 2 or 3
            0: install failed
            1: error platform
            2: no avx
            3: no avaliable backend
    """
    msg = "Paddle installed failed!\n"
    if state == 0:
        msg += (
            "Reasons might be python version/pip version/requirements/avx/GPU model... "
            "You can visit https://www.paddlepaddle.org.cn for help and install it manually.\n"
        )
    elif state == 1:
        msg += (
            "your system is",
            arg,
            ", paddle only supported Window/Linux/macOS now, sorry.\n",
        )
    elif state == 2:
        msg += (
            "avx is not supported by your machine. "
            "Sorry, paddle develop is not supported in noavx machine now."
        )
    elif state == 3:
        msg += (
            "No available backend found, you can manually install one of "
            "paddle/tensorflow.compat.va/tensorflow/pytorch/jax.\n"
        )
    msg += (
        "You can run deepxde again and select backend by 'DDE_BACKEND=xxx'.\n"
        "Running stopped!"
    )
    sys.exit(msg)


def run_install(command):
    """Send command to terminal and print it.

    Args:
        command (str): command to be sent to terminal.
    """
    print("Install command:", command, "\n")
    installed = os.system(command)
    if installed == 0:
        print("Paddle installed successfully!\n", file=sys.stderr, flush=True)
    else:
        exit_install(0)


def get_platform():
    """Get user's platform.

    Returns:
        platform (str): "windows", "linux" or "darwin"
    """
    platform = sys.platform
    if platform in ["win32", "cygwin"]:
        platform = "windows"
    elif platform in ["linux", "linux2"]:
        platform = "linux"
    elif platform == "darwin":
        pass
    else:
        exit_install(1, platform)

    return platform


def check_cuda(platform):
    """Check whether cuda is avaliable and its version.

    Returns:
        cuda_verion (str) or None
    """
    if platform == "linux":
        cuda_list = [101, 102, 110, 111, 112, 116, 117, 118]
    elif platform == "windows":
        cuda_list = [101, 102, 110, 111, 112, 113, 114, 115, 116, 117, 118]
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
                f"Your Cuda version is {cuda_version},",
                f"however paddle only supports Cuda {msg_cl} for {platform} now.\n",
                file=sys.stderr,
                flush=True,
            )
        else:
            return cuda_version

    return None


def check_rocm():
    """Check whether ROCm4.0 is avaliable.

    Returns:
        bool
    """
    roc_text1 = os.popen("/opt/rocm/bin/rocminfo").read()
    roc_text2 = os.popen("/opt/rocm/opencl/bin/clinfo").read()
    if roc_text1 != "" and roc_text2 != "":
        return True

    print("There is no avaliable ROCm4.0.\n", file=sys.stderr, flush=True)
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
        exit_install(2)


def check_executable():
    """Check user's python version.

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
            "Paddle only can be installed in macOS with cpu by pip now, ",
            "installing cpu version...",
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
        print(f"Installing cuda {cuda_version} version...", file=sys.stderr, flush=True)
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

    print("Installing cpu version...", file=sys.stderr, flush=True)
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
    py_exec = check_executable()

    # get user's device and generate cmd
    if platform == "darwin":
        cmd = generate_cmd(py_exec, platform)
    else:
        cuda_version = check_cuda(platform)
        has_rocm = check_rocm() if cuda_version is None else False

        cmd = generate_cmd(py_exec, platform, cuda_version, has_rocm)

    # run command
    run_install(cmd)


def interactive_install_paddle():
    """Ask the user for installing paddle."""

    try:
        notice = "{}{}".format(
            "No available backend found. ",
            "Do you want to install the recommended backend paddle (y/n): ",
        )
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

    exit_install(3)
