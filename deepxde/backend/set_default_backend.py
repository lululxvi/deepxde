import argparse
import json
import os


def set_default_backend(backend_name):
    default_dir = os.path.join(os.path.expanduser("~"), ".deepxde")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump({"backend": backend_name.lower()}, config_file)
    print(
        'Setting the default backend to "{}". You can change it in the '
        "~/.deepxde/config.json file or export the DDE_BACKEND environment variable. "
        "Valid options are: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle (all lowercase)".format(
            backend_name
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        nargs=1,
        type=str,
        choices=["tensorflow.compat.v1", "tensorflow", "pytorch", "jax", "paddle"],
        help="Set default backend",
    )
    args = parser.parse_args()
    set_default_backend(args.backend[0])
