"""
Transform an example script into an integration test, by:
- reducing the number of iterations to a minimum
- disabling interactive plots

Usage: python sample_to_test.py input_file.py
The transformed code is written to stdout.
"""

import re
import sys


EPOCHS_RE = re.compile(r"^(.*)epochs\s*=\s*(\d+)(.*)$", re.DOTALL)

PROLOG = """
from deepxde.optimizers import set_LBFGS_options
set_LBFGS_options(maxiter=1)

import matplotlib
matplotlib.use('template')
"""


def transform(line, file_name):
    """Apply transformations to line."""
    m = re.match(EPOCHS_RE, line)
    if m is not None:
        line = m.expand(r"\1epochs=1\3")

    # Burgers_RAR.py has an additional convergence loop: force 1 single pass
    if file_name == "Burgers_RAR.py" and line.startswith("while"):
        line = (
            "first_iteration = True\n"
            + line[:-2]
            + " and first_iteration:\n"
            + "    first_iteration = False\n"
        )

    return line


if __name__ == "__main__":
    file_name = sys.argv[1]

    print(PROLOG)
    with open(file_name, "r") as input:
        for line in input:
            sys.stdout.write(transform(line, file_name))
