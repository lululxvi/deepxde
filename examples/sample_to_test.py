"""
Transform an example script into an integration test, by:
- reducing the number of iterations to a minimum
- disabling interactive plots

Usage: python sample_to_test.py input_file.py
The transformed code is written to stdout.
"""

import re
import sys

import deepxde as dde


ITERATIONS_RE = re.compile(r"^(.*)iterations\s*=\s*(\d+)(.*)$", re.DOTALL)

PROLOG = """
from deepxde.optimizers import set_LBFGS_options
set_LBFGS_options(maxiter=1)

import matplotlib
matplotlib.use('template')
"""


def transform(line, file_name):
    """Apply transformations to line."""
    m = re.match(ITERATIONS_RE, line)
    if m is not None:
        line = m.expand(r"\1iterations=1\3")

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
        if file_name not in (
            # this example uses skopt which is not maintained
            # and uses deprecated numpy API
            "Allen_Cahn.py",
            # the below examples have different code for different
            # backends
            "Beltrami_flow.py",
            "Helmholtz_Dirichlet_2d_HPO.py",
            "Laplace_disk.py",
            "Lotka_Volterra.py",
            "Poisson_Dirichlet_1d.py",
            "Poisson_periodic_1d.py",
            "diffusion_1d_exactBC.py",
            "diffusion_1d_resample.py",
            "diffusion_1d.py",
            "diffusion_reaction.py",
            "fractional_diffusion_1d.py",
            "fractional_Poisson_1d.py",
            "fractional_Poisson_2d.py",
            "fractional_Poisson_3d.py",
            "ide.py",
            "diffusion_1d_inverse.py",
            "fractional_Poisson_1d_inverse.py",
            "fractional_Poisson_2d_inverse.py",
            "Poisson_Lshape.py",
            "ode_system.py",
            "Lorenz_inverse.py",
            # gives error with tensorflow.compat.v1
            "Volterra_IDE.py",
            # the dataset is large and not included in repo
            "antiderivative_unaligned.py",
            "antiderivative_aligned.py",
        ):
            lines = input.readlines()
            if dde.backend.get_preferred_backend() in lines[0].replace(",", "").replace(
                '"""', ""
            ).replace("\n", "").split(" "):
                for line in lines:
                    sys.stdout.write(transform(line, file_name))
