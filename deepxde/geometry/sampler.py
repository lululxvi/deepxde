from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skopt


def sample(n_samples, dimension, sampler="pseudo"):
    """Generate random or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin hypercube sampling), "Halton"
            (Halton sequence), "Hammersley" (Hammersley sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudo(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampler is not available.")


def pseudo(n_samples, dimension):
    """Pseudo random."""
    rng = np.random.default_rng()
    return rng.random((n_samples, dimension))


def quasirandom(n_samples, dimension, sampler):
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which are too special and may cause
        # some error.
        sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
    space = [(0.0, 1.0)] * dimension
    return np.array(sampler.generate(space, n_samples))
