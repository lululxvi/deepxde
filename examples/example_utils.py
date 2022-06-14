"""
Utility functions for the example programs.
The behaviour depends on whether the DDE_INTEGRATION_TEST variable is set; in that case,
training runs with a very little number of iterations/epochs, and plotting & saving of
results should be disabled.
"""

import os

from deepxde.optimizers import set_LBFGS_options


def is_in_integration_test():
    """
    Returns True if the code should run as an integration test
    """
    return os.environ.get('DDE_INTEGRATION_TEST') is not None


def is_interactive():
    """
    Returns True if the code is run interactively (not as an integration test)
    """
    return not is_in_integration_test()


def get_number_of_steps(default_value=None):
    """
    Overrides the number of steps if in an integration test
    """
    return 1 if is_in_integration_test() else default_value


# For external optimizers, the model.train() API does not let us override the number of epochs,
# but the number of iterations can be set apriori
set_LBFGS_options(maxiter=get_number_of_steps(15000))
