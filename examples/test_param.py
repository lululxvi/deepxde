import os

from deepxde.optimizers import set_LBFGS_options


def get_value(key, default_value, mapper):
    value = os.environ.get(key, default_value)
    if value is None:
        return None
    return mapper(value)


def get_steps(default_value=None):
    return get_value('DDE_STEPS_OVERRIDE', default_value, int)


def get_save_flag(default_value):
    return get_value('DDE_SAVE_RESULTS', default_value, int) > 0


set_LBFGS_options(maxiter=get_steps(15000))
