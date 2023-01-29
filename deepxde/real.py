import numpy as np

from . import backend as bkd


class Real:
    def __init__(self, precision):
        self.precision = None
        self.reals = None
        if precision == 16:
            self.set_float16()
        elif precision == 32:
            self.set_float32()
        elif precision == 64:
            self.set_float64()

    def __call__(self, package):
        return self.reals[package]

    def set_float16(self):
        self.precision = 16
        self.reals = {np: np.float16, bkd.lib: bkd.float16}

    def set_float32(self):
        self.precision = 32
        self.reals = {np: np.float32, bkd.lib: bkd.float32}

    def set_float64(self):
        self.precision = 64
        self.reals = {np: np.float64, bkd.lib: bkd.float64}
