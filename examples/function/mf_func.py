"""Backend supported: tensorflow.compat.v1"""
from test_param import *

import deepxde as dde
import numpy as np


train_steps = get_steps(80000)
report_flag = get_save_flag(1)


def func_lo(x):
    A, B, C = 0.5, 10, -5
    return A * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5) + C


def func_hi(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


geom = dde.geometry.Interval(0, 1)
num_test = 1000
data = dde.data.MfFunc(geom, func_lo, func_hi, 100, 6, num_test)

activation = "tanh"
initializer = "Glorot uniform"
regularization = ["l2", 0.01]
net = dde.nn.MfNN(
    [1] + [20] * 4 + [1],
    [10] * 2 + [1],
    activation,
    initializer,
    regularization=regularization,
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=train_steps)

dde.saveplot(losshistory, train_state, issave=report_flag, isplot=report_flag)
