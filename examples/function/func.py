"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
from test_param import *

import deepxde as dde
import numpy as np


train_steps = get_steps(10000)
report_flag = get_save_flag(1)


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return x * np.sin(5 * x)


geom = dde.geometry.Interval(-1, 1)
num_train = 16
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=train_steps)

dde.saveplot(losshistory, train_state, issave=report_flag, isplot=report_flag)
