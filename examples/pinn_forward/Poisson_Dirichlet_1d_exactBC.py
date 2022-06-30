"""
Implementation of the 1D Poisson equation (Section 3.2.1) in paper https://arxiv.org/abs/2111.02801.
"""

import deepxde as dde
from deepxde.backend import tf
import numpy as np

geom = dde.geometry.Interval(0, np.pi)

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    summation = sum([i * tf.sin(i * x) for i in range(1, 5)])
    return -dy_xx - summation - 8 * tf.sin(8 * x)

def func(x):
    summation = sum([np.sin(i * x) / i for i in range(1, 5)])
    return x + summation + np.sin(8 * x) / 8

data = dde.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=250)

layer_size = [1] + [50] * 3 + [1]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

def  output_transform(x, y):
    return x * (np.pi - x) * y + x 

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, decay = ("inverse time", 1000, 0.3), metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=30000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
