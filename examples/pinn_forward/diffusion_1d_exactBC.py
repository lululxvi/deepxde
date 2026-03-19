"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle
1D Diffusion Equation with a Time-Dependent Source Term.

This example solves the heat equation:
∂y/∂t - ∂²y/∂x² = f(x, t)
where the source term f(x, t) is chosen such that the analytical solution is y = e^(-t) * sin(πx).
"""
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Backend pytorch
# import torch
# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle


def pde(x, y):
    # Most backends
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
   
    # Physics Note: The following term is the forced heat source f(x, t)
    # required to satisfy the analytical solution y = e^(-t)sin(πx).
    source_term_val = dde.backend.exp(-x[:, 1:]) * (
        dde.backend.sin(np.pi * x[:, 0:1]) - np.pi**2 * dde.backend.sin(np.pi * x[:, 0:1])
    )
    # Backend tensorflow.compat.v1 or tensorflow, pytorch, jax, paddle
    return (
        dy_t
        - dy_xx
        + source_term_val
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(geomtime, pde, [], num_domain=40, solution=func, num_test=10000)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(
    # This works for TensorFlow, PyTorch, JAX, and Paddle
    lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + dde.backend.sin(np.pi * x[:, 0:1])
)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
