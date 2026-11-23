"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle
1D Diffusion Equation with Adaptive Point Resampling.

This example solves the heat equation using the PDEPointResampler callback:
∂y/∂t - ∂²y/∂x² = f(x, t)
Analytical solution: y = e^(-t) * sin(πx).
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

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

# Adaptive Resampling Callback
# Period=100 means every 100 iterations, the model redistributes 
# the domain points to where the PDE residual is highest.
resampler = dde.callbacks.PDEPointResampler(period=100)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=2000, callbacks=[resampler])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
