"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np

# Define function
if dde.backend.backend_name == "paddle":
    # Backend paddle
    import paddle

    sin = paddle.sin
elif dde.backend.backend_name == "pytorch":
    # Backend pytorch
    import torch

    sin = torch.sin
elif dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    # Backend tensorflow.compat.v1 or tensorflow
    from deepxde.backend import tf

    sin = tf.sin
elif dde.backend.backend_name == "jax":
    # Backend jax
    import jax.numpy as jnp

    sin = jnp.sin


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - np.pi**2 * sin(np.pi * x)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def func(x):
    return np.sin(np.pi * x)


geom = dde.geometry.Interval(-1, 1)
bc1 = dde.icbc.DirichletBC(geom, func, boundary_l)
bc2 = dde.icbc.PeriodicBC(geom, 0, boundary_r)
data = dde.data.PDE(geom, pde, [bc1, bc2], 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
