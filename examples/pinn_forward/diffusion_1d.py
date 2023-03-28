"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np

# Define function
if dde.backend.backend_name == "paddle":
    # Backend paddle
    import paddle

    sin = paddle.sin
    exp = paddle.exp
elif dde.backend.backend_name == "pytorch":
    # Backend pytorch
    import torch

    sin = torch.sin
    exp = torch.exp
elif dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    # Backend tensorflow.compat.v1 or tensorflow
    from deepxde.backend import tf

    sin = tf.sin
    exp = tf.exp
elif dde.backend.backend_name == "jax":
    # Backend jax
    import jax.numpy as jnp

    sin = jnp.sin
    exp = jnp.exp


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    # Backend tensorflow.compat.v1 or tensorflow
    return (
        dy_t
        - dy_xx
        + exp(-x[:, 1:])
        * (sin(np.pi * x[:, 0:1]) - np.pi**2 * sin(np.pi * x[:, 0:1]))
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
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
