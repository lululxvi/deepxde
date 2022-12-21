"""Backend supported: tensorflow.compat.v1, paddle

Implementation of the Poisson 1D example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np

A = 2
B = 50


# Define sine function
if dde.backend.backend_name in ["tensorflow.compat.v1", "tensorflow"]:
    from deepxde.backend import tf

    sin = tf.sin
elif dde.backend.backend_name == "paddle":
    import paddle

    sin = paddle.sin


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return (
        dy_xx
        + (np.pi * A) ** 2 * sin(np.pi * A * x)
        + 0.1 * (np.pi * B) ** 2 * sin(np.pi * B * x)
    )


def func(x):
    return np.sin(np.pi * A * x) + 0.1 * np.sin(np.pi * B * x)


geom = dde.geometry.Interval(0, 1)
bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(
    geom,
    pde,
    bc,
    1280,
    2,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

layer_size = [1] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])

model = dde.Model(data, net)
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    decay=("inverse time", 2000, 0.9),
)

pde_residual_resampler = dde.callbacks.PDEResidualResampler(period=1)
model.train(iterations=20000, callbacks=[pde_residual_resampler])

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True)
