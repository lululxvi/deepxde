"""
Poisson-like 2D problem
Backend supported: tensorflow.compat.v1
"""
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt


# Two target variables: A and B
# Equations: dA_xx = f, dB_tt = f
def equation(x, y, f):
    A = y[:, 0:1]
    B = y[:, 1:2]
    dA_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dB_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)
    return [dA_xx - f, dB_tt - f]


# Define space/time geometry
geomtime = dde.geometry.GeometryXTime(
    dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1)
)

# Boundary conditions for A and B
A_bc = dde.icbc.DirichletBC(
    geomtime,
    lambda _: 0,
    lambda _, on_boundary: on_boundary and np.isclose(_[0], 0),
    component=0,
)
B_bc = dde.icbc.DirichletBC(
    geomtime,
    lambda _: 0,
    lambda _, on_boundary: on_boundary and np.isclose(_[0], 1),
    component=1,
)

space = dde.data.GRF2D()
evaluation_points = geomtime.uniform_points(10)

data = dde.data.PDEOperatorCartesianProd(
    dde.data.TimePDE(
        geomtime, equation, [A_bc, B_bc], num_domain=1000, num_boundary=10
    ),
    space,
    evaluation_points,
    num_function=10,
)

# Define DeepONet with two outputs
net = dde.nn.DeepONetCartesianProd(
    [evaluation_points.shape[0], 100, 100],
    [geomtime.dim, 100, 100],
    activation="tanh",
    kernel_initializer="Glorot normal",
    num_outputs=2,
)

# Train model
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.train(iterations=5000)
