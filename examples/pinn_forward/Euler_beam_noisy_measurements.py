"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + 1


def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


# Closed-form solution
def func(x):
    return -(x**4) / 24 + x**3 / 6 - x**2 / 4


# ---------------------------------------------------------
# Synthetic FEM / experimental measurements
# ---------------------------------------------------------

np.random.seed(42)

num_measurements = 10

# Random measurement locations
X_measurement = np.random.uniform(
    0, 1, (num_measurements, 1)
)

# Exact displacement at measurement locations
Y_exact = func(X_measurement)

# Simulated numerical/measurement error
noise_level = 0.005  # 0.5%

noise = np.random.normal(
    loc=0.0,
    scale=noise_level,
    size=Y_exact.shape
)

# FEM-like noisy measurements
Y_measurement = Y_exact * (1.0 + noise)


# Treat measurements as observed displacement data
measurement_bc = dde.icbc.PointSetBC(
    X_measurement,
    Y_measurement,
    component=0,
)


# ---------------------------------------------------------
# Geometry and boundary conditions
# ---------------------------------------------------------

geom = dde.geometry.Interval(0, 1)

bc1 = dde.icbc.DirichletBC(
    geom,
    lambda x: 0,
    boundary_l
)

bc2 = dde.icbc.NeumannBC(
    geom,
    lambda x: 0,
    boundary_l
)

bc3 = dde.icbc.OperatorBC(
    geom,
    lambda x, y, _: ddy(x, y),
    boundary_r
)

bc4 = dde.icbc.OperatorBC(
    geom,
    lambda x, y, _: dddy(x, y),
    boundary_r
)


# ---------------------------------------------------------
# PINN data
# ---------------------------------------------------------

data = dde.data.PDE(
    geom,
    pde,
    [
        bc1,
        bc2,
        bc3,
        bc4,
        measurement_bc,
    ],
    num_domain=10,
    num_boundary=2,
    solution=func,
    num_test=100,
)


# ---------------------------------------------------------
# Neural network
# ---------------------------------------------------------

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"

net = dde.nn.FNN(
    layer_size,
    activation,
    initializer
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

model = dde.Model(data, net)

model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"]
)

losshistory, train_state = model.train(
    iterations=10000
)

dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True
)