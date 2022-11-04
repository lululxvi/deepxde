"""Backend supported: tensorflow.compat.v1"""
import deepxde as dde
import numpy as np


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx - 2


def dy_x(x, y, X):
    dy_x = dde.grad.jacobian(y, x)
    return dy_x


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1)


def func(x):
    return (x + 1) ** 2


def d_func(x):
    return 2 * (x + 1)


geom = dde.geometry.Interval(-1, 1)
bc_l = dde.icbc.DirichletBC(geom, func, boundary_l)
boundary_pts = geom.random_boundary_points(2)
r_boundary_pts = boundary_pts[np.isclose(boundary_pts, 1)].reshape(-1, 1)
bc_r = dde.icbc.PointSetOperatorBC(r_boundary_pts, d_func(r_boundary_pts), dy_x)

data = dde.data.PDE(
    geom, pde, [bc_l, bc_r], num_domain=16, num_boundary=2, solution=func, num_test=100
)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)


def dy_x(x, y):
    dy_x = dde.grad.jacobian(y, x)
    return dy_x


def dy_xx(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx


# Print out first and second derivatives into a file during training on the boundary points
first_derivative = dde.callbacks.OperatorPredictor(
    geom.random_boundary_points(2), op=dy_x, period=200, filename="first_derivative.txt"
)
second_derivative = dde.callbacks.OperatorPredictor(
    geom.random_boundary_points(2),
    op=dy_xx,
    period=200,
    filename="second_derivative.txt",
)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(
    iterations=10000, callbacks=[first_derivative, second_derivative]
)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
