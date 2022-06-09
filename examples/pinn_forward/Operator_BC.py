"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t)
    d2y_dt2 = dde.grad.hessian(y, t)
    return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t

def func(t):
    return 50/81 + t * 5/9  - 2 * np.exp(t) + (31/81) * np.exp(9 * t)

geom = dde.geometry.TimeDomain(0, 0.25)

def boundary_l(t, on_boundry):
    return on_boundry and np.isclose(t[0], 0)

def bc_func1(inputs, outputs, X):
    return outputs + 1

def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2

bc1 = dde.icbc.OperatorBC(geom, bc_func1, boundary_l)
bc2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)
#bc = dde.icbc.DirichletBC(geomtime, func, boundary)

data = dde.data.TimePDE(
    geom, ode, [bc1, bc2], 16, 2, solution=func, num_test=100
)
layer_size = [1] + [50]*3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

# model.compile("adam", lr=.01, metrics=["l2 relative error"])
# losshistory, train_state = model.train(epochs=30000)
model.compile("adam", lr=.001, metrics=["l2 relative error"], loss_weights=[0.01,1,1])
losshistory, train_state = model.train(epochs=15000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
