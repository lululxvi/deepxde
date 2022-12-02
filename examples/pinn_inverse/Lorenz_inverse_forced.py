"""Backend supported: tensorflow.compat.v1, tensorflow, paddle

Identification of the parameters of the modified Lorenz attractor (with exogenous input)
see https://github.com/lululxvi/deepxde/issues/79
"""
import re

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint


# Generate data
# true values, see p. 15 in https://arxiv.org/abs/1907.04502
C1true = 10
C2true = 15
C3true = 8 / 3

# time points
maxtime = 3
time = np.linspace(0, maxtime, 200)
ex_input = 10 * np.sin(2 * np.pi * time)  # exogenous input

# interpolate time / lift vectors (for using exogenous variable without fixed time stamps)
def ex_func(t):
    spline = sp.interpolate.Rbf(
        time, ex_input, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t)


# Modified Lorenz system (with exogenous input)
def LorezODE(x, t):
    x1, x2, x3 = x
    dxdt = [
        C1true * (x2 - x1),
        x1 * (C2true - x3) - x2,
        x1 * x2 - C3true * x3 + ex_func(t),
    ]
    return dxdt


# initial condition
x0 = [-8, 7, 27]

# solve ODE
x = odeint(LorezODE, x0, time)

# plot results
plt.plot(time, x, time, ex_input)
plt.xlabel("time")
plt.ylabel("x(t)")
plt.show()

time = time.reshape(-1, 1)

# Perform identification
# parameters to be identified
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

# interpolate time / lift vectors (for using exogenous variable without fixed time stamps)
def ex_func2(t):
    spline = sp.interpolate.Rbf(
        time, ex_input, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t[:, 0:])


# define system ODEs
def Lorenz_system(x, y, ex):
    """Modified Lorenz system (with exogenous input).
    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (28 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3 + u
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3 - ex,
    ]


def boundary(_, on_initial):
    return on_initial


# define time domain
geom = dde.geometry.TimeDomain(0, maxtime)

# Initial conditions
ic1 = dde.icbc.IC(geom, lambda X: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: x0[2], boundary, component=2)

# Get the training data
observe_t, ob_y = time, x
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

# define data object
data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
    auxiliary_var_function=ex_func2,
)

plt.plot(observe_t, ob_y)
plt.xlabel("Time")
plt.legend(["x", "y", "z"])
plt.title("Training data")
plt.show()

# define FNN architecture and compile
net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3])

# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1, C2, C3], period=100, filename=fnamevar)

model.train(iterations=60000, callbacks=[variable])

# Plots
# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()
# read output data in fnamevar (this line is a long story...)
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = Chat.shape
plt.plot(range(l), Chat[:, 0], "r-")
plt.plot(range(l), Chat[:, 1], "k-")
plt.plot(range(l), Chat[:, 2], "g-")
plt.plot(range(l), np.ones(Chat[:, 0].shape) * C1true, "r--")
plt.plot(range(l), np.ones(Chat[:, 1].shape) * C2true, "k--")
plt.plot(range(l), np.ones(Chat[:, 2].shape) * C3true, "g--")
plt.legend(["C1hat", "C2hat", "C3hat", "True C1", "True C2", "True C3"], loc="right")
plt.xlabel("Epoch")

yhat = model.predict(observe_t)
plt.figure()
plt.plot(observe_t, ob_y, "-", observe_t, yhat, "--")
plt.xlabel("Time")
plt.legend(["x", "y", "z", "xh", "yh", "zh"])
plt.title("Training data")
plt.show()
