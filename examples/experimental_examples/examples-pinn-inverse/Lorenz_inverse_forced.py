"""

Identification of the parameters of the modified Lorenz attractor (with exogenous input)
"""

import re

import brainstate as bst
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import odeint

import jax
import deepxde.experimental as deepxde

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

# Perform identification
# parameters to be identified
C1 = bst.ParamState(1.0)
C2 = bst.ParamState(1.0)
C3 = bst.ParamState(1.0)


# interpolate time / lift vectors (for using exogenous variable without fixed time stamps)
spline = sp.interpolate.Rbf(time, ex_input, function="thin_plate", smooth=0, episilon=0)


# define system ODEs
def Lorenz_system(x, y):
    """
    Modified Lorenz system (with exogenous input).
    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (28 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3 + u
    """
    y1, y2, y3 = y["y1"], y["y2"], y["y3"]
    jacobian = net.jacobian(x)
    dy1_x = jacobian["y1"]["t"]
    dy2_x = jacobian["y2"]["t"]
    dy3_x = jacobian["y3"]["t"]

    ex = jax.pure_callback(
        spline, jax.ShapeDtypeStruct(x["t"].shape, x["t"].dtype), x["t"]
    )
    return [
        dy1_x - C1.value * (y2 - y1),
        dy2_x - y1 * (C2.value - y3) + y2,
        dy3_x - y1 * y2 + C3.value * y3 - ex,
    ]


# define FNN architecture and compile
net = deepxde.nn.Model(
    deepxde.nn.DictToArray(t=None),
    deepxde.nn.FNN([1] + [40] * 3 + [3], "tanh"),
    deepxde.nn.ArrayToDict(y1=None, y2=None, y3=None),
)

# define time domain
geom = deepxde.geometry.TimeDomain(0, maxtime).to_dict_point("t")

# Initial conditions
ic = deepxde.icbc.IC(lambda x: {"y1": x0[0], "y2": x0[1], "y3": x0[2]})

# Get the training data
ob_y = x
observe_t = {"t": time}
observe_y = {"y1": ob_y[:, 0], "y2": ob_y[:, 1], "y3": ob_y[:, 2]}
bc = deepxde.icbc.PointSetBC(observe_t, observe_y)

# define data object
data = deepxde.problem.PDE(
    geom,
    Lorenz_system,
    [ic, bc],
    net,
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

plt.plot(time, ob_y)
plt.xlabel("Time")
plt.legend(["x", "y", "z"])
plt.title("Training data")
plt.show()

# callbacks for storing results
fnamevar = "variables.dat"
variable = deepxde.callbacks.VariableValue([C1, C2, C3], period=100, filename=fnamevar)

# train the model
trainer = deepxde.Trainer(data, external_trainable_variables=[C1, C2, C3])
trainer.compile(bst.optim.Adam(0.001)).train(iterations=60000, callbacks=[variable])

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

yhat = trainer.predict(observe_t)
yhat = deepxde.utils.dict_to_array(yhat)
plt.figure()
plt.plot(observe_t["t"], ob_y, "-", observe_t["t"], yhat, "--")
plt.xlabel("Time")
plt.legend(["x", "y", "z", "xh", "yh", "zh"])
plt.title("Training data")
plt.show()
