"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch

Implementation of Brinkman-Forchheimer equation example in paper https://arxiv.org/pdf/2111.02801.pdf.
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import re

g = 1
v = 1e-3
e = 0.4
K = 1e-3
H = 1

v_e = dde.Variable(0.1)


def sol(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def pde(x, y):
    u = y
    du_xx = dde.grad.hessian(y, x)
    return -v_e / e * du_xx + v * u / K - g


def output_transform(x, y):
    return x * (1 - x) * y


geom = dde.geometry.Interval(0, 1)
ob_x, ob_u = gen_traindata(5)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

data = dde.data.PDE(
    geom,
    pde,
    solution=sol,
    bcs=[observe_u],
    num_domain=100,
    num_boundary=0,
    train_distribution="uniform",
    num_test=500,
)

net = dde.maps.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
variable = dde.callbacks.VariableValue([v_e], period=200, filename="variables1.dat")

losshistory, train_state = model.train(epochs=30000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

lines = open("variables1.dat", "r").readlines()
v_ehat1 = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = v_ehat1.shape
v_etrue = 1e-3

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(v_ehat1[:, 0].shape) * v_etrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), v_ehat1[:, 0], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(top=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$\nu_e$")

plt.show()
