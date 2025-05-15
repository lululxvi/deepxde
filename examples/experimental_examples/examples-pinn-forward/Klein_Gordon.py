"""
We will solve a Klein-Gordon equation:

$$
\frac{\partial^2 y}{\partial t^2}+\alpha \frac{\partial^2 y}{\partial x^2}+\beta y+\gamma y^k=-x \cos (t)+x^2 \cos ^2(t), \quad x \in[-1,1], \quad t \in[0,10]
$$

with initial conditions

$$
y(x, 0)=x, \quad \frac{\partial y}{\partial t}(x, 0)=0
$$

and Dirichlet boundary conditions

$$
y(-1, t)=-\cos (t), \quad y(1, t)=\cos (t)
$$


We also specify the following parameters for the equation:

$$
\alpha=-1, \beta=0, \gamma=1, k=2 .
$$


The reference solution is $y(x, t)=x \cos (t)$.


"""

import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(-1, 1)
timedomain = deepxde.geometry.TimeDomain(0, 10)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point("x", "t")

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, t=None),
    deepxde.nn.FNN([2] + [40] * 2 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)

alpha, beta, gamma = -1, 0, 1


def pde(x, y):
    hessian = net.hessian(x)
    dy_tt = hessian["y"]["t"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    x, t = x["x"], x["t"]
    y = y["y"]
    return (
        dy_tt
        + alpha * dy_xx
        + beta * y
        + gamma * (y**2)
        + x * u.math.cos(t)
        - (x**2) * (u.math.cos(t) ** 2)
    )


def func(x):
    return {"y": x["x"] * u.math.cos(x["t"])}


bc = deepxde.icbc.DirichletBC(func)
ic_1 = deepxde.icbc.IC(func)
ic_2 = deepxde.icbc.OperatorBC(lambda x, y: {"y": net.jacobian(x)["y"]["t"]})
data = deepxde.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2],
    net,
    num_domain=30000,
    num_boundary=1500,
    num_initial=1500,
    solution=func,
    num_test=6000,
)

model = deepxde.Trainer(data)
model.compile(
    bst.optim.Adam(bst.optim.InverseTimeDecayLR(1e-3, 3000, 0.9)),
    metrics=["l2 relative error"],
).train(iterations=20000)
model.compile(bst.optim.LBFGS(1e-3), metrics=["l2 relative error"]).train(
    2000, display_every=200
)

model.saveplot(issave=True, isplot=True)

x = np.linspace(-1, 1, 256)
t = np.linspace(0, 10, 256)
X, T = np.meshgrid(x, t)

X_star = dict(x=np.ravel(X), t=np.ravel(T))
prediction = model.predict(X_star)

v = griddata(
    np.stack((X_star["x"], X_star["t"]), axis=-1),
    prediction["y"],
    (X, T),
    method="cubic",
)

fig, ax = plt.subplots()
ax.set_title("Results")
ax.set_ylabel("Prediction")
ax.imshow(
    v.T,
    interpolation="nearest",
    cmap="viridis",
    extent=(0, 10, -1, 1),
    origin="lower",
    aspect="auto",
)
plt.show()
