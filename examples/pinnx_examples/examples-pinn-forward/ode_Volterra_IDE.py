import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt

import numpy as np
from deepxde import pinnx


def ide(x, y, int_mat):
    jacobian = net.jacobian(x)['y']['x']
    y = y['y']
    rhs = u.math.matmul(int_mat, y)
    return (jacobian + y)[: len(rhs)] - rhs


def kernel(x, s):
    return np.exp(s - x)


def func(x):
    return {'y': u.math.exp(-x['x']) * u.math.cosh(x['x'])}


geom = pinnx.geometry.TimeDomain(0, 5).to_dict_point('x')
ic = pinnx.icbc.IC(func)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

data = pinnx.problem.IDE(
    geom,
    ide,
    ic,
    quad_deg=20,
    approximator=net,
    kernel=kernel,
    num_domain=10,
    num_boundary=2,
    train_distribution="uniform",
)

model = pinnx.Trainer(data)
model.compile(bst.optim.LBFGS(1e-3)).train(5000, display_every=200)

X = geom.uniform_points(100)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X['x'], y_true['y'], "-")
plt.plot(X['x'], y_pred['y'], "o")
plt.show()
