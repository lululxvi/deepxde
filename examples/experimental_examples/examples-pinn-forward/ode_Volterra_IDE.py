import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt

import numpy as np
import deepxde.experimental as deepxde


def ide(x, y, int_mat):
    jacobian = net.jacobian(x)['y']['x']
    y = y['y']
    rhs = u.math.matmul(int_mat, y)
    return (jacobian + y)[: len(rhs)] - rhs


def kernel(x, s):
    return np.exp(s - x)


def func(x):
    return {'y': u.math.exp(-x['x']) * u.math.cosh(x['x'])}


geom = deepxde.geometry.TimeDomain(0, 5).to_dict_point('x')
ic = deepxde.icbc.IC(func)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN([1] + [20] * 3 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)

data = deepxde.problem.IDE(
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

model = deepxde.Trainer(data)
model.compile(bst.optim.LBFGS(1e-3)).train(5000, display_every=200)

X = geom.uniform_points(100)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", deepxde.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X['x'], y_true['y'], "-")
plt.plot(X['x'], y_pred['y'], "o")
plt.show()
