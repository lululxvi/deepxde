import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from deepxde import pinnx


def ide(x, y, int_mat):
    """int_0^x y(t)dt"""
    lhs2 = net.jacobian(x)['y']['x']
    lhs2 = u.math.squeeze(lhs2)
    lhs1 = int_mat @ y['y']
    lhs1 = u.math.squeeze(lhs1)
    rhs = 2 * np.pi * u.math.cos(2 * np.pi * x['x']) + u.math.sin(np.pi * x['x']) ** 2 / np.pi
    rhs = u.math.squeeze(rhs)
    return lhs1 + (lhs2 - rhs)[: len(lhs1)]


def func(x):
    return {'y': u.math.sin(2 * u.math.pi * x['x'])}


geom = pinnx.geometry.TimeDomain(0, 1).to_dict_point('x')
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
    quad_deg=16,
    approximator=net,
    num_domain=16,
    num_boundary=2
)


trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001)).train(iterations=10000)

X = geom.uniform_points(100, True)
y_true = func(X)
y_pred = trainer.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X['x'], y_true['y'], "-")
plt.plot(X['x'], y_pred['y'], "o")
plt.show()
