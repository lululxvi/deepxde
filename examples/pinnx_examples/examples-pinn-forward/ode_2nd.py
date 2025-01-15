import brainstate as bst
import brainunit as u
import numpy as np

from deepxde import pinnx


def ode(x, y):
    dy_dt = net.jacobian(x)['y']['t']
    d2y_dt2 = net.hessian(x)['y']['t']['t']
    return d2y_dt2 - 10 * dy_dt + 9 * y['y'] - 5 * x['t']


def func(x):
    t = x['t']
    y = 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)
    return {'y': y}


geom = pinnx.geometry.TimeDomain(0, 0.25).to_dict_point("t")


def boundary_l(x, on_initial):
    return u.math.logical_and(on_initial, pinnx.utils.isclose(x['t'], 0))


def bc_func(inputs, outputs):
    return {'y': net.jacobian(inputs)['y']['t'] - 2}


ic1 = pinnx.icbc.IC(lambda x: {'y': -1})
ic2 = pinnx.icbc.OperatorBC(bc_func, boundary_l)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(t=None),
    pinnx.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

data = pinnx.problem.TimePDE(
    geom,
    ode,
    [ic1, ic2],
    approximator=net,
    num_domain=16,
    num_boundary=2,
    solution=func,
    num_test=500,
    loss_weights=[0.01, 1, 1]
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
