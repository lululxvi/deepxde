"""
Implementation of Brinkman-Forchheimer equation example in paper https://arxiv.org/pdf/2111.02801.pdf.
"""

import brainstate as bst
import brainunit as u
import numpy as np

from deepxde import pinnx

g = 1 * u.meter / u.second ** 2
v = 1e-3 * u.meter ** 2 / u.second
e = 0.4
H = 1 * u.meter


v_e = bst.ParamState(0.1 * u.meter ** 2 / u.second)
K = bst.ParamState(0.001 * u.meter ** 2)


def sol(x):
    x = x['x'].mantissa
    r = (v.mantissa * e / (1e-3 * 1e-3)) ** 0.5
    y = (
        g.mantissa * 1e-3 /
        v.mantissa *
        (1 - u.math.cosh(r * (x - H.mantissa / 2)) /
         u.math.cosh(r * H.mantissa / 2))
    )
    return {'y': y * u.meter / u.second}

geom = pinnx.geometry.Interval(0, 1).to_dict_point(x=u.meter)


def pde(x, y):
    du_xx = net.hessian(x)["y"]["x"]["x"]
    return -v_e.value / e * du_xx + v * y['y'] / K.value - g


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    x = {'x': xvals * u.meter}
    return x, sol(x)

ob_x, ob_u = gen_traindata(5)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=u.meter),
    pinnx.nn.FNN(
        [1] + [20] * 3 + [1], "tanh",
        output_transform=lambda x, y: x * (1 - x) * y
    ),
    pinnx.nn.ArrayToDict(y=u.meter / u.second),
)

problem = pinnx.problem.PDE(
    geom,
    pde,
    approximator=net,
    solution=sol,
    constraints=[observe_u],
    num_domain=100,
    num_boundary=0,
    train_distribution="uniform",
    num_test=500,
)

variable = pinnx.callbacks.VariableValue([v_e, K], period=200, filename="./variables1.dat")
trainer = pinnx.Trainer(problem, external_trainable_variables=[v_e, K])
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=30000, callbacks=[variable])
trainer.saveplot(issave=True, isplot=True)
