import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import deepxde.experimental as deepxde

ub = 200
rb = 20


def ode_system(x, y):
    jacobian = net.jacobian(x)
    r = y['r']
    p = y['p']
    dr_t = jacobian['r']['t']
    dp_t = jacobian['p']['t']
    return [
        dr_t - 1 / ub * rb * (2.0 * ub * r - 0.04 * ub * r * ub * p),
        dp_t - 1 / ub * rb * (0.02 * r * ub * p * ub - 1.06 * p * ub),
    ]


def input_transform(t):
    return u.math.concatenate(
        (
            t,
            u.math.sin(t),
            u.math.sin(2 * t),
            u.math.sin(3 * t),
            u.math.sin(4 * t),
            u.math.sin(5 * t),
            u.math.sin(6 * t),
        ),
        axis=-1,
    )


def output_transform(t, y):
    # hard constraints: x(0) = 100, y(0) = 15
    y1 = y[..., 0:1]
    y2 = y[..., 1:2]
    return u.math.concatenate(
        [y1 * u.math.tanh(t) + 100 / ub,
         y2 * u.math.tanh(t) + 15 / ub],
        axis=-1
    )


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(t=None),
    deepxde.nn.FNN(
        [7] + [64] * 6 + [2], "tanh",
        input_transform=input_transform,
        output_transform=output_transform
    ),
    deepxde.nn.ArrayToDict(r=None, p=None),
)

geom = deepxde.geometry.TimeDomain(0, 1.0).to_dict_point('t')
problem = deepxde.problem.PDE(
    geom,
    ode_system,
    [],
    net,
    num_domain=3000,
    num_boundary=2,
    num_test=3000
)

trainer = deepxde.Trainer(problem)
trainer.compile(bst.optim.Adam(0.001)).train(iterations=50000)
trainer.compile(bst.optim.LBFGS(1e-3)).train(1000)
trainer.saveplot(issave=True, isplot=True)


def func(t, r):
    x, y = r
    dx_t = 1 / ub * rb * (2.0 * ub * x - 0.04 * ub * x * ub * y)
    dy_t = 1 / ub * rb * (0.02 * ub * x * ub * y - 1.06 * ub * y)
    return dx_t, dy_t


def gen_truedata():
    t = np.linspace(0, 1, 100)

    sol = integrate.solve_ivp(func, (0, 10), (100 / ub, 15 / ub), t_eval=t)
    x_true, y_true = sol.y
    x_true = x_true.reshape(100, 1)
    y_true = y_true.reshape(100, 1)

    return x_true, y_true


t = np.linspace(0, 1, 100)
x_true, y_true = gen_truedata()
plt.plot(t, x_true, color="black", label="x_true")
plt.plot(t, y_true, color="blue", label="y_true")

sol_pred = trainer.predict({'t': t})
x_pred = sol_pred['r']
y_pred = sol_pred['p']

plt.plot(t, x_pred, color="red", linestyle="dashed", label="x_pred")
plt.plot(t, y_pred, color="orange", linestyle="dashed", label="y_pred")
plt.legend()

plt.xlabel("t")
plt.ylabel("population")

plt.show()
