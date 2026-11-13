import brainstate as bst
import brainunit as u

import deepxde.experimental as deepxde

unit_of_x = u.meter
unit_of_t = u.second
unit_of_f = 1 / u.second

C = bst.ParamState(2.0 * u.meter ** 2 / u.second)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x, xi='x', xj='x')

    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    source = (
        u.math.exp(-x['t'] / unit_of_t) *
        (u.math.sin(u.math.pi * x['x'] / unit_of_x) -
         u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'] / unit_of_x))
    )

    return dy_t - C.value * dy_xx + source * unit_of_f


geom = deepxde.geometry.Interval(-1, 1)
timedomain = deepxde.geometry.TimeDomain(0, 1)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain).to_dict_point(x=unit_of_x, t=unit_of_t)


def func(x):
    y = u.math.sin(u.math.pi * x['x'] / unit_of_x) * u.math.exp(-x['t'] / unit_of_t)
    return {'y': y}


bc = deepxde.icbc.DirichletBC(func)
ic = deepxde.icbc.IC(func)

x = {
    'x': u.math.linspace(-1, 1, num=10) * unit_of_x,
    't': u.math.full((10,), 1) * unit_of_t,
}
observe_y = deepxde.icbc.PointSetBC(x, func(x))

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=unit_of_x, t=unit_of_t),
    deepxde.nn.FNN([2] + [32] * 3 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)

problem = deepxde.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    net,
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=x,
    solution=func,
    num_test=10000,
)

variable = deepxde.callbacks.VariableValue(C, period=1000)
trainer = deepxde.Trainer(problem, external_trainable_variables=C)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=50000, callbacks=[variable])
trainer.saveplot(issave=True, isplot=True)
