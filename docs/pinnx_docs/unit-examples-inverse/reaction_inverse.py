import brainstate as bst
import brainunit as u
import numpy as np

import deepxde.experimental as deepxde

unit_of_x = u.meter
unit_of_t = u.second
unit_of_c = u.mole / u.meter ** 3

kf = bst.ParamState(0.05 * u.meter ** 6 / u.mole ** 2 / u.second)
D = bst.ParamState(1.0 * u.meter ** 2 / u.second)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x)
    ca, cb = y['ca'], y['cb']
    dca_t = jacobian['ca']['t']
    dcb_t = jacobian['cb']['t']
    dca_xx = hessian['ca']['x']['x']
    dcb_xx = hessian['cb']['x']['x']
    eq_a = dca_t - 1e-3 * D.value * dca_xx + kf.value * ca * cb ** 2
    eq_b = dcb_t - 1e-3 * D.value * dcb_xx + 2 * kf.value * ca * cb ** 2
    return [eq_a, eq_b]


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=unit_of_x, t=unit_of_t),
    deepxde.nn.FNN([2] + [20] * 3 + [2], "tanh"),
    deepxde.nn.ArrayToDict(ca=unit_of_c, cb=unit_of_c),
)

geom = deepxde.geometry.Interval(0, 1)
timedomain = deepxde.geometry.TimeDomain(0, 10)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point(x=unit_of_x, t=unit_of_t)


def fun_bc(x):
    c = (1 - x['x'] / unit_of_x) * unit_of_c
    return {'ca': c, 'cb': c}


bc = deepxde.icbc.DirichletBC(fun_bc)


def fun_init(x):
    return {
        'ca': u.math.exp(-20 * x['x'] / unit_of_x) * unit_of_c,
        'cb': u.math.exp(-20 * x['x'] / unit_of_x) * unit_of_c,
    }


ic = deepxde.icbc.IC(fun_init)


def gen_traindata():
    data = np.load("../dataset/reaction.npz")
    t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
    X, T = np.meshgrid(x, t)
    x = {'x': X.flatten() * unit_of_x, 't': T.flatten() * unit_of_t}
    y = {'ca': ca.flatten() * unit_of_c, 'cb': cb.flatten() * unit_of_c}
    return x, y


observe_x, observe_y = gen_traindata()
observe_bc = deepxde.icbc.PointSetBC(observe_x, observe_y)

data = deepxde.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_bc],
    net,
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=50000,
)

variable = deepxde.callbacks.VariableValue([kf, D], period=1000, filename="./variables.dat")
model = deepxde.Trainer(data, external_trainable_variables=[kf, D])
model.compile(bst.optim.Adam(0.001)).train(iterations=80000, callbacks=[variable])
model.saveplot(issave=True, isplot=True)
