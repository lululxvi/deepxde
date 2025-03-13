import brainstate as bst
import brainunit as u

import deepxde.experimental as deepxde


def pde(x, y):
    dy_xx = net.hessian(x)['y']['x']['x']
    return -dy_xx - u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'])


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)


def boundary_l(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x['x'], -1))


def boundary_r(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x['x'], 1))


def func(x):
    return {'y': u.math.sin(u.math.pi * x['x'])}


geom = deepxde.geometry.Interval(-1, 1).to_dict_point('x')
bc1 = deepxde.icbc.DirichletBC(func, boundary_l)
bc2 = deepxde.icbc.PeriodicBC('y', 'x', boundary_r)
data = deepxde.problem.PDE(
    geom, pde, [bc1, bc2], net,
    num_domain=16, num_boundary=2, solution=func, num_test=100
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
