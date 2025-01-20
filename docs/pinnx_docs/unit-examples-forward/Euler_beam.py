import brainstate as bst
import brainunit as u

from deepxde import pinnx

unit_of_u = u.meter
unit_of_x = u.meter
unit_of_E = u.pascal
unit_of_I = u.meter ** 4
unit_of_p = u.kilogram / u.second ** 2

geom = pinnx.geometry.Interval(0, 1).to_dict_point(x=unit_of_x)

E = 1 * unit_of_E
I = 1 * unit_of_I
p = -1. * unit_of_p


def pde(x, y):
    dy_xxxx = net.gradient(x, order=4)['y']['x']['x']['x']['x']
    return E * I * dy_xxxx - p


def boundary_l(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'] / unit_of_x, 0))


def boundary_r(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'] / unit_of_x, 1))


bc1 = pinnx.icbc.DirichletBC(lambda x: {'y': 0 * unit_of_u}, boundary_l)
bc2 = pinnx.icbc.NeumannBC(lambda x: {'y': 0 * unit_of_u}, boundary_l)
bc3 = pinnx.icbc.OperatorBC(lambda x, y: net.hessian(x)['y']['x']['x'] / u.meter, boundary_r)
bc4 = pinnx.icbc.OperatorBC(lambda x, y: net.gradient(x, order=3)['y']['x']['x']['x'] / u.meter ** 2, boundary_r)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=unit_of_x),
    pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=unit_of_u),
)


def func(x):
    x = x['x'] / unit_of_x
    y = -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4
    return {'y': y * unit_of_u}


data = pinnx.problem.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    net,
    num_domain=100,
    num_boundary=20,
    solution=func,
    num_test=100,
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
