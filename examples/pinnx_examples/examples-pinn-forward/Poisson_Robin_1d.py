import brainstate as bst
import brainunit as u

from deepxde import pinnx


def pde(x, y):
    dy_xx = net.hessian(x)['y']['x']['x']
    return dy_xx - 2


def boundary_l(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'], -1))


def boundary_r(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'], 1))


def func(x):
    return {'y': (x['x'] + 1) ** 2}


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')
bc_l = pinnx.icbc.DirichletBC(func, boundary_l)
bc_r = pinnx.icbc.RobinBC(lambda X, y: y, boundary_r)
data = pinnx.problem.PDE(
    geom,
    pde,
    [bc_l, bc_r],
    net,
    num_domain=16,
    num_boundary=2,
    solution=func,
    num_test=100
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
