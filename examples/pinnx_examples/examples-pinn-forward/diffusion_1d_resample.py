import brainstate as bst
import brainunit as u
import numpy as np

from deepxde import pinnx

geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')


def func(x):
    return {'y': u.math.sin(np.pi * x['x']) * u.math.exp(-x['t'])}


bc = pinnx.icbc.DirichletBC(func)
ic = pinnx.icbc.IC(func)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x, xi='x', xj='x')
    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    return (
        dy_t
        - dy_xx
        + u.math.exp(-x['t'])
        * (u.math.sin(u.math.pi * x['x']) -
           u.math.pi ** 2 * u.math.sin(u.math.pi * x['x']))
    )


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, t=None),
    pinnx.nn.FNN(
        [2] + [32] * 3 + [1], 'tanh', bst.init.KaimingUniform()
    ),
    pinnx.nn.ArrayToDict(y=None)
)

problem = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    net,
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

trainer = pinnx.Trainer(problem)
resampler = pinnx.callbacks.PDEPointResampler(period=100)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=2000, callbacks=[resampler])
trainer.saveplot(issave=True, isplot=True)
