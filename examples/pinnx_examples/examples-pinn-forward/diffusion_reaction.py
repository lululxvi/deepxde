import brainstate as bst
import brainunit as u
import numpy as np

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(-np.pi, np.pi)
timedomain = deepxde.geometry.TimeDomain(0, 1)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x, xi='x', xj='x')
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    d = 1
    return (
        dy_t
        - d * dy_xx
        - u.math.exp(-x['t'])
        * (
            3 * u.math.sin(2 * x['x']) / 2
            + 8 * u.math.sin(3 * x['x']) / 3
            + 15 * u.math.sin(4 * x['x']) / 4
            + 63 * u.math.sin(8 * x['x']) / 8
        )
    )


def output_transform(x, y):
    x = deepxde.utils.array_to_dict(x, ['x', 't'], keep_dim=True)
    return (
        x['t'] * (np.pi ** 2 - x['x'] ** 2) * y
        + u.math.sin(x['x'])
        + u.math.sin(2 * x['x']) / 2
        + u.math.sin(3 * x['x']) / 3
        + u.math.sin(4 * x['x']) / 4
        + u.math.sin(8 * x['x']) / 8
    )


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, t=None),
    deepxde.nn.FNN(
        [2] + [30] * 6 + [1],
        "tanh",
        bst.init.KaimingUniform(),
        output_transform=output_transform
    ),
    deepxde.nn.ArrayToDict(y=None)
)


def func(x):
    return {
        'y': u.math.exp(-x['t']) * (
            u.math.sin(x['x'])
            + u.math.sin(2 * x['x']) / 2
            + u.math.sin(3 * x['x']) / 3
            + u.math.sin(4 * x['x']) / 4
            + u.math.sin(8 * x['x']) / 8
        )
    }


data = deepxde.problem.TimePDE(
    geomtime, pde, [], net,
    num_domain=320, solution=func, num_test=80000
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(1e-3), metrics=["l2 relative error"]).train(iterations=20000)
trainer.saveplot(issave=True, isplot=True)
