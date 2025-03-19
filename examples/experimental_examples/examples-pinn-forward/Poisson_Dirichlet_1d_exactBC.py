import brainstate as bst
import brainunit as u
import numpy as np

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(0, np.pi).to_dict_point('x')


def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["y"]["x"]["x"]
    x = x["x"]
    summation = sum([i * u.math.sin(i * x) for i in range(1, 5)])
    return -dy_xx - summation - 8 * u.math.sin(8 * x)


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN(
        [1] + [50] * 3 + [1], "tanh",
        output_transform=lambda x, y: x * (np.pi - x) * y + x
    ),
    deepxde.nn.ArrayToDict(y=None),
)


def func(x):
    x = x['x']
    summation = sum([np.sin(i * x) / i for i in range(1, 5)])
    y = x + summation + np.sin(8 * x) / 8
    return {'y': y}


problem = deepxde.problem.PDE(
    geom,
    pde,
    [],
    net,
    num_domain=64,
    solution=func,
    num_test=400
)

trainer = deepxde.Trainer(problem)
trainer.compile(
    bst.optim.Adam(bst.optim.InverseTimeDecayLR(0.001, 1000, 0.3)),
    metrics=["l2 relative error"]
).train(iterations=30000)
trainer.saveplot(issave=True, isplot=True)
