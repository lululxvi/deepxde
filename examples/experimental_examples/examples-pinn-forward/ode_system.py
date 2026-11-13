import brainstate as bst
import numpy as np

import deepxde.experimental as deepxde


def ode_system(x, y):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    jacobian = net.jacobian(x)

    y1, y2 = y["y1"], y["y2"]
    dy1_x = jacobian["y1"]["t"]
    dy2_x = jacobian["y2"]["t"]
    return [dy1_x - y2, dy2_x + y1]


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(t=None),
    deepxde.nn.FNN([1] + [50] * 3 + [2], "tanh"),
    deepxde.nn.ArrayToDict(y1=None, y2=None),
)


def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return {"y1": np.sin(x["t"]), "y2": np.cos(x["t"])}


geom = deepxde.geometry.TimeDomain(0, 10).to_dict_point("t")
ic = deepxde.icbc.IC(lambda x: {"y1": 0, "y2": 0})
data = deepxde.problem.PDE(
    geom,
    ode_system,
    [ic],
    net,
    num_domain=35,
    num_boundary=2,
    solution=func,
    num_test=100,
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=20000
)
trainer.saveplot(issave=True, isplot=True)
