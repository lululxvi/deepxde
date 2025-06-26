"""
Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""

import brainstate as bst
import braintools
import brainunit as u
import numpy as np
from scipy.io import loadmat

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(-1, 1)
timedomain = deepxde.geometry.TimeDomain(0, 1)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain).to_dict_point("x", "t")

d = 0.001


@bst.compile.jit
def pde(x, out):
    jacobian = net.jacobian(x)
    hessian = net.hessian(x, xi="x", xj="x")
    dy_t = jacobian["u"]["t"]
    dy_xx = hessian["u"]["x"]["x"]
    return dy_t - d * dy_xx - 5 * (out["u"] - out["u"] ** 3)


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, t=None),
    deepxde.nn.FNN(
        [2] + [20] * 3 + [1],
        activation="tanh",
        output_transform=lambda x, y: u.math.expand_dims(
            x[..., 0] ** 2 * u.math.cos(np.pi * x[..., 0])
            + x[..., 1] * (1 - x[..., 0] ** 2) * y,
            axis=-1,
        ),
    ),
    deepxde.nn.ArrayToDict(u=None),
)

problem = deepxde.problem.TimePDE(
    geomtime, pde, [], net, num_domain=8000, num_boundary=400, num_initial=800
)

trainer = deepxde.Trainer(problem)
trainer.compile(bst.optim.Adam(lr=1e-3)).train(iterations=15000)
trainer.compile(bst.optim.LBFGS(lr=1e-3)).train(2000, display_every=200)
trainer.saveplot(issave=True, isplot=True)


def gen_testdata():
    data = loadmat("../dataset/Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    xx, tt = np.meshgrid(x, t)
    X = dict(x=np.ravel(xx), t=np.ravel(tt))
    return X, u.flatten()


X, y_true = gen_testdata()
y_pred = trainer.predict(X)
f = pde(X, y_pred)
print("Mean residual:", u.math.mean(u.math.absolute(f)))
print("L2 relative error:", braintools.metric.l2_norm(y_true, y_pred["u"]))
