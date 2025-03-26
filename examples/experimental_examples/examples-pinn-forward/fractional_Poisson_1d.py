import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(0, 1).to_dict_point("x")

alpha = 1.5


def fpde(x, y, int_mat):
    """
    (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)
    """
    x = x["x"]
    y = y["y"]
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = ini_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    rhs = (
        gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    # lhs /= 2 * np.cos(alpha * np.pi / 2)
    # rhs = gamma(alpha + 2) * x
    return lhs - rhs[: len(lhs)]


def func(x):
    return {"y": x["x"] ** 3 * (1 - x["x"]) ** 3}


bc = deepxde.icbc.DirichletBC(func)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN(
        [1] + [20] * 4 + [1],
        "tanh",
        bst.init.KaimingUniform(),
        output_transform=lambda x, y: x * (1 - x) * y,
    ),
    deepxde.nn.ArrayToDict(y=None),
)

data_type = "static"  # 'static' or 'dynamic'

if data_type == "static":
    # Static auxiliary points
    data = deepxde.problem.FPDE(
        geom, fpde, alpha, bc, [101], approximator=net, meshtype="static", solution=func
    )

else:

    # Dynamic auxiliary points
    data = deepxde.problem.FPDE(
        geom,
        fpde,
        alpha,
        bc,
        [100],
        approximator=net,
        meshtype="dynamic",
        num_domain=20,
        num_boundary=2,
        solution=func,
        num_test=100,
    )

trainer = deepxde.Trainer(data)

trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
