import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(0, 1).to_dict_point("x")

alpha0 = 1.8
alpha = bst.ParamState(1.5)


def fpde(x, y, int_mat):
    """
    (D_{0+}^alpha + D_{1-}^alpha) u(x)
    """
    y = y["y"]
    x = x["x"]
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        int_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = int_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    lhs /= 2 * u.math.cos(alpha.value * np.pi / 2)
    rhs = gamma(alpha0 + 2) * x
    return lhs - rhs[: len(lhs)]


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


def func(x):
    return {"y": x["x"] * (u.math.abs(1 - x["x"] ** 2)) ** (alpha0 / 2)}


observe_x = {"x": np.linspace(-1, 1, num=20)}
observe_y = deepxde.icbc.PointSetBC(observe_x, func(observe_x))

data_type = "static"  # 'static' or 'dynamic'

if data_type == "static":
    # Static auxiliary points
    data = deepxde.problem.FPDE(
        geom,
        fpde,
        alpha,
        observe_y,
        [101],
        approximator=net,
        meshtype="static",
        anchors=observe_x,
        solution=func,
        loss_weights=[1, 100],
    )
else:
    # Dynamic auxiliary points
    data = deepxde.problem.FPDE(
        geom,
        fpde,
        alpha,
        observe_y,
        [100],
        approximator=net,
        meshtype="dynamic",
        num_domain=20,
        anchors=observe_x,
        solution=func,
        num_test=100,
        loss_weights=[1, 100],
    )

variable = deepxde.callbacks.VariableValue(alpha, period=1000)
trainer = deepxde.Trainer(data, external_trainable_variables=[alpha])
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000, callbacks=[variable])
trainer.saveplot(issave=True, isplot=True)
