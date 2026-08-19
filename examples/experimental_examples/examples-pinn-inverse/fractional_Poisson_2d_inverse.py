import brainstate as bst
import brainunit as u
import jax
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import deepxde.experimental as deepxde

alpha0 = 1.8
alpha = bst.ParamState(1.5)


def fpde(x, y, int_mat):
    r"""
    \int_theta D_theta^alpha u(x)
    """
    y = y["y"]
    x = deepxde.utils.dict_to_array(x)

    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        int_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = int_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    lhs *= -u.math.exp(
        jax.lax.lgamma((1 - alpha.value) / 2) + jax.lax.lgamma((2 + alpha.value) / 2)
    ) / (2 * np.pi**1.5)
    x = x[: len(lhs)]
    rhs = (
        2**alpha0
        * gamma(2 + alpha0 / 2)
        * gamma(1 + alpha0 / 2)
        * (1 - (1 + alpha0 / 2) * u.math.sum(x**2, axis=1))
    )
    return lhs - rhs


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x1=None, x2=None),
    deepxde.nn.FNN(
        [2] + [20] * 4 + [1],
        "tanh",
        bst.init.KaimingUniform(),
        output_transform=lambda x, y: (1 - u.math.sum(x**2, axis=1, keepdims=True)) * y,
    ),
    deepxde.nn.ArrayToDict(y=None),
)


def func(x):
    x = deepxde.utils.dict_to_array(x)
    y = (u.math.abs(1 - u.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (
        1 + alpha.value / 2
    )
    return {"y": y}


geom = deepxde.geometry.Disk([0, 0], 1).to_dict_point("x1", "x2")
observe_x = geom.random_points(30)
bc = deepxde.icbc.PointSetBC(observe_x, func(observe_x))

problem = deepxde.problem.FPDE(
    geom,
    fpde,
    alpha,
    bc,
    [8, 100],
    approximator=net,
    num_domain=64,
    anchors=observe_x,
    solution=func,
    loss_weights=[1, 100],
)

variable = deepxde.callbacks.VariableValue(alpha, period=1000)
model = deepxde.Trainer(problem, external_trainable_variables=[alpha])
model.compile(bst.optim.Adam(1e-3)).train(iterations=10000, callbacks=[variable])
model.saveplot(issave=True, isplot=True)
