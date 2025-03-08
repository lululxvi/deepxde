import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

from deepxde import pinnx

alpha = 1.8


# Backend tensorflow.compat.v1
def fpde(x, y, int_mat):
    """
    \int_theta D_theta^alpha u(x)
    """
    x = pinnx.utils.dict_to_array(x)
    y = y['y']
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        int_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
    lhs = int_mat @ y
    lhs *= gamma((1 - alpha) / 2) * gamma((3 + alpha) / 2) / (2 * np.pi ** 2)
    x = x[: len(lhs)]
    rhs = (
        2 ** alpha
        * gamma(2 + alpha / 2)
        * gamma((3 + alpha) / 2)
        / gamma(3 / 2)
        * (1 - (1 + alpha / 3) * u.math.sum(x ** 2, axis=1))
    )
    return lhs - rhs


def func(x):
    x = pinnx.utils.dict_to_array(x)
    y = (u.math.abs(1 - u.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (1 + alpha / 2)
    return {'y': y}


geom = pinnx.geometry.Sphere([0, 0, 0], 1).to_dict_point('x1', 'x2', 'x3')
bc = pinnx.icbc.DirichletBC(func)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x1=None, x2=None, x3=None),
    pinnx.nn.FNN([3] + [20] * 4 + [1], "tanh", bst.init.KaimingUniform(),
                 output_transform=lambda x, y: (1 - u.math.sum(x ** 2, axis=1, keepdims=True)) * y),
    pinnx.nn.ArrayToDict(y=None),
)

problem = pinnx.problem.FPDE(
    geom,
    fpde,
    alpha,
    bc,
    [8, 8, 100],
    net,
    num_domain=256,
    num_boundary=1,
    solution=func,
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000)
trainer.saveplot(issave=False, isplot=True)

X = geom.random_points(10000)
y_true = func(X)
y_pred = trainer.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
