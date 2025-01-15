import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

from deepxde import pinnx

geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point("x", "t")

alpha = 1.8


def fpde(x, y, int_mat):
    """
    du/dt + (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)
    """
    jacobian = net.jacobian(x)
    dy_t = jacobian['y']['t']
    y = y['y']
    x, t = x['x'], x['t']

    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = -(ini_mat @ y)
    else:
        lhs = -u.math.matmul(int_mat, y)

    rhs = -dy_t - u.math.exp(-t) * (
        x ** 3 * (1 - x) ** 3
        + gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    return lhs - rhs[..., len(lhs)]


def func(x):
    return {'y': u.math.exp(-x['t']) * x['x'] ** 3 * (1 - x['x']) ** 3}


def out_transform(x, y):
    x = pinnx.utils.array_to_dict(x, ['x', 't'], keep_dim=True)
    return x['x'] * (1 - x['x']) * x['t'] * y + x['x'] ** 3 * (1 - x['x']) ** 3


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, t=None),
    pinnx.nn.FNN(
        [2] + [20] * 4 + [1], "tanh", bst.init.KaimingUniform(),
        output_transform=out_transform
    ),
    pinnx.nn.ArrayToDict(y=None),
)

bc = pinnx.icbc.DirichletBC(func)
ic = pinnx.icbc.IC(func)

data_type = 'static'  # 'static',  or  'dynamic'

if data_type == 'static':
    # Static auxiliary points
    data = pinnx.problem.TimeFPDE(
        geomtime,
        fpde,
        alpha,
        [bc, ic],
        [52],
        net,
        meshtype=data_type,
        num_domain=400,
        solution=func,
    )
else:
    # Dynamic auxiliary points
    data = pinnx.problem.TimeFPDE(
        geomtime,
        fpde,
        alpha,
        [bc, ic],
        [100],
        net,
        meshtype=data_type,
        num_domain=20,
        num_boundary=1,
        num_initial=1,
        solution=func,
        num_test=50,
    )

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000)
trainer.saveplot(issave=False, isplot=True)

X = geomtime.random_points(1000)
y_true = func(X)
y_pred = trainer.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
