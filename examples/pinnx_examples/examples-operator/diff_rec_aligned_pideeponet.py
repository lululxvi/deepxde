import brainstate as bst
import jax
import matplotlib.pyplot as plt
import numpy as np

import deepxde
from deepxde import pinnx
from ADR_solver import solve_ADR


# PDE
def pde(x, y, aux):
    D = 0.01
    k = 0.01

    def solve_jac(inp1):
        f1 = lambda i: pinnx.grad.jacobian(lambda inp: net((x[0], inp))['y'][i], inp1, vmap=False)
        return jax.vmap(f1)(np.arange(x[0].shape[0]))

    dy_t = jax.vmap(solve_jac, out_axes=1)(jax.numpy.expand_dims(x[1], 1))[..., 1]

    def solve_hes(inp1):
        inp1 = pinnx.utils.array_to_dict(inp1, ['x', 't'])
        f1 = lambda i: pinnx.grad.hessian(lambda inp: net((x[0], pinnx.utils.dict_to_array(inp)))['y'][i],
                                          inp1,
                                          xi='x',
                                          xj='x',
                                          vmap=False)
        return jax.vmap(f1)(np.arange(x[0].shape[0]))

    dy_xx = jax.vmap(solve_hes, out_axes=1)(jax.numpy.expand_dims(x[1], 1))

    dy_t = jax.numpy.squeeze(dy_t)
    dy_xx = jax.numpy.squeeze(dy_xx['x']['x'])
    y = jax.numpy.squeeze(y['y'])

    return dy_t - D * dy_xx + k * y ** 2 - aux


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')

# Net
net = bst.nn.Sequential(
    pinnx.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
    ),
    pinnx.nn.ArrayToDict(y=None)
)

# Boundary condition
bc = pinnx.icbc.DirichletBC(lambda *args, **kwargs: {'y': 0})
ic = pinnx.icbc.IC(lambda *args, **kwargs: {'y': 0})

# Function space
func_space = deepxde.data.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.problem.PDEOperatorCartesianProd(
    geomtime,
    pde,
    [bc, ic],
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    function_variables=[0],
    batch_size=50,
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

model = pinnx.Trainer(data)
model.compile(bst.optim.Adam(0.0005)).train(iterations=20000)
model.saveplot(isplot=True)

func_feats = func_space.random(1)
xs = np.linspace(0, 1, num=100)[:, None]
v = func_space.eval_batch(func_feats, xs)[0]
x, t, u_true = solve_ADR(
    0,
    1,
    0,
    1,
    lambda x: 0.01 * np.ones_like(x),
    lambda x: np.zeros_like(x),
    lambda u: 0.01 * u ** 2,
    lambda u: 0.02 * u,
    lambda x, t: np.tile(v[:, None], (1, len(t))),
    lambda x: np.zeros_like(x),
    100,
    100,
)
u_true = u_true.T
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))['y']
u_pred = u_pred.reshape((100, 100))
print(pinnx.metrics.l2_relative_error(u_true, u_pred))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
