import brainstate as bst
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

import deepxde
import deepxde.experimental as deepxde_new
from ADR_solver import solve_ADR


# PDE
def pde(x, y, aux):
    D = 0.01
    k = 0.01

    def solve_jac(x_):
        return deepxde_new.grad.jacobian(
            lambda inp: net((x_[0], deepxde_new.utils.dict_to_array(inp))),
            {'x': x_[1][0], 't': x_[1][1]},
            x='t',
            vmap=False
        )

    dy_t = jax.vmap(solve_jac)(x)

    def solve_hes(x_):
        return deepxde_new.grad.hessian(
            lambda inp: net((x_[0], deepxde_new.utils.dict_to_array(inp))),
            {'x': x_[1][0], 't': x_[1][1]},
            xi='x',
            xj='x',
            vmap=False
        )

    dy_xx = jax.vmap(solve_hes)(x)

    dy_t = dy_t['y']['t']
    dy_xx = dy_xx['y']['x']['x']
    y = y['y']
    aux = u.math.squeeze(aux)
    return dy_t - D * dy_xx + k * y ** 2 - aux


geom = deepxde_new.geometry.Interval(0, 1)
timedomain = deepxde_new.geometry.TimeDomain(0, 1)
geomtime = deepxde_new.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')

# Net
net = bst.nn.Sequential(
    deepxde_new.nn.DeepONet(
        [50, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
    ),
    deepxde_new.nn.ArrayToDict(y=None)
)

# Boundary condition
bc = deepxde_new.icbc.DirichletBC(lambda *args, **kwargs: {'y': 0})
ic = deepxde_new.icbc.IC(lambda *args, **kwargs: {'y': 0})

# Function space
func_space = deepxde.data.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = deepxde_new.problem.PDEOperator(
    geomtime,
    pde,
    [bc, ic],
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    function_variables=[0],
    num_fn_test=1000,
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

model = deepxde_new.Trainer(data)
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

v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])[0]
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((np.tile(v_branch, (100 * 100, 1)), x_trunk))['y']
u_pred = u_pred.reshape((100, 100))
print(deepxde_new.metrics.l2_relative_error(u_true, u_pred))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
