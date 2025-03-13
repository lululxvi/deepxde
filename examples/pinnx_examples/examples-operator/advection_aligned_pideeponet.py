import brainstate as bst
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

import deepxde
import deepxde.experimental as deepxde_exp

geom = deepxde_exp.geometry.Interval(0, 1)
timedomain = deepxde_exp.geometry.TimeDomain(0, 1)
geomtime = deepxde_exp.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')


# PDE
def pde_eqs(x, y, aux):
    def solve_jac(inp1):
        f1 = lambda i: deepxde_exp.grad.jacobian(lambda inp: net((x[0], inp))['y'][i], inp1, vmap=False)
        return jax.vmap(f1)(np.arange(x[0].shape[0]))

    jacobian = jax.vmap(solve_jac, out_axes=1)(jax.numpy.expand_dims(x[1], 1))
    dy_x = u.math.squeeze(jacobian[..., 0])
    dy_t = u.math.squeeze(jacobian[..., 1])
    return dy_t + dy_x


# Net
def periodic(x):
    x, t = x[..., :1], x[..., 1:]
    x = x * 2 * u.math.pi
    return u.math.concatenate(
        [u.math.cos(x),
         u.math.sin(x),
         u.math.cos(2 * x),
         u.math.sin(2 * x),
         t],
        axis=-1
    )


dim_x = 5
net = bst.nn.Sequential(
    deepxde_exp.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [dim_x, 128, 128, 128],
        "tanh",
        input_transform=periodic,
    ),
    deepxde_exp.nn.ArrayToDict(y=None)
)

ic = deepxde_exp.icbc.IC(lambda x, aux: {'y': aux})

# Function space
func_space = deepxde.data.GRF(kernel="ExpSineSquared", length_scale=1)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
problem = deepxde_exp.problem.PDEOperatorCartesianProd(
    geomtime,
    pde_eqs,
    ic,
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    function_variables=[0],
    num_fn_test=100,
    batch_size=32,
    num_domain=250,
    num_initial=50,
    num_test=500
)

model = deepxde_exp.Trainer(problem)
model.compile(bst.optim.Adam(0.0005)).train(iterations=50000)
model.saveplot(issave=True, isplot=True)

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts).T
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))['y']
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
print(deepxde_exp.metrics.l2_relative_error(u_true, u_pred))
