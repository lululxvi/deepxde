import brainstate as bst
import brainunit as u
import jax.tree
import numpy as np

import deepxde.experimental as deepxde


Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - u.math.sqrt(1 / (4 * nu**2) + 4 * u.math.pi**2)


def pde(x, y):
    jacobian = net.jacobian(x)
    hessian = net.hessian(x)

    u_vel, v_vel, p = y["u_vel"], y["v_vel"], y["p"]
    u_vel_x = jacobian["u_vel"]["x"]
    u_vel_y = jacobian["u_vel"]["y"]
    u_vel_xx = hessian["u_vel"]["x"]["x"]
    u_vel_yy = hessian["u_vel"]["y"]["y"]

    v_vel_x = jacobian["v_vel"]["x"]
    v_vel_y = jacobian["v_vel"]["y"]
    v_vel_xx = hessian["v_vel"]["x"]["x"]
    v_vel_yy = hessian["v_vel"]["y"]["y"]

    p_x = jacobian["p"]["x"]
    p_y = jacobian["p"]["y"]

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return momentum_x, momentum_y, continuity


def bc_func(x):
    u_ = 1 - u.math.exp(l * x["x"]) * u.math.cos(2 * u.math.pi * x["y"])
    v = (
        l
        / (2 * u.math.pi)
        * u.math.exp(l * x["x"])
        * u.math.sin(2 * u.math.pi * x["y"])
    )
    p = 1 / 2 * (1 - u.math.exp(2 * l * x["x"]))
    return {"u_vel": u_, "v_vel": v, "p": p}


def boundary_outflow(x, on_boundary):
    return on_boundary and deepxde.utils.isclose(x[0], 1)


spatial_domain = deepxde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])
spatial_domain = spatial_domain.to_dict_point("x", "y")

bc = deepxde.icbc.DirichletBC(bc_func)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, y=None),
    deepxde.nn.FNN([2] + 4 * [50] + [3], "tanh"),
    deepxde.nn.ArrayToDict(u_vel=None, v_vel=None, p=None),
)

data = deepxde.problem.PDE(
    spatial_domain,
    pde,
    [bc],
    net,
    num_domain=2601,
    num_boundary=400,
    num_test=100000,
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=30000)
trainer.compile(bst.optim.LBFGS(1e-3)).train(iterations=2000)

X = spatial_domain.random_points(100000)
output = trainer.predict(X)

u_exact = bc_func(X)
l2_difference = deepxde.metrics.l2_relative_error(u_exact, output)

f = pde(X, output)
residual = jax.tree.map(lambda x: np.mean(np.absolute(x)), f)
print("Mean residual:", residual)
print("L2 relative error:", l2_difference)
