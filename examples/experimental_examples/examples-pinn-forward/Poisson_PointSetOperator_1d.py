import brainstate as bst
import brainunit as u
import jax.tree

import deepxde.experimental as deepxde


def pde(x, y):
    dy_xx = net.hessian(x)["y"]["x"]["x"]
    return dy_xx - 2


def boundary_l(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x["x"], -1))


def func(x):
    return {"y": (x["x"] + 1) ** 2}


geom = deepxde.geometry.Interval(-1, 1).to_dict_point("x")

bc_l = deepxde.icbc.DirichletBC(func, boundary_l)


def dy_x(x, y):
    dy_x = net.jacobian(x)["y"]["x"]
    return {"y": dy_x}


def d_func(x):
    return {"y": 2 * (x["x"] + 1)}


boundary_pts = geom.random_boundary_points(2)
r_boundary_pts = jax.tree.map(lambda x: x[deepxde.utils.isclose(x, 1)], boundary_pts)
bc_r = deepxde.icbc.PointSetOperatorBC(r_boundary_pts, d_func(r_boundary_pts), dy_x)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None),
    deepxde.nn.FNN([1] + [50] * 2 + [1], "tanh"),
    deepxde.nn.ArrayToDict(y=None),
)

problem = deepxde.problem.PDE(
    geom,
    pde,
    [bc_l, bc_r],
    net,
    num_domain=16,
    num_boundary=2,
    solution=func,
    num_test=100,
)

# Print out first and second derivatives into a file during training on the boundary points
first_derivative = deepxde.callbacks.OperatorPredictor(
    geom.random_boundary_points(2),
    op=lambda x, y: net.jacobian(x)["y"]["x"],
    period=200,
    filename="first_derivative.txt",
)
second_derivative = deepxde.callbacks.OperatorPredictor(
    geom.random_boundary_points(2),
    op=lambda x, y: net.hessian(x)["y"]["x"]["x"],
    period=200,
    filename="second_derivative.txt",
)

trainer = deepxde.Trainer(problem)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=10000, callbacks=[first_derivative, second_derivative]
)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=10000
)
trainer.saveplot(issave=True, isplot=True)
