"""

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""

import brainstate as bst
import brainunit as u

import deepxde.experimental as deepxde

lmbd = 1.0
mu = 0.5
Q = 4.0

geom = deepxde.geometry.Rectangle([0, 0], [1, 1]).to_dict_point("x", "y")

BC_type = ("hard",)  # hard  or  soft


def boundary_left(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x["x"], 0.0))


def boundary_right(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x["x"], 1.0))


def boundary_top(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x["y"], 1.0))


def boundary_bottom(x, on_boundary):
    return u.math.logical_and(on_boundary, deepxde.utils.isclose(x["y"], 0.0))


# Exact solutions
def func(x):
    ux = u.math.cos(2 * u.math.pi * x["x"]) * u.math.sin(u.math.pi * x["y"])
    uy = u.math.sin(u.math.pi * x["x"]) * Q * x["y"] ** 4 / 4

    E_xx = (
        -2
        * u.math.pi
        * u.math.sin(2 * u.math.pi * x["x"])
        * u.math.sin(u.math.pi * x["y"])
    )
    E_yy = u.math.sin(u.math.pi * x["x"]) * Q * x["y"] ** 3
    E_xy = 0.5 * (
        u.math.pi * u.math.cos(2 * u.math.pi * x["x"]) * u.math.cos(u.math.pi * x["y"])
        + u.math.pi * u.math.cos(u.math.pi * x["x"]) * Q * x["y"] ** 4 / 4
    )

    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu

    return {"u": ux, "v": uy, "s": Sxx, "c": Syy, "e": Sxy}


# Soft Boundary Conditions
ux_top_bc = deepxde.icbc.DirichletBC(lambda x: {"u": 0}, boundary_top)
ux_bottom_bc = deepxde.icbc.DirichletBC(lambda x: {"u": 0}, boundary_bottom)
uy_left_bc = deepxde.icbc.DirichletBC(lambda x: {"v": 0}, boundary_left)
uy_bottom_bc = deepxde.icbc.DirichletBC(lambda x: {"v": 0}, boundary_bottom)
uy_right_bc = deepxde.icbc.DirichletBC(lambda x: {"v": 0}, boundary_right)
sxx_left_bc = deepxde.icbc.DirichletBC(lambda x: {"s": 0}, boundary_left)
sxx_right_bc = deepxde.icbc.DirichletBC(lambda x: {"s": 0}, boundary_right)
syy_top_bc = deepxde.icbc.DirichletBC(
    lambda x: {"c": (2 * mu + lmbd) * Q * u.math.sin(u.math.pi * x["x"])},
    boundary_top,
)


# Hard Boundary Conditions
def hard_BC(x, f):
    x = deepxde.utils.array_to_dict(x, ["x", "y"])
    f = deepxde.utils.array_to_dict(f, ["u", "v", "s", "c", "e"])
    Ux = f["u"] * x["y"] * (1 - x["y"])
    Uy = f["v"] * x["x"] * (1 - x["x"]) * x["y"]

    Sxx = f["s"] * x["x"] * (1 - x["x"])
    Syy = f["c"] * (1 - x["y"]) + (lmbd + 2 * mu) * Q * u.math.sin(u.math.pi * x["x"])
    Sxy = f["e"]
    return u.math.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def fx(x):
    return (
        -lmbd
        * (
            4
            * u.math.pi**2
            * u.math.cos(2 * u.math.pi * x["x"])
            * u.math.sin(u.math.pi * x["y"])
            - Q * x["y"] ** 3 * u.math.pi * u.math.cos(u.math.pi * x["x"])
        )
        - mu
        * (
            u.math.pi**2
            * u.math.cos(2 * u.math.pi * x["x"])
            * u.math.sin(u.math.pi * x["y"])
            - Q * x["y"] ** 3 * u.math.pi * u.math.cos(u.math.pi * x["x"])
        )
        - 8
        * mu
        * u.math.pi**2
        * u.math.cos(2 * u.math.pi * x["x"])
        * u.math.sin(u.math.pi * x["y"])
    )


def fy(x):
    return (
        lmbd
        * (
            3 * Q * x["y"] ** 2 * u.math.sin(u.math.pi * x["x"])
            - 2
            * u.math.pi**2
            * u.math.cos(u.math.pi * x["y"])
            * u.math.sin(2 * u.math.pi * x["x"])
        )
        - mu
        * (
            2
            * u.math.pi**2
            * u.math.cos(u.math.pi * x["y"])
            * u.math.sin(2 * u.math.pi * x["x"])
            + (Q * x["y"] ** 4 * u.math.pi**2 * u.math.sin(u.math.pi * x["x"])) / 4
        )
        + 6 * Q * mu * x["y"] ** 2 * u.math.sin(u.math.pi * x["x"])
    )


def pde(x, y):
    jacobian = net.jacobian(x)

    E_xx = jacobian["u"]["x"]
    E_yy = jacobian["u"]["y"]
    E_xy = 0.5 * (jacobian["u"]["y"] + jacobian["v"]["x"])

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian["s"]["x"]
    Syy_y = jacobian["c"]["y"]
    Sxy_x = jacobian["e"]["x"]
    Sxy_y = jacobian["e"]["y"]

    momentum_x = Sxx_x + Sxy_y - fx(x)
    momentum_y = Sxy_x + Syy_y - fy(x)

    stress_x = S_xx - y["s"]
    stress_y = S_yy - y["c"]
    stress_xy = S_xy - y["e"]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, y=None),
    deepxde.nn.PFNN(
        [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5],
        "tanh",
        bst.init.KaimingUniform(),
        output_transform=hard_BC if BC_type == "hard" else None,
    ),
    deepxde.nn.ArrayToDict(u=None, v=None, s=None, c=None, e=None),
)

if BC_type == "hard":
    bcs = []
else:
    bcs = [
        ux_top_bc,
        ux_bottom_bc,
        uy_left_bc,
        uy_bottom_bc,
        uy_right_bc,
        sxx_left_bc,
        sxx_right_bc,
        syy_top_bc,
    ]

data = deepxde.problem.PDE(
    geom,
    pde,
    bcs,
    net,
    num_domain=500,
    num_boundary=500,
    solution=func,
    num_test=100,
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(
    iterations=1000
)
trainer.saveplot(issave=True, isplot=True)
