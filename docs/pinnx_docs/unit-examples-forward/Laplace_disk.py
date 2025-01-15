import brainstate as bst
import brainunit as u
import numpy as np

from deepxde import pinnx

# geom = pinnx.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
# geom = geom.to_dict_point("r", "theta")

geom = pinnx.geometry.Rectangle(
    xmin=[0, 0],
    xmax=[1, 2 * np.pi],
).to_dict_point(r=u.meter, theta=u.radian)

uy = u.volt / u.meter
bc = pinnx.icbc.DirichletBC(
    lambda x: {'y': u.math.cos(x['theta']) * uy},
    lambda x, on_boundary: u.math.logical_and(on_boundary, u.math.allclose(x['r'], 1 * u.meter)),
)


def solution(x):
    r, theta = x['r'], x['theta']
    # TODO: Why add more divide u.meter?
    return {'y': r * u.math.cos(theta) * uy / u.meter}


def pde(x, y):
    jacobian = net.jacobian(x)
    hessian = net.hessian(x)

    dy_r = jacobian["y"]["r"]
    dy_rr = hessian["y"]["r"]["r"]
    dy_thetatheta = hessian["y"]["theta"]["theta"]
    return x['r'] * dy_r + x['r'] ** 2 * dy_rr + dy_thetatheta


# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
def feature_transform(x):
    x = pinnx.utils.array_to_dict(x, ["r", "theta"], keep_dim=True)
    return u.math.concatenate([x['r'] * u.math.sin(x['theta']),
                               x['r'] * u.math.cos(x['theta'])], axis=-1)


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(r=u.meter, theta=u.radian),
    pinnx.nn.FNN([2] + [20] * 3 + [1], "tanh", input_transform=feature_transform),
    pinnx.nn.ArrayToDict(y=uy),
)

problem = pinnx.problem.PDE(
    geom,
    pde,
    bc,
    net,
    num_domain=2540,
    num_boundary=80,
    solution=solution
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(1e-3), metrics=["l2 relative error"]).train(iterations=15000)
trainer.saveplot(issave=True, isplot=True)
