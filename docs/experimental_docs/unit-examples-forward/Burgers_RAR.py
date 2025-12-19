import brainstate as bst
import brainunit as u
import jax.tree
import numpy as np

import deepxde.experimental as deepxde

geom = deepxde.geometry.Interval(-1, 1)
timedomain = deepxde.geometry.TimeDomain(0, 0.99)
geomtime = deepxde.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point(x=u.meter, t=u.second)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=u.meter, t=u.second),
    deepxde.nn.FNN([2] + [20] * 3 + [1], "tanh", bst.init.KaimingUniform()),
    deepxde.nn.ArrayToDict(y=u.meter / u.second),
)
v = 0.01 / u.math.pi * u.meter ** 2 / u.second


def pde(x, y):
    jacobian = net.jacobian(x)
    hessian = net.hessian(x, xi='x', xj='x')

    dy_x = jacobian['y']['x']
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t + y['y'] * dy_x - v * dy_xx


bc = deepxde.icbc.DirichletBC(lambda x: {'y': 0 * u.meter / u.second})
ic = deepxde.icbc.IC(lambda x: {'y': -u.math.sin(u.math.pi * x['x'] / u.meter) * u.meter / u.second})

problem = deepxde.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    net,
    num_domain=2500,
    num_boundary=100,
    num_initial=160
)

trainer = deepxde.Trainer(problem)

trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000)
trainer.compile(bst.optim.LBFGS(1e-3)).train(1000)

X = geomtime.random_points(100000)
err = 1
while u.get_magnitude(err) > 0.012:
    f = trainer.predict(X, operator=pde)
    err_eq = u.math.absolute(f)
    err = u.math.mean(err_eq)
    print(f"Mean residual: {err:.3f}")

    x_id = u.math.argmax(err_eq)
    new_xs = jax.tree.map(lambda x: x[[x_id]], X)
    print("Adding new point:", new_xs, "\n")
    problem.add_anchors(new_xs)
    early_stopping = deepxde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    trainer.compile(bst.optim.Adam(1e-3)).train(iterations=10000,
                                                disregard_previous_best=True,
                                                callbacks=[early_stopping])
    trainer.compile(bst.optim.LBFGS(1e-3)).train(1000, display_every=100)

trainer.saveplot(issave=True, isplot=True)


def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = {'x': np.ravel(xx) * u.meter, 't': np.ravel(tt) * u.second}
    y = {'y': exact.flatten() * u.meter / u.second}
    return X, y


X, y_true = gen_testdata()
y_pred = trainer.predict(X)
print("L2 relative error:", deepxde.metrics.l2_relative_error(y_true, y_pred))
