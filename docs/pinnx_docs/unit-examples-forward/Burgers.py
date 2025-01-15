import brainstate as bst
import brainunit as u
import numpy as np

from deepxde import pinnx

geometry = pinnx.geometry.GeometryXTime(
    geometry=pinnx.geometry.Interval(-1, 1.),
    timedomain=pinnx.geometry.TimeDomain(0, 0.99)
).to_dict_point(x=u.meter, t=u.second)

uy = u.meter / u.second
bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0. * uy})
ic = pinnx.icbc.IC(lambda x: {'y': -u.math.sin(u.math.pi * x['x'] / u.meter) * uy})

v = 0.01 / u.math.pi * u.meter ** 2 / u.second


def pde(x, y):
    jacobian = approximator.jacobian(x)
    hessian = approximator.hessian(x)
    dy_x = jacobian['y']['x']
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    residual = dy_t + y['y'] * dy_x - v * dy_xx
    return residual


approximator = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=u.meter, t=u.second),
    pinnx.nn.FNN(
        [geometry.dim] + [20] * 3 + [1],
        "tanh",
        bst.init.KaimingUniform()
    ),
    pinnx.nn.ArrayToDict(y=uy)
)

problem = pinnx.problem.TimePDE(
    geometry,
    pde,
    [bc, ic],
    approximator,
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(1e-3)).train(iterations=15000)
trainer.compile(bst.optim.LBFGS(1e-3)).train(2000, display_every=500)
trainer.saveplot(issave=True, isplot=True)


def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = {'x': np.ravel(xx) * u.meter, 't': np.ravel(tt) * u.second}
    y = exact.flatten()[:, None]
    return X, y * uy


X, y_true = gen_testdata()
y_pred = trainer.predict(X)
f = pde(X, y_pred)
print("Mean residual:", u.math.mean(u.math.absolute(f)))
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred['y']))
