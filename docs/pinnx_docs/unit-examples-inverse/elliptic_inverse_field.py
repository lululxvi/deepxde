import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from deepxde import pinnx

unit_of_x = u.meter
unit_of_u = u.newton
unit_of_q = u.newton / unit_of_x ** 2


def pde(x, y):
    du_xx = net.hessian(x, y='u')['u']['x']['x']
    return -du_xx + y['q']


geom = pinnx.geometry.Interval(-1, 1).to_dict_point(x=unit_of_x)


def sol(x):
    # solution is u(x) = sin(pi*x), q(x) = -pi^2 * sin(pi*x)
    # return {'u': u.math.sin(u.math.pi * x['x']), }
    return {
        'u': u.math.sin(u.math.pi * x['x'] / unit_of_x) * unit_of_u,
        'q': -u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'] / unit_of_x) * unit_of_q
    }


bc = pinnx.icbc.DirichletBC(sol)


def gen_traindata(num):
    # generate num equally-spaced points from -1 to 1
    xvals = np.linspace(-1, 1, num)
    uvals = np.sin(np.pi * xvals)
    return {'x': xvals * unit_of_x}, {'u': uvals * unit_of_u}


ob_x, ob_u = gen_traindata(100)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=unit_of_x),
    pinnx.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(u=unit_of_u, q=unit_of_q),
)
problem = pinnx.problem.PDE(
    geom,
    pde,
    [bc, observe_u],
    net,
    num_domain=200,
    num_boundary=2,
    anchors=ob_x,
    num_test=1000,
    loss_weights=[1, 100, 1000],
)

model = pinnx.Trainer(problem)
model.compile(bst.optim.Adam(0.0001)).train(iterations=20000)
model.saveplot(issave=True, isplot=True)

# view results
x = geom.uniform_points(500)
yhat = model.predict(x)
uhat = yhat['u'] / unit_of_u
qhat = yhat['q'] / unit_of_q
x = x['x'] / unit_of_x

utrue = np.sin(np.pi * x)
print("l2 relative error for u: " + str(pinnx.metrics.l2_relative_error(utrue, uhat)))
plt.figure()
plt.plot(x, utrue, "-", label="u_true")
plt.plot(x, uhat, "--", label="u_NN")
plt.legend()

qtrue = -np.pi ** 2 * np.sin(np.pi * x)
print("l2 relative error for q: " + str(pinnx.metrics.l2_relative_error(qtrue, qhat)))
plt.figure()
plt.plot(x, qtrue, "-", label="q_true")
plt.plot(x, qhat, "--", label="q_NN")
plt.legend()

plt.show()
