import brainstate as bst
import numpy as np
import optax
import brainunit as u
import os

from deepxde import pinnx


def heat_eq_exact_solution(x, t):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    a_value = a.to_decimal(u.meter2 / u.second).item()
    n_value = n.to_decimal(u.Hz).item()
    L_value = L.to_decimal(u.meter).item()
    return np.exp(-(n_value ** 2 * np.pi ** 2 * a_value * t) / (L_value ** 2)) * np.sin(n_value * np.pi * x / L_value)


def gen_exact_solution():
    """Generates exact solution for the heat equation for the given values of x and t."""
    if os.path.exists("heat_eq_data.npz"):
        return
    # Number of points in each dimension:
    x_dim, t_dim = (256, 201)

    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.0)
    x_max, t_max = (L.to_decimal(u.meter), 1.0)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Obtain the value of the exact solution for each generated point:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i, j] = heat_eq_exact_solution(x[i], t[j])

    # Save solution:
    np.savez("heat_eq_data", x=x, t=t, usol=usol)

    # Save solution:
    np.savez("heat_eq_data", x=x, t=t, usol=usol)
    # Load solution:
    # data = np.load("heat_eq_data.npz")


def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    data = np.load("heat_eq_data.npz")
    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T
    # Process the data and flatten it out (like labels and features):
    xx, tt = np.meshgrid(x, t)
    X = {'x': np.ravel(xx) * u.meter, 't': np.ravel(tt) * u.second}
    y = exact.flatten()[:, None]
    return X, y * uy


# Problem parameters:
a = 0.4 * u.meter2 / u.second  # Thermal diffusivity
L = 1 * u.meter # Length of the bar
n = 1 * u.Hz # Frequency of the sinusoidal initial conditions

# Generate a dataset with the exact solution (if you dont have one):
gen_exact_solution()

uy = u.kelvin / u.second

# Computational geometry:
geomtime = pinnx.geometry.GeometryXTime(
    geometry=pinnx.geometry.Interval(0., 1.),
    timedomain=pinnx.geometry.TimeDomain(0., 1.)
).to_dict_point(x=u.meter, t=u.second)

# Initial and boundary conditions:
bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0. * uy})
ic = pinnx.icbc.IC(
    lambda x: {'y': u.math.sin(n * u.math.pi * x['x'][:] / L, unit_to_scale=u.becquerel) * uy},
)


@bst.compile.jit
def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    jacobian = approximator.jacobian(x)
    hessian = approximator.hessian(x)
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t - a * dy_xx


# Define the PDE problem and configurations of the network:

approximator = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=u.meter, t=u.second),
    pinnx.nn.FNN(
        [2] + [20] * 3 + [1],
        "tanh",
        bst.init.KaimingUniform()
    ),
    pinnx.nn.ArrayToDict(y=uy)
)

problem = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    approximator,
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

trainer = pinnx.Trainer(problem)

# Build and train the trainer:
trainer.compile(bst.optim.Adam(1e-3))

pde_resampler = pinnx.callbacks.PDEPointResampler(period=10)
trainer.train(iterations=10000, callbacks=[pde_resampler])
trainer.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
# TODO: train method must has iteration param
# losshistory, train_state = trainer.train(callbacks=[pde_resampler])
trainer.train(iterations=1000, callbacks=[pde_resampler])

# Plot/print the results
# TODO: losshistory, train_state params for saveplot?
# trainer.saveplot(losshistory, train_state, issave=True, isplot=True)
trainer.saveplot(issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = trainer.predict(X)
f = trainer.predict(X, operator=pde)
print("Mean residual:", u.math.mean(u.math.absolute(f)))
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred['y']))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
