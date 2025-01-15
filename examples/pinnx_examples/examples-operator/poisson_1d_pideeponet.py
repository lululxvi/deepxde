import brainstate as bst
import matplotlib.pyplot as plt
import numpy as np
import jax
from deepxde import pinnx
import brainunit as u


# Poisson equation: -u_xx = f
def equation(x, y, aux):

    def solve_hes(inp1):
        f1 = lambda i: pinnx.grad.hessian(lambda inp: net((x[0], inp))['u'][i], inp1, vmap=False)
        return jax.vmap(f1)(np.arange(x[0].shape[0]))

    dy_xx = jax.vmap(solve_hes, out_axes=1)(jax.numpy.expand_dims(x[1], 1))
    dy_xx = u.math.squeeze(dy_xx)
    return -dy_xx - aux


# Domain is interval [0, 1]
geom = pinnx.geometry.Interval(0, 1).to_dict_point('x')

bc = pinnx.icbc.DirichletBC(lambda x, aux: {'u': 0.})

# Function space for f(x) are polynomials
degree = 3
space = pinnx.fnspace.PowerSeries(N=degree + 1)

# Choose evaluation points
num_eval_points = 10
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)
evaluation_points = pinnx.utils.dict_to_array(evaluation_points)

# Setup DeepONet
dim_x = 1
p = 32
net = bst.nn.Sequential(
    pinnx.nn.DeepONetCartesianProd(
        [num_eval_points, 32, p],
        [dim_x, 32, p],
        activation="tanh",
    ),
    pinnx.nn.ArrayToDict(u=None)
)

# Define PDE operator
pde_op = pinnx.problem.PDEOperatorCartesianProd(
    geom,
    equation,
    bc,
    space,
    evaluation_points,
    approximator=net,
    num_function=100,
    num_domain=100,
    num_boundary=2
)

# Define and train trainer
model = pinnx.Trainer(pde_op)
model.compile(bst.optim.Adam(0.0005)).train(iterations=20000)
model.saveplot(isplot=True)

# Plot realisations of f(x)
n = 3
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))['u']

# Setup figure
fig = plt.figure(figsize=(7, 8))
plt.subplot(2, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

# Plot source term f(x)
for i in range(n):
    plt.plot(evaluation_points, fx[i], "--")

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)
for i in range(n):
    plt.plot(x, y[i], "-")
plt.xlabel("x")

plt.show()
