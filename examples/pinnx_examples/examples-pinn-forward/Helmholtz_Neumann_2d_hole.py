import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import deepxde.experimental as deepxde

# General parameters
n = 1
length = 1
R = 1 / 4

precision_train = 15
precision_test = 30

weight_inner = 10
weight_outer = 100
iterations = 5000
learning_rate = 1e-3
num_dense_layers = 3
num_dense_nodes = 350
activation = u.math.sin

k0 = 2 * np.pi * n
wave_len = 1 / n


def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["y"]["x"]["x"]
    dy_yy = hessian["y"]["y"]["y"]
    f = k0 ** 2 * u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])
    return -dy_xx - dy_yy - k0 ** 2 * y['y'] - f


def func(x):
    x = deepxde.array_to_dict(x, ['x', 'y'])
    return np.sin(k0 * x['x']) * np.sin(k0 * x['y'])


def neumann(x):
    x_ = deepxde.array_to_dict(x, ['x', 'y'])
    grad = np.array(
        [
            k0 * np.cos(k0 * x_['x']) * np.sin(k0 * x_['y']),
            k0 * np.sin(k0 * x_['x']) * np.cos(k0 * x_['y']),
        ]
    )

    normal = -inner.boundary_normal(x)
    normal = np.array(normal).T
    result = np.sum(grad * normal, axis=0)
    return result


outer = deepxde.geometry.Rectangle([-length / 2, -length / 2], [length / 2, length / 2])
inner = deepxde.geometry.Disk([0, 0], R)
geom = outer - inner
geom = deepxde.geometry.DictPointGeometry(geom, 'x', 'y')


def boundary_outer(x, on_boundary):
    return u.math.logical_and(on_boundary, outer.on_boundary(deepxde.utils.dict_to_array(x)))


def boundary_inner(x, on_boundary):
    return u.math.logical_and(on_boundary, inner.on_boundary(deepxde.utils.dict_to_array(x)))


hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

bc_inner = deepxde.icbc.NeumannBC(neumann, boundary_inner)
bc_outer = deepxde.icbc.DirichletBC(func, boundary_outer)

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, y=None),
    deepxde.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [1], activation),
    deepxde.nn.ArrayToDict(y=None),
)

data = deepxde.problem.PDE(
    geom,
    pde,
    [bc_inner, bc_outer],
    net,
    num_domain=nx_train ** 2,
    num_boundary=16 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
    loss_weights=[1, weight_inner, weight_outer]
)

trainer = deepxde.Trainer(data)
trainer.compile(bst.optim.Adam(learning_rate), metrics=["l2 relative error"]).train(iterations=iterations)
trainer.saveplot(issave=True, isplot=True)

# Plot the solution over a square grid with 100 points per wavelength in each direction
Nx = int(np.ceil(wave_len * 100))
Ny = Nx

# Grid points
xmin, xmax, ymin, ymax = [-length / 2, length / 2, -length / 2, length / 2]
plot_grid = np.mgrid[xmin: xmax: Nx * 1j, ymin: ymax: Ny * 1j]
points = np.vstack(
    (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
)

points_2d = points[:2, :]
u = trainer.predict(points[:2, :].T)
u = u.reshape((Nx, Ny))

ide = np.sqrt(points_2d[0, :] ** 2 + points_2d[1, :] ** 2) < R
ide = ide.reshape((Nx, Nx))

u_exact = func(points.T)
u_exact = u_exact.reshape((Nx, Ny))
diff = u_exact - u
error = np.linalg.norm(diff) / np.linalg.norm(u_exact)
print("Relative error = ", error)

plt.rc("font", family="serif", size=22)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(24, 12))

matrix = np.fliplr(u).T
matrix = np.ma.masked_where(ide, matrix)
pcm = ax1.imshow(
    matrix,
    extent=[-length / 2, length / 2, -length / 2, length / 2],
    cmap=plt.cm.get_cmap("seismic"),
    interpolation="spline16",
    label="PINN",
)

fig.colorbar(pcm, ax=ax1)

matrix = np.fliplr(u_exact).T
matrix = np.ma.masked_where(ide, matrix)
pcm = ax2.imshow(
    matrix,
    extent=[-length / 2, length / 2, -length / 2, length / 2],
    cmap=plt.cm.get_cmap("seismic"),
    interpolation="spline16",
    label="Exact",
)

ax1.set_title("PINNs")
ax2.set_title("Exact")
fig.colorbar(pcm, ax=ax2)

# Add the boundary normal vectors
p = inner.random_boundary_points(16 * nx_train)
px, py = p.T
nx, ny = inner.boundary_normal(p).T
ax1.quiver(px, py, nx, ny)
ax2.quiver(px, py, nx, ny)
# plt.savefig("plot_manufactured.pdf")
plt.show()
