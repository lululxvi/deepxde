import brainstate as bst
import brainunit as u
import numpy as np
from scipy.special import jv, hankel1

import deepxde.experimental as deepxde

# General parameters
weights = 1
iterations = 10000
learning_rate = 1e-3
num_dense_layers = 3
num_dense_nodes = 350
activation = "tanh"

# Problem parameters
k0 = 2
wave_len = 2 * np.pi / k0
length = 2 * np.pi
R = np.pi / 4
n_wave = 20
h_elem = wave_len / n_wave
nx = int(length / h_elem)

# Computational domain
outer = deepxde.geometry.Rectangle([-length / 2, -length / 2], [length / 2, length / 2])
inner = deepxde.geometry.Disk([0, 0], R)
inner_geom = inner.to_dict_point('x', 'y')
outer_geom = outer.to_dict_point('x', 'y')
geom = (outer - inner).to_dict_point('x', 'y')


# Definition of the pde
def pde(x, y):
    hessian = net.hessian(x)

    y0, y1 = y['y0'], y['y1']
    y0_xx = hessian['y0']['x']['x']
    y0_yy = hessian['y0']['y']['y']
    y1_xx = hessian['y1']['x']['x']
    y1_yy = hessian['y1']['y']['y']

    return [-y0_xx - y0_yy - k0 ** 2 * y0, -y1_xx - y1_yy - k0 ** 2 * y1]


def boundary_outer(x, on_boundary):
    return u.math.logical_and(on_boundary, outer_geom.on_boundary(x))


def boundary_inner(x, on_boundary):
    return u.math.logical_and(on_boundary, inner_geom.on_boundary(x))


def inner_bc(x):
    normal = inner_geom.boundary_normal(x)
    g = 1j * k0 * u.math.exp(1j * k0 * x['x']) * -normal['x']
    y0 = u.math.real(-g)

    g = 1j * k0 * u.math.exp(1j * k0 * x['x']) * -normal['x']
    y1 = u.math.imag(-g)

    return {'y0': y0, 'y1': y1}


def outer_bc(x, y):
    y0 = -k0 * y['y1']
    y1 = k0 * y['y0']
    return {'y0': y0, 'y1': y1}


# ABCs
bc_inner = deepxde.icbc.NeumannBC(inner_bc, boundary_inner)
bc_outer = deepxde.icbc.RobinBC(outer_bc, boundary_outer)

loss_weights = [1, 1, weights, weights]

net = deepxde.nn.Model(
    deepxde.nn.DictToArray(x=None, y=None),
    deepxde.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [2], activation),
    deepxde.nn.ArrayToDict(y0=None, y1=None),
)


# Exact solution
def sound_hard_circle(points):
    fem_xx = points['x']
    fem_xy = points['y']
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, axis=0)
    n_terms = int(30 + (k0 * R) ** 1.01)

    u_sc = np.zeros((npts,), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n - 1, k0 * R) - n / (k0 * R) * jv(n, k0 * R)
        hankel_deriv = n / (k0 * R) * hankel1(n, k0 * R) - hankel1(n + 1, k0 * R)
        u_sc += (
            -(1j ** n)
            * (bessel_deriv / hankel_deriv)
            * hankel1(n, k0 * r)
            * np.exp(1j * n * theta)
        ).ravel()
    return u_sc


def sol(x):
    result = sound_hard_circle(x)
    real = np.real(result)
    imag = np.imag(result)
    return {'y0': real, 'y1': imag}


problem = deepxde.problem.PDE(
    geom,
    pde,
    [bc_inner, bc_outer],
    net,
    num_domain=nx ** 2,
    num_boundary=8 * nx,
    num_test=5 * nx ** 2,
    solution=sol,
    loss_weights=loss_weights,
)

trainer = deepxde.Trainer(problem)
trainer.compile(bst.optim.Adam(learning_rate), metrics=["l2 relative error"]).train(iterations=iterations)
trainer.saveplot(issave=True, isplot=True)
