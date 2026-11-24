"""Backend supported: pytorch

The lowest eigenvalues and eigenfunctions of a Sturm-Liouville problem:

    -psi''(x) + V(x) psi(x) = E psi(x),    V(x) = x^4 / 4 + x,

with psi(-4.5) = psi(3.5) = 0. A network with K outputs learns the K eigenfunctions,
and K trainable variables learn the eigenvalues.
"""

import deepxde as dde
import numpy as np

# L-BFGS stops too early in float32
dde.config.set_default_float("float64")

# Number of eigenpairs
K = 3
a, b = -4.5, 3.5


def V(x):
    return x**4 / 4 + x


def reference_eigenvalues(N=100):
    """Chebyshev collocation with N + 1 points (cheb.m in L. N. Trefethen, Spectral
    Methods in MATLAB, SIAM, 2000). Accurate to about 11 digits."""
    t = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    T = np.tile(t, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (T - T.T + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    # Map [-1, 1] to [a, b], and drop the end points, where psi = 0
    x = (b - a) / 2 * t + (a + b) / 2
    D2 = (2 / (b - a)) ** 2 * (D @ D)[1:-1, 1:-1]
    H = -D2 + np.diag(V(x[1:-1]))
    return np.sort(np.linalg.eigvals(H).real)[:K]


# Unknown eigenvalues, with initial guess 0
E = [dde.Variable(0.0) for _ in range(K)]


def pde(x, psi):
    residuals = [
        -dde.grad.hessian(psi, x, component=k) + (V(x) - E[k]) * psi[:, k : k + 1]
        for k in range(K)
    ]
    # Orthonormality: the integral of psi_j * psi_k over [a, b] is 1 if j = k and 0
    # otherwise. This also rules out the trivial solution psi = 0. The points are a
    # uniform grid inside (a, b) and psi = 0 at both ends, so the trapezoidal rule
    # is h times the sum over the points.
    h = (b - a) / (len(x) + 1)
    orthonormality = [
        h * dde.backend.reduce_sum(psi[:, j] * psi[:, k]) - float(j == k)
        for j in range(K)
        for k in range(j, K)
    ]
    # Repeated at every point, like the other residuals
    orthonormality = dde.backend.reshape(dde.backend.stack(orthonormality, 0), [1, -1])
    return residuals + [orthonormality + 0 * x]


geom = dde.geometry.Interval(a, b)
data = dde.data.PDE(
    geom, pde, [], num_domain=400, num_test=1000, train_distribution="uniform"
)

net = dde.nn.FNN([1] + [32] * 3 + [K], "tanh", "Glorot normal")


# Exact boundary conditions psi(a) = psi(b) = 0
def output_transform(x, y):
    return (x - a) * (b - x) / 16 * y


net.apply_output_transform(output_transform)

model = dde.Model(data, net)

variable = dde.callbacks.VariableValue(E, period=1000, precision=6)
# A large weight, so that the outputs stay orthonormal
weights = [1] * K + [1000]
model.compile("adam", lr=0.001, loss_weights=weights, external_trainable_variables=E)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
model.compile("L-BFGS", loss_weights=weights, external_trainable_variables=E)
losshistory, train_state = model.train(callbacks=[variable])

# The outputs come in no particular order: sort them by eigenvalue
E_pred = np.array([dde.backend.to_numpy(e) for e in E])
order = np.argsort(E_pred)
E_ref = reference_eigenvalues()

# Sturm's oscillation theorem: the n-th eigenfunction has exactly n zeros.
# Count sign changes, ignoring the tails where |psi| < 0.1% of its maximum.
x = geom.uniform_points(1000)
psi = model.predict(x)
for n, k in enumerate(order):
    p = psi[np.abs(psi[:, k]) > 1e-3 * np.abs(psi[:, k]).max(), k]
    zeros = np.sum(np.sign(p[1:]) != np.sign(p[:-1]))
    error = abs(E_pred[k] - E_ref[n]) / E_ref[n]
    print(
        f"E_{n} = {E_pred[k]:.8f} (reference {E_ref[n]:.8f}), "
        f"relative error {error:.1e}, zeros: {zeros}"
    )

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
