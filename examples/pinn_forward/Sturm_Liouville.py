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
# Reference eigenvalues (Chebyshev collocation, 12 digits)
E_ref = [0.354123262748, 2.336634228099, 4.644604790037]

a, b = -4.5, 3.5

# Unknown eigenvalues, with initial guess 0
E = [dde.Variable(0.0) for _ in range(K)]


def pde(x, psi):
    residuals = [
        -dde.grad.hessian(psi, x, component=k)
        + (x**4 / 4 + x - E[k]) * psi[:, k : k + 1]
        for k in range(K)
    ]
    # Orthonormality: the integral of psi_j * psi_k over [a, b] is 1 if j = k and 0
    # otherwise. The integrals are approximated by the mean over the points.
    # This also rules out the trivial solution psi = 0.
    orthonormality = [
        (b - a) * dde.backend.reduce_mean(psi[:, j] * psi[:, k]) - float(j == k)
        for j in range(K)
        for k in range(j, K)
    ]
    # Repeated at every point, like the other residuals
    orthonormality = dde.backend.reshape(dde.backend.stack(orthonormality, 0), [1, -1])
    return residuals + [orthonormality + 0 * x]


geom = dde.geometry.Interval(a, b)
data = dde.data.PDE(geom, pde, [], num_domain=400, num_test=1000)

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

# Sturm's oscillation theorem: the n-th eigenfunction has exactly n zeros.
# Count sign changes, ignoring the tails where |psi| < 0.1% of its maximum.
x = geom.uniform_points(1000)
psi = model.predict(x)
for n, k in enumerate(order):
    p = psi[np.abs(psi[:, k]) > 1e-3 * np.abs(psi[:, k]).max(), k]
    zeros = np.sum(np.sign(p[1:]) != np.sign(p[:-1]))
    error = abs(E_pred[k] - E_ref[n]) / E_ref[n]
    print(f"E_{n} = {E_pred[k]:.8f}, relative error {error:.1e}, zeros: {zeros}")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
