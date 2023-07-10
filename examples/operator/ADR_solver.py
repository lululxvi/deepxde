import matplotlib.pyplot as plt
import numpy as np


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def main():
    xmin, xmax = -1, 1
    tmin, tmax = 0, 1
    k = lambda x: x**2 - x**2 + 1
    v = lambda x: np.ones_like(x)
    g = lambda u: u**3
    dg = lambda u: 3 * u**2
    f = (
        lambda x, t: np.exp(-t) * (1 + x**2 - 2 * x)
        - (np.exp(-t) * (1 - x**2)) ** 3
    )
    u0 = lambda x: (x + 1) * (1 - x)
    u_true = lambda x, t: np.exp(-t) * (1 - x**2)

    # xmin, xmax = 0, 1
    # tmin, tmax = 0, 1
    # k = lambda x: np.ones_like(x)
    # v = lambda x: np.zeros_like(x)
    # g = lambda u: u ** 2
    # dg = lambda u: 2 * u
    # f = lambda x, t: x * (1 - x) + 2 * t - t ** 2 * (x - x ** 2) ** 2
    # u0 = lambda x: np.zeros_like(x)
    # u_true = lambda x, t: t * x * (1 - x)

    Nx, Nt = 100, 100
    x, t, u = solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    plt.plot(x, u)
    plt.show()


if __name__ == "__main__":
    main()
