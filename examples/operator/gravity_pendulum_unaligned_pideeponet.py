"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

sin = dde.backend.sin


def pde(x, y, v):
    # theta''(t) = -sin(theta(t)) + u(t)
    dy_tt = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt + sin(y) - v


def boundary_ic(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


geom = dde.geometry.TimeDomain(0, 1)
ic1 = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_ic)  # theta(0) = 0
ic2 = dde.icbc.NeumannBC(geom, lambda x: 0.0, boundary_ic)  # theta'(0) = 0
pde = dde.data.PDE(geom, pde, [ic1, ic2], num_domain=200, num_boundary=20)
func_space = dde.data.GRF(length_scale=0.2)
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperator(pde, func_space, eval_pts, 500, num_test=100)
net = dde.nn.DeepONet(
    [50] + [32] * 3,
    [1] + [32] * 3,
    "tanh",
    "Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
model.train(iterations=50000)


def solve_pendulum(u_func, t):
    """
    Solve the forced pendulum ODE using a 4th-order Runge–Kutta method

    theta''(t) = -sin(theta(t)) + u(t),
    theta(0) = 0,
    theta'(0) = 0.
    """

    # State variables at each grid point
    theta = np.zeros(t.shape[0], dtype=float)  # angle
    theta_dot = np.zeros(t.shape[0], dtype=float)  # angular velocity

    def rhs(t_local, state):
        th, om = state
        return np.array(
            [om, -np.sin(th) + u_func(t_local)],
            dtype=float,
        )

    # RK4 time stepping
    for n in range(t.shape[0] - 1):
        dt = t[n + 1] - t[n]
        y_n = np.array([theta[n], theta_dot[n]], dtype=float)
        k1 = rhs(t[n], y_n)
        k2 = rhs(t[n] + 0.5 * dt, y_n + 0.5 * dt * k1)
        k3 = rhs(t[n] + 0.5 * dt, y_n + 0.5 * dt * k2)
        k4 = rhs(t[n] + dt, y_n + dt * k3)
        y_next = y_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        theta[n + 1], theta_dot[n + 1] = y_next
    return theta


def u_sin(t_local):
    # example forcing function u(t)
    # sinusoid, amplitude 0.5, freq 2.0 Hz
    return 0.5 * np.sin(4.0 * np.pi * t_local)


# Compare solver and PI-DeepONet
t = np.arange(0, 1, 0.01)
v = u_sin(eval_pts)
solved_theta = solve_pendulum(u_sin, t)
predicted_theta = model.predict((np.tile(v.T, (t.size, 1)), t.reshape(-1, 1)))

plt.figure()
plt.plot(t, solved_theta, label="theta(t) [solver]")
plt.plot(t, predicted_theta, label="theta(t) [PI-DeepONet]")
plt.xlabel("t")
plt.ylabel("theta")
plt.legend()
plt.tight_layout()
plt.show()
