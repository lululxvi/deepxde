Forced pendulum with unaligned points
=====================================

Problem setup
-------------

This example learns the solution operator

.. math::
   G: u \mapsto \theta

for the forced pendulum equation

.. math::

   \theta''(t) = -\sin(\theta(t)) + u(t), \qquad t \in [0, 1],

with initial conditions :math:`\theta(0)=0` and :math:`\theta'(0)=0`.

The forcing function :math:`u(t)` is sampled from a Gaussian random field (GRF). A physics-informed DeepONet is trained so that its prediction satisfies both the ODE residual and the initial conditions.

Implementation
--------------

The problem is posed on :math:`[0,1]`, with initial conditions :math:`\theta(0)=0` and :math:`\theta'(0)=0` enforced at :math:`t=0` via a Dirichlet condition and a Neumann condition:

.. code-block:: python

    geom = dde.geometry.TimeDomain(0, 1)
    ic1 = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_ic)
    ic2 = dde.icbc.NeumannBC(geom, lambda x: 0.0, boundary_ic)
    pde = dde.data.PDE(geom, pendulum_ode, [ic1, ic2], num_domain=200, num_boundary=20)

The forcing function is sampled from a Gaussian random field (GRF) at 50 sensor points, and ``dde.data.PDEOperator`` is constructed with 500 training functions and 100 test functions:

.. code-block:: python

    func_space = dde.data.GRF(length_scale=0.2)
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.data.PDEOperator(pde, func_space, eval_pts, 500, num_test=100)

A ``DeepONet`` is then defined with a branch net that takes the 50 sampled values of :math:`u` and a trunk net that takes the query time :math:`t`:

.. code-block:: python

    net = dde.nn.DeepONet(
        [50] + [32] * 3,
        [1] + [32] * 3,
        "tanh",
        "Glorot normal",
    )

We define a ``Model``, and then train it using Adam with learning rate of 0.0005 for 50,000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0005, metrics=["l2 relative error"])
    model.train(iterations=50000)

Numerical comparison
--------------------

For a sample forcing

.. math::
   u(t) = 0.5\sin(4\pi t),

the script computes a reference solution with a 4th-order Runge--Kutta solver, subject to the same initial conditions.

.. code-block:: python

    def solve_pendulum(u_func, t):
        theta = np.zeros(t.shape[0], dtype=float)
        theta_dot = np.zeros(t.shape[0], dtype=float)

        def rhs(t_local, state):
            th, om = state
            return np.array(
                [om, -np.sin(th) + u_func(t_local)],
                dtype=float,
            )

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

The RK4 trajectory can then be compared with the PI-DeepONet prediction.

Complete code
-------------

.. literalinclude:: ../../../examples/operator/gravity_pendulum_unaligned_pideeponet.py
  :language: python
