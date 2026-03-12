Inverse problem for the Klein-Gordon equation
================

Problem setup
--------------

We will solve an inverse problem for the Klein-Gordon equation to recover the unknown mass parameter :math:`m^2`:

.. math:: \frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + m^2 u = 0

for :math:`x \in [-1, 1]` and :math:`t \in [0, 1]` with Dirichlet boundary conditions

.. math:: u(-1, t) = u_{\text{exact}}(-1, t), \quad u(1, t) = u_{\text{exact}}(1, t)

and initial conditions

.. math:: u(x, 0) = \sin(\pi x), \quad \frac{\partial u}{\partial t}(x, 0) = 0.

The exact solution is :math:`u(x, t) = \sin(\pi x) \cos(\omega t)` where :math:`\omega = \sqrt{\pi^2 + m^2}`.

The true mass parameter is :math:`m^2 = 4`. We initialize at :math:`m^2 = 1` and recover the true value using 50 randomly sampled observation points in the interior of the domain.

Implementation
--------------

This description goes through the implementation of a solver for the above described inverse Klein-Gordon problem step-by-step.

We define the true mass parameter :math:`m^2 = 4` and the corresponding frequency :math:`\omega`:

.. code-block:: python

    m_sq_true = 4.0
    omega = np.sqrt(np.pi**2 + m_sq_true)

The learnable parameter :math:`m^2` is initialized far from its true value:

.. code-block:: python

    m_sq = dde.Variable(1.0)

We define the PDE residual :math:`u_{tt} - u_{xx} + m^2 u = 0`:

.. code-block:: python

    def pde(x, y):
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_tt - dy_xx + m_sq * y

The exact solution ``func`` is defined as:

.. code-block:: python

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.cos(omega * x[:, 1:2])

We set up the computational domain as a 1D interval crossed with a time domain:

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

The boundary and initial conditions are specified. We use Dirichlet BCs on the full boundary, and two initial conditions — one for the field value and one for the time derivative:

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda _, on_initial: on_initial,
    )

We create 50 sparse observation points as random ``(x, t)`` pairs in the interior to provide data for parameter recovery. The ``component=0`` argument specifies that the first (and only) network output must match the observations:

.. code-block:: python

    rng = np.random.default_rng(42)
    observe_x = np.column_stack(
        [rng.uniform(-1, 1, 50), rng.uniform(0, 1, 50)]  # columns are (x, t)
    )
    observe_y = func(observe_x)
    # component=0 means the first (and only) network output must match observations
    ptset = dde.icbc.PointSetBC(observe_x, observe_y, component=0)

The PDE problem is assembled with 2000 domain points, 200 boundary points, and 200 initial condition points:

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic_1, ic_2, ptset],
        num_domain=2000,
        num_boundary=200,
        num_initial=200,
        anchors=observe_x,
        solution=func,
        num_test=5000,
    )

We use a fully connected network with 3 hidden layers of width 40:

.. code-block:: python

    net = dde.nn.FNN([2] + [40] * 3 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

Phase 1 trains with Adam for 30,000 iterations. The ``VariableValue`` callback logs the current value of :math:`m^2` every 1,000 iterations to ``variables.dat``, letting us track how the parameter converges over training:

.. code-block:: python

    model.compile(
        "adam",
        lr=0.001,
        metrics=["l2 relative error"],
        external_trainable_variables=m_sq,
    )
    # Logs m_sq every 1000 iterations so we can plot its convergence trajectory
    variable = dde.callbacks.VariableValue(m_sq, period=1000, filename="variables.dat")
    losshistory, train_state = model.train(iterations=30000, callbacks=[variable])

Phase 2 refines with L-BFGS for further convergence:

.. code-block:: python

    model.compile(
        "L-BFGS",
        metrics=["l2 relative error"],
        external_trainable_variables=m_sq,
    )
    losshistory, train_state = model.train(callbacks=[variable])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Training results
-----------------

After 30,000 Adam iterations, :math:`m^2` converges from 1.0 to approximately 2.0 with an L2 relative error of ~4.4%. The subsequent L-BFGS phase (~5,900 additional iterations) drives :math:`m^2` to **4.00** (the exact true value) with a final L2 relative error of **0.047%**.

Total runtime is approximately 8 minutes on a GPU (NVIDIA RTX 4060, PyTorch backend):

- Adam (30k iterations): ~290 s
- L-BFGS (~5.9k iterations): ~180 s

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_inverse/Klein_Gordon_inverse.py
  :language: python
