Black-Scholes Equation
======================

Problem Setup
-------------

We solve the Black-Scholes partial differential equation for pricing a European call option.
The problem is formulated in terms of **time-to-maturity** :math:`\tau = T - t`.

The PDE is given by:

.. math::
    \frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V, \qquad S \in [0, S_{max}], \quad \tau \in [0, T]

where:
 - :math:`V(S, \tau)` is the option price
 - :math:`S` is the underlying asset price
 - :math:`\tau` is the time to maturity
 - :math:`\sigma=0.2` is the volatility
 - :math:`r=0.05` is the risk-free interest rate

The boundary and initial conditions are:

1. **Initial Condition** (at maturity :math:`\tau=0`):

.. math:: V(S, 0) = \max(S - K, 0)

where :math:`K=50` is the strike price.

2. **Boundary Conditions**:

.. math::
    V(0, \tau) &= 0 \\
    V(S_{max}, \tau) &= S_{max} - K e^{-r\tau}

Implementation
--------------

This example uses a Physics-Informed Neural Network (PINN) to solve the equation.

First, we import DeepXDE and define the problem parameters:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from scipy.stats import norm

    K = 50.0      # Strike price
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    T = 1.0       # Time to maturity
    S_max = 150.0 # Maximum stock price

We define the computational geometry. We use ``GeometryXTime`` to combine the asset price interval :math:`[0, 150]` and the time domain :math:`[0, 1]`:

.. code-block:: python

    geom = dde.geometry.Interval(0, S_max)
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

The PDE residual is defined below. Here ``x[:, 0]`` is the asset price
:math:`S`, ``x[:, 1]`` is the time to maturity :math:`\tau`, and ``y``
is the option price :math:`V(S, \tau)` output by the network.

.. code-block:: python

    def pde(x, y):
        # x[:, 0]: asset price S
        # x[:, 1]: time to maturity tau
        # y:       option price V(S, tau)
        S = x[:, 0:1]
        dV_tau = dde.grad.jacobian(y, x, i=0, j=1)
        dV_S   = dde.grad.jacobian(y, x, i=0, j=0)
        dV_SS  = dde.grad.hessian(y, x, i=0, j=0)
        return dV_tau - (0.5 * sigma**2 * S**2 * dV_SS + r * S * dV_S - r * y)


For the boundary conditions, we define the payoff function at maturity (:math:`\tau=0`) and the Dirichlet conditions at :math:`S=0` and :math:`S=S_{max}`:

.. code-block:: python

    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.maximum(x[:, 0:1] - K, 0.0),
        lambda _, on_initial: on_initial,
    )
    bc_lower = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )
    bc_upper = dde.icbc.DirichletBC(
        geomtime,
        lambda x: x[:, 0:1] - K * np.exp(-r * x[:, 1:2]),
        lambda x, on_boundary: on_boundary and np.isclose(x[0], S_max),
    )

**Sampling Strategy**:
To improve domain coverage and training stability, we use the **Sobol sequence** for sampling points. We select sample sizes as powers of 2 to avoid Sobol sequence warnings (2048 domain points, 64 boundary points, 128 initial points):

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic, bc_lower, bc_upper],
        num_domain=2047, # 2048 - 1 center point
        num_boundary=63,
        num_initial=127,
        num_test=2047,
        train_distribution="Sobol",
    )

**Network Architecture**:
We use a fully connected neural network with **4 hidden layers** and **28 neurons** per layer, using ``tanh`` activation and ``Glorot normal`` initialization:

.. code-block:: python

    net = dde.nn.FNN([2] + [28] * 4 + [1], "tanh", "Glorot normal")

**Training Strategy**:
We employ a two-stage training process: first using the **Adam** optimizer for 20,000 iterations, followed by **L-BFGS** to fine-tune the solution and achieve lower residual error.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=20000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

Validation
----------

The model is validated against the analytical Black-Scholes formula. We expect an L2 relative error around ``1e-3`` and a mean PDE residual around ``0.05``.

References
----------

.. [1] Tanios, R. (2021). "Physics Informed Neural Networks in Computational Finance: High Dimensional Forward & Inverse Option Pricing". ETH Zurich Master's Thesis. https://doi.org/10.3929/ethz-b-000491555

Complete code
-------------

.. literalinclude:: ../../../examples/pinn_forward/black_scholes_call.py
   :language: python

