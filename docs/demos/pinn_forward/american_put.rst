American Put Option
===================

Problem Setup
-------------

We solve the Black-Scholes partial differential equation for pricing an American put option.
Unlike European options, American options can be exercised at any time before maturity. This leads to a free-boundary problem, which is formulated as a Linear Complementarity Problem (LCP). 
The problem is formulated in terms of **time-to-maturity** :math:`\tau = T - t`.

The LCP is given by:

.. math::
    L(V) &= -\frac{\partial V}{\partial \tau} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \le 0 \\
 
    V(S, \tau) &\ge \max(K - S, 0) \\

    (V(S, \tau) - \max(K - S, 0)) \cdot L(V) &= 0

where:
 - :math:`V(S, \tau)` is the option price
 - :math:`S` is the underlying asset price
 - :math:`\tau` is the time to maturity
 - :math:`K=1.0` is the strike price
 - :math:`\sigma=0.2` is the volatility
 - :math:`r=0.05` is the risk-free interest rate

The boundary and initial conditions are:

1. **Initial Condition** (at maturity :math:`\tau=0`):

.. math:: V(S, 0) = \max(K - S, 0)

2. **Boundary Conditions**:

.. math::
    V(0, \tau) &= K \\
    V(S_{max}, \tau) &= 0

where :math:`S_{max} = 4.0` limits the domain to a reasonable multiple of the strike price.

Implementation
--------------

This example uses a Physics-Informed Neural Network (PINN) to solve the American put LCP using the **Fischer-Burmeister** function.

First, we import DeepXDE, PyTorch, and define the problem parameters:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    import torch

    K = 1.0       # Strike price
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    T = 1.0       # Time to maturity
    S_max = 4.0   # Maximum stock price

We define the computational geometry. We use ``GeometryXTime`` to combine the asset price interval :math:`[0, 4.0]` and the time domain :math:`[0, 1.0]`:

.. code-block:: python

    geom = dde.geometry.Interval(0, S_max)
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

For the American put, the PDE residual must account for the early exercise constraint. **This is very important to make sure the model does not learn European Options.**
We formulate this using soft penalties and the Fischer-Burmeister equation :math:`\phi(a, b) = \sqrt{a^2 + b^2} - (a + b) = 0` to satisfy the complementarity condition:

.. code-block:: python

    def pde(x, y):
        s = x[:, 0:1]
        v = y
        
        dv_dtau = dde.grad.jacobian(y, x, i=0, j=1) 
        dv_ds = dde.grad.jacobian(y, x, i=0, j=0)
        dv_ds2 = dde.grad.hessian(y, x, i=0, j=0)
        
        # Black-Scholes Operator
        bs_op = -dv_dtau + 0.5 * (sigma**2) * (s**2) * dv_ds2 + r * s * dv_ds - r * v
        
        # Intrinsic Value (Payoff)
        payoff = dde.backend.relu(K - s)
        
        # LCP Constraints
        res_pde = dde.backend.relu(bs_op)            # Soft penalty for L(V) > 0
        penalty = dde.backend.relu(payoff - v)       # Soft penalty for V < Payoff
        
        # Fischer-Burmeister Complementarity
        a = v - payoff
        b = -bs_op 
        # Add 1e-8 for numerical stability of the derivative of sqrt at 0
        res_fb = torch.sqrt(a**2 + b**2 + 1e-8) - (a + b)
        
        return [res_pde, penalty, res_fb]

For the boundary conditions, we define the payoff function at maturity (:math:`\tau=0`) and the Dirichlet conditions at :math:`S=0` and :math:`S=S_{max}`:

.. code-block:: python

    def initial_condition(x):
        s = x[:, 0:1]
        return np.maximum(K - s, 0)

    IC = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

    LB = dde.icbc.DirichletBC(
        geomtime, 
        lambda x: K, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0), 
        component=0
    )
    HB = dde.icbc.DirichletBC(
        geomtime, 
        lambda x: 0, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], S_max), 
        component=0
    )

**Sampling Strategy**:
To perfectly capture the non-differentiable kink at the strike price, we densely sample **anchor points** around :math:`S=1.0`. These are combined with the standard domain and boundary points:

.. code-block:: python

    S_anchors = np.random.uniform(0.5 * K, 1.5 * K, (1000, 1))
    tau_anchors = np.random.uniform(0, T, (1000, 1))
    anchors = np.hstack((S_anchors, tau_anchors))

    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [IC, LB, HB],
        num_domain=6000, 
        num_boundary=1500, 
        num_initial=1500,
        anchors=anchors
    )

**Network Architecture**:
We use a fully connected neural network with **4 hidden layers** and **128 neurons** per layer, increasing the width compared to the European option to handle the sharper Fischer-Burmeister gradients:

.. code-block:: python

    net = dde.nn.FNN([2] + [128] * 4 + [1], "tanh", "Glorot normal")

**Training Strategy**:
We employ a two-stage training process. First, we use the **Adam** optimizer for 12,000 iterations, heavily weighting the boundary conditions and the intrinsic payoff penalty. Then, we use **L-BFGS** with a ``PDEPointResampler`` to adaptively refine the solution.

.. code-block:: python

    model = dde.Model(data, net)
    
    # Heavily weight the initial/boundary conditions and the payoff penalty
    model.compile("adam", lr=0.001, loss_weights=[1, 10, 5, 100, 100, 100])
    model.train(iterations=12000)
    
    model.compile("L-BFGS")
    checker = dde.callbacks.PDEPointResampler(period=1000)
    losshistory, train_state = model.train(callbacks=[checker])

Validation
----------

The model is validated against a standard **Binomial Tree** pricing model. Evaluating the results alongside the analytical binomial ground truth allows us to compute the relative L2 error and visually confirm the model captures the early exercise premium.

Complete code
-------------

.. literalinclude:: ../../../examples/pinn_forward/american_put.py
   :language: python