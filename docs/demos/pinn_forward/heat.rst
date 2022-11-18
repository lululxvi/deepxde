Heat equation
=============

Problem setup
-------------

We will solve a heat equation:

.. math:: \frac{\partial u}{\partial t}=\alpha \frac{\partial^2u}{\partial x^2}, \qquad x \in [-1, 1], \quad t \in [0, 1]

where :math:`\alpha=0.4` is the thermal diffusivity constant.

With Dirichlet boundary conditions:

.. math:: u(0,t) = u(1,t)=0,

and periodic(sinusoidal) inital condition:

.. math:: u(x,0) = \sin (\frac{n\pi x}{L}),\qquad 0<x<L, \quad n = 1,2,.....

where :math:`L=1` is the length of the bar, :math:`n=1` is the frequency of the sinusoidal initial conditions.

The exact solution is :math:`u(x,t) = e^{\frac{-n ^2\pi ^2 \alpha t}{L^2}}\sin (\frac{n\pi x}{L})`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Heat equation step-by-step.

First, the DeepXDE are imported:

.. code-block:: python

    import deepxde as dde

We begin by defining the parameters of the equation:

.. code-block:: python

    a = 0.4
    L = 1
    n = 1

Next, we define a computational geometry and time domain. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(0, L)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Next, we express the PDE residual of the Heat equation:

.. code-block:: python

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - a * dy_xx

The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0]``) is :math:`x`-coordinate and the second componenet (``x[:,1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x,t)`, but here we use ``y`` as the name of the variable.

Next, we consider the boundary/initial condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain in considered as the boundary condition. We include the ``geomtime`` space, time geometry created above and ``on_boundary`` as the BCs in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the inital condition for the burgers equation and we use the computational domain, initial function, and ``on_initial`` to specify the IC. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
        lambda _, on_initial: on_initial,
    )

Now, we have specified the geometry, PDE residual, and boundary/initial condition. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        num_test=2540,
    )

The number 2540 is the number of training residual points sampled inside the domain, and the number 80 is the number of training points sampled on the boundary. We also include 160 initial residual points for the initial conditions.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:

.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    
   
We then train the model for 20000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=20000)
    
After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS-B")
    losshistory, train_state = model.train() 

Complete code
-------------

.. literalinclude:: ../../../examples/pinn_forward/heat.py
  :language: python
