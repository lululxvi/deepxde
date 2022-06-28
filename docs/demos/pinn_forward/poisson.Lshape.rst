Poisson equation over L-shaped domain
========================================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: - u_{xx}-u_{yy} = 1, \qquad  \Omega = [-1,1]^2 \backslash [0,1]^2

with the Dirichlet boundary conditions

.. math:: u(x,y)=0, \qquad (x,y)\in \partial \Omega

Implementation
--------------

This description goes through the implementation of a solver for the above described Poisson equation step-by-step.

First, the DeepXDE module is imported:

.. code-block:: python

    import deepxde as dde

We begin by defining a computational geometry. We can use a built-in class ``Polygon`` as follows

.. code-block:: python

    geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])

The geometry in this case is a polygon with six line segments, which is in a L-shape.

Next, we express the PDE residual of the Poisson equation:

.. code-block:: python

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return -dy_xx - dy_yy - 1

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the Dirichlet boundary condition. A simple Python function, returning a boolean, is used to define the subdomain for the Dirichlet boundary condition. The function should return ``True`` for those points inside the subdomain and ``False`` for the points outside. In our case, the points :math:`(x,y)` of the Dirichlet boundary condition are :math:`(x,y) \in \partial\{ [-1,1]^2\backslash [0,1]^2 \}`.

.. code-block:: python

    def boundary(_, on_boundary):
        return on_boundary

Then, the Dirichlet boundary condition is

.. code-block:: python

    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

Now, we have specified the geometry, PDE residual, and Dirichlet boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
The number 1200 is the number of training residual points sampled inside the domain, and the number 120 is the number of training points sampled on the boundary. We use 1500 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 5 (i.e., 4 hidden layers) and width 50. Besides, we choose ``tanh`` as activation function and ``Glorot uniform`` as initializer :

.. code-block:: python

    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)

We first train the model for 50000 iterations with Adam optimizer:

.. code-block:: python

    model.train(iterations=50000)

And then we train the model again using L-BFGS

.. code-block:: python

    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_Lshape.py
  :language: python
