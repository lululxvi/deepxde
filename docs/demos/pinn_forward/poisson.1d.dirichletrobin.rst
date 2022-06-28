Poisson equation in 1D with Dirichlet/Robin boundary conditions
=================================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: \Delta u = 2, \qquad x \in [-1, 1],

with the Robin boundary conditions on the right boundary

.. math:: \frac{du}{dx} = u

and Dirichlet boundary conditions on the left boundary

.. math:: u(-1) = 0.

The exact solution is :math:`u(x) = (x+1)^2`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Poisson equation step-by-step.
First, the DeepXDE module is imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)

Next, we express the PDE residual of the Poisson equation:

.. code-block:: python

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return dy_xx - 2

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the Robin boundary condition and Dirichlet boundary condition respectively.

The location of the Robin boundary condition is defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`x=1` and ``False`` otherwise (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent). In this function, the argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the boundary of the geometry, in this case Robin boundary when it reaches the right endpoint of the interval, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)


The location of the Dirichlet boundary condition is defined in a similar way that the function should return ``True`` for those points satisfying :math:`x=-1` and ``False`` otherwise. The arguments in this function are similar to ``boundary_r``, and the only difference is that in this case Dirichlet boundary condition is used when it reaches the left endpoint of the interval.

.. code-block:: python

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)


Next, we define a function to return the value of :math:`u(x)` for the points :math:`x` on the Dirichlet boundary. In this case, it is :math:`u(x)=0`. For example, :math:`(x+1)^2` is 0 on the boundary, and thus we can also use


.. code-block:: python

    def func(x):
        return (x + 1) ** 2

Then, the Dirichlet boundary condition is

.. code-block:: python

    bc_l = dde.icbc.DirichletBC(geom, func, boundary_1)

For Robin boundary condition, rather than define a function to return the value of :math:`u(x)` on the boundary, we use a lambda function that maps ``x`` and ``y`` to ``y``, where x is the input and y is the output.  Then Robin boundary condition is defined

.. code-block:: python

    bc_r = dde.icbc.RobinBC(geom, lambda X, y: y, boundary_r)

Now, we have specified the geometry, PDE residual, Dirichlet boundary condition and Robin boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)

The number 16 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 100 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We bulid a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

We also compute the :math:`L^2` relative error as a metric during training.

We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=10000)

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_Robin_1d.py
  :language: python
