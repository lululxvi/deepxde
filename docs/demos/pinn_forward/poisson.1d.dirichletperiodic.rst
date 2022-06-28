Poisson equation in 1D with Dirichlet/Periodic boundary conditions
==================================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: -\Delta u = \pi^2\sin(\pi x), \qquad x \in [-1, 1],

with the Periodic boundary conditions on the right boundary

.. math:: u(0)=u(1)

and Dirichlet boundary conditions on the left boundary

.. math:: u(-1)=0.

The exact solution is :math:`u(x) = \sin(\pi x)`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Poisson equation step-by-step.

First, the DeepXDE and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)

Next, we express the PDE residual of the Poisson equation:

.. code-block:: python

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the Periodic boundary condition and Dirichlet boundary condition respectively.

The location of the Periodic boundary condition is defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`x=1` and ``False`` otherwise (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent). In this function, the argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the boundary of the geometry, in this case Periodic boundary when it reaches the right endpoint of the interval, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)


The location of the Dirichlet boundary condition is defined in a similar way that the function should return ``True`` for those points satisfying :math:`x=-1` and ``False`` otherwise. The arguments in this function are similar to ``boundary_r``, and the only difference is that in this case Dirichlet boundary condition is used when it reaches the left endpoint of the interval.

.. code-block:: python

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)


Next, we define a function to return the value of :math:`u(x)` for the points :math:`x` on the Dirichlet boundary. In this case, it is :math:`u(x)=0`. For example, :math:`\sin(\pi * x)` is 0 on the boundary, and thus we can also use


.. code-block:: python

    def func(x):
        return np.sin(np.pi * x)

Then, the Dirichlet boundary condition is defined as

.. code-block:: python

    bc1 = dde.icbc.DirichletBC(geom, func, boundary_l)

and the Periodic boundary condition is

.. code-block:: python

    bc2 = dde.icbc.PeriodicBC(geom, 0, boundary_r)

Now, we have specified the geometry, PDE residual, Dirichlet boundary condition and Periodic boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(geom, pde, [bc1, bc2], 16, 2, solution=func, num_test=100)

The number 16 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 100 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

We also compute the :math:`L^2` relative error as a metric during training.

We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=10000)

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_periodic_1d.py
  :language: python
