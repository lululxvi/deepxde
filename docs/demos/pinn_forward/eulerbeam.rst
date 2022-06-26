Euler beam
==========

Problem setup
--------------

We will solve a Euler beam problem:

.. math:: \frac{\partial^{4} u}{\partial x^4} + 1 = 0, \qquad x \in [0, 1],

with two boundary conditions on the right boundary,

.. math:: u''(1)=0,   u'''(1)=0

and one Dirichlet boundary condition on the left boundary,

.. math:: u(0)=0

along with one Neumann boundary condition on the left boundary,

.. math:: u'(0)=0

The exact solution is :math:`u(x) = -\frac{1}{24}x^4+\frac{1}{6}x^3-\frac{1}{4}x^2`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Euler beam problem step-by-step.

First, the DeepXDE is imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(0, 1)

The Hessian matrix and the Jacobian maxtrix are defined to calculate the second and the third derivatives respectively.

.. code-block:: python

    def ddy(x, y):
        return dde.grad.hessian(y, x)
    
    def dddy(x, y):
        return dde.grad.jacobian(ddy(x, y), x)

Next, we express the PDE residual of the Poisson equation. 

.. code-block:: python

    def pde(x, y):
        dy_xx = ddy(x, y)
        dy_xxxx = dde.grad.hessian(dy_xx, x)
        return dy_xxxx + 1

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the left and right boundary condition respectively.

Two boundary conditions on the left including one Dirichlet boundary condition and one Neumann boundary condition are employed. The location of the left boundary condition is defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`x=0` and ``False`` otherwise (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent). In this function, the argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the boundary of the geometry, in this case Periodic boundary when it reaches the left endpoint of the interval, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

Two boundary conditions applied on the right boundary. The location of these two boundary condition is defined in a similar way that the function should return ``True`` for those points satisfying :math:`x=1` and ``False`` otherwise. The arguments in this function are similar to ``boundary_l``, and the only difference is that in these case general operator boundary conditions are used when it reaches the right endpoint of the interval.

.. code-block:: python

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

Next, for better comparsion, we define a function which is the exact solution to the problem.

.. code-block:: python

    def func(x):
        return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

The Dirichlet boundary condition and the Neumann boundary condition on the left are defined as following.

.. code-block:: python

    bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)

The right boundaries in this problem are of higher order so that the Hessian matrix and the Jacobian maxtrix are utilized when calculating the right boundary conditions. The right boundary is defined as,

.. code-block:: python

    bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

Now, we have specified the geometry, PDE residual and boundary conditions. We then define the PDE problem as

.. code-block:: python
 
    data = dde.data.PDE(
        geom,
        pde,
        [bc1, bc2, bc3, bc4],
        num_domain=10,
        num_boundary=2,
        solution=func,
        num_test=100,
    )

The number 10 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 100 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:

.. code-block:: python

    layer_size = [1] + [20] * 3 + [1]
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

.. literalinclude:: ../../../examples/pinn_forward/Euler_beam.py
  :language: python
