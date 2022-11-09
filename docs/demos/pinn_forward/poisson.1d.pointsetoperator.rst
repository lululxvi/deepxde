Poisson equation in 1D with Dirichlet/PointSetOperator boundary conditions
=================================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: \Delta u = 2, \qquad x \in [-1, 1],

with the Neumann boundary conditions on the right boundary

.. math:: \left.\dfrac{du}{dx}\right|_{x=1} =4

and Dirichlet boundary conditions on the left boundary

.. math:: u(-1)=0.

The exact solution is :math:`u(x) = (x+1)^2`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Poisson equation step-by-step.
First, the DeepXDE and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    from deepxde.backend import tf

We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)

Next, we express the PDE residual of the Poisson equation:

.. code-block:: python

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return dy_xx - 2

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the Dirichlet boundary condition (BC) and Neumann boundary condition (BC) respectively.

The Dirichlet boundary conditionis defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`x=0` and ``False`` otherwise (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent). In this function, the argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the boundary of the geometry, in this case Dirichlet boundary when it reaches the left endpoint of the interval, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)


Next, we define a function to return the value of :math:`u(x)` for the points :math:`x` on the Dirichlet boundary. In this case, it is :math:`u(x)=0`. For example, :math:`(x+1)^2` is 0 on the boundary, and thus we can also use


.. code-block:: python

    def func(x):
        return (x + 1) ** 2

Then, the Dirichlet boundary condition is

.. code-block:: python

    bc_l = dde.icbc.DirichletBC(geom, func, boundary_l)

For Neumann boundary condition, rather than using `NeumannBC()`, we use `PointSetOperatorBC()` which needs the following inputs --> points on the Neumann boundary, actual solution for Neumann BC and the function for predicted Neumann BC. We start with the actual solution for Neumann BC. Since the actual solution :math:`u=(x+1)^2` is known, we can define a function to calculate the actual Neumann BC (it is the first derivative) as:

.. code-block:: python

    def d_func(x):
        return 2 * (x + 1)

Next, we define a function to calculate the predicted Neumann BC

.. code-block:: python

    def dy_x(x, y, X):
        dy_x = dde.grad.jacobian(y, x)
        return dy_x

Finally we define `PointSetOperatorBC()` on the points that lie on the right boundary in a similar way as Dirichlet BC.

.. code-block:: python

    boundary_pts = geom.random_boundary_points(2)
    r_boundary_pts = boundary_pts[np.isclose(boundary_pts, 1)].reshape(-1, 1)
    bc_r = dde.icbc.PointSetOperatorBC(r_boundary_pts, d_func(r_boundary_pts), dy_x)

Now, we have specified the geometry, PDE residual, Dirichlet boundary condition and Neumann boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)

The number 16 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 100 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We build the ``Model``.

.. code-block:: python

    model = dde.Model(data, net)

To evaluate the intermediate values for any given function during training, we can use `OperatorPredictor`. Let's say we would like to write the first and second derivatives on the boundary points into two different files. To achieve that, we first define the functions for first and second derivatives 

.. code-block:: python

    def dy_x(x, y):
        dy_x = dde.grad.jacobian(y, x)
        return dy_x
    
    def dy_xx(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return dy_xx

Then we define the `OperatorPredictor` callbacks:

.. code-block:: python

    first_derivative = dde.callbacks.OperatorPredictor(
        geom.random_boundary_points(2), op=dy_x, period=200, filename="first_derivative.txt"
    )

    second_derivative = dde.callbacks.OperatorPredictor(
        geom.random_boundary_points(2),
        op=dy_xx,
        period=200,
        filename="second_derivative.txt",
    )

For optimization, we set the optimizer and learning rate. We also compute the :math:`L^2` relative error as a metric during training. We then train the model for 10000 iterations:

.. code-block:: python
    
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(
        iterations=10000, callbacks=[first_derivative, second_derivative]
    )

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_PointSetOperator_1d.py
  :language: python
