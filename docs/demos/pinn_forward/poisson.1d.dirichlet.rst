Poisson equation in 1D with Dirichlet boundary conditions
=========================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: -\Delta u = \pi^2 \sin(\pi x), \qquad x \in [-1, 1],

with the Dirichlet boundary conditions

.. math:: u(-1)=0, \quad u(1)=0.

The exact solution is :math:`u(x) = \sin(\pi x)`.

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
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we consider the Dirichlet boundary condition. A simple Python function, returning a boolean, is used to define the subdomain for the Dirichlet boundary condition (:math:`\{-1, 1\}`). The function should return ``True`` for those points inside the subdomain and ``False`` for the points outside. In our case, the points :math:`x` of the Dirichlet boundary condition are :math:`x=-1` and :math:`x=1`. (Note that because of rounding-off errors, it is often wise to use ``np.isclose`` to test whether two floating point values are equivalent.)

.. code-block:: python

    def boundary(x, _):
        return np.isclose(x[0], -1) or np.isclose(x[0], 1)

The argument ``x`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. To facilitate the implementation of ``boundary``, a boolean ``on_boundary`` is used as the second argument. If the point ``x`` (the first argument) is on the entire boundary of the geometry (the left and right endpoints of the interval in this case), then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``. Thus, we can also define ``boundary`` in a simpler way:

.. code-block:: python

    def boundary(x, on_boundary):
        return on_boundary

Next, we define a function to return the value of :math:`u(x)` for the points :math:`x` on the boundary. In this case, it is :math:`u(x)=0`.

.. code-block:: python

    def func(x):
        return 0

If the function value is not a constant, we can also use NumPy to compute. For example, :math:`\sin(\pi x)` is 0 on the boundary, and thus we can also use

.. code-block:: python

    def func(x):
        return np.sin(np.pi * x)

Then, the Dirichlet boundary condition is

.. code-block:: python

    bc = dde.icbc.DirichletBC(geom, func, boundary)

Now, we have specified the geometry, PDE residual, and Dirichlet boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)    

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

We also compute the :math:`L^2` relative error as a metric during training. We can also use ``callbacks`` to save the model and the movie during training, which is optional.

.. code-block:: python

    checkpointer = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", verbose=1, save_better_only=True
    )
    # ImageMagick (https://imagemagick.org/) is required to generate the movie.
    movie = dde.callbacks.MovieDumper(
        "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
    )
  
We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(
        iterations=10000, callbacks=[checkpointer, movie]
    )

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_Dirichlet_1d.py
  :language: python
