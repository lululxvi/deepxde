Poisson equation in 1D with Multi-scale Fourier feature networks
===================================================================================================

Problem setup
--------------

We will solve a Poisson equation:

.. math:: \Delta u = -(\pi A)^2 \sin(\pi A x)- 0.1 (\pi B)^2 \sin(\pi B x),
.. math:: \qquad x \in [-1, 1]

with Dirichlet boundary conditions

.. math:: u(0) = 0, \quad u(1) = 0

and two user-specified hyper-parameters that implies the fluctuation of sine functions

.. math:: A = 2, B = 50

The exact solution is :math:`u(x) = \sin(\pi A x)+0.1 \sin(\pi B x)`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Poisson equation step-by-step.
First, the DeepXDE module is imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

We begin by defining a computational geometry. We can use a built-in class ``Interval`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(0, 1)


Next, we express the PDE residual of the Poisson equation:

.. code-block:: python

    A = 2
    B = 50

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return (
            dy_xx
            + (np.pi * A) ** 2 * tf.sin(np.pi * A * x)
            + 0.1 * (np.pi * B) ** 2 * tf.sin(np.pi * B * x)
        )

The first argument to ``pde`` is the network input, i.e., the :math:`x`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Next, we define a function to return the value of :math:`u(x)` for the points :math:`x` on the Dirichlet boundary. In this case, it is :math:`u(x)=0`.

.. code-block:: python

    def func(x):
        return np.sin(np.pi * A * x) + 0.1 * np.sin(np.pi * B * x)

Then, the Dirichlet boundary condition is

.. code-block:: python

    bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

Now, we have specified the geometry, PDE residual and Dirichlet boundary condition. We then define the PDE problem as

.. code-block:: python

    data = dde.data.PDE(
        geom,
        pde,
        bc,
        1280,
        2,
        train_distribution="pseudo",
        solution=func,
        num_test=10000,
    )

The number 1280 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``train_distribution = 'pseudo'`` means that the sample training points follows a pseudo-random distribution.  The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 10000 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a multi-scale Fourier feature networks of depth 4 (i.e., 3 hidden layers) and width 100. ``sigmas`` is the list of standard deviation of the distribution of fourier feature embeddings. In this example, the network consists of a Fourier feature layer and fully-connected layers. The Fourier feature network is constructed using a random Fourier feature mapping as a coordinate embedding of the inputs. Specifically, we multiply the input vector with matrix whose entry is sampled from a Gaussian distribution with mean zero and variance sigma, and then concatenate the list of tensors horizontally with cosine and sine functions. Then, the network is:

.. code-block:: python

    layer_size = [1] + [100] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])

Now, we have the PDE problem and the network. We bulid a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=0.001,
        metrics=["l2 relative error"],
        decay=("inverse time", 2000, 0.9),
    )
We also compute the :math:`L^2` relative error as a metric during training.

We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=10000)

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Poisson_multiscale_1d.py
  :language: python
