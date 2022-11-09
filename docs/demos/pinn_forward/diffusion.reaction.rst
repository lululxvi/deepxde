Diffusion-reaction equation
===========================

Problem setup
--------------

We will solve the following 1D diffusion-reaction equation:

.. math:: \frac{\partial y}{\partial t} = d \frac{\partial^2 y}{\partial x^2} + e^{-t} (3\frac{\sin{2x}}{2} + \frac{8\sin{3x}}{3} + \frac{15\sin{4x}}{4} + \frac{63\sin{8x}}{8})

with the initial condition

.. math:: y(x, 0) = \sin{x} + \frac{\sin{2x}}{2} + \frac{\sin{3x}}{3} + \frac{\sin{4x}}{4} + \frac{\sin{8x}}{8}, \quad x \in [-\pi, \pi]

and the Dirichlet boundary condition

.. math:: y(t, -\pi) = y(t, \pi) = 0, \quad t \in [0, 1]

We also specify the following parameters for the equation:

.. math:: d = 1 

The exact solution is

.. math:: y(x, t) = e^{-t}( \sin{x} + \frac{\sin{2x}}{2} + \frac{\sin{3x}}{3} + \frac{\sin{4x}}{4} + \frac{\sin{8x}}{8})

Implementation
--------------

This description goes through the implementation of a solver for the above described diffusion-reaction equation step-by-step.

First, DeepXDE, Numpy and Tensorflow libraries are imported:

.. code-block:: python

    import DeepXDE as dde
    import numpy as np
    import tensorflow as tf

Next, we express the PDE residual of the diffusion-reaction equation:

.. code-block:: python

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        d = 1
        # Backend tensorflow.compact.v1 or tensorflow
        return (
            dy_t
            - d * dy_xx
            - tf.exp(-x[:, 1:])
            * (
                3 * tf.sin(2 * x[:, 0:1]) / 2
                + 8 * tf.sin(3 * x[:, 0:1]) / 3
                + 15 * tf.sin(4 * x[:, 0:1]) / 4
                + 63 * tf.sin(8 * x[:, 0:1]) / 8
            )
        )

If the backend is Pytorch, the residual should look like this:

.. code-block:: python

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        d = 1
        # Backend pytorch
        return (
            dy_t
            - d * dy_xx
            - torch.exp(-x[:, 1:])
            * (
                3 * torch.sin(2 * x[:, 0:1]) / 2
                + 8 * torch.sin(3 * x[:, 0:1]) / 3
                + 15 * torch.sin(4 * x[:, 0:1]) / 4
                + 63 * torch.sin(8 * x[:, 0:1]) / 8
            )
        )

The first argument to ``pde`` is the 2 dimensional vector where the first component(``x[:, 0]``) is the :math:`x`-coordinate, and the second component(``x[:, 1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x)`, but here we use ``y`` as the name of the variable.

Then we define the solution to the PDE:

.. code-block:: python

    def func(x):
        return np.exp(-x[:, 1:]) * (
            np.sin(x[:, 0:1])
            + np.sin(2 * x[:, 0:1]) / 2
            + np.sin(3 * x[:, 0:1]) / 3
            + np.sin(4 * x[:, 0:1]) / 4
            + np.sin(8 * x[:, 0:1]) / 8
        )


Now we can define a computational geometry and time domain. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Now, we have specified the geometry and the PDE residual. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime, pde, [], num_domain=320, solution=func, num_test=80000
    )

The number 320 is the number of training residual points sampled inside the domain, and the number 80000 is the number of points sampled inside
the domain for testing the PDE loss.

Next, we choose the network. Here, we use a fully connected neural network of depth 7 (i.e., 6 hidden layers) and width 30:

.. code-block:: python

    layer_size = [2] + [30] * 6 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)


Then we construct a function that satisfies both the initial and the boundary conditions to tansform the network output.
In this case, :math:`t(\pi^2 - x^2)y + \sin{x} + \frac{\sin{2x}}{2} + \frac{\sin{3x}}{3} + \frac{\sin{4x}}{4} + \frac{\sin{8x}}{8}` is used.
If :math:`t` is equal to 0, the initial condition is recovered. When :math:`x` is equal to :math:`-\pi` or :math:`\pi`, the boundary condition is recovered.
Hence the initial and boundary conditions are both hard conditions.

.. code-block:: python

    def output_transform(x, y):
        return (
            x[:, 1:2] * (np.pi ** 2 - x[:, 0:1] ** 2) * y
            + tf.sin(x[:, 0:1])
            + tf.sin(2 * x[:, 0:1]) / 2
            + tf.sin(3 * x[:, 0:1]) / 3
            + tf.sin(4 * x[:, 0:1]) / 4
            + tf.sin(8 * x[:, 0:1]) / 8
        )
    
    net.apply_output_transform(output_transform)

If the backend is Pytorch, the code should look like this:

.. code-block:: python

    def output_transform(x, y):
        return (
            x[:, 1:2] * (np.pi ** 2 - x[:, 0:1] ** 2) * y
            + torch.sin(x[:, 0:1])
            + torch.sin(2 * x[:, 0:1]) / 2
            + torch.sin(3 * x[:, 0:1]) / 3
            + torch.sin(4 * x[:, 0:1]) / 4
            + torch.sin(8 * x[:, 0:1]) / 8
        )
    
    net.apply_output_transform(output_transform)

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate. We then train the model for 20000 iterations.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=20000)

We also save and plot the best trained result and loss history.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete code
-------------

.. literalinclude:: ../../../examples/pinn_forward/diffusion_reaction.py
  :language: python
