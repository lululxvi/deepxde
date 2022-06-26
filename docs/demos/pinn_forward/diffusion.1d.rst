Diffusion equation
===================

Problem setup
--------------

We will solve a diffusion equation:

.. math:: \frac{\partial y}{\partial t} = \frac{\partial^2y}{\partial x^2} - e^{-t}(\sin(\pi x) - \pi^2\sin(\pi x)),   \qquad x \in [-1, 1], \quad t \in [0, 1]

with the initial condition

.. math:: y(x, 0) = \sin(\pi x)

and the Dirichlet boundary condition 

.. math:: y(-1, t) = y(1, t) = 0.

The reference solution is :math:`y = e^{-t} \sin(\pi x)`.

Implementation
--------------

This description goes through the implementation of a solver for the above described diffusion equation step-by-step.

First, the DeepXDE, NumPy (``np``), and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    from deepxde.backend import tf

We begin by defining computational geometries. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Next, we express the PDE residual of the diffusion equation:

.. code-block:: python

    def pde(x, y):
    	dy_t = dde.grad.jacobian(y, x, j=1)
    	dy_xx = dde.grad.hessian(y, x, j=0)
    	return (
            dy_t
            - dy_xx
            + tf.exp(-x[:, 1:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )


The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0:1]``) is :math:`x`-coordinate and the second component (``x[:,1:]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`y(x, t)`.

Next, we consider the boundary/initial condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain as the boundary condition. We include the ``geotime`` space , time geometry created above and ``on_boundary`` as the BC in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the initial condition for the diffusion equation and we use the computational domain, initial function, and ``on_initial`` to specify the IC. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

The reference solution ``func`` is defined as:

.. code-block:: python

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

Now, we have specified the geometry, the PDE residual and the boundary/initial conditions. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        solution=func,
        num_test=10000,
    )

The number 40 is the number of training residual points sampled inside the domain, and the number 20 is the number of training points sampled on the boundary (the left and right endpoints of the interval). We also include 10 initial residual points for the initial conditions and 10000 points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 32:

.. code-block:: python

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate. We then train the model for 10000 iterations.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=10000)
    
We also save and plot the best trained result and loss history.

.. code-block:: python

   dde.saveplot(losshistory, train_state, issave=True, isplot=True)  

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/diffusion_1d.py
  :language: python
