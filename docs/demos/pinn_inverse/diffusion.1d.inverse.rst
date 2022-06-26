Inverse problem for the diffusion equation
================

Problem setup
--------------

We will solve an inverse problem for the diffusion equation with an unknown parameter :math:`C`:

.. math:: \frac{\partial y}{\partial t} = C\frac{\partial^2y}{\partial x^2} - e^{-t}(\sin(\pi x) - \pi^2\sin(\pi x)),   \qquad x \in [-1, 1], \quad t \in [0, 1]

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
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return (
            dy_t
            - C * dy_xx
            + tf.exp(-x[:, 1:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )


The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0:1]``) is :math:`x`-coordinate and the second component (``x[:,1:]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`y(x, t)`.
Here ``C`` is an unknown parameter with initial value 2.0:

.. code-block:: python

    C = dde.Variable(2.0)

Next, we consider the boundary/initial condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain in considered as the boundary condition. We include the ``geotime`` space , time geometry created above and ``on_boundary`` as the BC in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the initial condition for the diffusion equation and we use the computational domain, initial function, and ``on_initial`` to specify the IC. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

The reference solution ``func`` is defined as:

.. code-block:: python

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

In this problem, we provide extra information on some training points and the parameter :math:`C` can be identified from these observations. We generate a 2-dimensional array ``observe_x`` of 10 equally-spaced input points :math:`(x, t)` as the first argument to ``PointSetBC``, where :math:`x` is in :math:`[-1, 1]` and :math:`t=1`. The second argument ``func(observe_x)`` is the corresponding exact solution. ``PointSetBC`` compares ``observe_x`` and ``func(observe_x)``, and they satisfy the Dirichlet boundary condition.

.. code-block:: python

    observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
    observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)

Now, we have specified the geometry, PDE residual, boundary/initial condition, and extra observations. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_y],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        anchors=observe_x,
        solution=func,
        num_test=10000,
    )

The number 40 is the number of training residual points sampled inside the domain, and the number 20 is the number of training points sampled on the boundary (the left and right endpoints of the interval). We also include 10 initial residual points for the initial conditions and 10000 points for testing the PDE residual. The argument ``anchors`` is the above described training points in addition to the ``num_domain``, ``num_initial``, and ``num_boundary`` sampled points.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 32:

.. code-block:: python

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate, and the unknown parameter ``C`` is passed as ``external_trainable_variables``:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C)
    
   
We then train the model for 50000 iterations. During the training process, we output the value of :math:`C` every 1000 iterations:

.. code-block:: python

    variable = dde.callbacks.VariableValue(C, period=1000)
    losshistory, train_state = model.train(iterations=50000, callbacks=[variable])
    
We also save and plot the best trained result and loss history.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)  

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_inverse/diffusion_1d_inverse.py
  :language: python
