Diffusion equation with hard initial and boundary conditions
=============================================================

Problem setup
--------------

We will solve a diffusion equation with hard initial and boundary conditions:

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

The reference solution ``func`` is defined as:

.. code-block:: python

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

Now, we have specified the geometry and the PDE residual. However, in order to apply hard boundary and initial conditions, they are not specified and excluded from the loss function. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=40, solution=func, num_test=10000)

The number 40 is the number of training residual points sampled inside the domain. 10000 points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 32:

.. code-block:: python

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Then we construct a function that spontaneously satisfies both the initial and the boundary conditions to transform the network output. In this case, :math:`t(1-x^2)y + sin(\pi x)` is used. When :math:`t` is equal to 0, the initial condition :math:`sin(\pi x)` is recovered. When :math:`x` is equal to -1 or 1, the boundary condition :math:`y(-1, t) = y(1, t) = 0` is recovered. Hence the initial and boundary conditions are both hard conditions.

.. code-block:: python
    
    net.apply_output_transform(
        lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + tf.sin(np.pi * x[:, 0:1])
    )

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

.. literalinclude:: ../../../examples/pinn_forward/diffusion_1d_exactBC.py
  :language: python
