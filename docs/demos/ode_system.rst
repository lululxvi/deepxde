A simple ODE system
================

Problem setup
--------------

We will solve an ODE system:

.. math:: \frac{dy_1}{dt} = y_2, \qquad \frac{dy_2}{dt} = - y_1, \qquad \text{where} \quad t \in [0,10],

with the initial conditions  

.. math:: y_1(0) = 0, \quad y_2(0) = 1.

The reference solution is :math:`y_1 = \sin(t), \quad y_2 = \cos(t)`.

Implementation
--------------

This description goes through the implementation of a solver for the above ODE system step-by-step.

First, the DeepXDE and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a computational geometry. We can use a built-in class ``TimeDomain`` to define a time domain as follows

.. code-block:: python

    geom = dde.geometry.TimeDomain(0, 10)

Next, we express the ODE system:

.. code-block:: python

    def ode_system(x, y):
        y1, y2 = y[:, 0:1], y[:, 1:]
        dy1_x = dde.grad.jacobian(y, x, i=0)
        dy2_x = dde.grad.jacobian(y, x, i=1)
        return [dy1_x - y2, dy2_x + y1]


The first argument to ``ode_system`` is the network input, i.e., the :math:`t`-coordinate, and here we represent it as ``x``. The second argument to ``ode_system`` is the network output, which is a 2-dimensional vector where the first component(``y[:, 0:1]``) is :math:`y_1`-coordinate and the second component (``y[:, 1:]``) is :math:`y_2`-coordinate. 

Next, we consider the initial condition. We can use a boundary function in our code, in which ``on_initial`` returns true if we should apply initial conditions:

.. code-block:: python

    def boundary(_, on_initial):
        return on_initial

Then the initial conditions are specified using the computational domain, initial function and boundary. The argument ``component`` refers to if this IC is for the first component or the second component.

.. code-block:: python

    ic1 = dde.IC(geom, np.sin, boundary, component=0)
    ic2 = dde.IC(geom, np.cos, boundary, component=1)
   
Now, we have specified the geometry, ODEs, and initial conditions. Since `PDE` is also an ODE solver, we then define the ``ODE`` problem as

.. code-block:: python

    data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

The number 35 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. We use 100 points for testing the ODE residual. The argument  ``solution=func`` is the reference solution to compute the error of our solution, and we define it as follows:

.. code-block:: python

    def func(x):
        return np.hstack((np.sin(x), np.cos(x)))

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

Now, we have the ODE problem and the network. We bulid a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
   
We then train the model for 20000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(epochs=20000)
    
We also save and plot the best trained result and loss history.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)  

Complete code
--------------

.. literalinclude:: ../../examples/ode_system.py
  :language: python
