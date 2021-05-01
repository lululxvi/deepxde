ODE System 
================

Problem setup
--------------

We will solve a system of Ordinary Differential Equations :

.. math:: \frac{dy}{dx} = z 
.. math:: \frac{dz}{dx} = -y , \qquad t \in [0, 10]

The solution of the ODE System is :

.. math:: y = sin(x) , z = cos(x)

Implementation
--------------

This description goes through the implementation of a solver for the above described Burgers equation step-by-step.

First, the DeepXDE and Numpy (``np``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a time domain. We can use a built-in class ``TimeDomain`` as follows

.. code-block:: python

    geom = dde.geometry.TimeDomain(0, 10)

Next, we express the PDE residual of the ODE System:

.. code-block:: python

    def ode_system(x, y):
        """ODE system.
        dy1/dx = y2
        dy2/dx = -y1
        """
        y1, y2 = y[:, 0:1], y[:, 1:]
        dy1_x = dde.grad.jacobian(y, x, i=0)
        dy2_x = dde.grad.jacobian(y, x, i=1)
        return [dy1_x - y2, dy2_x + y1]

Next, we consider initial conditions. ``on_intial`` is chosen here. We define two ``IC`` since there are two equations in the ODE system. 

.. code-block:: python

    def boundary(_, on_initial):
        return on_initial
        
    ic1 = dde.IC(geom, np.sin, boundary, component=0)
    ic2 = dde.IC(geom, np.cos, boundary, component=1)
    
Now, we have specified the geometry, solution ``func``, PDE residual, and initial conditions. We then define the ``PDE`` problem as

.. code-block:: python

    def func(x):
        """
        y1 = sin(x)
        y2 = cos(x)
        """
        return np.hstack((np.sin(x), np.cos(x)))

    data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

The number 35 is the number of training residual points sampled inside the domain, The number 2 is the number of training residual points sampled on the boundary and 100 is the number of testing residual points sampled.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We bulid a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
   
We then train the model for 20000 iterations:

.. code-block:: python

   losshistory, train_state = model.train(epochs=20000)
    

Complete code
--------------

.. literalinclude:: ../../examples/ode_system.py
  :language: python
