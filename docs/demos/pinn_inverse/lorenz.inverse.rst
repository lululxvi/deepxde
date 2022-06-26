Inverse problem for the Lorenz system
=====================================

Problem setup
--------------

We will solve the Lorenz system:

.. math:: \frac{dx}{dt} = \sigma(y-x), \quad \frac{dy}{dt} = x (\rho - z) - y, \quad \frac{dz}{dt} = x y - \beta z \qquad t \in [0, 3]

with initial conditions 

.. math:: x(0) = -8, \quad y(0) = 7, \quad z(0) = 27.

The reference solution is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Lorenz.npz>`_, where the parameters :math:`\sigma`, :math:`\rho`, and :math:`\beta` are to be identified from observations of the system at certain times and whose true values are 10, 15, and 8/3, respectivly. 

Implementation
--------------

This description goes through the implementation of a solver for the above Lorenz system step-by-step.

First, the DeepXDE and NumPy (``np``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
        
We also want to define our three unknown variables, :math:`\sigma`, :math:`\rho`, and :math:`\beta` which will now be called C1, C2, and C3, respectivly. These variables are given an initial guess of 1.0.

.. code-block:: python
    
    C1 = dde.Variable(1.0)
    C2 = dde.Variable(1.0)
    C3 = dde.Variable(1.0)

Now we can begin by creating a ``TimeDomain`` class.

.. code-block:: python
    
    geom = dde.geometry.TimeDomain(0, 3)
    
Next, we create the Lorenz system to solve using the ``dde.grad.jacobian`` function.

.. code-block:: python

    def Lorenz_system(x, y):
        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
        dy1_x = dde.grad.jacobian(y, x, i=0)
        dy2_x = dde.grad.jacobian(y, x, i=1)
        dy3_x = dde.grad.jacobian(y, x, i=2)
        return [
            dy1_x - C1 * (y2 - y1),
            dy2_x - y1 * (C2 - y3) + y2,
            dy3_x - y1 * y2 + C3 * y3,
        ]

The first argument to ``Lorenz_system`` is the network input, i.e., the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`y(x,y,z)`, but here we use ``y1, y2, y3`` as the name of the coordinates x, y, and z, which correspond to the columns of datapoints in the 2D array, :math:`y`. 

Next, we consider the initial conditions. We need to implement a function, which should return ``True`` for points inside the subdomain and ``False`` for the points outside. 

.. code-block:: python

    def boundary(_, on_initial):
        return on_initial

Then the initial conditions are specified using the computational domain, initial function, and boundary. The argument ``component`` refers to if this IC is for the first component (:math:`x`), the second component (:math:`y`), or the third component (:math:`z`). Note that in our case, the point :math:`t` of the initial condition is :math:`t = 0`. 

.. code-block:: python
    
    ic1 = dde.icbc.IC(geom, lambda X: -8, boundary, component=0)
    ic2 = dde.icbc.IC(geom, lambda X: 7, boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda X: 27, boundary, component=2)
    
Now me must assign the data from ``Lorenz.npz`` to the corresponding :math:`t`, :math:`x`, :math:`y`, and :math:`z` values for training. First we retrieve the data to train the model. The data is split into ``"t"`` and ``"y"``, which correspond to time datapoints and the cartesian coodrinate datapoints (x, y, and z), respectivly. 

.. code-block:: python

    def gen_traindata():
        data = np.load("dataset/Lorenz.npz")
        return data["t"], data["y"]
        
Then we organize and assign the train data. 

.. code-block:: python

    observe_t, ob_y = gen_traindata()
    observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
    observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
    observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
  
Now that the problem is fully setup, we define the PDE as: 

.. code-block:: python  
  
    data = dde.data.PDE(
        geom,
        Lorenz_system,
        [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
        num_domain=400,
        num_boundary=2,
        anchors=observe_t,
    )

Where ``num_domain`` is the number of points inside the domain, and ``num_boundary`` is the number of points on the boundary. ``anchors`` are extra points beyond ``num_domain`` and ``num_boundary`` used for training. 

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 40:

.. code-block:: python

    net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
    
Now that the PDE problem and network have been created, we build a ``Model`` and choose the optimizer, learning rate, and provide the trainable variables C1, C2, and C3:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3])
      variable = dde.callbacks.VariableValue(
      [C1, C2, C3], period=600, filename="variables.dat"
    )

We then train the model for 60000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(epochs=60000, callbacks=[variable])

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_inverse/Lorenz_inverse.py
  :language: python
