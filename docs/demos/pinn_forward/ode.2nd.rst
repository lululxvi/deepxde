Second order ODE
================

Problem setup
-------------

We will solve an ODE:

.. math:: y''(t)-10y'(t)+9y(t) = 5t

with initial conditions

.. math:: y(0)=-1,  y'(0)=2

For :math:`t \in [0,0.25]`. The exact solution is :math:`y(t)=\frac{50}{81}+\frac{5}{9}t + \frac{31}{81} e^{9t} - 2e^t`.

Implementation
--------------

This description goes through the implementation of a solver for the above ODE step-by-step.
First, the DeepXDE module is imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a computational geometry. We can use a built-in class ``TimeDomain`` as follows

.. code-block:: python

    geom = dde.geometry.TimeDomain(0, 0.25)

Next, we express the residual of the ODE:

.. code-block:: python

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t

The first argument to ``ode`` is the network input, i.e., the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`y(t)`, but here we use ``y`` as the name of the variable.

We define the initial condition, setting the value of the function at :math:`t=0` to -1.
   
.. code-block:: python

    ic1 = dde.icbc.IC(geom, lambda x: -1, lambda _, on_initial: on_initial)
    
Now we deal with the initial condition :math:`y'(0)=2`.

The location of the intial condition is defined by a simple Python function. The function should return ``True`` for those points satisfying :math:`t=0` and ``False`` otherwise (note that because of rounding-off errors, it is often wise to use ``dde.utils.isclose`` to test whether two floating point values are equivalent). In this function, the argument ``t`` to ``boundary`` is the network input and is a :math:`d`-dim vector, where :math:`d` is the dimension and :math:`d=1` in this case. Then a boolean ``on_boundary`` is used as the second argument. If the point ``t`` (the first argument) is on the boundary of the geometry, then ``on_boundary`` is ``True``, otherwise, ``on_boundary`` is ``False``.

.. code-block:: python

    def boundary_l(t, on_boundary):
        return on_boundary and dde.utils.isclose(t[0], 0)

Now we define a function that returns the error of the initial condition, :math:`y'(0)=2`, which is the difference between the derivative of the output of the network at 0, and 2. The function takes arguments (``inputs``, ``outputs``, ``X``) and outputs a tensor of size ``N x 1``, where ``N`` is the length of ``inputs``. ``inputs`` and ``outputs`` are the network input and output tensors, respectively; ``X`` is the NumPy array of the ``inputs``.

.. code-block:: python

    def error_2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2

Then, the initial condition is defined by

.. code-block:: python

    ic2 = dde.icbc.OperatorBC(geom, error_2, boundary_l)

Now, we have specified the geometry, PDE residual, and the initial conditions. We then define the PDE problem. Note: If you use `X` in `func`, then do not set ``num_test`` when you define ``dde.data.PDE`` or ``dde.data.TimePDE``, otherwise DeepXDE would throw an error. In this case, the training points will be used for testing, and this will not affect the network training and training loss. This is a bug of DeepXDE, which cannot be fixed in an easy way for all backends.

.. code-block:: python

    data = dde.data.TimePDE(geom, ode, [ic1, ic2], 16, 2, solution=func, num_test=500)

The number 16 is the number of training residual points sampled inside the domain, and the number 2 is the number of training points sampled on the boundary. The argument ``solution=func`` is the reference solution to compute the error of our solution, and can be ignored if we don't have a reference solution. We use 500 residual points for testing the PDE residual.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 50:

.. code-block:: python

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

Now, we have the PDE problem and the network. We build a ``Model``, choose the optimizer, set the learning rate to 0.001, and train the network for 15000 iterations. We set the weight of the ODE loss to 0.01, and the weights of the two ICs to 1. We also compute the :math:`L^2` relative error as a metric during training.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=.001, loss_weights=[0.01, 1, 1], metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=15000)


Complete code
-------------

.. literalinclude:: ../../../examples/pinn_forward/ode_2nd.py
  :language: python
