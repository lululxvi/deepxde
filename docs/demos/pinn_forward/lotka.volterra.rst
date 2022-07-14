Lotka-Volterra equation
================

Problem setup
--------------

We will solve a Lotka-Volterra equation:

.. math:: \frac{dr}{dt} = \frac{R}{U}(2Ur - 0.04U^2rp)
.. math:: \frac{dp}{dt} = \frac{R}{U}(0.02U^2rp - 1.06Up)

with the initial condition   

.. math:: r(0) = \frac{100}{U}, \quad p(0) = \frac{15}{U}

and two user-specified parameters 

.. math:: U = 200, R = 20,

the first of which approximates the upper bound of the range, and the second is the right bound of the domain. These two will be used for scaling.

The reference solution is generated using ``integrate.solve_ivp()`` from ``scipy``.

Implementation
--------------

This description goes through the implementation of a solver for the above described Lotka-Volterra equation step-by-step.

First, the DeepXDE, numpy, matplotlib, and scipy packages are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate
    from deepxde.backend import tf

We begin by defining the approximate upper bound of the range and the right bound of the domain. Later, we scale by these factors to obtain a graph between 0 and 1 when graphing population vs. time.

.. code-block:: python

    ub = 200
    rb = 20

We now define a time domain. We do this by using the built-in class ``TimeDomain``:

.. code-block:: python

    geom = dde.geometry.TimeDomain(0.0, 1.0)

Next, we express the ODE system:

.. code-block:: python

    def ode_system(x, y):
        r = y[:, 0:1]
        p = y[:, 1:2]
        dr_t = dde.grad.jacobian(y, x, i=0)
        dp_t = dde.grad.jacobian(y, x, i=1)
        return [
            dr_t - 1 / ub * rb * (2.0 * ub * r - 0.04 * ub * r * ub * p),
            dp_t - 1 / ub * rb * (0.02 * r * ub * p * ub - 1.06 * p * ub),
        ]

The first argument to ``ode_system`` is the :math:`t`-coordinate, represented by ``x``. The second argument is a 2-dimensional vector, represented as ``y``, which contains :math:`r(t)` and :math:`p(t)`.
      
Now, we define the ODE problem as 

.. code-block:: python 

    data = dde.data.PDE(geom, ode_system, [], 3000, 2, num_test = 3000)

Note that when solving this equation, we want to have hard constraints on the initial conditions, so we define this later when creating the network rather than as part of the PDE.

We have 3000 training residual points inside the domain and 2 points on the boundary. We use 3000 points for testing the ODE residual. We now create the network:

.. code-block:: python

    layer_size = [1] + [64] * 6 + [2]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.nn.FNN(layer_size, activation, initializer)

This is a neural network of depth 7 with 6 hidden layers of width 50. We use :math:`\tanh` as the activation function. Since we expect to have periodic behavior in the Lotka-Volterra equation, we add a feature layer with :math:`\sin(kt)`. This forces the prediction to be periodic and therefore more accurate.

.. code-block:: python

    def input_transform(t):
        return tf.concat(
            (
                t,
                tf.sin(t),
                tf.sin(2 * t),
                tf.sin(3 * t),
                tf.sin(4 * t),
                tf.sin(5 * t),
                tf.sin(6 * t),
            ),
            axis=1,
        )

As mentioned earlier, we want the initial conditions :math:`r(0)=\frac{100}{U}` and :math:`p(0)=\frac{15}{U}` to be hard constraints, so we transform the output:

.. code-block:: python

    def output_transform(t, y):
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]

        return tf.concat(
            [y1 * tf.tanh(t) + 100 / ub, y2 * tf.tanh(t) + 15 / ub], axis=1
        )

We add these layers:

.. code-block:: python

    net.apply_feature_transform(input_transform)
    net.apply_output_transform(output_transform)

Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate, and train it for 50000 iterations:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=50000)

After training with Adam, we continue with L-BFGS to have an even smaller loss:

.. code-block:: python

    model.compile("L-BFGS")
    losshistory, train_state = model.train()  
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Lotka_Volterra.py
  :language: python
