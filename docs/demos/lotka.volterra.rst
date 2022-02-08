<<<<<<< HEAD
Lotka-Volterra equation
=======
Lotka-Volterra Equation
>>>>>>> 4ec7688 (fixed names)
================

Problem setup
--------------

We will solve a Lotka-Volterra equation:

.. math:: \frac{dr}{dt} = \frac{R}{U}(2Ur - 0.04U^2rp)
.. math:: \frac{dp}{dt} = \frac{R}{U}(0.02U^2rp - 1.06Up)

<<<<<<< HEAD
with the initial condition   

.. math:: r(0) = \frac{100}{U}, \quad p(0) = \frac{15}{U}
=======
with the Dirichlet boundary conditions   

.. math:: r(0) = 100, \quad p(0) = 15
>>>>>>> 4ec7688 (fixed names)

and two user-specified parameters 

.. math:: U = 200, R = 20,

<<<<<<< HEAD
<<<<<<< HEAD
the first of which approximates the upper bound of the range, and the second is the right bound of the domain. These two will be used for scaling.
=======
the first of which approximates the upper bound of the range, and the second is the right bound of the domain, which will be used for scaling.
>>>>>>> 4ec7688 (fixed names)
=======
the first of which approximates the upper bound of the range, and the second is the right bound of the domain. These two will be used for scaling.
>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)

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
<<<<<<< HEAD
<<<<<<< HEAD
      
Now, we define the ODE problem as 
=======

Next, we consider the boundary. We use ``on_initial`` to specify the domain of the function:

.. code-block:: python 

    def boundary(_, on_initial):
        return on_initial
      
Since we wish to use a hard condition for the initial conditions, we define this later while creating the network. Now, we define the ODE problem as 
>>>>>>> 4ec7688 (fixed names)
=======
      
Now, we define the ODE problem as 
>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)

.. code-block:: python 

    data = dde.data.PDE(geom, ode_system, [], 3000, 2, num_test = 3000)

<<<<<<< HEAD
<<<<<<< HEAD
Note that when solving this equation, we want to have hard constraints on the initial conditions, so we define this later when creating the network rather than as part of the PDE.

=======
>>>>>>> 4ec7688 (fixed names)
=======
Note that when solving this equation, we want to have hard constraints on the initial conditions, so we define this later when creating the network rather than as part of the PDE.

>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)
We have 3000 training residual points inside the domain and 2 points on the boundary. We use 3000 points for testing the ODE residual. We now create the network:

.. code-block:: python

    layer_size = [1] + [64] * 6 + [2]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.maps.FNN(layer_size, activation, initializer)

<<<<<<< HEAD
<<<<<<< HEAD
This is a neural network of depth 7 with 6 hidden layers of width 50. We use :math:`\tanh` as the activation function. Since we expect to have periodic behavior in the Lotka-Volterra equation, we add a feature layer with :math:`\sin(kt)`. This forces the prediction to be periodic and therefore more accurate.
=======
This is a neural network of depth 7 with 6 hidden layers of width 50. We use :math:`\tanh` as the activation function. Since we observe periodic behavior in the Lotka-Volterra equation, we add a feature layer with :math:`\sin(kt)`:
>>>>>>> 4ec7688 (fixed names)
=======
This is a neural network of depth 7 with 6 hidden layers of width 50. We use :math:`\tanh` as the activation function. Since we expect to have periodic behavior in the Lotka-Volterra equation, we add a feature layer with :math:`\sin(kt)`. This forces the prediction to be periodic and therefore more accurate.
>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)

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

<<<<<<< HEAD
As mentioned earlier, we want the initial conditions :math:`r(0)=\frac{100}{U}` and :math:`p(0)=\frac{15}{U}` to be hard constraints, so we transform the output:
=======
As mentioned earlier, we want the initial conditions :math:`r(0)=100` and :math:`p(0)=15` to be hard constraints, so we transform the output:
>>>>>>> 4ec7688 (fixed names)

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

<<<<<<< HEAD
<<<<<<< HEAD
Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate, and train it for 50000 iterations:
=======
Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate, and train it for 10000 iterations:
>>>>>>> 4ec7688 (fixed names)
=======
Now that we have defined the neural network, we build a ``Model``, choose the optimizer and learning rate, and train it for 50000 iterations:
>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
<<<<<<< HEAD
<<<<<<< HEAD
    losshistory, train_state = model.train(epochs=50000)
=======
    losshistory, train_state = model.train(epochs=10000)
>>>>>>> 4ec7688 (fixed names)
=======
    losshistory, train_state = model.train(epochs=50000)
>>>>>>> cc1e4dc (small wording changes, changed amount of epochs)

After training with Adam, we continue with L-BFGS to have an even smaller loss:

.. code-block:: python

    model.compile("L-BFGS")
    losshistory, train_state = model.train()  
<<<<<<< HEAD
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

=======

After that, we save the best result:

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Now that we have a prediction for the model, we generate data to compare by using ``integrate.solve_ivp()`` from ``scipy``:

.. code-block:: python

    def gen_truedata():
        t = np.linspace(0, 1, 100)
        sol = integrate.solve_ivp(func, (0, 10), (100 / ub, 15 / ub), t_eval=t)
        x_true, y_true = sol.y
        x_true = x_true.reshape(100, 1)
        y_true = y_true.reshape(100, 1)

        return x_true, y_true


We call this function to get the true data and plot it with ``matplotlib``:

.. code-block:: python

    t = np.linspace(0, 1, 100)
    x_true, y_true = gen_truedata()
    plt.plot(t, x_true, color="black", label="x_true")
    plt.plot(t, y_true, color="blue", label="y_true")

We also plot the predicted data from earlier:

.. code-block:: python

    t = t.reshape(100, 1)
    sol_pred = model.predict(t)
    x_pred = sol_pred[:, 0:1]
    y_pred = sol_pred[:, 1:2]

    plt.plot(t, x_pred, color="red", linestyle="dashed", label="x_pred")
    plt.plot(t, y_pred, color="orange", linestyle="dashed", label="y_pred")

Lastly, we add a legend and show this graph:

.. code-block:: python

    plt.legend()
    plt.show()
>>>>>>> 4ec7688 (fixed names)

Complete code
--------------

.. literalinclude:: ../../examples/Lotka_Volterra.py
  :language: python