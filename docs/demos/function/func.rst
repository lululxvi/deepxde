Learning a function from a formula
===================

Problem setup
-------------

We will solve a simple function approximation problem from a formula:

.. math:: f(x) = x * \sin(5x)

Implementation
--------------

This description goes through the implementation of a solver for the above function step-by-step.

First, the DeepXDE and NumPy (``np``) modules are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

We begin by defining a simple function which will be approximated. 

.. code-block:: python

    def func(x):
        """
        x: array_like, N x D_in
        y: array_like, N x D_out
        """
        return x * np.sin(5 * x)

The argument ``x`` to ``func`` is the network input. The ``func`` simply returns the corresponding function values from the given ``x``. 

Then, we define a computational domain. We can use a built-in class ``Interval`` as follows:

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)

Now, we have specified the geometry, we need to define the problem using a built-in class ``Function``

.. code-block:: python

    num_train = 16
    num_test = 100
    data = dde.data.Function(geom, func, num_train, num_test)

Here, we use 16 points for training sampled inside the domain, and 100 points for testing.

Next, we choose a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20 with ``tanh`` as the activation function and ``Glorot uniform`` as the initializer:

.. code-block:: python

    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

Now, we have the function approximation problem and the network. We bulid a ``Model`` and choose the optimizer ``adam`` and the learning rate of ``0.001``:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
   
We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=10000)
    
We also save and plot the best trained result and loss history.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete code
-------------

.. literalinclude:: ../../../examples/function/func.py
  :language: python
