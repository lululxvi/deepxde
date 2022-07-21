Klein-Gordon equation
================

Problem setup
--------------

We will solve a Klein-Gordon equation:

.. math:: \frac{\partial^2y}{\partial t^2} + \alpha \frac{\partial^2y}{\partial x^2} + \beta y + \gamma y^k = -x\cos(t) + x^2\cos^2(t), \qquad x \in [-1, 1], \quad t \in [0, 10]

with initial conditions

.. math:: y(x, 0) = x, \quad \frac{\partial y}{\partial t}(x, 0) = 0

and Dirichlet boundary conditions

.. math:: y(-1, t) = -\cos(t), \quad y(1, t) = \cos(t)

We also specify the following parameters for the equation:

.. math:: \alpha = -1, \beta = 0, \gamma = 1, k = 2.

The reference solution is :math:`y(x, t) = x\cos(t)`.

Implementation
--------------

This description goes through the implementation of a solver for the above described Klein-Gordon equation step-by-step.

First, the DeepXDE, NumPy, TensorFlow, Maplotlib, and SciPy modules are imported.

.. code-block:: python

    import deepxde as dde
    import matplotlib.pyplot as plt
    import numpy as np
    from deepxde.backend import tf
    from scipy.interpolate import griddata

We begin by defining computational geometries. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 10)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
Next, we express the PDE residual of the Klein-Gordon equation:

.. code-block:: python

    def pde(x, y):
        alpha, beta, gamma, k = -1, 0, 1, 2
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        x, t = x[:, 0:1], x[:, 1:2]
        return (
            dy_tt
            + alpha * dy_xx
            + beta * y
            + gamma * (y ** k)
            + x * tf.cos(t)
            - (x ** 2) * (tf.cos(t) ** 2)
        )
        
The first argument to ``pde`` is a 2-dimensional vector where the first component(``x[:, 0:1]``) is the :math:`x`-coordinate and the second component (``x[:, 1:2]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`y(x, t)`.

The reference solution ``func`` is then defined as the following.

.. code-block:: python

    def func(x):
        return x[:, 0:1] * np.cos(x[:, 1:2])
        
Next, we consider the boundary/initial conditions. ``on_boundary`` is chosen here to use the whole boundary of the computational domain as the boundary condition. We include the ``geomtime`` space/time geometry created above and ``on_boundary`` as the BC in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the initial conditon for the Klein-Gordon equation, and we use the computational domain, initial function, and ``on_initial`` to specify the IC. Finally, we specify the initial condition for the first derivative of the :math:`y`-coordinate with respect to the :math:`t`-coordinate through the ``OperatorBC`` function of DeepXDE. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda _, on_initial: on_initial,
    )
    
Now, we have specified the geometry, PDE residual, and the boundary/initial conditions. We then define the ``TimePDE`` problem as the following.

.. code-block:: python
    
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic_1, ic_2],
        num_domain=30000,
        num_boundary=1500,
        num_initial=1500,
        solution=func,
        num_test=6000, 
    )

The number 30000 is the number of training residual points sampled inside of the domain, and the number 1500 is the number of training residual points sampled on the boundary. We also include 1500 initial residual points for the initial conditions and 6000 points for testing the PDE residual. 

Next, we choose the network. Here, we use a fully connected neural network of depth 3 (i.e., 2 hidden layers) and width 40.

.. code-block:: python

    layer_size = [2] + [40] * 2 + [1]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = dde.nn.FNN(layer_size, activation, initializer)
    
Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate. We also implement a learning rate decay to reduce overfitting of the model.

.. code-block:: python

    model = dde.Model(data, net)
    model.compile(
        "adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 3000, 0.9)
    )
    
We also compute the :math:`L^2` relative error as a metric during training.

We then train the model for 20000 iterations.

.. code-block:: python

    model.train(iterations=20000)
    
After we train the network with Adam, we compile again and continue to train the network using L-BFGS to achieve a smaller loss.

.. code-block:: python
    
    model.compile('L-BFGS', metrics=['l2 relative error')
    losshistory, train_state = model.train()
    
We then save and plot the best trained result and loss history of the model.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
Finally, we use the trained model to plot the predicted solution of the Klein-Gordon equation.

.. code-block:: python

    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 10, 256)
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    prediction = model.predict(X_star, operator=None)

    v = griddata(X_star, prediction[:, 0], (X, T), method='cubic')

    fig, ax = plt.subplots()
    ax.set_title("Results")
    ax.set_ylabel("Prediction")
    ax.imshow(
        v.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[0, 10, -1, 1],
        origin="lower",
        aspect="auto",
    )
    plt.show()
    
Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Klein_Gordon.py
  :language: python
