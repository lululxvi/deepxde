Burgers equation with residual-based adaptive refinement
================

Problem setup
--------------

We will solve a Burgers equation:

.. math:: \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2u}{\partial x^2}, \qquad x \in [-1, 1], \quad t \in [0, 1]

with the Dirichlet boundary conditions and initial conditions  

.. math:: u(-1,t)=u(1,t)=0, \quad u(x,0) = - \sin(\pi x).

The reference solution is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz>`_.

Implementation
--------------

This description goes through the implementation of a solver for the above described Burgers equation step-by-step.

First, the DeepXDE and TensorFlow (``tf``) modules are imported:

.. code-block:: python

    import deepxde as dde
    from deepxde.backend import tf

We begin by defining a computational geometry and time domain. We can use a built-in class ``Interval`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows

.. code-block:: python

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

Next, we express the PDE residual of the Burgers equation:

.. code-block:: python

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

The first argument to ``pde`` is 2-dimensional vector where the first component(``x[:,0]``) is :math:`x`-coordinate and the second componenet (``x[:,1]``) is the :math:`t`-coordinate. The second argument is the network output, i.e., the solution :math:`u(x,t)`, but here we use ``y`` as the name of the variable.

Next, we consider the boundary/initial condition. ``on_boundary`` is chosen here to use the whole boundary of the computational domain in considered as the boundary condition. We include the ``geomtime`` space, time geometry created above and ``on_boundary`` as the BCs in the ``DirichletBC`` function of DeepXDE. We also define ``IC`` which is the inital condition for the burgers equation and we use the computational domain, initial function, and ``on_initial`` to specify the IC. 

.. code-block:: python

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
    
Now, we have specified the geometry, PDE residual, and boundary/initial condition. We then define the ``TimePDE`` problem as

.. code-block:: python

    data = dde.data.TimePDE(geomtime, pde, [bc, ic], 
                            num_domain=2500, num_boundary=100, num_initial=100)    

The number 2500 is the number of training residual points sampled inside the domain, and the number 100 is the number of training points sampled on the boundary. We also include 100 initial residual points for the initial conditions.

Next, we choose the network. Here, we use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20:

.. code-block:: python

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

Now, we have the PDE problem and the network. We build a ``Model`` and choose the optimizer and learning rate:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    
   
We then train the model for 10000 iterations:

.. code-block:: python

    losshistory, train_state = model.train(iterations=10000)
    
After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()      

Because we only use 2500 residual points for training, the accuracy is low. Next, we improve the accuracy by the residual-based adaptive refinement (RAR) method. Because the Burgers equation has a sharp front, intuitively, we should put more points near the sharp front. First, we randomly generate 100000 points from our domain to calculate the PDE residual.

.. code-block:: python

    X = geomtime.random_points(100000)
    err = 1

We will repeatedly add points while the mean residual is greater than 0.005. Each iteration, we use our model to generate predictions for inputs in ``X`` and compute the absolute values of the errors. We then print the mean residual. Next, we find the points where the residual is greatest and add these new points for training PDE loss. Furthermore, we define a callback function to check whether the network converges. If there is significant improvement in the model's accuracy, as judged by the callback function, we continue to train the model. As before, after we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    while err > 0.005:
        f = model.predict(X, operator=pde)
        err_eq = np.absolute(f)
        err = np.mean(err_eq)
        print("Mean residual: %.3e" % (err))

        x_id = np.argmax(err_eq)
        print("Adding new point:", X[x_id], "\n")
        data.add_anchors(X[x_id])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        model.compile("adam", lr=1e-3)
        model.train(iterations=10000, disregard_previous_best=True, callbacks=[early_stopping])
        model.compile("L-BFGS")
        losshistory, train_state = model.train()

Finally, we display a graph depicting train loss and test loss over time, along with a graph displaying the predicted solution to the PDE.

.. code-block:: python

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

Complete code
--------------

.. literalinclude:: ../../../examples/pinn_forward/Burgers_RAR.py
  :language: python
