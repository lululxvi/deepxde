
Beltrami-flow
=============

Problem setup
-------------

We will solve the incompressible Navier-Stokes equation:

.. math:: \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = \frac{1}{Re} \Delta \mathbf{u} - \nabla p
.. math:: \nabla \cdot u = 0
.. math:: \mathbf{x} \in [-1, 1]^3, \quad t \in [0, 1]

In `this paper <https://www.ljll.math.upmc.fr/~frey/papers/Navier-Stokes/Ethier%20C.R.,%20Steinman%20D.A.,%20Exact%20fully%203d%20Navier-Stokes%20solutions%20for%20benchmarking.pdf>`_, the
authors derive, under some constraints on :math:`\mathbf{u}`, a closed-form Beltrami-flow solution, which is higly likely physically unattainable (but good for testing numerical solves nonetheless):

.. math:: u = -a[e^{ax}\sin{(ay + dz)} + e^{az}\cos{(ax + dy)}]e^{-d^2 t}
.. math:: v = -a[e^{ay}\sin{(az + dx)} + e^{ax}\cos{(ay + dz)}]e^{-d^2 t}
.. math:: w = -a[e^{az}\sin{(ax + dy)} + e^{ay}\cos{(az + dx)}]e^{-d^2 t}
.. math:: p = -\frac{a^2}{2}\left[e^{2ax} + e^{2ay} + e^{2az} + 2\sin{(ax + dy)}\cos{(az + dx)}e^{a(y + z)} + 2\sin{(ay + dz)}\cos{(ax + dy)}e^{a(z + x)} + 2\sin{(az + dx)}\cos{(ay + dz)}e^{a(x + y)}\right] e^{-2d^2t}

Here :math:`u, v` and :math:`w` are the velocity components, :math:`p` is the pressure and :math:`a` and :math:`d` are real and arbitrary numbers.

We also specify the following parameters for the function:

.. math:: a = d = Re = 1

Implementation
--------------

This description goes through the implementation of a solver for the above described Navier-Stokes equation step-by-step.

First, DeepXDE and Numpy libraries are imported:

.. code-block:: python

    import deepxde as dde
    import numpy as np

Next, we define the PDE and the residual:

.. code-block:: python

    def pde(x, u):
        u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
        u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
        u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
        u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
         u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

        v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
        v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
        v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
        v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
        v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

        w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
        w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
        w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
        w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
        w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
        w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
        w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

        p_x = dde.grad.jacobian(u, x, i=3, j=0)
        p_y = dde.grad.jacobian(u, x, i=3, j=1)
        p_z = dde.grad.jacobian(u, x, i=3, j=2)

        momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
            + p_x
            - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
        )
        momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
            + p_y
            - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
        )
        momentum_z = (
            w_vel_t
            + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
            + p_z
            - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
        )
        continuity = u_vel_x + v_vel_y + w_vel_z

        return [momentum_x, momentum_y, momentum_z, continuity]

The first argument to ``pde`` is the vector of the :math:`t`-coordinates, and the second argument is a 2-dimensional vector where the first column (``u[:, 0]``) is the first velocity component, the second column(``u[:, 1]``) is the the second velocity component, the third column (``u[:, 2]``) is the third velocity component, and the last column (``u[:, 3]``) is the pressure.
Here ``[momentum_x, momentum_y, momentum_z]`` is the residual vector for the velocity components, and ``continuity`` is the residual for the solution to be divergence-free.

Then we define the solution function:

.. code-block:: python

    def u_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
              + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def v_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def w_func(x):
        return (
            -a
            * (
                np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            )
            * np.exp(-(d ** 2) * x[:, 3:4])
        )


    def p_func(x):
        return (
            -0.5
            * a ** 2
            * (
                np.exp(2 * a * x[:, 0:1])
                + np.exp(2 * a * x[:, 1:2])
                + np.exp(2 * a * x[:, 2:3])
                + 2
                * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
                * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
                * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
                + 2
                * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
                * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
                * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
                + 2
                * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
                * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
                * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
            )
            * np.exp(-2 * d ** 2 * x[:, 3:4])
        )


Now we can define a computational geometry and time domain. We can use a built-in class ``Cuboid`` and ``TimeDomain`` and we combine both the domains using ``GeometryXTime`` as follows:

.. code-block:: python

    spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
    temporal_domain = dde.geometry.TimeDomain(0, 1)
    spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

Then we set the initial and boundary condition for the three components:

.. code-block:: python

    boundary_condition_u = dde.icbc.DirichletBC(
        spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
    )
    boundary_condition_v = dde.icbc.DirichletBC(
        spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
    )
    boundary_condition_w = dde.icbc.DirichletBC(
        spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
    )

    initial_condition_u = dde.icbc.IC(
        spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
    )
    initial_condition_v = dde.icbc.IC(
        spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
    )
    initial_condition_w = dde.icbc.IC(
        spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
    )

Here, we pass in our computational geometry, the functions for the different components to compute the boundary/initial values, and a function which returns ``True`` if a point satisfies the boundary/initial condition and ``False`` otherwise, plus the component axis on which the boundary/initial condition is satisfied.

Now we have eveything we need: the PDE residuals, the initial and boundary conditions. We then define the ``TimePDE`` problem as

.. code-block:: python
    
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            boundary_condition_w,
            initial_condition_u,
            initial_condition_v,
            initial_condition_w,
        ],
        num_domain=50000,
        num_boundary=5000,
        num_initial=5000,
        num_test=10000,
    )

The number 50000 is the number of training residual points sampled inside the domain, and the number 5000 is the number of training points sampled on the boundary. We also include 5000 initial residual points for the initial conditions and 10000 points for testing the PDE residual.

Next, we define the network structure. Here we use a fully connected neural network of depth 5 (i.e., 4 hidden layers) and width 50:

.. code-block:: python

    net = dde.nn.FNN([4] + 4 * [50] + [4], "tanh", "Glorot normal")
    model = dde.Model(data, net)

First, we train the model with the ``Adam`` optimizer for 30000 epochs, and putting more weight on the initial and boundary conditions:

.. code-block:: python

    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    model.train(epochs=30000)

After we train the network with Adam, we compile again and continue to train the network using L-BFGS to achieve a smaller loss:

.. code-block:: python

    model.compile("L-BFGS", loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    losshistory, train_state = model.train()

Next, we will use the trained model to predict the solution to the incompressible Navier-Stokes equation:
First we create an equidistant mesh in the domain:

.. code-block:: python

    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
    )

    X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

We will evaluate both at the initial state :math:`t = 0` and the final state :math:`t=1`:

.. code-block:: python

    t_0 = np.zeros(1000).reshape(1000, 1)
    t_1 = np.ones(1000).reshape(1000, 1)

    X_0 = np.hstack((X, t_0))
    X_1 = np.hstack((X, t_1))

    output_0 = model.predict(X_0)
    output_1 = model.predict(X_1)

First, we assign the predicted and exact velocity components and pressure at :math:`t=0`:

.. code-block:: python

    u_pred_0 = output_0[:, 0].reshape(-1)
    v_pred_0 = output_0[:, 1].reshape(-1)
    w_pred_0 = output_0[:, 2].reshape(-1)
    p_pred_0 = output_0[:, 3].reshape(-1)

    u_exact_0 = u_func(X_0).reshape(-1)
    v_exact_0 = v_func(X_0).reshape(-1)
    w_exact_0 = w_func(X_0).reshape(-1)
    p_exact_0 = p_func(X_0).reshape(-1)

Then at :math:`t=1`:

.. code-block:: python

    u_pred_1 = output_1[:, 0].reshape(-1)
    v_pred_1 = output_1[:, 1].reshape(-1)
    w_pred_1 = output_1[:, 2].reshape(-1)
    p_pred_1 = output_1[:, 3].reshape(-1)

    u_exact_1 = u_func(X_1).reshape(-1)
    v_exact_1 = v_func(X_1).reshape(-1)
    w_exact_1 = w_func(X_1).reshape(-1)
    p_exact_1 = p_func(X_1).reshape(-1)

Next, we calculate the PDE residuals:

.. code-block:: python

    f_0 = model.predict(X_0, operator=pde)
    f_1 = model.predict(X_1, operator=pde)

Then we calculate the :math:`L^{2}` relative error for each velocity component plus pressure, and the mean residual, for both timepoints:

.. code-block:: python

    l2_difference_u_0 = dde.metrics.l2_relative_error(u_exact_0, u_pred_0)
    l2_difference_v_0 = dde.metrics.l2_relative_error(v_exact_0, v_pred_0)
    l2_difference_w_0 = dde.metrics.l2_relative_error(w_exact_0, w_pred_0)
    l2_difference_p_0 = dde.metrics.l2_relative_error(p_exact_0, p_pred_0)
    residual_0 = np.mean(np.absolute(f_0))

    l2_difference_u_1 = dde.metrics.l2_relative_error(u_exact_1, u_pred_1)
    l2_difference_v_1 = dde.metrics.l2_relative_error(v_exact_1, v_pred_1)
    l2_difference_w_1 = dde.metrics.l2_relative_error(w_exact_1, w_pred_1)
    l2_difference_p_1 = dde.metrics.l2_relative_error(p_exact_1, p_pred_1)
    residual_1 = np.mean(np.absolute(f_1))

Finally we plot the results:

.. code-block:: python

    print("Accuracy at t = 0:")
    print("Mean residual:", residual_0)
    print("L2 relative error in u:", l2_difference_u_0)
    print("L2 relative error in v:", l2_difference_v_0)
    print("L2 relative error in w:", l2_difference_w_0)
    print("\n")
    print("Accuracy at t = 1:")
    print("Mean residual:", residual_1)
    print("L2 relative error in u:", l2_difference_u_1)
    print("L2 relative error in v:", l2_difference_v_1)
    print("L2 relative error in w:", l2_difference_w_1)

Complete code
-------------

.. literarinclude:: ../../../examples/pinn_forward/Beltrami-flow.py
    :language: python
