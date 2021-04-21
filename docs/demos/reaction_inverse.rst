Inverse problem for DIffusion-Reaction Systems 
==============================================

Problem setup
--------------

We will solve the Diffusion reaction system in porous media, the equations are:

.. math:: \frac{\partial C_A}{\partial t} = D\frac{\partial^2 C_A}{\partial x^2} -{k_f}{C_A}{{C_B}^2},
.. math:: \frac{\partial C_B}{\partial t} = D\frac{\partial^2 C_B}{\partial x^2} -2{k_f}{C_A}{{C_B}^2}, \qquad x \in [0, 1], \qquad t \in [0, 10]

with the Dirichlet boundary conditions

.. math:: C_A(x,0)=C_B(x,0)=e^{-20x}, \quad C_A(0,t)=C_B(0,t)=1, \quad C_A(1,t)=C_B(1,t)=0.

The reference solution is `here <https://github.com/lululxvi/deepxde/blob/master/examples/dataset/reaction.npz>`_

Implementation
--------------
1)We first define the ``Geometry`` , ``Time``, and combine both of them using ``GeometryXTime``

.. code-block:: python

   geom = dde.geometry.Interval(0, 1)
   timedomain = dde.geometry.TimeDomain(0, 10)
   geomtime = dde.geometry.GeometryXTime(geom, timedomain)

      
2)Define the ``pde residual`` function of Poisson equation

.. code-block:: python

  def pde(x, y):
        ca, cb = y[:, 0:1], y[:, 1:2]
        dca_t = dde.grad.jacobian(y, x, i=0, j=1)
        dca_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dcb_t = dde.grad.jacobian(y, x, i=1, j=1)
        dcb_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        eq_a = dca_t - 1e-3 * D * dca_xx + kf * ca * cb ** 2
        eq_b = dcb_t - 1e-3 * D * dcb_xx + 2 * kf * ca * cb ** 2
        return [eq_a, eq_b]
        
3)Define ``boundary conditions`` 2 sets as there are 2 gases  , ``Initail conditions`` and ``Data`` which is geometry + pde + BC + IC + training points using DeepXDE inbuilt functions as shown

.. code-block:: python
  
  bc_a = dde.DirichletBC(geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0)
  bc_b = dde.DirichletBC(geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1)
  ic1 = dde.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
  ic2 = dde.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)
    
4)Define ``PointSetBC`` for observing the changes in gases concentration  and ``Data`` which is geometry + pde + BC + IC + training points using DeepXDE inbuilt functions as shown

.. code-block:: python  

  observe_x, Ca, Cb = gen_traindata()
  observe_y1 = dde.PointSetBC(observe_x, Ca, component=0)
  observe_y2 = dde.PointSetBC(observe_x, Cb, component=1)

  data = dde.data.TimePDE(geomtime,pde,[bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],num_domain=2000,num_boundary=100,num_initial=100,anchors=observe_x,num_test=50000)    
    
5)We use a fully connected neural network of depth 4 (i.e., 3 hidden layers) and width 20

.. code-block:: python

  net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")
     
6)Bulid the Model and Compile using the ``MODEL`` and ``Complile`` functions of DeepXDE as shown

.. code-block:: python

  model = dde.Model(data, net)
  model.compile("adam", lr=0.001)
  variable = dde.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")

7)Train the model using ``Train`` function of deepXDE and include the callbacks created above

.. code-block:: python

  losshistory, train_state = model.train(epochs=80000, callbacks=[variable])
  
8)save the plot of model history using the ``saveplot`` function of DeepXDE

.. code-block:: python 

  dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Complete code
--------------

.. literalinclude:: ../../examples/reaction_inverse.py
  :language: python
